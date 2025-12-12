# Bug 机制详解：bucket_by_minute 与 force_agg 的关系

## 核心问题

**这个 bug 与 `force_agg` 无关！**

`force_agg` 只控制**是否触发聚合计算**，但这个 bug 导致**根本没有数据可以计算**。

## 完整数据流程对比

### 正常流程（Bug修复后）

```
1️⃣ batch_generate_psta_same_ts 被调用
   timestamp_iso = "2025-10-03T23:00:00+00:00"
   agg_minutes = 15
   agg_mode = "boundary"
   force_agg = False
   ↓

2️⃣ 调用 collect_items_for_psta
   ↓
   生成 15 个分钟刻度:
   ["23:00", "22:59", "22:58", ..., "22:46"]
   ↓
   查询数据库 (PurchasingShopPriceRecord)
   WHERE recorded_at BETWEEN "22:46" AND "23:00"
   ↓
   ✅ 查询到 500 条原始数据
   ↓
   初始化 bucket_by_minute = {
     "23:00": [],
     "22:59": [],
     "22:58": [],
     ...
     "22:46": []
   }
   ↓
   遍历 500 条数据:
   - 数据1: recorded_at="22:55:30" → 向下取整到 "22:55"
     ✅ "22:55" in bucket_by_minute? Yes
     ✅ bucket_by_minute["22:55"].append(0)

   - 数据2: recorded_at="22:56:12" → 向下取整到 "22:56"
     ✅ "22:56" in bucket_by_minute? Yes
     ✅ bucket_by_minute["22:56"].append(1)

   - 数据500: recorded_at="23:00:45" → 向下取整到 "23:00"
     ✅ "23:00" in bucket_by_minute? Yes
     ✅ bucket_by_minute["23:00"].append(499)
   ↓
   返回结果:
   bucket_minute_key = {
     "22:46": {"shop:1|iphone:10": [5, 12], ...},
     "22:47": {"shop:2|iphone:10": [8], ...},
     ...
     "23:00": {"shop:1|iphone:10": [490, 499], ...}
   }
   rows = [500条数据]
   ↓

3️⃣ batch_generate_psta_same_ts 创建子任务
   ↓
   遍历 bucket_minute_key 的 15 个分钟:

   - 分钟 "22:46":
     is_boundary? _floor_to_step("22:46", 15) = "22:45"
     "22:46" == "22:45"? No ❌
     do_agg = force_agg or is_boundary = False or False = False
     rows_for_this_minute = [数据5, 数据12, ...]  ✅ 有数据
     创建子任务: psta_process_minute_bucket(
       ts_iso="22:46",
       rows=[数据5, 数据12, ...],
       do_agg=False  ← 非边界，不聚合
     )

   - 分钟 "22:47":
     is_boundary? No ❌
     do_agg = False
     rows=[数据8, ...]  ✅ 有数据
     创建子任务(do_agg=False)  ← 不聚合

   ...

   - 分钟 "22:45":
     is_boundary? _floor_to_step("22:45", 15) = "22:45"
     "22:45" == "22:45"? Yes ✅
     do_agg = False or True = True  ← 边界时刻！
     rows=[...]  ✅ 有数据
     创建子任务: psta_process_minute_bucket(
       ts_iso="22:45",
       rows=[...],
       do_agg=True  ← 边界，会聚合！
     )

   - 分钟 "23:00":
     is_boundary? _floor_to_step("23:00", 15) = "23:00"
     "23:00" == "23:00"? Yes ✅
     do_agg = False or True = True
     rows=[数据490, 数据499, ...]  ✅ 有数据
     创建子任务: psta_process_minute_bucket(
       ts_iso="23:00",
       rows=[数据490, 数据499, ...],
       do_agg=True  ← 边界，会聚合！
     )
   ↓

4️⃣ 执行子任务 psta_process_minute_bucket

   对于非边界分钟（do_agg=False）:
   - 写入 PurchasingShopTimeAnalysis（原始数据对齐）
   - ❌ 跳过 _run_aggregation（不聚合）
   - ❌ 不计算 FeatureSnapshot

   对于边界分钟（do_agg=True，如 "22:45", "23:00"）:
   - 写入 PurchasingShopTimeAnalysis
   - ✅ 调用 _run_aggregation
     ↓
     查询数据库：
     SELECT * FROM PurchasingShopTimeAnalysis
     WHERE Timestamp_Time BETWEEN "22:45" AND "23:00"
     ↓
     ✅ 找到数据（因为前面的子任务已写入）
     ↓
     调用 _agg_feature_combos（4种组合）:
     - Case 1: shop:1|iphone:10 → 计算 mean, std, median...
     - Case 2: shopcohort:full_store|iphone:10 → ...
     - Case 3: shop:1|cohort:iphone15_series → ...
     - Case 4: shopcohort:full_store|cohort:iphone15_series → ...
     ↓
     ✅ 写入 FeatureSnapshot 表
```

### Bug 流程（修复前）

```
1️⃣ batch_generate_psta_same_ts 被调用
   timestamp_iso = "2025-10-03T23:00:00+00:00"
   agg_minutes = 15
   agg_mode = "boundary"
   force_agg = False
   ↓

2️⃣ 调用 collect_items_for_psta
   ↓
   生成 15 个分钟刻度:
   ["23:00", "22:59", "22:58", ..., "22:46"]
   ↓
   查询数据库 (PurchasingShopPriceRecord)
   WHERE recorded_at BETWEEN "22:46" AND "23:00"
   ↓
   ✅ 查询到 500 条原始数据
   ↓
   ❌ BUG: 初始化 bucket_by_minute = {}  ← 空字典！
   ↓
   遍历 500 条数据:
   - 数据1: recorded_at="22:55:30" → 向下取整到 "22:55"
     ❌ "22:55" in bucket_by_minute? No（字典是空的）
     ❌ continue（跳过这条数据）

   - 数据2: recorded_at="22:56:12" → 向下取整到 "22:56"
     ❌ "22:56" in bucket_by_minute? No
     ❌ continue（跳过）

   - 数据500: recorded_at="23:00:45" → 向下取整到 "23:00"
     ❌ "23:00" in bucket_by_minute? No
     ❌ continue（跳过）
   ↓
   所有 500 条数据都被跳过了！
   ↓
   返回结果:
   bucket_minute_key = {}  ← 空的！
   rows = [500条数据]  ← 虽然rows还在，但bucket_minute_key是空的
   ↓

3️⃣ batch_generate_psta_same_ts 创建子任务
   ↓
   遍历 bucket_minute_key:
   ❌ bucket_minute_key 是空的 {}
   ❌ 循环体一次都不执行
   ↓
   subtasks = []  ← 空列表！
   ↓
   或者即使有数据（从其他渠道），每个分钟的 rows 也是空的:

   - 分钟 "23:00":
     is_boundary? Yes ✅
     do_agg = True  ← 边界，应该聚合
     ❌ 但是 rows = []（空的，因为bucket_minute_key["23:00"]不存在）
     创建子任务: psta_process_minute_bucket(
       ts_iso="23:00",
       rows=[],  ← 空数据！
       do_agg=True
     )
   ↓

4️⃣ 执行子任务 psta_process_minute_bucket

   对于边界分钟（do_agg=True）:
   - ❌ rows=[]，没有数据可以写入 PurchasingShopTimeAnalysis
   - ✅ 调用 _run_aggregation（因为 do_agg=True）
     ↓
     查询数据库：
     SELECT * FROM PurchasingShopTimeAnalysis
     WHERE Timestamp_Time BETWEEN "22:45" AND "23:00"
     ↓
     ❌ 没有找到数据（因为没有写入）
     ↓
     调用 _agg_feature_combos:
     ❌ 没有数据可以计算
     ↓
     ❌ FeatureSnapshot 表没有数据写入
```

## 关键区别对比表

| 步骤 | 修复后（正常） | 修复前（Bug） |
|------|--------------|--------------|
| **collect_items_for_psta** | | |
| 查询原始数据 | ✅ 500条 | ✅ 500条 |
| bucket_by_minute 初始化 | ✅ {15个key: []} | ❌ {} 空字典 |
| 遍历数据填充bucket | ✅ 500条全部添加 | ❌ 500条全部跳过 |
| bucket_minute_key | ✅ {15个分钟: 数据} | ❌ {} 空字典 |
| **batch_generate_psta_same_ts** | | |
| 创建子任务数量 | ✅ 15个 | ❌ 0个或15个空任务 |
| 每个子任务的rows | ✅ 有数据 | ❌ [] 空数组 |
| 边界分钟的do_agg | ✅ True | ✅ True（但没用） |
| 非边界分钟的do_agg | ✅ False | ✅ False |
| **psta_process_minute_bucket** | | |
| 写入原始数据 | ✅ 有数据写入 | ❌ 无数据可写 |
| 边界分钟调用聚合 | ✅ 调用且有数据 | ✅ 调用但无数据 |
| FeatureSnapshot生成 | ✅ 有数据 | ❌ 无数据 |

## force_agg 的真正作用

`force_agg` 参数**只影响 do_agg 的值**，不影响数据收集：

### force_agg=False（默认）

```python
# 在 batch_generate_psta_same_ts 中
for minute_iso in bucket_minute_key.keys():
    mdt = _to_aware(minute_iso)
    boundary = _floor_to_step(mdt, 15)  # 对齐到15分钟边界
    is_boundary = (mdt == boundary)

    do_agg = False or is_boundary  # force_agg=False
    # 只有边界分钟 do_agg=True
```

| 分钟 | 是否边界 | do_agg | 是否聚合 |
|------|---------|--------|---------|
| 22:46 | ❌ | False | ❌ |
| 22:47 | ❌ | False | ❌ |
| ... | ... | ... | ... |
| 22:45 | ✅ | True | ✅ |
| ... | ... | ... | ... |
| 23:00 | ✅ | True | ✅ |

**结果**：15个分钟中，只有2个边界分钟（22:45, 23:00）触发聚合

### force_agg=True

```python
# 在 batch_generate_psta_same_ts 中
for minute_iso in bucket_minute_key.keys():
    mdt = _to_aware(minute_iso)
    boundary = _floor_to_step(mdt, 15)
    is_boundary = (mdt == boundary)

    do_agg = True or is_boundary  # force_agg=True
    # 所有分钟 do_agg=True
```

| 分钟 | 是否边界 | do_agg | 是否聚合 |
|------|---------|--------|---------|
| 22:46 | ❌ | True | ✅ |
| 22:47 | ❌ | True | ✅ |
| ... | ... | ... | ... |
| 22:45 | ✅ | True | ✅ |
| ... | ... | ... | ... |
| 23:00 | ✅ | True | ✅ |

**结果**：15个分钟全部触发聚合

## 为什么 force_agg=True 也不能解决 Bug？

即使 `force_agg=True` 让所有分钟都设置 `do_agg=True`，但：

```python
# psta_process_minute_bucket 收到的参数
psta_process_minute_bucket(
    ts_iso="23:00",
    rows=[],  # ❌ 空数据！（因为bucket_by_minute bug）
    do_agg=True,  # ✅ 是的，会调用聚合
)

# 在函数内部
if rows:  # ❌ rows是空的，条件为False
    # 写入 PurchasingShopTimeAnalysis
    pass

if do_agg:  # ✅ True，会执行
    _run_aggregation(...)  # ✅ 调用了
    # 但是查询数据库时找不到数据
    # 因为上面的 rows=[] 导致没有数据写入数据库
```

**关键点**：
- `force_agg=True` 确保调用 `_run_aggregation`
- 但 `_run_aggregation` 需要从数据库查询数据
- Bug 导致数据根本没有写入数据库
- 所以即使调用聚合函数，也没有数据可以计算

## 数据写入的两个阶段

### 阶段1：写入原始对齐数据（PurchasingShopTimeAnalysis）

```python
# psta_process_minute_bucket 中
for row in rows:  # ← rows 如果是空的，这里就不执行
    PurchasingShopTimeAnalysis.objects.update_or_create(
        shop_id=row['shop_id'],
        iphone_id=row['iphone_id'],
        Timestamp_Time=ts_dt,
        defaults={'New_Product_Price': row['price_new']}
    )
```

**Bug 影响**：rows=[] 导致这一步没有数据写入

### 阶段2：聚合计算（FeatureSnapshot）

```python
# _run_aggregation → _agg_feature_combos 中
if do_agg:  # ← force_agg=True 让这里为 True
    # 从数据库查询数据
    base_qs = PurchasingShopTimeAnalysis.objects.filter(
        Timestamp_Time__gte=bucket_start,
        Timestamp_Time__lt=bucket_end,
        ...
    )
    # ❌ 但是查询结果是空的（因为阶段1没有写入）
```

**Bug 影响**：即使调用了聚合，也查不到数据

## 总结

### Bug 的真正影响

❌ **数据在收集阶段就丢失了**
- `collect_items_for_psta` 中的 `bucket_by_minute` bug
- 导致所有原始数据在分配到分钟桶时被跳过
- `bucket_minute_key` 是空的
- 子任务收到的 `rows=[]`

### force_agg 的作用范围

✅ **force_agg 只控制是否触发聚合**
- `force_agg=False`：只有边界分钟触发聚合
- `force_agg=True`：所有分钟都触发聚合

❌ **force_agg 不能解决数据丢失问题**
- 因为数据在收集阶段就丢失了
- 不是聚合调用的问题
- 是根本没有数据可以聚合

### 修复的关键

只需修复 `bucket_by_minute` 初始化：
```python
# 修复前
bucket_by_minute = {}  # ❌

# 修复后
bucket_by_minute = {tick: [] for tick in ticks_iso}  # ✅
```

修复后，无论 `force_agg` 是 True 还是 False，数据都能正确收集和处理。
