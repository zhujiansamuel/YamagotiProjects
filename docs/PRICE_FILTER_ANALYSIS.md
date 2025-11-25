# 历史记录重算分析报告

## 📋 问题分析

### API 调用链
```
POST /AppleStockChecker/purchasing-time-analyses/dispatch_ts/
    ↓
dispatch_psta_batch_same_ts (api.py)
    ↓
batch_generate_psta_same_ts (Celery 任务)
    ↓
psta_process_minute_bucket (子任务)
    ↓
_process_minute_rows (数据写入)
```

---

## ✅ 好消息：不会跳过已有记录

### 核心逻辑（`_process_minute_rows` 函数，1550-1584行）

```python
with transaction.atomic():
    inst = (
        PurchasingShopTimeAnalysis.objects
        .select_for_update()
        .filter(
            shop_id=shop_id,
            iphone_id=iphone_id,
            Timestamp_Time=ts_dt,
        )
        .first()
    )

    if inst:
        # ✅ 已有记录：更新（不跳过）
        inst.Job_ID = job_id
        inst.Original_Record_Time_Zone = orig_tz
        inst.Timestamp_Time_Zone = ts_tz
        inst.Record_Time = rec_dt
        inst.Alignment_Time_Difference = align_diff
        inst.New_Product_Price = int(new_price)
        inst.Update_Count = (inst.Update_Count or 0) + 1  # 更新计数
        inst.save()
    else:
        # 没有记录：创建新记录
        inst = PurchasingShopTimeAnalysis.objects.create(...)
```

### 关键点

1. **不会跳过**：代码使用 `update_or_create` 逻辑（手动实现）
2. **幂等操作**：重复运行会覆盖旧数据（Last Write Wins）
3. **更新计数**：`Update_Count` 字段记录更新次数
4. **使用行锁**：`select_for_update()` 防止并发竞争

---

## ⚠️ 但有一个问题：固定价格阈值过滤

### 当前代码（1542-1543行）

```python
# 区间外：直接跳过
if price < PRICE_MIN or price > PRICE_MAX:
    continue
```

### 问题

这段代码仍然使用**固定阈值** `PRICE_MIN=10000, PRICE_MAX=350000`，**没有使用我们新增的动态价格区间**。

这意味着：
- 重算历史记录时，固定阈值会过滤掉一些本应保留的数据
- 对于低价商品（如 iPhone SE），固定阈值过于宽松
- 对于高价商品（如 iPhone 17 Pro Max 2TB），固定阈值可能过于严格

---

## 🔧 需要修复的地方

### 修改 `_process_minute_rows` 函数

**修改前（1542-1543行）**：
```python
# 区间外：直接跳过
if price < PRICE_MIN or price > PRICE_MAX:
    continue
```

**应改为**：
```python
# 使用动态价格区间过滤
if not is_price_valid(price, iphone_id, ts_dt):
    logger.debug(
        f"价格超出动态区间: shop_id={shop_id}, iphone_id={iphone_id}, "
        f"price={price}, timestamp={ts_dt}"
    )
    continue
```

或者更详细的版本：
```python
# 使用动态价格区间过滤
price_min, price_max = get_dynamic_price_range(iphone_id, ts_dt)
if not (price_min <= price <= price_max):
    logger.debug(
        f"价格超出动态区间: shop_id={shop_id}, iphone_id={iphone_id}, "
        f"price={price}, 区间=[{price_min:.0f}, {price_max:.0f}]"
    )
    continue
```

---

## 📊 影响范围

### 1️⃣ 数据写入阶段（`_process_minute_rows`）
- **当前**：使用固定阈值 [10000, 350000]
- **需要**：使用动态价格区间

### 2️⃣ 统计聚合阶段（`_calculate_overallbar_stats`）
- **已修复**：✅ 使用动态价格区间

### 3️⃣ 特征计算阶段（`_calculate_features`）
- **已修复**：✅ 使用动态价格区间

---

## 🎯 完整性检查

### 代码中所有使用 PRICE_MIN/PRICE_MAX 的位置

| 位置 | 状态 | 说明 |
|------|------|------|
| `_process_minute_rows` (1542行) | ❌ 需要修复 | 数据写入时的价格过滤 |
| `_calculate_overallbar_stats` (435-436行) | ✅ 已修复 | 使用动态区间 |
| `_calculate_features` (692-693行) | ✅ 已修复 | 使用动态区间 |
| 注释代码（2161行等） | ⚠️ 忽略 | 已注释的旧代码 |

---

## 💡 建议

### 立即修复
修改 `_process_minute_rows` 函数中的价格过滤逻辑，使用动态价格区间。

### 渐进式修复（可选）
如果担心动态区间计算开销，可以：
1. 在函数开始时预计算所有 `iphone_id` 的价格区间
2. 缓存到字典中，避免重复查询

```python
# 在 _process_minute_rows 函数开始处添加
unique_iphone_ids = {r.get("iphone_id") for r in rows if r.get("iphone_id")}
price_ranges = {
    iphone_id: get_dynamic_price_range(iphone_id, ts_dt)
    for iphone_id in unique_iphone_ids
}

# 然后在循环中使用
price_min, price_max = price_ranges.get(iphone_id, (PRICE_MIN, PRICE_MAX))
if not (price_min <= price <= price_max):
    continue
```

---

## 🧪 测试建议

### 1. 验证不跳过已有记录
```bash
# 第一次运行
curl -X POST "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/" \
     -H "Content-Type: application/json" \
     -d '{"timestamp_iso": "2025-01-20T10:00:00+09:00"}'

# 检查数据库
SELECT shop_id, iphone_id, Update_Count FROM purchasing_shop_time_analysis
WHERE Timestamp_Time = '2025-01-20 10:00:00+09:00';

# 第二次运行（相同时间）
curl -X POST "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/" \
     -H "Content-Type: application/json" \
     -d '{"timestamp_iso": "2025-01-20T10:00:00+09:00"}'

# 再次检查 - Update_Count 应该增加
SELECT shop_id, iphone_id, Update_Count FROM purchasing_shop_time_analysis
WHERE Timestamp_Time = '2025-01-20 10:00:00+09:00';
```

### 2. 验证动态价格区间
```python
from AppleStockChecker.tasks.timestamp_alignment_task import get_dynamic_price_range
from django.utils import timezone

# 测试不同型号的价格区间
iphone_ids = [1, 2, 3, 4, 5]  # 不同型号
reference_time = timezone.now()

for iphone_id in iphone_ids:
    price_min, price_max = get_dynamic_price_range(iphone_id, reference_time)
    print(f"iPhone {iphone_id}: [{price_min:.0f}, {price_max:.0f}]")
```

---

## 📝 总结

### ✅ 确认事项
1. **重算不会跳过已有记录**：代码使用 update 逻辑，幂等操作
2. **统计聚合已使用动态区间**：OverallBar 和特征计算已修复
3. **有更新计数器**：`Update_Count` 字段追踪重算次数

### ❌ 需要修复
1. **数据写入阶段仍用固定阈值**：`_process_minute_rows` 函数需要修改
2. 应使用 `get_dynamic_price_range()` 替换固定的 `PRICE_MIN/PRICE_MAX`

### 🎯 修复优先级
**高优先级**：修改 `_process_minute_rows` 以保持一致性，让所有阶段都使用动态价格区间。

---

生成时间：2025-11-25
文件：AppleStockChecker/tasks/timestamp_alignment_task.py
相关行数：1542-1543, 1550-1584
