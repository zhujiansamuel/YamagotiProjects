# Bug 修复：边界聚合时 rows 为空导致无法生成 FeatureSnapshot

## Bug 症状

使用 `force_agg=False` (边界模式) 时，即使时间戳是正确的边界时间（如 `2025-09-20T05:00:00+00:00`），也无法生成 FeatureSnapshot 数据。但使用 `force_agg=True` 时能正常生成数据。

## 用户报告

```python
# 用户的调用方式
payload = {
    "timestamp_iso": "2025-09-20T05:00:00+00:00",  # ✅ 正确的边界时间
    "agg_minutes": 15,
    "agg_mode": "boundary",
    "force_agg": False,  # ❌ 不生成数据
}

# force_agg=True 时能生成数据
payload = {
    "timestamp_iso": "2025-09-20T05:00:00+00:00",
    "agg_minutes": 15,
    "agg_mode": "boundary",
    "force_agg": True,  # ✅ 能生成数据
}
```

## 边界时间验证

使用诊断工具验证时间戳确实是边界时间：

```bash
$ python scripts/test_boundary_simple.py "2025-09-20T05:00:00+00:00"

测试时间戳: 2025-09-20T05:00:00+00:00
解析后的时间: 2025-09-20 05:00:00+00:00
  分: 0
  秒: 0
  微秒: 0

边界判断: mdt == boundary: True

do_agg 计算 (boundary 模式):
  force_agg=False: do_agg = False or True = True  ✅ 应该会执行聚合

分钟检查:
  minute % 15 = 0  ✅
  秒数为 0  ✅
  微秒为 0  ✅

结论: ✅ 这是一个边界时间，force_agg=False 时会执行聚合
```

时间戳本身没有问题，边界判断逻辑也正确。

## 根本原因

Bug 位于 `AppleStockChecker/tasks/timestamp_alignment_task.py:694-695`：

```python
def _agg_feature_combos(...):
    # —— 预取本桶出现过的 shop/iphone —— #
    shops_seen = sorted({int(r.get("shop_id")) for r in rows if r.get("shop_id")})
    iphones_seen = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})

    if use_window:
        base_qs = (
            PurchasingShopTimeAnalysis.objects
            .filter(
                Timestamp_Time__gte=bucket_start,
                Timestamp_Time__lt=bucket_end,
                shop_id__in=shops_seen,  # ⬅️ Bug: 如果 shops_seen=[]，查询为空！
                iphone_id__in=iphones_seen,
            )
            ...
        )
```

### 问题机制

**边界聚合的执行流程**：

```python
batch_generate_psta_same_ts(timestamp_iso="2025-09-20T05:00:00+00:00")
  ↓
  处理 15 个分钟桶 (05:00 - 05:14)
  ↓
  每个分钟桶创建子任务:
    for minute_iso in bucket_minute_key:
        mdt = _to_aware(minute_iso)
        boundary = _floor_to_step(mdt, 15)
        is_boundary = (mdt == boundary)

        if agg_mode == "boundary":
            do_agg_local = force_agg or is_boundary

        psta_process_minute_bucket(
            ts_iso=minute_iso,
            rows=minute_rows,  # ⬅️ 该分钟的数据（可能为空！）
            do_agg=do_agg_local,
        )
```

**边界分钟的特殊情况**：

```
15分钟窗口: 05:00 - 05:14

分钟桶分布:
  05:00: rows=[10条数据], do_agg=True (边界), 写入 PSTA + 聚合
  05:01: rows=[8条数据],  do_agg=False (非边界), 只写入 PSTA
  05:02: rows=[5条数据],  do_agg=False (非边界), 只写入 PSTA
  ...
  05:14: rows=[],         do_agg=False (非边界), 跳过

问题场景 1: 边界分钟无新数据
  05:00: rows=[] (该分钟没有新数据采集)
         ↓
         is_boundary=True, do_agg=True ✅
         ↓
         _run_aggregation(rows=[]) ⬅️ 空的 rows
         ↓
         shops_seen = []  (从空 rows 提取)
         iphones_seen = []
         ↓
         base_qs = PSTA.filter(shop_id__in=[], iphone_id__in=[])
         ↓
         查询结果为空！即使窗口内有 05:01-05:14 的历史数据 ❌
         ↓
         data_by_si = {}
         ↓
         不生成 FeatureSnapshot
```

**为什么 force_agg=True 能工作**：

```
force_agg=True 时:
  05:00: rows=[], do_agg=True, shops_seen=[], 查询失败 ❌
  05:01: rows=[8条], do_agg=True (强制), shops_seen=[1,2,3], 查询成功 ✅
  05:02: rows=[5条], do_agg=True (强制), shops_seen=[1,2], 查询成功 ✅
  ...

因为所有分钟都聚合，至少有一些分钟的 rows 不为空，
能正确提取 shops_seen/iphones_seen 并查询到数据。
```

## 代码对比

### 修复前（Bug）

```python
def _agg_feature_combos(...):
    try:
        # —— 预取本桶出现过的 shop/iphone —— #
        shops_seen = sorted({int(r.get("shop_id")) for r in rows if r.get("shop_id")})
        iphones_seen = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})
        # ⬆️ Bug: 从 rows 提取，如果 rows=[]，则 shops_seen=[], iphones_seen=[]

        if use_window:
            base_qs = (
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    shop_id__in=shops_seen,  # ⬅️ 如果 shops_seen=[]，查询为空
                    iphone_id__in=iphones_seen,
                    New_Product_Price__isnull=False,
                )
                ...
            )
```

### 修复后（正确）

```python
def _agg_feature_combos(...):
    try:
        # —— 预取本桶出现过的 shop/iphone —— #
        # Bug修复: 窗口模式时应该从窗口内的 PSTA 数据提取 shop/iphone，而不是从 rows（单分钟数据）
        # 原因: 边界分钟的 rows 可能为空，但窗口内仍有历史数据需要聚合
        if use_window:
            # 窗口模式: 从数据库查询窗口内所有的 shop_id 和 iphone_id
            shops_seen = sorted(set(
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    New_Product_Price__isnull=False,
                )
                .values_list('shop_id', flat=True)
                .distinct()
            ))
            iphones_seen = sorted(set(
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    New_Product_Price__isnull=False,
                )
                .values_list('iphone_id', flat=True)
                .distinct()
            ))
        else:
            # 单分钟模式: 从 rows 提取（原有逻辑）
            shops_seen = sorted({int(r.get("shop_id")) for r in rows if r.get("shop_id")})
            iphones_seen = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})

        if use_window:
            base_qs = (
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    shop_id__in=shops_seen,  # ✅ 现在 shops_seen 从窗口数据提取，不会为空
                    iphone_id__in=iphones_seen,
                    New_Product_Price__isnull=False,
                )
                ...
            )
```

## 修复原理

### 修复前的逻辑问题

```
rows (单分钟数据) → 提取 shops_seen/iphones_seen → 查询窗口数据
  ⬆️ 错误: 单分钟的 shop/iphone 可能不完整或为空
```

### 修复后的正确逻辑

```
窗口模式:
  窗口数据 → 提取所有 shops_seen/iphones_seen → 查询窗口数据
  ✅ 正确: 从窗口内的实际数据提取完整的 shop/iphone 列表

单分钟模式:
  rows → 提取 shops_seen/iphones_seen → 查询单分钟数据
  ✅ 正确: 单分钟模式下，rows 就是全部数据
```

## 影响范围

### 受影响的场景

1. ✅ **修复后工作**：`force_agg=False` + 边界时间 + 边界分钟无新数据

   ```python
   # 典型场景: 周期任务在边界时间触发，但该分钟恰好没有新数据采集
   batch_generate_psta_same_ts(
       timestamp_iso="2025-09-20T05:00:00+00:00",
       agg_mode="boundary",
       force_agg=False,  # ✅ 现在能正常工作
   )
   ```

2. ✅ **修复后工作**：窗口内有历史数据但边界分钟为空

   ```
   窗口 05:00-05:14:
     05:01-05:14: 有数据 → 写入 PSTA
     05:00: 边界分钟无数据 → 仍然能聚合 05:01-05:14 的数据 ✅
   ```

### 未受影响的场景

1. ✅ `force_agg=True`：本来就工作（绕过了 bug）
2. ✅ 单分钟模式 (`agg_minutes=1`)：不使用窗口查询
3. ✅ 边界分钟有数据：`rows` 不为空，能提取到 shop/iphone

## 性能影响

### 额外查询

修复引入了 2 个额外的数据库查询（仅在窗口模式）：

```python
# 查询 1: 提取窗口内所有 shop_id
shops_seen = PSTA.filter(
    Timestamp_Time__gte=bucket_start,
    Timestamp_Time__lt=bucket_end,
).values_list('shop_id', flat=True).distinct()

# 查询 2: 提取窗口内所有 iphone_id
iphones_seen = PSTA.filter(
    Timestamp_Time__gte=bucket_start,
    Timestamp_Time__lt=bucket_end,
).values_list('iphone_id', flat=True).distinct()
```

**影响评估**：

- ✅ 查询简单（只取 ID，带 distinct）
- ✅ 窗口时间范围小（15分钟）
- ✅ 有时间索引 (`Timestamp_Time`)
- ✅ 结果集小（通常 < 100 个 shop/iphone）
- ⚠️ 每次边界聚合增加 2 次查询
- ✅ 但修复了功能性 bug，权衡合理

**优化建议**（可选）：

```python
# 合并为单次查询（如果性能成为问题）
from django.db.models import Q

ids = (
    PurchasingShopTimeAnalysis.objects
    .filter(
        Timestamp_Time__gte=bucket_start,
        Timestamp_Time__lt=bucket_end,
        New_Product_Price__isnull=False,
    )
    .values('shop_id', 'iphone_id')
    .distinct()
)

shops_seen = sorted(set(row['shop_id'] for row in ids))
iphones_seen = sorted(set(row['iphone_id'] for row in ids))
```

## 测试验证

### 测试用例 1: 边界分钟无数据

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from AppleStockChecker.models import FeatureSnapshot, PurchasingShopTimeAnalysis
from datetime import datetime, timezone
from uuid import uuid4

# 准备测试数据: 只在 05:01-05:14 有数据，05:00 无数据
# (模拟边界分钟恰好无新数据采集的场景)

# 清理
bucket = datetime(2025, 9, 20, 5, 0, 0, tzinfo=timezone.utc)
FeatureSnapshot.objects.filter(bucket=bucket).delete()

# 执行
result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso="2025-09-20T05:00:00+00:00",
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=False,  # ⬅️ 测试修复后的边界模式
    sequential=True,
)

# 验证
count = FeatureSnapshot.objects.filter(bucket=bucket).count()
print(f"生成的 FeatureSnapshot 记录数: {count}")
assert count > 0, "修复后应该能生成数据"
```

### 测试用例 2: 对比 force_agg=True/False

```python
# 测试 1: force_agg=False（边界模式）
result_boundary = batch_generate_psta_same_ts(
    timestamp_iso="2025-09-20T05:00:00+00:00",
    force_agg=False,
)

count_boundary = FeatureSnapshot.objects.filter(
    bucket=datetime(2025, 9, 20, 5, 0, 0, tzinfo=timezone.utc)
).count()

# 测试 2: force_agg=True（强制模式）
FeatureSnapshot.objects.filter(
    bucket=datetime(2025, 9, 20, 5, 0, 0, tzinfo=timezone.utc)
).delete()

result_force = batch_generate_psta_same_ts(
    timestamp_iso="2025-09-20T05:00:00+00:00",
    force_agg=True,
)

count_force = FeatureSnapshot.objects.filter(
    bucket=datetime(2025, 9, 20, 5, 0, 0, tzinfo=timezone.utc)
).count()

# 验证: 修复后，两者应该生成相同数量的数据
print(f"force_agg=False: {count_boundary} 条")
print(f"force_agg=True:  {count_force} 条")
assert count_boundary == count_force, "修复后两者应该一致"
```

## 历史回顾

### 为什么之前没发现这个 Bug？

1. **初始化使用 `force_agg=True`**：

   ```bash
   # 初始化脚本默认使用 force_agg=True
   scripts/initialize_feature_snapshot.sh --auto
   ```

   绕过了 bug，能正常生成数据。

2. **周期任务可能运气好**：

   如果周期任务触发时，边界分钟恰好有新数据，`rows` 不为空，就能正常工作。

3. **Bug 只在特定条件触发**：

   - ✅ 必须是窗口模式 (`agg_minutes > 1`)
   - ✅ 必须是边界分钟
   - ✅ 边界分钟的 `rows` 必须为空
   - ✅ 使用 `force_agg=False`

   这些条件同时满足的概率较低，所以不容易发现。

## 相关文档

- `docs/FORCE_AGG_VS_BOUNDARY.md` - force_agg 参数详解
- `docs/BOUNDARY_TRIGGER_DIAGNOSIS.md` - 边界触发诊断
- `docs/BUG_FIX_BUCKET_BY_MINUTE.md` - bucket_by_minute bug
- `scripts/diagnose_boundary_issue.py` - 边界判断诊断工具
- `scripts/test_boundary_simple.py` - 时间戳边界测试

## 总结

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| 边界模式 + 边界分钟无数据 | ❌ 不生成 FeatureSnapshot | ✅ 正常生成 |
| force_agg=True | ✅ 能生成（绕过 bug） | ✅ 能生成 |
| 单分钟模式 | ✅ 能生成 | ✅ 能生成 |
| 窗口模式查询逻辑 | ❌ 从 rows 提取（错误） | ✅ 从窗口数据提取（正确） |
| 额外数据库查询 | 0 | 2（可接受） |

**关键改进**：

- ✅ 修复了边界聚合在特定条件下失败的 bug
- ✅ `force_agg=False` 现在能正常工作
- ✅ 周期任务更稳定（不依赖边界分钟有无数据）
- ✅ 代码逻辑更合理（窗口数据从窗口查询，而非从单分钟数据推断）

**用户可以安全地使用 `force_agg=False` 进行边界聚合了！**
