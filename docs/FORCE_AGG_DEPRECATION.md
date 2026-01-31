# force_agg 参数废弃说明

## 变更摘要

**变更日期**：2025-12-22

**变更内容**：`force_agg` 参数不再影响聚合行为，边界模式（boundary mode）现在**只在边界分钟**执行聚合。

## 问题背景

### 旧行为（有问题）

```python
# 边界模式的旧逻辑
else:  # boundary
    do_agg_local = bool(force_agg) or is_boundary
    agg_start_iso = boundary.isoformat()
```

**问题**：

当 `force_agg=True` 时，**所有15个分钟桶**都会执行聚合：

```
15分钟窗口: 05:00 - 05:14

force_agg=True 时的行为:
  05:00: do_agg=True (边界分钟)     → 聚合 05:00-05:14 的数据 ✅ 正确
  05:01: do_agg=True (force_agg)    → 聚合 05:01-05:15 的数据 ❌ 错误（窗口不完整）
  05:02: do_agg=True (force_agg)    → 聚合 05:02-05:16 的数据 ❌ 错误（窗口不完整）
  ...
  05:14: do_agg=True (force_agg)    → 聚合 05:14-05:28 的数据 ❌ 错误（窗口不完整）
```

**结果**：产生大量**趋近于零或不准确的数值**，因为非边界分钟的窗口数据不完整。

### 新行为（已修复）

```python
# 边界模式的新逻辑
else:  # boundary
    # 只在边界分钟聚合，不在所有分钟都聚合
    do_agg_local = is_boundary
    agg_start_iso = boundary.isoformat()
```

**修复后**：

无论 `force_agg` 为 True 或 False，都**只在边界分钟**执行聚合：

```
15分钟窗口: 05:00 - 05:14

修复后的行为（force_agg 无影响）:
  05:00: do_agg=True (is_boundary=True)  → 聚合 05:00-05:14 的数据 ✅ 正确
  05:01: do_agg=False (is_boundary=False) → 不聚合 ✅ 正确
  05:02: do_agg=False (is_boundary=False) → 不聚合 ✅ 正确
  ...
  05:14: do_agg=False (is_boundary=False) → 不聚合 ✅ 正确
  05:15: do_agg=True (is_boundary=True)  → 聚合 05:15-05:29 的数据 ✅ 正确（下一个边界）
```

## 代码对比

### 修复前

```python
# AppleStockChecker/tasks/timestamp_alignment_task.py:3437

else:  # boundary
    do_agg_local = bool(force_agg) or is_boundary  # ❌ force_agg 影响所有分钟
    agg_start_iso = boundary.isoformat()
```

### 修复后

```python
# AppleStockChecker/tasks/timestamp_alignment_task.py:3437

else:  # boundary
    # 修复：只在边界分钟聚合，不在所有分钟都聚合
    # force_agg 参数保留用于向后兼容，但不再影响非边界分钟的聚合行为
    # 原问题：force_agg=True 会让所有15个分钟桶都聚合，产生大量趋近于零的不准确数值
    do_agg_local = is_boundary  # ✅ 只看 is_boundary，忽略 force_agg
    agg_start_iso = boundary.isoformat()
```

## 影响范围

### 1. API 调用

**修复前**：
```python
# force_agg=True 会导致所有分钟都聚合（错误）
payload = {
    "timestamp_iso": "2025-09-20T05:00:00+00:00",
    "agg_minutes": 15,
    "agg_mode": "boundary",
    "force_agg": True,  # ❌ 会让 15 个分钟都聚合
}
```

**修复后**：
```python
# force_agg 参数无效，只在边界分钟聚合（正确）
payload = {
    "timestamp_iso": "2025-09-20T05:00:00+00:00",
    "agg_minutes": 15,
    "agg_mode": "boundary",
    "force_agg": False,  # ✅ 推荐使用 False（或省略，效果相同）
}
```

### 2. 初始化脚本

**修复前**：
```python
# scripts/initialize_feature_snapshot.py
result = batch_generate_psta_same_ts(
    timestamp_iso=ts_iso,
    agg_mode="boundary",
    force_agg=True,  # ❌ 导致所有分钟都聚合
)
```

**修复后**：
```python
# scripts/initialize_feature_snapshot.py
result = batch_generate_psta_same_ts(
    timestamp_iso=ts_iso,
    agg_mode="boundary",
    force_agg=False,  # ✅ 已废弃，但保留以兼容旧代码
)
```

### 3. 周期任务配置

**建议配置**（无需修改，但可以移除 `force_agg` 参数）：

```python
# Celery Beat 配置
CELERY_BEAT_SCHEDULE = {
    'psta-每分钟处理': {
        'task': 'AppleStockChecker.tasks.batch_generate_psta_same_ts',
        'schedule': crontab(minute='*/1'),
        'kwargs': {
            'agg_minutes': 15,
            'agg_mode': 'boundary',
            # 'force_agg': False,  # ← 可以移除，不再需要
        },
    },
}
```

## 向后兼容性

### force_agg 参数保留

为了向后兼容，`force_agg` 参数**仍然存在**于 API 签名中，但**不再影响行为**：

```python
@shared_task(bind=True, name="AppleStockChecker.tasks.batch_generate_psta_same_ts")
def batch_generate_psta_same_ts(
    self,
    *,
    # ... 其他参数 ...
    force_agg: bool = False,  # ← 保留但无效
    # ... 其他参数 ...
) -> Dict[str, Any]:
```

**推荐做法**：

- 新代码：使用 `force_agg=False` 或省略该参数
- 旧代码：无需修改，`force_agg=True` 不会产生副作用

### 迁移指南

**场景1：历史数据回填脚本**

```python
# 旧代码（无需修改，但建议更新）
for ts in timestamps:
    batch_generate_psta_same_ts(
        timestamp_iso=ts,
        force_agg=True,  # ← 旧代码可继续使用，无副作用
    )

# 推荐更新为
for ts in timestamps:
    batch_generate_psta_same_ts(
        timestamp_iso=ts,
        # force_agg=False,  # ← 可以移除或设为 False
    )
```

**场景2：初始化脚本**

```bash
# 旧脚本（仍然能工作）
scripts/initialize_feature_snapshot.sh --auto

# 内部使用 force_agg=False（已更新）
```

## 为什么废弃 force_agg？

### 1. Bug 修复使其不再必要

之前需要 `force_agg=True` 的原因是边界分钟可能无数据导致聚合失败（`shops_seen=[]`）。修复后（从窗口数据提取 shop/iphone ID），边界分钟即使无数据也能正常聚合。

### 2. 边界语义已明确

**边界模式的正确语义**：

- 只在边界分钟（0, 15, 30, 45）聚合
- 每次聚合处理完整的15分钟窗口数据
- 非边界分钟只写入分钟数据，不聚合

`force_agg=True` 破坏了这个语义，导致非边界分钟也聚合，产生不正确的结果。

### 3. 简化理解和使用

**修复前**（复杂）：

| force_agg | 边界分钟 | 非边界分钟 |
|-----------|----------|-----------|
| False | 聚合 ✅ | 不聚合 ✅ |
| True | 聚合 ✅ | 聚合 ❌（错误）|

**修复后**（简单）：

| force_agg | 边界分钟 | 非边界分钟 |
|-----------|----------|-----------|
| False | 聚合 ✅ | 不聚合 ✅ |
| True | 聚合 ✅ | 不聚合 ✅（已修复）|

现在无论 `force_agg` 如何设置，行为都是正确的。

## 测试验证

### 测试用例1：验证非边界分钟不聚合

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from AppleStockChecker.models import FeatureSnapshot
from datetime import datetime, timezone

# 清理
bucket = datetime(2025, 9, 20, 5, 1, 0, tzinfo=timezone.utc)  # 非边界分钟
FeatureSnapshot.objects.filter(bucket=bucket).delete()

# 测试 force_agg=True（旧代码可能这样用）
result = batch_generate_psta_same_ts(
    timestamp_iso="2025-09-20T05:01:00+00:00",  # 非边界分钟（05:01）
    force_agg=True,
    agg_mode="boundary",
)

# 验证：非边界分钟不应生成 FeatureSnapshot
count = FeatureSnapshot.objects.filter(bucket=bucket).count()
assert count == 0, f"非边界分钟不应聚合，但生成了 {count} 条记录"
print("✅ 测试通过：非边界分钟不聚合（即使 force_agg=True）")
```

### 测试用例2：验证边界分钟正常聚合

```python
# 清理
bucket = datetime(2025, 9, 20, 5, 0, 0, tzinfo=timezone.utc)  # 边界分钟
FeatureSnapshot.objects.filter(bucket=bucket).delete()

# 测试边界分钟（force_agg 无影响）
result = batch_generate_psta_same_ts(
    timestamp_iso="2025-09-20T05:00:00+00:00",  # 边界分钟（05:00）
    force_agg=False,  # 使用 False
    agg_mode="boundary",
)

# 验证：边界分钟应生成 FeatureSnapshot
count = FeatureSnapshot.objects.filter(bucket=bucket).count()
assert count > 0, f"边界分钟应该聚合，但没有生成数据"
print(f"✅ 测试通过：边界分钟正常聚合（生成 {count} 条记录）")
```

## 相关文档更新

以下文档已更新以反映此变更：

- `docs/FORCE_AGG_VS_BOUNDARY.md` - force_agg 参数说明（需要更新）
- `docs/FEATURE_SNAPSHOT_INITIALIZATION.md` - 初始化指南（需要更新）
- `scripts/initialize_feature_snapshot.py` - 初始化脚本（已更新）
- `AppleStockChecker/tasks/timestamp_alignment_task.py` - 核心逻辑（已修复）

## 常见问题

### Q1: 我的旧脚本使用了 force_agg=True，需要修改吗？

**不需要**。`force_agg` 参数保留用于向后兼容，设为 True 也不会产生副作用（不会让非边界分钟聚合）。

但**建议更新**为 `force_agg=False` 或移除该参数，使代码意图更清晰。

### Q2: 为什么之前用 force_agg=True 能生成数据？

因为 `force_agg=True` 会让所有分钟（包括边界分钟）都聚合。边界分钟聚合是正确的，所以能生成数据。

但副作用是非边界分钟也聚合，产生大量错误数据。

### Q3: 修复后初始化脚本还能工作吗？

**能**。初始化脚本已更新为使用 `force_agg=False`。由于传入的都是边界时间（15分钟间隔），边界分钟仍然会正常聚合。

### Q4: rolling 模式受影响吗？

**不受影响**。rolling 模式的逻辑未改变：

```python
elif MODE == "rolling":
    do_agg_local = True  # 所有分钟都聚合（符合 rolling 语义）
    agg_start_iso = _rolling_start(mdt, int(agg_minutes)).isoformat()
```

rolling 模式下每分钟都应该聚合（滚动窗口），这是正确的。

### Q5: 能否完全移除 force_agg 参数？

技术上可以，但为了向后兼容保留了该参数。未来版本可能标记为 deprecated 并最终移除。

## 总结

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| 边界分钟聚合 | ✅ 正确 | ✅ 正确 |
| 非边界分钟（force_agg=False） | ✅ 正确（不聚合） | ✅ 正确（不聚合） |
| 非边界分钟（force_agg=True） | ❌ 错误（聚合） | ✅ 正确（不聚合）|
| 向后兼容 | N/A | ✅ 完全兼容 |
| 数据准确性 | ❌ 产生错误数据 | ✅ 只在边界聚合 |

**关键改进**：

- ✅ 修复了 `force_agg=True` 导致的非边界分钟聚合问题
- ✅ 简化了边界模式的语义（只在边界聚合，一目了然）
- ✅ 保持向后兼容（旧代码无需修改）
- ✅ 提高数据准确性（避免趋近于零的错误数值）

**用户行动**：

- 旧代码：无需修改，继续能工作
- 新代码：使用 `force_agg=False` 或省略该参数
- 回填脚本：只传入边界时间（15分钟间隔），无需 `force_agg=True`
