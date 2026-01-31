# FeatureSnapshot 数据初始化指南

## 问题背景

FeatureSnapshot 表为空时会出现"循环依赖"问题：

```
需要历史 FeatureSnapshot 数据 → 才能计算新的聚合指标
    ↓
但表为空 → 没有历史数据
    ↓
新的聚合被跳过 → 表仍然为空
    ↓
永远无法生成数据 ❌
```

**解决方案**：使用 `force_agg=True` 强制执行初始聚合，填充第一批 FeatureSnapshot 数据。

## 初始化脚本

### 脚本位置

- `scripts/initialize_feature_snapshot.py` - Python 主脚本
- `scripts/initialize_feature_snapshot.sh` - Shell 包装器（自动检测环境）

### 核心参数

所有初始化任务使用以下参数：

- `force_agg=True` - **强制执行聚合**（即使没有历史数据）
- `sequential=True` - 顺序执行（避免数据库连接压力）
- `agg_minutes=15` - 15分钟聚合窗口
- `agg_mode="boundary"` - 边界模式

## 使用方法

### 1. 检查当前状态

```bash
# 查看 FeatureSnapshot 表状态
scripts/initialize_feature_snapshot.sh --check-only
```

**输出示例**：
```
============================================================
FeatureSnapshot 表状态检查
============================================================
总记录数: 0
⚠️  表为空，需要初始化
```

### 2. 自动初始化（推荐）

自动查找最早的可用原始数据并初始化：

```bash
# 从最早数据开始，初始化 24 小时范围
scripts/initialize_feature_snapshot.sh --auto

# 自定义时间范围（例如 48 小时）
scripts/initialize_feature_snapshot.sh --auto --hours 48

# 先演习看看会处理哪些时间点
scripts/initialize_feature_snapshot.sh --auto --dry-run
```

**工作流程**：

1. 自动查找 `PurchasingShopPriceRecord` 中最早的记录
2. 向上取整到 15 分钟边界（例如：23:07 → 23:15）
3. 生成时间戳序列（15分钟间隔）
4. 顺序处理每个时间点
5. 验证每个时间点是否生成了 FeatureSnapshot 数据

**输出示例**：
```
查找最早的可用原始数据...
✅ 找到最早记录: 2025-10-03 23:07:42+00:00
建议起始时间（15分钟边界）: 2025-10-03 23:15:00+00:00

自动初始化参数:
  起始: 2025-10-03 23:15:00+00:00
  结束: 2025-10-04 23:15:00+00:00
  范围: 24.0 小时

确认开始初始化? (yes/no): yes

============================================================
初始化 FeatureSnapshot 数据
============================================================
起始时间: 2025-10-03 23:15:00+00:00
结束时间: 2025-10-04 23:15:00+00:00
时间点数量: 97
参数: force_agg=True, sequential=True
============================================================

[1/97] 处理时间点: 2025-10-03T23:15:00+00:00
  任务 ID: 7f3a2b1c4d5e6f7a8b9c0d1e2f3a4b5c
  参数: agg_minutes=15, force_agg=True, sequential=True
  ✅ 处理完成
  total_buckets: 15
  FeatureSnapshot 记录数: 156

[2/97] 处理时间点: 2025-10-03T23:30:00+00:00
  ✅ 处理完成
  total_buckets: 15
  FeatureSnapshot 记录数: 156

...

============================================================
初始化完成
============================================================
成功: 97
跳过: 0
失败: 0
生成的 FeatureSnapshot 记录总数: 15132
```

### 3. 指定时间范围初始化

```bash
# 初始化特定时间范围
scripts/initialize_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-04T01:00:00+00:00"

# 先演习不实际执行
scripts/initialize_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-04T01:00:00+00:00" \
  --dry-run
```

### 4. 初始化单个时间点

```bash
# 只处理一个时间点（用于测试或补数据）
scripts/initialize_feature_snapshot.sh \
  --timestamp "2025-10-03T23:00:00+00:00"
```

## 验证初始化结果

### 方法 1：使用检查脚本

```bash
# 查看整体状态
scripts/initialize_feature_snapshot.sh --check-only
```

**期望输出**：
```
============================================================
FeatureSnapshot 表状态检查
============================================================
总记录数: 15132
最早时间: 2025-10-03 23:15:00+00:00
最晚时间: 2025-10-04 23:00:00+00:00

各 scope 记录数:
  shop:1|cohort:iphone-14-pro-max: 3784
  shop:1|iphone:5: 3784
  shopcohort:beijing|cohort:iphone-14-pro-max: 3784
  shopcohort:beijing|iphone:5: 3784
```

### 方法 2：使用验证脚本

```bash
scripts/verify_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-04T01:00:00+00:00" \
  --verbose
```

### 方法 3：数据库查询

```python
from AppleStockChecker.models import FeatureSnapshot
from datetime import datetime, timezone

# 查询特定时间点的数据
bucket_time = datetime(2025, 10, 3, 23, 15, 0, tzinfo=timezone.utc)
snapshots = FeatureSnapshot.objects.filter(bucket=bucket_time)

print(f"找到 {snapshots.count()} 条记录")

# 查看前几条
for snap in snapshots[:5]:
    print(f"{snap.scope} | {snap.name} = {snap.value}")
```

**期望输出**：
```
找到 156 条记录
shop:1|iphone:5 | mean_price = 5699.0
shop:1|iphone:5 | median_price = 5699.0
shop:1|iphone:5 | std_price = 0.0
shop:1|iphone:5 | count = 12
shop:1|iphone:5 | min_price = 5699.0
```

## 每个时间点应生成多少条记录？

**计算公式**：
```
4 种 scope 组合 × N 个统计指标 = 总记录数
```

**4 种 scope 组合**：
1. `shop:X|iphone:Y` - 特定店铺 × 特定机型
2. `shop:X|cohort:Y` - 特定店铺 × 机型组
3. `shopcohort:X|iphone:Y` - 店铺组 × 特定机型
4. `shopcohort:X|cohort:Y` - 店铺组 × 机型组

**常见统计指标**（约 39 个）：
- `mean_price`, `median_price`, `std_price`, `min_price`, `max_price`
- `count`, `sum_price`
- `q25_price`, `q75_price`, `iqr_price`
- `cv_price`, `range_price`
- `skew_price`, `kurtosis_price`
- ... 等等

**典型场景**：
- 4 种组合 × 39 个指标 = **156 条记录/时间点**
- 如果有多个 shop/iphone 组合，记录数会成倍增加

## 常见问题

### Q1: 初始化后还是没有数据？

**可能原因**：

1. **原始数据不足** - 查询窗口内没有足够的 `PurchasingShopPriceRecord`

   ```bash
   # 检查原始数据
   from AppleStockChecker.models import PurchasingShopPriceRecord
   from datetime import datetime, timezone

   start = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)
   end = datetime(2025, 10, 3, 23, 15, 0, tzinfo=timezone.utc)

   records = PurchasingShopPriceRecord.objects.filter(
       recorded_at__gte=start,
       recorded_at__lte=end,
   )
   print(f"原始记录数: {records.count()}")
   ```

2. **shop_ids/iphone_ids 过滤** - 检查是否因为过滤条件导致没有数据

   ```python
   # 查看有哪些 shop 和 iphone
   from AppleStockChecker.models import PurchasingShopPriceRecord

   shops = PurchasingShopPriceRecord.objects.values_list('shop_id', flat=True).distinct()
   iphones = PurchasingShopPriceRecord.objects.values_list('iphone_id', flat=True).distinct()

   print(f"可用的 shop_ids: {list(shops)}")
   print(f"可用的 iphone_ids: {list(iphones)}")
   ```

3. **collectors.py bug 未修复** - 确认 bucket_by_minute 和 new_price 的修复已应用

   ```bash
   # 检查修复是否存在
   git log --oneline | grep -E "(bucket_by_minute|new_price)"
   ```

### Q2: 为什么需要 force_agg=True？

**正常流程**（需要历史数据）：

```python
# _run_aggregation 查询历史窗口
historical_data = FeatureSnapshot.objects.filter(
    bucket__gte=window_start,
    bucket__lt=window_end,
)

if historical_data.count() < min_required:
    # 跳过聚合
    return
```

**force_agg=True 流程**（强制执行）：

```python
if force_agg:
    # 即使历史数据不足也执行聚合
    # 使用当前时间点的 PurchasingShopTimeAnalysis 数据
    compute_aggregations()
```

### Q3: 初始化后周期任务能接上吗？

能。初始化后：

1. FeatureSnapshot 表有了初始数据
2. 周期任务（每分钟触发）继续运行
3. 在边界时间（每15分钟）自动聚合
4. 新的聚合能查询到历史数据（初始化生成的）
5. 数据链条完整 ✅

**验证方法**：

```bash
# 等待 15 分钟后检查是否有新数据
scripts/verify_feature_snapshot.sh \
  --start "$(date -u -d '15 minutes ago' '+%Y-%m-%dT%H:%M:00+00:00')" \
  --end "$(date -u '+%Y-%m-%dT%H:%M:00+00:00')" \
  --verbose
```

### Q4: 可以重复运行初始化脚本吗？

可以。脚本会**跳过**已有数据的时间点：

```
[1/10] 处理时间点: 2025-10-03T23:00:00+00:00
  ⏭️  已有 156 条记录，跳过

[2/10] 处理时间点: 2025-10-03T23:15:00+00:00
  任务 ID: abc123...
  ✅ 处理完成
  FeatureSnapshot 记录数: 156
```

### Q5: 初始化会影响数据库性能吗？

使用 `sequential=True` 参数，任务**顺序执行**：

- ✅ 只使用 1 个数据库连接（不是 N 个）
- ✅ 每个时间点处理完再处理下一个
- ✅ 可以随时中断（Ctrl+C）
- ❌ 速度较慢（适合历史数据补数）

对于实时数据，周期任务使用 `sequential=False`（默认并行）提高速度。

## 推荐工作流程

### 首次初始化

```bash
# 1. 检查当前状态
scripts/initialize_feature_snapshot.sh --check-only

# 2. 演习自动初始化（看看会处理多少数据）
scripts/initialize_feature_snapshot.sh --auto --dry-run

# 3. 确认后执行初始化（24小时范围）
scripts/initialize_feature_snapshot.sh --auto

# 4. 验证结果
scripts/initialize_feature_snapshot.sh --check-only
```

### 补充历史数据

```bash
# 补充特定时间范围
scripts/initialize_feature_snapshot.sh \
  --start "2025-10-05T00:00:00+00:00" \
  --end "2025-10-05T12:00:00+00:00"

# 补充单个缺失的时间点
scripts/initialize_feature_snapshot.sh \
  --timestamp "2025-10-05T15:00:00+00:00"
```

### 监控周期任务

初始化完成后，周期任务应该能正常接上：

```bash
# 每 15 分钟检查一次新数据
watch -n 900 'scripts/verify_feature_snapshot.sh \
  --start "$(date -u -d "30 minutes ago" "+%Y-%m-%dT%H:%M:00+00:00")" \
  --end "$(date -u "+%Y-%m-%dT%H:%M:00+00:00")"'
```

## 故障排查

### 日志位置

```bash
# Celery worker 日志
docker compose logs -f worker

# Django 日志
docker compose logs -f web

# 查看特定任务的日志
docker compose exec web python manage.py shell
>>> from celery.result import AsyncResult
>>> result = AsyncResult('task-id-here')
>>> print(result.traceback)
```

### 手动触发单个时间点（调试）

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from uuid import uuid4

result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso="2025-10-03T23:00:00+00:00",
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=True,
    sequential=True,
)

print("Result:", result)

# 检查是否生成数据
from AppleStockChecker.models import FeatureSnapshot
from datetime import datetime, timezone

bucket_time = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)
count = FeatureSnapshot.objects.filter(bucket=bucket_time).count()
print(f"Generated {count} FeatureSnapshot records")
```

## 相关文档

- `docs/FIXES_SUMMARY.md` - Bug 修复总结
- `docs/SEQUENTIAL_EXECUTION.md` - 顺序执行模式
- `docs/BOUNDARY_TRIGGER_DIAGNOSIS.md` - 边界触发诊断
- `docs/BUG_FIX_BUCKET_BY_MINUTE.md` - bucket_by_minute bug 详解

## 总结

1. **问题**：FeatureSnapshot 表空 → 无法计算新聚合 → 永远为空
2. **解决**：使用 `force_agg=True` 强制生成初始数据
3. **工具**：`scripts/initialize_feature_snapshot.sh --auto`
4. **验证**：`scripts/verify_feature_snapshot.sh` 或 `--check-only`
5. **后续**：周期任务自动接上，持续生成新数据
