# force_agg 参数与边界聚合的关系

## 问题现象

使用 `force_agg=True` 能生成 FeatureSnapshot 数据，但切换到 `force_agg=False` 后，边界时间点仍然没有执行聚合。

## 原因分析

### do_agg 的计算逻辑

在 `batch_generate_psta_same_ts` 函数中（`timestamp_alignment_task.py:3369`）：

```python
# 边界判断
mdt = _to_aware(minute_iso)
boundary = _floor_to_step(mdt, int(agg_minutes))
is_boundary = (mdt == boundary)

if MODE == "off":
    do_agg_local = False
    agg_start_iso = None
elif MODE == "rolling":
    do_agg_local = True
    agg_start_iso = _rolling_start(mdt, int(agg_minutes)).isoformat()
else:  # boundary
    do_agg_local = bool(force_agg) or is_boundary  # ⬅️ 关键行
    agg_start_iso = boundary.isoformat()
```

**逻辑**：
```
do_agg_local = bool(force_agg) or is_boundary
```

这意味着：
- `force_agg=True`：所有分钟都执行聚合（`do_agg_local` 总是 `True`）
- `force_agg=False`：只有边界分钟执行聚合（`do_agg_local = is_boundary`）

### 边界判断条件

**is_boundary 的判断**：

```python
def _floor_to_step(dt: timezone.datetime, step_min: int) -> timezone.datetime:
    return dt - timezone.timedelta(
        minutes=dt.minute % step_min,
        seconds=dt.second,
        microseconds=dt.microsecond
    )

# 使用
boundary = _floor_to_step(mdt, int(agg_minutes))
is_boundary = (mdt == boundary)
```

**判断为边界的条件**（以 `agg_minutes=15` 为例）：

1. ✅ `minute % 15 == 0`（分钟能被15整除）
2. ✅ `second == 0`（秒数为0）
3. ✅ `microsecond == 0`（微秒为0）

**三个条件必须同时满足**，才会判断为边界。

### 常见错误情况

#### 情况1：秒数不为0

```python
# 错误示例
timestamp_iso = "2025-10-03T23:00:05+00:00"  # ❌ 秒数为5
                                           # minute=0 ✅
                                           # second=5 ❌
                                           # is_boundary = False

# 正确示例
timestamp_iso = "2025-10-03T23:00:00+00:00"  # ✅ 秒数为0
                                           # minute=0 ✅
                                           # second=0 ✅
                                           # is_boundary = True
```

#### 情况2：微秒不为0

```python
# 从 datetime.now() 获取时间
now = datetime.now(timezone.utc)  # 例如: 2025-10-03 23:00:00.123456+00:00
                                  # ❌ microsecond=123456

# 需要清零秒和微秒
now = now.replace(second=0, microsecond=0)  # ✅ 2025-10-03 23:00:00+00:00
```

#### 情况3：分钟不在边界上

```python
# 错误示例（agg_minutes=15）
timestamp_iso = "2025-10-03T23:07:00+00:00"  # minute=7
                                            # 7 % 15 = 7 ≠ 0
                                            # is_boundary = False

# 边界时间（agg_minutes=15）
边界分钟：0, 15, 30, 45
例如：23:00, 23:15, 23:30, 23:45
```

## 诊断工具

使用诊断脚本检查边界判断逻辑：

```bash
# 检查特定时间点
scripts/diagnose_boundary_issue.sh --timestamp "2025-10-03T23:00:00+00:00"

# 自动选择最近的边界时间进行检查
scripts/diagnose_boundary_issue.sh --auto

# 测试 force_agg=True 和 force_agg=False 的效果
scripts/diagnose_boundary_issue.sh --timestamp "2025-10-03T23:00:00+00:00" --test-force-agg
```

### 诊断输出示例

**边界时间**：
```
======================================================================
边界判断诊断：2025-10-03T23:00:00+00:00
======================================================================

输入时间：2025-10-03 23:00:00+00:00
  年: 2025
  月: 10
  日: 3
  时: 23
  分: 0
  秒: 0

边界时间：2025-10-03 23:00:00+00:00
  年: 2025
  月: 10
  日: 3
  时: 23
  分: 0
  秒: 0

是否为边界：✅ 是

分钟检查：
  minute % 15 = 0
  是否整除：✅ 是
  秒数是否为0：✅ 是
  微秒是否为0：✅ 是

do_agg 计算逻辑：
  force_agg=True:  do_agg = True or True = True  ✅ 会执行聚合
  force_agg=False: do_agg = False or True = True  ✅ 会执行聚合
```

**非边界时间**：
```
======================================================================
边界判断诊断：2025-10-03T23:07:05+00:00
======================================================================

输入时间：2025-10-03 23:07:05+00:00
  分: 7
  秒: 5

边界时间：2025-10-03 23:00:00+00:00
  分: 0
  秒: 0

是否为边界：❌ 否
  差异：425.0 秒
  原因：输入时间不在 15 分钟边界上
  最近的边界时间：2025-10-03 23:00:00+00:00
  下一个边界时间：2025-10-03 23:15:00+00:00

分钟检查：
  minute % 15 = 7
  是否整除：❌ 否
  秒数是否为0：❌ 否 (秒=5)
  微秒是否为0：✅ 是

do_agg 计算逻辑：
  force_agg=True:  do_agg = True or False = True  ✅ 会执行聚合
  force_agg=False: do_agg = False or False = False  ❌ 不会执行聚合
```

## 解决方案

### 方案1：确保使用边界时间

**手动构造边界时间**：

```python
from datetime import datetime, timezone, timedelta

# 方法1：从当前时间向下取整
now = datetime.now(timezone.utc)
minute = (now.minute // 15) * 15  # 向下取整到15的倍数
boundary = now.replace(minute=minute, second=0, microsecond=0)

# 方法2：使用 _floor_to_step 函数
from AppleStockChecker.tasks.timestamp_alignment_task import _floor_to_step, _to_aware

ts_dt = _to_aware("2025-10-03T23:07:42+00:00")
boundary = _floor_to_step(ts_dt, 15)  # 2025-10-03 23:00:00+00:00

# 方法3：直接构造
boundary = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)  # ✅ 确保秒和微秒为0
```

**边界时间列表生成**：

```python
from datetime import datetime, timezone, timedelta

def generate_boundaries(start_dt, end_dt, step_minutes=15):
    """生成边界时间序列"""
    # 确保起始时间在边界上
    minute = (start_dt.minute // step_minutes) * step_minutes
    current = start_dt.replace(minute=minute, second=0, microsecond=0)

    boundaries = []
    while current <= end_dt:
        boundaries.append(current)
        current += timedelta(minutes=step_minutes)

    return boundaries

# 使用
start = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 10, 4, 1, 0, 0, tzinfo=timezone.utc)
boundaries = generate_boundaries(start, end, 15)

for b in boundaries:
    print(b.isoformat())
```

### 方案2：使用 force_agg=True（初始化场景）

如果 FeatureSnapshot 表为空，使用 `force_agg=True` 强制生成初始数据：

```bash
# 使用初始化脚本（内部使用 force_agg=True）
scripts/initialize_feature_snapshot.sh --auto
```

**适用场景**：
- ✅ FeatureSnapshot 表为空时的冷启动
- ✅ 需要重新处理历史数据
- ✅ 补充缺失的非边界时间点数据

**不适用场景**：
- ❌ 正常的周期任务（应使用边界模式）
- ❌ 生产环境的实时计算（性能开销大）

### 方案3：检查 _run_aggregation 内部逻辑

即使 `do_agg=True`，`_run_aggregation` 函数内部可能还有其他条件导致跳过聚合。

**可能的原因**：

1. **窗口内没有数据**

```python
# 在 _run_aggregation 中
psta_data = PurchasingShopTimeAnalysis.objects.filter(
    bucket__gte=window_start,
    bucket__lt=window_end,
)

if not psta_data.exists():
    # 跳过聚合
    return
```

**检查方法**：

```python
from AppleStockChecker.models import PurchasingShopTimeAnalysis
from datetime import datetime, timezone, timedelta

boundary = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)
window_start = boundary
window_end = boundary + timedelta(minutes=15)

count = PurchasingShopTimeAnalysis.objects.filter(
    bucket__gte=window_start,
    bucket__lt=window_end,
).count()

print(f"窗口内 PSTA 数据：{count} 条")
```

如果为0，说明：
- 分钟数据没有被 `psta_process_minute_bucket` 写入
- 需要检查 `rows` 是否为空或 `bucket_by_minute` bug 是否已修复

2. **其他业务逻辑条件**

检查 `_run_aggregation` 函数中是否有其他跳过逻辑：

```bash
# 查找 _run_aggregation 函数
grep -A 100 "def _run_aggregation" AppleStockChecker/tasks/timestamp_alignment_task.py | grep -E "(if|return)" | head -20
```

## 验证边界聚合是否工作

### 测试步骤

```bash
# 1. 使用诊断工具检查边界时间
scripts/diagnose_boundary_issue.sh --auto --check-data-only

# 2. 测试 force_agg=False（边界模式）
scripts/diagnose_boundary_issue.sh --timestamp "2025-10-03T23:00:00+00:00" --test-force-agg

# 3. 验证生成的数据
scripts/verify_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-03T23:00:00+00:00" \
  --verbose
```

### 预期结果

**force_agg=False 且 is_boundary=True 时**：

```
✅ 执行成功
结果：
  total_buckets: 15
  FeatureSnapshot 记录数: 156
  ✅ 成功生成数据
```

**force_agg=False 且 is_boundary=False 时**：

```
✅ 执行成功（任务不报错）
结果：
  total_buckets: 15  （处理了15个分钟桶）
  FeatureSnapshot 记录数: 0  （非边界分钟不执行聚合）
  ⚠️  警告：没有生成 FeatureSnapshot 数据
```

## 周期任务配置

确保周期任务使用边界时间：

```python
# Celery Beat 配置示例
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'psta-每分钟处理': {
        'task': 'AppleStockChecker.tasks.batch_generate_psta_same_ts',
        'schedule': crontab(minute='*/1'),  # 每分钟执行
        'kwargs': {
            'agg_minutes': 15,
            'agg_mode': 'boundary',
            'force_agg': False,  # ⬅️ 使用边界模式
        },
    },
}
```

**工作原理**：

- 每分钟触发一次任务
- 处理过去15分钟的数据（`query_window_minutes=15`）
- 只在边界分钟（0, 15, 30, 45）执行聚合
- 其他分钟只写入 `PurchasingShopTimeAnalysis`，不执行聚合

## 总结

| 场景 | force_agg | 时间类型 | 是否聚合 | 适用场景 |
|------|-----------|----------|----------|----------|
| 1 | `True` | 任意时间 | ✅ 是 | 初始化、历史数据重算 |
| 2 | `False` | 边界时间 | ✅ 是 | 正常周期任务 |
| 3 | `False` | 非边界时间 | ❌ 否 | 正常周期任务（中间分钟） |

**关键要点**：

1. `force_agg=False` 时，**必须**使用边界时间才会执行聚合
2. 边界时间要求：`minute % 15 == 0 && second == 0 && microsecond == 0`
3. 使用诊断工具 `scripts/diagnose_boundary_issue.sh` 检查边界判断
4. 初始化时使用 `force_agg=True`，正常运行时使用 `force_agg=False`
5. 检查 `PurchasingShopTimeAnalysis` 窗口内是否有数据

**调试建议**：

如果 `force_agg=False` 在边界时间仍然不工作：

1. 使用诊断工具确认 `is_boundary=True`
2. 检查 `PurchasingShopTimeAnalysis` 窗口内数据
3. 检查 `_run_aggregation` 函数内部逻辑
4. 查看 Celery worker 日志中的聚合相关输出
