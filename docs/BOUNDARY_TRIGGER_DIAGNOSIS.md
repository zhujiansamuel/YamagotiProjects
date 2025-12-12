# 边界触发诊断说明

## 问题描述

使用以下调用方式期望在边界时刻自动触发统计指标计算，但似乎没有触发：

```python
result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso=ts,
    query_window_minutes=15,
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=False,
)
```

## 原因分析

### 1. 时间窗口生成机制

`collect_items_for_psta` 函数会**总是生成 15 个分钟桶**（从 `timestamp_iso` 向前推算）：

```python
# collectors.py:62-69
ticks_dt: List = []
cur = ts_dt
for _ in range(15):
    ticks_dt.append(_floor_to_minute(cur))
    cur = cur - timedelta(minutes=1)
# 保持从新到旧
ticks_iso: List[str] = [_iso(x) for x in ticks_dt]
```

**示例**：
- 如果 `timestamp_iso="2025-12-12T10:07:00+09:00"`
- 生成的 15 个分钟桶：`10:07, 10:06, 10:05, 10:04, 10:03, 10:02, 10:01, 10:00, 09:59, 09:58, 09:57, 09:56, 09:55, 09:54, 09:53`

### 2. 边界判断逻辑

在 `batch_generate_psta_same_ts` 中，对**每个分钟**都会判断是否是边界：

```python
# timestamp_alignment_task.py:3324-3336
mdt = _to_aware(minute_iso)
boundary = _floor_to_step(mdt, int(agg_minutes))  # 向下对齐到步长
is_boundary = (mdt == boundary)

if MODE == "boundary":
    do_agg_local = bool(force_agg) or is_boundary
    agg_start_iso = boundary.isoformat()
```

**边界计算**（假设 `agg_minutes=15`）：
- `_floor_to_step` 会将时间向下对齐到 15 分钟边界
- 对齐规则：`minute - (minute % 15)`

| 当前分钟 | 边界时刻 | 是否边界 | do_agg |
|---------|---------|---------|--------|
| 10:07 | 10:00 | ❌ | False |
| 10:06 | 10:00 | ❌ | False |
| 10:05 | 10:00 | ❌ | False |
| ... | ... | ... | ... |
| 10:00 | 10:00 | ✅ | True |
| 09:59 | 09:45 | ❌ | False |
| ... | ... | ... | ... |
| 09:45 | 09:45 | ✅ | True |

### 3. 实际触发情况

在上述示例中（`timestamp_iso=10:07, agg_minutes=15`）：
- **15 个分钟桶**都会创建子任务
- 但**只有 2 个边界分钟**会触发聚合：`10:00` 和 `09:45`
- 其他 13 个分钟只会写入原始数据（`PurchasingShopTimeAnalysis`），**不触发聚合计算**

### 4. OverallBar/CohortBar 已被禁用

即使在边界分钟触发了聚合（`do_agg=True`），也**不会计算 OverallBar 和 CohortBar**，因为这两个计算已经被注释掉了：

```python
# timestamp_alignment_task.py:1645 (_run_aggregation)
# ===== 已禁用：OverallBar 和 CohortBar 计算 =====
# 原因：主要使用 FeatureSnapshot 四类组合，无需全店聚合统计

# # 1) OverallBar
# _agg_overallbar(...)

# # 2) CohortBar
# _agg_cohortbar(...)
```

**只会计算**：
- ✅ FeatureSnapshot（4种组合）
- ✅ 时间序列指标（基于 FeatureSnapshot 的历史数据）

## 解决方案

### 方案 1：确保传入边界时刻

如果您希望每次调用都触发聚合，请确保 `timestamp_iso` 是边界时刻：

```python
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

# 生成边界时刻（每 15 分钟一个）
start = datetime(2025, 12, 12, 10, 0, 0, tzinfo=JST)  # 10:00（边界）
timestamps = [
    (start + timedelta(minutes=i*15)).isoformat()
    for i in range(24)  # 生成 24 个边界时刻（覆盖 6 小时）
]

for ts in timestamps:
    result = batch_generate_psta_same_ts(
        job_id=uuid4().hex,
        timestamp_iso=ts,  # 边界时刻
        query_window_minutes=15,
        agg_minutes=15,
        agg_mode="boundary",
        force_agg=False,
    )
```

**边界时刻示例**（`agg_minutes=15`）：
```
10:00, 10:15, 10:30, 10:45, 11:00, 11:15, ...
```

### 方案 2：使用 force_agg=True 强制聚合

如果您希望**无论是否边界都触发聚合**，可以设置 `force_agg=True`：

```python
result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso=ts,  # 任意时刻
    query_window_minutes=15,
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=True,  # ⚠️ 强制所有分钟都触发聚合
)
```

**注意**：
- `force_agg=True` 会让**所有 15 个分钟**都触发聚合（不仅是边界）
- 这会增加计算量和数据库压力
- 可能产生重复数据（同一时间窗口被多次聚合）

### 方案 3：使用 rolling 模式

如果您希望每个分钟都基于其前 N 分钟的滑动窗口进行聚合：

```python
result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso=ts,
    query_window_minutes=15,
    agg_minutes=15,
    agg_mode="rolling",  # 滑动窗口模式
    force_agg=False,
)
```

**rolling 模式特点**：
- 每个分钟都会触发聚合（`do_agg=True`）
- 聚合窗口：`[当前分钟 - (agg_minutes-1), 当前分钟]`
- 例如 10:07 的窗口：`[09:53, 10:07]`

### 方案 4：恢复 OverallBar/CohortBar 计算

如果您期望看到 OverallBar/CohortBar 数据，需要取消注释：

1. 打开 `AppleStockChecker/tasks/timestamp_alignment_task.py`
2. 找到 `_run_aggregation` 函数（约 line 1645）
3. 取消以下代码的注释：

```python
# 1) OverallBar
_agg_overallbar(
    ts_iso=ts_iso,
    ts_dt=ts_dt,
    rows=rows,
    use_window=use_window,
    bucket_start=bucket_start,
    bucket_end=bucket_end,
    is_final_bar=is_final_bar,
    agg_ctx=agg_ctx,
    ob_has_iphone=ob_has_iphone,
)

# 2) CohortBar
_agg_cohortbar(...)
```

4. 同时需要恢复时间序列特征中的 OverallBar/CohortBar 基值收集（line 1082-1100）

## 验证方法

### 检查是否触发了聚合

1. **查看日志**：
```python
# 在 _run_aggregation 函数开始处会有日志
logger.info(f"[_run_aggregation] ts={ts_iso}, do_agg={do_agg}, agg_minutes={agg_minutes}")
```

2. **检查 FeatureSnapshot 数据**：
```python
from AppleStockChecker.models import FeatureSnapshot
from django.utils import timezone

# 查询指定时间桶的数据
bucket_time = timezone.datetime(2025, 12, 12, 10, 0, 0, tzinfo=timezone.get_current_timezone())
snapshots = FeatureSnapshot.objects.filter(bucket=bucket_time)

print(f"找到 {snapshots.count()} 条 FeatureSnapshot 记录")
for snap in snapshots[:10]:
    print(f"  scope={snap.scope}, name={snap.name}, value={snap.value}")
```

3. **检查 OverallBar 数据**（如果已恢复）：
```python
from AppleStockChecker.models import OverallBar

bucket_time = timezone.datetime(2025, 12, 12, 10, 0, 0, tzinfo=timezone.get_current_timezone())
bars = OverallBar.objects.filter(bucket=bucket_time)

print(f"找到 {bars.count()} 条 OverallBar 记录")
for bar in bars[:10]:
    print(f"  iphone_id={bar.iphone_id}, mean={bar.mean}, std={bar.std}")
```

### 调试单个分钟的聚合

```python
from AppleStockChecker.tasks.timestamp_alignment_task import psta_process_minute_bucket
from uuid import uuid4

# 直接调用单个分钟的处理任务
result = psta_process_minute_bucket(
    ts_iso="2025-12-12T10:00:00+09:00",  # 边界时刻
    rows=[],  # 可以传入空行，仅触发聚合
    job_id=uuid4().hex,
    do_agg=True,  # 强制聚合
    agg_start_iso="2025-12-12T10:00:00+09:00",
    agg_minutes=15,
)

print(result)
```

## 常见误区

### 误区 1：以为 query_window_minutes 控制聚合窗口

❌ **错误理解**：`query_window_minutes=15` 表示聚合窗口是 15 分钟

✅ **正确理解**：
- `query_window_minutes=15`：查询数据库时向前拉取 15 分钟的原始数据
- `agg_minutes=15`：聚合窗口大小（边界间隔）
- 这两个参数通常保持一致，但含义不同

### 误区 2：以为每次调用都会触发聚合

❌ **错误理解**：调用 `batch_generate_psta_same_ts` 就会触发聚合

✅ **正确理解**：
- 只有边界分钟才会触发聚合（boundary 模式 + force_agg=False）
- 或者设置 `force_agg=True` 强制所有分钟触发
- 或者使用 `rolling` 模式让所有分钟都触发

### 误区 3：以为会看到 OverallBar/CohortBar 数据

❌ **错误理解**：触发聚合后会生成 OverallBar/CohortBar 数据

✅ **正确理解**：
- 这两个计算已被禁用（2025-12-12 的修改）
- 只会生成 FeatureSnapshot 数据
- 如需 OverallBar/CohortBar，需要取消注释恢复计算

## 推荐配置

### 实时数据处理（每分钟触发）

```python
# 每分钟调用一次，只在边界触发聚合
result = batch_generate_psta_same_ts(
    timestamp_iso=None,  # 自动使用当前时间
    query_window_minutes=15,
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=False,
)
```

### 历史数据重算（精确边界）

```python
# 只处理边界时刻，避免重复计算
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
start = datetime(2025, 10, 23, 7, 0, 0, tzinfo=JST)
end = datetime(2025, 10, 23, 20, 0, 0, tzinfo=JST)

# 只生成边界时刻
current = start
while current <= end:
    result = batch_generate_psta_same_ts(
        timestamp_iso=current.isoformat(),
        query_window_minutes=15,
        agg_minutes=15,
        agg_mode="boundary",
        force_agg=False,
        sequential=True,  # 顺序执行，避免数据库压力
    )
    current += timedelta(minutes=15)  # 下一个边界
```

### 滑动窗口分析

```python
# 每分钟都计算滑动窗口统计
result = batch_generate_psta_same_ts(
    timestamp_iso=ts,
    query_window_minutes=15,
    agg_minutes=15,
    agg_mode="rolling",  # 每分钟都基于前 15 分钟计算
    force_agg=False,
)
```

## 总结

**关键要点**：
1. ⏰ **边界对齐**：`timestamp_iso` 必须是边界时刻才会触发聚合（boundary 模式）
2. 🔢 **15个分钟桶**：`collect_items_for_psta` 总是生成 15 个分钟，但只有边界分钟聚合
3. 🚫 **OverallBar/CohortBar 已禁用**：目前只计算 FeatureSnapshot
4. ✅ **FeatureSnapshot 正常**：4种组合的统计数据会正常计算
5. 🔧 **三种模式**：boundary（边界）、rolling（滑动）、off（不聚合）

**推荐阅读**：
- `docs/DISABLE_OVERALLBAR_COHORTBAR.md` - OverallBar/CohortBar 禁用说明
- `docs/SEQUENTIAL_EXECUTION.md` - 顺序执行模式说明
