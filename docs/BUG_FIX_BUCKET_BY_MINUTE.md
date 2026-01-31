# Bug 修复：bucket_by_minute 初始化导致数据丢失

## 问题描述

在使用 `batch_generate_psta_same_ts` 任务处理历史数据时，虽然任务正常创建（返回 `total_buckets: 5-6`），但 `FeatureSnapshot` 表中没有任何数据生成。

## 根本原因

`AppleStockChecker/collectors.py` 中的 `collect_items_for_psta` 函数存在严重 bug：

### Bug 代码（修复前）

```python
# Line 62-69: 生成 15 个分钟刻度
ticks_dt: List = []
cur = ts_dt
for _ in range(15):
    ticks_dt.append(_floor_to_minute(cur))
    cur = cur - timedelta(minutes=1)
ticks_iso: List[str] = [_iso(x) for x in ticks_dt]

# Line 97: ❌ 错误！初始化为空字典
bucket_by_minute: Dict[str, List[int]] = {}

# Line 99-115: 遍历原始数据
for idx, r in enumerate(rows):
    # ...
    minute_iso = _iso(_floor_to_minute(rec_dt))
    if minute_iso not in bucket_by_minute:  # ❌ bucket_by_minute 是空的，总是 True
        # 不在 15 分钟窗口内的行，跳过
        continue  # ❌ 所有数据都被跳过了！
```

### 问题分析

1. **`ticks_iso`** 包含 15 个有效的分钟时间戳
2. **`bucket_by_minute`** 初始化为空字典 `{}`
3. 在遍历原始数据时，检查 `minute_iso not in bucket_by_minute`
4. 因为 `bucket_by_minute` 是空的，所以**所有数据的 `minute_iso` 都不在其中**
5. 所有数据都被 `continue` 跳过了

### 后果

```
原始数据查询（PurchasingShopPriceRecord）
  ↓ 有数据
遍历 rows 并填充 bucket_by_minute
  ↓ ❌ 所有数据被跳过
bucket_by_minute = {}（仍然是空的）
  ↓
bucket_minute_key = {}（也是空的）
  ↓
创建子任务，但 rows=[]（空数据）
  ↓
psta_process_minute_bucket 收到空数据
  ↓ do_agg=True（边界时刻）
_run_aggregation 执行，但没有数据
  ↓
_agg_feature_combos 查询数据库
  ↓ ❌ 数据库中也没有新数据（因为原始数据没写入）
FeatureSnapshot 没有数据生成
```

## 修复方案

### 修复代码（修复后）

```python
# Line 97: ✅ 正确！预先初始化所有 15 个分钟桶
bucket_by_minute: Dict[str, List[int]] = {tick: [] for tick in ticks_iso}

# Line 99-115: 遍历原始数据
for idx, r in enumerate(rows):
    # ...
    minute_iso = _iso(_floor_to_minute(rec_dt))
    if minute_iso not in bucket_by_minute:  # ✅ 现在只有不在窗口内的才会 True
        # 不在 15 分钟窗口内的行，跳过
        continue  # ✅ 只跳过真正不在窗口内的数据

    # ✅ 窗口内的数据正常添加
    bucket_by_minute[minute_iso].append(idx)
```

### 修复效果

```
原始数据查询（PurchasingShopPriceRecord）
  ↓ 有数据
bucket_by_minute 预先初始化 15 个 key
  ↓
遍历 rows 并填充 bucket_by_minute
  ↓ ✅ 数据正常添加到对应的分钟桶
bucket_by_minute = {
  "2025-10-03T23:00:00+00:00": [0, 5, 12, ...],
  "2025-10-03T23:01:00+00:00": [1, 3, 8, ...],
  ...
}
  ↓
创建子任务，rows=[...]（有数据）
  ↓
psta_process_minute_bucket 收到数据
  ↓ do_agg=True（边界时刻）
写入 PurchasingShopTimeAnalysis
  ↓
_run_aggregation 执行
  ↓
_agg_feature_combos 计算 4 种组合
  ↓ ✅ 写入 FeatureSnapshot
FeatureSnapshot 有数据！
```

## 验证方法

### 1. 检查修复是否生效

```bash
# 运行验证脚本
scripts/verify_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-04T01:00:00+00:00" \
  --verbose
```

### 2. 重新运行历史数据处理

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from uuid import uuid4
from datetime import datetime, timedelta, timezone

# 重新处理边界时刻
JST = timezone(timedelta(hours=9))
ts = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)

result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso=ts.isoformat(),
    query_window_minutes=15,
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=False,
    sequential=True,  # 顺序执行便于调试
)

print(result)
```

### 3. 查询 FeatureSnapshot 数据

```python
from AppleStockChecker.models import FeatureSnapshot
from django.utils import timezone
from datetime import datetime, timedelta

# 查询指定时间桶的数据
bucket_time = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)
snapshots = FeatureSnapshot.objects.filter(bucket=bucket_time)

print(f"找到 {snapshots.count()} 条 FeatureSnapshot 记录")

# 查看前几条
for snap in snapshots[:10]:
    print(f"{snap.scope} | {snap.name} | {snap.value}")
```

## 相关文件

### 修改的文件

- `AppleStockChecker/collectors.py` (line 97)
  - 修复前: `bucket_by_minute: Dict[str, List[int]] = {}`
  - 修复后: `bucket_by_minute: Dict[str, List[int]] = {tick: [] for tick in ticks_iso}`

### 新增的文件

- `scripts/verify_feature_snapshot.py` - 验证脚本
- `scripts/verify_feature_snapshot.sh` - Shell 包装脚本
- `docs/BUG_FIX_BUCKET_BY_MINUTE.md` - 本文档

## 为什么会出现这个 Bug？

### 原始意图

代码的原始意图是：
1. 生成 15 个分钟刻度（`ticks_iso`）
2. 遍历原始数据，将数据分配到对应的分钟桶
3. 如果数据的时间不在这 15 个分钟内，就跳过

### 实现错误

但实现时犯了逻辑错误：
- 检查 `minute_iso not in bucket_by_minute` 想要判断"是否在 15 个分钟窗口内"
- 但 `bucket_by_minute` 初始化为空字典，所以**无法判断**哪些分钟在窗口内
- 应该检查 `minute_iso not in ticks_iso`，或者预先初始化 `bucket_by_minute`

### 正确实现（两种方案）

**方案 1：预先初始化（已采用）**
```python
bucket_by_minute: Dict[str, List[int]] = {tick: [] for tick in ticks_iso}

# 然后检查
if minute_iso not in bucket_by_minute:  # 现在可以正确判断了
    continue
```

**方案 2：直接检查 ticks_iso**
```python
bucket_by_minute: Dict[str, List[int]] = {}

# 检查是否在 ticks_iso 中
if minute_iso not in ticks_iso:
    continue

# 添加数据
bucket_by_minute.setdefault(minute_iso, []).append(idx)
```

我们采用了方案 1，因为：
- 更清晰：`bucket_by_minute` 明确表示有哪些分钟桶
- 更安全：避免后续代码访问不存在的 key
- 更高效：不需要每次都调用 `setdefault`

## 影响范围

### 受影响的功能

- ✅ **FeatureSnapshot 计算**（4种组合）- **主要影响**
- ✅ 时间序列指标（基于 FeatureSnapshot 的历史数据）
- ⚠️  OverallBar/CohortBar（已禁用，但如果恢复也会受影响）

### 受影响的时间范围

**所有历史数据处理都受影响**，因为：
- `collect_items_for_psta` 是所有数据收集的入口
- 无论是实时处理还是历史重算，都会调用这个函数
- Bug 存在期间处理的数据都没有正确写入

### 需要重新处理的数据

**所有之前处理过的数据都需要重新处理**，特别是：
- 2025-10-03 到 2025-10-04 的数据（用户提到的时间范围）
- 任何其他使用 `batch_generate_psta_same_ts` 处理过的历史数据

## 重新处理历史数据

### 生成需要重新处理的时间戳列表

```python
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
start = datetime(2025, 10, 3, 23, 0, 0, tzinfo=UTC)
end = datetime(2025, 10, 4, 1, 0, 0, tzinfo=UTC)

# 生成所有边界时刻（15分钟间隔）
timestamps = []
current = start
while current <= end:
    timestamps.append(current.isoformat())
    current += timedelta(minutes=15)

print(f"需要重新处理 {len(timestamps)} 个时间点：")
for ts in timestamps:
    print(f"  {ts}")
```

### 批量重新处理

```python
import requests
import time
from uuid import uuid4

url = "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/"

for i, ts in enumerate(timestamps, 1):
    payload = {
        "timestamp_iso": ts,
        "agg_minutes": 15,
        "agg_mode": "boundary",
        "force_agg": False,
        "sequential": True,  # 顺序执行，避免数据库压力
    }

    try:
        response = requests.post(url, json=payload)
        result = response.json()
        print(f"[{i}/{len(timestamps)}] {ts} -> {result}")
    except Exception as e:
        print(f"[{i}/{len(timestamps)}] {ts} -> 错误: {e}")

    # 控制速率
    if i % 10 == 0:
        time.sleep(5)  # 每10个时间戳休息5秒
    time.sleep(1)  # 每个请求间隔1秒
```

## 预防措施

### 1. 添加单元测试

为 `collect_items_for_psta` 添加测试，确保：
- `bucket_by_minute` 包含所有 15 个分钟 key
- 数据正确分配到对应的分钟桶
- 不在窗口内的数据被正确跳过

### 2. 添加数据验证

在任务完成后验证：
- `bucket_by_minute` 不为空
- 至少有一些分钟桶包含数据
- 如果所有桶都是空的，记录警告日志

### 3. 监控数据生成

定期检查 FeatureSnapshot 表：
- 每个边界时刻是否有数据
- 数据量是否合理
- 是否有长时间的数据缺口

## 总结

这是一个**严重的逻辑错误**，导致：
- ❌ 所有原始数据在收集阶段就被丢弃
- ❌ 子任务收到的是空数据
- ❌ FeatureSnapshot 没有数据可计算
- ❌ 时间序列指标无法计算

修复非常简单，只需一行代码：
```python
# 修复前
bucket_by_minute: Dict[str, List[int]] = {}

# 修复后
bucket_by_minute: Dict[str, List[int]] = {tick: [] for tick in ticks_iso}
```

但影响范围很大，需要：
- ✅ 验证修复是否生效
- ✅ 重新处理所有受影响的历史数据
- ✅ 添加测试和监控防止再次发生
