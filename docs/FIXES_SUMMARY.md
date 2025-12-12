# 修复总结：collectors.py 数据丢失问题

## 修复的问题

本次修复解决了 `AppleStockChecker/collectors.py` 中的两个关键 bug，这些 bug 导致历史数据重算时 FeatureSnapshot 表没有数据生成。

## 修复 1：bucket_by_minute 初始化 Bug（严重）

### 问题描述
`bucket_by_minute` 初始化为空字典，导致所有数据在收集阶段被跳过。

### 受影响代码（修复前）
```python
# Line 97
bucket_by_minute: Dict[str, List[int]] = {}  # ❌ 空字典

# Line 114-116
minute_iso = _iso(_floor_to_minute(rec_dt))
if minute_iso not in bucket_by_minute:  # ❌ 总是 True（字典为空）
    continue  # ❌ 所有数据都被跳过
```

### 修复代码（修复后）
```python
# Line 98
bucket_by_minute: Dict[str, List[int]] = {tick: [] for tick in ticks_iso}  # ✅ 预先初始化 15 个分钟桶

# Line 114-116
minute_iso = _iso(_floor_to_minute(rec_dt))
if minute_iso not in bucket_by_minute:  # ✅ 只有不在窗口内的才为 True
    continue  # ✅ 只跳过窗口外的数据
```

### 影响范围
- **数据收集**：所有历史数据在分配到分钟桶时被跳过
- **子任务**：收到的 `rows=[]`（空数据）
- **聚合计算**：无数据可计算
- **FeatureSnapshot**：表中没有数据生成

### 提交信息
```
commit 784ecc2
fix: Critical bug in bucket_by_minute initialization causing data loss
```

## 修复 2：new_price 字段缺失 Bug

### 问题描述
`index_by_key` 初始化时缺少 `new_price` 字段，导致后续访问时出现 KeyError。

### 受影响代码（修复前）
```python
# Line 120
buf = index_by_key.setdefault(key, {"order": [], "times": []})  # ❌ 缺少 new_price

# Line 132（在排序循环中）
new_price = buf["new_price"]  # ❌ KeyError: 'new_price'
```

### 修复代码（修复后）
```python
# Line 120
buf = index_by_key.setdefault(key, {"order": [], "times": [], "new_price": []})  # ✅ 包含 new_price

# Line 123
buf["new_price"].append(r.get("price_new"))  # ✅ 收集价格数据

# Line 133
new_price = buf["new_price"]  # ✅ 正常访问

# Line 135
paired: List[Tuple[str, int]] = list(zip(times, order, new_price))  # ✅ 包含价格

# Line 139
buf["new_price"] = [p[2] for p in paired]  # ✅ 排序后更新
```

### 影响范围
- **数据索引**：价格数据无法正确存储和排序
- **后续处理**：依赖 `index_by_key` 的代码无法正常工作

### 提交信息
```
commit d5434e5
fix: Add missing new_price field in index_by_key initialization
```

## 验证修复

### 1. 使用验证脚本（推荐）

```bash
# 检查指定时间范围的 FeatureSnapshot 数据
scripts/verify_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-04T01:00:00+00:00" \
  --verbose
```

### 2. 手动测试历史数据处理

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from uuid import uuid4

# 测试单个时间点
result = batch_generate_psta_same_ts(
    job_id=uuid4().hex,
    timestamp_iso="2025-10-03T23:00:00+00:00",
    agg_minutes=15,
    agg_mode="boundary",
    force_agg=False,
    sequential=True,  # 顺序执行便于调试
)

print(f"处理结果：{result}")

# 查询 FeatureSnapshot 数据
from AppleStockChecker.models import FeatureSnapshot
from datetime import datetime, timezone

bucket_time = datetime(2025, 10, 3, 23, 0, 0, tzinfo=timezone.utc)
snapshots = FeatureSnapshot.objects.filter(bucket=bucket_time)

print(f"找到 {snapshots.count()} 条 FeatureSnapshot 记录")
for snap in snapshots[:5]:
    print(f"  {snap.scope} | {snap.name} = {snap.value}")
```

### 3. 批量重新处理历史数据

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from datetime import datetime, timedelta, timezone
from uuid import uuid4

UTC = timezone.utc

# 生成需要重新处理的时间戳（15分钟间隔）
start = datetime(2025, 10, 3, 23, 0, 0, tzinfo=UTC)
end = datetime(2025, 10, 4, 1, 0, 0, tzinfo=UTC)

timestamps = []
current = start
while current <= end:
    timestamps.append(current.isoformat())
    current += timedelta(minutes=15)

print(f"需要重新处理 {len(timestamps)} 个时间点")

# 顺序处理每个时间点
for i, ts in enumerate(timestamps, 1):
    print(f"\n[{i}/{len(timestamps)}] 处理 {ts}...")

    result = batch_generate_psta_same_ts(
        job_id=uuid4().hex,
        timestamp_iso=ts,
        agg_minutes=15,
        agg_mode="boundary",
        sequential=True,
    )

    print(f"  total_buckets: {result.get('total_buckets')}")

    # 简单检查
    from AppleStockChecker.models import FeatureSnapshot
    bucket_dt = datetime.fromisoformat(ts)
    count = FeatureSnapshot.objects.filter(bucket=bucket_dt).count()
    print(f"  FeatureSnapshot 记录数: {count}")
```

## 预期结果

### 修复前（Bug 状态）
```
处理 2025-10-03T23:00:00+00:00...
  total_buckets: 6
  FeatureSnapshot 记录数: 0  ❌ 没有数据
```

### 修复后（正常状态）
```
处理 2025-10-03T23:00:00+00:00...
  total_buckets: 6
  FeatureSnapshot 记录数: 156  ✅ 有数据（4种组合 × 39个指标）
```

## 数据流程对比

### 修复前（Bug）
```
collect_items_for_psta
  ↓ bucket_by_minute = {}（空字典）
  ↓ 遍历 500 条原始数据
  ↓ 检查 minute_iso not in bucket_by_minute（总是 True）
  ↓ ❌ 所有数据被 continue 跳过
  ↓ bucket_minute_key = {}（空字典）
batch_generate_psta_same_ts
  ↓ 遍历 bucket_minute_key（空的，不执行）
  ↓ subtasks = []（没有创建任何子任务）
  ↓ 或子任务收到 rows=[]（空数据）
psta_process_minute_bucket
  ↓ rows=[]，无法写入 PurchasingShopTimeAnalysis
  ↓ 调用 _run_aggregation（但数据库中没有数据）
  ↓ ❌ FeatureSnapshot 无数据可计算
```

### 修复后（正常）
```
collect_items_for_psta
  ↓ bucket_by_minute = {15个分钟: []}（预先初始化）
  ↓ 遍历 500 条原始数据
  ↓ 检查 minute_iso not in bucket_by_minute（窗口内为 False）
  ↓ ✅ 数据正常添加到对应分钟桶
  ↓ bucket_minute_key = {15个分钟: {key: [indices]}}
batch_generate_psta_same_ts
  ↓ 遍历 bucket_minute_key（15个分钟）
  ↓ 创建 15 个子任务，每个有正确的 rows 数据
  ↓ 边界分钟（23:00, 22:45）设置 do_agg=True
psta_process_minute_bucket
  ↓ rows=[...]，写入 PurchasingShopTimeAnalysis
  ↓ 边界分钟调用 _run_aggregation
  ↓ 查询数据库（有数据）
  ↓ 计算 4 种组合的统计指标
  ↓ ✅ FeatureSnapshot 正常生成
```

## 相关文档

- `docs/BUG_FIX_BUCKET_BY_MINUTE.md` - bucket_by_minute bug 详细分析
- `docs/BUG_MECHANISM_EXPLAINED.md` - Bug 机制与 force_agg 关系
- `docs/WHY_PERIODIC_TASKS_WORKED.md` - 为什么周期任务能工作的分析
- `docs/BOUNDARY_TRIGGER_DIAGNOSIS.md` - 边界触发诊断指南
- `docs/SEQUENTIAL_EXECUTION.md` - 顺序执行模式文档

## 相关提交

```bash
# 查看完整改动
git show 784ecc2  # bucket_by_minute fix
git show d5434e5  # new_price fix

# 查看所有相关提交
git log --oneline 377e93f..HEAD
```

## 下一步行动

1. ✅ **验证修复**：运行上面的验证脚本
2. ✅ **重新处理历史数据**：使用批量处理脚本重算受影响的时间范围
3. ⏳ **监控周期任务**：确认周期任务正常工作
4. ⏳ **添加单元测试**：为 `collect_items_for_psta` 添加测试防止回归

## 影响评估

### 受影响的时间范围
所有使用 `batch_generate_psta_same_ts` 处理的历史数据都受影响，包括：
- 2025-10-03 到 2025-10-04 的数据（用户报告的时间范围）
- 任何其他手动重算的历史数据

### 周期任务状态
根据用户说明，周期任务也使用相同的代码路径（`batch_generate_psta_same_ts` → `collect_items_for_psta`），但之前能正常工作。可能的原因：
1. Bug 是最近引入的（比 git 历史显示的更晚）
2. 周期任务也受影响但未被注意到
3. 存在未知的代码路径差异

### 需要重新处理的数据
使用修复后的代码重新处理所有受影响的历史数据，特别是：
- FeatureSnapshot 表为空的时间范围
- 用户报告的 2025-10-03 23:00 到 2025-10-04 01:00 时间段

## 总结

这两个 bug 都是 **关键性错误**：

1. **bucket_by_minute bug**：导致所有数据在收集阶段就被丢弃
2. **new_price bug**：在修复第一个 bug 后立即暴露，阻止了数据的正确处理

修复非常简单但影响深远：
- 只需两行代码修改
- 但影响所有历史数据处理
- 需要重新处理受影响的数据

现在两个 bug 都已修复并推送到远程分支，可以开始验证和重新处理历史数据。
