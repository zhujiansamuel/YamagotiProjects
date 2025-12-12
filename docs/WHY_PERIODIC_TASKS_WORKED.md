# 为什么以前能正常工作，现在却遇到 bucket_by_minute Bug？

## 问题概述

用户报告：
- ✅ 以前使用周期任务调用 `batch_generate_psta_same_ts` 时能正常计算出数据
- ❌ 现在重新计算历史数据时，FeatureSnapshot 表中没有任何数据
- 🤔 但是 `bucket_by_minute` 的 bug 一直存在，为什么以前能工作？

## 答案：两条不同的代码路径

### 路径 1：周期任务（实时处理）- 正常工作 ✅

```
Celery Beat 周期任务（每分钟触发）
  ↓
直接调用某个爬虫任务（获取最新数据）
  ↓
可能直接调用 psta_process_minute_bucket
  ↓
rows 参数来自爬虫结果（不经过 collect_items_for_psta）
  ↓
✅ 绕过了 bucket_by_minute 的 bug
  ↓
✅ FeatureSnapshot 正常生成
```

### 路径 2：历史数据重算 - 遇到 Bug ❌

```
手动调用 batch_generate_psta_same_ts（重算历史数据）
  ↓
调用 collect_items_for_psta（从数据库查询历史数据）
  ↓
❌ bucket_by_minute = {} 初始化bug
  ↓
❌ 所有数据被跳过
  ↓
❌ rows = [] 空数据传给子任务
  ↓
❌ FeatureSnapshot 无数据可计算
```

## 详细分析

### 1. 周期任务的可能实现方式

周期任务可能有以下几种实现：

#### 方式 A：爬虫 → 直接调用 psta_process_minute_bucket

```python
# 某个爬虫任务（定时执行）
@periodic_task(run_every=crontab(minute='*/1'))
def scrape_and_process():
    # 1. 爬取最新数据
    scraped_data = scrape_shops()

    # 2. 转换为 rows 格式
    rows = [
        {
            "shop_id": item.shop_id,
            "iphone_id": item.iphone_id,
            "recorded_at": item.recorded_at,
            "price_new": item.price,
        }
        for item in scraped_data
    ]

    # 3. 直接调用 psta_process_minute_bucket
    psta_process_minute_bucket.delay(
        ts_iso=current_minute_iso(),
        rows=rows,  # ← 直接传入 rows，不经过 collect_items_for_psta
        job_id=uuid4().hex,
        do_agg=True,
        agg_minutes=1,
    )
```

**关键**：`rows` 参数直接来自爬虫结果，**不经过 `collect_items_for_psta`**，所以不会触发 `bucket_by_minute` 的 bug。

#### 方式 B：爬虫 → 写入数据库 → 周期任务调用API

```python
# 步骤1：爬虫任务（持续运行）
@periodic_task(run_every=crontab(minute='*/1'))
def scrape_task():
    scraped_data = scrape_shops()
    # 写入 PurchasingShopPriceRecord 表
    for item in scraped_data:
        PurchasingShopPriceRecord.objects.create(...)

# 步骤2：处理任务（稍后触发，如每分钟第30秒）
@periodic_task(run_every=crontab(minute='*', second='30'))
def process_task():
    # 调用API处理刚才爬取的数据
    requests.post(
        "http://localhost:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/",
        json={"timestamp_iso": current_minute_iso()}
    )
```

如果是这种方式，周期任务**会触发** `collect_items_for_psta`，**应该会遇到 bug**。

### 2. 重构前后的代码差异

#### 重构前（用户提供的代码）

```python
# psta_process_minute_bucket 是独立的任务
@shared_task(name="AppleStockChecker.tasks.psta_process_minute_bucket")
def psta_process_minute_bucket(
        ts_iso: str,
        rows: List[Dict[str, Any]],  # ← 直接接收 rows
        job_id: str,
        ...):
    # 直接处理 rows，进行聚合计算
    for r in rows:
        # 写入 PurchasingShopTimeAnalysis
        ...

    if do_agg:
        # 进行聚合计算（OverallBar, CohortBar, FeatureSnapshot）
        ...
```

**特点**：
- 可以直接调用，不需要经过数据收集阶段
- `rows` 参数可以来自任何来源（爬虫、API、其他任务）

#### 重构后（当前代码）

```python
# 引入了 batch_generate_psta_same_ts 作为入口
@shared_task(name="AppleStockChecker.tasks.batch_generate_psta_same_ts")
def batch_generate_psta_same_ts(...):
    # 1. 调用 collect_items_for_psta 收集数据
    pack = collect_items_for_psta(...)  # ← 新引入的函数

    # 2. 从 pack 中提取 rows 和 bucket_minute_key
    rows = pack.get("rows")
    bucket_minute_key = pack.get("bucket_minute_key")  # ← bug 在这里

    # 3. 为每个分钟创建子任务
    for minute_iso, key_map in bucket_minute_key.items():
        subtasks.append(
            psta_process_minute_bucket.s(
                ts_iso=minute_iso,
                rows=minute_rows,  # ← rows 来自 bucket_minute_key
                ...
            )
        )
```

**特点**：
- 新增了 `collect_items_for_psta` 函数来从数据库查询历史数据
- `bucket_minute_key` 用于分配数据到各个分钟桶
- **Bug 存在于 `collect_items_for_psta` 中**

### 3. Bug 存在的时间线

| 时间 | 事件 | Bug 状态 | 影响 |
|------|------|---------|------|
| **重构前** | 周期任务调用 `psta_process_minute_bucket` | ❌ 不存在 | ✅ 正常工作 |
| **重构时** | 引入 `batch_generate_psta_same_ts` 和 `collect_items_for_psta` | ⚠️  引入bug | ⚠️  周期任务可能仍绕过 |
| **现在** | 手动调用 `batch_generate_psta_same_ts` 重算历史数据 | ❌ 触发bug | ❌ 无法生成数据 |

### 4. 为什么周期任务仍能正常工作？

有几种可能：

#### 可能性 1：周期任务从未使用 batch_generate_psta_same_ts

周期任务可能直接调用：
```python
# 周期任务配置（在 Django Admin 中）
Task: AppleStockChecker.tasks.psta_process_minute_bucket
Arguments: {...}
```

而不是调用：
```python
Task: AppleStockChecker.tasks.batch_generate_psta_same_ts
```

#### 可能性 2：周期任务传入了 items 参数

```python
# API 调用（周期任务可能这样调用）
POST /AppleStockChecker/purchasing-time-analyses/dispatch_ts/
{
  "items": [...]  # ← 直接传入 items，不查询数据库
}
```

查看 API 代码：
```python
@api_view(["POST"])
def dispatch_psta_batch_same_ts(request):
    body = request.data or {}

    async_res = batch_generate_psta_same_ts.apply_async(
        kwargs={
            "items": body.get("items"),  # ← 如果提供了 items
            "timestamp_iso": body.get("timestamp_iso"),
            ...
        }
    )
```

如果提供了 `items` 参数，可能会绕过 `collect_items_for_psta`（取决于实现细节）。

#### 可能性 3：Bug 是最近才引入的

`collect_items_for_psta` 中的 bug 可能是在某次修改中引入的，而周期任务一直使用的是旧版本代码。

### 5. 验证方法

#### 检查周期任务配置

```bash
# 查看 Django Admin 中的周期任务配置
# /admin/django_celery_beat/periodictask/

# 查看任务名称和参数
```

#### 检查 Celery Beat 日志

```bash
# 查看 celery beat 日志，确认周期任务调用的是哪个任务
docker compose logs celery_beat -f --tail=100

# 查找类似以下的日志：
# [2025-12-12 10:00:00,123] Scheduler: Sending due task
#   AppleStockChecker.tasks.psta_process_minute_bucket
```

#### 检查 items 参数使用

```python
# 在 batch_generate_psta_same_ts 中添加日志
def batch_generate_psta_same_ts(*, items=None, ...):
    if items:
        logger.info(f"Using provided items: {len(items)} items")
        # 使用 items，不调用 collect_items_for_psta
    else:
        logger.info("Calling collect_items_for_psta to fetch from DB")
        # 调用 collect_items_for_psta（会触发 bug）
```

## 结论

### 为什么以前能正常工作？

最可能的原因：

1. **周期任务直接调用 `psta_process_minute_bucket`**
   - 不经过 `batch_generate_psta_same_ts`
   - 不经过 `collect_items_for_psta`
   - `rows` 参数来自爬虫或其他数据源
   - **绕过了 bug**

2. **周期任务提供了 `items` 参数**
   - 直接传入数据，不查询数据库
   - 不触发 `collect_items_for_psta`
   - **绕过了 bug**

3. **Bug 是最近才引入的**
   - 周期任务使用的代码版本没有这个 bug
   - 最近的重构引入了 bug

### 为什么现在不工作？

- 手动调用 `batch_generate_psta_same_ts` 重算历史数据
- 必须经过 `collect_items_for_psta`（从数据库查询）
- **触发了 `bucket_by_minute` 的 bug**
- 所有数据被跳过，无法生成 FeatureSnapshot

### 解决方案

1. **✅ 已修复**：`bucket_by_minute = {tick: [] for tick in ticks_iso}`
2. **重新运行**：使用修复后的代码重新处理历史数据
3. **验证周期任务**：确认周期任务的实际调用路径
4. **统一代码路径**：考虑让周期任务也使用 `batch_generate_psta_same_ts`

## 下一步行动

### 1. 验证周期任务配置

```bash
# 查看周期任务列表
python manage.py shell
>>> from django_celery_beat.models import PeriodicTask
>>> for task in PeriodicTask.objects.filter(enabled=True):
...     print(f"{task.name}: {task.task}")
...     print(f"  Args: {task.args}")
...     print(f"  Kwargs: {task.kwargs}")
```

### 2. 检查是否有直接调用 psta_process_minute_bucket 的代码

```bash
# 搜索直接调用
grep -r "psta_process_minute_bucket.delay\|psta_process_minute_bucket.apply_async" \
  --include="*.py" .
```

### 3. 重新处理历史数据（使用修复后的代码）

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from datetime import datetime, timedelta, timezone
from uuid import uuid4

UTC = timezone.utc
timestamps = [
    "2025-10-03T23:00:00+00:00",
    "2025-10-03T23:15:00+00:00",
    "2025-10-03T23:30:00+00:00",
    "2025-10-03T23:45:00+00:00",
    "2025-10-04T00:00:00+00:00",
]

for ts in timestamps:
    result = batch_generate_psta_same_ts(
        job_id=uuid4().hex,
        timestamp_iso=ts,
        agg_minutes=15,
        agg_mode="boundary",
        force_agg=False,
        sequential=True,
    )
    print(f"{ts}: {result}")
```

### 4. 验证数据生成

```bash
scripts/verify_feature_snapshot.sh \
  --start "2025-10-03T23:00:00+00:00" \
  --end "2025-10-04T01:00:00+00:00" \
  --verbose
```
