# 顺序执行模式说明

## 概述

`batch_generate_psta_same_ts` 任务现在支持两种执行模式：

1. **并发模式（默认）**：使用 Celery `chord` 并发执行所有子任务，速度快但可能对数据库造成压力
2. **顺序模式**：逐个执行子任务，每个任务完成后再执行下一个，适合数据重算场景

## 使用方法

### API 调用

通过 `dispatch_psta_batch_same_ts` API 传递 `sequential=true` 参数：

#### 并发执行（默认）

```bash
curl -X POST "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp_iso": "2025-12-12T10:00:00+09:00",
    "agg_minutes": 15,
    "agg_mode": "boundary"
  }'
```

#### 顺序执行

```bash
curl -X POST "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp_iso": "2025-12-12T10:00:00+09:00",
    "agg_minutes": 15,
    "agg_mode": "boundary",
    "sequential": true
  }'
```

### 直接调用 Celery 任务

```python
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts

# 并发执行
result = batch_generate_psta_same_ts.apply_async(
    kwargs={
        "timestamp_iso": "2025-12-12T10:00:00+09:00",
        "agg_minutes": 15,
        "agg_mode": "boundary",
        "sequential": False,  # 默认值
    }
)

# 顺序执行
result = batch_generate_psta_same_ts.apply_async(
    kwargs={
        "timestamp_iso": "2025-12-12T10:00:00+09:00",
        "agg_minutes": 15,
        "agg_mode": "boundary",
        "sequential": True,  # 启用顺序执行
    }
)
```

## 执行模式对比

| 特性 | 并发模式 (sequential=false) | 顺序模式 (sequential=true) |
|------|----------------------------|---------------------------|
| **执行方式** | 所有子任务同时执行 | 逐个执行，每个完成后再执行下一个 |
| **速度** | 快 | 慢 |
| **数据库压力** | 高（多个连接同时写入） | 低（单个连接顺序写入） |
| **数据一致性** | 可能出现竞争 | 完全可控 |
| **适用场景** | 实时数据处理 | 历史数据重算、调试 |
| **进度监控** | 通过 chord 回调 | 实时报告每个子任务进度 |
| **错误处理** | 某个失败不影响其他任务 | 某个失败后继续执行剩余任务 |

## 应用场景

### 使用并发模式（默认）

适合以下场景：
- ✅ 实时数据处理（每分钟触发）
- ✅ 数据量不大
- ✅ 数据库资源充足
- ✅ 需要快速完成

### 使用顺序模式

适合以下场景：
- ✅ **历史数据重算**（避免数据库连接槽耗尽）
- ✅ 大批量数据处理
- ✅ 调试和验证（逐个观察结果）
- ✅ 数据库资源受限
- ✅ 需要严格的执行顺序

## 历史数据重算示例

对于大规模历史数据重算，建议使用顺序模式：

```python
from datetime import datetime, timedelta, timezone
import requests
import time

JST = timezone(timedelta(hours=9))
start = datetime(2025, 10, 23, 7, 0, 0, tzinfo=JST)
end = datetime(2025, 10, 23, 20, 25, 0, tzinfo=JST)

minutes = int((end - start).total_seconds() // 60)
timestamps = [
    (start + timedelta(minutes=i)).isoformat(timespec="seconds")
    for i in range(minutes - 1, -1, -1)
]

url = "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/"

for i, ts in enumerate(timestamps, 1):
    payload = {
        "timestamp_iso": ts,
        "agg_minutes": 15,
        "agg_mode": "boundary",
        "sequential": True,  # 使用顺序执行
    }

    response = requests.post(url, json=payload)
    print(f"[{i}/{len(timestamps)}] {ts} -> {response.json()}")

    # 间隔控制（顺序模式已经在内部控制，外部可以减少等待）
    if i % 100 == 0:
        time.sleep(10)  # 每100个时间戳休息10秒
    time.sleep(1)  # 每个请求间隔1秒
```

## 进度监控

### 并发模式

进度通过 WebSocket 或轮询 Celery 任务状态获取：
- 所有子任务提交后立即返回 `chord_id`
- 通过 `AsyncResult(chord_id).ready()` 检查是否完成
- 完成后通过 `psta_finalize_buckets` 回调获取最终结果

### 顺序模式

进度实时报告：
- 每完成一个子任务后通过 `notify_progress_all` 推送进度
- 返回结果中包含 `sequential: true` 标记
- 返回结果中包含完整的 `result` 字段（finalize 结果）

示例进度通知：
```json
{
  "status": "running",
  "step": "processing_bucket_5",
  "progress": 33,
  "current": 5,
  "total": 15,
  "timestamp": "2025-12-12T10:00:00+09:00"
}
```

## 性能影响

### 并发模式性能

假设有 15 个子任务（15分钟窗口，每分钟一个）：
- 总时间：约 5-10 秒（所有任务并发执行）
- 数据库连接：同时使用 15 个连接
- 内存占用：所有任务同时在内存中

### 顺序模式性能

假设有 15 个子任务，每个子任务耗时 2 秒：
- 总时间：约 30 秒（15 × 2 秒）
- 数据库连接：始终使用 1 个连接
- 内存占用：单个任务的内存占用

## 故障恢复

### 并发模式

- 如果某个子任务失败，其他任务继续执行
- 最终回调会收集所有结果（包括失败的）
- 需要检查最终结果中的 `error_hist` 字段

### 顺序模式

- 如果某个子任务失败，会记录错误但继续执行下一个
- 所有结果（包括失败的）都会收集到最终结果中
- 错误信息包含在返回结果的 `error_hist` 中

## 注意事项

1. **顺序模式会阻塞主任务**：
   - 主任务会等待所有子任务完成
   - 确保 Celery worker 有足够的超时设置
   - 建议设置 `CELERY_TASK_SOFT_TIME_LIMIT` 和 `CELERY_TASK_TIME_LIMIT`

2. **数据库连接**：
   - 顺序模式虽然只使用一个连接，但会长时间占用
   - 确保连接不会超时（调整 `CONN_MAX_AGE` 设置）

3. **任务队列**：
   - 顺序模式的主任务会长时间运行
   - 考虑使用专门的队列处理这类任务
   - 避免阻塞其他实时任务

4. **监控和调试**：
   - 顺序模式更容易调试（逐个观察结果）
   - 可以通过日志追踪每个子任务的执行

## 配置建议

### Celery 配置（settings.py）

```python
# 针对顺序执行的长任务
CELERY_TASK_SOFT_TIME_LIMIT = 3600  # 1小时软限制
CELERY_TASK_TIME_LIMIT = 7200  # 2小时硬限制

# 数据库连接池设置
DATABASES = {
    'default': {
        # ...
        'CONN_MAX_AGE': 600,  # 连接保持10分钟
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}
```

### 队列分离（可选）

```python
CELERY_TASK_ROUTES = {
    'AppleStockChecker.tasks.batch_generate_psta_same_ts': {
        'queue': 'sequential_tasks',  # 专门的队列
    },
}
```

启动专门的 worker：
```bash
celery -A AppleStockChecker worker -Q sequential_tasks -c 2 --loglevel=info
```

## 版本历史

- **v1.0** (2025-12-12): 添加 `sequential` 参数支持顺序执行模式
