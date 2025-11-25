# Celery 队列配置与 WebScraper 专用 Worker

## 概述

本项目配置了 **Celery 队列隔离**，将 WebScraper 相关任务路由到专用队列，由 **2 个专用 Worker** 处理。这样可以：

- 防止 WebScraper 任务阻塞其他关键任务
- 提高任务处理效率
- 更好地控制资源使用
- 便于监控和故障排查

---

## 队列架构

```
Redis Broker
    ↓
├── default 队列 (4 workers)
│   ├── 时间对齐任务
│   ├── 数据聚合任务
│   └── 其他通用任务
│
└── webscraper 队列 (2 workers)
    ├── task_process_webscraper_job
    ├── task_process_xlsx
    └── task_ingest_json_shop1
```

---

## 队列配置

### 1. 队列定义

| 队列名称 | Worker 数量 | 并发数 | 任务类型 | 说明 |
|---------|------------|--------|---------|------|
| `default` | 1 | 4 | 通用任务 | 处理所有未指定队列的任务 |
| `celery` | 1 (共享) | - | 系统任务 | Celery 内部任务 |
| `webscraper` | 1 | 2 | WebScraper | 专门处理爬虫数据导入 |

### 2. 任务路由

在 `YamagotiProjects/celery.py` 中配置：

```python
app.conf.task_routes = {
    # WebScraper 相关任务 → webscraper 队列
    "AppleStockChecker.tasks.webscraper_tasks.task_process_webscraper_job": {
        "queue": "webscraper",
        "routing_key": "webscraper.process_job",
    },
    "AppleStockChecker.tasks.webscraper_tasks.task_process_xlsx": {
        "queue": "webscraper",
        "routing_key": "webscraper.process_xlsx",
    },
    "AppleStockChecker.tasks.webscraper_tasks.task_ingest_json_shop1": {
        "queue": "webscraper",
        "routing_key": "webscraper.ingest_json",
    },
    # 其他任务 → default 队列（默认）
}
```

### 3. Worker 配置

#### 默认队列 Worker
```bash
celery -A YamagotiProjects worker \
    -Q default,celery \
    -l info \
    -c 4 \
    --max-tasks-per-child=1000
```

- **队列**: `default` + `celery`（系统任务）
- **并发数**: 4（可根据 CPU 核心数调整）
- **任务限制**: 每个 Worker 进程最多处理 1000 个任务后重启（防止内存泄漏）

#### WebScraper 专用 Worker
```bash
celery -A YamagotiProjects worker \
    -Q webscraper \
    -l info \
    -c 2 \
    --max-tasks-per-child=100
```

- **队列**: `webscraper`（专用）
- **并发数**: 2（适合 I/O 密集型任务）
- **任务限制**: 每个 Worker 进程最多处理 100 个任务后重启

---

## 使用方法

### 本地开发

#### 方式 1: 使用脚本启动

```bash
# 终端 1: 启动默认队列 Worker
./scripts/celery_worker_default.sh

# 终端 2: 启动 WebScraper 专用 Worker
./scripts/celery_worker_webscraper.sh

# 终端 3: 启动定时任务调度器
./scripts/celery_beat.sh

# 终端 4: 启动 Flower 监控（可选）
./scripts/flower.sh
```

#### 方式 2: 手动启动

```bash
# 默认队列 Worker
celery -A YamagotiProjects worker -Q default,celery -l info -c 4

# WebScraper 专用 Worker
celery -A YamagotiProjects worker -Q webscraper -l info -c 2

# Celery Beat
celery -A YamagotiProjects beat -l info

# Flower
celery -A YamagotiProjects flower --port=5555
```

### Docker Compose 生产环境

```bash
# 启动所有服务（包含所有 Workers）
./scripts/prod_up.sh

# 查看 Worker 状态
docker compose ps

# 查看 WebScraper Worker 日志
docker compose logs -f celery_worker_webscraper

# 查看默认 Worker 日志
docker compose logs -f celery_worker_default
```

---

## 发送任务到队列

### 方式 1: 自动路由（推荐）

任务会根据 `celery.py` 中的路由配置自动分配到对应队列：

```python
# 自动路由到 webscraper 队列
from AppleStockChecker.tasks.webscraper_tasks import task_process_webscraper_job

result = task_process_webscraper_job.delay(
    job_id="34172550",
    source_name="shop3",
    dry_run=False
)
```

### 方式 2: 手动指定队列

```python
# 手动指定队列（覆盖路由配置）
result = task_process_webscraper_job.apply_async(
    args=("34172550", "shop3"),
    kwargs={"dry_run": False},
    queue="webscraper"  # 显式指定队列
)
```

---

## 监控与调试

### 1. Flower Web UI

访问 `http://localhost:5555` 查看：

- **Workers**: 所有 Worker 状态、并发数、活跃任务
- **Tasks**: 任务执行历史、成功率、失败原因
- **Queues**: 队列长度、任务分布
- **Monitor**: 实时任务流

### 2. 命令行监控

```bash
# 查看所有活跃任务
celery -A YamagotiProjects inspect active

# 查看已注册任务
celery -A YamagotiProjects inspect registered

# 查看队列长度
celery -A YamagotiProjects inspect stats

# 查看 Worker 状态
celery -A YamagotiProjects status

# 查看特定队列的任务
celery -A YamagotiProjects inspect active -d worker_webscraper@hostname
```

### 3. Redis 监控

```bash
# 查看所有队列
redis-cli KEYS "celery*"

# 查看 webscraper 队列长度
redis-cli LLEN webscraper

# 查看 default 队列长度
redis-cli LLEN default

# 实时监控 Redis 命令
redis-cli MONITOR
```

---

## 性能调优

### 1. 并发数调整

根据任务类型和服务器资源调整：

| 任务类型 | 推荐并发数 | 说明 |
|---------|----------|------|
| CPU 密集型 | CPU 核心数 | 数据处理、计算 |
| I/O 密集型 | CPU 核心数 × 2-4 | 网络请求、文件读写 |
| 混合型 | CPU 核心数 × 1.5 | 兼顾 CPU 和 I/O |

```bash
# 调整并发数
celery -A YamagotiProjects worker -Q webscraper -c 4  # 从 2 增加到 4
```

### 2. 任务优先级

```python
# 高优先级任务
task_process_webscraper_job.apply_async(
    args=("34172550", "shop3"),
    priority=9  # 0-10，数字越大优先级越高
)

# 低优先级任务
task_process_xlsx.apply_async(
    args=[...],
    priority=1
)
```

### 3. 任务超时

在 `settings.py` 中配置：

```python
CELERY_TASK_TIME_LIMIT = 600        # 硬超时（10分钟）
CELERY_TASK_SOFT_TIME_LIMIT = 540   # 软超时（9分钟，可捕获异常）
```

针对特定任务：

```python
@shared_task(
    soft_time_limit=300,  # 5分钟软超时
    time_limit=360,       # 6分钟硬超时
)
def my_task():
    pass
```

### 4. 任务重试

```python
@shared_task(
    autoretry_for=(OperationalError,),  # 自动重试的异常类型
    retry_backoff=30,                   # 重试间隔（秒）: 30, 60, 120, ...
    retry_backoff_max=600,              # 最大重试间隔（10分钟）
    retry_jitter=True,                  # 添加随机抖动
    max_retries=5,                      # 最大重试次数
)
def my_task():
    pass
```

---

## 故障排查

### 问题 1: 任务卡在队列不执行

**原因**：
- 没有 Worker 监听该队列
- Worker 已停止或崩溃

**解决方案**：
```bash
# 检查 Worker 状态
celery -A YamagotiProjects status

# 检查队列长度
redis-cli LLEN webscraper

# 重启 Worker
./scripts/celery_worker_webscraper.sh
```

### 问题 2: Worker 内存持续增长

**原因**：
- 内存泄漏
- 没有设置 `max_tasks_per_child`

**解决方案**：
```bash
# 设置任务限制，定期重启 Worker 进程
celery -A YamagotiProjects worker -Q webscraper --max-tasks-per-child=100
```

### 问题 3: 任务执行缓慢

**原因**：
- 并发数不足
- 数据库连接池耗尽
- 任务本身耗时长

**解决方案**：
```bash
# 增加并发数
celery -A YamagotiProjects worker -Q webscraper -c 4

# 启用 PgBouncer 连接池
export USE_PGBOUNCER=true

# 查看任务执行时间分布
# 在 Flower 中查看 Runtime 统计
```

### 问题 4: 任务重复执行

**原因**：
- 没有设置 `CELERY_TASK_ACKS_LATE`
- Worker 在任务完成前崩溃

**解决方案**：
```python
# settings.py
CELERY_TASK_ACKS_LATE = True                # 任务完成后才确认
CELERY_TASK_REJECT_ON_WORKER_LOST = True    # Worker 崩溃时重新入队
```

---

## 最佳实践

### 1. 任务设计

✅ **推荐**：
- 任务幂等（可重复执行）
- 任务粒度适中（1-10分钟）
- 避免长时间占用数据库连接
- 使用 `batch_id` 追踪批次

❌ **避免**：
- 任务过大（超过 30 分钟）
- 任务过小（频繁调度开销）
- 在任务中使用全局状态
- 不处理异常

### 2. 队列隔离

按业务重要性和资源需求隔离队列：

```python
# 关键业务任务 → 高优先级队列
"critical_task": {"queue": "critical", "priority": 10}

# 后台任务 → 低优先级队列
"background_task": {"queue": "background", "priority": 1}

# 资源密集型任务 → 专用队列（低并发）
"heavy_task": {"queue": "heavy", "priority": 5}
```

### 3. 监控告警

设置告警规则：
- 队列长度超过阈值（如 1000）
- 任务失败率超过 5%
- Worker 离线超过 5 分钟
- 任务平均执行时间异常

---

## WebScraper 任务说明

### 任务列表

| 任务名称 | 功能 | 输入 | 耗时 |
|---------|------|------|------|
| `task_process_webscraper_job` | 处理 WebScraper 导出数据 | `job_id`, `source_name` | 1-5 分钟 |
| `task_process_xlsx` | 处理 Excel/CSV 文件 | `file_bytes`, `filename`, `source_name` | 30秒-3分钟 |
| `task_ingest_json_shop1` | 处理 JSON 格式店铺数据 | `records`, `opts` | 30秒-2分钟 |

### 调用示例

```python
from AppleStockChecker.tasks.webscraper_tasks import (
    task_process_webscraper_job,
    task_process_xlsx,
    task_ingest_json_shop1,
)

# 示例 1: 处理 WebScraper Job
result = task_process_webscraper_job.delay(
    job_id="34172550",
    source_name="shop3",
    dry_run=False,
    create_shop=True,
    dedupe=True,
    upsert=False,
    batch_id="batch_20251125_001"
)

# 示例 2: 处理 Excel 文件
with open("data.xlsx", "rb") as f:
    file_bytes = f.read()

result = task_process_xlsx.delay(
    file_bytes=file_bytes,
    filename="data.xlsx",
    source_name="shop3",
    dry_run=False
)

# 示例 3: 处理 JSON 数据
records = [
    {"model": "iPhone 17 Pro", "capacity": "256GB", "price": 150000},
    # ...
]
opts = {
    "dry_run": False,
    "dedupe": True,
    "upsert": False,
    "batch_id": "batch_20251125_002",
    "source": "shop1"
}
result = task_ingest_json_shop1.delay(records, opts)

# 获取任务结果
print(result.get(timeout=600))  # 等待最多 10 分钟
```

---

## 参考资料

- [Celery 官方文档](https://docs.celeryq.dev/)
- [Celery 最佳实践](https://docs.celeryq.dev/en/stable/userguide/optimizing.html)
- [Flower 监控工具](https://flower.readthedocs.io/)

---

## 更新历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-11-25 | v1.0 | 初始版本 - WebScraper 专用队列配置 |
