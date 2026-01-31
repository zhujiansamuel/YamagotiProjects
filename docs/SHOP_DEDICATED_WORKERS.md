# Shop 专用 Worker 配置指南

本文档说明如何为每个不同的 shop 分配唯一的 Celery Worker，用于独立处理各个店铺的数据清洗任务。

---

## 概述

### 功能说明

通过 **分离数据接收与数据清洗** 的架构，系统可以将不同店铺的数据清洗任务分配到各自专用的 Celery Worker 队列中，实现任务隔离和并行处理。

### 架构特点

- **数据接收任务**（webscraper 队列）：负责解析/拉取数据，存入 Redis，触发清洗任务
- **数据清洗任务**（shop_* 队列）：从 Redis 读取数据，执行清洗，写入数据库

### 使用场景

- **测试环境**: 为每个 shop 独立测试数据清洗，避免相互干扰
- **性能优化**: 多个 shop 的数据同时清洗时，充分利用多核 CPU
- **问题排查**: 隔离问题 shop，不影响其他 shop 的数据处理
- **并发控制**: 为不同 shop 设置不同的并发数

---

## 架构说明

### 新架构数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                      webscraper 队列（数据接收）                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  task_process_xlsx                                                   │
│    └─ 解析 xlsx/csv → 存 Redis → 触发 task_clean_shop_data          │
│                                                                      │
│  task_process_webscraper_job                                         │
│    └─ 拉取 WebScraper 数据 → 存 Redis → 触发 task_clean_shop_data   │
│                                                                      │
│  task_ingest_json_shop1                                              │
│    └─ 接收 JSON → 存 Redis → 触发 task_clean_shop1_json             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 触发清洗任务（动态路由）
┌─────────────────────────────────────────────────────────────────────┐
│                      shop_* 队列（数据清洗）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  shop_shop1 队列 ─── Worker 1                                        │
│    └─ task_clean_shop_data / task_clean_shop1_json                  │
│                                                                      │
│  shop_shop2 队列 ─── Worker 2                                        │
│    └─ task_clean_shop_data                                          │
│                                                                      │
│  shop_shop5 队列 ─── Worker 5（处理 shop5_1~4）                      │
│    └─ task_clean_shop_data                                          │
│                                                                      │
│  shop_shop6 队列 ─── Worker 6（处理 shop6_1~4）                      │
│    └─ task_clean_shop_data                                          │
│                                                                      │
│  ... 其他 shop 队列 ...                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 队列分组

| 队列名称 | 处理的店铺 | 说明 |
|----------|-----------|------|
| `shop_shop1` | shop1 | 独立队列 |
| `shop_shop2` | shop2 | 独立队列 |
| `shop_shop3` | shop3 | 独立队列 |
| `shop_shop4` | shop4 | 独立队列 |
| `shop_shop5` | shop5_1, shop5_2, shop5_3, shop5_4 | 合并队列（后续会合并清洗器） |
| `shop_shop6` | shop6_1, shop6_2, shop6_3, shop6_4 | 合并队列（后续会合并清洗器） |
| `shop_shop7` | shop7 | 独立队列 |
| `shop_shop8` | shop8 | 独立队列 |
| `shop_shop9` | shop9 | 独立队列 |
| `shop_shop10` | shop10 | 独立队列 |
| `shop_shop11` | shop11 | 独立队列 |
| `shop_shop12` | shop12 | 独立队列 |
| `shop_shop13` | shop13 | 独立队列 |
| `shop_shop14` | shop14 | 独立队列 |
| `shop_shop15` | shop15 | 独立队列 |
| `shop_shop16` | shop16 | 独立队列 |
| `shop_shop17` | shop17 | 独立队列 |
| `shop_shop18` | shop18 | 独立队列 |
| `shop_shop20` | shop20 | 独立队列 |

**共 19 个队列**

---

## 快速开始

### 1. 启动 Shop 专用 Workers

#### 方式 1: 批量启动（推荐）

使用批量启动脚本一次性为多个 shop 启动 workers：

```bash
# 编辑脚本配置（可选）
vim scripts/start_shop_workers.sh

# 启动所有配置的 workers
./scripts/start_shop_workers.sh
```

**输出示例:**
```
🚀 批量启动 Shop Workers...

✓ 将在 tmux session 'shop_workers' 中启动 workers
  查看 workers: tmux attach -t shop_workers
  切换窗口: Ctrl+b 然后按 0-9
  退出但保持运行: Ctrl+b 然后按 d

启动 Worker: shop1 (并发: 1)
  ✓ 已在 tmux 窗口 'shop1' 中启动
启动 Worker: shop2 (并发: 1)
  ✓ 已在 tmux 窗口 'shop2' 中启动
...

✅ 已启动 19 个 Shop Workers

📊 管理 Workers:
  查看所有 workers: tmux attach -t shop_workers
  列出所有窗口: tmux list-windows -t shop_workers
  关闭所有 workers: tmux kill-session -t shop_workers
```

#### 方式 2: 单独启动

为单个 shop 启动专用 worker：

```bash
# 为 shop1 启动 worker，并发数 1（默认）
./scripts/celery_worker_shop.sh shop1

# 为 shop2 启动 worker，并发数 2
./scripts/celery_worker_shop.sh shop2 2

# 为 shop5 启动 worker（处理 shop5_1~4），并发数 2
./scripts/celery_worker_shop.sh shop5 2
```

### 2. 使用 API 导入数据

数据接收任务会自动将清洗任务路由到对应的 shop 队列：

```bash
# 导入 xlsx 文件（清洗任务自动路由到 shop_shop1 队列）
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "files=@shop1.xlsx"
```

**响应示例:**
```json
{
  "accepted": true,
  "dry_run": false,
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "tasks": [
    {
      "file": "shop1.xlsx",
      "task_id": "abc123...",
      "source": "shop1",
      "cleaning_task_id": "def456...",
      "cleaning_queue": "shop_shop1"
    }
  ]
}
```

### 3. 查询清洗任务结果

数据接收任务返回的是 `cleaning_task_id`，需要查询该任务获取实际清洗结果：

```bash
# 查询清洗任务状态
curl "http://localhost:8000/AppleStockChecker/tasks/def456.../status/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 4. 监控任务执行

#### 使用 tmux 查看 worker 日志

```bash
# 连接到 tmux session
tmux attach -t shop_workers

# 在 tmux 中切换窗口
Ctrl+b 然后按 0    # 切换到窗口 0 (shop1)
Ctrl+b 然后按 1    # 切换到窗口 1 (shop2)

# 退出 tmux（worker 继续运行）
Ctrl+b 然后按 d
```

#### 使用 Flower 监控

```bash
# 启动 Flower
./scripts/flower.sh

# 访问 http://localhost:5555
```

### 5. 停止所有 Shop Workers

```bash
./scripts/stop_shop_workers.sh
```

---

## 详细配置

### Redis 临时存储

数据在接收任务和清洗任务之间通过 Redis 传递：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| Key 格式 | `ingest:temp:<batch_id>:<source_name>` | 唯一标识 |
| 存储格式 | JSON | DataFrame 序列化 |
| TTL | 3600 秒（1 小时） | 自动过期 |

**注意**: 清洗任务失败时，Redis 数据不会被删除，依赖 TTL 自动过期，方便排查问题。

### 队列命名规则

- **队列名称**: `shop_<shop_name>`
- **路由键**: 动态指定

**特殊规则**:
- `shop5-1`, `shop5_1` → 队列 `shop_shop5`
- `shop6-1`, `shop6_1` → 队列 `shop_shop6`

### Worker 配置

每个 shop worker 的默认配置：

```bash
celery -A YamagotiProjects worker \
    -Q shop_<shop_name> \              # 队列名称
    -l info \                          # 日志级别
    -c <concurrency> \                 # 并发数（默认 1）
    --max-tasks-per-child=100 \        # 处理 100 个任务后重启进程
    --hostname=worker_shop_<shop_name>@%h  # Worker 名称
```

---

## 任务说明

### 数据接收任务（webscraper 队列）

| 任务 | 说明 | 触发的清洗任务 |
|------|------|--------------|
| `task_process_xlsx` | 解析 xlsx/csv 文件 | `task_clean_shop_data` |
| `task_process_webscraper_job` | 拉取 WebScraper 数据 | `task_clean_shop_data` |
| `task_ingest_json_shop1` | 接收 shop1 JSON 数据 | `task_clean_shop1_json` |

### 数据清洗任务（shop_* 队列）

| 任务 | 说明 | 队列 |
|------|------|------|
| `task_clean_shop_data` | 通用清洗任务 | 动态路由到 `shop_<shop_name>` |
| `task_clean_shop1_json` | shop1 JSON 专用清洗 | `shop_shop1` |

---

## 使用示例

### 示例 1: 测试环境独立验证

**场景**: 为 3 个 shop 分别导入测试数据，互不干扰

```bash
# 1. 启动 3 个 shop 的 workers
./scripts/celery_worker_shop.sh shop1 1 &
./scripts/celery_worker_shop.sh shop2 1 &
./scripts/celery_worker_shop.sh shop3 1 &

# 2. 导入测试数据（清洗任务自动路由）
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?dry_run=1" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@test_shop1.xlsx" \
  -F "files=@test_shop2.xlsx" \
  -F "files=@test_shop3.xlsx"

# 3. 在 Flower 中分别查看各 shop 队列的处理情况
# http://localhost:5555
```

### 示例 2: 生产环境大批量导入

**场景**: 同时为多个 shop 导入历史数据

```bash
# 1. 启动所有 workers
./scripts/start_shop_workers.sh

# 2. 批量上传
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@shop1_history.xlsx" \
  -F "files=@shop2_history.xlsx" \
  -F "files=@shop5-1_history.xlsx" \
  -F "files=@shop5-2_history.xlsx"

# 3. 在 Flower 中监控所有队列
# http://localhost:5555
```

### 示例 3: 为特定 shop 单独调试

**场景**: shop3 数据清洗有问题，单独启动 worker 进行调试

```bash
# 1. 只为 shop3 启动 worker，增加日志级别
celery -A YamagotiProjects worker \
    -Q shop_shop3 \
    -l debug \
    -c 1

# 2. 只上传 shop3 的数据
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?dry_run=1" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@shop3_test.xlsx"

# 3. 查看详细的 debug 日志
```

---

## 性能对比

### 场景: 同时导入 5 个 shop 的数据，每个文件 10000 行

| 模式 | Worker 配置 | 总耗时 | 吞吐量 |
|------|------------|--------|--------|
| 旧架构（单队列） | 1 个 webscraper worker (并发 2) | ~25 分钟 | 2000 行/分钟 |
| 新架构（分离队列） | 1 个 webscraper + 5 个 shop workers | ~10 分钟 | 5000 行/分钟 |
| 新架构（高并发） | 1 个 webscraper + 5 个 shop workers (各并发 2) | ~5 分钟 | 10000 行/分钟 |

**优势**:
- ✅ **并行处理**: 多个 shop 同时清洗，提升 2-5 倍性能
- ✅ **任务隔离**: 某个 shop 的清洗失败不影响其他 shop
- ✅ **资源控制**: 为重要 shop 分配更多并发数
- ✅ **问题定位**: 快速识别哪个 shop 的清洗有问题
- ✅ **数据接收快速**: 接收任务轻量化，快速响应

---

## 常见问题

### Q1: 清洗任务显示 "Redis 数据不存在或已过期"

**原因**: 清洗任务启动时，Redis 中的临时数据已过期（TTL 默认 1 小时）

**解决方案**:
1. 确保 shop worker 已启动并正常运行
2. 检查 Redis 连接是否正常
3. 如需增加 TTL，修改 `redis_temp_storage.py` 中的 `DEFAULT_TTL`

### Q2: 如何查看某个 shop 队列的长度？

```bash
# 使用 Redis CLI
redis-cli LLEN shop_shop1
redis-cli LLEN shop_shop5

# 或使用 Celery 命令
celery -A YamagotiProjects inspect active
```

### Q3: 可以动态添加新的 shop worker 吗？

**可以**。无需重启现有 workers，只需为新 shop 启动专用 worker：

```bash
# 新增 shop4 的 worker
./scripts/celery_worker_shop.sh shop4 1
```

### Q4: shop5-1 和 shop5_1 有什么区别？

两者等价，系统会自动规范化：
- `shop5-1` (WebScraper/文件名) → 队列 `shop_shop5`
- `shop5_1` (清洗器名称) → 队列 `shop_shop5`

### Q5: 如何查看清洗任务的详细结果？

数据接收任务返回 `cleaning_task_id`，使用该 ID 查询清洗结果：

```bash
# 查询任务状态
celery -A YamagotiProjects result <cleaning_task_id>

# 或通过 API
curl "http://localhost:8000/AppleStockChecker/tasks/<cleaning_task_id>/status/"
```

### Q6: 如何在 Docker 环境中使用？

为每个 shop 创建 Docker Compose 服务配置：

```yaml
# docker-compose.yml
services:
  # 数据接收 worker
  celery_worker_webscraper:
    build: .
    command: celery -A YamagotiProjects worker -Q webscraper -c 2
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  # Shop 清洗 workers
  celery_worker_shop1:
    build: .
    command: celery -A YamagotiProjects worker -Q shop_shop1 -c 1
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  celery_worker_shop2:
    build: .
    command: celery -A YamagotiProjects worker -Q shop_shop2 -c 1
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  # shop5 处理 shop5_1~4
  celery_worker_shop5:
    build: .
    command: celery -A YamagotiProjects worker -Q shop_shop5 -c 2
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
```

---

## 最佳实践

### 1. 并发数配置建议

| Shop 类型 | 数据量 | 推荐并发数 | 说明 |
|----------|--------|-----------|------|
| 测试 shop | 小 (< 1000 行) | 1 | 节省资源 |
| 普通 shop | 中 (1000-10000 行) | 1-2 | 平衡性能和资源 |
| 大型 shop | 大 (> 10000 行) | 2-4 | 加快处理速度 |
| 合并 shop (shop5/6) | 多数据源 | 2-4 | 处理多个子店铺 |

### 2. 资源监控

定期检查：
- CPU 使用率 (`top` / `htop`)
- 内存使用率
- 数据库连接数
- Redis 内存使用

### 3. 日志管理

```bash
# 创建日志目录
mkdir -p logs

# 启动时重定向日志
./scripts/celery_worker_shop.sh shop1 1 > logs/shop1.log 2>&1 &

# 定期清理旧日志
find logs/ -name "*.log" -mtime +7 -delete
```

### 4. 错误处理

- 启用任务重试机制（已在 `task_clean_shop_data` 中配置）
- 设置合理的超时时间（清洗任务 9000 秒）
- 监控失败任务并及时排查
- Redis 临时数据 1 小时后自动过期

---

## 技术细节

### 代码实现位置

- **Redis 临时存储**: `AppleStockChecker/utils/redis_temp_storage.py`
- **队列映射**: `AppleStockChecker/utils/shop_queue_mapping.py`
- **任务定义**: `AppleStockChecker/tasks/webscraper_tasks.py`
- **Celery 配置**: `YamagotiProjects/celery.py`

### 数据流程

```python
# 1. 数据接收任务（webscraper 队列）
task_process_xlsx:
    df = _read_tabular(filename, file_bytes)  # 解析文件
    redis_key = store_dataframe(batch_id, source_name, df)  # 存 Redis
    task_clean_shop_data.apply_async(queue=get_shop_queue(source_name))  # 触发清洗

# 2. 数据清洗任务（shop_* 队列）
task_clean_shop_data:
    df = retrieve_dataframe(redis_key)  # 从 Redis 读取
    df_clean = run_cleaner(cleaner_name, df)  # 执行清洗
    _write_records_to_db(df_clean, ...)  # 写入数据库
```

---

## 参考资料

- [Celery 路由文档](https://docs.celeryq.dev/en/stable/userguide/routing.html)
- [Celery Workers 文档](https://docs.celeryq.dev/en/stable/userguide/workers.html)
- [项目 Celery 队列配置](./CELERY_QUEUES.md)

---

## 更新历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-12-12 | v1.0 | 初始版本 - Shop 专用 Worker 配置 |
| 2026-01-31 | v2.0 | 重构 - 分离数据接收与数据清洗任务，使用 Redis 临时存储 |
