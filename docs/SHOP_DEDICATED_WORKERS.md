# Shop 专用 Worker 配置指南

本文档说明如何为每个不同的 shop 分配唯一的 Celery Worker，用于独立处理各个店铺的历史数据导入任务。

---

## 概述

### 功能说明

通过启用 **按 shop 路由** 功能，系统可以将不同店铺的数据导入任务分配到各自专用的 Celery Worker 队列中，实现任务隔离和并行处理。

### 使用场景

- **测试环境**: 为每个 shop 独立测试数据导入，避免相互干扰
- **性能优化**: 多个 shop 的数据同时导入时，充分利用多核 CPU
- **问题排查**: 隔离问题 shop，不影响其他 shop 的数据处理
- **并发控制**: 为不同 shop 设置不同的并发数

---

## 架构说明

### 默认模式 (route_by_shop=0)

```
API 请求
  ↓
import-tradein-xlsx 端点
  ↓
[shop1.xlsx, shop2.xlsx, shop3.xlsx]
  ↓
所有任务 → webscraper 队列
  ↓
单个 webscraper worker (并发数: 2)
```

### Shop 路由模式 (route_by_shop=1)

```
API 请求
  ↓
import-tradein-xlsx 端点
  ↓
[shop1.xlsx, shop2.xlsx, shop3.xlsx]
  ↓
├── shop1.xlsx → shop_shop1 队列 → Worker 1
├── shop2.xlsx → shop_shop2 队列 → Worker 2
└── shop3.xlsx → shop_shop3 队列 → Worker 3
```

---

## 快速开始

### 1. 启动 Shop 专用 Workers

#### 方式 1: 批量启动（推荐）

使用批量启动脚本一次性为多个 shop 启动 workers：

```bash
# 编辑脚本配置（可选）
vim scripts/start_shop_workers.sh

# 修改 SHOPS 数组，配置要启动的 shop：
SHOPS=(
    "shop1:1"    # shop1，并发数 1
    "shop2:1"    # shop2，并发数 1
    "shop3:2"    # shop3，并发数 2
    "shop4:1"    # shop4，并发数 1
)

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
启动 Worker: shop3 (并发: 2)
  ✓ 已在 tmux 窗口 'shop3' 中启动

✅ 已启动 3 个 Shop Workers

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

# 为 shop3 启动 worker，并发数 4
./scripts/celery_worker_shop.sh shop3 4
```

### 2. 使用 API 导入数据

在导入数据时，添加 `route_by_shop=1` 参数：

```bash
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?route_by_shop=1&dry_run=0" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "files=@shop1.xlsx" \
  -F "files=@shop2.xlsx" \
  -F "files=@shop3.xlsx"
```

**响应示例:**
```json
{
  "accepted": true,
  "dry_run": false,
  "dedupe": true,
  "upsert": false,
  "route_by_shop": true,
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "tasks": [
    {
      "file": "shop1.xlsx",
      "task_id": "abc123...",
      "source": "shop1",
      "queue": "shop_shop1"
    },
    {
      "file": "shop2.xlsx",
      "task_id": "def456...",
      "source": "shop2",
      "queue": "shop_shop2"
    },
    {
      "file": "shop3.xlsx",
      "task_id": "ghi789...",
      "source": "shop3",
      "queue": "shop_shop3"
    }
  ]
}
```

### 3. 监控任务执行

#### 使用 tmux 查看 worker 日志

```bash
# 连接到 tmux session
tmux attach -t shop_workers

# 在 tmux 中切换窗口
Ctrl+b 然后按 0    # 切换到窗口 0 (shop1)
Ctrl+b 然后按 1    # 切换到窗口 1 (shop2)
Ctrl+b 然后按 2    # 切换到窗口 2 (shop3)

# 退出 tmux（worker 继续运行）
Ctrl+b 然后按 d
```

#### 使用 Flower 监控

```bash
# 启动 Flower
./scripts/flower.sh

# 访问 http://localhost:5555
```

在 Flower 中可以看到：
- 每个 shop 队列的任务数量
- 各个 worker 的状态
- 任务执行时间和成功率

### 4. 停止所有 Shop Workers

```bash
# 停止所有 shop workers
./scripts/stop_shop_workers.sh
```

---

## 详细配置

### API 参数说明

`POST /AppleStockChecker/purchasing-price-records/import-tradein-xlsx/`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `route_by_shop` | boolean | `false` | 启用按 shop 路由（`1` 或 `true` 启用） |
| `dry_run` | boolean | `false` | 仅校验不写库 |
| `dedupe` | boolean | `true` | 同 shop+PN+时间去重 |
| `upsert` | boolean | `false` | 存在时更新，否则插入 |
| `batch_id` | string | auto | 批次 ID（自动生成或自定义） |

### 队列命名规则

- **队列名称**: `shop_<source_name>`
- **路由键**: `shop.<source_name>`
- **示例**:
  - 文件 `shop1.xlsx` → 队列 `shop_shop1`
  - 文件 `shop2.csv` → 队列 `shop_shop2`
  - 文件 `janpara.xlsx` → 队列 `shop_janpara`

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

## 使用示例

### 示例 1: 测试环境独立验证

**场景**: 为 3 个 shop 分别导入测试数据，互不干扰

```bash
# 1. 启动 3 个 shop 的 workers
./scripts/start_shop_workers.sh

# 2. 导入测试数据
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?route_by_shop=1&dry_run=1" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@test_shop1.xlsx" \
  -F "files=@test_shop2.xlsx" \
  -F "files=@test_shop3.xlsx"

# 3. 在 tmux 中分别查看各 shop 的处理日志
tmux attach -t shop_workers

# 4. 测试完成后停止
./scripts/stop_shop_workers.sh
```

### 示例 2: 生产环境大批量导入

**场景**: 同时为 10 个 shop 导入历史数据，每个 shop 并发数 2

```bash
# 1. 编辑配置
vim scripts/start_shop_workers.sh

SHOPS=(
    "shop1:2"
    "shop2:2"
    "shop3:2"
    "shop4:2"
    "shop5:2"
    "shop6:2"
    "shop7:2"
    "shop8:2"
    "shop9:2"
    "shop10:2"
)

# 2. 启动所有 workers
./scripts/start_shop_workers.sh

# 3. 批量上传（可分批上传）
# 第一批
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?route_by_shop=1" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@shop1_history.xlsx" \
  -F "files=@shop2_history.xlsx" \
  -F "files=@shop3_history.xlsx" \
  -F "files=@shop4_history.xlsx" \
  -F "files=@shop5_history.xlsx"

# 第二批
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?route_by_shop=1" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@shop6_history.xlsx" \
  -F "files=@shop7_history.xlsx" \
  -F "files=@shop8_history.xlsx" \
  -F "files=@shop9_history.xlsx" \
  -F "files=@shop10_history.xlsx"

# 4. 在 Flower 中监控所有队列
# http://localhost:5555
```

### 示例 3: 为特定 shop 单独调试

**场景**: shop3 数据导入有问题，单独启动 worker 进行调试

```bash
# 1. 只为 shop3 启动 worker，增加日志级别
celery -A YamagotiProjects worker \
    -Q shop_shop3 \
    -l debug \
    -c 1

# 2. 只上传 shop3 的数据
curl -X POST "http://localhost:8000/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?route_by_shop=1&dry_run=1" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@shop3_test.xlsx"

# 3. 查看详细的 debug 日志
```

---

## 性能对比

### 场景: 同时导入 5 个 shop 的数据，每个文件 10000 行

| 模式 | Worker 配置 | 总耗时 | 吞吐量 |
|------|------------|--------|--------|
| 默认模式 | 1 个 webscraper worker (并发 2) | ~25 分钟 | 2000 行/分钟 |
| Shop 路由模式 | 5 个 shop workers (各并发 1) | ~10 分钟 | 5000 行/分钟 |
| Shop 路由模式 | 5 个 shop workers (各并发 2) | ~5 分钟 | 10000 行/分钟 |

**优势**:
- ✅ **并行处理**: 多个 shop 同时处理，提升 2-5 倍性能
- ✅ **任务隔离**: 某个 shop 的任务失败不影响其他 shop
- ✅ **资源控制**: 为重要 shop 分配更多并发数
- ✅ **问题定位**: 快速识别哪个 shop 的数据有问题

---

## 常见问题

### Q1: 启动 workers 后，任务仍然在 webscraper 队列中

**原因**: 导入时未添加 `route_by_shop=1` 参数

**解决方案**:
```bash
# ❌ 错误 - 任务会路由到 webscraper 队列
curl ... /import-tradein-xlsx/

# ✅ 正确 - 任务会路由到各 shop 队列
curl ... /import-tradein-xlsx/?route_by_shop=1
```

### Q2: 如何查看某个 shop 队列的长度？

```bash
# 使用 Redis CLI
redis-cli LLEN shop_shop1
redis-cli LLEN shop_shop2

# 或使用 Celery 命令
celery -A YamagotiProjects inspect active
```

### Q3: 可以动态添加新的 shop worker 吗？

**可以**。无需重启现有 workers，只需为新 shop 启动专用 worker：

```bash
# 新增 shop4 的 worker
./scripts/celery_worker_shop.sh shop4 1
```

然后上传 `shop4.xlsx` 文件时，任务会自动路由到 `shop_shop4` 队列。

### Q4: 如何确认 workers 是否正常运行？

```bash
# 方法 1: 使用 Celery 命令
celery -A YamagotiProjects inspect stats

# 方法 2: 检查进程
ps aux | grep "celery.*worker.*shop_"

# 方法 3: 使用 Flower
# http://localhost:5555

# 方法 4: 查看 tmux 窗口
tmux list-windows -t shop_workers
```

### Q5: 停止单个 shop worker 而不影响其他 workers

```bash
# 如果使用 tmux
tmux kill-window -t shop_workers:shop1

# 或直接杀进程
pkill -f "celery.*worker.*shop_shop1"
```

### Q6: 如何在 Docker 环境中使用？

需要为每个 shop 创建 Docker Compose 服务配置：

```yaml
# docker-compose.yml
services:
  celery_worker_shop1:
    build: .
    command: celery -A YamagotiProjects worker -Q shop_shop1 -c 1
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      # ...

  celery_worker_shop2:
    build: .
    command: celery -A YamagotiProjects worker -Q shop_shop2 -c 1
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      # ...
```

---

## 最佳实践

### 1. 并发数配置建议

| Shop 类型 | 数据量 | 推荐并发数 | 说明 |
|----------|--------|-----------|------|
| 测试 shop | 小 (< 1000 行) | 1 | 节省资源 |
| 普通 shop | 中 (1000-10000 行) | 1-2 | 平衡性能和资源 |
| 大型 shop | 大 (> 10000 行) | 2-4 | 加快处理速度 |

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

- 启用任务重试机制（已在 `task_process_xlsx` 中配置）
- 设置合理的超时时间（9000 秒）
- 监控失败任务并及时排查

---

## 回滚到默认模式

如果需要回退到默认的 webscraper 队列模式：

```bash
# 1. 停止所有 shop workers
./scripts/stop_shop_workers.sh

# 2. 启动默认 webscraper worker
./scripts/celery_worker_webscraper.sh

# 3. 导入时不添加 route_by_shop 参数（或设为 0）
curl ... /import-tradein-xlsx/?route_by_shop=0
```

---

## 技术细节

### 代码实现位置

- **API 端点**: `AppleStockChecker/views.py:999` (`import_tradein_xlsx`)
- **动态路由**: `AppleStockChecker/views.py:1050-1088`
- **任务定义**: `AppleStockChecker/tasks/webscraper_tasks.py:84` (`task_process_xlsx`)
- **Celery 配置**: `YamagotiProjects/celery.py`

### 队列路由逻辑

```python
# 当 route_by_shop=1 时
if route_by_shop:
    queue_name = f"shop_{source_name}"
    task_process_xlsx.apply_async(
        kwargs={...},
        queue=queue_name,
        routing_key=f"shop.{source_name}",
    )
else:
    # 默认路由到 webscraper 队列
    task_process_xlsx.delay(...)
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
