# 项目更新说明 - PgBouncer 连接池 & Celery 队列优化

## 更新时间
2025-11-25

## 概述

本次更新为项目添加了两个重要优化：

1. **PgBouncer 连接池**：解决 PostgreSQL "too many clients already" 错误
2. **Celery 专用队列**：为 WebScraper 任务配置 2 个专用 Worker

---

## 🎯 主要变更

### 1. 新增文件

#### Docker 相关
- `docker-compose.yml` - Docker Compose 完整配置
- `Dockerfile` - Python 应用容器镜像
- `.env.example` - 环境变量模板

#### 脚本 (`scripts/` 目录)
- `dev_up.sh` - 启动开发环境
- `dev_down.sh` - 停止开发环境
- `prod_up.sh` - 启动生产环境（含所有 Workers）
- `prod_down.sh` - 停止生产环境
- `celery_worker_default.sh` - 启动默认队列 Worker（本地）
- `celery_worker_webscraper.sh` - 启动 WebScraper 队列 Worker（本地）
- `celery_beat.sh` - 启动定时任务调度器
- `flower.sh` - 启动 Celery 监控工具
- `check_health.sh` - 检查所有服务健康状态
- `migrate.sh` - 执行数据库迁移
- `run_tests.sh` - 运行测试
- `lint.sh` - 代码质量检查

#### 文档 (`docs/` 目录)
- `PGBOUNCER_SETUP.md` - PgBouncer 配置详细指南
- `CELERY_QUEUES.md` - Celery 队列配置与使用
- `README_CHANGES.md` - 本文件（变更说明）

### 2. 修改文件

#### `YamagotiProjects/settings.py`
- 添加环境变量支持 (`USE_PGBOUNCER`)
- 配置 PgBouncer 数据库连接（事务池模式）
- 配置 PostgreSQL 直连（开发环境）
- 关键配置：
  - `CONN_MAX_AGE=0`（PgBouncer 模式）
  - `DISABLE_SERVER_SIDE_CURSORS=True`（必须）

#### `YamagotiProjects/celery.py`
- 添加任务队列路由配置
- 配置 `webscraper` 专用队列
- 配置任务优先级和默认队列
- 优化 Redis 连接池

---

## 🚀 快速开始

### 开发环境

```bash
# 1. 启动基础服务（PostgreSQL + Redis + PgBouncer）
./scripts/dev_up.sh

# 2. 运行 Django 开发服务器
python manage.py runserver

# 3. 启动 Celery Workers（开启 3 个终端）
./scripts/celery_worker_default.sh      # 终端 1
./scripts/celery_worker_webscraper.sh   # 终端 2
./scripts/celery_beat.sh                # 终端 3

# 4. 启动 Flower 监控（可选）
./scripts/flower.sh
```

### 生产环境

```bash
# 启动所有服务（Docker Compose）
./scripts/prod_up.sh

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f celery_worker_webscraper
```

### 停止服务

```bash
# 开发环境
./scripts/dev_down.sh

# 生产环境
./scripts/prod_down.sh
```

---

## 📊 服务架构

### 服务列表

| 服务 | 容器名称 | 端口 | 说明 |
|------|---------|------|------|
| PostgreSQL | `yapp_postgres` | 5433 | 数据库（max_connections=200） |
| PgBouncer | `yapp_pgbouncer` | 6432 | 连接池（事务池模式） |
| Redis | `yapp_redis` | 6379 | 消息代理 |
| Celery Default | `yapp_celery_default` | - | 默认队列 Worker（4并发） |
| Celery WebScraper | `yapp_celery_webscraper` | - | WebScraper 队列 Worker（2并发） |
| Celery Beat | `yapp_celery_beat` | - | 定时任务调度器 |
| Flower | `yapp_flower` | 5555 | Celery 监控工具 |

### 连接架构

```
Django/Celery
    ↓
┌─ 开发环境 ────────────────┐
│  PostgreSQL:5433          │
│  (直连, CONN_MAX_AGE=60) │
└──────────────────────────┘
    ↓
┌─ 生产环境 ────────────────┐
│  PgBouncer:6432           │
│  (事务池, 25-50连接)      │
│      ↓                    │
│  PostgreSQL:5432          │
│  (max_connections=200)    │
└──────────────────────────┘
```

### 队列架构

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

## ⚙️ 配置说明

### 环境变量

复制 `.env.example` 并根据环境修改：

```bash
cp .env.example .env
```

关键变量：

```bash
# 开发环境（直连 PostgreSQL）
USE_PGBOUNCER=false
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433

# 生产环境（使用 PgBouncer）
USE_PGBOUNCER=true
PGBOUNCER_HOST=127.0.0.1
PGBOUNCER_PORT=6432

# Celery 并发配置
CELERY_CONCURRENCY_DEFAULT=4
CELERY_CONCURRENCY_WEBSCRAPER=2
```

### PgBouncer 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `POOL_MODE` | `transaction` | 事务池模式（推荐） |
| `MAX_CLIENT_CONN` | `1000` | 最大客户端连接数 |
| `DEFAULT_POOL_SIZE` | `25` | 默认连接池大小 |
| `MAX_DB_CONNECTIONS` | `50` | 每个数据库的最大连接数 |

### Celery 队列配置

| 队列 | Worker 数 | 并发数 | 任务类型 |
|------|----------|--------|---------|
| `default` | 1 | 4 | 通用任务 |
| `webscraper` | 1 | 2 | WebScraper 任务 |

---

## 🔍 监控与调试

### 健康检查

```bash
# 检查所有服务
./scripts/check_health.sh

# 手动检查 PgBouncer
pg_isready -h 127.0.0.1 -p 6432

# 查看 PgBouncer 连接池状态
psql -h 127.0.0.1 -p 6432 -U samuelzhu -d pgbouncer -c "SHOW POOLS;"
```

### Celery 监控

```bash
# Flower Web UI
# 访问: http://localhost:5555

# 命令行查看活跃任务
celery -A YamagotiProjects inspect active

# 查看队列长度
redis-cli LLEN webscraper
redis-cli LLEN default
```

---

## 🐛 故障排查

### 问题 1: "too many clients already"

**解决方案**：
1. 确认使用 PgBouncer：`export USE_PGBOUNCER=true`
2. 检查 PgBouncer 运行：`docker compose ps pgbouncer`
3. 验证配置：`CONN_MAX_AGE=0` 且 `DISABLE_SERVER_SIDE_CURSORS=True`

### 问题 2: WebScraper 任务不执行

**解决方案**：
1. 检查 Worker 状态：`celery -A YamagotiProjects status`
2. 查看队列长度：`redis-cli LLEN webscraper`
3. 重启 Worker：`./scripts/celery_worker_webscraper.sh`

### 问题 3: 任务执行缓慢

**解决方案**：
1. 增加并发数：`-c 4`（从 2 增加到 4）
2. 检查数据库连接池：`SHOW POOLS;`
3. 查看 Flower 中的 Runtime 统计

---

## 📈 性能优化建议

### 1. 连接池大小调优

根据并发量和 CPU 核心数调整：

```yaml
# docker-compose.yml
DEFAULT_POOL_SIZE: 25    # 推荐：CPU 核心数 × 2-4
MAX_DB_CONNECTIONS: 50   # 不超过 max_connections 的 70%
```

### 2. Celery 并发数

根据任务类型调整：

| 任务类型 | 推荐并发数 |
|---------|----------|
| CPU 密集型 | CPU 核心数 |
| I/O 密集型 | CPU 核心数 × 2-4 |
| 混合型 | CPU 核心数 × 1.5 |

### 3. PostgreSQL 优化

```ini
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 5MB
```

---

## 📝 待办事项

### 已完成 ✅
- [x] PgBouncer 配置和 Docker Compose 服务
- [x] Django settings 支持环境变量和 PgBouncer
- [x] Celery 队列路由配置
- [x] WebScraper 专用队列（2 个 Worker）
- [x] 启动脚本和健康检查脚本
- [x] 详细文档（PgBouncer + Celery）

### 建议后续优化 ⏳
- [ ] 添加 Prometheus + Grafana 监控
- [ ] 配置告警规则（队列长度、失败率等）
- [ ] 添加 CI/CD Pipeline
- [ ] 配置日志聚合（ELK Stack）
- [ ] 添加自动化测试（pytest + coverage）
- [ ] 配置备份策略（pg_dump 定时任务）

---

## 🔗 参考文档

- [PgBouncer 配置指南](./PGBOUNCER_SETUP.md)
- [Celery 队列配置](./CELERY_QUEUES.md)
- [项目规范](../CLAUDE.md)

---

## 📞 联系方式

如有问题，请：
1. 查看 `docs/PGBOUNCER_SETUP.md` 和 `docs/CELERY_QUEUES.md`
2. 运行 `./scripts/check_health.sh` 检查服务状态
3. 查看日志：`docker compose logs -f [service_name]`

---

## 📜 变更历史

### v1.0 (2025-11-25)
- ✨ 添加 PgBouncer 连接池（事务池模式）
- ✨ 配置 Celery WebScraper 专用队列（2 个 Worker）
- ✨ 创建 Docker Compose 完整配置
- ✨ 添加所有启动脚本和文档
- 🔧 优化 Django 数据库配置（支持环境变量）
- 🔧 优化 Celery 配置（队列路由、连接池）

---

**重要提示**：
1. 生产环境请务必设置 `USE_PGBOUNCER=true`
2. 不要将真实密码提交到版本控制
3. 定期监控连接池和队列状态
4. 根据实际负载调整并发数和连接池大小
