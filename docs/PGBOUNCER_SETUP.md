# PgBouncer 连接池配置指南

## 概述

本项目已配置 **PgBouncer** 作为 PostgreSQL 连接池，用于解决数据库连接数过多的问题（`FATAL: sorry, too many clients already`）。

### 主要特性

- **事务池模式** (Transaction Pooling)：每个事务结束后立即释放连接
- **连接复用**：多个客户端共享少量数据库连接
- **高并发支持**：支持 1000+ 客户端连接，后端仅使用 25-50 个数据库连接
- **透明代理**：应用无需修改代码，只需修改连接配置

---

## 架构

```
Django/Celery (1000+ connections)
         ↓
    PgBouncer (端口 6432)
    连接池: 25-50 connections
         ↓
   PostgreSQL (端口 5432)
   最大连接数: 200
```

---

## 配置说明

### 1. Docker Compose 服务

项目包含以下服务（`docker-compose.yml`）：

- **db**: PostgreSQL 16 数据库（端口 5433）
- **pgbouncer**: PgBouncer 连接池（端口 6432）
- **redis**: Redis 消息代理（端口 6379）
- **celery_worker_default**: 默认队列 Worker（4并发）
- **celery_worker_webscraper**: WebScraper 专用 Worker（2并发）
- **celery_beat**: 定时任务调度器
- **flower**: Celery 监控工具（端口 5555）

### 2. PgBouncer 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `POOL_MODE` | `transaction` | 事务池模式（推荐） |
| `MAX_CLIENT_CONN` | `1000` | 最大客户端连接数 |
| `DEFAULT_POOL_SIZE` | `25` | 默认连接池大小 |
| `MIN_POOL_SIZE` | `5` | 最小连接池 |
| `RESERVE_POOL_SIZE` | `5` | 保留连接池 |
| `MAX_DB_CONNECTIONS` | `50` | 每个数据库的最大连接数 |
| `SERVER_IDLE_TIMEOUT` | `600` | 服务器空闲超时（10分钟） |
| `SERVER_LIFETIME` | `3600` | 连接生命周期（1小时） |

### 3. Django 数据库配置

项目根据环境变量 `USE_PGBOUNCER` 自动选择连接方式：

#### 开发环境（直连 PostgreSQL）
```bash
export USE_PGBOUNCER=false
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5433
```

#### 生产环境（通过 PgBouncer）
```bash
export USE_PGBOUNCER=true
export PGBOUNCER_HOST=127.0.0.1
export PGBOUNCER_PORT=6432
```

**关键配置（自动处理）**：
- `CONN_MAX_AGE=0`：事务池模式必须禁用连接复用
- `DISABLE_SERVER_SIDE_CURSORS=True`：禁用服务器端游标

---

## 使用方法

### 开发环境启动

```bash
# 1. 启动基础服务（PostgreSQL + Redis + PgBouncer）
./scripts/dev_up.sh

# 2. 运行 Django 开发服务器（本地）
python manage.py runserver

# 3. 启动 Celery Workers（本地）
# 终端 1：默认队列 Worker
./scripts/celery_worker_default.sh

# 终端 2：WebScraper 专用 Worker
./scripts/celery_worker_webscraper.sh

# 终端 3：定时任务调度器
./scripts/celery_beat.sh

# 4. 启动 Flower 监控（可选）
./scripts/flower.sh
```

### 生产环境启动

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

## 健康检查

```bash
# 检查所有服务状态
./scripts/check_health.sh

# 手动检查 PgBouncer
pg_isready -h 127.0.0.1 -p 6432

# 查看 PgBouncer 连接池状态
psql -h 127.0.0.1 -p 6432 -U samuelzhu -d pgbouncer -c "SHOW POOLS;"

# 查看客户端连接
psql -h 127.0.0.1 -p 6432 -U samuelzhu -d pgbouncer -c "SHOW CLIENTS;"

# 查看服务器连接
psql -h 127.0.0.1 -p 6432 -U samuelzhu -d pgbouncer -c "SHOW SERVERS;"

# 查看统计信息
psql -h 127.0.0.1 -p 6432 -U samuelzhu -d pgbouncer -c "SHOW STATS;"
```

---

## 监控指标

### Celery Flower

访问 `http://localhost:5555` 查看：
- 任务执行情况
- Worker 状态
- 队列长度
- 任务成功/失败统计

### PgBouncer 统计

```sql
-- 连接池使用率
SHOW POOLS;

-- 客户端连接数
SELECT COUNT(*) FROM SHOW CLIENTS;

-- 等待队列
SELECT COUNT(*) FROM SHOW CLIENTS WHERE state = 'waiting';
```

---

## 故障排查

### 问题 1: "too many clients already"

**原因**：数据库连接数超过 PostgreSQL `max_connections` 限制

**解决方案**：
1. 确认使用 PgBouncer（`USE_PGBOUNCER=true`）
2. 检查 PgBouncer 运行状态：`docker compose ps pgbouncer`
3. 验证 Django 配置了 `CONN_MAX_AGE=0`

### 问题 2: Django 连接超时

**原因**：PgBouncer 连接池耗尽

**解决方案**：
1. 增加 `DEFAULT_POOL_SIZE` 或 `MAX_DB_CONNECTIONS`
2. 检查是否有长时间运行的查询占用连接
3. 确认没有使用服务器端游标（Django QuerySet `.iterator()` 需禁用）

### 问题 3: Celery 任务失败

**原因**：数据库连接问题

**解决方案**：
1. 检查 Celery Worker 是否使用 PgBouncer
2. 确认环境变量正确设置
3. 查看 Worker 日志：`docker compose logs -f celery_worker_webscraper`

---

## 性能优化建议

### 1. 连接池大小调优

```yaml
# docker-compose.yml
DEFAULT_POOL_SIZE: 25    # 根据并发量调整（推荐：CPU核心数 * 2-4）
MAX_DB_CONNECTIONS: 50   # 不超过 PostgreSQL max_connections 的 70%
```

### 2. PostgreSQL 配置优化

```ini
# PostgreSQL
max_connections = 200           # 足够的后端连接
shared_buffers = 256MB          # 缓存大小
effective_cache_size = 1GB      # 系统缓存估计
work_mem = 5MB                  # 每个查询的工作内存
```

### 3. Celery 优化

```python
# settings.py
CELERY_TASK_ACKS_LATE = True               # 任务失败时重试
CELERY_TASK_REJECT_ON_WORKER_LOST = True   # Worker 崩溃时重新入队
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000   # 防止内存泄漏
```

---

## 环境变量参考

复制 `.env.example` 并根据环境修改：

```bash
cp .env.example .env
```

关键变量：

```bash
# PgBouncer 配置
USE_PGBOUNCER=true
PGBOUNCER_HOST=127.0.0.1
PGBOUNCER_PORT=6432
PGBOUNCER_DATABASE=applestockchecker_dev
PGBOUNCER_USER=samuelzhu
PGBOUNCER_PASSWORD=YOUR_PASSWORD

# Celery 并发配置
CELERY_CONCURRENCY_DEFAULT=4
CELERY_CONCURRENCY_WEBSCRAPER=2
```

---

## 安全注意事项

1. **密码管理**：
   - 生产环境使用强密码
   - 不要将密码提交到版本控制
   - 使用 `.env` 文件或 Secret Manager

2. **网络隔离**：
   - PgBouncer 和 PostgreSQL 应在内部网络
   - 限制外部访问端口

3. **连接限制**：
   - 设置合理的 `MAX_CLIENT_CONN` 防止资源耗尽
   - 监控连接数和队列长度

---

## 参考资料

- [PgBouncer 官方文档](https://www.pgbouncer.org/)
- [Django 数据库连接池](https://docs.djangoproject.com/en/stable/ref/databases/#connection-management)
- [Celery 最佳实践](https://docs.celeryq.dev/en/stable/userguide/optimizing.html)

---

## 更新历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2025-11-25 | v1.0 | 初始版本 - PgBouncer 配置与 Celery 队列优化 |
