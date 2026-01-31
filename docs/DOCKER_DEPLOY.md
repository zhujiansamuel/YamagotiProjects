# Docker Compose 部署指南

本文档介绍如何使用 Docker Compose 在本地或服务器上运行 YamagotiProjects。

## 目录

- [快速开始](#快速开始)
- [服务架构](#服务架构)
- [开发环境](#开发环境)
- [生产环境](#生产环境)
- [常用命令](#常用命令)
- [故障排除](#故障排除)
- [迁移到服务器](#迁移到服务器)

## 快速开始

### 前提条件

- Docker 20.10+
- Docker Compose V2 (`docker compose` 命令)

### 首次启动

```bash
# 1. 克隆项目
git clone <repository-url>
cd YamagotiProjects

# 2. 复制环境变量文件
cp .env.example .env

# 3. 启动开发环境
./scripts/dev_up.sh

# 4. 创建超级用户
docker compose -f docker-compose.dev.yml exec web python manage.py createsuperuser

# 5. 访问应用
# Web: http://localhost:8000
# Admin: http://localhost:8000/admin/
```

## 服务架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose 网络                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌───────────┐    ┌─────────────────────┐   │
│  │  Web    │───▶│ PgBouncer │───▶│    PostgreSQL       │   │
│  │ :8000   │    │   :6432   │    │       :5433         │   │
│  └─────────┘    └───────────┘    └─────────────────────┘   │
│       │                                                     │
│       │         ┌─────────────────────────────────────┐    │
│       └────────▶│             Redis                   │    │
│                 │             :6379                   │    │
│                 └─────────────────────────────────────┘    │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ Celery   │    │ Celery   │    │ Celery   │            │
│   │ Default  │    │WebScraper│    │  Beat    │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│                                                             │
│   ┌──────────┐                                             │
│   │  Flower  │  (监控面板 :5555)                           │
│   └──────────┘                                             │
└─────────────────────────────────────────────────────────────┘
```

### 服务说明

| 服务 | 端口 | 说明 |
|------|------|------|
| web | 8000 | Django Web 应用 (ASGI/Daphne) |
| db | 5433 | PostgreSQL 16 数据库 |
| pgbouncer | 6432 | 连接池，减少数据库连接开销 |
| redis | 6379 | Celery Broker + 缓存 + Channels |
| celery_worker_default | - | 默认任务队列 |
| celery_worker_webscraper | - | 网页抓取专用队列 |
| celery_beat | - | 定时任务调度器 |
| flower | 5555 | Celery 任务监控 |

## 开发环境

### 配置文件

- `docker-compose.dev.yml` - 轻量开发配置
- `docker-compose.yml` - 完整生产配置

### 启动模式

```bash
# 仅 Web + 基础服务（推荐日常开发）
./scripts/dev_up.sh

# Web + Celery Workers
./scripts/dev_up.sh --celery

# 全部服务（模拟生产环境）
./scripts/dev_up.sh --all
```

### 开发特性

- 代码热重载 (使用 runserver)
- 挂载本地目录，修改代码立即生效
- Celery workers 可选启动

### 停止服务

```bash
# 停止服务（保留数据）
./scripts/dev_down.sh

# 停止并清理数据卷
./scripts/dev_down.sh --clean
```

## 生产环境

### 启动

```bash
# 使用完整配置启动所有服务
./scripts/prod_up.sh

# 或直接使用 docker compose
docker compose up -d
```

### 生产配置要点

1. **环境变量**：修改 `.env` 中的敏感信息
2. **DJANGO_DEBUG**：设为 `False`
3. **SECRET_KEY**：使用强随机密钥
4. **ALLOWED_HOSTS**：添加域名

### 停止

```bash
./scripts/prod_down.sh
```

## 常用命令

### 日志

```bash
# 查看所有日志
docker compose -f docker-compose.dev.yml logs -f

# 查看特定服务日志
docker compose -f docker-compose.dev.yml logs -f web
docker compose -f docker-compose.dev.yml logs -f celery_worker
```

### Django 管理

```bash
# 进入 Web 容器
docker compose -f docker-compose.dev.yml exec web bash

# 数据库迁移
docker compose -f docker-compose.dev.yml exec web python manage.py migrate

# 创建超级用户
docker compose -f docker-compose.dev.yml exec web python manage.py createsuperuser

# 收集静态文件
docker compose -f docker-compose.dev.yml exec web python manage.py collectstatic

# Django Shell
docker compose -f docker-compose.dev.yml exec web python manage.py shell
```

### 数据库

```bash
# 连接 PostgreSQL
docker compose -f docker-compose.dev.yml exec db psql -U samuelzhu -d applestockchecker_dev

# 备份数据库
docker compose -f docker-compose.dev.yml exec db pg_dump -U samuelzhu applestockchecker_dev > backup.sql

# 恢复数据库
cat backup.sql | docker compose -f docker-compose.dev.yml exec -T db psql -U samuelzhu -d applestockchecker_dev
```

### 构建

```bash
# 重新构建镜像
docker compose -f docker-compose.dev.yml build

# 强制重新构建（无缓存）
docker compose -f docker-compose.dev.yml build --no-cache
```

## 故障排除

### Web 容器启动失败

```bash
# 查看详细日志
docker compose -f docker-compose.dev.yml logs web

# 常见原因：
# 1. 数据库未就绪 - 等待几秒后重试
# 2. 迁移失败 - 进入容器手动检查
docker compose -f docker-compose.dev.yml exec web python manage.py migrate --check
```

### 数据库连接失败

```bash
# 检查 PostgreSQL 状态
docker compose -f docker-compose.dev.yml exec db pg_isready

# 检查 PgBouncer 状态
docker compose -f docker-compose.dev.yml exec pgbouncer pg_isready -h localhost
```

### Redis 连接失败

```bash
# 检查 Redis
docker compose -f docker-compose.dev.yml exec redis redis-cli ping
```

### 清理重建

```bash
# 完全清理并重建
./scripts/dev_down.sh --clean
docker compose -f docker-compose.dev.yml build --no-cache
./scripts/dev_up.sh
```

## 迁移到服务器

### 1. 准备服务器

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 安装 Docker Compose V2 (通常已包含)
docker compose version
```

### 2. 传输项目

```bash
# 方法 A: Git
git clone <repository-url>

# 方法 B: rsync
rsync -avz --exclude '.git' --exclude '__pycache__' \
  ./YamagotiProjects/ user@server:/opt/YamagotiProjects/
```

### 3. 配置环境

```bash
# 复制并编辑环境变量
cp .env.example .env
vim .env

# 重要修改：
# - DJANGO_DEBUG=False
# - DJANGO_SECRET_KEY=<strong-random-key>
# - DJANGO_ALLOWED_HOSTS=your-domain.com
# - 数据库密码
```

### 4. 启动服务

```bash
# 构建并启动
docker compose up -d --build

# 初始化
docker compose exec web python manage.py migrate
docker compose exec web python manage.py collectstatic --noinput
docker compose exec web python manage.py createsuperuser
```

### 5. 配置反向代理 (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /opt/YamagotiProjects/staticfiles/;
    }

    location /media/ {
        alias /opt/YamagotiProjects/media/;
    }
}
```

### 6. 持久化配置

确保数据卷正确挂载：
- `postgres_data` - 数据库持久化
- `redis_data` - Redis 持久化
- `static_data` - 静态文件
- `media_data` - 上传文件

## 附录

### 端口映射

| 服务 | 容器端口 | 主机端口 |
|------|----------|----------|
| Web | 8000 | 8000 |
| PostgreSQL | 5432 | 5433 |
| PgBouncer | 5432 | 6432 |
| Redis | 6379 | 6379 |
| Flower | 5555 | 5555 |

### 环境变量参考

详见 `.env.example` 文件。
