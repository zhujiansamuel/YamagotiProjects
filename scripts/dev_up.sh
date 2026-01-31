#!/bin/bash
# 启动开发环境服务
# 用法:
#   ./scripts/dev_up.sh          # 仅启动基础服务 + Web
#   ./scripts/dev_up.sh --celery # 启动基础服务 + Web + Celery workers
#   ./scripts/dev_up.sh --all    # 启动所有服务（等同于 prod 模式）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 解析参数
WITH_CELERY=false
ALL_SERVICES=false

for arg in "$@"; do
    case $arg in
        --celery)
            WITH_CELERY=true
            shift
            ;;
        --all)
            ALL_SERVICES=true
            shift
            ;;
    esac
done

echo "========================================"
echo "  YamagotiProjects 开发环境启动"
echo "========================================"

if [ "$ALL_SERVICES" = true ]; then
    echo "模式: 全服务 (使用 docker-compose.yml)"
    docker compose up -d
elif [ "$WITH_CELERY" = true ]; then
    echo "模式: Web + Celery (使用 docker-compose.dev.yml)"
    docker compose -f docker-compose.dev.yml --profile celery up -d
else
    echo "模式: 仅 Web (使用 docker-compose.dev.yml)"
    docker compose -f docker-compose.dev.yml up -d
fi

echo ""
echo "等待服务启动..."
sleep 5

# 检查服务健康状态
echo ""
echo "服务状态:"
if [ "$ALL_SERVICES" = true ]; then
    docker compose ps
else
    docker compose -f docker-compose.dev.yml ps
fi

echo ""
echo "========================================"
echo "  开发环境已启动"
echo "========================================"
echo ""
echo "服务端口:"
echo "  - Django Web:     http://localhost:8000"
echo "  - Django Admin:   http://localhost:8000/admin/"
echo "  - PostgreSQL:     localhost:5433"
echo "  - PgBouncer:      localhost:6432"
echo "  - Redis:          localhost:6379"
if [ "$WITH_CELERY" = true ] || [ "$ALL_SERVICES" = true ]; then
echo "  - Flower:         http://localhost:5555"
fi
echo ""
echo "常用命令:"
echo "  - 查看日志:        docker compose -f docker-compose.dev.yml logs -f"
echo "  - 查看 Web 日志:   docker compose -f docker-compose.dev.yml logs -f web"
echo "  - 进入容器:        docker compose -f docker-compose.dev.yml exec web bash"
echo "  - 运行 migrate:    docker compose -f docker-compose.dev.yml exec web python manage.py migrate"
echo "  - 创建超级用户:    docker compose -f docker-compose.dev.yml exec web python manage.py createsuperuser"
echo "  - 停止服务:        ./scripts/dev_down.sh"
echo ""
