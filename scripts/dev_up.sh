#!/bin/bash
# 启动开发环境服务
# 用法:
#   ./scripts/dev_up.sh               # 仅启动基础服务 + Web
#   ./scripts/dev_up.sh --celery      # 启动基础服务 + Web + Celery workers（不含 shop 专用）
#   ./scripts/dev_up.sh --shop-workers # 启动基础服务 + Web + 19个 Shop 专用 workers
#   ./scripts/dev_up.sh --all         # 启动所有服务（等同于 prod 模式）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 解析参数
WITH_CELERY=false
WITH_SHOP_WORKERS=false
ALL_SERVICES=false

for arg in "$@"; do
    case $arg in
        --celery)
            WITH_CELERY=true
            shift
            ;;
        --shop-workers)
            WITH_SHOP_WORKERS=true
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

# =============================================================================
# 检查 local-pg 容器是否运行 (仅在非 --all 模式下)
# docker-compose.yml 依赖外部的 local-pg 容器提供 PostgreSQL 服务
# =============================================================================
    echo ""
    echo "[检查] PostgreSQL 容器 (local-pg)..."

    # 检查 local-pg 容器是否存在且运行
    if docker ps --format '{{.Names}}' | grep -q '^local-pg$'; then
        echo "[OK] local-pg 容器正在运行"
    else
        # 检查容器是否存在但未运行
        if docker ps -a --format '{{.Names}}' | grep -q '^local-pg$'; then
            echo "[警告] local-pg 容器存在但未运行，正在启动..."
            docker start local-pg
            echo "[OK] local-pg 容器已启动"
        else
            echo "[错误] local-pg 容器不存在，请先手动创建："
            echo "  docker run -d --name local-pg -p 5433:5432 -e POSTGRES_PASSWORD=localpass -v pgdata:/var/lib/postgresql/data --restart unless-stopped postgres:16"
            exit 1
        fi

        # 等待 PostgreSQL 就绪
        echo "[等待] PostgreSQL 服务就绪..."
        for i in {1..30}; do
            if docker exec local-pg pg_isready -U postgres -d applestockchecker_dev > /dev/null 2>&1; then
                echo "[OK] PostgreSQL 服务已就绪"
                break
            fi
            if [ $i -eq 30 ]; then
                echo "[错误] PostgreSQL 服务启动超时"
                exit 1
            fi
            sleep 1
        done
    fi
    echo ""

# =============================================================================
# 检查 Ollama 服务 (如果启动 shop-workers)
# =============================================================================
if [ "$WITH_SHOP_WORKERS" = true ]; then
    echo "[检查] Ollama 服务..."
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        OLLAMA_VERSION=$(curl -s http://localhost:11434/api/version | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
        echo "[OK] Ollama 服务正在运行 (版本: $OLLAMA_VERSION)"
    else
        echo "[警告] Ollama 服务未运行，部分 shop workers (shop3/4/9/11/12/14/15/16) 的 LLM 功能将不可用"
    fi
    echo ""
fi

# =============================================================================
# 启动服务
# =============================================================================
if [ "$ALL_SERVICES" = true ]; then
    echo "模式: 全服务 (使用 docker-compose.yml)"
    docker compose up -d
elif [ "$WITH_SHOP_WORKERS" = true ]; then
    echo "模式: Web + 19个 Shop 专用 Workers (使用 docker-compose.yml)"
    echo ""
    echo "启动的 Shop Workers:"
    echo "  - shop1, shop2, shop3*, shop4*"
    echo "  - shop5 (并发 2, 处理 shop5_1~4)"
    echo "  - shop6 (并发 2, 处理 shop6_1~4)"
    echo "  - shop7, shop8, shop9*, shop10, shop11*, shop12*"
    echo "  - shop13, shop14*, shop15*, shop16*, shop17, shop18, shop20"
    echo "  (* 表示使用 LLM)"
    echo ""
    docker compose --profile shop-workers up -d
elif [ "$WITH_CELERY" = true ]; then
    echo "模式: Web + Celery (使用 docker-compose.yml，不含 shop 专用 workers)"
    docker compose --profile celery up -d
else
    echo "模式: 仅 Web (使用 docker-compose.yml)"
    docker compose up -d
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
    docker compose ps
fi

echo ""
echo "========================================"
echo "  开发环境已启动"
echo "========================================"
echo ""
echo "服务端口:"
echo "  - Django Web:     http://localhost:8000"
echo "  - Django Admin:   http://localhost:8000/admin/"
echo "  - PostgreSQL:     localhost:5433 (local-pg 容器)"
echo "  - Redis:          localhost:6379"
if [ "$WITH_CELERY" = true ] || [ "$WITH_SHOP_WORKERS" = true ] || [ "$ALL_SERVICES" = true ]; then
echo "  - Flower:         http://localhost:5555"
fi
if [ "$WITH_SHOP_WORKERS" = true ]; then
echo "  - Ollama:         http://localhost:11434 (宿主机服务)"
echo ""
echo "Shop Workers (共 19 个):"
echo "  - shop1~4, shop7~18, shop20: 并发数 1"
echo "  - shop5, shop6: 并发数 2 (处理多个子店铺)"
echo "  - 使用 LLM: shop3, shop4, shop9, shop11, shop12, shop14, shop15, shop16"
fi
echo ""
echo "常用命令:"
echo "  - 查看日志:        docker compose logs -f"
echo "  - 查看 Web 日志:   docker compose logs -f web"
echo "  - 查看 Shop 日志:  docker compose logs -f worker_shop1"
echo "  - 进入容器:        docker compose exec web bash"
echo "  - 运行 migrate:    docker compose exec web python manage.py migrate"
echo "  - 创建超级用户:    docker compose exec web python manage.py createsuperuser"
echo "  - 停止服务:        ./scripts/dev_down.sh"
echo ""
echo "数据库恢复命令 (首次启动后执行):"
echo "  docker exec -i local-pg pg_restore -U postgres -d applestockchecker_dev --clean --if-exists < ~/sources/local-pg-applestockchecker_dev.dump"
echo ""
