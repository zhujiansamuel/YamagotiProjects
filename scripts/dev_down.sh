#!/bin/bash
# 停止开发环境服务
# 用法:
#   ./scripts/dev_down.sh         # 停止服务但保留数据
#   ./scripts/dev_down.sh --clean # 停止服务并删除卷（清理数据）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

CLEAN_VOLUMES=false

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_VOLUMES=true
            shift
            ;;
    esac
done

echo "========================================"
echo "  停止开发环境服务"
echo "========================================"

# 停止 dev 配置的服务
docker compose -f docker-compose.dev.yml --profile celery down 2>/dev/null || true

# 同时也停止主配置的服务（如果有）
docker compose down 2>/dev/null || true

if [ "$CLEAN_VOLUMES" = true ]; then
    echo ""
    echo "清理数据卷..."
    docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
    docker compose down -v 2>/dev/null || true
    echo "数据卷已清理"
fi

echo ""
echo "开发环境已停止"
