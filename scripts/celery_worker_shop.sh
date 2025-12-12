#!/bin/bash
# 为指定 shop 启动专用 Celery Worker（测试/开发环境）
# 用法: ./scripts/celery_worker_shop.sh <shop_name> [concurrency]
#
# 示例:
#   ./scripts/celery_worker_shop.sh shop1          # 启动 shop1 的 worker，并发数 1
#   ./scripts/celery_worker_shop.sh shop2 2        # 启动 shop2 的 worker，并发数 2
#   ./scripts/celery_worker_shop.sh shop3 4        # 启动 shop3 的 worker，并发数 4

set -e

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供 shop 名称"
    echo ""
    echo "用法: $0 <shop_name> [concurrency]"
    echo ""
    echo "示例:"
    echo "  $0 shop1          # 启动 shop1 的 worker，并发数 1"
    echo "  $0 shop2 2        # 启动 shop2 的 worker，并发数 2"
    echo "  $0 shop3 4        # 启动 shop3 的 worker，并发数 4"
    exit 1
fi

SHOP_NAME=$1
CONCURRENCY=${2:-1}  # 默认并发数为 1
QUEUE_NAME="shop_${SHOP_NAME}"

echo "🚀 启动 Celery Worker for ${SHOP_NAME}..."
echo "   队列名称: ${QUEUE_NAME}"
echo "   并发数: ${CONCURRENCY}"
echo ""

# 设置环境变量
export DJANGO_SETTINGS_MODULE=YamagotiProjects.settings

# 启动 Celery Worker - 专门处理该 shop 的队列
celery -A YamagotiProjects worker \
    -Q "${QUEUE_NAME}" \
    -l info \
    -c "${CONCURRENCY}" \
    --max-tasks-per-child=100 \
    --hostname="worker_${QUEUE_NAME}@%h"

echo "✅ Celery Worker (${QUEUE_NAME}) 已停止"
