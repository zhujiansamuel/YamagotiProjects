#!/bin/bash
# 为指定 shop 启动专用 Celery Worker（数据清洗任务）
# 用法: ./scripts/celery_worker_shop.sh <shop_name> [concurrency]
#
# 架构说明:
#   新架构将任务分为两类:
#   1. 数据接收任务 (webscraper 队列) - 解析/拉取数据，存 Redis
#   2. 数据清洗任务 (shop_* 队列) - 从 Redis 读取，清洗，写库
#
#   本脚本启动的是数据清洗任务的 worker
#
# 队列分组:
#   - shop_shop5 队列处理: shop5_1, shop5_2, shop5_3, shop5_4
#   - shop_shop6 队列处理: shop6_1, shop6_2, shop6_3, shop6_4
#   - 其他队列: 独立处理对应 shop
#
# 示例:
#   ./scripts/celery_worker_shop.sh shop1          # 启动 shop1 清洗 worker，并发数 1
#   ./scripts/celery_worker_shop.sh shop2 2        # 启动 shop2 清洗 worker，并发数 2
#   ./scripts/celery_worker_shop.sh shop5 2        # 启动 shop5 清洗 worker（处理 shop5_1~4）

set -e

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 请提供 shop 名称"
    echo ""
    echo "用法: $0 <shop_name> [concurrency]"
    echo ""
    echo "示例:"
    echo "  $0 shop1          # 启动 shop1 清洗 worker，并发数 1"
    echo "  $0 shop2 2        # 启动 shop2 清洗 worker，并发数 2"
    echo "  $0 shop5 2        # 启动 shop5 清洗 worker（处理 shop5_1~4）"
    echo ""
    echo "队列分组说明:"
    echo "  - shop5 队列处理: shop5_1, shop5_2, shop5_3, shop5_4"
    echo "  - shop6 队列处理: shop6_1, shop6_2, shop6_3, shop6_4"
    exit 1
fi

SHOP_NAME=$1
CONCURRENCY=${2:-1}  # 默认并发数为 1
QUEUE_NAME="shop_${SHOP_NAME}"

echo "========================================"
echo "  启动 Shop 数据清洗 Worker"
echo "========================================"
echo ""
echo "  Shop 名称: ${SHOP_NAME}"
echo "  队列名称: ${QUEUE_NAME}"
echo "  并发数: ${CONCURRENCY}"
echo ""

# 显示队列分组信息
case "${SHOP_NAME}" in
    shop5)
        echo "  注意: 此队列处理以下店铺的数据:"
        echo "        shop5_1, shop5_2, shop5_3, shop5_4"
        echo ""
        ;;
    shop6)
        echo "  注意: 此队列处理以下店铺的数据:"
        echo "        shop6_1, shop6_2, shop6_3, shop6_4"
        echo ""
        ;;
esac

# 设置环境变量
export DJANGO_SETTINGS_MODULE=YamagotiProjects.settings

# 启动 Celery Worker - 专门处理该 shop 的数据清洗任务
celery -A YamagotiProjects worker \
    -Q "${QUEUE_NAME}" \
    -l info \
    -c "${CONCURRENCY}" \
    --max-tasks-per-child=100 \
    --hostname="worker_${QUEUE_NAME}@%h"

echo ""
echo "Celery Worker (${QUEUE_NAME}) 已停止"
