#!/bin/bash
# 批量启动多个 shop 的专用 Celery Workers（数据清洗任务）
# 用法: ./scripts/start_shop_workers.sh
#
# 架构说明:
# - webscraper 队列: 数据接收任务（需要单独启动 webscraper worker）
# - shop_* 队列: 数据清洗任务（本脚本启动的 workers）
#
# 队列分组:
# - shop_shop5: 处理 shop5_1, shop5_2, shop5_3, shop5_4
# - shop_shop6: 处理 shop6_1, shop6_2, shop6_3, shop6_4
# - 其他: 独立队列

set -e

# 配置要启动的 shop 列表（根据需要修改）
# 格式: "queue_suffix:concurrency"
# 注意: 队列名称会自动加上 "shop_" 前缀
SHOPS=(
    "shop1:1"
    "shop2:1"
    "shop3:1"
    "shop4:1"
    "shop5:2"    # 处理 shop5_1~4，并发数 2
    "shop6:2"    # 处理 shop6_1~4，并发数 2
    "shop7:1"
    "shop8:1"
    "shop9:1"
    "shop10:1"
    "shop11:1"
    "shop12:1"
    "shop13:1"
    "shop14:1"
    "shop15:1"
    "shop16:1"
    "shop17:1"
    "shop18:1"
    "shop20:1"
)

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Shop 数据清洗 Workers 批量启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}架构说明:${NC}"
echo "  - webscraper 队列: 数据接收任务（需单独启动）"
echo "  - shop_* 队列: 数据清洗任务（本脚本启动）"
echo ""

# 检查 tmux 是否安装
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}警告: 未安装 tmux，将在后台启动 workers${NC}"
    echo "   建议安装 tmux 以便更好地管理多个 workers: sudo apt install tmux"
    echo ""
    USE_TMUX=false
else
    USE_TMUX=true
    SESSION_NAME="shop_workers"

    # 检查是否已有同名 session
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}tmux session '$SESSION_NAME' 已存在${NC}"
        echo "   选项:"
        echo "   1) 连接到现有 session: tmux attach -t $SESSION_NAME"
        echo "   2) 删除现有 session 并重新创建: tmux kill-session -t $SESSION_NAME && $0"
        exit 1
    fi

    echo -e "${GREEN}将在 tmux session '$SESSION_NAME' 中启动 workers${NC}"
    echo "  查看 workers: tmux attach -t $SESSION_NAME"
    echo "  切换窗口: Ctrl+b 然后按 0-9 或 n/p"
    echo "  退出但保持运行: Ctrl+b 然后按 d"
    echo ""
fi

# 创建日志目录
mkdir -p logs

# 启动计数器
STARTED=0

echo -e "${BLUE}启动 Shop Workers...${NC}"
echo ""

for shop_config in "${SHOPS[@]}"; do
    # 解析 shop_name 和 concurrency
    IFS=':' read -r shop_name concurrency <<< "$shop_config"
    concurrency=${concurrency:-1}  # 默认并发数为 1

    queue_name="shop_${shop_name}"

    echo -e "  ${BLUE}[$((STARTED + 1))/${#SHOPS[@]}]${NC} 启动 Worker: ${GREEN}${shop_name}${NC} (队列: ${queue_name}, 并发: ${concurrency})"

    if [ "$USE_TMUX" = true ]; then
        # 使用 tmux
        if [ $STARTED -eq 0 ]; then
            # 创建新 session 和第一个窗口
            tmux new-session -d -s "$SESSION_NAME" -n "$shop_name" \
                "./scripts/celery_worker_shop.sh $shop_name $concurrency"
        else
            # 创建新窗口
            tmux new-window -t "$SESSION_NAME" -n "$shop_name" \
                "./scripts/celery_worker_shop.sh $shop_name $concurrency"
        fi
        echo -e "       ${GREEN}已在 tmux 窗口 '$shop_name' 中启动${NC}"
    else
        # 不使用 tmux，直接后台启动
        nohup ./scripts/celery_worker_shop.sh "$shop_name" "$concurrency" \
            > "logs/celery_${queue_name}.log" 2>&1 &
        PID=$!
        echo -e "       ${GREEN}已在后台启动 (PID: $PID)${NC}"
        echo -e "       日志: logs/celery_${queue_name}.log"
    fi

    STARTED=$((STARTED + 1))
    sleep 0.5  # 避免同时启动太多进程
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  已启动 ${STARTED} 个 Shop Workers${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ "$USE_TMUX" = true ]; then
    echo -e "${BLUE}管理 Workers:${NC}"
    echo "  查看所有 workers: tmux attach -t $SESSION_NAME"
    echo "  列出所有窗口: tmux list-windows -t $SESSION_NAME"
    echo "  关闭所有 workers: tmux kill-session -t $SESSION_NAME"
    echo ""

    # 显示窗口列表
    echo -e "${BLUE}当前窗口列表:${NC}"
    sleep 2  # 等待 tmux 窗口完全创建
    tmux list-windows -t "$SESSION_NAME" | head -20
    if [ ${#SHOPS[@]} -gt 20 ]; then
        echo "  ... (共 ${#SHOPS[@]} 个窗口)"
    fi
else
    echo -e "${BLUE}管理 Workers:${NC}"
    echo "  查看日志: tail -f logs/celery_shop_*.log"
    echo "  查看所有进程: ps aux | grep 'celery.*worker.*shop_'"
    echo "  停止所有 workers: ./scripts/stop_shop_workers.sh"
fi

echo ""
echo -e "${YELLOW}重要提醒:${NC}"
echo "  1. 确保 webscraper worker 也已启动（处理数据接收）:"
echo "     ./scripts/celery_worker_webscraper.sh"
echo ""
echo "  2. 新架构下，清洗任务会自动路由到对应队列，无需手动指定 route_by_shop"
echo ""
echo "  3. 队列分组说明:"
echo "     - shop_shop5 队列处理: shop5_1, shop5_2, shop5_3, shop5_4"
echo "     - shop_shop6 队列处理: shop6_1, shop6_2, shop6_3, shop6_4"
echo ""
