#!/bin/bash
# 批量启动多个 shop 的专用 Celery Workers（测试/开发环境）
# 用法: ./scripts/start_shop_workers.sh
#
# 默认配置: 为 shop1, shop2, shop3 各启动一个 worker
# 可以通过编辑此脚本中的 SHOPS 数组来自定义 shop 列表

set -e

# 配置要启动的 shop 列表（根据需要修改）
# 格式: "shop_name:concurrency"
SHOPS=(
    "shop1:1"
    "shop2:1"
    "shop3:1"
    # 添加更多 shop...
    # "shop4:2"
    # "shop5:1"
)

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 批量启动 Shop Workers...${NC}"
echo ""

# 检查 tmux 是否安装
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}⚠️  警告: 未安装 tmux，将在后台启动 workers${NC}"
    echo "   建议安装 tmux 以便更好地管理多个 workers: sudo apt install tmux"
    echo ""
    USE_TMUX=false
else
    USE_TMUX=true
    SESSION_NAME="shop_workers"

    # 检查是否已有同名 session
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  tmux session '$SESSION_NAME' 已存在${NC}"
        echo "   选项:"
        echo "   1) 连接到现有 session: tmux attach -t $SESSION_NAME"
        echo "   2) 删除现有 session 并重新创建: tmux kill-session -t $SESSION_NAME && $0"
        exit 1
    fi

    echo -e "${GREEN}✓ 将在 tmux session '$SESSION_NAME' 中启动 workers${NC}"
    echo "  查看 workers: tmux attach -t $SESSION_NAME"
    echo "  切换窗口: Ctrl+b 然后按 0-9"
    echo "  退出但保持运行: Ctrl+b 然后按 d"
    echo ""
fi

# 启动计数器
STARTED=0

for shop_config in "${SHOPS[@]}"; do
    # 解析 shop_name 和 concurrency
    IFS=':' read -r shop_name concurrency <<< "$shop_config"
    concurrency=${concurrency:-1}  # 默认并发数为 1

    queue_name="shop_${shop_name}"

    echo -e "${BLUE}启动 Worker: ${shop_name} (并发: ${concurrency})${NC}"

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
        echo -e "${GREEN}  ✓ 已在 tmux 窗口 '$shop_name' 中启动${NC}"
    else
        # 不使用 tmux，直接后台启动
        nohup ./scripts/celery_worker_shop.sh "$shop_name" "$concurrency" \
            > "logs/celery_${queue_name}.log" 2>&1 &
        PID=$!
        echo -e "${GREEN}  ✓ 已在后台启动 (PID: $PID)${NC}"
        echo "     日志: logs/celery_${queue_name}.log"
    fi

    STARTED=$((STARTED + 1))
    sleep 1  # 避免同时启动太多进程
done

echo ""
echo -e "${GREEN}✅ 已启动 ${STARTED} 个 Shop Workers${NC}"
echo ""

if [ "$USE_TMUX" = true ]; then
    echo -e "${BLUE}📊 管理 Workers:${NC}"
    echo "  查看所有 workers: tmux attach -t $SESSION_NAME"
    echo "  列出所有窗口: tmux list-windows -t $SESSION_NAME"
    echo "  关闭所有 workers: tmux kill-session -t $SESSION_NAME"
    echo ""

    # 显示窗口列表
    echo -e "${BLUE}📋 当前窗口列表:${NC}"
    sleep 2  # 等待 tmux 窗口完全创建
    tmux list-windows -t "$SESSION_NAME"
else
    echo -e "${BLUE}📊 管理 Workers:${NC}"
    echo "  查看日志: tail -f logs/celery_shop_*.log"
    echo "  查看所有进程: ps aux | grep 'celery.*worker.*shop_'"
    echo "  停止所有 workers: pkill -f 'celery.*worker.*shop_'"
fi

echo ""
echo -e "${YELLOW}💡 提示:${NC}"
echo "  测试时记得添加 route_by_shop=1 参数:"
echo "  POST /AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?route_by_shop=1"
