#!/bin/bash
# 停止所有 shop 专用的 Celery Workers
# 用法: ./scripts/stop_shop_workers.sh

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 停止 Shop Workers...${NC}"
echo ""

SESSION_NAME="shop_workers"

# 检查 tmux session 是否存在
if command -v tmux &> /dev/null && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}发现 tmux session: $SESSION_NAME${NC}"
    echo "正在停止所有窗口中的 workers..."

    # 杀死 tmux session
    tmux kill-session -t "$SESSION_NAME"
    echo -e "${GREEN}✅ tmux session '$SESSION_NAME' 已停止${NC}"
fi

# 查找并停止所有 shop worker 进程
echo ""
echo -e "${YELLOW}查找后台运行的 shop worker 进程...${NC}"

# 查找进程
PIDS=$(pgrep -f "celery.*worker.*shop_" || true)

if [ -z "$PIDS" ]; then
    echo -e "${GREEN}✓ 没有发现运行中的 shop worker 进程${NC}"
else
    echo -e "${YELLOW}发现以下 shop worker 进程:${NC}"
    ps -fp $PIDS || true
    echo ""

    echo -e "${RED}正在停止这些进程...${NC}"
    pkill -f "celery.*worker.*shop_" || true

    # 等待一下，如果还有进程就强制杀死
    sleep 2
    REMAINING=$(pgrep -f "celery.*worker.*shop_" || true)
    if [ ! -z "$REMAINING" ]; then
        echo -e "${RED}强制停止剩余进程...${NC}"
        pkill -9 -f "celery.*worker.*shop_" || true
    fi

    echo -e "${GREEN}✅ 所有 shop worker 进程已停止${NC}"
fi

echo ""
echo -e "${BLUE}📊 验证:${NC}"
REMAINING=$(pgrep -f "celery.*worker.*shop_" || true)
if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}✓ 确认：没有运行中的 shop worker 进程${NC}"
else
    echo -e "${RED}⚠️  警告：仍有进程在运行:${NC}"
    ps -fp $REMAINING || true
fi

echo ""
echo -e "${GREEN}完成！${NC}"
