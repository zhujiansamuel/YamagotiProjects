#!/bin/bash
# Shell wrapper for clear_feature_snapshots.py
# 自动检测环境并运行（支持本地/Docker）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查是否在容器内
if [ -f "/.dockerenv" ]; then
    # 在容器内直接运行
    python scripts/clear_feature_snapshots.py "$@"
# 检查是否有 Docker Compose 环境
elif docker compose ps web &>/dev/null 2>&1; then
    # 通过 docker compose 运行
    docker compose exec web python scripts/clear_feature_snapshots.py "$@"
else
    # 本地环境直接运行
    python scripts/clear_feature_snapshots.py "$@"
fi
