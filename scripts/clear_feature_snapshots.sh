#!/bin/bash
# Shell wrapper for clear_feature_snapshots.py
# 便于直接运行（自动处理 Docker 容器）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查是否在容器内
if [ -f "/.dockerenv" ]; then
    # 在容器内直接运行
    python scripts/clear_feature_snapshots.py "$@"
else
    # 在容器外通过 docker compose 运行
    docker compose exec web python scripts/clear_feature_snapshots.py "$@"
fi
