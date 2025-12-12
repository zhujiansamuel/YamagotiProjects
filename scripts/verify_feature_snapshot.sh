#!/bin/bash
# 验证 FeatureSnapshot 数据生成的包装脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 自动检测环境
if [ -f "/.dockerenv" ]; then
    # 在 Docker 容器内
    python "$SCRIPT_DIR/verify_feature_snapshot.py" "$@"
elif docker compose ps web &>/dev/null 2>&1; then
    # Docker Compose 环境存在
    docker compose exec web python scripts/verify_feature_snapshot.py "$@"
else
    # 本地环境
    cd "$PROJECT_DIR"
    python "$SCRIPT_DIR/verify_feature_snapshot.py" "$@"
fi
