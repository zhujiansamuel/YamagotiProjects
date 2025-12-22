#!/bin/bash
# FeatureSnapshot 冷启动初始化的 Shell 包装脚本

set -e

if [ -f "/.dockerenv" ]; then
    python scripts/coldstart_feature_snapshot.py "$@"
elif docker compose ps web &>/dev/null 2>&1; then
    docker compose exec web python scripts/coldstart_feature_snapshot.py "$@"
else
    python scripts/coldstart_feature_snapshot.py "$@"
fi
