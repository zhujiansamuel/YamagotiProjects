#!/bin/bash
# 初始化 FeatureSnapshot 数据的 Shell 包装脚本
# 自动检测环境并在 Docker 容器或本地执行

set -e

# 检测是否在 Docker 容器内
if [ -f "/.dockerenv" ]; then
    # 在容器内直接运行
    python scripts/initialize_feature_snapshot.py "$@"
elif docker compose ps web &>/dev/null 2>&1; then
    # 在容器外，通过 docker compose exec 运行
    docker compose exec web python scripts/initialize_feature_snapshot.py "$@"
else
    # 本地环境直接运行
    python scripts/initialize_feature_snapshot.py "$@"
fi
