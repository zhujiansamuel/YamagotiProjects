#!/bin/bash
# 停止生产环境

set -e

echo "🛑 停止生产环境..."

# 停止所有服务
docker compose down

echo "✅ 生产环境已停止！"
