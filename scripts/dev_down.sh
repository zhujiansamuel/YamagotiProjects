#!/bin/bash
# 停止开发环境服务

set -e

echo "🛑 停止开发环境..."

# 停止所有 Docker Compose 服务
docker compose down

echo "✅ 开发环境已停止！"
