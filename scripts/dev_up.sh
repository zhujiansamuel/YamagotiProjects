#!/bin/bash
# 启动开发环境服务

set -e

echo "🚀 启动开发环境..."

# 启动 Docker Compose 服务
docker compose up -d db redis pgbouncer

echo "⏳ 等待服务启动..."
sleep 5

# 检查服务健康状态
echo "🔍 检查服务状态..."
docker compose ps

echo "✅ 开发环境已启动！"
echo ""
echo "📊 服务端口："
echo "  - PostgreSQL (直连): localhost:5433"
echo "  - PgBouncer (连接池): localhost:6432"
echo "  - Redis: localhost:6379"
echo ""
echo "💡 提示："
echo "  - 开发环境默认直连 PostgreSQL (USE_PGBOUNCER=false)"
echo "  - 如需使用 PgBouncer，请设置环境变量: export USE_PGBOUNCER=true"
echo "  - 查看日志: docker compose logs -f"
echo "  - 停止服务: ./scripts/dev_down.sh"
