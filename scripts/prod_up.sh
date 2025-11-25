#!/bin/bash
# 启动生产环境（包含 Celery Workers）

set -e

echo "🚀 启动生产环境..."

# 检查环境变量
if [ ! -f .env ]; then
    echo "⚠️  警告：未找到 .env 文件，使用默认配置"
    echo "   建议复制 .env.example 并配置："
    echo "   cp .env.example .env"
fi

# 启动所有服务
docker compose up -d

echo "⏳ 等待服务启动..."
sleep 10

# 检查服务健康状态
echo "🔍 检查服务状态..."
docker compose ps

echo "✅ 生产环境已启动！"
echo ""
echo "📊 服务端口："
echo "  - PostgreSQL (直连): localhost:5433"
echo "  - PgBouncer (连接池): localhost:6432"
echo "  - Redis: localhost:6379"
echo "  - Flower (监控): http://localhost:5555"
echo ""
echo "🔧 Celery Workers："
echo "  - celery_worker_default: 4 并发，处理 default 队列"
echo "  - celery_worker_webscraper: 2 并发，专门处理 webscraper 队列"
echo ""
echo "💡 提示："
echo "  - 生产环境应设置 USE_PGBOUNCER=true"
echo "  - 查看日志: docker compose logs -f [service_name]"
echo "  - 停止服务: ./scripts/prod_down.sh"
