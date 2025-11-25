#!/bin/bash
# 检查所有服务的健康状态

set -e

echo "🔍 检查服务健康状态..."
echo ""

# 检查 PostgreSQL
echo "📊 PostgreSQL (直连 5433):"
if pg_isready -h 127.0.0.1 -p 5433 -U samuelzhu > /dev/null 2>&1; then
    echo "  ✅ 运行正常"
else
    echo "  ❌ 未响应"
fi

# 检查 PgBouncer
echo ""
echo "📊 PgBouncer (连接池 6432):"
if pg_isready -h 127.0.0.1 -p 6432 > /dev/null 2>&1; then
    echo "  ✅ 运行正常"
else
    echo "  ❌ 未响应"
fi

# 检查 Redis
echo ""
echo "📊 Redis (6379):"
if redis-cli -h 127.0.0.1 -p 6379 ping > /dev/null 2>&1; then
    echo "  ✅ 运行正常"
else
    echo "  ❌ 未响应"
fi

# 检查 Docker Compose 服务
echo ""
echo "📊 Docker Compose 服务:"
docker compose ps

# 检查 Celery Workers（如果在 Docker 中运行）
echo ""
echo "📊 Celery Workers:"
if docker compose ps | grep -q "celery"; then
    docker compose exec celery_worker_default celery -A YamagotiProjects inspect active 2>/dev/null || echo "  ⚠️  无法获取 worker 状态"
else
    echo "  ⚠️  未在 Docker 中运行"
fi

echo ""
echo "✅ 健康检查完成！"
