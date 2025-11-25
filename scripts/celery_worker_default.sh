#!/bin/bash
# 启动默认队列的 Celery Worker（本地开发）

set -e

echo "🚀 启动 Celery Worker (default 队列)..."

# 设置环境变量（如果需要）
export DJANGO_SETTINGS_MODULE=YamagotiProjects.settings

# 启动 Celery Worker
celery -A YamagotiProjects worker \
    -Q default,celery \
    -l info \
    -c 4 \
    --max-tasks-per-child=1000 \
    --hostname=worker_default@%h

echo "✅ Celery Worker (default) 已停止"
