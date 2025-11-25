#!/bin/bash
# 启动 WebScraper 专用队列的 Celery Worker（本地开发）

set -e

echo "🚀 启动 Celery Worker (webscraper 队列)..."

# 设置环境变量（如果需要）
export DJANGO_SETTINGS_MODULE=YamagotiProjects.settings

# 启动 Celery Worker - 专门处理 webscraper 队列
celery -A YamagotiProjects worker \
    -Q webscraper \
    -l info \
    -c 2 \
    --max-tasks-per-child=100 \
    --hostname=worker_webscraper@%h

echo "✅ Celery Worker (webscraper) 已停止"
