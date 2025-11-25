#!/bin/bash
# 启动 Celery Beat 定时任务调度器（本地开发）

set -e

echo "🚀 启动 Celery Beat..."

# 设置环境变量
export DJANGO_SETTINGS_MODULE=YamagotiProjects.settings

# 启动 Celery Beat
celery -A YamagotiProjects beat \
    -l info \
    --scheduler django_celery_beat.schedulers:DatabaseScheduler

echo "✅ Celery Beat 已停止"
