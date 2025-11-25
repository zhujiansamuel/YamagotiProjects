#!/bin/bash
# 启动 Flower - Celery 监控工具

set -e

echo "🌸 启动 Flower..."

# 设置环境变量
export DJANGO_SETTINGS_MODULE=YamagotiProjects.settings

# 启动 Flower
celery -A YamagotiProjects flower --port=5555

echo ""
echo "✅ Flower 已启动！"
echo "   访问地址: http://localhost:5555"
