#!/bin/bash
# 执行数据库迁移

set -e

echo "🔄 执行数据库迁移..."

# 生成迁移文件
python manage.py makemigrations

# 应用迁移
python manage.py migrate

echo "✅ 数据库迁移完成！"
