#!/bin/bash
# -*- coding: utf-8 -*-
# 数据库权限修复脚本

set -e

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_DIR"

echo "======================================"
echo "数据库权限诊断和修复工具"
echo "======================================"
echo ""
echo "项目目录: $PROJECT_DIR"
echo ""

# 读取数据库配置
DB_NAME="${POSTGRES_DATABASE:-applestockchecker_dev}"
DB_USER="${POSTGRES_USER:-samuelzhu}"
DB_PASSWORD="${POSTGRES_PASSWORD:-Xdb73008762}"
DB_HOST="${POSTGRES_HOST:-127.0.0.1}"
DB_PORT="${POSTGRES_PORT:-5433}"

echo "当前数据库配置:"
echo "  数据库: $DB_NAME"
echo "  用户: $DB_USER"
echo "  主机: $DB_HOST"
echo "  端口: $DB_PORT"
echo ""

# 检查 psql 是否可用
if ! command -v psql &> /dev/null; then
    echo "❌ 错误: psql 命令不可用"
    echo "请安装 PostgreSQL 客户端工具"
    exit 1
fi

# 检测 Python 命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到 Python 命令"
    echo "请安装 Python 3"
    exit 1
fi

echo "使用 Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# 选择操作
echo "请选择操作："
echo "  1) 诊断权限问题（推荐先运行）"
echo "  2) 修复权限（授予完整权限）"
echo "  3) 测试连接"
echo "  4) 退出"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "======================================"
        echo "运行权限诊断..."
        echo "======================================"
        $PYTHON_CMD scripts/diagnose_db_permissions.py
        ;;
    2)
        echo ""
        echo "======================================"
        echo "修复数据库权限..."
        echo "======================================"
        echo "⚠️  警告: 此操作需要超级用户权限（通常是 postgres 用户）"
        read -p "继续吗？(y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "请输入 PostgreSQL 超级用户（通常是 postgres）："
            read -p "超级用户名 [postgres]: " SUPERUSER
            SUPERUSER=${SUPERUSER:-postgres}

            PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$SUPERUSER" -d "$DB_NAME" -f scripts/fix_db_permissions.sql

            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ 权限修复成功！"
                echo ""
                echo "现在可以运行迁移了："
                echo "  python manage.py makemigrations"
                echo "  python manage.py migrate"
            else
                echo ""
                echo "❌ 权限修复失败"
                echo ""
                echo "可能的原因："
                echo "  1. 超级用户密码错误"
                echo "  2. 无法连接到数据库"
                echo "  3. 使用的用户不是超级用户"
                echo ""
                echo "手动修复方法："
                echo "  1. 使用 postgres 用户登录："
                echo "     psql -U postgres -d $DB_NAME"
                echo "  2. 运行以下命令："
                echo "     GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
                echo "     GRANT ALL PRIVILEGES ON SCHEMA public TO $DB_USER;"
                echo "     GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_USER;"
            fi
        else
            echo "操作已取消"
        fi
        ;;
    3)
        echo ""
        echo "======================================"
        echo "测试数据库连接..."
        echo "======================================"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();"

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 数据库连接成功！"
        else
            echo ""
            echo "❌ 数据库连接失败"
            echo ""
            echo "请检查："
            echo "  1. 数据库服务是否运行"
            echo "  2. 连接参数是否正确"
            echo "  3. 防火墙设置"
        fi
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效的选项"
        exit 1
        ;;
esac
