#!/bin/bash
# -*- coding: utf-8 -*-
# Docker 环境数据库权限修复脚本

set -e

echo "======================================"
echo "Docker 数据库权限修复工具"
echo "======================================"
echo ""

# 容器名称
CONTAINER_NAME="${DB_CONTAINER_NAME:-yapp_postgres}"

# 数据库配置
DB_NAME="${POSTGRES_DATABASE:-applestockchecker_dev}"
DB_USER="${POSTGRES_USER:-samuelzhu}"
SUPERUSER="${POSTGRES_SUPERUSER:-postgres}"

echo "Docker 容器: $CONTAINER_NAME"
echo "数据库: $DB_NAME"
echo "普通用户: $DB_USER"
echo "超级用户: $SUPERUSER"
echo ""

# 检查 Docker 是否可用
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: docker 命令不可用"
    echo "请确保 Docker 已安装并正在运行"
    exit 1
fi

# 检查容器是否运行
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ 错误: 容器 $CONTAINER_NAME 未运行"
    echo ""
    echo "请先启动容器："
    echo "  docker-compose up -d db"
    echo ""
    echo "或查看所有运行的容器："
    echo "  docker ps"
    exit 1
fi

echo "✅ 容器 $CONTAINER_NAME 正在运行"
echo ""

# 选择操作
echo "请选择操作："
echo "  1) 诊断权限问题（推荐先运行）"
echo "  2) 修复权限（授予完整权限）"
echo "  3) 测试连接"
echo "  4) 进入 PostgreSQL 命令行"
echo "  5) 查看容器日志"
echo "  6) 退出"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "======================================"
        echo "运行权限诊断..."
        echo "======================================"
        echo ""
        echo "方法 1: 在容器内运行诊断脚本"
        echo "----------------------------------------"

        # 复制脚本到容器
        docker cp scripts/diagnose_db_permissions.py ${CONTAINER_NAME}:/tmp/

        # 在容器内运行诊断
        docker exec -it ${CONTAINER_NAME} psql -U ${SUPERUSER} -d ${DB_NAME} -c "
            SELECT
                'Current User: ' || current_user as info
            UNION ALL
            SELECT
                'Database: ' || current_database()
            UNION ALL
            SELECT
                'Table Owner: ' || tableowner
            FROM pg_tables
            WHERE tablename = 'django_migrations'
            UNION ALL
            SELECT
                CASE
                    WHEN has_table_privilege('${DB_USER}', 'django_migrations', 'SELECT')
                    THEN '✅ SELECT 权限: YES'
                    ELSE '❌ SELECT 权限: NO'
                END
            UNION ALL
            SELECT
                CASE
                    WHEN has_table_privilege('${DB_USER}', 'django_migrations', 'INSERT')
                    THEN '✅ INSERT 权限: YES'
                    ELSE '❌ INSERT 权限: NO'
                END
            UNION ALL
            SELECT
                CASE
                    WHEN has_table_privilege('${DB_USER}', 'django_migrations', 'UPDATE')
                    THEN '✅ UPDATE 权限: YES'
                    ELSE '❌ UPDATE 权限: NO'
                END
            UNION ALL
            SELECT
                CASE
                    WHEN has_table_privilege('${DB_USER}', 'django_migrations', 'DELETE')
                    THEN '✅ DELETE 权限: YES'
                    ELSE '❌ DELETE 权限: NO'
                END;
        " 2>/dev/null || echo "⚠️ django_migrations 表可能不存在（首次迁移时是正常的）"

        echo ""
        echo "方法 2: 查看所有表的权限"
        echo "----------------------------------------"
        docker exec -it ${CONTAINER_NAME} psql -U ${SUPERUSER} -d ${DB_NAME} -c "
            SELECT
                schemaname,
                tablename,
                tableowner,
                has_table_privilege('${DB_USER}', schemaname||'.'||tablename, 'SELECT') as can_select,
                has_table_privilege('${DB_USER}', schemaname||'.'||tablename, 'INSERT') as can_insert
            FROM pg_tables
            WHERE schemaname = 'public'
            LIMIT 10;
        "
        ;;

    2)
        echo ""
        echo "======================================"
        echo "修复数据库权限..."
        echo "======================================"
        echo "⚠️  警告: 此操作将授予用户 $DB_USER 完整的数据库权限"
        read -p "继续吗？(y/N): " confirm

        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo ""
            echo "开始修复权限..."

            # 在容器内执行权限修复 SQL
            docker exec -i ${CONTAINER_NAME} psql -U ${SUPERUSER} -d ${DB_NAME} <<EOF
-- 授予数据库所有权限
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};

-- 授予 public schema 的所有权限
GRANT ALL PRIVILEGES ON SCHEMA public TO ${DB_USER};

-- 授予现有所有表的所有权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DB_USER};

-- 授予现有所有序列的所有权限
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};

-- 设置默认权限（对未来创建的对象）
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO ${DB_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO ${DB_USER};

-- 显示结果
SELECT '✅ 权限修复成功！' as status;
EOF

            if [ $? -eq 0 ]; then
                echo ""
                echo "======================================"
                echo "✅ 权限修复成功！"
                echo "======================================"
                echo ""
                echo "现在可以运行迁移了："
                echo "  python manage.py makemigrations"
                echo "  python manage.py migrate"
                echo ""
                echo "或在 Docker 中运行："
                echo "  docker-compose exec web python manage.py migrate"
            else
                echo ""
                echo "❌ 权限修复失败"
                echo "请检查容器和数据库状态"
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

        echo ""
        echo "从主机连接 (端口 5433):"
        echo "----------------------------------------"
        psql -h 127.0.0.1 -p 5433 -U ${DB_USER} -d ${DB_NAME} -c "SELECT version();" 2>&1 || \
            echo "⚠️ 从主机连接失败（可能需要 PGPASSWORD 环境变量）"

        echo ""
        echo "从容器内连接:"
        echo "----------------------------------------"
        docker exec ${CONTAINER_NAME} psql -U ${DB_USER} -d ${DB_NAME} -c "SELECT version();"

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 数据库连接成功！"
        fi
        ;;

    4)
        echo ""
        echo "======================================"
        echo "进入 PostgreSQL 命令行..."
        echo "======================================"
        echo ""
        echo "使用超级用户 $SUPERUSER 登录"
        echo "输入 \\q 或 Ctrl+D 退出"
        echo ""
        docker exec -it ${CONTAINER_NAME} psql -U ${SUPERUSER} -d ${DB_NAME}
        ;;

    5)
        echo ""
        echo "======================================"
        echo "查看容器日志..."
        echo "======================================"
        echo ""
        docker logs --tail 50 ${CONTAINER_NAME}
        ;;

    6)
        echo "退出"
        exit 0
        ;;

    *)
        echo "无效的选项"
        exit 1
        ;;
esac
