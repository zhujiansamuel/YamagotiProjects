#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库权限诊断脚本
用于检查数据库用户权限和表访问权限
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'YamagotiProjects.settings')
django.setup()

from django.db import connection
from django.conf import settings


def check_database_permissions():
    """检查数据库权限"""
    print("=" * 60)
    print("数据库权限诊断")
    print("=" * 60)

    db_config = settings.DATABASES['default']
    print(f"\n当前数据库配置:")
    print(f"  数据库: {db_config['NAME']}")
    print(f"  用户: {db_config['USER']}")
    print(f"  主机: {db_config['HOST']}")
    print(f"  端口: {db_config['PORT']}")
    print(f"  USE_PGBOUNCER: {os.getenv('USE_PGBOUNCER', 'false')}")

    with connection.cursor() as cursor:
        print("\n" + "=" * 60)
        print("1. 检查当前连接用户")
        print("=" * 60)
        cursor.execute("SELECT current_user, current_database();")
        user, db = cursor.fetchone()
        print(f"  当前用户: {user}")
        print(f"  当前数据库: {db}")

        print("\n" + "=" * 60)
        print("2. 检查用户角色和权限")
        print("=" * 60)
        cursor.execute("""
            SELECT
                r.rolname,
                r.rolsuper,
                r.rolcreatedb,
                r.rolcreaterole
            FROM pg_roles r
            WHERE r.rolname = current_user;
        """)
        role_info = cursor.fetchone()
        if role_info:
            print(f"  角色名: {role_info[0]}")
            print(f"  超级用户: {role_info[1]}")
            print(f"  可创建数据库: {role_info[2]}")
            print(f"  可创建角色: {role_info[3]}")

        print("\n" + "=" * 60)
        print("3. 检查 django_migrations 表权限")
        print("=" * 60)
        try:
            cursor.execute("""
                SELECT
                    has_table_privilege(current_user, 'django_migrations', 'SELECT') as can_select,
                    has_table_privilege(current_user, 'django_migrations', 'INSERT') as can_insert,
                    has_table_privilege(current_user, 'django_migrations', 'UPDATE') as can_update,
                    has_table_privilege(current_user, 'django_migrations', 'DELETE') as can_delete;
            """)
            perms = cursor.fetchone()
            print(f"  SELECT 权限: {perms[0]}")
            print(f"  INSERT 权限: {perms[1]}")
            print(f"  UPDATE 权限: {perms[2]}")
            print(f"  DELETE 权限: {perms[3]}")

            if not all(perms):
                print("\n  ⚠️ 权限不足！需要所有权限才能进行迁移。")
        except Exception as e:
            print(f"  ❌ 无法检查表权限: {e}")
            print(f"  可能是表不存在或没有权限查询")

        print("\n" + "=" * 60)
        print("4. 检查数据库对象所有权")
        print("=" * 60)
        try:
            cursor.execute("""
                SELECT
                    schemaname,
                    tablename,
                    tableowner
                FROM pg_tables
                WHERE tablename = 'django_migrations';
            """)
            table_info = cursor.fetchone()
            if table_info:
                print(f"  Schema: {table_info[0]}")
                print(f"  表名: {table_info[1]}")
                print(f"  所有者: {table_info[2]}")

                if table_info[2] != user:
                    print(f"\n  ⚠️ 警告: 表所有者 ({table_info[2]}) 与当前用户 ({user}) 不一致")
            else:
                print("  ℹ️ django_migrations 表不存在（这是正常的，如果是首次迁移）")
        except Exception as e:
            print(f"  ❌ 无法检查表所有权: {e}")

        print("\n" + "=" * 60)
        print("5. 检查数据库级权限")
        print("=" * 60)
        cursor.execute("""
            SELECT
                datname,
                has_database_privilege(current_user, datname, 'CREATE') as can_create,
                has_database_privilege(current_user, datname, 'CONNECT') as can_connect
            FROM pg_database
            WHERE datname = current_database();
        """)
        db_perms = cursor.fetchone()
        print(f"  数据库: {db_perms[0]}")
        print(f"  CREATE 权限: {db_perms[1]}")
        print(f"  CONNECT 权限: {db_perms[2]}")

        print("\n" + "=" * 60)
        print("6. 检查 Schema 权限")
        print("=" * 60)
        cursor.execute("""
            SELECT
                nspname,
                has_schema_privilege(current_user, nspname, 'CREATE') as can_create,
                has_schema_privilege(current_user, nspname, 'USAGE') as can_usage
            FROM pg_namespace
            WHERE nspname = 'public';
        """)
        schema_perms = cursor.fetchone()
        print(f"  Schema: {schema_perms[0]}")
        print(f"  CREATE 权限: {schema_perms[1]}")
        print(f"  USAGE 权限: {schema_perms[2]}")

        print("\n" + "=" * 60)
        print("诊断完成")
        print("=" * 60)


if __name__ == '__main__':
    try:
        check_database_permissions()
    except Exception as e:
        print(f"\n❌ 诊断过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
