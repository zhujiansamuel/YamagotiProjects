-- 修复数据库权限脚本
-- 使用方法: psql -U postgres -d applestockchecker_dev -f fix_db_permissions.sql

-- ============================================================
-- 方案 1: 授予用户完整的数据库权限（推荐用于开发环境）
-- ============================================================

-- 授予数据库所有权限
GRANT ALL PRIVILEGES ON DATABASE applestockchecker_dev TO samuelzhu;

-- 授予 public schema 的所有权限
GRANT ALL PRIVILEGES ON SCHEMA public TO samuelzhu;

-- 授予现有所有表的所有权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO samuelzhu;

-- 授予现有所有序列的所有权限
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO samuelzhu;

-- 设置默认权限（对未来创建的对象）
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO samuelzhu;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO samuelzhu;

-- 如果表已存在，修改所有者
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
    LOOP
        EXECUTE 'ALTER TABLE ' || quote_ident(r.tablename) || ' OWNER TO samuelzhu;';
    END LOOP;
END $$;

-- 修改序列所有者
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public')
    LOOP
        EXECUTE 'ALTER SEQUENCE ' || quote_ident(r.sequence_name) || ' OWNER TO samuelzhu;';
    END LOOP;
END $$;

SELECT 'Permissions fixed successfully!' as status;

-- ============================================================
-- 方案 2: 只授予 django_migrations 表的权限（最小权限）
-- ============================================================
-- 如果你只想修复 django_migrations 的权限，取消下面的注释：

-- GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE django_migrations TO samuelzhu;
-- ALTER TABLE django_migrations OWNER TO samuelzhu;
