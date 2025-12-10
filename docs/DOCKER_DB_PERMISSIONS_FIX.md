# Docker 环境数据库权限快速修复指南

## 🐳 你的环境

- 数据库运行在 Docker 容器中：`yapp_postgres`
- 端口映射：`5433:5432`（主机端口 → 容器端口）
- 数据库：`applestockchecker_dev`
- 用户：`samuelzhu`

## ⚡ 快速修复（推荐）

### 方法 1：使用自动化脚本（最简单）

```bash
./scripts/docker_fix_permissions.sh
```

然后选择：
- **选项 1**：诊断权限问题
- **选项 2**：一键修复权限

### 方法 2：单行命令修复

```bash
docker exec -i yapp_postgres psql -U postgres -d applestockchecker_dev <<'EOF'
GRANT ALL PRIVILEGES ON DATABASE applestockchecker_dev TO samuelzhu;
GRANT ALL PRIVILEGES ON SCHEMA public TO samuelzhu;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO samuelzhu;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO samuelzhu;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO samuelzhu;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO samuelzhu;
SELECT '✅ 权限修复成功！' as status;
EOF
```

### 方法 3：使用 SQL 文件

```bash
docker exec -i yapp_postgres psql -U postgres -d applestockchecker_dev < scripts/fix_db_permissions.sql
```

## 🔍 诊断权限问题

### 快速检查

```bash
docker exec yapp_postgres psql -U postgres -d applestockchecker_dev -c "
SELECT
    'User' as type, current_user as name
UNION ALL
SELECT
    'Database', current_database()
UNION ALL
SELECT
    'Can SELECT', CASE WHEN has_table_privilege('samuelzhu', 'django_migrations', 'SELECT') THEN '✅ YES' ELSE '❌ NO' END
UNION ALL
SELECT
    'Can INSERT', CASE WHEN has_table_privilege('samuelzhu', 'django_migrations', 'INSERT') THEN '✅ YES' ELSE '❌ NO' END;
"
```

### 查看所有表的所有者

```bash
docker exec yapp_postgres psql -U postgres -d applestockchecker_dev -c "
SELECT tablename, tableowner
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
"
```

## 🛠️ 常用 Docker 数据库命令

### 进入 PostgreSQL 命令行

```bash
# 使用超级用户
docker exec -it yapp_postgres psql -U postgres -d applestockchecker_dev

# 使用普通用户
docker exec -it yapp_postgres psql -U samuelzhu -d applestockchecker_dev
```

### 查看容器状态

```bash
# 检查容器是否运行
docker ps | grep yapp_postgres

# 查看容器日志
docker logs yapp_postgres

# 查看最近 50 行日志
docker logs --tail 50 yapp_postgres
```

### 在 Docker 容器内运行 Django 命令

```bash
# 运行迁移
docker-compose exec web python manage.py migrate

# 创建迁移
docker-compose exec web python manage.py makemigrations

# 查看迁移状态
docker-compose exec web python manage.py showmigrations
```

## 🔧 修复后验证

修复权限后，验证是否成功：

```bash
# 1. 测试连接
docker exec yapp_postgres psql -U samuelzhu -d applestockchecker_dev -c "SELECT 1;"

# 2. 运行迁移
docker-compose exec web python manage.py migrate

# 或从主机运行（如果 Django 不在容器中）
python manage.py migrate
```

## 📋 完整的权限修复流程

### 步骤 1：确保容器运行

```bash
docker-compose up -d db
```

### 步骤 2：运行修复脚本

```bash
./scripts/docker_fix_permissions.sh
# 选择选项 2
```

### 步骤 3：验证修复

```bash
./scripts/docker_fix_permissions.sh
# 选择选项 1（诊断）
```

### 步骤 4：运行迁移

```bash
python manage.py migrate
```

## ⚠️ 常见问题

### 问题 1：容器未运行

**错误**：`Error: No such container: yapp_postgres`

**解决**：
```bash
docker-compose up -d db
```

### 问题 2：从主机连接失败

**错误**：`psql: error: connection to server at "127.0.0.1", port 5433 failed`

**原因**：端口未映射或容器未启动

**解决**：
1. 检查容器状态：`docker ps | grep yapp_postgres`
2. 检查端口映射：`docker port yapp_postgres`
3. 重启容器：`docker-compose restart db`

### 问题 3：权限修复后仍然报错

**可能原因**：
- 使用了 PgBouncer（端口 6432）而不是直连 PostgreSQL（端口 5433）
- 环境变量配置错误

**解决**：
```bash
# 临时禁用 PgBouncer
export USE_PGBOUNCER=false
export POSTGRES_PORT=5433
export POSTGRES_HOST=127.0.0.1

# 运行迁移
python manage.py migrate

# 恢复环境变量
unset USE_PGBOUNCER
unset POSTGRES_PORT
unset POSTGRES_HOST
```

### 问题 4：Django 找不到模块

**错误**：`ModuleNotFoundError: No module named 'YamagotiProjects'`

**解决**：确保从项目根目录运行命令
```bash
cd /path/to/YamagotiProjects
python manage.py migrate
```

## 🔐 权限最佳实践

### 开发环境（推荐）

授予完整权限，方便开发：

```sql
GRANT ALL PRIVILEGES ON DATABASE applestockchecker_dev TO samuelzhu;
ALTER DATABASE applestockchecker_dev OWNER TO samuelzhu;
```

### 生产环境（最小权限）

只授予必要的权限：

```sql
GRANT CONNECT ON DATABASE applestockchecker_dev TO samuelzhu;
GRANT USAGE ON SCHEMA public TO samuelzhu;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO samuelzhu;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO samuelzhu;
```

## 📚 相关文档

- [完整权限故障排除指南](./DB_PERMISSIONS_TROUBLESHOOTING.md)
- [Docker Compose 配置](../docker-compose.yml)
- [数据库设置](../YamagotiProjects/settings.py)

## 🆘 需要帮助？

如果以上方法都无法解决问题：

1. **查看详细日志**：
   ```bash
   docker logs yapp_postgres 2>&1 | grep -i "error\|permission"
   ```

2. **检查数据库配置**：
   ```bash
   docker exec yapp_postgres psql -U postgres -c "SHOW all;"
   ```

3. **重新初始化数据库**（⚠️ 会删除所有数据）：
   ```bash
   docker-compose down -v
   docker-compose up -d db
   # 然后重新运行权限修复
   ```

## 🎯 快速参考

| 操作 | 命令 |
|------|------|
| 修复权限 | `./scripts/docker_fix_permissions.sh` → 选项 2 |
| 诊断问题 | `./scripts/docker_fix_permissions.sh` → 选项 1 |
| 进入数据库 | `docker exec -it yapp_postgres psql -U postgres -d applestockchecker_dev` |
| 运行迁移 | `python manage.py migrate` 或 `docker-compose exec web python manage.py migrate` |
| 查看日志 | `docker logs yapp_postgres` |
| 重启容器 | `docker-compose restart db` |
