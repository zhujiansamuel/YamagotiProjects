# 数据库权限问题故障排除指南

## 问题描述

运行 `python manage.py makemigrations` 或 `python manage.py migrate` 时出现错误：

```
django.db.utils.ProgrammingError: permission denied for table django_migrations
```

## 原因分析

这个错误表明数据库用户没有足够的权限访问或操作 `django_migrations` 表。可能的原因包括：

1. **用户权限不足**：数据库用户没有对表的 SELECT、INSERT、UPDATE、DELETE 权限
2. **表所有权问题**：表的所有者不是当前连接的用户
3. **Schema 权限问题**：用户没有 public schema 的 USAGE 或 CREATE 权限
4. **PgBouncer 限制**：在事务池模式下可能遇到权限检查问题

## 解决方案

### 方案 1：使用自动化脚本（推荐）

我们提供了自动化的诊断和修复工具：

```bash
# 1. 先诊断问题
./scripts/fix_permissions.sh
# 选择选项 1 进行诊断

# 2. 如果确认是权限问题，运行修复
./scripts/fix_permissions.sh
# 选择选项 2 进行修复
```

### 方案 2：手动诊断

运行诊断脚本查看详细信息：

```bash
python scripts/diagnose_db_permissions.py
```

这个脚本会检查：
- 当前连接用户和数据库
- 用户角色和权限
- django_migrations 表权限
- 数据库对象所有权
- 数据库级和 Schema 级权限

### 方案 3：手动修复权限

#### 3.1 使用 PostgreSQL 超级用户修复

```bash
# 连接到数据库（使用 postgres 超级用户）
psql -U postgres -d applestockchecker_dev

# 执行以下 SQL 命令：
```

```sql
-- 授予数据库所有权限
GRANT ALL PRIVILEGES ON DATABASE applestockchecker_dev TO samuelzhu;

-- 授予 public schema 的所有权限
GRANT ALL PRIVILEGES ON SCHEMA public TO samuelzhu;

-- 授予现有所有表的所有权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO samuelzhu;

-- 授予现有所有序列的所有权限
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO samuelzhu;

-- 设置默认权限（对未来创建的对象）
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON TABLES TO samuelzhu;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON SEQUENCES TO samuelzhu;
```

#### 3.2 或者使用提供的 SQL 脚本

```bash
psql -U postgres -d applestockchecker_dev -f scripts/fix_db_permissions.sql
```

### 方案 4：Docker 环境修复

如果你在 Docker 中运行：

```bash
# 进入 PostgreSQL 容器
docker exec -it yapp_postgres psql -U samuelzhu -d applestockchecker_dev

# 如果需要超级用户权限
docker exec -it yapp_postgres psql -U postgres -d applestockchecker_dev

# 然后执行上述 SQL 命令
```

或者直接在容器中执行脚本：

```bash
docker exec -i yapp_postgres psql -U postgres -d applestockchecker_dev < scripts/fix_db_permissions.sql
```

## 特殊情况处理

### 情况 1：使用 PgBouncer

如果使用 PgBouncer（`USE_PGBOUNCER=true`），某些操作可能受限。建议：

1. **临时直连数据库进行迁移**：
   ```bash
   # 临时禁用 PgBouncer
   export USE_PGBOUNCER=false
   python manage.py migrate
   # 之后恢复
   unset USE_PGBOUNCER
   ```

2. **或者连接到 PostgreSQL 而不是 PgBouncer**：
   ```bash
   # 修改端口从 6432 (PgBouncer) 到 5432 (PostgreSQL)
   export POSTGRES_PORT=5433
   python manage.py migrate
   ```

### 情况 2：表已存在但权限不足

如果 `django_migrations` 表已经存在但没有权限：

```sql
-- 检查表所有者
SELECT tableowner FROM pg_tables WHERE tablename = 'django_migrations';

-- 如果所有者不是 samuelzhu，修改所有者
ALTER TABLE django_migrations OWNER TO samuelzhu;

-- 或者授予权限
GRANT ALL PRIVILEGES ON TABLE django_migrations TO samuelzhu;
```

### 情况 3：首次迁移

如果是首次运行迁移，确保用户有创建表的权限：

```sql
-- 授予 schema 的 CREATE 权限
GRANT CREATE ON SCHEMA public TO samuelzhu;

-- 或者让用户成为数据库所有者
ALTER DATABASE applestockchecker_dev OWNER TO samuelzhu;
```

## 验证修复

修复后，验证权限是否正确：

```bash
# 1. 运行诊断脚本
python scripts/diagnose_db_permissions.py

# 2. 尝试运行迁移
python manage.py makemigrations
python manage.py migrate

# 3. 检查迁移记录
python manage.py showmigrations
```

## 预防措施

为避免将来出现权限问题：

### 1. 数据库初始化时设置正确权限

在创建数据库时：

```sql
CREATE DATABASE applestockchecker_dev OWNER samuelzhu;
```

### 2. 使用一致的用户

确保所有 Django 操作都使用同一个数据库用户。

### 3. 在 Docker Compose 中配置

如果使用 Docker，在 `docker-compose.yml` 中确保：

```yaml
db:
  environment:
    POSTGRES_USER: samuelzhu
    POSTGRES_PASSWORD: your_password
    POSTGRES_DB: applestockchecker_dev
```

### 4. 定期备份

在进行权限修改前备份数据库：

```bash
# 使用提供的备份脚本
./scripts/pg_dump.sh
```

## 常见错误信息

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `permission denied for table django_migrations` | 表权限不足 | 授予表权限或修改所有者 |
| `permission denied for schema public` | Schema 权限不足 | `GRANT ALL ON SCHEMA public TO user` |
| `must be owner of table django_migrations` | 不是表所有者 | `ALTER TABLE ... OWNER TO user` |
| `permission denied for database` | 数据库权限不足 | `GRANT ALL ON DATABASE ... TO user` |

## 获取帮助

如果上述方案都无法解决问题：

1. **查看完整错误堆栈**：
   ```bash
   python manage.py migrate --verbosity 3
   ```

2. **检查 PostgreSQL 日志**：
   ```bash
   # Docker 环境
   docker logs yapp_postgres

   # 或查看日志文件
   tail -f /var/log/postgresql/postgresql-*.log
   ```

3. **检查数据库连接**：
   ```bash
   ./scripts/fix_permissions.sh
   # 选择选项 3 测试连接
   ```

4. **联系数据库管理员**：如果在生产环境，请联系 DBA 授予适当权限。

## 相关文档

- [PostgreSQL 权限管理](https://www.postgresql.org/docs/current/ddl-priv.html)
- [Django 数据库配置](https://docs.djangoproject.com/en/stable/ref/databases/)
- [PgBouncer 最佳实践](https://www.pgbouncer.org/usage.html)

## 更新历史

- 2025-12-08: 初始版本，添加自动化诊断和修复工具
