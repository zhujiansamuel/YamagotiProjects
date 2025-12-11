# FeatureSnapshot 清理脚本使用指南

## 📋 脚本说明

`clear_feature_snapshots.py` 用于清除 `FeatureSnapshot` 表的所有数据。

### 适用场景

- ✅ 统计指标重构后清除历史数据
- ✅ 测试环境数据重置
- ✅ 修复数据错误后的全量重算
- ✅ 释放数据库存储空间

### 安全特性

- ✅ 默认 Dry-run 模式（仅查看统计）
- ✅ 显示详细统计信息（记录数、表大小、时间范围等）
- ✅ 二次确认机制（需输入 `DELETE` 确认）
- ✅ 分批删除避免长事务锁表
- ✅ 进度显示和性能统计

---

## 🚀 使用方法

### 方法 1: 使用 Shell 包装器（推荐）

```bash
# 1. Dry-run（仅查看统计，不删除）
./scripts/clear_feature_snapshots.sh

# 2. 实际删除（需要二次确认）
./scripts/clear_feature_snapshots.sh --execute

# 3. 静默删除（跳过确认，危险！）
./scripts/clear_feature_snapshots.sh --execute --force

# 4. 自定义批量大小
./scripts/clear_feature_snapshots.sh --execute --batch-size 5000
```

### 方法 2: 通过 Docker Compose

```bash
# Dry-run
docker compose exec web python scripts/clear_feature_snapshots.py

# 实际删除
docker compose exec web python scripts/clear_feature_snapshots.py --execute

# 静默删除
docker compose exec web python scripts/clear_feature_snapshots.py --execute --force
```

### 方法 3: 在容器内直接运行

```bash
# 进入容器
docker compose exec web bash

# 运行脚本
python scripts/clear_feature_snapshots.py
python scripts/clear_feature_snapshots.py --execute
```

---

## 📊 输出示例

### Dry-run 模式（默认）

```
======================================================================
 FeatureSnapshot 清理脚本
======================================================================
 时间: 2025-12-11 14:30:00+09:00
 模式: 🟢 试运行模式 (DRY-RUN)

======================================================================
 FeatureSnapshot 表统计信息
======================================================================

📊 总记录数: 1,234,567
   表大小: 245 MB
   时间范围: 2025-10-01 00:00:00+09:00 ~ 2025-12-11 14:00:00+09:00

📈 数据最终化状态:
   is_final=True:  800,000
   is_final=False: 434,567

🏷️  Top 10 特征名 (name):
   mean                     500,000 条
   median                   500,000 条
   std                      100,000 条
   dispersion                80,000 条
   count                     50,000 条
   ...

🎯 Top 10 作用域前缀 (scope):
   shop                     ~50,000 个
   shopcohort               ~20,000 个
   overall                  ~10,000 个
   cohort                    ~5,000 个

======================================================================

💡 提示：这是 dry-run 模式，数据未被删除
   如需实际删除，请使用: --execute
   删除前建议备份: ./scripts/pg_dump.sh
```

### Execute 模式

```
======================================================================
 FeatureSnapshot 清理脚本
======================================================================
 时间: 2025-12-11 14:30:00+09:00
 模式: 🔴 执行模式 (EXECUTE)

[显示统计信息...]

⚠️  警告：即将删除 1,234,567 条 FeatureSnapshot 记录！
   此操作不可逆，建议先执行数据库备份：./scripts/pg_dump.sh

   确认删除？请输入 'DELETE' 继续，或按 Enter 取消: DELETE

🔄 开始删除 1,234,567 条记录（批量大小: 10,000）...
   Batch #1: 已删除 10,000/1,234,567 (0.8%)
   Batch #2: 已删除 20,000/1,234,567 (1.6%)
   Batch #3: 已删除 30,000/1,234,567 (2.4%)
   ...
   Batch #124: 已删除 1,234,567/1,234,567 (100.0%)

✅ 删除完成！
   总删除记录: 1,234,567
   总批次数: 124
   耗时: 45.32 秒
   速度: 27,245 条/秒

✅ 验证通过：FeatureSnapshot 表已完全清空
```

---

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--execute` | flag | False | 实际执行删除（默认为 dry-run） |
| `--force` | flag | False | 跳过二次确认（危险！） |
| `--batch-size` | int | 10000 | 每批删除的记录数 |

---

## ⚠️ 重要注意事项

### 1. 删除前务必备份

```bash
# 执行数据库备份
./scripts/pg_dump.sh
```

### 2. 删除操作不可逆

FeatureSnapshot 记录删除后无法恢复，除非从备份还原。

### 3. 生产环境使用建议

- ✅ 选择业务低峰期执行
- ✅ 提前通知相关人员
- ✅ 确认依赖该数据的服务已停止
- ✅ 准备回滚方案（数据库备份）

### 4. 性能考虑

- 默认批量大小 10,000 适合大多数场景
- 如果数据库性能较差，可降低批量大小（`--batch-size 5000`）
- 如果数据库性能很好，可增大批量大小（`--batch-size 20000`）

### 5. 相关操作

清空 FeatureSnapshot 后，通常需要重新运行聚合任务来重新生成数据：

```bash
# 调用 API 重新生成统计数据
curl -X POST http://localhost:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/ \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp_iso": "2025-12-11T14:00:00+09:00",
    "agg_minutes": 15,
    "agg_mode": "boundary",
    "force_agg": true
  }'
```

---

## 🔧 故障排查

### 问题 1: 权限不足

```bash
# 确保脚本有执行权限
chmod +x scripts/clear_feature_snapshots.py
chmod +x scripts/clear_feature_snapshots.sh
```

### 问题 2: Django 环境错误

确保通过 Docker 容器运行，或者在正确的 Python 虚拟环境中。

### 问题 3: 删除速度慢

- 调整 `--batch-size` 参数
- 检查数据库性能和连接
- 确认没有其他进程锁表

### 问题 4: 中断恢复

如果删除过程被中断（Ctrl+C），部分数据已被删除：
- 重新运行脚本会继续删除剩余数据
- 或者使用 `--force` 跳过确认直接删除剩余数据

---

## 📝 日志记录

脚本输出可以重定向到日志文件：

```bash
./scripts/clear_feature_snapshots.sh --execute 2>&1 | tee clear_feature_snapshots_$(date +%Y%m%d_%H%M%S).log
```

---

## 🔗 相关文档

- [CLAUDE.md](/home/user/YamagotiProjects/CLAUDE.md) - 项目开发规范
- [timestamp_alignment_task.py](/home/user/YamagotiProjects/AppleStockChecker/tasks/timestamp_alignment_task.py) - 统计指标计算逻辑
- [api.py](/home/user/YamagotiProjects/AppleStockChecker/api.py) - 触发聚合任务的 API

---

## 版本历史

- `v1.0` (2025-12-11): 初始版本
  - 支持 dry-run 模式
  - 分批删除
  - 二次确认机制
  - 详细统计信息
