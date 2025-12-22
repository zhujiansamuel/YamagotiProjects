# FeatureSnapshot 聚合日志说明

## 日志格式

当 FeatureSnapshot 聚合执行时，会输出以下特别的日志行，方便你监控运行情况：

### 1. 聚合开始

```
🔄 [FeatureSnapshot 聚合] 开始计算 | 时间点: 2025-09-20T05:00:00+00:00 | 窗口: 2025-09-20T05:00:00+00:00 → 2025-09-20T05:15:00+00:00 | 聚合步长: 15分钟 | 模式: 窗口
```

**标识**：`🔄 [FeatureSnapshot 聚合] 开始计算`

**包含信息**：
- 时间点：当前处理的时间戳
- 窗口：聚合窗口的起止时间
- 聚合步长：15分钟（或其他配置值）
- 模式：窗口模式或单分钟模式

### 2. 数据源信息

```
  📊 [数据源] shops: 15个, iphones: 8个 | 来源: 窗口PSTA数据
```

**标识**：`📊 [数据源]`

**包含信息**：
- shops 数量：窗口内有多少个店铺
- iphones 数量：窗口内有多少个 iPhone 机型
- 来源：数据来自哪里
  - `窗口PSTA数据`：从数据库窗口查询（正常）
  - `rows参数`：从单分钟 rows 提取（单分钟模式）

### 3. 特征写入统计

```
  ✍️  [特征写入] Case1(shop×iphone): 120, Case2(shopcohort×iphone): 8, Case3(shop×cohort): 15, Case4(shopcohort×cohort): 1 | 总计: 144 个组合
```

**标识**：`✍️ [特征写入]`

**包含信息**：
- Case1 (shop×iphone)：各店铺 × 各机型（单值，原始曲线）
- Case2 (shopcohort×iphone)：店铺组合 × 各机型（加权）
- Case3 (shop×cohort)：各店铺 × 机型组合（加权）
- Case4 (shopcohort×cohort)：店铺组合 × 机型组合（加权）
- 总计：4 种组合的总数

### 4. 聚合完成

```
✅ [FeatureSnapshot 聚合] 完成 | 时间点: 2025-09-20T05:00:00+00:00 | bucket: 2025-09-20T05:00:00+00:00 | 生成记录数: 156 条
```

**标识**：`✅ [FeatureSnapshot 聚合] 完成`

**包含信息**：
- 时间点：处理的时间戳
- bucket：FeatureSnapshot 的 bucket 字段值
- 生成记录数：实际写入数据库的 FeatureSnapshot 记录数
  - 每个组合有约 39 个统计指标（mean, median, std, count, ...）
  - 典型值：144 个组合 × 39 个指标 ≈ 156 条记录（部分指标可能为空）

## 完整示例

```
2025-12-22 10:30:00 INFO 🔄 [FeatureSnapshot 聚合] 开始计算 | 时间点: 2025-09-20T05:00:00+00:00 | 窗口: 2025-09-20T05:00:00+00:00 → 2025-09-20T05:15:00+00:00 | 聚合步长: 15分钟 | 模式: 窗口
2025-12-22 10:30:01 INFO   📊 [数据源] shops: 15个, iphones: 8个 | 来源: 窗口PSTA数据
2025-12-22 10:30:05 INFO   ✍️  [特征写入] Case1(shop×iphone): 120, Case2(shopcohort×iphone): 8, Case3(shop×cohort): 15, Case4(shopcohort×cohort): 1 | 总计: 144 个组合
2025-12-22 10:30:06 INFO ✅ [FeatureSnapshot 聚合] 完成 | 时间点: 2025-09-20T05:00:00+00:00 | bucket: 2025-09-20T05:00:00+00:00 | 生成记录数: 156 条
```

## 日志级别

所有 FeatureSnapshot 聚合日志都使用 `INFO` 级别，可以通过以下方式查看：

### Docker 环境

```bash
# 查看 Celery worker 日志
docker compose logs -f worker

# 只看 FeatureSnapshot 相关日志
docker compose logs -f worker | grep "FeatureSnapshot 聚合"

# 只看开始和完成
docker compose logs -f worker | grep -E "🔄|✅"
```

### 本地环境

```bash
# 查看 Django 日志（如果配置了日志文件）
tail -f logs/celery.log | grep "FeatureSnapshot 聚合"

# 只看开始和完成
tail -f logs/celery.log | grep -E "🔄|✅"
```

## 故障排查

### 场景 1: 看到开始但没有完成

```
🔄 [FeatureSnapshot 聚合] 开始计算 ...
(没有 ✅ 完成日志)
```

**可能原因**：
- 任务执行失败（查看错误日志）
- 任务还在执行（等待或检查 Celery worker 状态）
- 数据库连接问题

**检查方法**：
```bash
# 查看错误日志
docker compose logs worker | grep -E "ERROR|Exception|Traceback"

# 查看 Celery 任务状态
docker compose exec web python manage.py shell
>>> from celery.result import AsyncResult
>>> result = AsyncResult('task-id')
>>> result.status
```

### 场景 2: 生成记录数为 0

```
✅ [FeatureSnapshot 聚合] 完成 | ... | 生成记录数: 0 条
```

**可能原因**：
- 窗口内没有数据（shops: 0个, iphones: 0个）
- 数据被价格过滤器过滤掉
- bucket_by_minute bug 未修复（检查是否是最新代码）

**检查方法**：
```bash
# 查看数据源日志
docker compose logs worker | grep "📊 \[数据源\]"

# 如果 shops/iphones 都为 0，检查窗口内是否有 PSTA 数据
docker compose exec web python manage.py shell
>>> from AppleStockChecker.models import PurchasingShopTimeAnalysis
>>> from datetime import datetime, timezone
>>> start = datetime(2025, 9, 20, 5, 0, 0, tzinfo=timezone.utc)
>>> end = datetime(2025, 9, 20, 5, 15, 0, tzinfo=timezone.utc)
>>> PurchasingShopTimeAnalysis.objects.filter(
...     Timestamp_Time__gte=start,
...     Timestamp_Time__lt=end
... ).count()
```

### 场景 3: 特征写入为 0

```
  ✍️  [特征写入] Case1(shop×iphone): 0, Case2(shopcohort×iphone): 0, Case3(shop×cohort): 0, Case4(shopcohort×cohort): 0 | 总计: 0 个组合
```

**可能原因**：
- 数据被异常值过滤器过滤掉（`_filter_outliers_by_mean_band`）
- 价格数据全部为 None
- ShopWeightProfile 或 Cohort 配置缺失（影响 Case2-4）

**检查方法**：
```bash
# 查看价格过滤日志
docker compose logs worker | grep "动态价格过滤"

# 检查 ShopWeightProfile 配置
docker compose exec web python manage.py shell
>>> from AppleStockChecker.models import ShopWeightProfile
>>> ShopWeightProfile.objects.count()
```

## 性能监控

### 计算时间

通过开始和完成日志的时间戳可以计算聚合耗时：

```
2025-12-22 10:30:00 INFO 🔄 [FeatureSnapshot 聚合] 开始计算 ...
2025-12-22 10:30:06 INFO ✅ [FeatureSnapshot 聚合] 完成 ...

耗时：6 秒
```

**典型耗时**（参考）：
- 窗口内数据量 < 1000 条：1-3 秒
- 窗口内数据量 1000-5000 条：3-10 秒
- 窗口内数据量 > 5000 条：10-30 秒

### 吞吐量监控

```bash
# 统计每小时生成多少个 FeatureSnapshot
docker compose logs worker --since 1h | grep "✅ \[FeatureSnapshot 聚合\] 完成" | wc -l

# 统计生成的总记录数
docker compose logs worker --since 1h | grep "✅ \[FeatureSnapshot 聚合\] 完成" | \
  grep -oP '生成记录数: \K\d+' | awk '{sum+=$1} END {print sum}'
```

## 日志配置

### 调整日志级别

如果日志太多，可以调整级别：

**Django settings.py**:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'AppleStockChecker.tasks.timestamp_alignment_task': {
            'handlers': ['console'],
            'level': 'INFO',  # 或 'WARNING' 减少日志
        },
    },
}
```

### 过滤特定日志

使用 grep 过滤只看关键信息：

```bash
# 只看聚合的开始和完成
docker compose logs -f worker | grep -E "🔄.*开始计算|✅.*完成"

# 只看失败的聚合（生成记录数为 0）
docker compose logs -f worker | grep "生成记录数: 0 条"

# 只看特定时间点
docker compose logs -f worker | grep "2025-09-20T05:00:00"
```

## 总结

使用这些日志，你可以：

1. **实时监控**：知道聚合何时开始、何时完成
2. **数据验证**：确认窗口内有多少 shop/iphone 数据
3. **性能分析**：计算聚合耗时
4. **故障诊断**：快速定位问题（无数据、过滤、配置缺失等）
5. **生产监控**：统计每小时生成的 FeatureSnapshot 数量

**关键标识**：
- 🔄 = 开始
- 📊 = 数据源
- ✍️ = 特征写入
- ✅ = 完成

在日志中搜索这些 emoji 可以快速定位 FeatureSnapshot 聚合的相关信息！
