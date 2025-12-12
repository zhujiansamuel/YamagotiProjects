# 禁用 OverallBar 和 CohortBar 计算说明

## 📋 修改概述

**日期**: 2025-12-12
**修改人**: Claude
**影响范围**: 统计指标计算流程

---

## ✅ 修改内容

### 已禁用的计算模块

在 `AppleStockChecker/tasks/timestamp_alignment_task.py` 中注释掉了以下计算：

1. **OverallBar 计算** (`_agg_overallbar`)
   - 位置：`_run_aggregation` 函数第1704-1715行
   - 功能：每个 iPhone 的全店统计（mean, median, std, dispersion, shop_count）

2. **CohortBar 计算** (`_agg_cohortbar`)
   - 位置：`_run_aggregation` 函数第1717-1724行
   - 功能：机型组合的全店统计（mean, median, std, dispersion, n_models, shop_count_agg）

3. **时间序列指标中的 OverallBar/CohortBar 基值收集**
   - 位置：`_agg_time_series_features` 函数第1082-1100行
   - 功能：为 `scope="overall:iphone:*"` 和 `scope="cohort:*"` 的时间序列指标提供基值

---

## 🔍 保留的计算模块

### ✅ FeatureSnapshot 四类组合（完全不受影响）

以下计算**继续正常工作**，因为它们直接从原始数据 `PurchasingShopTimeAnalysis` 计算：

1. **Case 1**: 各店 × 各 iPhone
   - Scope: `shop:1|iphone:10`
   - 计算：mean, median, std, dispersion, count

2. **Case 2**: 组合店 × 各 iPhone（带店铺权重 + 时效权重）
   - Scope: `shopcohort:premium|iphone:10`
   - 计算：mean, median, std, dispersion, count

3. **Case 3**: 各店 × 组合 iPhone（带机型权重 + 时效权重）
   - Scope: `shop:1|cohort:flagship`
   - 计算：mean, median, std, dispersion, count

4. **Case 4**: 组合店 × 组合 iPhone（带店铺权重 + 机型权重 + 时效权重）
   - Scope: `shopcohort:premium|cohort:flagship`
   - 计算：mean, median, std, dispersion, count

### ✅ 时间序列指标（部分支持）

以下时间序列指标**继续正常工作**（基于四类组合的历史值）：

- **SMA/EMA/WMA**: 基于 FeatureSnapshot 的历史 mean 值
  - 示例：`scope="shop:1|iphone:10"`, `name="sma"`, `version="sma_15"`

- **Bollinger Bands**: 基于 FeatureSnapshot 的历史 mean 值
  - 示例：`scope="shop:1|iphone:10"`, `name="boll_mid/boll_up/boll_low/boll_width"`

---

## ⚠️ 不再支持的功能

### ❌ 基于 OverallBar 的时间序列指标

如果你配置了以下类型的时间序列指标，**将无法计算**：

- Scope: `overall:iphone:10`
- Scope: `cohort:flagship`

**原因**: OverallBar 和 CohortBar 不再更新，无法提供历史基值。

**解决方案**: 如需此类指标，请改用四类组合的 scope：
- ❌ `overall:iphone:10` → ✅ `shopcohort:all_shops|iphone:10`（定义一个包含所有店铺的 ShopWeightProfile）
- ❌ `cohort:flagship` → ✅ `shop:1|cohort:flagship`（选择特定店铺）

---

## 📊 性能影响

### 性能提升

| 维度 | 影响 |
|------|------|
| **数据库写入** | 减少 OverallBar/CohortBar 表的写入操作 |
| **计算开销** | 减少两次聚合计算（每个 iPhone 和每个 Cohort） |
| **查询压力** | 减少对 OverallBar/CohortBar 表的读取 |

### 存储节省

- OverallBar 表不再增长
- CohortBar 表不再增长

**估算**：假设 10 个 iPhone × 15 分钟间隔 × 24 小时 = 每天减少约 960 条 OverallBar 记录

---

## 🔄 如何恢复

如果需要恢复 OverallBar 和 CohortBar 的计算：

### 步骤 1: 恢复聚合调用

在 `timestamp_alignment_task.py` 的 `_run_aggregation` 函数中：

```python
# 取消注释第1704-1715行
_agg_overallbar(
    ts_iso=ts_iso,
    ts_dt=ts_dt,
    rows=rows,
    use_window=use_window,
    bucket_start=bucket_start,
    bucket_end=bucket_end,
    is_final_bar=is_final_bar,
    agg_ctx=agg_ctx,
    ob_has_iphone=ob_has_iphone,
)

# 取消注释第1717-1724行
_agg_cohortbar(
    ts_iso=ts_iso,
    ob_bucket=ob_bucket,
    is_final_bar=is_final_bar,
    agg_ctx=agg_ctx,
    ob_has_iphone=ob_has_iphone,
)
```

### 步骤 2: 恢复时间序列基值收集

在 `timestamp_alignment_task.py` 的 `_agg_time_series_features` 函数中：

```python
# 取消注释第1082-1100行
# 4.b OverallBar.mean -> overall:iphone:<id>
if ob_has_iphone:
    for row in OverallBar.objects.filter(bucket=ob_bucket).values("iphone_id", "mean"):
        if row["mean"] is not None:
            base_now[f"overall:iphone:{row['iphone_id']}"] = float(row["mean"])

# 4.c CohortBar.mean -> cohort:<slug>
for row in CohortBar.objects.filter(bucket=ob_bucket).select_related("cohort").values("cohort__slug", "mean"):
    if row["mean"] is not None and row["cohort__slug"]:
        base_now[f"cohort:{row['cohort__slug']}"] = float(row["mean"])
```

---

## 🧹 清理旧数据（可选）

如果不再需要历史 OverallBar 和 CohortBar 数据：

### 清理 OverallBar 数据

```bash
# 查看统计
python manage.py shell -c "from AppleStockChecker.models import OverallBar; print(f'OverallBar 记录数: {OverallBar.objects.count()}')"

# 清空数据（谨慎！）
python manage.py shell -c "from AppleStockChecker.models import OverallBar; OverallBar.objects.all().delete()"
```

### 清理 CohortBar 数据

```bash
# 查看统计
python manage.py shell -c "from AppleStockChecker.models import CohortBar; print(f'CohortBar 记录数: {CohortBar.objects.count()}')"

# 清空数据（谨慎！）
python manage.py shell -c "from AppleStockChecker.models import CohortBar; CohortBar.objects.all().delete()"
```

### 使用清理脚本

可以参考 `scripts/clear_feature_snapshots.py` 创建类似的清理脚本。

---

## 📝 验证修改

### 验证 FeatureSnapshot 正常工作

```bash
# 1. 触发聚合任务
curl -X POST http://localhost:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/ \
  -d '{"agg_minutes": 15, "agg_mode": "boundary"}'

# 2. 检查 FeatureSnapshot 是否有新数据
python manage.py shell -c "
from AppleStockChecker.models import FeatureSnapshot
from django.utils import timezone
recent = FeatureSnapshot.objects.filter(
    bucket__gte=timezone.now() - timezone.timedelta(hours=1)
).count()
print(f'最近1小时的 FeatureSnapshot 记录: {recent}')
"

# 3. 检查四类组合的数据
python manage.py shell -c "
from AppleStockChecker.models import FeatureSnapshot
scopes = FeatureSnapshot.objects.values('scope').distinct()[:10]
for s in scopes:
    print(s['scope'])
"
```

### 验证 OverallBar/CohortBar 已停止更新

```bash
# 检查 OverallBar 最近更新时间
python manage.py shell -c "
from AppleStockChecker.models import OverallBar
latest = OverallBar.objects.order_by('-updated_at').first()
if latest:
    print(f'OverallBar 最后更新: {latest.updated_at}')
else:
    print('OverallBar 表为空')
"

# 检查 CohortBar 最近更新时间
python manage.py shell -c "
from AppleStockChecker.models import CohortBar
latest = CohortBar.objects.order_by('-updated_at').first()
if latest:
    print(f'CohortBar 最后更新: {latest.updated_at}')
else:
    print('CohortBar 表为空')
"
```

---

## 🔗 相关文档

- [timestamp_alignment_task.py](/home/user/YamagotiProjects/AppleStockChecker/tasks/timestamp_alignment_task.py) - 核心聚合逻辑
- [models.py](/home/user/YamagotiProjects/AppleStockChecker/models.py) - 数据模型定义
- [api.py](/home/user/YamagotiProjects/AppleStockChecker/api.py) - 触发聚合的 API

---

## 📌 关键要点总结

1. ✅ **FeatureSnapshot 四类组合不受影响**，继续正常工作
2. ✅ **基于四类组合的时间序列指标不受影响**（SMA/EMA/WMA/Bollinger）
3. ❌ **OverallBar 和 CohortBar 不再更新**
4. ❌ **基于 OverallBar/CohortBar 的时间序列指标将失效**（需要改用四类组合）
5. ⚡ **性能提升**：减少数据库写入和计算开销
6. 🔄 **可轻松恢复**：只需取消相关代码的注释

---

## 版本历史

- **v1.0** (2025-12-12): 初始版本，禁用 OverallBar 和 CohortBar 计算
