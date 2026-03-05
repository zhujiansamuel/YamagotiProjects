# GPU Engine ↔ Celery Task 双向对齐修改计划（v4 — 最终版）

## 核心决策汇总

| 决策 | 结论 |
|------|------|
| 统计指标配置 | 两侧硬编码 `FEATURE_WINDOWS = [30, 60, 75, 120, 900, 1800]`，不读 FeatureSpec |
| Bollinger 中线 | 统一 SMA（Celery 删除 center_mode/EMA 分支） |
| Bollinger std | 统一 rolling std (ddof=1)，只取最近 W 个点 |
| logb 输出 | CH features_wide 列: `logb_30` ... `logb_1800` |
| Celery → CH | 新增写入路径，run_id = `"live"` |
| _fetch_prev_base | 转向 CH 读取（用户承诺回填） |
| PG FeatureSnapshot | **废弃写入**（唯一消费者 `_fetch_prev_base` 已转 CH；API 已从 CH 读） |
| recency_weight | Celery 删除，与 GPU 对齐（纯静态权重） |
| EMA half-life | Celery 新增 `ema_hl_30`, `ema_hl_60` |
| overall:iphone:* / cohort:* 时序 | 继续跳过 |

---

## Phase 1: CH Schema — clickhouse/init.sql

### 1.1 新增 logb 列

CREATE TABLE 定义中追加：
```sql
-- Market Log Premium (6 窗口)
logb_30              Nullable(Float64),
logb_60              Nullable(Float64),
logb_75              Nullable(Float64),
logb_120             Nullable(Float64),
logb_900             Nullable(Float64),
logb_1800            Nullable(Float64),
```

ALTER TABLE 迁移语句：
```sql
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_30 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_60 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_75 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_120 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_900 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_1800 Nullable(Float64);
```

---

## Phase 2: engine/aggregate.py — GPU 跨店聚合

### 2.1 MAD 异常值过滤 (A1)

新增 `_mad_filter_dim1(data, k=3.0)`:
- 沿 dim=1 (shop 维度)，对每个 (iphone, bucket)：
  - nanmedian → MAD = nanmedian(|x - median|) → threshold = k × 1.4826 × MAD
  - 超出 threshold 的值置 NaN
- 在 `aggregate_cross_shop()` 开头调用

### 2.2 标准中位数 (A4)

`_nanmedian_dim1()` 改为偶数取两中间值平均：
```python
sorted_v = valid.sort().values
n = sorted_v.numel()
if n % 2 == 1:
    result[i, b] = sorted_v[n // 2]
else:
    result[i, b] = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
```

### 2.3 动态价格区间过滤 (A5)

新增 `apply_dynamic_price_filter(tensor, lookback_buckets=2, tolerance=0.10, min_samples=3, fallback_range=(100_000, 350_000))`:
- 对每个 (iphone, bucket)：取前 N 桶参考价 → 超出 ref ± tolerance 的值置 NaN
- 在 pipeline.py 中 `aggregate_cross_shop()` 之前调用

---

## Phase 3: engine/features.py — GPU 特征计算

### 3.1 SMA 缩窗 (B2)

`compute_sma_batch()` 改为 cumsum 实现：
```python
cumsum = series.cumsum(dim=1)
for t in range(n):
    w = min(t + 1, window)
    sma[:, t] = (cumsum[:, t] - cumsum[:, t - w]) / w  # t >= w
    sma[:, t] = cumsum[:, t] / (t + 1)                  # t < w
```

### 3.2 WMA 缩窗 (B3)

`compute_wma_batch()` 改为逐步缩窗线性权重：
```python
for t in range(n):
    w = min(t + 1, window)
    segment = series[:, t - w + 1 : t + 1]
    weights = arange(1, w + 1)
    wma[:, t] = (segment * weights).sum(dim=1) / weights.sum()
```

---

## Phase 4: engine/pipeline.py — GPU 主流程

### 4.1 集成过滤 (A1, A5)

```python
tensor = build_price_tensor(...)
tensor = apply_dynamic_price_filter(tensor)   # A5
agg = aggregate_cross_shop(tensor, ...)       # A1 MAD 在内部
```

### 4.2 输出精度 round(v, 2) (B6)

`_agg_to_features_df()`, `_per_shop_features_df()`, `_per_profile_features_df()` 对所有数值 round 2 位。

### 4.3 新增 shop × cohort scope (D2)

新增 `_per_shop_cohort_features_df()`:
- scope = `shop:{sid}|cohort:{slug}`
- 按 model_weight 加权 mean → 加权 std → 特征计算

### 4.4 新增 shopcohort × cohort scope (D3)

新增 `_per_profile_cohort_features_df()`:
- scope = `shopcohort:{prof}|cohort:{slug}`
- 双重权重: shop_weight × model_weight（无 recency）

### 4.5 logb 列 (E1)

新增 `_compute_market_log_premium()`:
- 从 `shopcohort:full_store|iphone:*` 行的 `wma_{W}` 列
- `logb_{W} = log(wma / official_price)`
- 追加到同一行的 DataFrame 中

---

## Phase 5: tasks/timestamp_alignment_task.py — Celery 侧全面对齐

这是改动最大的文件。按功能分块：

### 5.1 删除 recency_weight (D1/D6 对齐)

**删除**: `recency_weight()` 函数 + `AGE_CAP_MIN`, `RECENCY_HALF_LIFE_MIN`, `RECENCY_DECAY` 配置

**修改 Cases 2-4**: 去掉时效衰减，仅用静态权重：
- Case 2: `w = shop_weight`（不乘 recency）
- Case 3: `w = model_weight`
- Case 4: `w = shop_weight × model_weight`

### 5.2 MAD 过滤 (A1)

删除 `_filter_outliers_by_mean_band()`，新增 `_filter_outliers_by_mad(vals, k=3.0)`:
```python
def _filter_outliers_by_mad(vals, k=3.0):
    med = 标准 median（偶数取平均）
    MAD = median(|v - med|)
    threshold = k × 1.4826 × MAD
    if threshold == 0: return list(vals)
    filtered = [v for v in vals if |v - med| <= threshold]
    return filtered if filtered else list(vals)
```

### 5.3 样本标准差 ddof=1 (A2)

重命名 `_pop_std()` → `_sample_std()`: 除以 `n-1`。全文替换调用点。

### 5.4 离散度 = 变异系数 (A3)

`_stats()` 中改为 `disp_v = std_v / mean_v if mean_v else 0.0`。
删除 `_quantile()` 函数。

### 5.5 EMA/SMA/WMA 硬编码窗口 + 删除 FeatureSpec 读取

`_agg_time_series_features()` 重写：

1. 删除 `FeatureSpec.objects.filter(...)` 查询
2. 硬编码:
```python
FEATURE_WINDOWS = [30, 60, 75, 120, 900, 1800]   # 分钟
BUCKET_MIN = 15
EMA_HL_WINDOWS = [30, 60]

for W in FEATURE_WINDOWS:
    W_buckets = W // BUCKET_MIN
    for scope, x_t in base_now.items():
        prev = _fetch_prev_base(scope, "mean", W_buckets - 1, anchor_bucket)
        series = list(reversed(prev)) + [float(x_t)]
        # EMA: alpha = 2/(W_buckets+1)
        # SMA: window = W_buckets
        # WMA: window = W_buckets, 线性权重
```
3. **新增 EMA half-life**:
```python
for W in EMA_HL_WINDOWS:
    W_buckets = W // BUCKET_MIN
    alpha = 1.0 - 0.5 ** (1.0 / W_buckets)
    # 计算 ema_hl_{W}
```

### 5.6 _fetch_prev_base limit 单位修正 + 转向 CH

**删除** 旧 PG 实现，改为从 CH features_wide 读取：

```python
def _fetch_prev_base(scope: str, column: str, limit: int, anchor_dt):
    """从 CH features_wide 读取历史基值序列（新→旧），limit 为桶数。
    返回 [float | None, ...] 保留等间距（None 表示该桶无数据）。
    """
    ch = ClickHouseService()
    rows = ch.client.execute(
        "SELECT %(col)s FROM features_wide FINAL "
        "WHERE run_id = 'live' AND scope = %(scope)s AND bucket < %(dt)s "
        "ORDER BY bucket DESC LIMIT %(lim)s",
        {"col": column, "scope": scope, "dt": anchor_dt, "lim": limit},
    )
    return [float(r[0]) if r[0] is not None else None for r in rows]
```

**关键变化**:
- 参数 `base_name` + `base_version` → `column`（直接对应 CH 列名，如 `"mean"`）
- limit 现在是**桶数**（不再是分钟数）
- 返回值**保留 None**（等间距序列），不再过滤
- 使用 `FINAL` 关键字保证去重

### 5.7 EMA/SMA/WMA 处理 None 值

series 中可能含 None（某桶无数据），各函数的 None 处理策略：

- **EMA**: None 时沿用前一个 ema 值（等效 GPU 的 ffill）:
```python
def _ema_from_series(series, alpha):
    ema = series[0] if series[0] is not None else 0.0
    for v in series[1:]:
        if v is None:
            continue  # ema 不更新 = ffill
        ema = alpha * float(v) + (1 - alpha) * ema
    return ema
```
- **SMA**: window 中只取非 None 值计算均值
- **WMA**: window 中只取非 None 值，权重按位置分配

### 5.8 Bollinger 硬编码 + SMA + rolling std (C1, C2, C3)

`_agg_bollinger_bands()` 重写：

1. 删除 FeatureSpec 查询 + `_parse_center_mode()`
2. 硬编码:
```python
for W in FEATURE_WINDOWS:
    W_buckets = W // BUCKET_MIN
    k = 2.0
    for scope, x_t in base_now.items():
        prev = _fetch_prev_base(scope, "mean", W_buckets - 1, anchor_bucket)
        series = list(reversed(prev)) + [float(x_t)]
        non_none = [v for v in series if v is not None]
        if len(non_none) < 2:
            continue
        mid = sum(non_none[-W_buckets:]) / len(non_none[-W_buckets:])  # SMA
        window_vals = non_none[-W_buckets:]
        std = _sample_std(window_vals)  # rolling std, ddof=1
        up = mid + k * std
        low = mid - k * std
        width = up - low
```

### 5.9 logb 硬编码 (E1)

`_agg_market_log_premium()` 重写：

1. 删除从 PG FeatureSnapshot 查 WMA 记录的逻辑
2. 直接使用 5.5 中刚计算的 WMA 值（通过参数传入或 wide_row 累积器）
3. 按 FEATURE_WINDOWS 计算: `logb_{W} = log(wma_{W} / official_price)`

### 5.10 Celery 写入 CH features_wide（新增核心）

**架构**: 在各计算函数中用累积器收集 wide-format row，最后批量写入 CH。

```python
# 在 _run_aggregation() 中：
wide_rows = {}  # key: (bucket, scope) → value: {col: val}

# _agg_feature_combos 中:
for scope in [shop:*|iphone:*, shopcohort:*|iphone:*, shop:*|cohort:*, shopcohort:*|cohort:*]:
    wide_rows[(bucket, scope)] = {
        "mean": mean_w, "median": med, "std": st,
        "shop_count": n, "dispersion": disp,
    }

# _agg_time_series_features 中:
for scope, W in product(scopes, FEATURE_WINDOWS):
    wide_rows[(bucket, scope)][f"ema_{W}"] = ema_val
    wide_rows[(bucket, scope)][f"sma_{W}"] = sma_val
    wide_rows[(bucket, scope)][f"wma_{W}"] = wma_val
for scope, W in product(scopes, EMA_HL_WINDOWS):
    wide_rows[(bucket, scope)][f"ema_hl_{W}"] = ema_hl_val

# _agg_bollinger_bands 中:
for scope, W in product(scopes, FEATURE_WINDOWS):
    wide_rows[(bucket, scope)][f"boll_mid_{W}"] = mid
    wide_rows[(bucket, scope)][f"boll_up_{W}"] = up
    wide_rows[(bucket, scope)][f"boll_low_{W}"] = low
    wide_rows[(bucket, scope)][f"boll_width_{W}"] = width

# _agg_market_log_premium 中:
for scope, W in product(logb_scopes, FEATURE_WINDOWS):
    wide_rows[(bucket, scope)][f"logb_{W}"] = logb_val

# 最终: 批量写入 CH
df = pd.DataFrame([{"bucket": k[0], "scope": k[1], **v} for k, v in wide_rows.items()])
ch = ClickHouseService()
ch.insert_features(df, run_id="live")
```

### 5.11 废弃 PG FeatureSnapshot 写入

- 删除 `FeatureWriter` 使用（`writer.write()`, `writer.write_many()`）
- 删除 `safe_upsert_feature_snapshot()` 调用
- 保留 FeatureWriter 类本身（api.py）不动——其他地方可能引用，但 Celery 不再调用
- `_run_aggregation()` 中的 writer 参数可改为传入 wide_rows 累积器

### 5.12 前端 logb 映射 (views.py)

FeaturePointsViewSet._FE_NAME_MAP 中追加 logb 映射:
```python
"logb30m": "logb_30", "logb60m": "logb_60", "logb75m": "logb_75",
"logb120m": "logb_120", "logb900m": "logb_900", "logb1800m": "logb_1800",
```

---

## 修改文件清单（最终版）

### GPU 侧

| 文件 | 修改内容 | 差异点 |
|------|---------|--------|
| `engine/aggregate.py` | `_mad_filter_dim1()`; `_nanmedian_dim1()` 偶数; `apply_dynamic_price_filter()` | A1, A4, A5 |
| `engine/features.py` | SMA cumsum 缩窗; WMA 缩窗 | B2, B3 |
| `engine/pipeline.py` | A5 过滤; round(v,2); D2/D3 新 scope; logb 列 | A5, B6, D2, D3, E1 |

### Celery 侧

| 文件 | 修改内容 | 差异点 |
|------|---------|--------|
| `tasks/timestamp_alignment_task.py` | 删除 recency_weight; MAD 过滤; ddof=1; CV; EMA/SMA/WMA/Bollinger 硬编码; EMA half-life 新增; _fetch_prev_base → CH + limit 修正 + 保留 None; logb 硬编码; CH 写入; 废弃 PG 写入 | D1/D6, A1-A3, B2-B5, C1-C4, E1, 新需求 |

### 共用 / 其他

| 文件 | 修改内容 |
|------|---------|
| `clickhouse/init.sql` | 新增 logb_30..1800 列 |
| `views.py` | FeaturePointsViewSet._FE_NAME_MAP 追加 logb 映射 |

---

## 无需修改确认

| 差异点 | 状态 |
|--------|------|
| A2/A3/A4 GPU | ✅ 已满足 |
| B1 GPU | ✅ 双模式 ffill + skipnan 已有 |
| C1/C2/C3 GPU | ✅ Bollinger 已是 SMA + rolling std (ddof=1) |
| D4 GPU | ✅ 加权 std 已满足 |
| D5 GPU | ✅ median = mean 已满足 |
| E2-E4, F1-F2 | ✅ 无需处理 |
| `services/clickhouse_service.py` | ✅ insert_features 已是动态列，无需修改 |
| `features/api.py` | ✅ FeatureWriter 类保留不动 |

---

## 实施顺序

1. **Phase 1**: clickhouse/init.sql — logb 列 DDL
2. **Phase 2**: engine/aggregate.py — MAD, median, 动态过滤
3. **Phase 3**: engine/features.py — SMA/WMA 缩窗
4. **Phase 4**: engine/pipeline.py — A5 集成, B6 精度, D2/D3 scope, E1 logb
5. **Phase 5**: tasks/timestamp_alignment_task.py — Celery 全面对齐
6. **Phase 5.12**: views.py — logb 前端映射
7. **测试**: 关键函数单元测试

---

## 无待讨论问题

所有决策已确认：
- ✅ recency_weight: 删除
- ✅ EMA half-life: Celery 新增
- ✅ _fetch_prev_base: 转 CH + limit 修正 (桶数)
- ✅ PG FeatureSnapshot: 废弃写入（API 已从 CH 读）
- ✅ overall/cohort 时序: 跳过
- ✅ run_id: "live"
