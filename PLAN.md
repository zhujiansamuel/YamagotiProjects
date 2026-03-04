# GPU Engine ↔ Celery Task 双向对齐修改计划（v3）

核心决策：
- **Celery 是主要实时路径**，GPU 是批量回填/实验路径
- **Celery 也写入 CH features_wide**（新增）
- 两侧统计指标硬编码，窗口/列完全一致: `[30, 60, 75, 120, 900, 1800]`
- 所有 Bollinger 统一 SMA 中线
- logb 作为 features_wide 的额外列 (`logb_30` ... `logb_1800`)
- Celery EMA 跳 None 逻辑直接删除（见 Phase 4.5）

---

## Phase 1: engine/aggregate.py — GPU 跨店聚合

### 1.1 MAD 异常值过滤 (A1)

新增 `_mad_filter_dim1(data, k=3.0)`: 沿 shop 维度 MAD 过滤，异常值置 NaN。
在 `aggregate_cross_shop()` 开头调用。

### 1.2 标准中位数 (A4)

`_nanmedian_dim1()` 改为偶数取两中间值平均。

### 1.3 动态价格区间过滤 (A5)

新增 `apply_dynamic_price_filter(tensor, ...)`: 基于前 N 桶参考价 ± 10% 过滤。
在 pipeline.py aggregate 步骤中 `aggregate_cross_shop()` 之前调用。

---

## Phase 2: engine/features.py — GPU 特征计算

### 2.1 SMA 缩窗 (B2)

`compute_sma_batch()` 改为 cumsum 实现，不足窗口用实际可用长度。

### 2.2 WMA 缩窗 (B3)

`compute_wma_batch()` 改为逐步缩窗线性权重。

---

## Phase 3: engine/pipeline.py — GPU 主流程

### 3.1 输出精度 round(v, 2) (B6)

`_agg_to_features_df()`, `_per_shop_features_df()`, `_per_profile_features_df()` 对所有数值 round 2 位。

### 3.2 新增 shop × cohort scope (D2)

新增 `_per_shop_cohort_features_df()`: scope = `shop:{sid}|cohort:{slug}`

### 3.3 新增 shopcohort × cohort scope (D3)

新增 `_per_profile_cohort_features_df()`: scope = `shopcohort:{prof}|cohort:{slug}`

### 3.4 logb 列 (E1)

新增 `_compute_market_log_premium()`:
- 从 `shopcohort:full_store|iphone:*` 行的 `wma_{W}` 列计算 `logb_{W} = log(wma / official_price)`
- 写入同一行的 `logb_30` ... `logb_1800` 列

---

## Phase 4: tasks/timestamp_alignment_task.py — Celery 侧对齐

### 4.1 MAD 过滤 (A1)

删除 `_filter_outliers_by_mean_band()`，新增 `_filter_outliers_by_mad(vals, k=3.0)`:
```python
def _filter_outliers_by_mad(vals, k=3.0):
    med = 标准 median（偶数取平均）
    MAD = median(|v - med|)
    threshold = k × 1.4826 × MAD
    return [v for v in vals if |v - med| <= threshold]
```
`_stats()` 中调用替换。

### 4.2 样本标准差 ddof=1 (A2)

`_pop_std()` → `_sample_std()`: 除以 `n-1` 而非 `n`。全文替换调用点。

### 4.3 离散度 = 变异系数 (A3)

`_stats()` 中 `disp_v = std_v / mean_v`，删除 p10/p90 和 `_quantile` 函数。

### 4.4 Bollinger 统一 SMA + rolling std (C1, C2, C3)

`_agg_bollinger_bands()`:
1. 删除 `_parse_center_mode()` 和 EMA 分支，统一 `mid = _sma(series, W)`
2. std 改为只取最近 W 个点的样本 std: `std = _sample_std(series[-W:])`
3. **删除 FeatureSpec 读取**，改用硬编码窗口 `[30, 60, 75, 120, 900, 1800]`:
```python
FEATURE_WINDOWS = [30, 60, 75, 120, 900, 1800]
for W in FEATURE_WINDOWS:
    for scope, x_t in base_now.items():
        ...  # 计算 boll_mid, boll_up, boll_low, boll_width
```

### 4.5 EMA/SMA/WMA 硬编码窗口 + 删除 FeatureSpec 读取

`_agg_time_series_features()`:
1. **删除 FeatureSpec 查询**（当前从 DB 读 active specs）
2. 改为硬编码，与 GPU 完全一致:
```python
FEATURE_WINDOWS = [30, 60, 75, 120, 900, 1800]
EMA_HL_WINDOWS = [30, 60]

for W in FEATURE_WINDOWS:
    for scope, x_t in base_now.items():
        # EMA: alpha = 2/(W+1), name = f"ema_{W}"
        # SMA: name = f"sma_{W}"
        # WMA: name = f"wma_{W}"
for W in EMA_HL_WINDOWS:
    # EMA half-life: alpha = 1 - 0.5^(1/hl_buckets)
```
3. **EMA 跳 None 逻辑删除**: `_ema_from_series` 中没有跳 None 逻辑（series 已由 `_fetch_prev_base` 预过滤 None）。真正的变化是：`_fetch_prev_base` 返回的是**连续非 None 值序列**（中间跳过 None 桶），这导致时间不等间距。修改方案：

   - **`_fetch_prev_base` 改为保留 None 位置**（返回含 None 的等间距序列）
   - `_ema_from_series` 遇到 None 时直接沿用前一个 ema 值（不更新）
   - `_sma`, `_wma_linear` 的 window slice 中只取非 None 值计算

   这与 GPU 的 ffill 行为一致：遇到 NaN 时 forward-fill 使 EMA 状态不变。

### 4.6 logb 硬编码 (E1)

`_agg_market_log_premium()`:
- 删除从 FeatureSnapshot 查 WMA 记录的逻辑
- 改为直接使用 4.5 中刚计算的 WMA 值
- 生成 `logb_{W}` 列值

### 4.7 Celery 写入 CH features_wide（新增）

**当前**: Celery 只写 PG FeatureSnapshot（逐条 upsert via FeatureWriter）
**新增**: 每个 bucket 处理完后，汇总为一行 wide-format dict，写入 CH

实现方案:
1. 在 `_agg_feature_combos` + `_agg_time_series_features` + `_agg_bollinger_bands` + `_agg_market_log_premium` 执行完后，**收集本桶所有 (scope, name, value) 结果**
2. 按 scope 分组，转为 wide-format row:
   ```python
   # 例如 scope="shopcohort:full_store|iphone:42" 的一行:
   {
       "bucket": bucket,
       "scope": scope,
       "mean": ..., "median": ..., "std": ..., "shop_count": ..., "dispersion": ...,
       "ema_30": ..., "ema_60": ..., ..., "ema_1800": ...,
       "sma_30": ..., ..., "sma_1800": ...,
       "wma_30": ..., ..., "wma_1800": ...,
       "ema_hl_30": ..., "ema_hl_60": ...,
       "boll_mid_30": ..., "boll_up_30": ..., ..., "boll_width_1800": ...,
       "logb_30": ..., ..., "logb_1800": ...,
   }
   ```
3. 调用 `ClickHouseService.insert_features(df, run_id="live")`

**run_id 约定**: Celery 实时写入用 `"live"`（与 GPU 回填的 run_id 区分）

**PG FeatureSnapshot 保留还是废弃?**
→ 待讨论（见下方问题）

---

## Phase 5: CH Schema 变更 — init.sql

### 5.1 新增 logb 列

```sql
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_30 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_60 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_75 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_120 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_900 Nullable(Float64);
ALTER TABLE yamagoti.features_wide ADD COLUMN IF NOT EXISTS logb_1800 Nullable(Float64);
```

同步更新 CREATE TABLE 定义。

---

## 修改文件清单（最终版）

### GPU 侧

| 文件 | 修改内容 | 差异点 |
|------|---------|--------|
| `engine/aggregate.py` | `_mad_filter_dim1()`; `_nanmedian_dim1()` 偶数; `apply_dynamic_price_filter()` | A1, A4, A5 |
| `engine/features.py` | SMA cumsum 缩窗; WMA 缩窗 | B2, B3 |
| `engine/pipeline.py` | round(v,2); D2/D3 scope; logb 列 | B6, D2, D3, E1 |

### Celery 侧

| 文件 | 修改内容 | 差异点 |
|------|---------|--------|
| `tasks/timestamp_alignment_task.py` | MAD 过滤; ddof=1; CV; Bollinger 硬编码+SMA+rolling; EMA/SMA/WMA 硬编码窗口; 删除 FeatureSpec 读取; EMA 等间距处理; logb 硬编码; **CH 写入** | A1-A3, C1-C3, B4/B5/C4, E1, 新需求 |

### 共用

| 文件 | 修改内容 |
|------|---------|
| `clickhouse/init.sql` | 新增 logb_30..1800 列 |
| `services/clickhouse_service.py` | （可能无需修改，insert_features 已是动态列） |

---

## 无需修改确认

| 差异点 | 状态 |
|--------|------|
| A2/A3/A4 GPU | ✅ 已满足 |
| B1 GPU | ✅ 双模式已有 |
| B4/B5/C4 | ✅ 硬编码（两侧统一） |
| C1/C2/C3 GPU | ✅ 已满足 |
| D1/D6 | ✅ 不加时效权重 |
| D4/D5 | ✅ GPU 已满足 |
| E2-E4, F1-F2 | ✅ 无需处理 |

---

## 实施顺序

1. Phase 5: CH schema (logb 列) — 先建表
2. Phase 1: aggregate.py (A1, A4, A5)
3. Phase 2: features.py (B2, B3)
4. Phase 3: pipeline.py (B6, D2, D3, E1)
5. Phase 4: timestamp_alignment_task.py (全部 Celery 侧变更)
6. 测试: 关键函数单元测试

---

## 待讨论问题

**Q1**: Celery 写 CH 后，**PG FeatureSnapshot 是否保留?**
- 选项 A: 保留双写（PG + CH），PG 作为 Celery 侧时序特征的中间状态（`_fetch_prev_base` 从 PG 读历史）
- 选项 B: 废弃 PG 写入，`_fetch_prev_base` 改从 CH 读历史
- 建议选 A（保留双写）: PG FeatureSnapshot 是 `_fetch_prev_base` 的数据源，Bollinger/EMA/SMA/WMA 都依赖它读历史。改成从 CH 读会引入较大的架构变更。

**Q2**: Celery 的 `_fetch_prev_base` 读 OverallBar/CohortBar 的路径已被注释。对于 `overall:iphone:*` 和 `cohort:*` 这两类 scope 的时间序列特征，是否仍然跳过？

**Q3**: Celery run_id 用 `"live"` 是否 OK？GPU 回填通常用什么 run_id？
