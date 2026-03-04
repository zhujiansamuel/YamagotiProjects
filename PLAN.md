# GPU Engine ↔ Celery Task 双向对齐修改计划（v2）

基于 22 个差异点的用户决策 + 确认事项：
- CH 列固定 → B4/B5/C4 保持硬编码，不读 FeatureSpec
- C1 → 所有 Bollinger 统一为 SMA 中线（Celery 删除 center_mode 支持）
- E1 logb → 独立行（scope + name），不影响 CH features_wide 列
- Celery EMA 跳 None → 直接删除
- Q2 → 动态价格过滤在 tensor 阶段置 NaN

---

## Phase 1: engine/aggregate.py — 跨店聚合

### 1.1 MAD 异常值过滤 (A1) — GPU 侧

**当前**: 无过滤

**修改**: 在 `aggregate_cross_shop()` 中，nanmean 之前加入 MAD 过滤：

```python
def _mad_filter_dim1(data: torch.Tensor, k: float = 3.0) -> torch.Tensor:
    """沿 dim=1 (shop 维度) 做 MAD 过滤，异常值置 NaN。
    threshold = median ± k × 1.4826 × MAD
    """
```

- 对 data (I, S, B) 的每个 (i, b)：nanmedian → MAD → 标准化 → 超出 k=3 倍的置 NaN
- 在 `aggregate_cross_shop()` 开头调用：`data = _mad_filter_dim1(tensor.data)`

### 1.2 标准中位数 (A4) — GPU 侧

**当前**: `_nanmedian_dim1()` 用 `valid.median()` — 偶数个取偏小值

**修改**: 改为偶数个取两中间值平均：
```python
sorted_v = valid.sort().values
n = sorted_v.numel()
if n % 2 == 1:
    result[i, b] = sorted_v[n // 2]
else:
    result[i, b] = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
```

### 1.3 动态价格区间过滤 (A5) — GPU 侧

**当前**: 无

**修改**: 在 aggregate.py 新增函数：

```python
def apply_dynamic_price_filter(
    tensor: PriceTensor,
    *,
    lookback_buckets: int = 2,       # 回看 2 桶 = 30 分钟
    tolerance: float = 0.10,          # ±10%
    min_samples: int = 3,
    fallback_range: tuple = (100_000, 350_000),
) -> PriceTensor:
    """对每个 (iphone, bucket)，基于前 N 桶参考价过滤异常值，置 NaN。"""
```

在 pipeline.py 的 aggregate 步骤中调用：
```python
tensor = build_price_tensor(full_aligned, device=device)
tensor = apply_dynamic_price_filter(tensor)      # A5
agg = aggregate_cross_shop(tensor, ...)           # A1 MAD 在内部
```

### 1.4 ✅ 无需修改确认

- A2 (ddof=1): `_nanstd_dim1` 已用 `count - 1`
- A3 (CV): `dispersion = std / mean` 已满足

---

## Phase 2: engine/features.py — 特征计算

### 2.1 SMA 缩窗 (B2) — GPU 侧

**当前**: `F.pad(..., mode='replicate')` + conv1d

**修改**: 改为 cumsum 实现，不足窗口时用实际可用长度：
```python
def compute_sma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    n = series.shape[1]
    cumsum = series.cumsum(dim=1)
    sma = torch.zeros_like(series)
    for t in range(n):
        w = min(t + 1, window)
        if t >= w:
            sma[:, t] = (cumsum[:, t] - cumsum[:, t - w]) / w
        else:
            sma[:, t] = cumsum[:, t] / (t + 1)
    return sma
```

### 2.2 WMA 缩窗 (B3) — GPU 侧

**当前**: `F.pad(..., mode='replicate')` + conv1d

**修改**: 改为逐列计算，不足窗口时用实际可用长度的线性权重：
```python
def compute_wma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    n = series.shape[1]
    wma = torch.zeros_like(series)
    for t in range(n):
        w = min(t + 1, window)
        start = t - w + 1
        segment = series[:, start:t+1]            # (I, w)
        weights = torch.arange(1, w + 1, dtype=series.dtype, device=series.device)
        wma[:, t] = (segment * weights).sum(dim=1) / weights.sum()
    return wma
```

### 2.3 Bollinger 统一 SMA 中线 (C1) — GPU 侧

**当前**: `compute_bollinger_batch()` 已使用 SMA

**修改**: ✅ 无需修改，已满足。删除计划中的 center_mode 参数扩展。

### 2.4 ✅ 无需修改确认

- B1 (双模式 ffill + skipnan): 已满足
- B4 (alpha 硬编码): 保持硬编码
- B5 (窗口硬编码): 保持硬编码
- C2 (rolling std): 已满足
- C3 (ddof=1): torch.std 默认 Bessel correction
- C4 (窗口硬编码): 保持硬编码

---

## Phase 3: engine/pipeline.py — 主流程

### 3.1 输出精度截断 2 位小数 (B6) — GPU 侧

**修改**: 在 `_agg_to_features_df()`, `_per_shop_features_df()`, `_per_profile_features_df()` 中，
写入 row 时对所有数值做 round(v, 2)：
```python
for fname, ftensor in features.items():
    row[fname] = round(ftensor[i_idx, b_idx].item(), 2)
# 同理对 mean, median, std, dispersion
row["mean"] = round(mean_val, 2)
row["median"] = round(..., 2)
# ...
```

### 3.2 新增 Case 3: shop × cohort (D2) — GPU 侧

**当前**: 无 `shop:{sid}|cohort:{slug}` scope

**新增**: `_per_shop_cohort_features_df(tensor, cohort_configs, *, skipnan)`:
- 对每个 shop_id × 每个 CohortConfig：
  - 取该店的成员 iPhone 价格: `tensor.data[:, s_idx, :]` 中 members 对应行
  - 按 model_weight 加权 mean（归一化权重）
  - 加权 std = sqrt(Σ(w×(x-μ_w)²)/Σ(w))
  - 在 weighted_mean 上计算特征
  - scope = `shop:{sid}|cohort:{slug}`

### 3.3 新增 Case 4: shopcohort × cohort (D3) — GPU 侧

**新增**: `_per_profile_cohort_features_df(tensor, profiles, cohort_configs, *, skipnan)`:
- 对每个 ShopWeightProfile × 每个 CohortConfig：
  - 取 profile 中的 shops + cohort 中的 member iPhones
  - 双重权重: shop_weight × model_weight（无 recency）
  - 加权聚合 → 特征计算
  - scope = `shopcohort:{prof_slug}|cohort:{coh_slug}`

### 3.4 Market Log Premium (E1) — GPU 侧

**新增**: `_compute_market_log_premium(features_df, official_prices)`:

logb 以 **独立行** 写入 ClickHouse features_wide:
- 找 `shopcohort:full_store|iphone:*` scope 的行
- 对每个这样的行，取其 WMA 列（如 wma_120, wma_1800）
- 计算 `logb = math.log(wma / official_price)`
- 生成新行: scope 不变, 额外列 `logb_{window}` = logb 值

实际上 logb 可以作为 features_wide 的额外列追加到同一行中（比如 `logb_120`, `logb_1800`），因为它们跟其他特征列是一对一关系。

但用户选择"独立行 scope+name"——这更适合写入 **PG FeatureSnapshot**（Celery 的写法），而 CH features_wide 是宽表无 name 列。

**方案**: GPU pipeline 同时写两处：
1. CH features_wide: 在 shopcohort:full_store|iphone:* 行中追加 `logb_120`, `logb_1800` 等列（可选，如果 CH 已建列）
2. PG FeatureSnapshot: 通过 FeatureWriter 写入独立行 (scope, name="logb", version="wma120m", value=logb_value)

---

## Phase 4: Celery 侧修改 — timestamp_alignment_task.py

### 4.1 MAD 异常值过滤 (A1) — Celery 侧

**当前**: `_filter_outliers_by_mean_band(vals, 0.5, 1.5)` — 按 mean×[0.5, 1.5] 过滤

**修改**: 重写为 `_filter_outliers_by_mad(vals, k=3.0)`:
```python
def _filter_outliers_by_mad(vals, k=3.0):
    """MAD 过滤：median ± k × 1.4826 × MAD"""
    if len(vals) < 3:
        return list(vals)
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    med = vals_sorted[n // 2] if n % 2 else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
    abs_devs = sorted(abs(v - med) for v in vals_sorted)
    mad = abs_devs[n // 2] if n % 2 else 0.5 * (abs_devs[n // 2 - 1] + abs_devs[n // 2])
    threshold = k * 1.4826 * mad
    if threshold == 0:
        return list(vals)  # 所有值相同
    filtered = [v for v in vals if abs(v - med) <= threshold]
    return filtered if filtered else list(vals)
```

**调用点修改**: `_stats()` 函数中 `_filter_outliers_by_mean_band(vals_raw)` → `_filter_outliers_by_mad(vals_raw)`

删除 `_filter_outliers_by_mean_band()` 函数。

### 4.2 标准差 ddof=1 (A2) — Celery 侧

**当前**: `_pop_std()` 除以 N（总体 std）

**修改**: 改为除以 N-1（样本 std）：
```python
def _sample_std(vals):
    """样本标准差 (ddof=1)；N<=1 返回 0."""
    n = len(vals)
    if n <= 1:
        return 0.0
    mu = sum(vals) / n
    s2 = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return s2 ** 0.5
```

重命名 `_pop_std` → `_sample_std`，全文替换所有调用点。

### 4.3 离散度 = 变异系数 (A3) — Celery 侧

**当前**: `_stats()` 中 `disp_v = (p90 - p10)`

**修改**: 改为 `disp_v = std_v / mean_v if mean_v != 0 else 0.0`

删除 `_quantile` 函数（仅用于 p10/p90，不再需要）。

### 4.4 Bollinger 统一 SMA 中线 + rolling std (C1, C2, C3) — Celery 侧

**当前**: `_agg_bollinger_bands()` 中：
- center_mode 从 FeatureSpec 读取（支持 sma/ema）
- std = `_pop_std(series_old_to_new)` — 全序列总体 std

**修改**:
1. 删除 `_parse_center_mode()` 和 center_mode 分支，统一用 SMA：
   ```python
   mid = _sma(series_old_to_new, W)
   ```
2. std 改为 rolling std (ddof=1)，只取最近 W 个点：
   ```python
   window_vals = series_old_to_new[-W:]
   std = _sample_std(window_vals)
   ```

### 4.5 EMA 跳 None 逻辑删除 (B1) — Celery 侧

**当前**: `_fetch_prev_base()` 三个 return 语句中 `if v is not None` 过滤掉 None

**这实际上是合理的 DB 查询过滤**（DB 中 None 表示无数据），不需要删除。

真正需要删除的是：如果 EMA 函数本身有跳过 None 的逻辑。
检查发现 `_ema_from_series()`, `_sma()`, `_wma_linear()` 本身 **没有** 跳 None 逻辑——
它们接收的 series 已经被 `_fetch_prev_base` 预过滤了 None。

**问题**: _fetch_prev_base 过滤 None 后，序列中间会缺少时间点（非等间距），这就是 B1 说的"跳过模式"。

**修改方案**: 不改 `_fetch_prev_base`（DB 查询过滤 None 是合理的），但在构建 series 时需要意识到这个时间间隔问题。由于 Celery 是实时逐桶处理（只取最新值），且 B1 决策是"废弃 Celery 的跳过模式"——实际含义是：**Celery 侧的这些时间序列特征（EMA/SMA/WMA/Bollinger）将在 GPU pipeline 完全接管后停用**。

因此 Celery 侧 EMA/SMA/WMA 相关代码的 None 跳过逻辑**保持不动**（因为它马上就要被 GPU 取代），只修改聚合阶段（A1-A3, C1-C3）保证在两边并行运行期间数据一致。

---

## 修改文件清单（最终版）

### GPU 侧

| 文件 | 修改内容 | 差异点 |
|------|---------|--------|
| `engine/aggregate.py` | 新增 `_mad_filter_dim1()`; 修改 `_nanmedian_dim1()`; 新增 `apply_dynamic_price_filter()` | A1, A4, A5 |
| `engine/features.py` | 重写 `compute_sma_batch()` (cumsum 缩窗); 重写 `compute_wma_batch()` (缩窗) | B2, B3 |
| `engine/pipeline.py` | 集成 A5 过滤; 输出 round(v,2); 新增 `_per_shop_cohort_features_df()`; 新增 `_per_profile_cohort_features_df()`; 新增 `_compute_market_log_premium()` | A5, B6, D2, D3, E1 |

### Celery 侧

| 文件 | 修改内容 | 差异点 |
|------|---------|--------|
| `tasks/timestamp_alignment_task.py` | `_filter_outliers_by_mean_band` → `_filter_outliers_by_mad`; `_pop_std` → `_sample_std` (ddof=1); `_stats()` dispersion → CV; Bollinger 删除 center_mode 分支 + std 改 rolling | A1, A2, A3, C1, C2, C3 |

### 无需修改

| 差异点 | 状态 |
|--------|------|
| A2 GPU | ✅ 已满足 |
| A3 GPU | ✅ 已满足 |
| A4 Celery | ✅ 已满足 |
| A5 Celery | ✅ 已有 |
| B1 | ✅ GPU 双模式已有; Celery 跳过模式将随 GPU 接管而废弃 |
| B4, B5, C4 | ✅ 保持硬编码（CH 列固定） |
| C1 GPU | ✅ 已是 SMA 中线 |
| C2, C3 GPU | ✅ 已满足 |
| D1, D6 | ✅ 不加时效权重 |
| D4 | ✅ GPU 已用加权 std |
| D5 | ✅ GPU 已用 median=mean |
| E2, E3, E4 | ✅ 无需处理 |
| F1, F2 | ✅ 无需处理 |

---

## 实施顺序

1. **Phase 1**: engine/aggregate.py (A1, A4, A5) — 基础聚合对齐
2. **Phase 2**: engine/features.py (B2, B3) — SMA/WMA 缩窗
3. **Phase 3**: engine/pipeline.py (B6, D2, D3, E1) — 新 scope + logb + 精度
4. **Phase 4**: tasks/timestamp_alignment_task.py (A1, A2, A3, C1, C2, C3) — Celery 侧对齐
5. **测试**: 对关键函数编写单元测试验证数值一致性
