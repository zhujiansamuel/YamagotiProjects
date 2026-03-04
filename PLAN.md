# GPU Engine 对齐修改计划

基于 22 个差异点的用户决策，以下是按文件组织的具体修改方案。

---

## Phase 1: engine/config.py — 配置基础设施

### 1.1 新增 FeatureSpec 读取工具函数 (B4, B5, C4)

**当前状态**: 特征窗口硬编码为 `FEATURE_WINDOWS = [30, 60, 75, 120, 900, 1800]`

**修改方案**: 新增 `load_feature_specs()` 函数，从 FeatureSpec DB 读取 active 的配置，按 family 分组返回。保留硬编码作为 fallback（DB 为空时）。

```python
# 新增到 config.py
def load_feature_specs() -> dict:
    """从 FeatureSpec 读取 active 配置，按 family 分组。
    返回 {'ema': [...], 'sma': [...], 'wma_linear': [...], 'boll': [...]}
    每项包含 slug, params, version 等字段。
    DB 为空时返回 None，调用方回退到硬编码。
    """
```

**问题 Q1**: FeatureSpec 目前有哪些 active 记录？我需要确认 DB 中的实际数据来决定 load_feature_specs 的解析逻辑是否需要处理特殊情况。能否运行 `python manage.py shell -c "from AppleStockChecker.models import FeatureSpec; print(list(FeatureSpec.objects.filter(active=True).values()))"` 提供结果？

---

## Phase 2: engine/aggregate.py — 跨店聚合

### 2.1 MAD 异常值过滤 (A1)

**当前状态**: `aggregate_cross_shop()` 无过滤，直接 nanmean

**修改方案**: 在 `aggregate_cross_shop()` 的 nanmean/nanmedian 之前，加入 MAD 过滤：
1. 对每个 (iphone, bucket) 的 shop 维度：
   - median_price = nanmedian(shop 维度)
   - MAD = nanmedian(|price - median_price|)
   - threshold = median_price ± 3 × 1.4826 × MAD（标准化 MAD）
   - 超出 threshold 的值置为 NaN
2. 过滤后再做聚合

实现方式: 新增 `_mad_filter_dim1(data: Tensor, k=3.0) -> Tensor` 辅助函数。

### 2.2 标准差 ddof=1 (A2)

**当前状态**: `_nanstd_dim1()` 已经使用 ddof=1 (`count - 1`)

**修改**: ✅ 无需修改，已满足。

### 2.3 离散度 = 变异系数 (A3)

**当前状态**: `dispersion = std / mean`

**修改**: ✅ 无需修改，已满足。

### 2.4 标准中位数（偶数取平均）(A4)

**当前状态**: `_nanmedian_dim1()` 使用 `valid.median()` — torch 偶数个元素时取偏小值

**修改方案**: 修改 `_nanmedian_dim1()`:
```python
if valid.numel() > 0:
    sorted_v = valid.sort().values
    n = sorted_v.numel()
    if n % 2 == 1:
        result[i, b] = sorted_v[n // 2]
    else:
        result[i, b] = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0
```

### 2.5 动态价格区间过滤 (A5)

**当前状态**: GPU engine 无此逻辑

**修改方案**: 在 `build_price_tensor()` 之后、`aggregate_cross_shop()` 之前，新增 `apply_dynamic_price_filter(tensor: PriceTensor) -> PriceTensor`:
1. 对每个 iphone_id 的每个 bucket：
   - 回看前 2 个桶（30 分钟），收集同 iphone 所有 shop 的有效价格
   - 计算 reference_price = mean
   - 如果样本数 < 3，使用 [100000, 350000] 固定范围
   - 否则范围 = [ref × 0.9, ref × 1.1]
   - 超出范围的值置为 NaN

实现放在 `aggregate.py` 中。

**问题 Q2**: 这个动态价格过滤是在 align 阶段做（即丢弃整行）还是在 tensor 阶段做（置 NaN）？GPU 批量模式下建议在 tensor 阶段置 NaN，与 A1 的 MAD 过滤形成两层防护。请确认。

---

## Phase 3: engine/features.py — 特征计算

### 3.1 SMA/WMA 缩窗模式 (B2, B3)

**当前状态**: `compute_sma_batch()` 和 `compute_wma_batch()` 使用 `F.pad(..., mode='replicate')`

**修改方案**: 改为不足窗口时缩窗计算：

SMA: 使用 cumsum 实现渐进窗口
```python
def compute_sma_batch(series, window):
    # cumsum 方式: 前 window-1 个点用 1..window-1 的平均
    cumsum = series.cumsum(dim=1)
    sma = torch.zeros_like(series)
    for t in range(series.shape[1]):
        w = min(t + 1, window)
        if t - w >= 0:
            sma[:, t] = (cumsum[:, t] - cumsum[:, t - w]) / w
        else:
            sma[:, t] = cumsum[:, t] / w
    return sma
```

WMA: 类似，前端点用实际可用长度的线性权重。

### 3.2 从 FeatureSpec 读取配置 (B4, B5)

**修改方案**: 修改 `compute_all_features()` 和 `compute_all_features_skipnan()` 的签名：
```python
def compute_all_features(
    agg_mean: torch.Tensor,
    windows: list[int] | None = None,
    feature_specs: list | None = None,  # 新参数
) -> dict[str, torch.Tensor]:
```

当 `feature_specs` 不为 None 时，从 specs 解析窗口/参数；否则 fallback 到硬编码。

### 3.3 数值精度截断 2 位小数 (B6)

**修改方案**: 在 `pipeline.py` 的 `_agg_to_features_df()` 中，写入 DataFrame 前对每个 value 做 `round(v, 2)`。不在 tensor 计算阶段截断（保持计算精度），仅在输出阶段截断。

### 3.4 Bollinger 支持 center_mode (C1)

**修改方案**: 扩展 `compute_bollinger_batch()`:
```python
def compute_bollinger_batch(
    series: torch.Tensor,
    window: int,
    k: float = 2.0,
    center_mode: str = "sma",   # 新参数: "sma" | "ema"
    center_window: int | None = None,  # None 表示同 window
) -> BollingerResult:
```

当 `center_mode == "ema"` 时，mid 使用 EMA 而非 SMA。

### 3.5 Bollinger rolling std + ddof=1 (C2, C3)

**当前状态**: `unfolded.std(dim=-1)` — 已经是 rolling std，torch 默认 ddof=1

**修改**: ✅ 无需修改，已满足。

---

## Phase 4: engine/pipeline.py — 主流程

### 4.1 集成 MAD 过滤 + 动态价格过滤 (A1, A5)

在 `aggregate` 步骤中：
```python
if "aggregate" in effective_steps:
    tensor = build_price_tensor(full_aligned, device=device)
    tensor = apply_dynamic_price_filter(tensor)   # A5: 新增
    agg = aggregate_cross_shop(tensor, min_quorum=config.min_quorum)
    # MAD 过滤已集成在 aggregate_cross_shop 内部
```

### 4.2 FeatureSpec 集成 (B4, B5, C1, C4)

在 `features` 步骤开头读取 FeatureSpec，传入 compute_all_features:
```python
from AppleStockChecker.engine.config import load_feature_specs
specs = load_feature_specs()
feat_fn = compute_all_features_skipnan if skipnan else compute_all_features
iphone_features = feat_fn(agg.mean, feature_specs=specs)
```

### 4.3 输出精度截断 (B6)

在 `_agg_to_features_df()` 中对所有数值列截断至 2 位小数。同样修改 `_per_shop_features_df()` 和 `_per_profile_features_df()`。

### 4.4 新增 Case 3: shop × cohort (D2)

**当前缺失**: GPU engine 无 `shop:{sid}|cohort:{slug}` scope

**修改方案**: 在 pipeline.py features 步骤中新增 `_per_shop_cohort_features_df()`:
- 输入: tensor (3D), cohort configs
- 对每个 shop × 每个 cohort:
  - 取该店的各成员 iPhone 价格，按 cohort 的 model_weight 加权
  - 计算 weighted mean, 加权 std (D4)
  - 在 weighted_mean 上计算全部特征
  - scope = `shop:{sid}|cohort:{slug}`

### 4.5 新增 Case 4: shopcohort × cohort (D3)

**修改方案**: 新增 `_per_profile_cohort_features_df()`:
- 输入: tensor (3D), profiles, cohort configs
- 对每个 profile × 每个 cohort:
  - 取 profile 中的 shops，取 cohort 中的 member iPhones
  - 三重权重: shop_weight × model_weight (无 recency，D1 决策)
  - 加权聚合 → 特征计算
  - scope = `shopcohort:{prof_slug}|cohort:{coh_slug}`

### 4.6 Market Log Premium (E1)

**修改方案**: 在 pipeline.py features 步骤末尾新增 `_compute_market_log_premium()`:
```python
def _compute_market_log_premium(
    features_df: pd.DataFrame,
    official_prices: dict[int, float],
) -> pd.DataFrame:
    """对 shopcohort:full_store|iphone:* 的 WMA 行计算 log premium。
    logb = log(wma_value / official_price)
    """
```

- 从 settings.IPHONE_OFFICIAL_PRICES 获取官方价
- 从已计算的 features 中找 `shopcohort:full_store|iphone:*` scope 的 WMA 列
- 计算 `logb = math.log(wma / official)`
- 生成新的 DataFrame 行，name="logb"

**问题 Q3**: Market Log Premium 的输出格式需要确认：是作为额外的 feature 列追加到同一行，还是作为独立的 scope/name 行写入 ClickHouse？从 Celery 代码看是独立行（name="logb", version=spec_slug），请确认 GPU 端是否也这样。

---

## Phase 5: 跨文件一致性

### 5.1 D4: Profile 加权 std

**当前状态**: `_per_profile_features_df()` 已使用加权 std `sqrt(Σ(w×(x-μ)²)/Σ(w))`

**修改**: ✅ 无需修改，已满足。

### 5.2 D5: Profile median = mean

**当前状态**: `_per_profile_features_df()` 已设 `"median": mean_val`

**修改**: ✅ 无需修改，已满足。

### 5.3 B1: 保留双模式

**当前状态**: compute_all_features (ffill) 和 compute_all_features_skipnan 已存在

**修改**: ✅ 无需修改，已满足。

---

## 修改文件清单

| 文件 | 修改类型 | 涉及差异点 |
|------|---------|-----------|
| `engine/config.py` | 新增 `load_feature_specs()` | B4, B5, C4 |
| `engine/aggregate.py` | 新增 MAD 过滤；修改 median；新增动态价格过滤 | A1, A4, A5 |
| `engine/features.py` | 改 SMA/WMA 为缩窗；支持 FeatureSpec 参数；Bollinger 支持 center_mode | B2, B3, B4, C1 |
| `engine/pipeline.py` | 集成过滤；FeatureSpec 读取；精度截断；新增 D2/D3 scope；Market Log Premium | A1, A5, B5, B6, C4, D2, D3, E1 |

## 无需修改项确认

| 差异点 | 原因 |
|--------|------|
| A2 (ddof=1) | ✅ 已满足 |
| A3 (CV) | ✅ 已满足 |
| B1 (双模式) | ✅ 已满足 |
| C2 (rolling std) | ✅ 已满足 |
| C3 (ddof=1) | ✅ 已满足 |
| D1 (无时效权重) | ✅ 用户决策不加 |
| D4 (加权 std) | ✅ 已满足 |
| D5 (median=mean) | ✅ 已满足 |
| D6 (无时效权重) | ✅ 用户决策不加 |
| E2, E3, E4 | ✅ 无需处理 |
| F1, F2 | ✅ 无需处理 |

---

## 待确认问题

1. **Q1**: FeatureSpec DB 中现有 active 记录有哪些？
2. **Q2**: 动态价格过滤放在 tensor 阶段（置 NaN）是否 OK？
3. **Q3**: Market Log Premium 输出格式 — 独立行还是额外列？
