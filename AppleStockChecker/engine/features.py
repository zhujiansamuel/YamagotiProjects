"""
PyTorch 向量化特征计算: EMA / SMA / WMA / Bollinger Bands。
全部 iPhone 在维度 0 并行, 时间在维度 1。
参考: docs/REFACTOR_PLAN_V1.md §6.3, §20 Phase 2 Step 2.2
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .config import FEATURE_WINDOWS, WINDOW_TO_BUCKETS

logger = logging.getLogger(__name__)


# ── forward-fill 工具函数 ─────────────────────────────────────────────

def _forward_fill_1d(series: torch.Tensor) -> torch.Tensor:
    """沿 dim=1 (时间轴) forward-fill NaN。

    前导 NaN 保持不变, 中间/尾部 NaN 用最近有效值填充。
    返回新 Tensor (不 in-place 修改)。

    Parameters
    ----------
    series : (n_iphones, n_buckets)

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    out = series.clone()
    mask = torch.isnan(out)
    for t in range(1, out.shape[1]):
        fill = mask[:, t]  # 当前列为 NaN 的行
        out[:, t] = torch.where(fill, out[:, t - 1], out[:, t])
    return out


# EMA half-life 窗口 (分钟) → 桶数 (÷15min)
EMA_HL_WINDOWS: list[int] = [30, 60]
EMA_HL_BUCKETS: dict[int, int] = {w: w // 15 for w in EMA_HL_WINDOWS}


@dataclass
class BollingerResult:
    mid: torch.Tensor       # (I, B)
    upper: torch.Tensor     # (I, B)
    lower: torch.Tensor     # (I, B)
    width: torch.Tensor     # (I, B)


# ── EMA ──────────────────────────────────────────────────────────────────

def compute_ema_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """指数移动平均。时间维串行, iPhone 维并行。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int  窗口 (桶数)

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    alpha = 2.0 / (window + 1.0)
    ema = torch.zeros_like(series)
    ema[:, 0] = series[:, 0]
    for t in range(1, series.shape[1]):
        ema[:, t] = alpha * series[:, t] + (1 - alpha) * ema[:, t - 1]
    return ema


def compute_ema_halflife_batch(series: torch.Tensor, hl_buckets: int) -> torch.Tensor:
    """半衰期 EMA。alpha = 1 - exp(-ln2 / hl_buckets)。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    hl_buckets : int  半衰期 (桶数)

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    import math as _math
    alpha = 1.0 - _math.exp(-_math.log(2) / hl_buckets)
    ema = torch.zeros_like(series)
    ema[:, 0] = series[:, 0]
    for t in range(1, series.shape[1]):
        ema[:, t] = alpha * series[:, t] + (1 - alpha) * ema[:, t - 1]
    return ema


# ── skip-nan EMA ─────────────────────────────────────────────────────

def compute_ema_batch_skipnan(series: torch.Tensor, window: int) -> torch.Tensor:
    """skip-nan 版 EMA: NaN 桶输出 NaN, 有效值恢复时重新初始化。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    alpha = 2.0 / (window + 1.0)
    n_rows, n_cols = series.shape
    ema = torch.full_like(series, float("nan"))
    ema[:, 0] = series[:, 0]  # 首列: NaN 保留, 有效值保留

    for t in range(1, n_cols):
        cur = series[:, t]
        prev = ema[:, t - 1]
        cur_valid = ~torch.isnan(cur)
        prev_valid = ~torch.isnan(prev)

        # 正常递推: cur 有效 且 prev 有效
        normal = cur_valid & prev_valid
        # 重新初始化: cur 有效 但 prev 为 NaN
        reinit = cur_valid & ~prev_valid

        ema[:, t] = torch.where(
            normal,
            alpha * cur + (1 - alpha) * prev,
            torch.where(reinit, cur, torch.full_like(cur, float("nan"))),
        )

    return ema


def compute_ema_halflife_batch_skipnan(series: torch.Tensor, hl_buckets: int) -> torch.Tensor:
    """skip-nan 版半衰期 EMA。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    hl_buckets : int

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    import math as _math
    alpha = 1.0 - _math.exp(-_math.log(2) / hl_buckets)
    n_rows, n_cols = series.shape
    ema = torch.full_like(series, float("nan"))
    ema[:, 0] = series[:, 0]

    for t in range(1, n_cols):
        cur = series[:, t]
        prev = ema[:, t - 1]
        cur_valid = ~torch.isnan(cur)
        prev_valid = ~torch.isnan(prev)

        normal = cur_valid & prev_valid
        reinit = cur_valid & ~prev_valid

        ema[:, t] = torch.where(
            normal,
            alpha * cur + (1 - alpha) * prev,
            torch.where(reinit, cur, torch.full_like(cur, float("nan"))),
        )

    return ema


# ── SMA ──────────────────────────────────────────────────────────────────

def compute_sma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """简单移动平均, cumsum 向量化实现, 缩窗: 不足窗口时用实际可用长度。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    n = series.shape[1]
    cumsum = series.cumsum(dim=1)
    # 在 dim=1 前面补一列 0, 方便做差
    padded = torch.cat([torch.zeros(series.shape[0], 1, dtype=series.dtype,
                                     device=series.device), cumsum], dim=1)
    # padded shape: (I, n+1), padded[:,t+1] = cumsum[:,t]

    sma = torch.zeros_like(series)
    w = min(window, n)

    # ── 缩窗部分: t = 0 .. min(window, n)-1, 分母 = t+1 ──
    # sma[:, t] = cumsum[:, t] / (t+1)
    divisors = torch.arange(1, w + 1, dtype=series.dtype, device=series.device)
    sma[:, :w] = cumsum[:, :w] / divisors.unsqueeze(0)

    # ── 满窗部分: t = window .. n-1, 分母 = window ──
    if n > window:
        # cumsum[:, t] - cumsum[:, t-window] = padded[:,t+1] - padded[:,t+1-window]
        sma[:, window:] = (padded[:, window + 1:] - padded[:, 1:n - window + 1]) / window

    return sma


# ── WMA ──────────────────────────────────────────────────────────────────

def compute_wma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """线性加权移动平均, conv1d 实现, 缩窗: 不足窗口时用实际可用长度的线性权重。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    n = series.shape[1]
    wma = torch.zeros_like(series)

    # ── 满窗部分: t >= window-1, 用 conv1d 一次完成 ──
    if n >= window:
        weights = torch.arange(1, window + 1, dtype=series.dtype, device=series.device)
        w_sum = weights.sum()
        # conv1d: output[i] = sum(input[i+k] * kernel[k]), k=0..W-1
        # 要让最新值(segment尾部)权重最大: kernel = [1, 2, ..., W]
        kernel = weights.reshape(1, 1, window)
        inp = series.unsqueeze(1)  # (I, 1, B)
        conv_out = torch.nn.functional.conv1d(inp, kernel, padding=0).squeeze(1)  # (I, B-W+1)
        wma[:, window - 1:] = conv_out / w_sum

    # ── 缩窗部分: t = 0 .. min(window-1, n-1), 权重 = [1..t+1] ──
    for t in range(min(window - 1, n)):
        w = t + 1
        weights_t = torch.arange(1, w + 1, dtype=series.dtype, device=series.device)
        wma[:, t] = (series[:, :w] * weights_t).sum(dim=1) / weights_t.sum()

    return wma


# ── Bollinger Bands ──────────────────────────────────────────────────────

def compute_bollinger_batch(
    series: torch.Tensor,
    window: int,
    k: float = 2.0,
) -> BollingerResult:
    """布林带: mid (SMA) ± k * rolling_std。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int
    k : float  标准差倍数 (默认 2)

    Returns
    -------
    BollingerResult
    """
    mid = compute_sma_batch(series, window)

    # rolling std via unfold
    n_buckets = series.shape[1]
    if n_buckets >= window:
        unfolded = series.unfold(dimension=1, size=window, step=1)  # (I, B-W+1, W)
        std = unfolded.std(dim=-1)  # (I, B-W+1)
        # 前 window-1 个点用 NaN 填充
        pad_val = torch.full(
            (series.shape[0], window - 1),
            float("nan"), dtype=series.dtype, device=series.device,
        )
        std = torch.cat([pad_val, std], dim=1)  # (I, B)
    else:
        std = torch.full_like(series, float("nan"))

    upper = mid + k * std
    lower = mid - k * std
    width = 2 * k * std

    return BollingerResult(mid=mid, upper=upper, lower=lower, width=width)


# ── 全部特征一次计算 ─────────────────────────────────────────────────────

def compute_all_features(
    agg_mean: torch.Tensor,
    windows: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """对每个窗口计算全部特征, 返回 {列名: Tensor}。

    Parameters
    ----------
    agg_mean : (n_iphones, n_buckets)
        跨店聚合后的 mean 序列
    windows : list[int]
        特征窗口 (分钟), 默认 FEATURE_WINDOWS

    Returns
    -------
    dict  keys 如 'ema_120', 'boll_up_900', ...
    """
    if windows is None:
        windows = FEATURE_WINDOWS

    # forward-fill NaN before computing features
    filled = _forward_fill_1d(agg_mean)

    features: dict[str, torch.Tensor] = {}

    for win_min in windows:
        win_buckets = WINDOW_TO_BUCKETS[win_min]

        features[f"ema_{win_min}"] = compute_ema_batch(filled, win_buckets)
        features[f"sma_{win_min}"] = compute_sma_batch(filled, win_buckets)
        features[f"wma_{win_min}"] = compute_wma_batch(filled, win_buckets)

        boll = compute_bollinger_batch(filled, win_buckets)
        features[f"boll_mid_{win_min}"] = boll.mid
        features[f"boll_up_{win_min}"] = boll.upper
        features[f"boll_low_{win_min}"] = boll.lower
        features[f"boll_width_{win_min}"] = boll.width

    # EMA half-life 系列
    for hl_min in EMA_HL_WINDOWS:
        hl_buckets = EMA_HL_BUCKETS[hl_min]
        features[f"ema_hl_{hl_min}"] = compute_ema_halflife_batch(filled, hl_buckets)

    logger.info("compute_all_features: %d features computed for %d windows",
                len(features), len(windows))
    return features


def compute_all_features_skipnan(
    agg_mean: torch.Tensor,
    windows: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """skip-nan 版全特征计算。

    EMA 系列用 skipnan 版, SMA/WMA/Bollinger 直接复用现有函数
    (窗口含 NaN → 输出 NaN, 已满足语义)。

    Parameters
    ----------
    agg_mean : (n_iphones, n_buckets)
    windows : list[int]

    Returns
    -------
    dict  keys 如 'ema_120', 'boll_up_900', ...
    """
    if windows is None:
        windows = FEATURE_WINDOWS

    features: dict[str, torch.Tensor] = {}

    for win_min in windows:
        win_buckets = WINDOW_TO_BUCKETS[win_min]

        features[f"ema_{win_min}"] = compute_ema_batch_skipnan(agg_mean, win_buckets)
        features[f"sma_{win_min}"] = compute_sma_batch(agg_mean, win_buckets)
        features[f"wma_{win_min}"] = compute_wma_batch(agg_mean, win_buckets)

        boll = compute_bollinger_batch(agg_mean, win_buckets)
        features[f"boll_mid_{win_min}"] = boll.mid
        features[f"boll_up_{win_min}"] = boll.upper
        features[f"boll_low_{win_min}"] = boll.lower
        features[f"boll_width_{win_min}"] = boll.width

    # EMA half-life 系列
    for hl_min in EMA_HL_WINDOWS:
        hl_buckets = EMA_HL_BUCKETS[hl_min]
        features[f"ema_hl_{hl_min}"] = compute_ema_halflife_batch_skipnan(agg_mean, hl_buckets)

    logger.info("compute_all_features_skipnan: %d features computed for %d windows",
                len(features), len(windows))
    return features
