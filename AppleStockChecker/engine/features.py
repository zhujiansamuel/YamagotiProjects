"""
PyTorch 向量化特征计算: EMA / SMA / WMA / Bollinger Bands。
全部 iPhone 在维度 0 并行, 时间在维度 1。
参考: docs/REFACTOR_PLAN_V1.md §6.3, §20 Phase 2 Step 2.2
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import FEATURE_WINDOWS, WINDOW_TO_BUCKETS

logger = logging.getLogger(__name__)


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


# ── SMA ──────────────────────────────────────────────────────────────────

def compute_sma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """简单移动平均, 全向量化 (F.conv1d)。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    kernel = torch.ones(1, 1, window, dtype=series.dtype, device=series.device) / window
    padded = F.pad(series.unsqueeze(1), (window - 1, 0), mode="replicate")
    return F.conv1d(padded, kernel).squeeze(1)


# ── WMA ──────────────────────────────────────────────────────────────────

def compute_wma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """线性加权移动平均, 全向量化 (F.conv1d)。

    权重: [1, 2, ..., window], 归一化后做卷积。

    Parameters
    ----------
    series : (n_iphones, n_buckets)
    window : int

    Returns
    -------
    Tensor  (n_iphones, n_buckets)
    """
    weights = torch.arange(1, window + 1, dtype=series.dtype, device=series.device)
    kernel = (weights / weights.sum()).flip(0).reshape(1, 1, window)
    padded = F.pad(series.unsqueeze(1), (window - 1, 0), mode="replicate")
    return F.conv1d(padded, kernel).squeeze(1)


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

    features: dict[str, torch.Tensor] = {}

    for win_min in windows:
        win_buckets = WINDOW_TO_BUCKETS[win_min]

        features[f"ema_{win_min}"] = compute_ema_batch(agg_mean, win_buckets)
        features[f"sma_{win_min}"] = compute_sma_batch(agg_mean, win_buckets)
        features[f"wma_{win_min}"] = compute_wma_batch(agg_mean, win_buckets)

        boll = compute_bollinger_batch(agg_mean, win_buckets)
        features[f"boll_mid_{win_min}"] = boll.mid
        features[f"boll_up_{win_min}"] = boll.upper
        features[f"boll_low_{win_min}"] = boll.lower
        features[f"boll_width_{win_min}"] = boll.width

    logger.info("compute_all_features: %d features computed for %d windows",
                len(features), len(windows))
    return features
