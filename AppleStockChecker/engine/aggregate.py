"""
跨店聚合: price_aligned DataFrame → 3D Tensor → 跨 shop 维度统计。
参考: docs/REFACTOR_PLAN_V1.md §6.2 Step 4-5, §20 Phase 2 Step 2.1
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class PriceTensor:
    """3D 价格张量及其索引映射。"""
    data: torch.Tensor          # (n_iphones, n_shops, n_buckets), NaN = 缺失
    iphone_ids: np.ndarray      # (n_iphones,) int
    shop_ids: np.ndarray        # (n_shops,) int
    bucket_index: pd.DatetimeIndex  # (n_buckets,)


@dataclass
class AggResult:
    """跨店聚合结果, 每个字段 shape (n_iphones, n_buckets)。"""
    mean: torch.Tensor
    median: torch.Tensor
    std: torch.Tensor
    shop_count: torch.Tensor    # int tensor
    dispersion: torch.Tensor    # std / mean (变异系数)
    iphone_ids: np.ndarray
    bucket_index: pd.DatetimeIndex


def build_price_tensor(
    aligned_df: pd.DataFrame,
    *,
    device: str = "cpu",
) -> PriceTensor:
    """将对齐后的 DataFrame 构建为 3D Tensor。

    Parameters
    ----------
    aligned_df : DataFrame
        必须包含: bucket, shop_id, iphone_id, price_new
    device : str
        PyTorch 设备

    Returns
    -------
    PriceTensor
    """
    df = aligned_df.copy()

    iphone_ids = np.sort(df["iphone_id"].unique())
    shop_ids = np.sort(df["shop_id"].unique())
    bucket_index = pd.DatetimeIndex(sorted(df["bucket"].unique()))

    iphone_map = {v: i for i, v in enumerate(iphone_ids)}
    shop_map = {v: i for i, v in enumerate(shop_ids)}
    bucket_map = {v: i for i, v in enumerate(bucket_index)}

    n_i, n_s, n_b = len(iphone_ids), len(shop_ids), len(bucket_index)

    data = torch.full((n_i, n_s, n_b), float("nan"), dtype=torch.float64, device=device)

    i_idx = df["iphone_id"].map(iphone_map).values
    s_idx = df["shop_id"].map(shop_map).values
    b_idx = df["bucket"].map(bucket_map).values
    prices = df["price_new"].values.astype(np.float64)

    data[i_idx, s_idx, b_idx] = torch.tensor(prices, dtype=torch.float64, device=device)

    logger.info(
        "build_price_tensor: shape=(%d iphones, %d shops, %d buckets)  device=%s",
        n_i, n_s, n_b, device,
    )
    return PriceTensor(data=data, iphone_ids=iphone_ids, shop_ids=shop_ids,
                       bucket_index=bucket_index)


def aggregate_cross_shop(
    tensor: PriceTensor,
    *,
    min_quorum: int = 16,
) -> AggResult:
    """对 shop 维度 (dim=1) 做 nanmean/nanmedian/nanstd 聚合。

    Parameters
    ----------
    tensor : PriceTensor
        shape (n_iphones, n_shops, n_buckets)
    min_quorum : int
        记录用，不强制跳过

    Returns
    -------
    AggResult  每个字段 shape (n_iphones, n_buckets)
    """
    data = tensor.data  # (I, S, B)

    # 非 NaN 计数
    valid_mask = ~torch.isnan(data)
    shop_count = valid_mask.sum(dim=1)  # (I, B)

    # nanmean
    mean = torch.nanmean(data, dim=1)  # (I, B)

    # nanmedian: torch.nanmedian 只返回单值，需要手动处理
    median = _nanmedian_dim1(data)  # (I, B)

    # nanstd (无偏)
    std = _nanstd_dim1(data, mean)  # (I, B)

    # 变异系数 dispersion = std / mean
    dispersion = std / mean
    dispersion = torch.where(torch.isnan(dispersion), torch.zeros_like(dispersion), dispersion)

    below_quorum = (shop_count < min_quorum).sum().item()
    if below_quorum > 0:
        logger.info(
            "aggregate_cross_shop: %d (iphone, bucket) pairs below quorum=%d (policy=use_anyway)",
            below_quorum, min_quorum,
        )

    logger.info(
        "aggregate_cross_shop: shape=(%d, %d)  shop_count range=[%d, %d]",
        mean.shape[0], mean.shape[1],
        shop_count.min().item(), shop_count.max().item(),
    )

    return AggResult(
        mean=mean, median=median, std=std,
        shop_count=shop_count, dispersion=dispersion,
        iphone_ids=tensor.iphone_ids,
        bucket_index=tensor.bucket_index,
    )


# ── 辅助函数 ─────────────────────────────────────────────────────────────

def _nanmedian_dim1(data: torch.Tensor) -> torch.Tensor:
    """沿 dim=1 计算 nanmedian, 返回 (I, B)。"""
    I, S, B = data.shape
    result = torch.full((I, B), float("nan"), dtype=data.dtype, device=data.device)

    for i in range(I):
        for b in range(B):
            vals = data[i, :, b]
            valid = vals[~torch.isnan(vals)]
            if valid.numel() > 0:
                result[i, b] = valid.median()
    return result


def _nanstd_dim1(data: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """沿 dim=1 计算无偏 nanstd, 返回 (I, B)。"""
    # mean shape: (I, B), data shape: (I, S, B)
    diff_sq = (data - mean.unsqueeze(1)) ** 2  # (I, S, B)
    valid_mask = ~torch.isnan(data)
    diff_sq = torch.where(valid_mask, diff_sq, torch.zeros_like(diff_sq))
    count = valid_mask.sum(dim=1).float()  # (I, B)
    # 无偏: ddof=1
    var = diff_sq.sum(dim=1) / (count - 1).clamp(min=1)
    return torch.sqrt(var)
