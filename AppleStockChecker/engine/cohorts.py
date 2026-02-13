"""
Cohort 加权聚合: 从 PG 读取 Cohort 配置, 对 iPhone 维度加权, 计算 cohort 级特征。
参考: docs/REFACTOR_PLAN_V1.md §6.2 Step 7, §20 Phase 2 Step 2.3
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .aggregate import AggResult
from .features import compute_all_features

logger = logging.getLogger(__name__)


@dataclass
class CohortConfig:
    """一个 Cohort 的完整配置。"""
    cohort_id: int
    slug: str
    members: list[dict] = field(default_factory=list)
    # members: [{'iphone_id': 42, 'weight': 1.0}, ...]
    shop_weights: dict[int, float] = field(default_factory=dict)
    # shop_weights: {shop_id: weight, ...}, 空 dict 表示等权


def load_cohort_configs() -> list[CohortConfig]:
    """从 PG 读取全部 Cohort + CohortMember + ShopWeightProfile。"""
    from AppleStockChecker.models import Cohort

    configs = []
    for cohort in Cohort.objects.prefetch_related(
        "members", "members__iphone",
        "shop_weight_profile", "shop_weight_profile__items",
    ).all():
        members = [
            {"iphone_id": m.iphone_id, "weight": m.weight}
            for m in cohort.members.all()
        ]

        shop_weights = {}
        if cohort.shop_weight_profile:
            for item in cohort.shop_weight_profile.items.all():
                shop_weights[item.shop_id] = item.weight

        configs.append(CohortConfig(
            cohort_id=cohort.id,
            slug=cohort.slug,
            members=members,
            shop_weights=shop_weights,
        ))

    logger.info("load_cohort_configs: %d cohorts loaded", len(configs))
    return configs


def compute_cohort_features(
    agg: AggResult,
    features: dict[str, torch.Tensor],
    configs: list[CohortConfig],
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """对每个 Cohort 做加权聚合, 返回 features_wide 格式的 DataFrame。

    Parameters
    ----------
    agg : AggResult
        iPhone 级跨店聚合结果
    features : dict[str, Tensor]
        iPhone 级特征 {name: (n_iphones, n_buckets)}
    configs : list[CohortConfig]
    device : str

    Returns
    -------
    DataFrame  columns: bucket, scope, mean, median, std, shop_count, dispersion, + 全部特征列
    """
    iphone_id_to_idx = {int(v): i for i, v in enumerate(agg.iphone_ids)}
    n_buckets = len(agg.bucket_index)
    all_rows = []

    for cfg in configs:
        # 找到成员 iPhone 在 agg 中的索引和权重
        indices = []
        weights = []
        for m in cfg.members:
            idx = iphone_id_to_idx.get(m["iphone_id"])
            if idx is not None:
                indices.append(idx)
                weights.append(m["weight"])

        if not indices:
            logger.warning("cohort %s: no members found in agg data, skipping", cfg.slug)
            continue

        w = torch.tensor(weights, dtype=torch.float64, device=device)
        w = w / w.sum()  # 归一化

        # 加权聚合基础指标
        member_mean = agg.mean[indices]  # (n_members, n_buckets)
        cohort_mean = (member_mean * w.unsqueeze(1)).sum(dim=0)  # (n_buckets,)

        member_median = agg.median[indices]
        cohort_median = (member_median * w.unsqueeze(1)).sum(dim=0)

        member_std = agg.std[indices]
        cohort_std = (member_std * w.unsqueeze(1)).sum(dim=0)

        member_count = agg.shop_count[indices].float()
        cohort_shop_count = member_count.sum(dim=0)  # 总覆盖店铺数

        cohort_dispersion = cohort_std / cohort_mean
        cohort_dispersion = torch.where(
            torch.isnan(cohort_dispersion),
            torch.zeros_like(cohort_dispersion),
            cohort_dispersion,
        )

        # 在 cohort_mean 上重新计算特征
        cohort_mean_2d = cohort_mean.unsqueeze(0)  # (1, n_buckets) for feature funcs
        cohort_features = compute_all_features(cohort_mean_2d)

        # 组装行
        scope = f"cohort:{cfg.slug}"
        for b_i in range(n_buckets):
            row = {
                "bucket": agg.bucket_index[b_i],
                "scope": scope,
                "mean": cohort_mean[b_i].item(),
                "median": cohort_median[b_i].item(),
                "std": cohort_std[b_i].item(),
                "shop_count": int(cohort_shop_count[b_i].item()),
                "dispersion": cohort_dispersion[b_i].item(),
            }
            for fname, ftensor in cohort_features.items():
                v = ftensor[0, b_i].item()
                row[fname] = v
            all_rows.append(row)

    if not all_rows:
        logger.warning("compute_cohort_features: no rows produced")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info("compute_cohort_features: %d cohorts → %d rows", len(configs), len(df))
    return df
