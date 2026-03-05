"""
主 Pipeline: reader → align → aggregate → features → cohorts → CH 写入。
参考: docs/REFACTOR_PLAN_V1.md §6.2, §20 Phase 2
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .config import ALL_STEPS, BucketConfig, FEATURE_WINDOWS

logger = logging.getLogger(__name__)


def run(
    run_id: str,
    date_from: date,
    date_to: date,
    *,
    device: str = "cpu",
    steps: list[str] | None = None,
    batch_days: int = 30,
    iphone_ids: list[int] | None = None,
    shop_ids: list[int] | None = None,
    nan_mode: str = "ffill",
) -> dict:
    """执行 pipeline。

    Parameters
    ----------
    run_id : str
        写入 CH 的 run_id 标识
    date_from, date_to : date
        数据范围 [date_from, date_to)
    device : str
        PyTorch 设备, e.g. 'cuda:0' | 'cpu'
    steps : list[str] | None
        要执行的步骤, 默认全部。可选: align, aggregate, features, cohorts
    batch_days : int
        每次从 PG 读取的天数 (align 阶段分段读取)
    iphone_ids, shop_ids : list[int] | None
        限定范围

    Returns
    -------
    dict  执行统计
    """
    from AppleStockChecker.engine.reader import read_price_records
    from AppleStockChecker.engine.align import align_to_buckets
    from AppleStockChecker.engine.aggregate import build_price_tensor, aggregate_cross_shop, apply_dynamic_price_filter
    from AppleStockChecker.engine.features import compute_all_features, compute_all_features_skipnan
    from AppleStockChecker.engine.cohorts import load_cohort_configs, compute_cohort_features
    from AppleStockChecker.services.clickhouse_service import ClickHouseService

    effective_steps = steps or ALL_STEPS
    skipnan = nan_mode == "skipnan"
    config = BucketConfig()
    ch = ClickHouseService()

    stats = {
        "run_id": run_id,
        "date_from": str(date_from),
        "date_to": str(date_to),
        "device": device,
        "steps": effective_steps,
        "batches": [],
    }

    logger.info(
        "pipeline.run START  run_id=%s  range=%s→%s  steps=%s  device=%s  batch_days=%d",
        run_id, date_from, date_to, effective_steps, device, batch_days,
    )
    t0 = time.time()

    # ── Step: align ──────────────────────────────────────────────────────
    all_aligned = []

    if "align" in effective_steps:
        t_step = time.time()
        total_aligned = 0
        cursor = datetime.combine(date_from, datetime.min.time())
        end = datetime.combine(date_to, datetime.min.time())

        while cursor < end:
            batch_end = min(cursor + timedelta(days=batch_days), end)
            bt = time.time()

            df = read_price_records(
                cursor, batch_end,
                shop_ids=shop_ids,
                iphone_ids=iphone_ids,
            )

            if df.empty:
                logger.info("  batch %s→%s: 0 rows, skip", cursor.date(), batch_end.date())
                cursor = batch_end
                continue

            aligned = align_to_buckets(df, config)
            inserted = ch.insert_price_aligned(aligned, run_id)
            all_aligned.append(aligned)
            elapsed = time.time() - bt

            batch_stat = {
                "from": str(cursor.date()),
                "to": str(batch_end.date()),
                "raw_rows": len(df),
                "aligned_rows": len(aligned),
                "inserted": inserted,
                "seconds": round(elapsed, 2),
            }
            stats["batches"].append(batch_stat)
            total_aligned += inserted

            logger.info(
                "  batch %s→%s: %d raw → %d aligned → %d inserted (%.1fs)",
                cursor.date(), batch_end.date(),
                len(df), len(aligned), inserted, elapsed,
            )
            cursor = batch_end

        stats["total_aligned"] = total_aligned
        stats["align_seconds"] = round(time.time() - t_step, 2)

    # 后续步骤需要全部 aligned 数据
    if all_aligned:
        full_aligned = pd.concat(all_aligned, ignore_index=True)
    else:
        full_aligned = pd.DataFrame()

    # ── Step: aggregate ──────────────────────────────────────────────────
    agg = None
    tensor = None
    if "aggregate" in effective_steps:
        t_step = time.time()

        if full_aligned.empty:
            logger.warning("  [aggregate] no aligned data available, skipping")
        else:
            tensor = build_price_tensor(full_aligned, device=device)
            tensor = apply_dynamic_price_filter(tensor)  # A5: 动态价格过滤
            agg = aggregate_cross_shop(tensor, min_quorum=config.min_quorum)

            stats["aggregate"] = {
                "n_iphones": len(agg.iphone_ids),
                "n_buckets": len(agg.bucket_index),
                "shop_count_min": int(agg.shop_count.min().item()),
                "shop_count_max": int(agg.shop_count.max().item()),
            }
            logger.info("  [aggregate] done: %d iphones × %d buckets",
                        len(agg.iphone_ids), len(agg.bucket_index))

        stats["aggregate_seconds"] = round(time.time() - t_step, 2)

    # ── Step: features ───────────────────────────────────────────────────
    iphone_features: dict[str, object] = {}
    if "features" in effective_steps:
        t_step = time.time()

        if agg is None:
            logger.warning("  [features] no agg data, skipping")
        else:
            feat_fn = compute_all_features_skipnan if skipnan else compute_all_features
            iphone_features = feat_fn(agg.mean)

            # 写入 CH: 组装 iphone 级 features_wide DataFrame
            iphone_df = _agg_to_features_df(agg, iphone_features)
            inserted = ch.insert_features(iphone_df, run_id)

            stats["features"] = {
                "n_feature_cols": len(iphone_features),
                "rows_inserted": inserted,
            }
            logger.info("  [features] done: %d feature cols → %d rows inserted",
                        len(iphone_features), inserted)

            # per-shop features (scope = shop:{sid}|iphone:{iid})
            if tensor is not None:
                shop_df = _per_shop_features_df(tensor, skipnan=skipnan)
                if not shop_df.empty:
                    shop_ins = ch.insert_features(shop_df, run_id)
                    stats["features_per_shop"] = {"rows_inserted": shop_ins}
                    logger.info("  [features/per-shop] %d rows inserted", shop_ins)

                # per-profile features (scope = shopcohort:{slug}|iphone:{iid})
                from AppleStockChecker.engine.cohorts import load_shop_weight_profiles
                profiles = load_shop_weight_profiles()
                profile_df = pd.DataFrame()
                if profiles:
                    profile_df = _per_profile_features_df(tensor, profiles, skipnan=skipnan)
                    if not profile_df.empty:
                        prof_ins = ch.insert_features(profile_df, run_id)
                        stats["features_per_profile"] = {"rows_inserted": prof_ins}
                        logger.info("  [features/per-profile] %d rows inserted", prof_ins)

                # D2: per-shop × cohort (scope = shop:{sid}|cohort:{slug})
                cohort_configs = load_cohort_configs()
                if cohort_configs:
                    sc_df = _per_shop_cohort_features_df(
                        tensor, cohort_configs, skipnan=skipnan,
                    )
                    if not sc_df.empty:
                        sc_ins = ch.insert_features(sc_df, run_id)
                        stats["features_shop_cohort"] = {"rows_inserted": sc_ins}
                        logger.info("  [features/shop-cohort] %d rows inserted", sc_ins)

                    # D3: per-profile × cohort (scope = shopcohort:{slug}|cohort:{slug})
                    if profiles:
                        pc_df = _per_profile_cohort_features_df(
                            tensor, profiles, cohort_configs, skipnan=skipnan,
                        )
                        if not pc_df.empty:
                            pc_ins = ch.insert_features(pc_df, run_id)
                            stats["features_profile_cohort"] = {"rows_inserted": pc_ins}
                            logger.info("  [features/profile-cohort] %d rows inserted", pc_ins)

        stats["features_seconds"] = round(time.time() - t_step, 2)

    # ── Step: cohorts ────────────────────────────────────────────────────
    if "cohorts" in effective_steps:
        t_step = time.time()

        if agg is None:
            logger.warning("  [cohorts] no agg data, skipping")
        else:
            configs = load_cohort_configs()
            if configs:
                cohort_df = compute_cohort_features(
                    agg, configs, device=device, skipnan=skipnan,
                )
                if not cohort_df.empty:
                    inserted = ch.insert_features(cohort_df, run_id)
                    stats["cohorts"] = {
                        "n_cohorts": len(configs),
                        "rows_inserted": inserted,
                    }
                    logger.info("  [cohorts] done: %d cohorts → %d rows inserted",
                                len(configs), inserted)
                else:
                    logger.warning("  [cohorts] no cohort rows produced")
            else:
                logger.info("  [cohorts] no cohort configs found in PG")

        stats["cohorts_seconds"] = round(time.time() - t_step, 2)

    elapsed_total = time.time() - t0
    stats["total_seconds"] = round(elapsed_total, 2)
    logger.info(
        "pipeline.run DONE  run_id=%s  total_time=%.1fs",
        run_id, elapsed_total,
    )
    return stats


# ── 内部工具 ─────────────────────────────────────────────────────────────

def _agg_to_features_df(
    agg,
    features: dict[str, object],
) -> pd.DataFrame:
    """将 AggResult + feature tensors 组装为 features_wide 格式 DataFrame。"""
    import math
    import torch

    n_iphones = len(agg.iphone_ids)
    n_buckets = len(agg.bucket_index)
    rows = []

    for i_idx in range(n_iphones):
        iphone_id = int(agg.iphone_ids[i_idx])
        scope = f"iphone:{iphone_id}"

        for b_idx in range(n_buckets):
            mean_val = agg.mean[i_idx, b_idx].item()
            # shop_count=0 的桶 mean 为 NaN，无有效数据，跳过
            if math.isnan(mean_val):
                continue
            row = {
                "bucket": agg.bucket_index[b_idx],
                "scope": scope,
                "mean": round(mean_val, 2),
                "median": round(agg.median[i_idx, b_idx].item(), 2),
                "std": round(agg.std[i_idx, b_idx].item(), 2),
                "shop_count": int(agg.shop_count[i_idx, b_idx].item()),
                "dispersion": round(agg.dispersion[i_idx, b_idx].item(), 2),
            }
            for fname, ftensor in features.items():
                row[fname] = round(ftensor[i_idx, b_idx].item(), 2)
            rows.append(row)

    return pd.DataFrame(rows)


def _per_shop_features_df(tensor, *, skipnan: bool = False) -> pd.DataFrame:
    """为每个 (shop_id, iphone_id) 计算特征，scope = shop:{sid}|iphone:{iid}。

    单店的 mean 就是价格本身 (shop_count=1, std=0)。
    """
    import math
    import torch
    from AppleStockChecker.engine.features import compute_all_features, compute_all_features_skipnan

    n_iphones, n_shops, n_buckets = tensor.data.shape
    rows = []

    for s_idx in range(n_shops):
        shop_id = int(tensor.shop_ids[s_idx])
        # 取该店的 2D 切片: (n_iphones, n_buckets)
        shop_slice = tensor.data[:, s_idx, :]

        # 检查该店是否有任何有效数据
        if torch.isnan(shop_slice).all():
            continue

        feat_fn = compute_all_features_skipnan if skipnan else compute_all_features
        features = feat_fn(shop_slice)

        for i_idx in range(n_iphones):
            iphone_id = int(tensor.iphone_ids[i_idx])
            scope = f"shop:{shop_id}|iphone:{iphone_id}"

            for b_idx in range(n_buckets):
                price_val = shop_slice[i_idx, b_idx].item()
                if math.isnan(price_val):
                    continue
                row = {
                    "bucket": tensor.bucket_index[b_idx],
                    "scope": scope,
                    "mean": round(price_val, 2),
                    "median": round(price_val, 2),
                    "std": 0.0,
                    "shop_count": 1,
                    "dispersion": 0.0,
                }
                for fname, ftensor in features.items():
                    row[fname] = round(ftensor[i_idx, b_idx].item(), 2)
                rows.append(row)

    logger.info("_per_shop_features_df: %d rows", len(rows))
    return pd.DataFrame(rows)


def _per_profile_features_df(tensor, profiles, *, skipnan: bool = False) -> pd.DataFrame:
    """为每个 ShopWeightProfile × iphone_id 计算加权特征。

    scope = shopcohort:{slug}|iphone:{iid}
    """
    import math
    import torch
    from AppleStockChecker.engine.features import compute_all_features, compute_all_features_skipnan

    shop_id_to_idx = {int(v): i for i, v in enumerate(tensor.shop_ids)}
    n_iphones = tensor.data.shape[0]
    n_buckets = tensor.data.shape[2]
    rows = []

    for profile in profiles:
        # 找到 profile 中在 tensor 里存在的 shop 及其权重
        indices = []
        weights = []
        for item in profile.items:
            s_idx = shop_id_to_idx.get(item["shop_id"])
            if s_idx is not None:
                indices.append(s_idx)
                weights.append(item["weight"])

        if not indices:
            logger.warning("profile %s: no shops found in tensor, skipping", profile.slug)
            continue

        w = torch.tensor(weights, dtype=torch.float64, device=tensor.data.device)
        w = w / w.sum()  # 归一化

        # 对每个 iphone_id，按 shop 权重加权聚合
        # tensor.data shape: (n_iphones, n_shops, n_buckets)
        # 取 indices 对应的 shops: (n_iphones, len(indices), n_buckets)
        shop_data = tensor.data[:, indices, :]

        # 将 NaN 替换为 0 以便加权求和，同时跟踪有效性
        valid_mask = ~torch.isnan(shop_data)  # (I, S_sub, B)
        shop_data_clean = torch.where(valid_mask, shop_data, torch.zeros_like(shop_data))

        # 加权有效掩码
        w_expanded = w.unsqueeze(0).unsqueeze(2)  # (1, S_sub, 1)
        valid_w = torch.where(valid_mask, w_expanded.expand_as(valid_mask), torch.zeros_like(shop_data))
        w_sum = valid_w.sum(dim=1)  # (I, B)

        # 加权平均
        weighted_mean = (shop_data_clean * w_expanded.expand_as(shop_data_clean)).sum(dim=1)  # (I, B)
        # 重新归一化 (只除以实际参与的权重之和)
        weighted_mean = torch.where(w_sum > 0, weighted_mean / w_sum, torch.full_like(weighted_mean, float("nan")))

        # 加权标准差: sqrt(sum(w_i * (x_i - mean)^2) / sum(w_i))
        diff_sq = (shop_data_clean - weighted_mean.unsqueeze(1)) ** 2  # (I, S_sub, B)
        weighted_var = torch.where(
            valid_mask, diff_sq * w_expanded.expand_as(diff_sq), torch.zeros_like(diff_sq),
        ).sum(dim=1)  # (I, B)
        weighted_std = torch.where(
            w_sum > 0, torch.sqrt(weighted_var / w_sum), torch.zeros_like(w_sum),
        )  # (I, B)
        weighted_disp = torch.where(
            weighted_mean != 0, weighted_std / weighted_mean, torch.zeros_like(weighted_std),
        )  # (I, B)

        feat_fn = compute_all_features_skipnan if skipnan else compute_all_features
        features = feat_fn(weighted_mean)

        # E1: 如果是 full_store profile，预加载 official_prices 用于 logb
        is_full_store = (profile.slug == "full_store")
        official_prices = {}
        if is_full_store:
            from django.conf import settings as _settings
            official_prices = getattr(_settings, "IPHONE_OFFICIAL_PRICES", {})

        for i_idx in range(n_iphones):
            iphone_id = int(tensor.iphone_ids[i_idx])
            scope = f"shopcohort:{profile.slug}|iphone:{iphone_id}"

            for b_idx in range(n_buckets):
                mean_val = weighted_mean[i_idx, b_idx].item()
                if math.isnan(mean_val):
                    continue
                row = {
                    "bucket": tensor.bucket_index[b_idx],
                    "scope": scope,
                    "mean": round(mean_val, 2),
                    "median": round(mean_val, 2),
                    "std": round(weighted_std[i_idx, b_idx].item(), 2),
                    "shop_count": int(valid_mask[i_idx, :, b_idx].sum().item()),
                    "dispersion": round(weighted_disp[i_idx, b_idx].item(), 2),
                }
                for fname, ftensor in features.items():
                    row[fname] = round(ftensor[i_idx, b_idx].item(), 2)

                # E1: logb 直接在此计算，避免二次 INSERT
                if is_full_store and official_prices:
                    official = official_prices.get(iphone_id)
                    if official and official > 0:
                        for W in FEATURE_WINDOWS:
                            wma_val = row.get(f"wma_{W}")
                            if wma_val is not None and not math.isnan(wma_val) and wma_val > 0:
                                row[f"logb_{W}"] = round(math.log(wma_val / official), 4)

                rows.append(row)

    logger.info("_per_profile_features_df: %d rows", len(rows))
    return pd.DataFrame(rows)


def _per_shop_cohort_features_df(
    tensor, cohort_configs, *, skipnan: bool = False,
) -> pd.DataFrame:
    """D2: 每个 shop × cohort 的加权特征。scope = shop:{sid}|cohort:{slug}"""
    import math
    import torch
    from AppleStockChecker.engine.features import compute_all_features, compute_all_features_skipnan

    iphone_id_to_idx = {int(v): i for i, v in enumerate(tensor.iphone_ids)}
    n_shops = tensor.data.shape[1]
    n_buckets = tensor.data.shape[2]
    rows = []

    for cfg in cohort_configs:
        member_indices = []
        member_weights = []
        for m in cfg.members:
            idx = iphone_id_to_idx.get(m["iphone_id"])
            if idx is not None:
                member_indices.append(idx)
                member_weights.append(m["weight"])
        if not member_indices:
            continue

        w = torch.tensor(member_weights, dtype=torch.float64, device=tensor.data.device)
        w = w / w.sum()

        for s_idx in range(n_shops):
            shop_id = int(tensor.shop_ids[s_idx])
            # (n_members, n_buckets)
            member_data = tensor.data[member_indices, s_idx, :]

            valid_mask = ~torch.isnan(member_data)
            clean = torch.where(valid_mask, member_data, torch.zeros_like(member_data))

            w_exp = w.unsqueeze(1)  # (n_members, 1)
            valid_w = torch.where(valid_mask, w_exp.expand_as(valid_mask), torch.zeros_like(clean))
            w_sum = valid_w.sum(dim=0)  # (n_buckets,)

            weighted_sum = (clean * w_exp.expand_as(clean)).sum(dim=0)
            wmean = torch.where(w_sum > 0, weighted_sum / w_sum, torch.full_like(w_sum, float("nan")))

            diff_sq = (clean - wmean.unsqueeze(0)) ** 2
            wvar = torch.where(valid_mask, diff_sq * w_exp.expand_as(diff_sq), torch.zeros_like(diff_sq)).sum(dim=0)
            wstd = torch.where(w_sum > 0, torch.sqrt(wvar / w_sum), torch.zeros_like(w_sum))
            wdisp = torch.where(wmean != 0, wstd / wmean, torch.zeros_like(wstd))

            wmean_2d = wmean.unsqueeze(0)  # (1, B)
            feat_fn = compute_all_features_skipnan if skipnan else compute_all_features
            features = feat_fn(wmean_2d)

            scope = f"shop:{shop_id}|cohort:{cfg.slug}"
            for b_idx in range(n_buckets):
                mv = wmean[b_idx].item()
                if math.isnan(mv):
                    continue
                row = {
                    "bucket": tensor.bucket_index[b_idx],
                    "scope": scope,
                    "mean": round(mv, 2),
                    "median": round(mv, 2),
                    "std": round(wstd[b_idx].item(), 2),
                    "shop_count": 1,
                    "dispersion": round(wdisp[b_idx].item(), 2),
                }
                for fname, ftensor in features.items():
                    row[fname] = round(ftensor[0, b_idx].item(), 2)
                rows.append(row)

    logger.info("_per_shop_cohort_features_df: %d rows", len(rows))
    return pd.DataFrame(rows)


def _per_profile_cohort_features_df(
    tensor, profiles, cohort_configs, *, skipnan: bool = False,
) -> pd.DataFrame:
    """D3: 每个 profile × cohort 的加权特征。scope = shopcohort:{prof}|cohort:{slug}"""
    import math
    import torch
    from AppleStockChecker.engine.features import compute_all_features, compute_all_features_skipnan

    shop_id_to_idx = {int(v): i for i, v in enumerate(tensor.shop_ids)}
    iphone_id_to_idx = {int(v): i for i, v in enumerate(tensor.iphone_ids)}
    n_buckets = tensor.data.shape[2]
    rows = []

    for profile in profiles:
        shop_indices = []
        shop_weights = []
        for item in profile.items:
            s_idx = shop_id_to_idx.get(item["shop_id"])
            if s_idx is not None:
                shop_indices.append(s_idx)
                shop_weights.append(item["weight"])
        if not shop_indices:
            continue

        sw = torch.tensor(shop_weights, dtype=torch.float64, device=tensor.data.device)
        sw = sw / sw.sum()

        for cfg in cohort_configs:
            member_indices = []
            member_weights = []
            for m in cfg.members:
                idx = iphone_id_to_idx.get(m["iphone_id"])
                if idx is not None:
                    member_indices.append(idx)
                    member_weights.append(m["weight"])
            if not member_indices:
                continue

            mw = torch.tensor(member_weights, dtype=torch.float64, device=tensor.data.device)
            mw = mw / mw.sum()

            # (n_members, n_shops_sub, n_buckets)
            sub = tensor.data[member_indices][:, shop_indices, :]
            valid = ~torch.isnan(sub)
            clean = torch.where(valid, sub, torch.zeros_like(sub))

            # Combined weight: shop_weight × model_weight → (n_members, n_shops_sub, 1)
            combined_w = (mw.unsqueeze(1) * sw.unsqueeze(0)).unsqueeze(2)  # (M, S, 1)
            valid_w = torch.where(valid, combined_w.expand_as(valid), torch.zeros_like(clean))
            w_sum = valid_w.sum(dim=(0, 1))  # (B,)

            weighted_sum = (clean * combined_w.expand_as(clean)).sum(dim=(0, 1))  # (B,)
            wmean = torch.where(w_sum > 0, weighted_sum / w_sum, torch.full_like(w_sum, float("nan")))

            diff_sq = (clean - wmean.unsqueeze(0).unsqueeze(0)) ** 2
            wvar = torch.where(
                valid, diff_sq * combined_w.expand_as(diff_sq), torch.zeros_like(diff_sq),
            ).sum(dim=(0, 1))
            wstd = torch.where(w_sum > 0, torch.sqrt(wvar / w_sum), torch.zeros_like(w_sum))
            wdisp = torch.where(wmean != 0, wstd / wmean, torch.zeros_like(wstd))

            wmean_2d = wmean.unsqueeze(0)
            feat_fn = compute_all_features_skipnan if skipnan else compute_all_features
            features = feat_fn(wmean_2d)

            shop_count_per_bucket = valid.any(dim=0).sum(dim=0)  # (B,)

            scope = f"shopcohort:{profile.slug}|cohort:{cfg.slug}"
            for b_idx in range(n_buckets):
                mv = wmean[b_idx].item()
                if math.isnan(mv):
                    continue
                row = {
                    "bucket": tensor.bucket_index[b_idx],
                    "scope": scope,
                    "mean": round(mv, 2),
                    "median": round(mv, 2),
                    "std": round(wstd[b_idx].item(), 2),
                    "shop_count": int(shop_count_per_bucket[b_idx].item()),
                    "dispersion": round(wdisp[b_idx].item(), 2),
                }
                for fname, ftensor in features.items():
                    row[fname] = round(ftensor[0, b_idx].item(), 2)
                rows.append(row)

    logger.info("_per_profile_cohort_features_df: %d rows", len(rows))
    return pd.DataFrame(rows)

