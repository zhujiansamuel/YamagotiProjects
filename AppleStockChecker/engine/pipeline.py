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

from .config import ALL_STEPS, BucketConfig

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
    from AppleStockChecker.engine.aggregate import build_price_tensor, aggregate_cross_shop
    from AppleStockChecker.engine.features import compute_all_features
    from AppleStockChecker.engine.cohorts import load_cohort_configs, compute_cohort_features
    from AppleStockChecker.services.clickhouse_service import ClickHouseService

    effective_steps = steps or ALL_STEPS
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
    if "aggregate" in effective_steps:
        t_step = time.time()

        if full_aligned.empty:
            logger.warning("  [aggregate] no aligned data available, skipping")
        else:
            tensor = build_price_tensor(full_aligned, device=device)
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
            iphone_features = compute_all_features(agg.mean)

            # 写入 CH: 组装 iphone 级 features_wide DataFrame
            iphone_df = _agg_to_features_df(agg, iphone_features)
            inserted = ch.insert_features(iphone_df, run_id)

            stats["features"] = {
                "n_feature_cols": len(iphone_features),
                "rows_inserted": inserted,
            }
            logger.info("  [features] done: %d feature cols → %d rows inserted",
                        len(iphone_features), inserted)

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
                    agg, iphone_features, configs, device=device,
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
    import torch

    n_iphones = len(agg.iphone_ids)
    n_buckets = len(agg.bucket_index)
    rows = []

    for i_idx in range(n_iphones):
        iphone_id = int(agg.iphone_ids[i_idx])
        scope = f"iphone:{iphone_id}"

        for b_idx in range(n_buckets):
            row = {
                "bucket": agg.bucket_index[b_idx],
                "scope": scope,
                "mean": agg.mean[i_idx, b_idx].item(),
                "median": agg.median[i_idx, b_idx].item(),
                "std": agg.std[i_idx, b_idx].item(),
                "shop_count": int(agg.shop_count[i_idx, b_idx].item()),
                "dispersion": agg.dispersion[i_idx, b_idx].item(),
            }
            for fname, ftensor in features.items():
                row[fname] = ftensor[i_idx, b_idx].item()
            rows.append(row)

    return pd.DataFrame(rows)
