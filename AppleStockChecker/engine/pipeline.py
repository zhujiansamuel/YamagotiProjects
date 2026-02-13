"""
主 Pipeline: 串联 reader → align → (后续 aggregate → features → cohorts)。
Phase 1 只实现 align 步骤。
参考: docs/REFACTOR_PLAN_V1.md §6.2, §20 Phase 1
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

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
        PyTorch 设备 (Phase 1 不使用)
    steps : list[str] | None
        要执行的步骤, 默认全部。可选: align, aggregate, features, cohorts
    batch_days : int
        每次从 PG 读取的天数
    iphone_ids, shop_ids : list[int] | None
        限定范围

    Returns
    -------
    dict  执行统计
    """
    # 延迟导入避免 Django 未初始化时出错
    from AppleStockChecker.engine.reader import read_price_records
    from AppleStockChecker.engine.align import align_to_buckets
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
    if "align" in effective_steps:
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

    # ── Step: aggregate (Phase 2) ────────────────────────────────────────
    if "aggregate" in effective_steps:
        logger.info("  [aggregate] not yet implemented — skipping (Phase 2)")

    # ── Step: features (Phase 2) ─────────────────────────────────────────
    if "features" in effective_steps:
        logger.info("  [features] not yet implemented — skipping (Phase 2)")

    # ── Step: cohorts (Phase 2) ──────────────────────────────────────────
    if "cohorts" in effective_steps:
        logger.info("  [cohorts] not yet implemented — skipping (Phase 2)")

    elapsed_total = time.time() - t0
    stats["total_seconds"] = round(elapsed_total, 2)
    logger.info(
        "pipeline.run DONE  run_id=%s  total_time=%.1fs  stats=%s",
        run_id, elapsed_total, stats,
    )
    return stats
