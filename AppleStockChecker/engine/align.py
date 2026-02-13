"""
时间对齐: PriceRecord DataFrame → 15 分钟桶对齐后的 price_aligned 格式。
参考: docs/REFACTOR_PLAN_V1.md §4.2, §6.2 Step 2
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import BucketConfig

logger = logging.getLogger(__name__)


def align_to_buckets(
    df: pd.DataFrame,
    config: BucketConfig | None = None,
) -> pd.DataFrame:
    """将原始 PriceRecord 对齐到 15 分钟整点桶。

    Parameters
    ----------
    df : DataFrame
        必须包含列: shop_id, iphone_id, price_new, price_a, price_b, recorded_at
    config : BucketConfig
        桶配置 (默认 15 分钟)

    Returns
    -------
    DataFrame  columns: bucket, shop_id, iphone_id, price_new, price_a, price_b,
                        alignment_diff_sec, record_time
    """
    if config is None:
        config = BucketConfig()

    if df.empty:
        logger.warning("align_to_buckets: empty input")
        return pd.DataFrame(columns=[
            "bucket", "shop_id", "iphone_id",
            "price_new", "price_a", "price_b",
            "alignment_diff_sec", "record_time",
        ])

    freq = f"{config.interval_min}min"

    out = df.copy()

    # 确保 recorded_at 是 datetime
    out["recorded_at"] = pd.to_datetime(out["recorded_at"], utc=True)

    # floor 到 15min 整点
    out["bucket"] = out["recorded_at"].dt.floor(freq)

    # 每 (shop, iphone, bucket) 取 recorded_at 最新的一行
    out = (
        out.sort_values("recorded_at")
        .groupby(["bucket", "shop_id", "iphone_id"], as_index=False)
        .last()
    )

    # alignment_diff_sec = recorded_at - bucket (秒)
    out["alignment_diff_sec"] = (
        (out["recorded_at"] - out["bucket"]).dt.total_seconds().astype(int)
    )
    out.rename(columns={"recorded_at": "record_time"}, inplace=True)

    # 转为 Asia/Tokyo 用于写入 CH
    out["bucket"] = out["bucket"].dt.tz_convert("Asia/Tokyo")
    out["record_time"] = out["record_time"].dt.tz_convert("Asia/Tokyo")

    # 只保留需要的列
    cols = [
        "bucket", "shop_id", "iphone_id",
        "price_new", "price_a", "price_b",
        "alignment_diff_sec", "record_time",
    ]
    out = out[cols].reset_index(drop=True)

    logger.info("align_to_buckets: %d rows → %d aligned rows, buckets %s ~ %s",
                len(df), len(out),
                out["bucket"].min(), out["bucket"].max())
    return out
