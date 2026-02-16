from __future__ import annotations
"""
shop10 清洗器 — ドラゴンモバイル

  原始 DataFrame（data2 / price / time-scraped）
    │
    └─ clean_with_model_capacity_matching() ← 公共模板（model+cap→全色PN展开）

  注: shop10 保留 debug 日志功能（颜色/折扣模式检测），
  通过自定义 row_filter_fn 在模板外部实现。
"""
from typing import List, Optional
import logging
import re
import time

import pandas as pd

from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    extract_price_yen,
    assemble_output_df,
    validate_columns,
    _load_iphone17_info_df_from_db,
    log_cleaner_start,
    log_cleaner_complete,
    _truncate_for_log,
    parse_dt_aware,
    clean_with_model_capacity_matching,
)

logger = logging.getLogger(__name__)

def clean_shop10(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    """
    shop10 使用公共模板，但保留 debug 日志功能。
    debug 模式下，对匹配到颜色/折扣模式的行输出详细日志。
    """
    if debug:
        # debug 日志：检测含颜色/折扣信息的行
        _COLOR_DISCOUNT_PAT = re.compile(
            r"(ブラック|ホワイト|ブルー|グリーン|ピンク|レッド|イエロー|パープル|ゴールド|シルバー|"
            r"グラファイト|ミッドナイト|スターライト|ナチュラル|チタニウム|チタン|"
            r"Black|White|Blue|Green|Pink|Red|Yellow|Purple|Gold|Silver|Titanium|"
            r"値下げ|値引|割引|円引|OFF|オフ|[-−–]\s*\d|\d+\s*円\s*(?:引|OFF|オフ))",
            re.I
        )
        s_data2 = df["data2"].fillna("").astype(str) if "data2" in df.columns else pd.Series(dtype=str)
        s_price = df["price"].fillna("").astype(str) if "price" in df.columns else pd.Series(dtype=str)
        if not s_data2.empty:
            mask = s_data2.str.contains(_COLOR_DISCOUNT_PAT, na=False) | s_price.str.contains(_COLOR_DISCOUNT_PAT, na=False)
            logger.debug(
                "Debug mode: color/discount pattern matching statistics",
                extra={
                    "event_type": "debug_stats",
                    "shop_name": "ドラゴンモバイル",
                    "cleaner_name": "shop10",
                    "total_rows": len(df),
                    "hit_rows": int(mask.sum()),
                    "debug_limit": debug_limit,
                }
            )

    return clean_with_model_capacity_matching(
        df,
        cleaner_name="shop10",
        shop_name="ドラゴンモバイル",
        model_col="data2",
        price_col="price",
        add_model_norm=True,
        coerce_price=False,
    )
