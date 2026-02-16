from __future__ import annotations
"""
shop18 清洗器 — 買取オク

  使用公共模板 clean_with_jan_matching。
  JAN 优先匹配，fallback 通过 type 列的 (model, capacity, color) 回退匹配。
"""
from typing import Optional
import logging
import re

import pandas as pd

from ..cleaner_tools import (
    clean_with_jan_matching,
    validate_columns,
    _parse_capacity_gb,
    _normalize_model_generic,
    to_int_yen,
)

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop18"
SHOP_NAME = "買取オク"


def _match_by_type(type_text: str, info_df: pd.DataFrame) -> Optional[str]:
    """
    当 JAN 无法匹配时，根据 `type` 文本（如 'iPhone 17 Pro 512GB ディープブルー'）
    用 (model_norm, capacity_gb, color_norm) 回退匹配到 part_number。
    """
    if not type_text:
        return None
    txt = str(type_text).replace("\u3000", " ").replace("\xa0", " ").strip()
    model_norm = _normalize_model_generic(txt)
    cap_gb = _parse_capacity_gb(txt)
    if not model_norm or pd.isna(cap_gb):
        return None
    cap_gb = int(cap_gb)

    # 在该 (model, cap) 下，寻找哪个颜色名出现在 type 文本中
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    cand = df[(df["model_name_norm"] == model_norm) & (df["capacity_gb"] == cap_gb)]
    if cand.empty:
        return None

    # 直接用 "颜色原文子串" 命中（多数站点颜色在文案中能直接找到）
    for _, r in cand.iterrows():
        color_raw = str(r["color"])
        if color_raw and color_raw in txt:
            return str(r["part_number"])

    # 若未命中且候选仅有 1 个颜色，直接返回（保底）
    if len(cand) == 1:
        return str(cand.iloc[0]["part_number"])

    return None


def _iter_records_shop18(df: pd.DataFrame):
    """
    产出规范化记录。
    输入列：jan, type, price, time-scraped
    """
    for _, row in df.iterrows():
        yield {
            "JAN": row.get("jan"),
            "price": row.get("price"),
            "time-scraped": row.get("time-scraped"),
            "_type": row.get("type"),  # 传递给 fallback
        }


def _fallback_match_shop18(rec: dict, info_df: pd.DataFrame) -> Optional[str]:
    """JAN 无法匹配时，用 type 列做 (model, cap, color) 回退匹配。"""
    return _match_by_type(rec.get("_type"), info_df)


def clean_shop18(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入列: jan, type, price, time-scraped
    输出: part_number, shop_name, price_new, recorded_at
    """
    validate_columns(df, ["jan", "type", "price", "time-scraped"],
                     cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME)
    return clean_with_jan_matching(
        df,
        cleaner_name=CLEANER_NAME,
        shop_name=SHOP_NAME,
        iter_records_fn=_iter_records_shop18,
        price_extractor_fn=to_int_yen,
        fallback_match_fn=_fallback_match_shop18,
    )
