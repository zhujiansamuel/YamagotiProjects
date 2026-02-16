from __future__ import annotations
"""
shop8 清洗器 — 買取wiki

  原始 DataFrame（機種名 / 未開封 / time-scraped）
    │
    ├─ _extract_part_number()           ← 型番直接抽取（型番: XXXJ/A or PN 正则）
    └─ clean_with_model_capacity_matching() ← 公共模板（PN直提取 or model+cap展开）
"""
from typing import Optional
import logging
import re

import pandas as pd

from ..cleaner_tools import normalize_text_basic, clean_with_model_capacity_matching

logger = logging.getLogger(__name__)

PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")

def _extract_part_number(text: str) -> str | None:
    t = normalize_text_basic(text)
    # 1) 优先：显式 "型番: XXXXXJ/A"
    m = re.search(r"型番[:：]\s*([A-Z0-9]{4,6}\d{0,3}J/A)\b", t)
    if m:
        return m.group(1)
    # 2) 兜底：全文 PN 正则
    m2 = PN_REGEX.search(t)
    return m2.group(0) if m2 else None

def clean_shop8(df: pd.DataFrame) -> pd.DataFrame:
    return clean_with_model_capacity_matching(
        df,
        cleaner_name="shop8",
        shop_name="買取wiki",
        model_col="機種名",
        price_col="未開封",
        pn_extractor_fn=_extract_part_number,
        coerce_price=False,
    )
