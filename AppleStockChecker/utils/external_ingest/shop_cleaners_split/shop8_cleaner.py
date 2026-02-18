from __future__ import annotations
"""
shop8 清洗器 — 買取wiki

  原始 DataFrame（機種名 / 未開封 / time-scraped）
    │
    ├─ _extract_model_cap_color_shop8() ← 先 (model,cap,color) 匹配单一 PN
    ├─ _extract_part_number()           ← 型番兜底（型番: XXXJ/A or PN 正则）
    └─ clean_with_model_capacity_matching() ← 公共模板
"""
from typing import Optional, Tuple
import logging
import re

import pandas as pd

from ..cleaner_tools import (
    _normalize_model_generic,
    normalize_text_basic,
    clean_with_model_capacity_matching,
)

logger = logging.getLogger(__name__)

PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")


def _extract_model_cap_color_shop8(text: str) -> Optional[Tuple[str, int, str]]:
    """
    从機種名解析 (model_norm, cap_gb, color_raw)。
    颜色在型号容量之后、型番/全角括号之前，如: iPhone 17 Pro 256GB シルバー  （）\\n型番：MG854J/A
    """
    t = text.split("型番")[0].split("（")[0].strip()
    m = re.search(r"(\d{2,4})\s*GB\s+(.+)$", t)
    if not m:
        return None
    cap = int(m.group(1))
    color_raw = m.group(2).strip()
    if not color_raw:
        return None
    before_cap = re.sub(r"\d{2,4}\s*GB\s+.*$", "", t).strip()
    model_norm = _normalize_model_generic(before_cap)
    if not model_norm:
        return None
    return (model_norm, cap, color_raw)


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
        model_cap_color_extractor_fn=_extract_model_cap_color_shop8,
        pn_extractor_fn=_extract_part_number,
        coerce_price=False,
    )
