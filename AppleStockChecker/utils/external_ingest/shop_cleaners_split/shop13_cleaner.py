from __future__ import annotations
"""
shop13 清洗器 — 家電市場

  原始 DataFrame（新品価格 / 買取商品2 / time-scraped）
    │
    └─ clean_with_model_capacity_matching() ← 公共模板（一行 model+cap+color → 单一 PN）
"""
from typing import Optional, Tuple
import logging
import re

import pandas as pd

from ..cleaner_tools import (
    _normalize_model_generic,
    _parse_capacity_gb,
    clean_with_model_capacity_matching,
)

logger = logging.getLogger(__name__)


def _extract_model_cap_color_shop13(text: str) -> Optional[Tuple[str, int, str]]:
    """
    从買取商品2 解析 (model_norm, cap_gb, color_raw)。
    颜色在 [...] 内，如: iPhone 17 Pro 256GB SIMフリー [ディープブルー]
    """
    m_color = re.search(r"\[([^\]]+)\]", text)
    if not m_color:
        return None
    color_raw = m_color.group(1).strip()
    text_without_color = re.sub(r"\s*\[[^\]]+\]\s*", " ", text).strip()
    model_norm = _normalize_model_generic(text_without_color)
    cap = _parse_capacity_gb(text_without_color)
    if not model_norm or cap is None:
        return None
    return (model_norm, cap, color_raw)


def clean_shop13(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入列:
      - 「新品価格」: 价格
      - 「買取商品2」: 机种名 + 容量 + [颜色]
      - 「time-scraped」: 抓取时间

    输出: part_number, shop_name(=家電市場), price_new, recorded_at
    一行 (model, cap, color) → 匹配单一 PN。
    """
    return clean_with_model_capacity_matching(
        df,
        cleaner_name="shop13",
        shop_name="家電市場",
        model_col="買取商品2",
        price_col="新品価格",
        model_cap_color_extractor_fn=_extract_model_cap_color_shop13,
        coerce_price=False,
    )
