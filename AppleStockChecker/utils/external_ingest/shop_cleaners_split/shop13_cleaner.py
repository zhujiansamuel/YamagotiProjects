from __future__ import annotations
"""
shop13 清洗器 — 家電市場

  原始 DataFrame（新品価格 / 買取商品2 / time-scraped）
    │
    └─ clean_with_model_capacity_matching() ← 公共模板（model+cap→全色PN展开）
"""
import logging

import pandas as pd

from ..cleaner_tools import clean_with_model_capacity_matching

logger = logging.getLogger(__name__)

def clean_shop13(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入列:
      - 「新品価格」: 价格
      - 「買取商品2」: 机种名 + 容量
      - 「time-scraped」: 抓取时间

    输出: part_number, shop_name(=家電市場), price_new, recorded_at
    对同一 (机种, 容量) 展开所有颜色的 PN。
    """
    return clean_with_model_capacity_matching(
        df,
        cleaner_name="shop13",
        shop_name="家電市場",
        model_col="買取商品2",
        price_col="新品価格",
        coerce_price=False,
    )
