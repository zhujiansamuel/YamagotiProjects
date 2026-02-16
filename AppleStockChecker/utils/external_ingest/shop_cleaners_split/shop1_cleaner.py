from __future__ import annotations
"""
shop1 清洗器 — 買取商店

  原始 DataFrame（JAN / price / time-scraped 或 JSON 列）
    │
    ├─ _iter_records()           ← Step 1: 规范化记录（直列 or JSON 拉平）
    └─ clean_with_jan_matching() ← Step 2: JAN→PN 映射 + 价格/时间解析（公共模板）
"""
import logging
from typing import List
import re
import json
import pandas as pd

from ..cleaner_tools import clean_with_jan_matching

logger = logging.getLogger(__name__)


def _iter_records(df: pd.DataFrame):
    """
    产出规范化记录：{"JAN":..., "price":..., "time-scraped": ...}
    适配两种输入：
      A) 直列：JAN, price, time-scraped
      B) JSON 列：json（对象/数组/带 data 的对象），同行的 time-scraped 为默认时间
         - 兼容字段别名：jancode / goodsPrice / time_scraped / timestamp / keywords(兜底提取 JAN)
    """
    cols = {c.lower(): c for c in df.columns}

    # A) 直列
    if all(k in cols for k in ["jan", "price", "time-scraped"]):
        JAN_col, price_col, ts_col = cols["jan"], cols["price"], cols["time-scraped"]
        for _, row in df.iterrows():
            yield {"JAN": row.get(JAN_col), "price": row.get(price_col), "time-scraped": row.get(ts_col)}
        return

    # B) JSON 列
    json_col = cols.get("json")
    ts_col = cols.get("time-scraped") or cols.get("time_scraped")
    if not json_col:
        return

    for _, row in df.iterrows():
        default_ts = row.get(ts_col)
        cell = row.get(json_col)
        parsed = None

        if isinstance(cell, (dict, list)):
            parsed = cell
        elif isinstance(cell, str) and cell.strip():
            s = cell.strip().lstrip("\ufeff")
            # CSV 风格的 "" → "（若存在）
            if s.count('""') and not s.count('\\"'):
                s = s.replace('""', '"')
            try:
                parsed = json.loads(s)
            except Exception:
                continue
        else:
            continue

        # 统一拉平成若干对象
        items: List[dict] = []
        if isinstance(parsed, dict):
            items = [x for x in parsed.get("data", [parsed]) if isinstance(x, dict)]
        elif isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]

        for it in items:
            jan = it.get("JAN") or it.get("jan") or it.get("jancode") or it.get("jAN")
            if not jan:
                jan = it.get("keywords")  # 兜底：从文字里抽出 JAN
            price = it.get("price") or it.get("goodsPrice") or it.get("Price")
            ts = it.get("time-scraped") or it.get("time_scraped") or it.get("timestamp") or default_ts
            yield {"JAN": jan, "price": price, "time-scraped": ts}


def clean_shop1(df: pd.DataFrame) -> pd.DataFrame:
    """
    以 JAN 映射 part_number；price -> price_new；time-scraped -> recorded_at。
    shop_name 固定为「買取商店」。
    仅输出 _load_iphone17_info_df_from_db() 中存在的机型。
    """
    return clean_with_jan_matching(
        df,
        cleaner_name="shop1",
        shop_name="買取商店",
        iter_records_fn=_iter_records,
    )
