from __future__ import annotations
"""
shop20 清洗器

  使用公共模板 clean_with_jan_matching。
  从 JSON 列解析 data 数组，提取 jancode + goodsPrice。
"""
from typing import Optional, List
import json
import logging

import pandas as pd

from ..cleaner_tools import (
    clean_with_jan_matching,
    validate_columns,
    to_int_yen,
)

logger = logging.getLogger(__name__)


def _coerce_price(v) -> Optional[int]:
    """goodsPrice 既可能是数字也可能是字符串，统一转 int（日元）"""
    if v is None:
        return None
    if isinstance(v, (int, float)) and pd.notna(v):
        return int(round(float(v)))
    return to_int_yen(v)


def _iter_records_shop20(df: pd.DataFrame):
    """
    产出规范化记录。
    输入列：json, time-scraped
    从 json['data'] 数组中逐条提取 jancode + goodsPrice
    """
    for _, row in df.iterrows():
        raw_json = row.get("json")
        if not isinstance(raw_json, str) or not raw_json.strip():
            continue

        # 将 CSV 内部双引号转为标准 JSON 引号
        s = raw_json.replace('""', '"').strip()

        try:
            payload = json.loads(s)
        except Exception:
            s2 = s.lstrip("\ufeff").strip()
            try:
                payload = json.loads(s2)
            except Exception:
                continue

        data = payload.get("data")
        if not isinstance(data, list):
            continue

        rec_at = row.get("time-scraped")

        for item in data:
            if not isinstance(item, dict):
                continue

            jan = item.get("jancode") or item.get("jan")
            if not jan:
                jan = item.get("keywords")

            price = _coerce_price(item.get("goodsPrice"))

            yield {
                "JAN": jan,
                "price": price,
                "time-scraped": rec_at,
            }


def _price_passthrough(v):
    """shop20 的价格在 _iter_records 中已转为 int，直接返回。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


def clean_shop20(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入列: json, time-scraped
    输出: part_number, shop_name(=毎日買取), price_new, recorded_at
    """
    validate_columns(df, ["json", "time-scraped"],
                     cleaner_name="shop20", shop_name="毎日買取")
    return clean_with_jan_matching(
        df,
        cleaner_name="shop20",
        shop_name="毎日買取",
        iter_records_fn=_iter_records_shop20,
        price_extractor_fn=_price_passthrough,
    )
