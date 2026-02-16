# -*- coding: utf-8 -*-
"""
shop5 统一清洗器（森森買取 · shop5_1～shop5_4）

shop5_1～shop5_4 为同一店铺不同数据源变体，逻辑相同，统一在此实现。
通过多注册方式供 registry 映射 shop5_1, shop5_2, shop5_3, shop5_4。

使用公共模板 clean_with_jan_matching。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from ..cleaner_tools import clean_with_jan_matching, validate_columns

logger = logging.getLogger(__name__)


def _extract_jan_from_data(x: object) -> Optional[str]:
    """
    从 'data' 文本里抽取 13 位 JAN（例如 'JAN:4549995648300'）
    """
    if x is None:
        return None
    s = str(x)
    m = re.search(r"JAN[:：]?\s*(\d{13})", s)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{13})\b", s)
    return m2.group(1) if m2 else None


def _iter_records_shop5(df: pd.DataFrame):
    """
    产出规范化记录。
    输入列：price, data, name, time-scraped
    过滤：排除 name 含"中古"的行、time-scraped 为空的行
    JAN 从 data 列提取
    """
    for _, row in df.iterrows():
        name_val = str(row.get("name", ""))
        if "中古" in name_val:
            continue
        ts = row.get("time-scraped")
        if ts is None or str(ts).strip() == "":
            continue
        jan = _extract_jan_from_data(row.get("data"))
        yield {
            "JAN": jan,
            "price": row.get("price"),
            "time-scraped": ts,
        }


def _clean_shop5_soramimi(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """森森買取统一清洗逻辑。"""
    validate_columns(df, ["price", "data", "name", "time-scraped"],
                     cleaner_name=f"shop5-{variant}", shop_name="森森買取")
    return clean_with_jan_matching(
        df,
        cleaner_name=f"shop5-{variant}",
        shop_name="森森買取",
        iter_records_fn=_iter_records_shop5,
        coerce_price=False,
    )


def _make_shop5_cleaner(variant: str):
    """返回绑定 variant 的清洗器，供 registry 多注册。"""
    def _cleaner(df: pd.DataFrame) -> pd.DataFrame:
        return _clean_shop5_soramimi(df, variant)
    return _cleaner


# 供 registry 直接导入的四个清洗器
clean_shop5_1 = _make_shop5_cleaner("1")
clean_shop5_2 = _make_shop5_cleaner("2")
clean_shop5_3 = _make_shop5_cleaner("3")
clean_shop5_4 = _make_shop5_cleaner("4")
