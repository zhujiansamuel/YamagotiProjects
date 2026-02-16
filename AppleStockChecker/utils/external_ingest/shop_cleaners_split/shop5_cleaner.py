# -*- coding: utf-8 -*-
"""
shop5 统一清洗器（森森買取 · shop5_1～shop5_4）

shop5_1～shop5_4 为同一店铺不同数据源变体，逻辑相同，统一在此实现。
通过多注册方式供 registry 映射 shop5_1, shop5_2, shop5_3, shop5_4。
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import pandas as pd

from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import extract_price_yen, assemble_output_df, validate_columns, _load_jan_to_pn_from_csv, log_cleaner_start

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


def _clean_shop5_soramimi(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """
    森森買取统一清洗逻辑。

    输入列：price, data, name, time-scraped
    输出列：part_number, shop_name, price_new, recorded_at
    shop_name 固定 '森森買取'
    """
    # 必要列检查
    validate_columns(df, ["price", "data", "name", "time-scraped"],
                     cleaner_name=f"shop5-{variant}", shop_name="森森買取")

    log_cleaner_start(logger, cleaner_name=f"shop5-{variant}", shop_name="森森買取", input_rows=len(df))

    # 1) 过滤掉 name 含"中古"的行
    src = df.copy()
    mask_keep = ~src["name"].astype(str).str.contains("中古", na=False)
    src = src[mask_keep]

    # 2) 跳过 time-scraped 为空的行
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 3) 载入 JAN -> PN 映射
    jan_to_pn = _load_jan_to_pn_from_csv()

    # 4) 逐列解析
    jan_series = src["data"].map(_extract_jan_from_data)
    pn_series = jan_series.map(lambda j: jan_to_pn.get(j) if j and re.fullmatch(r"\d{13}", j) else None)

    price_new = src["price"].map(extract_price_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 5) 组装结果：必须有 PN & 价格
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_series.iat[i]
        p = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "森森買取",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = assemble_output_df(rows, coerce_price=False)
    return out


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
