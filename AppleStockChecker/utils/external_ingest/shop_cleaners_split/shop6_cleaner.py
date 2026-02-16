# -*- coding: utf-8 -*-
"""
shop6 统一清洗器（買取ルデヤ · shop6_1～shop6_4）

shop6_1～shop6_4 为同一店铺不同数据源变体，逻辑相同，统一在此实现。
通过多注册方式供 registry 映射 shop6_1, shop6_2, shop6_3, shop6_4。
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import pandas as pd

from ...external_ingest.cleaner_tools import parse_dt_aware
from ..cleaner_tools import extract_price_yen, assemble_output_df, validate_columns, _load_iphone17_info_df_from_db, _build_jan_map, log_cleaner_start

logger = logging.getLogger(__name__)

_PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")


def _extract_pn_from_text(text: object) -> Optional[str]:
    if text is None:
        return None
    s = str(text).replace("\u3000", " ")
    m = _PN_REGEX.search(s)
    return m.group(0) if m else None


def _clean_shop6_kaidoruya(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """
    買取ルデヤ统一清洗逻辑。

    输入列：data7, phone, data8, time-scraped
    输出列：part_number, shop_name, price_new, recorded_at
    shop_name 固定 '買取ルデヤ'
    """
    # 必要列检查
    validate_columns(df, ["data7", "phone", "data8", "time-scraped"],
                     cleaner_name=f"shop6-{variant}", shop_name="買取ルデヤ")

    log_cleaner_start(logger, cleaner_name=f"shop6-{variant}", shop_name="買取ルデヤ", input_rows=len(df))

    # 跳过 time-scraped 为空的行
    src = df.copy()
    mask_time = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_from_db()
    jan_to_pn = _build_jan_map(info_df)

    # 解析列：JAN 从 phone；PN 兜底从 data8
    jan_series = src["phone"].astype(str).str.replace(r"[^\d]", "", regex=True)
    pn_by_jan = jan_series.map(lambda j: jan_to_pn.get(j) if re.fullmatch(r"\d{13}", j or "") else None)
    pn_fallback = src["data8"].map(_extract_pn_from_text)

    price_new = src["data7"].map(extract_price_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 组装：优先 JAN→PN；无则 data8 提取；再无则丢弃
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_by_jan.iat[i] or pn_fallback.iat[i]
        p = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "買取ルデヤ",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = assemble_output_df(rows, coerce_price=False)
    return out


def _make_shop6_cleaner(variant: str):
    """返回绑定 variant 的清洗器，供 registry 多注册。"""
    def _cleaner(df: pd.DataFrame) -> pd.DataFrame:
        return _clean_shop6_kaidoruya(df, variant)
    return _cleaner


# 供 registry 直接导入的四个清洗器
clean_shop6_1 = _make_shop6_cleaner("1")
clean_shop6_2 = _make_shop6_cleaner("2")
clean_shop6_3 = _make_shop6_cleaner("3")
clean_shop6_4 = _make_shop6_cleaner("4")
