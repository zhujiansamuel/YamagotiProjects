# -*- coding: utf-8 -*-
"""
shop6 统一清洗器（買取ルデヤ · shop6_1～shop6_4）

shop6_1～shop6_4 为同一店铺不同数据源变体，逻辑相同，统一在此实现。
通过多注册方式供 registry 映射 shop6_1, shop6_2, shop6_3, shop6_4。

使用公共模板 clean_with_jan_matching。
JAN 从 phone 列提取，fallback 从 data8 列正则提取 PN。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from ..cleaner_tools import (
    clean_with_jan_matching,
    extract_price_yen,
    validate_columns,
)

logger = logging.getLogger(__name__)

_PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")


def _extract_pn_from_text(text: object) -> Optional[str]:
    if text is None:
        return None
    s = str(text).replace("\u3000", " ")
    m = _PN_REGEX.search(s)
    return m.group(0) if m else None


def _iter_records_shop6(df: pd.DataFrame):
    """
    产出规范化记录。
    输入列：data7(价格), phone(JAN), data8(PN兜底), time-scraped
    JAN 从 phone 列提取；若匹配失败，由 fallback_match_fn 从 data8 提取 PN
    过滤：time-scraped 为空的行跳过
    """
    for _, row in df.iterrows():
        ts = row.get("time-scraped")
        if ts is None or str(ts).strip() == "":
            continue
        # phone 列中提取纯数字作为 JAN
        phone_raw = str(row.get("phone", ""))
        jan = re.sub(r"[^\d]", "", phone_raw)
        if not re.fullmatch(r"\d{13}", jan):
            jan = None
        yield {
            "JAN": jan,
            "price": row.get("data7"),
            "time-scraped": ts,
            "_data8": row.get("data8"),  # 传递给 fallback
        }


def _fallback_match_shop6(rec: dict, info_df) -> Optional[str]:
    """JAN 无法匹配时，从 data8 列提取 PN 直接返回。"""
    return _extract_pn_from_text(rec.get("_data8"))


def _clean_shop6_kaidoruya(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """買取ルデヤ统一清洗逻辑。"""
    validate_columns(df, ["data7", "phone", "data8", "time-scraped"],
                     cleaner_name=f"shop6-{variant}", shop_name="買取ルデヤ")
    return clean_with_jan_matching(
        df,
        cleaner_name=f"shop6-{variant}",
        shop_name="買取ルデヤ",
        iter_records_fn=_iter_records_shop6,
        fallback_match_fn=_fallback_match_shop6,
        coerce_price=False,
    )


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
