# AppleStockChecker/utils/external_ingest/registry.py
from __future__ import annotations
from typing import Dict, Protocol
import pandas as pd

from .shop_cleaners_split.shop1_cleaner import clean_shop1
from .shop_cleaners_split.shop2_cleaner import clean_shop2
from .shop_cleaners_split.shop3_cleaner import clean_shop3
from .shop_cleaners_split.shop4_cleaner import clean_shop4
from .shop_cleaners_split.shop5_cleaner import clean_shop5_1, clean_shop5_2, clean_shop5_3, clean_shop5_4
from .shop_cleaners_split.shop6_cleaner import clean_shop6_1, clean_shop6_2, clean_shop6_3, clean_shop6_4
from .shop_cleaners_split.shop7_cleaner import clean_shop7
from .shop_cleaners_split.shop8_cleaner import clean_shop8
from .shop_cleaners_split.shop9_cleaner import clean_shop9
from .shop_cleaners_split.shop10_cleaner import clean_shop10
from .shop_cleaners_split.shop11_cleaner import clean_shop11
from .shop_cleaners_split.shop12_cleaner import clean_shop12
from .shop_cleaners_split.shop13_cleaner import clean_shop13
from .shop_cleaners_split.shop14_cleaner import clean_shop14
from .shop_cleaners_split.shop15_cleaner import clean_shop15
from .shop_cleaners_split.shop16_cleaner import clean_shop16
from .shop_cleaners_split.shop17_cleaner import clean_shop17
from .shop_cleaners_split.shop18_cleaner import clean_shop18
from .shop_cleaners_split.shop20_cleaner import clean_shop20


class Cleaner(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...


CLEANERS = {
    "shop1":  clean_shop1,
    "shop2":  clean_shop2,
    "shop3":  clean_shop3,
    "shop4":  clean_shop4,
    "shop5_1":  clean_shop5_1,
    "shop5_2":  clean_shop5_2,
    "shop5_3":  clean_shop5_3,
    "shop5_4":  clean_shop5_4,
    "shop6_1":  clean_shop6_1,
    "shop6_2":  clean_shop6_2,
    "shop6_3":  clean_shop6_3,
    "shop6_4":  clean_shop6_4,
    "shop7":  clean_shop7,
    "shop8":  clean_shop8,
    "shop9":  clean_shop9,
    "shop10":  clean_shop10,
    "shop11":  clean_shop11,
    "shop12":  clean_shop12,
    "shop13":  clean_shop13,
    "shop14":  clean_shop14,
    "shop15":  clean_shop15,
    "shop16":  clean_shop16,
    "shop17":  clean_shop17,
    "shop18":  clean_shop18,
    "shop20":  clean_shop20,

}


# def get_cleaner(name: str) -> Cleaner:
#     if name not in CLEANERS:
#         raise KeyError(f"未注册的清洗器: {name}")
#     return CLEANERS[name]

# def run_cleaner(name: str, df: pd.DataFrame) -> pd.DataFrame:
#     cleaner = get_cleaner(name)
#     return cleaner(df)



def has_cleaner(name: str) -> bool:
    if not name:
        return False
    return name.replace("-", "_") in CLEANERS

def get_cleaner(name: str) -> Cleaner:
    """真正取出 cleaner 的函数，后续如果需要用得到"""
    if not name:
        raise KeyError("未指定清洗器名称")
    normalized_name = name.replace("-", "_")
    if normalized_name not in CLEANERS:
        raise KeyError(f"未注册的清洗器: {name} (规范化名: {normalized_name})")
    return CLEANERS[normalized_name]



def run_cleaner(shop_key: str, df):
    from AppleStockChecker.utils.external_ingest.cleaner_tools import dedupe_output_keep_latest

    if not shop_key:
        raise KeyError("未指定清洗器名称")
    normalized_key = shop_key.replace("-", "_")
    if normalized_key not in CLEANERS:
        raise KeyError(f"未注册的清洗器: {shop_key} (规范化名: {normalized_key})")

    out = CLEANERS[normalized_key](df)
    if isinstance(out, pd.DataFrame) and not out.empty:
        out = dedupe_output_keep_latest(out)
    return out


