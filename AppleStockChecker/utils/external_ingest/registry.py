# AppleStockChecker/utils/external_ingest/registry.py
from __future__ import annotations
from typing import Callable, Dict
import pandas as pd
from .base_cleaners import CLEANERS, Cleaner

def get_cleaner(name: str) -> Cleaner:
    if name not in CLEANERS:
        raise KeyError(f"未注册的清洗器: {name}")
    return CLEANERS[name]

def run_cleaner(name: str, df: pd.DataFrame) -> pd.DataFrame:
    cleaner = get_cleaner(name)
    return cleaner(df)
