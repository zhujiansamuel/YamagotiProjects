from __future__ import annotations
"""
shop8 清洗器 — 買取wiki

  原始 DataFrame（機種名 / 未開封 / time-scraped）
    │
    ├─ _extract_part_number()    ← Step 1: 型番抽取（型番: XXXJ/A or PN 正则）
    ├─ to_int_yen()              ← Step 2: 价格解析
    ├─ parse_dt_aware()          ← Step 3: 时间解析
    └─ clean_shop8()             ← Step 4: 主函数，输出 part_number / price_new / recorded_at
"""
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import normalize_text_basic
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Tuple
from urllib.parse import urlparse
from typing import Dict, Optional, List, Iterable, Union
import os, re, json, pathlib
from datetime import datetime
import pytz
import time

PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")

def _extract_part_number(text: str) -> str | None:
    t = normalize_text_basic(text)
    # 1) 优先：显式 "型番: XXXXXJ/A"
    m = re.search(r"型番[:：]\s*([A-Z0-9]{4,6}\d{0,3}J/A)\b", t)
    if m:
        return m.group(1)
    # 2) 兜底：全文 PN 正则
    m2 = PN_REGEX.search(t)
    return m2.group(0) if m2 else None

def clean_shop8(df: pd.DataFrame) -> pd.DataFrame:
    print("shop8:買取wiki---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 列名容错：有些抓取器可能用不同大小写或空白
    # 这里统一抓关键列
    col_model = "機種名"
    col_price_new = "未開封"
    col_time = "time-scraped"

    for need in (col_model, col_price_new, col_time):
        if need not in df.columns:
            raise ValueError(f"shop8 清洗器缺少必要列: {need}")

    # 解析
    part_numbers = df[col_model].map(_extract_part_number)
    price_new = df[col_price_new].map(to_int_yen)
    recorded_at = df[col_time].map(parse_dt_aware)

    out = pd.DataFrame({
        "part_number": part_numbers,
        "shop_name": "買取wiki",
        "price_new": price_new,
        "recorded_at": recorded_at,
    })

    # 丢掉关键字段缺失的行（pn 或 price）
    out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)

    # 确保类型（避免 pandas 的 NA 类型导致后续 int() 失败）
    out["part_number"] = out["part_number"].astype(str)
    return out
