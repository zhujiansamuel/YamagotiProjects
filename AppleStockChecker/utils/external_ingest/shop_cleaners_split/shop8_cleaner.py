from __future__ import annotations
"""
shop8 清洗器 — 買取wiki

  原始 DataFrame（機種名 / 未開封 / time-scraped）
    │
    ├─ _extract_part_number()    ← Step 1: 型番抽取（型番: XXXJ/A or PN 正则）
    ├─ extract_price_yen()       ← Step 2: 价格解析（cleaner_tools 统一）
    ├─ parse_dt_aware()          ← Step 3: 时间解析
    └─ clean_shop8()             ← Step 4: 主函数，输出 part_number / price_new / recorded_at
"""
from typing import Optional, List
import logging
import re
import time

import pandas as pd

from ...external_ingest.cleaner_tools import parse_dt_aware
from ..cleaner_tools import normalize_text_basic, extract_price_yen, validate_columns, log_cleaner_start, log_cleaner_complete, assemble_output_df

logger = logging.getLogger(__name__)

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
    start_time = time.time()
    log_cleaner_start(logger, cleaner_name="shop8", shop_name="買取wiki", input_rows=len(df))
    # 列名容错：有些抓取器可能用不同大小写或空白
    # 这里统一抓关键列
    col_model = "機種名"
    col_price_new = "未開封"
    col_time = "time-scraped"

    validate_columns(df, [col_model, col_price_new, col_time],
                     cleaner_name="shop8", shop_name="買取wiki")

    if df.empty:
        log_cleaner_complete(logger, cleaner_name="shop8", shop_name="買取wiki", input_rows=len(df), output_records=0, start_time=start_time)
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 解析
    part_numbers = df[col_model].map(_extract_part_number)
    price_new = df[col_price_new].map(extract_price_yen)
    recorded_at = df[col_time].map(parse_dt_aware)

    rows: List[dict] = []
    for i in range(len(df)):
        pn = part_numbers.iat[i]
        p = price_new.iat[i]
        ts = recorded_at.iat[i]
        rows.append({
            "part_number": pn,
            "shop_name": "買取wiki",
            "price_new": p,
            "recorded_at": ts,
        })

    out = assemble_output_df(rows, coerce_price=False)
    log_cleaner_complete(logger, cleaner_name="shop8", shop_name="買取wiki", input_rows=len(df), output_records=len(out), start_time=start_time)
    return out
