from __future__ import annotations
"""
shop5_3 清洗器 — JAN/PN 解析子模块
  原始 data 列 → _extract_jan_from_data() / _extract_pn_from_text() → JAN or PN
  供 shop5 主清洗流程调用
"""
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import extract_price_yen
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

def _resolve_info_path() -> Path:
    try:
        from django.conf import settings
        p = getattr(settings, "EXTERNAL_IPHONE17_INFO_PATH", None)
        if p:
            return Path(p)
    except Exception:
        pass
    envp = os.getenv("IPHONE17_INFO_CSV")
    if envp and Path(envp).exists():
        return Path(envp)
    return Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv"

def _load_jan_to_pn() -> Dict[str, str]:
    """
    返回 { jan(13位字符串) : part_number } 的字典。
    若 info 文件没有 jan 列，则返回空字典（后续走 data8 的 PN 兜底）。
    """
    path = _resolve_info_path()
    if not path.exists():
        # 没找到映射文件时，仍允许仅走 data8 的 PN 兜底
        return {}
    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(path), re.I):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    if "part_number" not in df.columns:
        # 没有 PN 列，无法映射
        return {}

    # 允许 info 表没有 jan；有则清洗为 13 位
    if "jan" in df.columns:
        df = df.copy()
        df["jan"] = df["jan"].astype(str).str.replace(r"[^\d]", "", regex=True)
        df = df[df["jan"].str.fullmatch(r"\d{13}", na=False)]
        mapping = dict(zip(df["jan"].astype(str), df["part_number"].astype(str)))
        return mapping
    return {}

def _extract_jan_from_data(x: object) -> Optional[str]:
    """
    从 'data' 文本里抽取 13 位 JAN（例如 'JAN:4549995648300'）
    """
    if x is None:
        return None
    s = str(x)
    # 优先匹配 'JAN:XXXXXXXXXXXXX'
    m = re.search(r"JAN[:：]?\s*(\d{13})", s)
    if m:
        return m.group(1)
    # 兜底：任意 13 位数字
    m2 = re.search(r"\b(\d{13})\b", s)
    return m2.group(1) if m2 else None

def clean_shop5_3(df: pd.DataFrame) -> pd.DataFrame:
    print("shop5-3:森森買取---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入表头：web-scraper-order, web-scraper-start-url, pagination-selector, price, data, name, (time-scraped)
    要求：删除 name 含 '中古' 的行；通过 data 的 JAN 定位 PN；price -> price_new；time-scraped -> recorded_at
    shop_name 固定 '森森買取'
    """
    # 必要列检查（time-scraped 在问题描述中是必须的）
    need_cols = ["price", "data", "name", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop5-3 清洗器缺少必要列：{c}")

    # 1) 过滤掉 name 含“中古”的行
    src = df.copy()
    mask_keep = ~src["name"].astype(str).str.contains("中古", na=False)
    src = src[mask_keep]

    # 2) 跳过 time-scraped 为空的行（避免落库时间为空）
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 3) 载入 JAN -> PN 映射（允许为空字典；若 info 表无 jan 列则无法映射 PN）
    jan_to_pn = _load_jan_to_pn()

    # 4) 逐列解析
    jan_series = src["data"].map(_extract_jan_from_data)
    pn_series  = jan_series.map(lambda j: jan_to_pn.get(j) if j and re.fullmatch(r"\d{13}", j) else None)

    price_new   = src["price"].map(extract_price_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 5) 组装结果：必须有 PN & 价格
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_series.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "森森買取",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out
