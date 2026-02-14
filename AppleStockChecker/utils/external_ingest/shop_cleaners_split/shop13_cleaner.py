from __future__ import annotations
"""
shop13 清洗器 — 家電市場

  原始 DataFrame（新品価格 / 買取商品2 / time-scraped）
    │
    ├─ _normalize_model_generic()  ← Step 1: 机型归一化（cleaner_tools）
    ├─ _parse_capacity_gb()        ← Step 2: 容量解析（cleaner_tools）
    ├─ extract_price_yen()         ← Step 3: 价格提取（cleaner_tools）
    ├─ _load_iphone17_info_df_from_db()  ← Step 4: 机型信息（cleaner_tools）
    └─ clean_shop13()              ← Step 5: 主函数，输出 part_number / price_new / recorded_at
"""
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _normalize_model_generic, _load_iphone17_info_df_from_db, extract_price_yen
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

def clean_shop13(df: pd.DataFrame) -> pd.DataFrame:
    print("shop13:家電市場---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入列（来自 shop13.csv）：
      - 「新品価格」: 价格（可能含 '円'、'¥'、'～'、'万' 等）
      - 「買取商品2」: 含 机种名 + 容量 (+ 颜色等修饰)
      - 「time-scraped」: 抓取时间（输出 recorded_at）

    输出 DataFrame 列：
      - part_number, shop_name, price_new, recorded_at

    规则：
      - shop_name 固定为「家電市場」
      - 机种名统一用 _normalize_model_generic 归一（如 'iPhone 17 Pro Max' / 'iPhone Air'）
      - 容量用 _parse_capacity_gb 解析（GB/TB → 以 GB 计）
      - 通过 _load_iphone17_info_df_from_db() 对应（机种，容量）取**所有颜色**的 PN 列表并展开为多行
      - 仅输出在信息表中能匹配到的机型与容量
      - recorded_at = parse_dt_aware(time-scraped)
    """
    # --- 必要列检查 ---
    need_cols = ["新品価格", "買取商品2", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop13 清洗器缺少必要列：{c}")

    # --- 载入 iPhone17 信息（含颜色），并补充归一化机种名 ---
    info_df = _load_iphone17_info_df_from_db().copy()
    # 预期列：part_number, model_name, capacity_gb, color
    info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
    info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")

    # （model_name_norm, capacity_gb）→ 该组合下的所有颜色的 PN 列表
    groups = (
        info_df.groupby(["model_name_norm", "capacity_gb"])["part_number"]
        .apply(list).to_dict()
    )

    # --- 源数据解析 ---
    model_norm = df["買取商品2"].map(_normalize_model_generic)
    cap_gb     = df["買取商品2"].map(_parse_capacity_gb)

    price_new   = df["新品価格"].map(extract_price_yen)
    recorded_at = df["time-scraped"].map(parse_dt_aware)

    # --- 展开为行 ---
    rows: List[dict] = []
    for i in range(len(df)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p = price_new.iat[i]
        t = recorded_at.iat[i]

        # 关键字段缺失则跳过
        if not m or pd.isna(c) or (p is None):
            continue

        pn_list = groups.get((m, int(c)), [])
        if not pn_list:
            # 信息表中没有对应（机种, 容量）记录 → 跳过
            continue

        # 注意：按要求对同一机种+容量下的「所有颜色」展开
        for pn in pn_list:
            rows.append({
                "part_number": str(pn),
                "shop_name": "家電市場",
                "price_new": int(p),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out
