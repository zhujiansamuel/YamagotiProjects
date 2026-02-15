from __future__ import annotations
"""
shop18 清洗器 — 買取オク

  原始 DataFrame（jan / type / price / time-scraped）
    │
    ├─ _extract_jan_digits()       ← Step 1: JAN 提取（cleaner_tools）
    ├─ _build_jan_map()             ← Step 2: JAN → part_number 映射（cleaner_tools）
    ├─ _match_by_type()             ← Step 3: JAN 无法匹配时 type 回退（model/cap/color）
    ├─ to_int_yen()                 ← Step 4: 价格解析
    └─ clean_shop18()               ← Step 5: 主函数，输出 part_number / price_new / recorded_at
"""
from typing import Dict, Optional, List, Tuple
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _normalize_model_generic, _load_iphone17_info_df_from_db, _extract_jan_digits, _build_jan_map, assemble_output_df, validate_columns
import re
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
import pytz
import time

SHOP_NAME_OVERRIDE: Optional[str] = "買取オク"

def _match_by_type(type_text: str, info_df: pd.DataFrame) -> Optional[str]:
    """
    当 JAN 无法匹配时，根据 `type` 文本（如 'iPhone 17 Pro 512GB ディープブルー'）
    用 (model_norm, capacity_gb, color_norm) 回退匹配到 part_number。
    """
    if not type_text:
        return None
    txt = str(type_text).replace("\u3000", " ").replace("\xa0", " ").strip()
    model_norm = _normalize_model_generic(txt)
    cap_gb = _parse_capacity_gb(txt)
    if not model_norm or pd.isna(cap_gb):
        return None
    cap_gb = int(cap_gb)

    # 在该 (model, cap) 下，寻找哪个颜色名出现在 type 文本中
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    cand = df[(df["model_name_norm"] == model_norm) & (df["capacity_gb"] == cap_gb)]
    if cand.empty:
        return None

    # 直接用 "颜色原文子串" 命中（多数站点颜色在文案中能直接找到）
    for _, r in cand.iterrows():
        color_raw = str(r["color"])
        if color_raw and color_raw in txt:
            return str(r["part_number"])

    # 若未命中且候选仅有 1 个颜色，直接返回（保底）
    if len(cand) == 1:
        return str(cand.iloc[0]["part_number"])

    return None

def clean_shop18(df: pd.DataFrame) -> pd.DataFrame:
    print("shop18:買取オク---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop18.csv):
      - jan: 如 'JAN: 4549995648300'
      - type: 如 'iPhone 17 Pro  256GB ディープブルー'
      - price: '¥180,500' / '問い合わせ' 等
      - time-scraped
      - web-scraper-start-url: 用于默认派生 shop_name（域名）
    输出：
      - part_number, shop_name, price_new, recorded_at
    仅输出出现在 _load_iphone17_info_df_from_db() 的机型。
    """
    validate_columns(df, ["jan", "type", "price", "time-scraped"],
                     cleaner_name="shop18", shop_name="買取オク")

    info_df = _load_iphone17_info_df_from_db()
    jan_map = _build_jan_map(info_df)

    # 为回退匹配准备（按 (model, cap) 切片）
    rows: List[dict] = []

    for _, row in df.iterrows():
        # 价格（无价/“問い合わせ”跳过）
        price_new = to_int_yen(row.get("price"))
        if price_new is None:
            continue
        price_new = int(price_new)

        # 记录时间
        recorded_at = parse_dt_aware(row.get("time-scraped"))

        # 店名（若未覆盖，则用域名）
        if SHOP_NAME_OVERRIDE:
            shop_name = SHOP_NAME_OVERRIDE
        else:
            start_url = str(row.get("web-scraper-start-url") or "")
            netloc = urlparse(start_url).netloc or "shop18"
            shop_name = netloc

        # 先用 JAN 直接匹配
        jan_digits = _extract_jan_digits(row.get("jan"))
        part_number: Optional[str] = None
        if jan_digits and jan_digits in jan_map:
            part_number = jan_map[jan_digits]
        else:
            # 回退：用 type 匹配 (model, cap, color)
            part_number = _match_by_type(row.get("type"), info_df)

        if not part_number:
            # 无法匹配到信息表 → 跳过
            continue

        rows.append({
            "part_number": str(part_number),
            "shop_name": shop_name,
            "price_new": price_new,
            "recorded_at": recorded_at,
        })

    out = assemble_output_df(rows)
    return out
