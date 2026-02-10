from __future__ import annotations
from typing import Dict, Optional, List, Iterable, Union
from ..helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _load_iphone17_info_df_from_db, _extract_jan_digits, _build_jan_map
import re
import json
import pandas as pd
from datetime import datetime
import pytz
import time

def _iter_records(df: pd.DataFrame):
    """
    产出规范化记录：{"JAN":..., "price":..., "time-scraped": ...}
    适配两种输入：
      A) 直列：JAN, price, time-scraped
      B) JSON 列：json（对象/数组/带 data 的对象），同行的 time-scraped 为默认时间
         - 兼容字段别名：jancode / goodsPrice / time_scraped / timestamp / keywords(兜底提取 JAN)
    """
    cols = {c.lower(): c for c in df.columns}

    # A) 直列
    if all(k in cols for k in ["jan", "price", "time-scraped"]):
        JAN_col, price_col, ts_col = cols["jan"], cols["price"], cols["time-scraped"]
        for _, row in df.iterrows():
            yield {"JAN": row.get(JAN_col), "price": row.get(price_col), "time-scraped": row.get(ts_col)}
        return

    # B) JSON 列
    json_col = cols.get("json")
    ts_col = cols.get("time-scraped") or cols.get("time_scraped")
    if not json_col:
        return

    for _, row in df.iterrows():
        default_ts = row.get(ts_col)
        cell = row.get(json_col)
        parsed = None

        if isinstance(cell, (dict, list)):
            parsed = cell
        elif isinstance(cell, str) and cell.strip():
            s = cell.strip().lstrip("\ufeff")
            # CSV 风格的 "" → "（若存在）
            if s.count('""') and not s.count('\\"'):
                s = s.replace('""', '"')
            try:
                parsed = json.loads(s)
            except Exception:
                continue
        else:
            continue

        # 统一拉平成若干对象
        items: List[dict] = []
        if isinstance(parsed, dict):
            items = [x for x in parsed.get("data", [parsed]) if isinstance(x, dict)]
        elif isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]

        for it in items:
            jan = it.get("JAN") or it.get("jan") or it.get("jancode") or it.get("jAN")
            if not jan:
                jan = it.get("keywords")  # 兜底：从文字里抽出 JAN
            price = it.get("price") or it.get("goodsPrice") or it.get("Price")
            ts = it.get("time-scraped") or it.get("time_scraped") or it.get("timestamp") or default_ts
            yield {"JAN": jan, "price": price, "time-scraped": ts}

def clean_shop1(df: pd.DataFrame) -> pd.DataFrame:
    """
    以 JAN 映射 part_number；price -> price_new；time-scraped -> recorded_at。
    shop_name 固定为「買取商店」。
    仅输出 _load_iphone17_info_df_from_db() 中存在的机型。
    """
    # 准备 JAN->PN 映射
    info_df = _load_iphone17_info_df_from_db()
    jan_map = _build_jan_map(info_df)

    rows: List[dict] = []

    for rec in _iter_records(df):
        jan = _extract_jan_digits(rec.get("JAN"))

        if not jan:
            continue
        pn = jan_map.get(jan)

        if not pn:
            continue

        price_val = rec.get("price")
        # 既支持数值，也支持 "181,500" / "181500円"
        price_new = to_int_yen(price_val)
        if price_new is None:
            continue

        recorded_at = parse_dt_aware(rec.get("time-scraped"))

        rows.append({
            "part_number": str(pn),
            "shop_name": "買取商店",
            "price_new": int(price_new),
            "recorded_at": recorded_at,
        })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    # print("+++++++++++++++out",out)
    return out
