# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import pandas as pd

from .tradein_cleaner import parse_tradein_uploaded
from .color_norm import normalize_color  # (canon, is_all)

# —— 规则：哪些状态视为“新品/未开封” —— #
NEW_STATUS_RE = re.compile(r"(新品|未開封|未开封|new(?!er)|sealed|unopened|未使用)", re.I)


def _parse_price_int(v) -> int | None:
    """把 '¥105,000' / '105000' / '10.5万' 等转为 int（日元）"""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if "万" in s:
        m = re.search(r"([\d\.]+)\s*万", s)
        base = float(m.group(1)) if m else 0.0
        tail = 0
        m2 = re.search(r"万\s*([0-9,]+)", s)
        if m2:
            tail = int(re.sub(r"[^\d]", "", m2.group(1)))
        return int(base * 10000 + tail)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def _collect_prices(row: Dict[str, Any]) -> list[int]:
    """收集该行所有可能的价格列"""
    prices: list[int] = []
    for col in ("价格", "价格2", "价格3", "price", "price1", "price2", "price3"):
        p = _parse_price_int(row.get(col))
        if p is not None:
            prices.append(p)
    return sorted(set(prices)) if prices else []


def _parse_capacity_gb(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", text, re.I)
    if m:
        return int(float(m.group(1)) * 1024)
    m = re.search(r"(\d{2,4})\s*GB", text, re.I)
    if m:
        return int(m.group(1))
    return None


def _normalize_model(text: str | None) -> str:
    """去掉容量/噪声，保留“iPhone 16 Pro Max”主体"""
    if not text:
        return ""
    t = re.sub(r"\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB", "", text, flags=re.I)
    t = re.sub(r"SIMフリー|未開封|新品|中古|（.*?）|\(.*?\)", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _safe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    try:
        return pd.concat(dfs, ignore_index=True)
    except Exception:
        base_cols = set().union(*[set(df.columns) for df in dfs])
        aligned = []
        for df in dfs:
            for c in base_cols:
                if c not in df.columns:
                    df[c] = None
            aligned.append(df[list(base_cols)])
        return pd.concat(aligned, ignore_index=True)


def clean_and_aggregate_tradein(
    file_blobs: List[Tuple[bytes, str]],
) -> Dict[str, Any]:
    """
    输入：[(file_bytes, filename), ...]
    输出：
    {
      "rows_total": int,
      "errors": [{"file":..., "error":...}, ...],
      "records": [  # 聚合后的业务记录（写库前的“清洗产物”）
        {
          "key": "pn:<pn>||<shop>" 或 "mc:<model>|<cap_gb>||<shop>",
          "shop_name": "...",
          "price_new": 100000,   # 可能为 None
          "price_grade_a": 98000,# 可能为 None
          "price_grade_b": 92000,# 可能为 None
          "meta": {
            "pn": "...",
            "jan": "13位数字或空",
            "model_name": "...(规范化后)",
            "capacity_gb": 256 或 None,
            "color_raw": "...",
            "color_canon": "...(统一映射)",
            "color_any": True/False
          }
        }, ...
      ],
      "preview": [前10条records的摘要]
    }
    """
    cleaned: list[pd.DataFrame] = []
    errors: list[dict] = []

    # 1) 逐文件清洗为统一列
    for data, name in file_blobs:
        try:
            df = parse_tradein_uploaded(data, name)
            cleaned.append(df)
        except Exception as e:
            errors.append({"file": name, "error": str(e)})

    if not cleaned:
        return {"rows_total": 0, "errors": errors, "records": [], "preview": []}

    df_merged = _safe_concat(cleaned)
    rows_total = int(df_merged.shape[0])

    # 2) 聚合：有 PN 用 pn||shop，无 PN 用 model+capacity||shop
    agg = defaultdict(lambda: {
        "shop_name": "",
        "prices": {"new": set(), "others": set()},
        "meta": {"pn":"", "jan":"", "model_name":"", "capacity_gb": None,
                 "color_raw":"", "color_canon":"", "color_any": False},
    })

    for row in df_merged.to_dict("records"):
        pn   = (row.get("Part_number") or "").strip()
        shop = (row.get("店铺名") or "").strip()
        jan  = re.sub(r"\D", "", str(row.get("JAN") or ""))

        model_raw = row.get("iphone型号") or ""
        model_norm = _normalize_model(model_raw)
        cap_gb = _parse_capacity_gb(model_raw)

        color_raw = (row.get("颜色") or "").strip()
        canon_color, is_all = normalize_color(color_raw)

        status_txt = (row.get("状态") or "").strip()
        prices = _collect_prices(row)
        if not prices:
            continue

        # 分组键
        if pn:
            key = f"pn:{pn}||{shop}"
        else:
            key = f"mc:{model_norm}|{cap_gb}||{shop}"

        slot = agg[key]
        slot["shop_name"] = shop

        # 元信息
        meta = slot["meta"]
        if not meta["pn"] and pn: meta["pn"] = pn
        if not meta["jan"] and len(jan) == 13: meta["jan"] = jan
        if not meta["model_name"]: meta["model_name"] = model_norm
        if meta["capacity_gb"] is None: meta["capacity_gb"] = cap_gb
        if not meta["color_raw"] and color_raw: meta["color_raw"] = color_raw
        if not meta["color_canon"] and canon_color: meta["color_canon"] = canon_color
        meta["color_any"] = meta["color_any"] or is_all or (color_raw == "")

        # 价格分类：新品/未开封 → new；其他 → others
        if NEW_STATUS_RE.search(status_txt):
            for p in prices: slot["prices"]["new"].add(p)
        else:
            for p in prices: slot["prices"]["others"].add(p)

    # 3) 计算三档价 & 构造 records
    records: list[dict] = []
    preview: list[dict] = []
    for key, val in agg.items():
        new_list = sorted(val["prices"]["new"] or set(), reverse=True)
        oth_list = sorted(val["prices"]["others"] or set(), reverse=True)

        price_new = new_list[0] if new_list else None
        price_a = oth_list[0] if len(oth_list) >= 1 else None
        price_b = oth_list[1] if len(oth_list) >= 2 else None

        rec = {
            "key": key,
            "shop_name": val["shop_name"],
            "price_new": price_new,
            "price_grade_a": price_a,
            "price_grade_b": price_b,
            "meta": val["meta"],
        }
        records.append(rec)

        if len(preview) < 10:
            pv = {
                "key": key, "shop_name": val["shop_name"],
                "price_new": price_new, "price_grade_a": price_a, "price_grade_b": price_b,
                **val["meta"],  # pn/jan/model_name/capacity_gb/color_raw/color_canon/color_any
            }
            preview.append(pv)

    return {
        "rows_total": rows_total,
        "errors": errors,
        "records": records,
        "preview": preview,
    }
