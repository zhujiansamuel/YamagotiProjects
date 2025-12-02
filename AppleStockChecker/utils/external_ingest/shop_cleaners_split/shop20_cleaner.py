from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
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

JAN_DIGITS_RE = re.compile(r"(\d{8,})")  # 抓取连续 8+ 位数字

def _extract_jan_digits(s: str) -> Optional[str]:
    if not s:
        return None
    m = JAN_DIGITS_RE.search(str(s))
    return m.group(1) if m else None

def _load_iphone17_info_df_for_shop20() -> pd.DataFrame:
    """
    读取 AppleStockChecker/data/iphone17_info.csv 或 settings / env 指定的路径。
    输出列：part_number, model_name_norm, capacity_gb
    """


    try:
        from django.conf import settings
        p = getattr(settings, "EXTERNAL_IPHONE17_INFO_PATH", None)
        if p:
            path = str(p)
        else:
            raise AttributeError
    except Exception:
        path = os.getenv("IPHONE17_INFO_CSV") or str(Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv")
    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError(f"未找到 iphone17_info：{pth}")

    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(pth), re.I):
        df = pd.read_excel(pth)
    else:
        df = pd.read_csv(pth, encoding="utf-8-sig")

    need = {"part_number", "model_name", "capacity_gb","color"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"iphone17_info 缺少必要列：{missing}")

    df = df.copy()
    # df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number","color"])
    return df[["part_number", "model_name", "capacity_gb","color","jan"]]

def _pick_info_jan_column_shop20(info_df: pd.DataFrame) -> Optional[str]:
    """在信息表中寻找 JAN 列名（大小写/写法容错）"""
    candidates = [c for c in info_df.columns
                  if str(c).strip().lower() in {"jan", "jan_code", "jancode"}]
    return candidates[0] if candidates else None

def _build_jan_to_pn_map_shop20(info_df: pd.DataFrame) -> Dict[str, str]:
    """
    从信息表构建 { jan_digits -> part_number }。
    若无 JAN 列，返回空映射（此站点要求用 jancode 匹配，建议信息表带 jan）。
    """
    jan_map: Dict[str, str] = {}
    jcol = _pick_info_jan_column_shop20(info_df)
    if not jcol:
        return jan_map
    for _, r in info_df.iterrows():
        jan_digits = _extract_jan_digits(r.get(jcol))
        pn = r.get("part_number")
        if jan_digits and pd.notna(pn):
            jan_map[str(jan_digits)] = str(pn)
    return jan_map

def _coerce_price(v) -> Optional[int]:
    """goodsPrice 既可能是数字也可能是字符串，统一转 int（日元）"""
    if v is None:
        return None
    if isinstance(v, (int, float)) and pd.notna(v):
        return int(round(float(v)))
    return to_int_yen(v)

def clean_shop20(df: pd.DataFrame) -> pd.DataFrame:
    print("shop20:毎日買取---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop20.csv):
      - json: 形如 {""success"":true,""data"":[...]} 的 JSON 文本（需先把 "" → "）
      - time-scraped: 抓取时间
    输出:
      - part_number, shop_name(=買取当番), price_new, recorded_at
    规则:
      - 对 json['data'] 的每个项，取 jancode → 在信息表中匹配 PN；取 goodsPrice → price_new
      - 无法解析/缺少 jancode 或 goodsPrice 的条目跳过
      - recorded_at 使用该行的 time-scraped
    """
    # 必要列检查
    for c in ["json", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop20 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop20()
    jan_map = _build_jan_to_pn_map_shop20(info_df)

    rows: List[dict] = []

    for _, row in df.iterrows():
        raw_json = row.get("json")
        if not isinstance(raw_json, str) or not raw_json.strip():
            continue

        # 将 CSV 内部双引号转为标准 JSON 引号
        # 例如 {""success"":true} -> {"success":true}
        s = raw_json.replace('""', '"').strip()

        try:
            payload = json.loads(s)
        except Exception:
            # 解析失败，尝试去掉可能的 BOM/不可见字符后再试
            s2 = s.lstrip("\ufeff").strip()
            try:
                payload = json.loads(s2)
            except Exception:
                continue

        data = payload.get("data")
        if not isinstance(data, list):
            continue

        rec_at = parse_dt_aware(row.get("time-scraped"))

        for item in data:
            if not isinstance(item, dict):
                continue

            jan_digits = _extract_jan_digits(item.get("jancode") or item.get("jan"))
            if not jan_digits:
                # 一些接口把 JAN 也写进 keywords，如 "... 4549995xxxxxxx"
                jan_digits = _extract_jan_digits(item.get("keywords"))

            if not jan_digits:
                continue

            pn = jan_map.get(jan_digits)
            if not pn:
                # 信息表里找不到该 JAN → 跳过（只输出已知机型）
                continue

            price = _coerce_price(item.get("goodsPrice"))
            if price is None:
                # 无价格（或无法解析）→ 跳过
                continue

            rows.append({
                "part_number": pn,
                "shop_name": "毎日買取",
                "price_new": int(price),
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
