from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _normalize_model_generic
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

def _norm(s: str) -> str:
    return (s or "").strip()

def _load_iphone17_info_df_for_shop2() -> pd.DataFrame:
    """
    读取 iphone17_info，并尽量保留 jan 列以供 shop1 做 JAN→PN 映射。
    输出列至少包含：part_number, model_name, capacity_gb, color，
    若检测到任何 jan 列，则额外返回标准化列 'jan'。
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

    need = {"part_number", "model_name", "capacity_gb", "color"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"iphone17_info 缺少必要列：{missing}")

    df = df.copy()
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number", "color"])

    # ★ 检测并标准化 jan 列（尽最大可能适配命名）
    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):  # 兼容 'JANコード' 等
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))  # 去重保序

    cols = ["part_number", "model_name", "capacity_gb", "color"]
    if jan_candidates:
        src = jan_candidates[0]
        df["jan"] = df[src]
        cols.append("jan")

    return df[cols]

def _extract_jan_digits(s: str) -> Optional[str]:
    if not s:
        return None
    m = JAN_DIGITS_RE.search(str(s))
    return m.group(1) if m else None

def _build_maps(info_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[Tuple[str, int, str], str]]:
    """
    返回：
      jan_map: { jan_digits -> part_number }（若信息表含 'jan' 列）
      triple_map: { (model_norm, capacity_gb, color_norm) -> part_number }
    """
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df["color_norm"] = df["color"].map(lambda x: _norm(str(x)))

    # JAN 映射（可选）
    jan_map: Dict[str, str] = {}
    jan_col_candidates = [c for c in df.columns if str(c).lower() == "jan"]
    if jan_col_candidates:
        jcol = jan_col_candidates[0]
        jseries = df[jcol].map(lambda x: _extract_jan_digits(str(x)) if pd.notna(x) else None)
        for _, r in df.assign(jan_norm=jseries).dropna(subset=["jan_norm"]).iterrows():
            jan_map[str(r["jan_norm"])] = str(r["part_number"])

    triple_map: Dict[Tuple[str, int, str], str] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col = r["color_norm"]
        if not m or pd.isna(cap) or not col:
            continue
        triple_map[(m, int(cap), col)] = str(r["part_number"])
    return jan_map, triple_map

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
    仅输出出现在 _load_iphone17_info_df_for_shop2() 的机型。
    """
    need_cols = ["jan", "type", "price", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop18 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    jan_map, triple_map = _build_maps(info_df)

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

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
