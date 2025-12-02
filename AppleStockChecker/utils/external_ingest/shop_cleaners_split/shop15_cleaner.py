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

_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)

FAMILY_SYNONYMS = {
    "blue": ["ブルー"],
    "black": ["ブラック", "黒"],
    "white": ["ホワイト", "白"],
    "green": ["グリーン", "緑"],
    "red": ["レッド", "赤"],
    "pink": ["ピンク"],
    "purple": ["パープル", "紫"],
    "yellow": ["イエロー", "黄"],
    "gold": ["ゴールド"],
    "silver": ["シルバー"],
    "gray": ["グレー", "グレイ", "灰"],
    "natural": ["ナチュラル"],
}

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

def _normalize_model_generic(text: str) -> str:
    """
    统一型号主体：
      - iPhone17/16 + 后缀（Pro/Pro Max/Plus/mini）
      - iPhone Air（含“17 air”→ Air）
      - 允许紧凑写法：17pro / 17promax / 16Pro / 16Plus ...
    输出：'iPhone 17 Pro Max' / 'iPhone 17 Pro' / 'iPhone Air' / ...
    """
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    # 日文别名到英文
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini")
           .replace("エアー", "Air")
           .replace("エア", "Air"))

    # ❗ 在“数字后立即跟英文”的位置补一个空格：17pro -> 17 pro
    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)

    # 标准化大小写/形态：pro-max / ProMax / promáx → Pro Max；pro → Pro；plus → Plus；mini → mini
    t = re.sub(r"(?i)\bpro\s*max\b", "Pro Max", t)
    t = re.sub(r"(?i)\bpro\b", "Pro", t)
    t = re.sub(r"(?i)\bplus\b", "Plus", t)
    t = re.sub(r"(?i)\bmini\b", "mini", t)

    # 若没有 iPhone 前缀但出现纯数字代号，补上
    if "iPhone" not in t and re.search(r"\b1[0-9]\b", t):
        t = re.sub(r"\b(1[0-9])\b", r"iPhone \1", t, count=1)

    # 特例：'17 air' → iPhone Air（防止被当成 iPhone 17）
    t = re.sub(r"(?i)\biPhone\s+17\s+Air\b", "iPhone Air", t)

    # 去容量/SIM/括号噪声
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 1) 数字代号机型
    m = _NUM_MODEL_PAT.search(t)
    if m:
        base = f"{m.group(1)} {m.group(2)}"
        suf  = (m.group(3) or "").strip()
        return f"{base} {suf}".strip()

    # 2) Air
    m2 = _AIR_PAT.search(t)
    if m2:
        # 当前返回主体 'iPhone Air'；若以后真有 Air Plus 等可在此扩展
        return "iPhone Air"

    return ""

def _parse_capacity_gb(text: str) -> Optional[int]:
    if not text:
        return None
    t = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", t, flags=re.I)
    if m:
        return int(round(float(m.group(1)) * 1024))
    m = re.search(r"(\d{2,4})\s*GB", t, flags=re.I)
    if m:
        return int(m.group(1))
    return None

def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    宽松匹配一个 'label_raw' 是否命中某个颜色（color_raw / color_norm）。
    优先：
      - 归一化后完全相等
      - label_raw 子串包含于 color_raw
      - 英文族名（如 Blue）映射到日文家族词，并判断是否是 color_raw 的子串
    """
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    key = label_raw.strip().lower()
    if key in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[key]:
            if jp in str(color_raw):
                return True
    # 也尝试 label_norm 的英文键
    if label_norm in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[label_norm]:
            if jp in str(color_raw):
                return True
    return False

FIRST_YEN_RE_shop15 = re.compile(r"(\d[\d,]*)\s*円")  # 抓取 price 中第一个 “N円”（作为基准价）

COLOR_DELTA_IN_PRICE_RE_shop15 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／円¥]+)\s*   # 颜色标签
        (?P<sep>[：:\-])? \s*                # 分隔符（可无）
        (?P<sign>[+\-−－])? \s*              # 正负号（可无）
        (?P<amount>\d[\d,]*) \s* (?:円)?     # 金额（可跟円）
    """,
    re.UNICODE | re.VERBOSE,
)

def _extract_base_price(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)
    return to_int_yen(m.group(1))

def _extract_color_deltas_from_price(text: str) -> List[Tuple[str, int]]:
    """
    从 price 文本中抽取若干 (label_raw, delta_int)。
    先去掉第一个“基准价 N円”，在剩余文字里用 finditer 捕获所有“颜色±金额”。
    负号判定：显式 sign 优先；若 sign 缺省且 sep='-'，按负数处理。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    s = str(text)
    m0 = FIRST_YEN_RE_shop15.search(s)
    tail = s[m0.end():] if m0 else s

    # 为兼容 “　”（全角空格）等情况，不强制切片，直接全串 finditer
    for m in COLOR_DELTA_IN_PRICE_RE_shop15.finditer(tail):
        label = (m.group("label") or "").strip()
        if not label:
            continue
        sep = m.group("sep")
        sign = m.group("sign")
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        if sign:
            negative = sign in ("-", "−", "－")
        else:
            negative = sep in ("-", "−", "－") if sep else False
        delta = -int(amt) if negative else int(amt)
        out.append((label, delta))
    return out

def _build_color_map_shop15(info_df: pd.DataFrame) -> Dict[tuple, Dict[str, Tuple[str, str]]]:
    """
    (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }
    """
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df["color_norm"] = df["color"].map(lambda x: _norm(str(x)))
    cmap: Dict[tuple, Dict[str, Tuple[str, str]]] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]; cap = r["capacity_gb"]
        if not m or pd.isna(cap):
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        cmap[key][_norm(str(r["color"]))] = (str(r["part_number"]), str(r["color"]))
    return cmap

def clean_shop15(df: pd.DataFrame) -> pd.DataFrame:
    print("shop15:買取当番---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop15.csv):
      - data2: 机型（如 'iPhone 17 Pro Max 256GB'）
      - price: 基准价 + 颜色差额（如 '213,500円　ブルー-9,000円　シルバー-7,500円'）
      - time-scraped: 抓取时间
    输出:
      - part_number, shop_name(=買取当番), price_new, recorded_at
    规则：
      - 仅输出信息表存在的（机型, 容量, 颜色）
      - 命中的颜色：price = base + delta；未命中的颜色：price = base
    """
    # 必要列检查
    for c in ["price", "data2", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop15 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop15(info_df)

    rows: List[dict] = []

    for _, row in df.iterrows():
        model_text = str(row.get("data2") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)
        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        price_text = row.get("price")
        base_price = _extract_base_price(price_text)
        if base_price is None:
            continue
        base_price = int(base_price)

        # 解析颜色差额
        labels_and_deltas = _extract_color_deltas_from_price(price_text)
        color_deltas: Dict[str, int] = {}
        if labels_and_deltas:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in labels_and_deltas:
                    if _label_matches_color(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta  # 多命中时以后者为准

        rec_at = parse_dt_aware(row.get("time-scraped"))

        # 生成行：未命中的颜色用基准价
        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_deltas.get(col_norm, 0)
            rows.append({
                "part_number": pn,
                "shop_name": "買取当番",
                "price_new": int(base_price + delta),
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")
