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

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

def to_int_yen_shop11(v) -> Optional[int]:
    """
    将各种形式的日元表示解析为 int（日元），若无法解析返回 None。
    支持样例：
      "1,000" "1,000円" "¥1,000" "１，０００" "1000" 以及带空格的混合形式
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # 去掉括号内的备注
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()

    # 找到第一个数字段（允许全角数字与逗号/点），并取其附近
    # 首先把全角数字/逗号/点转换为半角
    # 半角映射（数字/标点）
    trans_map = str.maketrans({
        '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
        '，':',','．':'.','－':'-','＋':'+','¥':'','￥':''
    })
    s2 = s.translate(trans_map)

    # 移除非数字/逗号/点/+-/空格/円符号
    # 但先尝试用正则抓取像 -?¥?1,000 或 1,000 的金额
    m = re.search(r"([+\-−－]?)\s*(?:¥|￥)?\s*([\d][\d,]*)", s2)
    if not m:
        # 备用：从任何位置提取数字串
        m2 = re.search(r"([\d][\d,]*)", s2)
        if not m2:
            return None
        amt_txt = m2.group(1)
        sign = ""
    else:
        sign = m.group(1) or ""
        amt_txt = m.group(2)

    # 去逗号并转 int
    amt_digits = re.sub(r"[^\d]", "", amt_txt or "")
    if not amt_digits:
        return None
    try:
        val = int(amt_digits)
    except Exception:
        return None

    if sign in ("-", "−", "－"):
        val = -val
    return val

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

def _normalize_number_text(txt: str) -> str:
    if txt is None:
        return ""
    return str(txt).translate(_FZ_TO_HZ_TRANS).strip()

_COLOR_SEP_SPLIT_RE = re.compile(r"[／/、，,・\s]+")  # split labels by these

_COLOR_GROUP_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # 最多 80 char 的 label group（不会以数字或 +/- 开头）
    [：:]\s*                               # 必须有 冒号（：或:）作为分隔（这是最常见的情形）
    (?P<sign>[+\-−－]?)\s*                 # 可选 +/-
    (?P<amount>[\d０-９,，]+)              # 金额（含全角数字与逗号）
    (?:\s*円|\s*¥|\s*￥)?                  # 可选货币符
    """,
    re.UNICODE | re.VERBOSE
)

_COLOR_GROUP_FALLBACK_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # label group（保守）
    [\s]*?(?P<sign>[+\-−－])\s*
    (?P<amount>[\d０-９,，]+)
    (?:\s*円|\s*¥|\s*￥)?
    """,
    re.UNICODE | re.VERBOSE
)

def _extract_color_deltas_shop11(text: str) -> List[Tuple[str, int]]:
    """
    更鲁棒的颜色差额解析，返回 [(label_raw, delta_int), ...]
    支持：
      - "シルバー・ブルー：-1,000円(未開封)" -> ('シルバー', -1000), ('ブルー', -1000)
      - "ブルー、ブラック：-2,000円(未開封)"
      - "銀206000,青205500" 不在此函数处理（若需要可在外面加入绝对价解析）
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text).strip()
    # 去掉括号内备注 (未開封) 等
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()
    if not s:
        return out

    s_norm = _normalize_number_text(s)

    # 1) 主匹配：labelGroup：+/-?amount
    for m in _COLOR_GROUP_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = _normalize_number_text(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        # 把 labels 按常见分隔符拆成多个 label
        for lbl in _COLOR_SEP_SPLIT_RE.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 2) 回退匹配（如 "ブルー -4000"）
    for m in _COLOR_GROUP_FALLBACK_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = _normalize_number_text(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        for lbl in _COLOR_SEP_SPLIT_RE.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 去重：若同 label 多次解析到，以最后一个为准（保留最后出现的 delta）
    if out:
        tmp: Dict[str, int] = {}
        for lbl, d in out:
            tmp[lbl] = d
        return list(tmp.items())

    return out

def _label_matches_color_shop11(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    匹配策略（宽容）：
      - 归一化后（去空白、半角/全角数字转换）相等；
      - label_raw 为 color_raw 的子串；
      - label 在常见家族词典里，家族任一词出现在 color_raw 即匹配。
    """
    if not label_raw or not color_raw:
        return False
    lbl = str(label_raw).strip()
    cr = str(color_raw).strip()

    # 规范化：半角化 + 去两端空白
    lbl_norm = _norm(lbl)
    cr_norm = color_norm  # 传入时应已是 _norm(color_raw)

    # 1) 精确归一化相等
    if lbl_norm == cr_norm:
        return True

    # 2) 原文子串（精确子串）
    if lbl in cr:
        return True

    # 3) 分割后任一片段是 color_raw 的子串（处理 "シルバー SV" vs "シルバー" 之类）
    for tok in _COLOR_SEP_SPLIT_RE.split(lbl):
        tok = tok.strip()
        if not tok:
            continue
        if tok in cr:
            return True
        if _norm(tok) == cr_norm:
            return True

    # 4) 家族同义词（小词典，可按需扩充）
    FAMILY = {
        "blue": ["ブルー","青","blue"],
        "silver": ["シルバー","銀","silver"],
        "black": ["ブラック","黒","black"],
        "white": ["ホワイト","白","white"],
        "gold": ["ゴールド","金","gold"],
        "orange": ["オレンジ","橙"],
    }
    k = lbl.strip().lower()
    # 若 lbl 自身是家族词或家族内词，检查家族内的 token 是否出现在 color_raw
    for fam_tokens in FAMILY.values():
        if k in [t.lower() for t in fam_tokens] or any(tok in lbl for tok in fam_tokens):
            for tok in fam_tokens:
                if tok in cr:
                    return True

    return False

def _build_color_map_shop11(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """
    构建 (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }
    """
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df["color_norm"] = df["color"].map(lambda x: _norm(str(x)))
    cmap: Dict[Tuple[str, int], Dict[str, Tuple[str, str]]] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        if not m or pd.isna(cap):
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        cmap[key][_norm(str(r["color"]))] = (str(r["part_number"]), str(r["color"]))
    return cmap

def clean_shop11(df: pd.DataFrame) -> pd.DataFrame:
    print("shop11:モバステ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    shop11 清洗器：
      - storage_name -> model+capacity (使用 _normalize_model_generic / _parse_capacity_gb)
      - price_unopened 为基准价（解析为整数）
      - caution_empty 中的颜色差额（如 "シルバー・ブルー：-1,000円(未開封)"）会应用到该颜色对应的 part_number
      - 输出：part_number, shop_name("モバステ"), price_new, recorded_at
    """
    need_cols = ["storage_name", "price_unopened", "caution_empty", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop11 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop11(info_df)

    rows = []
    for _, row in df.iterrows():
        storage = str(row.get("storage_name") or "").strip()
        if not storage:
            continue
        model_norm = _normalize_model_generic(storage)
        cap_gb = _parse_capacity_gb(storage)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)
        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            # 若找不到该型号/容量的映射则跳过
            continue

        base_price = to_int_yen_shop11(row.get("price_unopened"))
        if base_price is None:
            continue
        rec_at_raw = row.get("time-scraped")
        # 不做时区修正，直接保留原值或解析为 datetime（如果需要）
        try:
            recorded_at = dateparser.parse(str(rec_at_raw)) if pd.notna(rec_at_raw) else None
        except Exception:
            recorded_at = rec_at_raw

        deltas = _extract_color_deltas_shop11(row.get("caution_empty") or "")
        color_deltas: Dict[str, int] = {}
        for col_norm, (pn, col_raw) in color_map.items():
            for label_raw, delta in deltas:
                if _label_matches_color_shop11(label_raw, col_raw, col_norm):
                    color_deltas[col_norm] = int(delta)   # 后匹配覆盖前匹配

        # 生成输出：每个颜色的 part_number 都产一行
        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_deltas.get(col_norm, 0)
            price_new = int(base_price + delta)
            rows.append({
                "part_number": pn,
                "shop_name": "モバステ",
                "price_new": price_new,
                "recorded_at": recorded_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
