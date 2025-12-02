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

def _price_from_shop3(x: object) -> Optional[int]:
    """
    data5 -> price_new
    - 预期形如 '¥177,000'；也兼容 '～177,000円'/'10.5万' 等；取区间最大值
    - 去除可能出现的修饰词（“新品/未開封”等）
    """
    if x is None:
        return None
    s = str(x)
    s = (s.replace("新品", "")
           .replace("新\u54c1", "")
           .replace("未開封", "")
           .replace("未开封", ""))  # 安全冗余
    return to_int_yen(s)

FAMILY_SYNONYMS_shop3 = { "blue": ["ブルー", "青", "ディープブルー"], "ブルー": ["ブルー", "青", "ディープブルー"], "青": ["ブルー", "青", "ディープブルー"], "ディープブルー": ["ディープブルー", "ブルー", "青"], "silver": ["シルバー", "銀"], "シルバー": ["シルバー", "銀"], "銀": ["シルバー", "銀"], }

_LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s；;]+")  # include ideographic middle dot, full-width space etc.

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

_DELTA_PATTERN_STRICT = re.compile(
    r"""(?P<labels>[^+\-−－\d¥￥円]+?)   # 标签片段（尽量少匹配到数字/符号）
        (?P<sign>[+\-−－])\s*            # 必须有 +/- 符号
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?  # 金额（允许全角数字与逗号）
    """,
    re.UNICODE | re.VERBOSE
)

_DELTA_PATTERN_LOOSE = re.compile(
    r"""(?P<labels>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE
)

def _clean_label_token(tok: str) -> str:
    """去除前后空白、括号内备注、奇怪符号"""
    if tok is None:
        return ""
    t = str(tok).strip()
    # 去掉尾部/头部的注释 (未開封) (例) 等
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"（.*?）", "", t)
    t = t.strip()
    return t

def _extract_color_deltas_shop3(text: str) -> List[Tuple[str, int]]:
    """
    从 text 中提取 (label_raw, delta_int) 多条记录，支持多标签共用金额的写法。
    例子都能被捕获：
      'ブルー、シルバー　-1000'
      'シルバー-3,000/ディープブルー-3,000'
      'シルバー　-3,000 ブルー -3,000'
      'シルバー/ディープブルー　-3000'
      'ブルー　-2000'
      'ブラック、ブルー　-4000'
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text).translate(_FZ_TO_HZ_TRANS).strip()  # 先半角化常见字符
    # 1) 先用严格规则全局匹配（要求有 +/-）
    for m in _DELTA_PATTERN_STRICT.finditer(s):
        labels_part = m.group("labels")
        sign = m.group("sign")
        amt_txt = m.group("amount")
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            continue
        if sign in ("-", "−", "－"):
            amt = -amt
        # 拆分 labels_part（可能是 'シルバー/ディープブルー' / 'ブルー、シルバー'）
        toks = [t for t in _LABEL_SPLIT_RE.split(labels_part) if t and t.strip()]
        for tok in toks:
            lbl = _clean_label_token(tok)
            if lbl:
                out.append((lbl, int(amt)))

    # 2) 若 strict 没匹配到任何结果，尝试 loose（更宽松）
    if not out:
        for m in _DELTA_PATTERN_LOOSE.finditer(s):
            labels_part = m.group("labels")
            sign = m.group("sign")
            amt_txt = m.group("amount")
            amt = _normalize_amount_text(amt_txt)
            if amt is None:
                continue
            if sign in ("-", "−", "－"):
                amt = -amt
            toks = [t for t in _LABEL_SPLIT_RE.split(labels_part) if t and t.strip()]
            for tok in toks:
                lbl = _clean_label_token(tok)
                if lbl:
                    out.append((lbl, int(amt)))

    return out

def _label_matches_color_shop3(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配：归一等值 | 文本子串 | 家族词命中"""
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop3:
            candidates.update(FAMILY_SYNONYMS_shop3[k])
    if not candidates:
        for _, toks in FAMILY_SYNONYMS_shop3.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break
    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map_shop3(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """
    (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }
    依赖 _load_iphone17_info_df_for_shop2()（含 color 列）
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

def clean_shop3(df: pd.DataFrame) -> pd.DataFrame:
    print("shop3:買取一丁目---------->进入清洗器时间：",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入列：
      web-scraper-order, web-scraper-start-url, data4, data5, data6, data8, title, 减价1, time-scraped
    规则：
      - shop_name 固定 '買取一丁目'
      - title 含“机种名 + 容量” → 归一(model_norm) + 解析容量(capacity_gb)
      - 通过 iphone17_info 对应 (model_norm, capacity_gb) 获取“所有颜色”的 part_number 并展开
      - data5 为新品基准价 price_new（解析日元/区间）
      - “减价1”里出现单色/多色的差额（±N円）时，对应颜色在基准价上加/减
      - time-scraped 为 recorded_at（为空行直接跳过）
    输出：part_number, shop_name, price_new, recorded_at
    """
    # 必要列检查
    need_cols = ["title", "data5", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop3 清洗器缺少必要列：{c}")

    # 过滤掉 time-scraped 为空的行
    src = df.copy()
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 载入信息表（含颜色）
    info_df = _load_iphone17_info_df_for_shop2()
    color_maps = _build_color_map_shop3(info_df)

    # 解析 model/cap
    model_norm = src["title"].map(_normalize_model_generic)
    cap_gb     = src["title"].map(_parse_capacity_gb)

    # 解析价格/时间
    try:
        base_price = src["data5"].map(_price_from_shop3)
    except Exception:
        base_price = src["data5"].map(to_int_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 减价文本（可能不存在）
    remark = src.get("减价1") if "减价1" in src.columns else None

    rows: List[dict] = []
    for i in range(len(src)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p0 = base_price.iat[i]
        t  = recorded_at.iat[i]
        if not m or pd.isna(c) or p0 is None:
            continue

        key = (m, int(c))
        cmap = color_maps.get(key)  # { color_norm: (pn, color_raw) }
        if not cmap:
            # 未收录（机型或容量不在信息表）
            continue

        # 默认所有颜色 = 基准价
        per_color_abs: Dict[str, int] = {}    # 若你后续想支持“绝对价”，可在此填入
        per_color_delta: Dict[str, int] = {}

        # 解析“减价1”中的差额
        rem_text = str(remark.iat[i]) if remark is not None else ""
        deltas = _extract_color_deltas_shop3(rem_text)
        if deltas:
            for col_norm, (pn, col_raw) in cmap.items():
                for label_raw, delta in deltas:
                    if _label_matches_color_shop3(label_raw, col_raw, col_norm):
                        per_color_delta[col_norm] = delta  # 多次命中以后者为准

        # 生成输出：若存在绝对价则优先（此处暂无），否则 base±delta
        for col_norm, (pn, col_raw) in cmap.items():
            if col_norm in per_color_abs:
                price_val = per_color_abs[col_norm]
            else:
                price_val = int(p0) + per_color_delta.get(col_norm, 0)
            rows.append({
                "part_number": str(pn),
                "shop_name": "買取一丁目",
                "price_new": int(price_val),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

def _normalize_amount_text(s: str) -> Optional[int]:
    """
    把全角数字/标点转半角，去掉非数字字符后尝试转换为 int。
    返回 None 表示无法解析。
    """
    if s is None:
        return None
    t = str(s).translate(_FZ_TO_HZ_TRANS)
    # 仅保留数字和逗号
    m = re.search(r"([0-9][0-9,]*)", t)
    if not m:
        return None
    numtxt = m.group(1).replace(",", "")
    try:
        return int(numtxt)
    except Exception:
        return None

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})
