"""Shared utilities for external ingest cleaners.

This module centralizes regex patterns, normalization helpers, price/color
parsers, and iphone17 info loaders used by multiple cleaners. Keeping these in
one place reduces duplication inside :mod:`base_cleaners` and clarifies the
responsibility boundaries between reusable helpers and cleaner
implementations.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..external_ingest.helpers import to_int_yen

_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")
PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")
_YEN_RE = re.compile(r"[^\d]+")
_CAP_RE = re.compile(r"(\d+)\s*(TB|GB)", re.IGNORECASE)

_COLOR_DELTA_RE = re.compile(
    r"""^\s*
        (?P<label>全色|[\S　 ]*?[^\s　])     # 颜色名或“全色”
        \s*
        (?P<sign>[+\-−－])?                  # 可选符号
        \s*
        (?P<amount>\d[\d,]*)\s*円?           # 金额
        \s*$
    """,
    re.VERBOSE,
)

COLOR_DELTA_ANY_RE = re.compile(
    r"""(?P<label>[^：:\s]+)\s*[：:]\s*(?P<sign>[+\-−－])?\s*(?P<amount>\d[\d,]*)\s*円""",
    re.UNICODE,
)

PAIRINGS: List[Tuple[str, str]] = [
    ("iPhone17 Pro Max7", "iPhone17 Pro Max3"),  # 1TB
    ("iPhone17 Pro Max16", "iPhone17 Pro Max10"),  # 512GB
    ("iPhone17 Pro5", "iPhone17 Pro2"),  # 256GB
]

# ---- 颜色差额解析 ----
# 支持：'ブルー：-2,000円' / 'Blue-2000円' / 'ブルー:-2000円' / 'ブルー -2000円'
COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\s]+)\s*
        (?P<sep>[：:\-])\s*           # 新增：捕获分隔符
        (?P<sign>[+\-−－])?\s*        # 显式正负号（可选）
        (?P<amount>\d[\d,]*)\s*円
    """,
    re.UNICODE | re.VERBOSE,
)

# 英文族名到常见日文关键词的宽松映射（可按需要补充）
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


def _parse_color_delta(line: str) -> Optional[Tuple[str, int]]:
    """
    解析“颜色 ± 金额” 或 “全色 ± 金额”。
    返回: ("全色"或颜色名, delta_int)
    若无法解析，返回 None。
    """
    if not line or not isinstance(line, str):
        return None
    s = line.strip()
    m = _COLOR_DELTA_RE.match(s)
    if not m:
        if "全色" in s:
            return ("全色", 0)
        return None
    label = m.group("label").strip()
    sign = m.group("sign") or "+"
    amt = to_int_yen(m.group("amount"))
    if sign in ("-", "−", "－"):
        amt = -amt
    return (label, amt)


def _find_base_price(df: pd.DataFrame, idx: int) -> Optional[int]:
    """
    按规范：机种行(data11非空)的上一行 data 是基准价。
    若上一行取不到，向上最多回溯 3 行找首个含“円”的金额。
    """
    for j in range(idx - 1, max(-1, idx - 4), -1):
        if j < 0:
            break
        s = str(df["data"].iat[j]) if "data" in df.columns else ""
        if s and ("円" in s or re.search(r"\d[\d,]*", s)):
            price = to_int_yen(s)
            if price:
                return int(price)
    return None


def _collect_adjustments(df: pd.DataFrame, start_idx: int) -> Dict[str, int]:
    """
    从机种行【同一行】开始收集“颜色±金额”（含“全色”）。
    一直向下，直到遇到下一个 data11 非空（下一机种）或到文件末尾。
    返回：{ color_norm | "ALL" : delta_int }
    """
    result: Dict[str, int] = {}
    n = len(df)
    for i in range(start_idx, n):
        if i > start_idx and pd.notna(df.get("data11", pd.Series(dtype=object)).iat[i]):
            break
        parsed = _parse_color_delta(str(df.get("data", pd.Series(dtype=object)).iat[i]))
        if not parsed:
            continue
        label, delta = parsed
        if "全色" in label:
            result["ALL"] = delta if isinstance(delta, int) else 0
        else:
            result[_norm(label)] = delta if isinstance(delta, int) else 0
    return result


def _parse_yen(val) -> int | None:
    """'¥177,000' / '177,000円' / '177000' -> 177000"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = _YEN_RE.sub("", s)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _norm(s: str) -> str:
    return (s or "").strip()


def _norm_model_token(s: str) -> str:
    """
    将 data2-1 的机型片段“宽松”规范化（仅用于和 iphone17_info 里的 model_name 做宽松匹配）
    规则：小写、去空格、去多余符号
    """
    s = (s or "").lower()
    s = re.sub(r"iphone\s*", "iphone ", s)
    s = re.sub(r"[^0-9a-z\s+]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_model_name_loose(model_token: str, iphone17_df: pd.DataFrame) -> str | None:
    """
    宽松匹配：在 iphone17_df['model_name'] 中选与 token 最匹配的项（不严格 Fuzzy，先简单包含匹配）
    """
    token = _norm_model_token(model_token)
    if not token:
        return None
    candidates = list(
        dict.fromkeys([_norm(x) for x in iphone17_df["model_name"].dropna().tolist()])
    )

    def norm_m(m):
        return _norm_model_token(m)

    hits = [m for m in candidates if token in norm_m(m) or norm_m(m) in token]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return sorted(hits, key=lambda m: len(m), reverse=True)[0]
    return None


def _parse_adjust_rule(s: str) -> dict:
    """
    解析 data5 的减价规则：
      '青-1000' → {'青': -1000}
      '銀-5000+++青-5000' → {'銀': -5000, '青': -5000}
    返回：{组名: 负数(或0)}
    """
    rules = {}
    if not s:
        return rules
    parts = re.split(r"\+{1,3}|[,、，\s]+", str(s))
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"(.+?)-(\d+)", p)
        if not m:
            continue
        group = m.group(1).strip()
        amt = -int(m.group(2))
        rules[group] = amt
    return rules


def _apply_adjust_for_colorname(color_name: str, rules: dict) -> int:
    """
    根据规则返回针对该“颜色名”的价格修正（和机型容量下实际存在的颜色匹配）。
    """
    c = color_name or ""
    adjust = 0
    for group, delta in rules.items():
        g = group.strip()
        if g in ("青", "ブルー", "ミストブルー", "ディープブルー", "スカイブルー"):
            if "ブルー" in c:
                adjust += delta
        elif g in (
            "銀",
            "シルバー",
        ):
            if "シルバー" in c:
                adjust += delta
        else:
            if g and g in c:
                adjust += delta
    return adjust


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
        path = os.getenv("IPHONE17_INFO_CSV") or str(
            Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv"
        )

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

    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))

    cols = ["part_number", "model_name", "capacity_gb", "color"]
    if jan_candidates:
        src = jan_candidates[0]
        df["jan"] = df[src]
        cols.append("jan")

    return df[cols]


def pick_first_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """
    在 df 里按顺序返回第一列；都没有则返回空 Series。
    用它替代 df.get('A') or df.get('B') 这种写法，避免 Series 布尔判断歧义。
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(dtype=object)


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
    t = (
        t.replace("プロマックス", "Pro Max")
        .replace("プロ", "Pro")
        .replace("プラス", "Plus")
        .replace("ミニ", "mini")
        .replace("エアー", "Air")
        .replace("エア", "Air")
    )
    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)
    t = re.sub(r"(?i)\bpro\s*max\b", "Pro Max", t)
    t = re.sub(r"(?i)\bpro\b", "Pro", t)
    t = re.sub(r"(?i)\bplus\b", "Plus", t)
    t = re.sub(r"(?i)\bmini\b", "mini", t)
    if "iPhone" not in t and re.search(r"\b1[0-9]\b", t):
        t = re.sub(r"\b(1[0-9])\b", r"iPhone \1", t, count=1)
    t = re.sub(r"(?i)\biPhone\s+17\s+Air\b", "iPhone Air", t)
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    m = _NUM_MODEL_PAT.search(t)
    if m:
        base = f"{m.group(1)} {m.group(2)}"
        suf = (m.group(3) or "").strip()
        return f"{base} {suf}".strip()
    m2 = _AIR_PAT.search(t)
    if m2:
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


def _resolve_iphone17_info_path() -> str:
    """
    解析 iphone17_info.csv 的路径：
      1) settings.EXTERNAL_IPHONE17_INFO_PATH
      2) 环境变量 IPHONE17_INFO_CSV
      3) 项目内默认 AppleStockChecker/data/iphone17_info.csv
    """
    try:
        from django.conf import settings

        p = getattr(settings, "EXTERNAL_IPHONE17_INFO_PATH", None)
        if p:
            return str(p)
    except Exception:
        pass

    envp = os.getenv("IPHONE17_INFO_CSV")
    if envp and Path(envp).exists():
        return envp

    here = Path(__file__).resolve()
    app_root = here.parents[2]
    default_path = app_root / "data" / "iphone17_info.csv"
    return str(default_path)


@lru_cache(maxsize=1)
def _load_iphone17_info_df() -> pd.DataFrame:
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
        path = os.getenv("IPHONE17_INFO_CSV") or str(
            Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv"
        )

    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError(f"未找到 iphone17_info：{pth}")

    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(pth), re.I):
        df = pd.read_excel(pth)
    else:
        df = pd.read_csv(pth, encoding="utf-8-sig")

    need = {"part_number", "model_name", "capacity_gb"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"iphone17_info 缺少必要列：{missing}")

    df = df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name_norm", "capacity_gb", "part_number"])
    return df[["part_number", "model_name_norm", "capacity_gb"]]


def _clean_text(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u3000", " ")
    s = s.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def _extract_part_number(text: str) -> str | None:
    t = _clean_text(text)
    m = re.search(r"型番[:：]\s*([A-Z0-9]{4,6}\d{0,3}J/A)\b", t)
    if m:
        return m.group(1)
    m2 = PN_REGEX.search(t)
    return m2.group(0) if m2 else None


def _extract_price_new(raw: object) -> Optional[int]:
    """
    从 'price' 字段取整数日元：
      - 去掉'新品'等字样与货币符号
      - 支持区间 '82,000～84,000' 取最大
      - 合理区间过滤由 to_int_yen 完成
    """
    if raw is None:
        return None
    s = str(raw)
    s = s.replace("新品", "").replace("新\u54c1", "")
    s = s.replace("未開封", "").replace("未开封", "")
    return to_int_yen(s)


def _price_from_shop7(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x)
    s = s.replace("新品", "").replace("新\u54c1", "")
    s = s.replace("未開封", "").replace("未开封", "")
    return to_int_yen(s)


def _norm_model_for_shop7(text: str) -> str:
    """
    在 _normalize_model_generic 之前做一点“shop7 特有”的宽松处理：
    然后交给 _normalize_model_generic 做最终归一。
    """
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)
    t = (
        t.replace("プロマックス", "Pro Max")
        .replace("プロ", "Pro")
        .replace("プラス", "Plus")
        .replace("ミニ", "mini")
        .replace("エアー", "Air")
        .replace("エア", "Air")
    )
    t = re.sub(r"(?i)pro[-\s]?max", "Pro Max", t)
    if re.search(r"(?i)\b17\s+air\b", t):
        t = re.sub(r"(?i)\b17\s+air\b", "iPhone Air", t)
    if "iPhone" not in t and re.search(r"\b1[0-9]\b", t):
        t = re.sub(r"\b(1[0-9])\b", r"iPhone \1", t, count=1)
    return _normalize_model_generic(t)


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


@lru_cache(maxsize=1)
def _load_jan_to_pn() -> Dict[str, str]:
    """
    返回 { jan(13位字符串) : part_number } 的字典。
    若 info 文件没有 jan 列，则返回空字典（后续走 data8 的 PN 兜底）。
    """
    path = _resolve_info_path()
    if not path.exists():
        return {}
    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(path), re.I):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    if "part_number" not in df.columns:
        return {}
    if "jan" in df.columns:
        df = df.copy()
        df["jan"] = df["jan"].astype(str).str.replace(r"[^\d]", "", regex=True)
        df = df[df["jan"].str.fullmatch(r"\d{13}", na=False)]
        mapping = dict(zip(df["jan"].astype(str), df["part_number"].astype(str)))
        return mapping
    return {}


def _extract_pn_from_text(text: object) -> Optional[str]:
    if text is None:
        return None
    s = str(text).replace("\u3000", " ")
    m = _PN_REGEX.search(s)
    return m.group(0) if m else None


def _price_from_shop6_data7(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x)
    s = (
        s.replace("新品", "")
        .replace("新\u54c1", "")
        .replace("未開封", "")
        .replace("未开封", "")
    )
    return to_int_yen(s)


def _extract_jan_from_data(x: object) -> Optional[str]:
    """
    从 'data' 文本里抽取 13 位 JAN（例如 'JAN:4549995648300'）
    """
    if x is None:
        return None
    s = str(x)
    m = re.search(r"JAN[:：]?\s*(\d{13})", s)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(\d{13})\b", s)
    return m2.group(1) if m2 else None


def _price_from_shop5(x: object) -> Optional[int]:
    """
    price 列 -> price_new：
      - 去掉前导 '～'、'新品'、'未開封' 等修饰
      - 支持区间 '～162,500円' / '162,500～170,000円'：取最大值
      - 支持 '10.5万'
    """
    if x is None:
        return None
    s = str(x).strip()
    s = s.lstrip("～")
    s = (
        s.replace("新品", "")
        .replace("新\u54c1", "")
        .replace("未開封", "")
        .replace("未开封", "")
    )
    return to_int_yen(s)


def _price_from_shop3(x: object) -> Optional[int]:
    """
    data5 -> price_new
    - 预期形如 '¥177,000'；也兼容 '～177,000円'/'10.5万' 等；取区间最大值
    - 去除可能出现的修饰词（“新品/未開封”等）
    """
    if x is None:
        return None
    s = str(x)
    s = (
        s.replace("新品", "")
        .replace("新\u54c1", "")
        .replace("未開封", "")
        .replace("未开封", "")
    )
    return to_int_yen(s)


def _extract_color_deltas(text: str) -> List[Tuple[str, int]]:
    """
    从机型列文本中提取若干 '颜色：±金额円' 片段。
    返回 [(原始标签, delta_int)]，不做颜色是否存在校验。
    """
    res: List[Tuple[str, int]] = []
    if not text:
        return res
    for m in COLOR_DELTA_ANY_RE.finditer(str(text)):
        label = m.group("label").strip()
        sign = m.group("sign") or "+"
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        if sign in ("-", "−", "－"):
            amt = -amt
        res.append((label, int(amt)))
    return res


def _build_color_maps(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """
    为每个 (model_norm, capacity_gb) 构建颜色映射：
      key: (model_norm, cap_gb)
      val: { color_norm : (part_number, color_raw) }
    """
    info_df = info_df.copy()
    info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
    info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")
    info_df["color_norm"] = info_df["color"].map(lambda x: _norm(str(x)))
    cmap: Dict[Tuple[str, int], Dict[str, Tuple[str, str]]] = {}
    for _, r in info_df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col_norm = r["color_norm"]
        pn = str(r["part_number"])
        col_raw = str(r["color"])
        if pd.isna(cap) or not m or not col_norm:
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        cmap[key][col_norm] = (pn, col_raw)
    return cmap


def _labels_to_color_deltas(
    labels_and_deltas: List[Tuple[str, int]],
    color_map_for_model: Dict[str, Tuple[str, str]],
) -> Dict[str, int]:
    """
    将从文本里抓到的 [(label_raw, delta)] 映射到具体颜色键 color_norm。
    匹配规则（按优先级）：
      1) label_norm 与某个 color_norm 完全相等 → 命中该颜色
      2) label_raw 子串出现在 color_raw 中（例如 'ブルー' 命中 'ディープブルー'）
    允许多标签命中同一颜色，后者会覆盖前者（以最后一次出现为准）。
    """
    out: Dict[str, int] = {}
    items = list(color_map_for_model.items())
    for label_raw, delta in labels_and_deltas:
        label_norm = _norm(label_raw)
        matched = False
        for col_norm, (_pn, col_raw) in items:
            if label_norm == col_norm:
                out[col_norm] = delta
                matched = True
        if matched:
            continue
        for col_norm, (_pn, col_raw) in items:
            if label_raw in col_raw:
                out[col_norm] = delta
    return out


def _extract_color_deltas_shop12(text: str) -> List[Tuple[str, int]]:
    """
    从 '備考1' 文本中提取若干 (label_raw, delta_int)。
    修复点：当 sign 缺省且分隔符 sep 为 '-'（含全角/数学负号），视为负数。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    for m in COLOR_DELTA_RE.finditer(str(text)):
        label = m.group("label").strip()
        sep = m.group("sep")
        sign = m.group("sign")
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        if sign:
            negative = sign in ("-", "−", "－")
        else:
            negative = sep in ("-", "−", "－")
        delta = -int(amt) if negative else int(amt)
        out.append((label, delta))
    return out


def _has_all_colors(text: str) -> Optional[int]:
    """
    若文本含“全色”，且可选出现 '全色 ± 金額'，返回统一 delta；
    若仅出现 '全色' 无金额，返回 0；
    若未出现 '全色'，返回 None。
    """
    if not text:
        return None
    s = str(text)
    if "全色" not in s:
        return None
    m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*(\d[\d,]*)\s*円", s)
    if m:
        sign = m.group(1) or "+"
        amt = to_int_yen(m.group(2)) or 0
        if sign in ("-", "−", "－"):
            amt = -amt
        return int(amt)
    return 0


def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配一个 'label_raw' 是否命中某个颜色。"""
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
    if label_norm in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[label_norm]:
            if jp in str(color_raw):
                return True
    return False


def _build_color_map(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
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


__all__ = [
    "_NUM_MODEL_PAT",
    "_AIR_PAT",
    "_PN_REGEX",
    "PN_REGEX",
    "_YEN_RE",
    "_CAP_RE",
    "_COLOR_DELTA_RE",
    "COLOR_DELTA_ANY_RE",
    "PAIRINGS",
    "COLOR_DELTA_RE",
    "FAMILY_SYNONYMS",
    "_parse_color_delta",
    "_find_base_price",
    "_collect_adjustments",
    "_parse_yen",
    "_norm",
    "_norm_model_token",
    "_pick_model_name_loose",
    "_parse_adjust_rule",
    "_apply_adjust_for_colorname",
    "_load_iphone17_info_df_for_shop2",
    "pick_first_col",
    "_normalize_model_generic",
    "_parse_capacity_gb",
    "_resolve_iphone17_info_path",
    "_load_iphone17_info_df",
    "_clean_text",
    "_extract_part_number",
    "_extract_price_new",
    "_price_from_shop7",
    "_norm_model_for_shop7",
    "_resolve_info_path",
    "_load_jan_to_pn",
    "_extract_pn_from_text",
    "_price_from_shop6_data7",
    "_extract_jan_from_data",
    "_price_from_shop5",
    "_price_from_shop3",
    "_extract_color_deltas",
    "_build_color_maps",
    "_labels_to_color_deltas",
    "_extract_color_deltas_shop12",
    "_has_all_colors",
    "_label_matches_color",
    "_build_color_map",
]
