from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ..external_ingest.helpers import to_int_yen, parse_dt_aware
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
    ("iPhone17 Pro Max7",  "iPhone17 Pro Max3"),   # 1TB
    ("iPhone17 Pro Max16", "iPhone17 Pro Max10"),  # 512GB
    ("iPhone17 Pro5",      "iPhone17 Pro2"),       # 256GB
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

class Cleaner(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...

CLEANERS: Dict[str, Cleaner] = {}

# === 可复用的小工具（如果你已有同名函数可删除这里并改用现有函数） ===

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
        # 某些行可能只有“全色”不带金额，视为 delta=0
        if "全色" in s:
            return ("全色", 0)
        return None
    label = m.group("label").strip()
    sign = m.group("sign") or "+"
    amt = to_int_yen(m.group("amount"))  # 只取绝对值
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
    for j in range(start_idx, n):
        # 下一个机种（且必须 j > start_idx 才算“下一个”）
        nxt_model = str(df["data11"].iat[j]) if "data11" in df.columns and df["data11"].iat[j] is not None else ""
        if j > start_idx and nxt_model.strip():
            break

        line = str(df["data"].iat[j]) if "data" in df.columns and df["data"].iat[j] is not None else ""
        parsed = _parse_color_delta(line)
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
    if val is None: return None
    s = str(val).strip()
    if not s: return None
    s = _YEN_RE.sub("", s)
    if not s: return None
    try:
        n = int(s)
        return n
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
    s = re.sub(r"[^0-9a-z\s+]", "", s)  # 仅保留 a-z0-9 和空格
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_model_name_loose(model_token: str, iphone17_df: pd.DataFrame) -> str | None:
    """
    宽松匹配：在 iphone17_df['model_name'] 中选与 token 最匹配的项（不严格 Fuzzy，先简单包含匹配）
    """
    token = _norm_model_token(model_token)
    if not token: return None
    # 候选（去重）
    candidates = list(dict.fromkeys([_norm(x) for x in iphone17_df["model_name"].dropna().tolist()]))
    # 简单策略：同样规范化后，包含则命中
    def norm_m(m): return _norm_model_token(m)
    hits = [m for m in candidates if token in norm_m(m) or norm_m(m) in token]
    if len(hits) == 1:
        return hits[0]
    # 多命中时偏向更长的 model_name（更具体）
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
    if not s: return rules
    # 允许 '+++', '+', '，' 等作为分隔
    parts = re.split(r"\+{1,3}|[,、，\s]+", str(s))
    for p in parts:
        p = p.strip()
        if not p: continue
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
    约定：
      - '青'：匹配包含「ブルー」的颜色（ミストブルー/ディープブルー/スカイブルー 等）
      - '銀'/'シルバー'：匹配包含「シルバー」
      - 可扩展其它组（例：'黒'->「ブラック」；'白'->「ホワイト/シルバー」等）
    """
    c = color_name or ""
    adjust = 0
    for group, delta in rules.items():
        g = group.strip()
        if g in ("青", "ブルー","ミストブルー","ディープブルー","スカイブルー"):
            if "ブルー" in c:
                adjust += delta
        elif g in ("銀", "シルバー",):
            if "シルバー" in c:
                adjust += delta
        else:
            # 精确匹配 group 文字（万一 data5 直接写具体颜色）
            if g and g in c:
                adjust += delta
    return adjust
#
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

def pick_first_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """
    在 df 里按顺序返回第一列；都没有则返回空 Series。
    用它替代 df.get('A') or df.get('B') 这种写法，避免 Series 布尔判断歧义。
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(dtype=object)
#
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

def _resolve_iphone17_info_path() -> str:
    """
    解析 iphone17_info.csv 的路径：
      1) settings.EXTERNAL_IPHONE17_INFO_PATH
      2) 环境变量 IPHONE17_INFO_CSV
      3) 项目内默认 AppleStockChecker/data/iphone17_info.csv
    """
    # 1) settings 覆盖
    try:
        from django.conf import settings
        p = getattr(settings, "EXTERNAL_IPHONE17_INFO_PATH", None)
        if p:
            return str(p)
    except Exception:
        pass

    # 2) 环境变量
    envp = os.getenv("IPHONE17_INFO_CSV")
    if envp and Path(envp).exists():
        return envp

    # 3) 项目内默认
    # 计算当前文件所在 app 的 data 目录
    here = Path(__file__).resolve()
    app_root = here.parents[2]  # .../AppleStockChecker/
    default_path = app_root / "data" / "iphone17_info.csv"
    return str(default_path)

@lru_cache(maxsize=1)
def _load_iphone17_info_df() -> pd.DataFrame:
    """
    读取 AppleStockChecker/data/iphone17_info.csv 或 settings / env 指定的路径。
    输出列：part_number, model_name_norm, capacity_gb
    """
    # 解析路径（settings > env > 默认）
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

    need = {"part_number", "model_name", "capacity_gb"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"iphone17_info 缺少必要列：{missing}")

    df = df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name_norm", "capacity_gb", "part_number"])
    return df[["part_number", "model_name_norm", "capacity_gb"]]

def register_cleaner(name: str):
    def deco(fn: Cleaner):
        CLEANERS[name] = fn
        return fn
    return deco

def _clean_text(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\u3000", " ")          # 全角空格
    s = s.replace("\r", " ").replace("\n", " ")  # 去换行
    return re.sub(r"\s+", " ", s).strip()

def _extract_part_number(text: str) -> str | None:
    t = _clean_text(text)
    # 1) 优先：显式 “型番: XXXXXJ/A”
    m = re.search(r"型番[:：]\s*([A-Z0-9]{4,6}\d{0,3}J/A)\b", t)
    if m:
        return m.group(1)
    # 2) 兜底：全文 PN 正则
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
    # 去掉“新品”等修饰词
    s = s.replace("新品", "").replace("新\u54c1", "")
    # 常见前缀也清理一下
    s = s.replace("未開封", "").replace("未开封", "")
    return to_int_yen(s)


def _price_from_shop7(x: object) -> Optional[int]:
    """data2 -> price_new：去掉“新品/未開封/货币符号/逗号”，区间取最大"""
    if x is None:
        return None
    s = str(x)
    s = s.replace("新品", "").replace("新\u54c1", "")
    s = s.replace("未開封", "").replace("未开封", "")
    return to_int_yen(s)


def _norm_model_for_shop7(text: str) -> str:
    """
    在 _normalize_model_generic 之前做一点“shop7 特有”的宽松处理：
      - ‘promax/ProMax/pro-max’ → ‘Pro Max’
      - ‘17 air’ → ‘iPhone Air’
      - 没有 iPhone 前缀但有 '17' 的，补成 ‘iPhone 17 ...’
    然后交给 _normalize_model_generic 做最终归一。
    """
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    # 日文/英文后缀标准化
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini")
           .replace("エアー", "Air")
           .replace("エア", "Air"))

    # promax 连写/大小写
    t = re.sub(r"(?i)pro[-\s]?max", "Pro Max", t)

    # 若没有 iPhone 前缀但出现 "17 air" / "17 pro max" / "17 pro" / "17 plus"
    # 先把 "17 air" 显式改成 "iPhone Air"（Air 没有数字后缀）
    if re.search(r"(?i)\b17\s+air\b", t):
        # 去掉“17 ”，以免 _normalize_model_generic 误识别为 iPhone 17
        t = re.sub(r"(?i)\b17\s+air\b", "iPhone Air", t)

    # 若没有 iPhone 单词但有纯数字代号（例如 "17 Pro Max 256GB"）
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
        # 没找到映射文件时，仍允许仅走 data8 的 PN 兜底
        return {}
    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(path), re.I):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    if "part_number" not in df.columns:
        # 没有 PN 列，无法映射
        return {}

    # 允许 info 表没有 jan；有则清洗为 13 位
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
    # 去一些修饰词
    s = (s.replace("新品", "")
           .replace("新\u54c1", "")
           .replace("未開封", "")
           .replace("未开封", ""))
    return to_int_yen(s)

def _extract_jan_from_data(x: object) -> Optional[str]:
    """
    从 'data' 文本里抽取 13 位 JAN（例如 'JAN:4549995648300'）
    """
    if x is None:
        return None
    s = str(x)
    # 优先匹配 'JAN:XXXXXXXXXXXXX'
    m = re.search(r"JAN[:：]?\s*(\d{13})", s)
    if m:
        return m.group(1)
    # 兜底：任意 13 位数字
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
    s = s.lstrip("～")  # '～162,500円' -> '162,500円'
    s = (s.replace("新品", "")
           .replace("新\u54c1", "")
           .replace("未開封", "")
           .replace("未开封", ""))
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
    s = (s.replace("新品", "")
           .replace("新\u54c1", "")
           .replace("未開封", "")
           .replace("未开封", ""))  # 安全冗余
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
    # 反查结构：norm -> (pn, raw)
    items = list(color_map_for_model.items())
    for label_raw, delta in labels_and_deltas:
        label_norm = _norm(label_raw)
        # 1) 完全相等
        matched = False
        for col_norm, (_pn, col_raw) in items:
            if label_norm == col_norm:
                out[col_norm] = delta
                matched = True
        if matched:
            continue
        # 2) 子串包含（适配“ブルー”命中“ディープブルー/スカイブルー/ミストブルー”）
        for col_norm, (_pn, col_raw) in items:
            if label_raw in col_raw:
                out[col_norm] = delta
    return out

def _extract_color_deltas_shop12(text: str) -> List[Tuple[str, int]]:
    """
    从 '備考1' 文本中提取若干 (label_raw, delta_int)。
    修复点：当 sign 缺省且分隔符 sep 为 '-'（含全角/数学负号），视为负数。
    支持示例：
      'Blue-2000円'         -> ('Blue', -2000)
      'ブルー：-2,000円'    -> ('ブルー', -2000)
      'ブルー:2000円'       -> ('ブルー', +2000)
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

        # 计算有效符号：显式 sign 优先；否则看分隔符是否是负号风格
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
    # 试图解析 "全色 ± n円"
    m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*(\d[\d,]*)\s*円", s)
    if m:
        sign = m.group(1) or "+"
        amt = to_int_yen(m.group(2)) or 0
        if sign in ("-", "−", "－"):
            amt = -amt
        return int(amt)
    return 0

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


# =============== shop2 清洗器 ===============
@register_cleaner("shop2")
def clean_shop2(shop2_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：
      - shop2_df: 读取自 shop2.csv（columns: web-scraper-order, web-scraper-start-url, data2-1, data2-2, ..., data5, ..., data3, time-scraped）
      - iphone17_df: 读取自 iphone17_info.csv（至少包含: model_name, capacity_gb, color, part_number）
    输出 DataFrame 列：
      - part_number, shop_name, price_new, recorded_at
    规则：
      - 仅 data2-2 含 'simfree' 且含 '未開封'（且不含 '開封'）的行
      - data2-1 解析机型+容量；若在 iphone17_df 找不到对应机型容量 → 跳过
      - 价格 data3；data5 减价规则（青/銀等组）会作用到对应颜色（蓝系/银系）
      - shop_name 固定 '海峡通信'；recorded_at = time-scraped
    """
    SHOP = "海峡通信"

    # 统一列名（小写）
    df = shop2_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 必要列存在性检查
    need_cols = ["data2-1","data2-2","data3","data5","time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = None

    # 只保留 simfree 未開封
    def _is_target(s: str) -> bool:
        s = (s or "").lower()
        return ("simfree" in s) and ("未開封" in s)

    df = df[df["data2-2"].apply(_is_target)]
    if df.empty:
        return pd.DataFrame(columns=["part_number","shop_name","price_new","recorded_at"])

    # iphone17_df 预处理
    info = _load_iphone17_info_df_for_shop2()
    # info = iphone17_df.copy()
    # info["model_name"] = info["model_name"].apply(_norm)
    # 容量转 int GB
    if "capacity_gb" not in info.columns:
        # 如果你的 info 表容量列叫别的名字，替换这里
        raise ValueError("iphone17_info.csv 需要包含 capacity_gb 列")
    # 颜色规范
    info["color"] = info["color"].apply(_norm)

    out_rows = []

    for _, row in df.iterrows():
        recorded_at = parse_dt_aware(row.get("time-scraped"))
        raw_modelcap = _norm(row.get("data2-1"))
        if not raw_modelcap:
            continue

        # 容量
        cap_gb = _parse_capacity_gb(raw_modelcap)
        if not cap_gb:
            continue

        # 机型（宽松匹配）
        model_name = _pick_model_name_loose(raw_modelcap, info)
        if not model_name:
            continue

        # 该机型容量下的所有颜色
        sub = info[(info["model_name"] == model_name) & (info["capacity_gb"] == cap_gb)].copy()
        if sub.empty:
            continue

        # 基础价格
        base_price = _parse_yen(row.get("data3"))
        if base_price is None:
            continue

        # 减价规则
        rules = _parse_adjust_rule(row.get("data5"))


        # 记录时间
        rec_raw = row.get("time-scraped")
        # 容忍多种日期格式
        try:
            rec_dt = pd.to_datetime(rec_raw, utc=True, errors="coerce")
            recorded_at = rec_dt.isoformat() if pd.notnull(rec_dt) else None
            recorded_at = parse_dt_aware(row.get("time-scraped"))
        except Exception:
            recorded_at = None

        # 为该机型容量下的每个颜色生成一条记录（套用 color-specific 调整）
        for _, it in sub.iterrows():
            part = _norm(it.get("part_number"))
            color = _norm(it.get("color"))
            if not part:
                continue
            adj = _apply_adjust_for_colorname(color, rules)
            price = base_price + adj
            if price <= 0:
                # 价格异常则跳过
                continue
            out_rows.append({
                "part_number": part,
                "shop_name": SHOP,
                "price_new": int(price),
                "recorded_at": recorded_at
            })

    if not out_rows:
        return pd.DataFrame(columns=["part_number","shop_name","price_new","recorded_at"])

    out = pd.DataFrame(out_rows, columns=["part_number","shop_name","price_new","recorded_at"])
    return out


FAMILY_SYNONYMS_shop3 = { "blue": ["ブルー", "青", "ディープブルー"], "ブルー": ["ブルー", "青", "ディープブルー"], "青": ["ブルー", "青", "ディープブルー"], "ディープブルー": ["ディープブルー", "ブルー", "青"], "silver": ["シルバー", "銀"], "シルバー": ["シルバー", "銀"], "銀": ["シルバー", "銀"], }
# 分隔标签的常见 token
_LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s；;]+")  # include ideographic middle dot, full-width space etc.

# 把全角数字、全角符号等转为半角的简易映射
_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

# 必须带符号版本： capture labels (任意非数字非符号片段) + sign + amount
_DELTA_PATTERN_STRICT = re.compile(
    r"""(?P<labels>[^+\-−－\d¥￥円]+?)   # 标签片段（尽量少匹配到数字/符号）
        (?P<sign>[+\-−－])\s*            # 必须有 +/- 符号
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?  # 金额（允许全角数字与逗号）
    """,
    re.UNICODE | re.VERBOSE
)

# 宽松降级版本（若 strict 未匹配到任何项），允许 sep 前后有一些噪音
_DELTA_PATTERN_LOOSE = re.compile(
    r"""(?P<labels>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE
)

def _normalize_amount_text(s: str) -> Optional[int]:
    """把可能的全角数字、千位分隔符等规范成 int。返回 None 时无法解析"""
    if s is None:
        return None
    t = str(s).translate(_FZ_TO_HZ_TRANS)
    m = re.search(r"([0-9][0-9,]*)", t)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None

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

@register_cleaner("shop3")
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


# 在文件顶部（或合适位置）加入：
LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s]+")   # 用于把 "シルバー/ディープブルー" 拆成两项

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

def _parse_color_delta_shop4(line: str) -> Optional[List[Tuple[str, int]]]:
    """
    解析“颜色 ± 金额” 或 “全色 ± 金额”。返回 list[(label, delta)]（可能为多项）。
    若无法解析，返回 None。
    Example inputs:
      "シルバー-1,000円" -> [("シルバー", -1000)]
      "シルバー/ディープブルー-3,000円" -> [("シルバー", -3000), ("ディープブルー", -3000)]
      "全色-2,000円" -> [("全色", -2000)]
      "全色" -> [("全色", 0)]
    """
    if not line or not isinstance(line, str):
        return None
    s = line.strip()
    # 快速兜底：仅含“全色”而无数字，视为 0
    if s == "全色" or s == "全 色":
        return [("全色", 0)]

    m = _COLOR_DELTA_RE.match(s)
    if not m:
        # 有些格式会把 label 和金额分开，比如 "シルバー/ディープブルー -3,000円"
        # 我们再尝试一个更宽松的匹配：先从行中找出金额，再把前面的部分视为 label group
        am = re.search(r"([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)\s*円?", s)
        if not am:
            # 若行仅含“全色”关键字仍处理
            if "全色" in s:
                return [("全色", 0)]
            return None
        sign = am.group(1) or "+"
        amt_txt = am.group(2)
        amt = _normalize_amount_text(amt_txt) if _normalize_amount_text(amt_txt) is not None else None
        if amt is None:
            # 最后尝试 to_int_yen 如果存在
            try:
                amt = to_int_yen(amt_txt)
            except Exception:
                amt = None
        if amt is None:
            return None
        if sign in ("-", "−", "－"):
            amt = -amt

        label_part = s[:am.start()].strip()
        if not label_part:
            # 没有 label，视为无法解析
            return None
        labels = [p for p in LABEL_SPLIT_RE.split(label_part) if p]
        return [(lbl.strip(), int(amt)) for lbl in labels]

    label_raw = m.group("label").strip()
    sign = m.group("sign") or "+"
    # 优先使用 to_int_yen，如果返回 None 使用 fallback
    amt_val = None
    try:
        amt_val = to_int_yen(m.group("amount"))
    except Exception:
        amt_val = None
    if amt_val is None:
        amt_val = _normalize_amount_text(m.group("amount"))

    if amt_val is None:
        return None

    if sign in ("-", "−", "－"):
        amt_val = -int(amt_val)
    else:
        amt_val = int(amt_val)

    # label_raw 可能为 "シルバー/ディープブルー" 等复合项，拆分
    labels = [p for p in LABEL_SPLIT_RE.split(label_raw) if p]
    if not labels:
        return None
    return [(lbl.strip(), int(amt_val)) for lbl in labels]

# 调整 _collect_adjustments 来兼容 _parse_color_delta 返回多项
def _collect_adjustments_shop4(df: pd.DataFrame, start_idx: int) -> Dict[str, int]:
    """
    从机种行【同一行】开始收集“颜色±金额”（含“全色”）。
    一直向下，直到遇到下一个 data11 非空（下一机种）或到文件末尾。
    返回：{ color_norm | "ALL" : delta_int }
    同一颜色若多次出现，以后出现的（靠近机种行的）为准（覆盖前者）。
    """
    result: Dict[str, int] = {}
    n = len(df)
    for j in range(start_idx, n):
        # 下一个机种（且必须 j > start_idx 才算“下一个”）
        nxt_model = ""
        if "data11" in df.columns:
            val = df["data11"].iat[j]
            nxt_model = str(val) if val is not None else ""
        if j > start_idx and nxt_model.strip():
            break

        line = ""
        if "data" in df.columns:
            val = df["data"].iat[j]
            line = str(val) if val is not None else ""
        parsed = _parse_color_delta_shop4(line)
        if not parsed:
            continue

        # parsed 是 list[(label, delta)]
        for label, delta in parsed:
            if not isinstance(delta, int):
                try:
                    delta = int(delta)
                except Exception:
                    continue
            if "全色" in label:
                result["ALL"] = delta
            else:
                # 以归一化后的 color key 存储（便于后续匹配）
                result[_norm(label)] = delta

    return result

# ---------- 主清洗器 ----------

@register_cleaner("shop4")
def clean_shop4(df: pd.DataFrame) -> pd.DataFrame:
    print("shop4:モバイルミックス---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop4.csv):
      - web-scraper-order, web-scraper-start-url, data, data11, time-scraped
    规则：
      - data11：机种名 + 容量（第二行起常见）
      - 该机种的“基准价” = 上一行的 data（金额，如 212,000円）
      - 机种行之后的若干行 data 可能出现“颜色 ± 金额”，对单色或全色调整
      - 若机种行同一行的 data 含“全色”（可带 ±金额），则所有颜色同价（基准价±统一调整）
      - 仅输出出现在 _load_iphone17_info_df_for_shop2() 的机种
      - shop_name 固定为「モバイルミックス」
      - recorded_at = parse_dt_aware(time-scraped)

    输出：
      - columns: part_number, shop_name, price_new, recorded_at
    """
    # 必要列
    for c in ["data", "data11", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop4 清洗器缺少必要列：{c}")

    # 归一化信息表并建立 (model_norm, cap) → {color_norm: pn}
    info_df = _load_iphone17_info_df_for_shop2().copy()
    # 预期含：part_number, model_name, capacity_gb, color
    info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
    info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")
    info_df["color_norm"] = info_df["color"].map(lambda x: _norm(str(x)))

    pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    for _, r in info_df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col = r["color_norm"]
        pn = str(r["part_number"])
        if pd.isna(cap) or not m or not col:
            continue
        key = (m, int(cap))
        pn_map.setdefault(key, {})
        pn_map[key][col] = pn

    rows: List[dict] = []

    n = len(df)

    for i in range(n):

        model_text = str(df["data11"].iat[i]) if df["data11"].iat[i] is not None else ""
        model_text = model_text.strip()
        if not model_text:
            continue

        # 从 data11 提取 model + capacity
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_to_pn = pn_map.get(key)

        if not color_to_pn:
            # 信息表没有该机种容量组合 → 跳过
            continue

        # 基准价：上一行 data
        base_price = _find_base_price(df, i)
        if base_price is None:
            # 没有可用基准价，跳过该机种
            continue

        # 同行 data 若写“全色 ± n円”，优先应用统一调整
        # 同行 data 若写“全色 ± n円”，优先应用统一调整
        same_line = str(df["data"].iat[i]) if df["data"].iat[i] is not None else ""
        same_line_parsed = _parse_color_delta_shop4(same_line)  # 可能 None 或 list[(label,delta)]
        global_delta = None
        if same_line_parsed:
            # 如果其中有 '全色'，以其 delta 为 global_delta（若多次出现，取最后一项）
            for lbl, d in same_line_parsed:
                if "全色" in lbl:
                    global_delta = d
                    # 不 break：如果后面还有 '全色'，后者覆盖前者

        # 其后续行的“颜色 ± n円”调整
        adjustments = _collect_adjustments_shop4(df, i)

        # 若同行已表明“全色”，且 adjustments 也包含 ALL，则以最近的声明为准：
        # 优先使用同行的 global_delta（更接近机种行）
        if global_delta is not None:
            adjustments["ALL"] = global_delta

        # recorded_at 取机种行 time-scraped
        rec_at = parse_dt_aware(df["time-scraped"].iat[i])

        # 价格生成：如果有 "ALL"，所有颜色都用 (base + ALL)
        if "ALL" in adjustments:
            final_price = base_price + adjustments["ALL"]
            for col_norm, pn in color_to_pn.items():
                rows.append({
                    "part_number": pn,
                    "shop_name": "モバイルミックス",
                    "price_new": int(final_price),
                    "recorded_at": rec_at,
                })
        else:
            # 单色调整：出现在 adjustments 的颜色使用 base+delta；其余颜色用 base
            for col_norm, pn in color_to_pn.items():

                delta = adjustments.get(col_norm, 0)
                rows.append({
                    "part_number": pn,
                    "shop_name": "モバイルミックス",
                    "price_new": int(base_price + delta),
                    "recorded_at": rec_at,
                })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out

@register_cleaner("shop5-1")
def clean_shop5_1(df: pd.DataFrame) -> pd.DataFrame:
    print("shop5-1:森森買取---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    """
    输入表头：web-scraper-order, web-scraper-start-url, pagination-selector, price, data, name, (time-scraped)
    要求：删除 name 含 '中古' 的行；通过 data 的 JAN 定位 PN；price -> price_new；time-scraped -> recorded_at
    shop_name 固定 '森森買取'
    """
    # 必要列检查（time-scraped 在问题描述中是必须的）
    need_cols = ["price", "data", "name", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop5-1 清洗器缺少必要列：{c}")

    # 1) 过滤掉 name 含“中古”的行
    src = df.copy()
    mask_keep = ~src["name"].astype(str).str.contains("中古", na=False)
    src = src[mask_keep]

    # 2) 跳过 time-scraped 为空的行（避免落库时间为空）
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 3) 载入 JAN -> PN 映射（允许为空字典；若 info 表无 jan 列则无法映射 PN）
    jan_to_pn = _load_jan_to_pn()

    # 4) 逐列解析
    jan_series = src["data"].map(_extract_jan_from_data)
    pn_series  = jan_series.map(lambda j: jan_to_pn.get(j) if j and re.fullmatch(r"\d{13}", j) else None)

    price_new   = src["price"].map(_price_from_shop5)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 5) 组装结果：必须有 PN & 价格
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_series.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "森森買取",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop5-2")
def clean_shop5_2(df: pd.DataFrame) -> pd.DataFrame:
    print("shop5-2:森森買取---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入表头：web-scraper-order, web-scraper-start-url, pagination-selector, price, data, name, (time-scraped)
    要求：删除 name 含 '中古' 的行；通过 data 的 JAN 定位 PN；price -> price_new；time-scraped -> recorded_at
    shop_name 固定 '森森買取'
    """
    # 必要列检查（time-scraped 在问题描述中是必须的）
    need_cols = ["price", "data", "name", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop5-2 清洗器缺少必要列：{c}")

    # 1) 过滤掉 name 含“中古”的行
    src = df.copy()
    mask_keep = ~src["name"].astype(str).str.contains("中古", na=False)
    src = src[mask_keep]

    # 2) 跳过 time-scraped 为空的行（避免落库时间为空）
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 3) 载入 JAN -> PN 映射（允许为空字典；若 info 表无 jan 列则无法映射 PN）
    jan_to_pn = _load_jan_to_pn()

    # 4) 逐列解析
    jan_series = src["data"].map(_extract_jan_from_data)
    pn_series  = jan_series.map(lambda j: jan_to_pn.get(j) if j and re.fullmatch(r"\d{13}", j) else None)

    price_new   = src["price"].map(_price_from_shop5)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 5) 组装结果：必须有 PN & 价格
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_series.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "森森買取",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop5-3")
def clean_shop5_3(df: pd.DataFrame) -> pd.DataFrame:
    print("shop5-3:森森買取---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入表头：web-scraper-order, web-scraper-start-url, pagination-selector, price, data, name, (time-scraped)
    要求：删除 name 含 '中古' 的行；通过 data 的 JAN 定位 PN；price -> price_new；time-scraped -> recorded_at
    shop_name 固定 '森森買取'
    """
    # 必要列检查（time-scraped 在问题描述中是必须的）
    need_cols = ["price", "data", "name", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop5-3 清洗器缺少必要列：{c}")

    # 1) 过滤掉 name 含“中古”的行
    src = df.copy()
    mask_keep = ~src["name"].astype(str).str.contains("中古", na=False)
    src = src[mask_keep]

    # 2) 跳过 time-scraped 为空的行（避免落库时间为空）
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 3) 载入 JAN -> PN 映射（允许为空字典；若 info 表无 jan 列则无法映射 PN）
    jan_to_pn = _load_jan_to_pn()

    # 4) 逐列解析
    jan_series = src["data"].map(_extract_jan_from_data)
    pn_series  = jan_series.map(lambda j: jan_to_pn.get(j) if j and re.fullmatch(r"\d{13}", j) else None)

    price_new   = src["price"].map(_price_from_shop5)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 5) 组装结果：必须有 PN & 价格
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_series.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "森森買取",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop5-4")
def clean_shop5_4(df: pd.DataFrame) -> pd.DataFrame:
    print("shop5-4:森森買取---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入表头：web-scraper-order, web-scraper-start-url, pagination-selector, price, data, name, (time-scraped)
    要求：删除 name 含 '中古' 的行；通过 data 的 JAN 定位 PN；price -> price_new；time-scraped -> recorded_at
    shop_name 固定 '森森買取'
    """
    # 必要列检查（time-scraped 在问题描述中是必须的）
    need_cols = ["price", "data", "name", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop5-4 清洗器缺少必要列：{c}")

    # 1) 过滤掉 name 含“中古”的行
    src = df.copy()
    mask_keep = ~src["name"].astype(str).str.contains("中古", na=False)
    src = src[mask_keep]

    # 2) 跳过 time-scraped 为空的行（避免落库时间为空）
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 3) 载入 JAN -> PN 映射（允许为空字典；若 info 表无 jan 列则无法映射 PN）
    jan_to_pn = _load_jan_to_pn()

    # 4) 逐列解析
    jan_series = src["data"].map(_extract_jan_from_data)
    pn_series  = jan_series.map(lambda j: jan_to_pn.get(j) if j and re.fullmatch(r"\d{13}", j) else None)

    price_new   = src["price"].map(_price_from_shop5)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 5) 组装结果：必须有 PN & 价格
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_series.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "森森買取",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop6-1")
def clean_shop6_1(df: pd.DataFrame) -> pd.DataFrame:
    print("shop6-1:買取ルデヤ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 必要列检查
    need_cols = ["data7", "phone", "data8", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop6-1 清洗器缺少必要列：{c}")

    # 跳过 time-scraped 为空的行
    src = df.copy()
    mask_time = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    jan_to_pn = _load_jan_to_pn()  # 可能为空字典（允许）

    # 解析列
    jan_series = src["phone"].astype(str).str.replace(r"[^\d]", "", regex=True)
    pn_by_jan = jan_series.map(lambda j: jan_to_pn.get(j) if re.fullmatch(r"\d{13}", j or "") else None)
    pn_fallback = src["data8"].map(_extract_pn_from_text)  # 从 data8 兜底提取 PN

    # 价格/时间
    price_new = src["data7"].map(_price_from_shop6_data7)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 组装：优先 JAN→PN；无则 data8 提取；再无则丢弃
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_by_jan.iat[i] or pn_fallback.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "買取ルデヤ",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop6-2")
def clean_shop6_2(df: pd.DataFrame) -> pd.DataFrame:
    print("shop6-2:買取ルデヤ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 必要列检查
    need_cols = ["data7", "phone", "data8", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop6-2 清洗器缺少必要列：{c}")

    # 跳过 time-scraped 为空的行
    src = df.copy()
    mask_time = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    jan_to_pn = _load_jan_to_pn()  # 可能为空字典（允许）

    # 解析列
    jan_series = src["phone"].astype(str).str.replace(r"[^\d]", "", regex=True)
    pn_by_jan = jan_series.map(lambda j: jan_to_pn.get(j) if re.fullmatch(r"\d{13}", j or "") else None)
    pn_fallback = src["data8"].map(_extract_pn_from_text)  # 从 data8 兜底提取 PN

    # 价格/时间
    price_new = src["data7"].map(_price_from_shop6_data7)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 组装：优先 JAN→PN；无则 data8 提取；再无则丢弃
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_by_jan.iat[i] or pn_fallback.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "買取ルデヤ",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop6-3")
def clean_shop6_3(df: pd.DataFrame) -> pd.DataFrame:
    print("shop6-3:買取ルデヤ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 必要列检查
    need_cols = ["data7", "phone", "data8", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop6-3 清洗器缺少必要列：{c}")

    # 跳过 time-scraped 为空的行
    src = df.copy()
    mask_time = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    jan_to_pn = _load_jan_to_pn()  # 可能为空字典（允许）

    # 解析列
    jan_series = src["phone"].astype(str).str.replace(r"[^\d]", "", regex=True)
    pn_by_jan = jan_series.map(lambda j: jan_to_pn.get(j) if re.fullmatch(r"\d{13}", j or "") else None)
    pn_fallback = src["data8"].map(_extract_pn_from_text)  # 从 data8 兜底提取 PN

    # 价格/时间
    price_new = src["data7"].map(_price_from_shop6_data7)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 组装：优先 JAN→PN；无则 data8 提取；再无则丢弃
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_by_jan.iat[i] or pn_fallback.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "買取ルデヤ",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop6-4")
def clean_shop6_4(df: pd.DataFrame) -> pd.DataFrame:
    print("shop6-4:買取ルデヤ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 必要列检查
    need_cols = ["data7", "phone", "data8", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop6-4 清洗器缺少必要列：{c}")

    # 跳过 time-scraped 为空的行
    src = df.copy()
    mask_time = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    jan_to_pn = _load_jan_to_pn()  # 可能为空字典（允许）

    # 解析列
    jan_series = src["phone"].astype(str).str.replace(r"[^\d]", "", regex=True)
    pn_by_jan = jan_series.map(lambda j: jan_to_pn.get(j) if re.fullmatch(r"\d{13}", j or "") else None)
    pn_fallback = src["data8"].map(_extract_pn_from_text)  # 从 data8 兜底提取 PN

    # 价格/时间
    price_new = src["data7"].map(_price_from_shop6_data7)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 组装：优先 JAN→PN；无则 data8 提取；再无则丢弃
    rows: List[dict] = []
    for i in range(len(src)):
        pn = pn_by_jan.iat[i] or pn_fallback.iat[i]
        p  = price_new.iat[i]
        ts = recorded_at.iat[i]
        if not pn or p is None:
            continue
        rows.append({
            "part_number": str(pn),
            "shop_name": "買取ルデヤ",
            "price_new": int(p),
            "recorded_at": ts,
        })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop7")
def clean_shop7(df: pd.DataFrame) -> pd.DataFrame:
    print("DEBUG: shop7:買取ホムラ ----------> 进入清洗器 时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    _SHORT_MODEL_REPLACEMENTS = [
    (re.compile(r'(?i)\b17\s*pro\s*max\b'), "iPhone 17 Pro Max"),
    (re.compile(r'(?i)\b17promax\b'), "iPhone 17 Pro Max"),
    (re.compile(r'(?i)\b17\s*pro\b'), "iPhone 17 Pro"),
    (re.compile(r'(?i)\b17pro\b'), "iPhone 17 Pro"),
    (re.compile(r'(?i)\b17\s*air\b'), "iPhone 17 Air"),
    (re.compile(r'(?i)\b17air\b'), "iPhone 17 Air"),
    (re.compile(r'(?i)\bi\s*phone\s*17\b'), "iPhone 17"),
    (re.compile(r'(?i)\b17\b'), "iPhone 17"),  # 小心：放最后做兜底
]
    def _norm_model_for_shop7(s: Optional[str]) -> str:
        """
        针对 shop7 的 model 字段做宽松归一化：
          - 跳过纯数字的行（返回空字符串）
          - 将 '17pro', '17promax', '17 pro max' 等短写扩展为 'iPhone 17 Pro Max'
          - 最后调用 _normalize_model_generic 生成最终的归一字符串（与 info 表匹配）
        返回 '' 表示无法识别（将被跳过）
        """
        if s is None:
            return ""
        txt = str(s).strip()
        if not txt:
            return ""

        # 跳过行号/序号（仅数字或少量标点）
        if re.fullmatch(r'[\d\-\.\s]+', txt):
            # 代表这是序号/编号行，例如 "1", "2" 等
            return ""

        # 统一全角空格 & 多余空白
        txt = re.sub(r'[\u3000\s]+', ' ', txt).strip()

        # 先把常用短写替换成标准形式
        expanded = txt
        for patt, repl in _SHORT_MODEL_REPLACEMENTS:
            expanded = patt.sub(repl, expanded)

        # 可能存在像 "17pro 256GB" 或 "17promax 256GB" 这样的，
        # 上面的替换会把它变成 "iPhone 17 Pro 256GB" 之类，交给 normalize 处理。
        try:
            norm = _normalize_model_generic(expanded)
        except Exception:
            # 若 _normalize_model_generic 出错，退化为简单返回 expanded（再由上级判定是否匹配）
            norm = expanded

        # 如果 normalize 后为空或仅数字，认为无法识别
        if not norm or re.fullmatch(r'[\d\-\.\s]+', str(norm).strip()):
            return ""
        return norm

    info_df = _load_iphone17_info_df_for_shop2()  # part_number, model_name_norm, capacity_gb, color

    # 必要列检查
    need_cols = ["data", "data2", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop7 清洗器缺少必要列：{c}")

    # 先把 time-scraped 为空的行排除，避免时间解析报错
    df = df.copy().reset_index(drop=True)
    mask_time_ok = df["time-scraped"].astype(str).str.strip().ne("") & df["time-scraped"].notna()
    df = df[mask_time_ok].reset_index(drop=True)
    if df.empty:
        print("DEBUG: 输入 df 为空或所有行 time-scraped 缺失，返回空 DataFrame")
        return pd.DataFrame(columns=["part_number","shop_name","price_new","recorded_at"])

    # data -> 机型&容量
    model_norm_series = df["data2"].map(_norm_model_for_shop7)
    cap_gb_series     = df["data2"].map(_parse_capacity_gb)

    # 价格/时间（注意 data2 里是价格，我们仅在行有 price 时才处理该行）
    price_series  = df["data3"].map(_price_from_shop7)
    recorded_at   = df["time-scraped"].map(parse_dt_aware)


    # ------------- 构建 (model_norm, cap) -> { color_norm: part_number } -------------
    info2 = info_df.copy()
    if "color" not in info2.columns:
        raise ValueError("info_df 缺少 'color' 列，无法进行颜色映射")
    info2["model_name_norm"] = info2["model_name"].map(_normalize_model_generic)
    info2["capacity_gb"] = pd.to_numeric(info2["capacity_gb"], errors="coerce").astype("Int64")
    info2["color_norm"] = info2["color"].map(lambda x: _norm(str(x)))

    pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    for _, r in info2.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col = r["color_norm"]
        pn = str(r["part_number"])
        if pd.isna(cap) or not m or not col:
            continue
        key = (m, int(cap))
        pn_map.setdefault(key, {})
        pn_map[key][col] = pn

    print(f"DEBUG: 建立了 pn_map，包含 {len(pn_map)} 个 (model,cap) 条目")

    # ----------------- 颜色减价解析函数（shop7 专用） -----------------
    DELTA_RE = re.compile(
        r"(?P<labels>[^\d¥￥円\+\-−－]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[0-9０-９,，]+)",
        re.UNICODE
    )

    FW_TO_ASC = str.maketrans({
        "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
        "，":",","．":".","－":"-","＋":"+","　":" "
    })

    def _to_int_amount(s: str) -> Optional[int]:
        if s is None:
            return None
        t = str(s).translate(FW_TO_ASC)
        m = re.search(r"([0-9][0-9,]*)", t)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None

    def _parse_color_deltas_shop7(text: str) -> Dict[str, int]:
        res: Dict[str, int] = {}
        if not text or not str(text).strip():
            return res
        s = str(text).strip()
        parts = []
        # 尝试先用 DELTA_RE 匹配整段里的所有有金额的片段；若没则按分隔符拆
        found = False
        for m in DELTA_RE.finditer(s):
            found = True
            labels_part = m.group("labels") or ""
            sign = m.group("sign") or "+"
            amt_txt = m.group("amount")
            amt = _to_int_amount(amt_txt)
            if amt is None:
                continue
            delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
            # labels_part 可能包含多个颜色，用常见分隔符拆
            for tok in re.split(r"[／/、，,・\s]+", labels_part):
                tok = tok.strip()
                if not tok:
                    continue
                key = _norm(tok)
                res[key] = delta
        if not found:
            # 退化处理：没有显式金额匹配的情况下，尝试找类似 "シルバー/ディープブルー-3000" 形式
            # 以最后出现的 +/- 数字为金额，前面子串作为标签
            # 例如 "シルバー/ディープブルー-3000"
            m2 = re.search(r"(?P<labels>.+?)[\s]*([+\-−－])\s*(?P<amount>[0-9０-９,，]+)", s)
            if m2:
                labels_part = m2.group("labels") or ""
                sign = m2.group(2) or "+"
                amt_txt = m2.group("amount")
                amt = _to_int_amount(amt_txt)
                if amt is not None:
                    delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
                    for tok in re.split(r"[／/、，,・\s]+", labels_part):
                        tok = tok.strip()
                        if tok:
                            res[_norm(tok)] = delta
        # Debug print for parsed deltas
        if res:
            print(f"DEBUG: 解析到 color deltas from '{text}': {res}")
        else:
            print(f"DEBUG: 未解析到 color deltas from '{text}'")
        return res

    # ----------------- 主循环：遍历含价格的行 -----------------
    rows: List[dict] = []
    n = len(df)
    for i in range(n):
        base_price = price_series.iat[i]
        if base_price is None:
            # 不是机种行（或者是下行的颜色行），跳过
            continue

        model_text_raw = df["data"].iat[i] if df["data"].iat[i] is not None else ""
        m = model_norm_series.iat[i]
        c = cap_gb_series.iat[i]
        t = recorded_at.iat[i]
        print(f"DEBUG: 处理行 i={i}, model_raw='{model_text_raw}', model_norm='{m}', cap='{c}', base_price={base_price}")

        if not m or pd.isna(c):
            print(f"DEBUG: 跳过 i={i} 因为 model/cap 缺失")
            continue
        c = int(c)
        key = (m, c)
        color_to_pn = pn_map.get(key)
        print(f"DEBUG: 对应的 color->pn 映射: {color_to_pn}")
        if not color_to_pn:
            print(f"DEBUG: info 表中未找到该机型容量 key={key}，跳过 i={i}")
            continue

        # 检查下一行是否为颜色减价行：下一行 data 非空 且 data2 为空
        deltas: Dict[str, int] = {}
        j = i + 1
        if j < n:
            nxt_data = df["data2"].iat[j] if df["data2"].iat[j] is not None else ""
            nxt_data2 = df["data3"].iat[j] if "data3" in df.columns and df["data2"].iat[j] is not None else ""
            if str(nxt_data).strip() and (str(nxt_data2).strip() == "" or pd.isna(nxt_data2)):
                print(f"DEBUG: 发现潜在颜色行 at i+1={j}: '{nxt_data}'")
                deltas = _parse_color_deltas_shop7(nxt_data)
            else:
                print(f"DEBUG: 下一行 i+1={j} 不是颜色行 (data='{nxt_data}' data2='{nxt_data2}')")

        # 生成每个颜色的 price_new
        for col_norm, pn in color_to_pn.items():
            delta = 0
            if deltas:
                if col_norm in deltas:
                    delta = deltas[col_norm]
                    print(f"DEBUG: 直接匹配 col_norm='{col_norm}' 得到 delta={delta} for pn={pn}")
                else:
                    # 进一步尝试在 info2 中查原始 color 文本匹配 deltas 的 key
                    matches = info2[
                        (info2["model_name_norm"] == m) &
                        (info2["capacity_gb"].astype("Int64") == c) &
                        (info2["part_number"].astype(str) == str(pn))
                    ]
                    raw_color = matches["color"].iat[0] if not matches.empty else ""
                    matched = False
                    for lbl_norm, dval in deltas.items():
                        # 尝试在 raw_color 的归一化字符串中查找 lbl_norm
                        if lbl_norm and lbl_norm in _norm(raw_color):
                            delta = dval
                            matched = True
                            print(f"DEBUG: 通过 raw_color='{raw_color}' 的归一化匹配 lbl='{lbl_norm}' -> delta={delta} for pn={pn}")
                            break
                        # 也尝试原文子串匹配
                        if lbl_norm and lbl_norm in raw_color:
                            delta = dval
                            matched = True
                            print(f"DEBUG: 通过 raw_color 原文匹配 lbl='{lbl_norm}' -> delta={delta} for pn={pn}")
                            break
                    if not matched and deltas:
                        print(f"DEBUG: 未匹配到颜色调整 for pn={pn} (raw_color='{raw_color}'), 使用 delta=0")

            price_final = int(base_price + delta)
            print(f"DEBUG: 输出 -> pn={pn}, base={base_price}, delta={delta}, final={price_final}")
            rows.append({
                "part_number": str(pn),
                "shop_name": "買取ホムラ",
                "price_new": price_final,
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    print("DEBUG: 完成 clean_shop7, 产出行数:", len(out))
    return out


@register_cleaner("shop8")
def clean_shop8(df: pd.DataFrame) -> pd.DataFrame:
    print("shop8:買取wiki---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 列名容错：有些抓取器可能用不同大小写或空白
    # 这里统一抓关键列
    col_model = "機種名"
    col_price_new = "未開封"
    col_time = "time-scraped"

    for need in (col_model, col_price_new, col_time):
        if need not in df.columns:
            raise ValueError(f"shop8 清洗器缺少必要列: {need}")

    # 解析
    part_numbers = df[col_model].map(_extract_part_number)
    price_new = df[col_price_new].map(to_int_yen)
    recorded_at = df[col_time].map(parse_dt_aware)

    out = pd.DataFrame({
        "part_number": part_numbers,
        "shop_name": "買取wiki",
        "price_new": price_new,
        "recorded_at": recorded_at,
    })

    # 丢掉关键字段缺失的行（pn 或 price）
    out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)

    # 确保类型（避免 pandas 的 NA 类型导致后续 int() 失败）
    out["part_number"] = out["part_number"].astype(str)
    return out

#ドラゴンモバイル    # shop10　　９
@register_cleaner("shop10")
def clean_shop10(df: pd.DataFrame) -> pd.DataFrame:
    print("shop10:ドラゴンモバイル---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    info_df = _load_iphone17_info_df()

    need_cols = ["data2", "price", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop10 清洗器缺少必要列：{c}")

    # ★ 改这里
    model_norm = df["data2"].map(_normalize_model_generic)
    cap_gb     = df["data2"].map(_parse_capacity_gb)

    def _price_from_shop10(x: object) -> int | None:
        if x is None: return None
        s = str(x).replace("新品", "").replace("新\u54c1", "").replace("未開封","").replace("未开封","")
        return to_int_yen(s)

    price_new   = df["price"].map(_price_from_shop10)
    recorded_at = df["time-scraped"].map(parse_dt_aware)

    groups = (
        info_df.groupby(["model_name_norm", "capacity_gb"])["part_number"]
        .apply(list).to_dict()
    )

    rows = []
    for i in range(len(df)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p = price_new.iat[i]
        t = recorded_at.iat[i]
        if not m or pd.isna(c) or p is None:
            continue
        pn_list = groups.get((m, int(c)), [])
        if not pn_list:
            continue
        for pn in pn_list:
            rows.append({
                "part_number": str(pn),
                "shop_name": "ドラゴンモバイル",
                "price_new": int(p),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

from dateutil import parser as dateparser

# -------- shop11 专用：金额解析（支持全角、千位逗号、带円/¥ 等） ----------
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

# -------- shop11 专用：颜色差额解析（返回列表 (label_raw, delta_int)） ----------
# 分隔色名的 token（也用于分割像 "シルバー・ブルー"）
_COLOR_SPLIT_RE = re.compile(r"[／/、，,・\s]+")
# 匹配 label 与 +/- 金额的主要正则（类似 shop14 的写法，但更宽容）
_COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\+\s/、，,・\(\)]+?)    # label：不应包含分隔符或括号
        \s*(?:[:：\s])\s*                      # 分隔符（：或:或冒号或空格）— 宽松匹配
        (?P<sign>[+\-−－])?\s*                 # 可选符号
        (?P<amount>[\d,]+|[０-９，]+)          # 金额（半角或全角，含千位分隔符）
        (?:\s*円|\s*¥|\s*￥)?                  # 可选货币词
    """,
    re.UNICODE | re.VERBOSE,
)

# 回退匹配：像 "ブルー -4000" 或 "ブルー-4000"（空格或无空格）
_COLOR_DELTA_FALLBACK_RE = re.compile(
    r"""(?P<label>[^+\-−－\d¥￥円/、，,・\(\)]+?)   # label 前缀（尽量不包含数字或 +/-）
        [\s]*?(?P<sign>[+\-−－])\s*               # 明确的 +/-
        (?P<amount>[\d,]+|[０-９，]+)            # 金额
        (?:\s*円|\s*¥|\s*￥)?                    # 可选 '円'
    """,
    re.UNICODE | re.VERBOSE,
)

def _normalize_digits_for_amount(txt: str) -> str:
    """把全角数字与全角逗号换成半角，去掉非数字字符（除了千分符）"""
    if txt is None:
        return ""
    trans = str.maketrans({'０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9','，':','})
    return (str(txt).translate(trans)).strip()


# -------- color map 构建与匹配（使用 info_df 的 color 字段） ----------
# 一个简单的同义词映射，便于匹配英文/日文/中文标签（可扩展）
FAMILY_SYNONYMS_shop11 = {
    "blue": ["ブルー", "青", "blue"],
    "black": ["ブラック", "黒", "black"],
    "white": ["ホワイト", "白", "white"],
    "silver": ["シルバー", "銀", "silver"],
    "gold": ["ゴールド", "金", "gold"],
    "orange": ["オレンジ", "橙"],
    "pink": ["ピンク"],
    "green": ["グリーン", "緑"],
    "red": ["レッド", "赤"],
    "yellow": ["イエロー", "黄"],
    "purple": ["パープル", "紫"],
    "gray": ["グレー", "灰", "グレイ"],
}

# ----------------- 替换 / 新实现开始 -----------------
import re
from typing import List, Tuple, Dict, Optional

# 用于把全角数字/标点转半角
_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

def _normalize_number_text(txt: str) -> str:
    if txt is None:
        return ""
    return str(txt).translate(_FZ_TO_HZ_TRANS).strip()

# 分隔颜色的字符（包含常见全角/半角分隔）
_COLOR_SEP_SPLIT_RE = re.compile(r"[／/、，,・\s]+")  # split labels by these

# 捕获 label-group 和 amount 的更宽松正则，
# 允许 label 中包含分隔符（我们后续再 split），并允许 amount 有千位逗号或全角数字
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

# 回退形式：label ... -NNNN 或 label-NNNN（例如 "ブルー -4000" / "シルバー-4000"）
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

# 更健壮的 label->color 匹配
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

# -------- 主清洗器 ----------
@register_cleaner("shop11")
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

@register_cleaner("shop9")
def clean_shop9(df: pd.DataFrame) -> pd.DataFrame:
    print("shop9:アキモバ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    增强的 shop9 清洗器，支持 price 字段中包含：
      - 绝对价（例：'橙175,000/青,銀174,000'）
      - 差额（例：'青,黒-2,000円'）
      - 全色（'全色'）
    输出列与以前一致：part_number, shop_name (固定 'アキモバ'), price_new, recorded_at
    """
    info_df = _load_iphone17_info_df_for_shop2()

    col_model = "機種名"
    col_price = "買取価格"
    col_time  = "time-scraped"
    for need in (col_model, col_price, col_time):
        if need not in df.columns:
            raise ValueError(f"shop9 清洗器缺少必要列：{need}")

    # --- helpers ----------------------------------------------------------
    SPLIT_LABELS = re.compile(r"[／/、，,・\s]+")  # 分隔多个颜色标签的符号集

    # 把全角数字/逗号/符号等做成半角并返回整数（或 None）
    def _norm_amount_to_int(s: str) -> Optional[int]:
        if s is None:
            return None
        # 简单半角化和去噪
        tt = str(s).replace("　", " ").replace("，", ",").replace("．", ".")
        # remove full-width digits -> use translate map if available, else fallback
        tt = tt.translate(str.maketrans({
            '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
            '－':'-','＋':'+','¥':'','￥':''
        }))
        m = re.search(r"([0-9][0-9,]*)", tt)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None

    # 从文本中提取绝对价：label + amount（仅在片段中没有 +/- 时视作绝对价）
    ABS_PRICE_RE = re.compile(
        r"""(?P<label>[^：:\-\+\s/、，,・¥￥円]+?)\s*(?:[:：]?\s*)        # 标签
            (?:(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)(?:円)?)  # 金额
        """,
        re.VERBOSE | re.UNICODE
    )

    # 从文本中提取差额：标签 + (必带 + 或 -) + 数字
    DELTA_RE = re.compile(
        r"""(?P<labels>[^+\-−－\d¥￥円]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
        re.VERBOSE | re.UNICODE
    )

    def _extract_abs_prices(text: str) -> List[Tuple[str, int]]:
        """返回 [(label_raw, abs_price_int), ...]。仅捕获那些看起来像绝对价的片段。"""
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        # split by common separators, then test each chunk for ABS_PRICE_RE without +/- signs
        parts = [p.strip() for p in re.split(r"[／/、，,;；\n]", s) if p and p.strip()]
        for part in parts:
            if "+" in part or "-" in part or "－" in part or "−" in part:
                # 若包含 +/-，跳过此片段（交给差额解析）
                continue
            # 可能是 "橙175,000" 或 "青,銀174,000"（需要拆标签）
            m = ABS_PRICE_RE.search(part)
            if not m:
                continue
            amount = _norm_amount_to_int(m.group("amount"))
            if amount is None:
                continue
            labels_part = m.group("label")
            # labels_part 可能含多个标签，用逗号或其他分隔符拆
            toks = [t.strip() for t in re.split(r"[／/、，,・\s]+", labels_part) if t.strip()]
            for tok in toks:
                out.append((tok, int(amount)))
        return out

    def _extract_deltas(text: str) -> List[Tuple[str, int]]:
        """返回 [(label_raw, delta_int), ...]。处理 '青,黒-2,000円' / 'シルバー-3,000/ディープブルー-3,000' 等情况。"""
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        # 全局查找所有带符号的匹配（支持多标签共享同一金额，如 'シルバー/ディープブルー-3000'）
        for m in DELTA_RE.finditer(s):
            labels_part = m.group("labels") or ""
            sign = m.group("sign") or "+"
            amt_txt = m.group("amount")
            amt = _norm_amount_to_int(amt_txt)
            if amt is None:
                continue
            if sign in ("-", "−", "－"):
                delta = -int(amt)
            else:
                delta = int(amt)
            # labels_part 可能包含多个标签
            toks = [t.strip() for t in re.split(r"[／/、，,・\s]+", labels_part) if t.strip()]
            if not toks:
                continue
            for tok in toks:
                out.append((tok, delta))
        # 退化处理：如果没有找到任何带符号的匹配，但字符串中包含 '全色'，返回全色 0（或查找后面是否带数）
        if not out and "全色" in s:
            # 查看是否有数值附带（例如 '全色-1000' 的情况已被上面捕获）
            m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)", s)
            if m:
                sign = m.group(1) or "+"
                amt = _norm_amount_to_int(m.group(2))
                if amt is None:
                    amt = 0
                out.append(("全色", -amt if sign in ("-", "−", "－") else amt))
            else:
                out.append(("全色", 0))
        return out

    # ---------------------------------------------------------------------

    # 先把 info_df 分组，建立 (model_norm, cap) -> { color_norm: part_number }
    info_df2 = info_df.copy()
    info_df2["model_name_norm"] = info_df2["model_name"].map(_normalize_model_generic)
    info_df2["capacity_gb"] = pd.to_numeric(info_df2["capacity_gb"], errors="coerce").astype("Int64")
    info_df2["color_norm"] = info_df2["color"].map(lambda x: _norm(str(x)))

    pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    for _, r in info_df2.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col = r["color_norm"]
        pn = str(r["part_number"])
        if pd.isna(cap) or not m or not col:
            continue
        key = (m, int(cap))
        pn_map.setdefault(key, {})
        pn_map[key][col] = pn

    # 处理输入列
    model_norm = df[col_model].map(_normalize_model_generic)
    cap_gb     = df[col_model].map(_parse_capacity_gb)
    recorded_at = df[col_time].map(lambda x: parse_dt_aware(x))  # shop9 你说不需要时区修正，但 parse_dt_aware 会处理，保留即可

    rows = []
    for i in range(len(df)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        t = recorded_at.iat[i]
        raw_price_cell = df[col_price].iat[i]

        if not m or pd.isna(c):
            continue
        c = int(c)

        key = (m, c)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            # info 表中没有该机型/容量，跳过
            continue

        s = str(raw_price_cell) if raw_price_cell is not None else ""
        # 先尝试解析绝对价（优先）
        abs_list = _extract_abs_prices(s)  # [(label, price), ...]
        deltas = _extract_deltas(s)       # [(label, delta), ...]
        base_price = to_int_yen(s)        # 基准价（若文本里有明显基价，会返回第一个数）
        # 优化逻辑：
        # - 如果 abs_list 存在，则对匹配到的颜色使用绝对价；未出现的颜色使用 base_price（若无 base，则跳过那些未给价的颜色）
        # - 否则若 deltas 中包含 ('全色', val) 或 '全色'，对所有颜色用 base + val
        # - 否则对有 delta 的颜色采用 base+delta，其他颜色采用 base
        color_abs_map: Dict[str, int] = {}
        color_delta_map: Dict[str, int] = {}
        # fill abs map: 把 label 映射到 color_norm
        if abs_list:
            for label_raw, amt in abs_list:
                lbl_norm = _norm(label_raw)
                # label 可能是多个逗号分隔形式，拆开再映射
                for tok in [t for t in re.split(r"[／/、，,・\s]+", label_raw) if t and t.strip()]:
                    n = _norm(tok)
                    # 找到 info_df 中与之匹配的颜色 key（宽松匹配：直接相等或子串）
                    matched = None
                    for col_norm, pn in color_to_pn.items():
                        if n == col_norm or tok in col_norm or tok in pn or tok in str(col_norm):
                            matched = col_norm
                            break
                        # 也可匹配 unicode exact substring of raw color, but we use available col_norm keys
                    if matched:
                        color_abs_map[matched] = int(amt)
            # 可能 abs_list 给到的是部分颜色；对未给到绝对价但 info 中有颜色，若 base_price 存在则回退到 base_price
        if deltas:
            for label_raw, delta in deltas:
                if label_raw == "全色":
                    color_delta_map["ALL"] = int(delta)
                    continue
                for tok in [t for t in re.split(r"[／/、，,・\s]+", label_raw) if t and t.strip()]:
                    n = _norm(tok)
                    matched = None
                    for col_norm in color_to_pn.keys():
                        # 宽松匹配：归一化相同或标签是颜色原文的子串
                        if n == col_norm or tok in col_norm or tok in str(col_norm):
                            matched = col_norm
                            break
                    if matched:
                        color_delta_map[matched] = int(delta)

        # 生成输出行
        # 1) 如果存在 ALL（全色），以 base + ALL（若 base 缺失则跳过）
        if "ALL" in color_delta_map:
            if base_price is None:
                # 没有基价但声明了全色差额，无法计算 -> 跳过整个机型
                continue
            final = int(base_price + color_delta_map["ALL"])
            for col_norm, pn in color_to_pn.items():
                rows.append({
                    "part_number": pn,
                    "shop_name": "アキモバ",
                    "price_new": int(final),
                    "recorded_at": t,
                })
            continue

        # 2) 若存在绝对价，优先使用其对应颜色；其余颜色用 base_price（若 base 不存在则跳过未指定颜色）
        if color_abs_map:
            for col_norm, pn in color_to_pn.items():
                if col_norm in color_abs_map:
                    price_val = color_abs_map[col_norm]
                    rows.append({
                        "part_number": pn,
                        "shop_name": "アキモバ",
                        "price_new": int(price_val),
                        "recorded_at": t,
                    })
                else:
                    # 回退：若 base_price 存在，用 base；否则跳过
                    if base_price is not None:
                        rows.append({
                            "part_number": pn,
                            "shop_name": "アキモバ",
                            "price_new": int(base_price),
                            "recorded_at": t,
                        })
            continue

        # 3) 否则按差额映射（部分颜色可能有 delta）
        if base_price is None:
            # 没有基价也没有绝对价 -> 跳过
            continue
        for col_norm, pn in color_to_pn.items():
            delta = color_delta_map.get(col_norm, 0)
            rows.append({
                "part_number": pn,
                "shop_name": "アキモバ",
                "price_new": int(base_price + delta),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out


# "トゥインクル", # shop12      12
@register_cleaner("shop12")
def clean_shop12(df: pd.DataFrame) -> pd.DataFrame:
    print("shop12:トゥインクル---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop12.csv):
      - モデルナンバー: 含 “iPhone17 Pro 256GB 未開封” 等机型信息；无价或纯标题行将被跳过
      - 備考1: 可能含 “全色” 或 若干 “Blue-2000円 / ブルー：-2000円” 等颜色差额
      - 買取価格: 基础价格（￥180,500 等）
      - time-scraped: 抓取时间
    输出:
      - part_number, shop_name, price_new, recorded_at
    """
    # 必要列检查
    for c in ["モデルナンバー", "備考1", "買取価格", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop12 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

    for _, row in df.iterrows():
        price_base = to_int_yen(row.get("買取価格"))
        if price_base is None:
            # 无价格 → 该行是无用标题/分隔信息，跳过
            continue

        model_text = str(row.get("モデルナンバー") or "").strip()
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
            # 信息表内没有该（机型, 容量）→ 跳过
            continue

        remark = row.get("備考1") or ""
        # 1) 优先识别“全色”场景
        all_delta = _has_all_colors(remark)
        rec_at = parse_dt_aware(row.get("time-scraped"))

        if all_delta is not None:
            # 全色统一价 = 基础价 + 统一 delta（无金额则 delta=0）
            final_price = int(price_base + all_delta)
            for _col_norm, (pn, _col_raw) in color_map.items():
                rows.append({
                    "part_number": pn,
                    "shop_name": "トゥインクル",
                    "price_new": final_price,
                    "recorded_at": rec_at,
                })
            continue

        # 2) 否则解析各色差额
        labels_and_deltas = _extract_color_deltas_shop12(remark)
        color_deltas: Dict[str, int] = {}
        if labels_and_deltas:
            # 将 label 映射到实际颜色
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in labels_and_deltas:
                    if _label_matches_color(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta  # 多个命中时以最后一次为准

        # 3) 生成行：未命中的颜色用基础价
        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_deltas.get(col_norm, 0)
            rows.append({
                "part_number": pn,
                "shop_name": "トゥインクル",
                "price_new": int(price_base + delta),
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out

#家電市場   # shop13           13
@register_cleaner("shop13")
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
      - 通过 _load_iphone17_info_df_for_shop2() 对应（机种，容量）取**所有颜色**的 PN 列表并展开为多行
      - 仅输出在信息表中能匹配到的机型与容量
      - recorded_at = parse_dt_aware(time-scraped)
    """
    # --- 必要列检查 ---
    need_cols = ["新品価格", "買取商品2", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop13 清洗器缺少必要列：{c}")

    # --- 载入 iPhone17 信息（含颜色），并补充归一化机种名 ---
    info_df = _load_iphone17_info_df_for_shop2().copy()
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

    def _price_from_shop13(x: object) -> Optional[int]:
        if x is None:
            return None
        # 去掉常见修饰词，交给 to_int_yen 解析（支持 '円'、'¥'、'～'、'万' 等）
        s = (
            str(x)
            .replace("新品", "")
            .replace("新\u54c1", "")
            .replace("未開封", "")
            .replace("未开封", "")
            .lstrip("～")
        )
        return to_int_yen(s)

    price_new   = df["新品価格"].map(_price_from_shop13)
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



# ---- 颜色差额解析（支持 “青-3000”“ブルー：-2,000円”“橙/銀+1000” 等）----
# 捕获分隔符 sep，以便在 sign 缺省时用 '-' 作为负号
COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s+\+\s+)|(?:\s*;\s*)")

# 英/中/日 颜色族与日文关键词的宽松映射（可按需要扩充）
FAMILY_SYNONYMS_shop14 = {
    # blue family
    "blue": ["ブルー", "青"],
    "ブルー": ["ブルー", "青"],
    "青": ["ブルー", "青"],

    # black
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],

    # white
    "white": ["ホワイト", "白"],
    "ホワイト": ["ホワイト", "白"],
    "白": ["ホワイト", "白"],

    # green
    "green": ["グリーン", "緑"],
    "グリーン": ["グリーン", "緑"],
    "緑": ["グリーン", "緑"],

    # red
    "red": ["レッド", "赤"],
    "レッド": ["レッド", "赤"],
    "赤": ["レッド", "赤"],

    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],

    # purple
    "purple": ["パープル", "紫"],
    "パープル": ["パープル", "紫"],
    "紫": ["パープル", "紫"],

    # yellow
    "yellow": ["イエロー", "黄"],
    "イエロー": ["イエロー", "黄"],
    "黄": ["イエロー", "黄"],

    # orange / silver / gold / gray / natural
    "orange": ["オレンジ", "橙"],
    "オレンジ": ["オレンジ", "橙"],
    "橙": ["オレンジ", "橙"],

    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],

    "gold": ["ゴールド", "金"],
    "ゴールド": ["ゴールド", "金"],
    "金": ["ゴールド", "金"],

    "gray": ["グレー", "グレイ", "灰"],
    "グレー": ["グレー", "グレイ", "灰"],
    "グレイ": ["グレー", "グレイ", "灰"],
    "灰": ["グレー", "グレイ", "灰"],

    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],
}
# 颜色绝对价（无 +/- 号）：'銀206000' / '青 205,500' / 'シルバー：206000円' / '青¥205500'
COLOR_ABS_PRICE_RE = re.compile(
    r"""(?P<label>[^：:\-\s/、／¥円]+)\s*      # 颜色标签（不能以 +/- 开头）
        (?:[:：]?\s*)                         # 可选分隔
        (?:¥|￥)?\s*                          # 可选货币符号
        (?P<amount>\d[\d,]*)\s*               # 金额
        (?:円)?\s*$                           # 可选 '円'
    """,
    re.UNICODE | re.VERBOSE,
)
COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])?\s*          # ← 这里改为可选 ?!
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)
# “全色”统一价解析
def _has_all_colors_shop14(text: str) -> Optional[int]:
    if not text:
        return None
    s = str(text)
    if "全色" not in s:
        return None
    m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*(\d[\d,]*)\s*(円)?", s)
    if m:
        sign = m.group(1) or "+"
        amt = to_int_yen(m.group(2)) or 0
        if sign in ("-", "−", "－"):
            amt = -amt
        return int(amt)
    return 0

def _extract_color_deltas_shop14(text: str) -> List[Tuple[str, int]]:
    """
    从 '减价条件2' 提取若干 (label_raw, delta_int)。
    允许多组，使用 '/', '／', '、', ',', '，', ';' 等分隔。
    例：
      '青-3000'          -> [('青', -3000)]
      '橙/銀+1000'       -> [('橙', +1000), ('銀', +1000)]
      'ブルー：-2,000円' -> [('ブルー', -2000)]
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    # 先分段，再逐段匹配
    parts = [p.strip() for p in SPLIT_TOKENS_RE.split(str(text)) if p and p.strip()]
    for part in parts:
        m = COLOR_DELTA_RE_shop14.search(part)
        if not m:
            continue
        label = m.group("label").strip()
        sep = m.group("sep")
        sign = m.group("sign")
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        # 有显式 sign 用之；否则以分隔符是否为负号判断
        if sign:
            negative = sign in ("-", "−", "－")
        else:
            negative = sep in ("-", "−", "－")
        delta = -int(amt) if negative else int(amt)
        out.append((label, delta))
    return out

def _label_matches_color_shop14(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    宽松匹配 label 是否命中颜色：
    1) 归一化精确相等
    2) label_raw 是 color_raw 的子串
    3) 同义族：label 无论是英文还是日文（如 “blue”“ブルー”“青”“銀”“橙”），
       都先取出该族的“日文关键词集合”，只要其中任意一个出现在 color_raw 中即命中。
    """
    label_norm = _norm(label_raw)

    # 1) 精确相等（归一化后）
    if label_norm == color_norm:
        return True

    # 2) 原文子串
    if label_raw and str(label_raw) in str(color_raw):
        return True

    # 3) 同义族匹配（正向键 + 反向值）
    # 3.1 直接以 label_raw/label_norm 作为键
    keys = {label_raw.strip().lower(), label_norm, label_raw.strip()}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop14:
            candidates.update(FAMILY_SYNONYMS_shop14[k])

    # 3.2 若还没命中，将 label 当作“族内词”去反查家族，再收集该家族的全部关键词
    if not candidates:
        for fam, tokens in FAMILY_SYNONYMS_shop14.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in tokens):
                candidates.update(tokens)
                break

    # 家族里的任一关键词是 color_raw 的子串即可
    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map_shop14(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
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

def _extract_color_abs_prices(text: str) -> List[Tuple[str, int]]:
    """
    抽取若干 (label_raw, abs_price) 绝对价。
    规则：
      - 仅当片段中 **不含** '+' / '-' 才视作绝对价，避免与差额混淆；
      - 支持“多标签共用金额”，如 '青/銀327000'：先缓存 '青'，遇到 '銀327000' 时一并赋值。
    例：
      '銀206000,青205500'     -> [('銀',206000), ('青',205500)]
      '青/銀327000'           -> [('青',327000), ('銀',327000)]
      'シルバー：206000円'     -> [('シルバー',206000)]
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    pending_labels: List[str] = []

    def _norm_label(lbl: str) -> str:
        return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

    for part in [p.strip() for p in SPLIT_TOKENS_RE.split(str(text)) if p and p.strip()]:
        # 有显式 + / - 的片段交给“差额”解析，不在此处理
        if '+' in part or '-' in part or '－' in part or '−' in part:
            continue

        m = COLOR_ABS_PRICE_RE.search(part)
        if m:
            label = _norm_label(m.group("label"))
            amt = to_int_yen(m.group("amount"))
            if label and amt is not None:
                # 当前标签的绝对价
                out.append((label, int(amt)))
                # 把之前缓存的“无金额标签”一并赋值
                for pl in pending_labels:
                    pln = _norm_label(pl)
                    if pln:
                        out.append((pln, int(amt)))
                pending_labels = []  # 清空缓存
            continue

        # 走到这里：这个片段没有金额，可能是“青”或“青/銀”的一段
        for tok in re.split(r"[／/]", part):
            tok = _norm_label(tok)
            if tok:
                pending_labels.append(tok)

    return out


@register_cleaner("shop14")
def clean_shop14(df: pd.DataFrame) -> pd.DataFrame:
    print("shop14:買取楽園---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop14.csv):
      - name: 机型（如 'iPhone17 Pro 256GB'）
      - data6: 'SIM FREE 未開封' / 'SIM FREE　開封'（仅取“未開封”）
      - price2: 基础价格（如 '新品: ¥180,500'）
      - 减价条件2: 颜色差额（如 '青-3000'、'橙/銀+1000'），无则所有颜色同价
      - time-scraped: 抓取时间
    输出:
      - part_number, shop_name(=買取楽園), price_new, recorded_at
    """
    # 必要列检查
    for c in ["name", "data6", "price2", "减价条件2", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop14 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop14(info_df)

    rows: List[dict] = []

    for _, row in df.iterrows():
        status = str(row.get("data6") or "")
        # 只处理 “SIM FREE 未開封”
        if "未開封" not in status:
            continue

        model_text = str(row.get("name") or "").strip()
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
            # 信息表中没有该机型/容量 → 跳过
            continue

        base_price = to_int_yen(row.get("price2"))
        if base_price is None:
            continue
        base_price = int(base_price)

        remark2 = row.get("减价条件2") or ""
        rec_at = parse_dt_aware(row.get("time-scraped"))

        # 先看“全色”
        all_delta = _has_all_colors(remark2)
        if all_delta is not None:
            final_price = base_price + all_delta
            for _col_norm, (pn, _raw) in color_map.items():
                rows.append({
                    "part_number": pn,
                    "shop_name": "買取楽園",
                    "price_new": int(final_price),
                    "recorded_at": rec_at,
                })
            continue

        # 解析颜色“绝对价”和“差额”
        abs_list = _extract_color_abs_prices(remark2)  # [('銀',206000), ('青',205500)]
        labels_and_deltas = _extract_color_deltas(remark2)  # [('銀',-3000), ...]
        color_abs: Dict[str, int] = {}
        color_deltas: Dict[str, int] = {}

        if abs_list:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, abs_price in abs_list:
                    if _label_matches_color(label_raw, col_raw, col_norm):
                        color_abs[col_norm] = abs_price  # 绝对价优先

        if labels_and_deltas:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in labels_and_deltas:
                    if _label_matches_color(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta

        # 生成：若有绝对价，用绝对价；否则用 基准价±差额；都没有则用基准价
        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm in color_abs:
                price_val = color_abs[col_norm]
            else:
                price_val = base_price + color_deltas.get(col_norm, 0)
            rows.append({
                "part_number": pn,
                "shop_name": "買取楽園",
                "price_new": int(price_val),
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out





FIRST_YEN_RE_shop15 = re.compile(r"(\d[\d,]*)\s*円")  # 抓取 price 中第一个 “N円”（作为基准价）
# 颜色差额：'ブルー-9,000円' / 'シルバー　-7,500円' / 'ブルー -4,000'
COLOR_DELTA_IN_PRICE_RE_shop15 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／円¥]+)\s*   # 颜色标签
        (?P<sep>[：:\-])? \s*                # 分隔符（可无）
        (?P<sign>[+\-−－])? \s*              # 正负号（可无）
        (?P<amount>\d[\d,]*) \s* (?:円)?     # 金额（可跟円）
    """,
    re.UNICODE | re.VERBOSE,
)
SPLIT_TOKENS_RE_shop15 = re.compile(r"[／/、，,]|(?:\s*;\s*)")


# 颜色家族同义词（含英文键与日文键，便于“青/銀/橙”等反向命中）
FAMILY_SYNONYMS_shop15= {
    # blue
    # green
    "green": ["グリーン", "緑"],
    "グリーン": ["グリーン", "緑"],
    "緑": ["グリーン", "緑"],
    # red
    "red": ["レッド", "赤"],
    "レッド": ["レッド", "赤"],
    "赤": ["レッド", "赤"],
    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],
    # purple
    "purple": ["パープル", "紫"],
    "パープル": ["パープル", "紫"],
    "紫": ["パープル", "紫"],
    # yellow
    "yellow": ["イエロー", "黄"],
    "イエロー": ["イエロー", "黄"],
    "黄": ["イエロー", "黄"],
    # orange
    "orange": ["オレンジ", "橙"],
    "オレンジ": ["オレンジ", "橙"],
    "橙": ["オレンジ", "橙"],
    # silver
    "金": ["ゴールド", "金"],
    # gray
    "グレイ": ["グレー", "グレイ", "灰"],
    "灰": ["グレー", "グレイ", "灰"],
    # natural
    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],
    # blue
    "blue": ["ブルー", "青", "ディープブルー", "スカイブルー", "ミストブルー", "マリン"],
    "ブルー": ["ブルー", "青", "ディープブルー", "スカイブルー", "ミストブルー", "マリン"],
    "青": ["ブルー", "青", "ディープブルー", "スカイブルー", "ミストブルー", "マリン"],
    "ディープブルー": ["ディープブルー", "ブルー", "青"],
    # silver
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],
    # 其他常见
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],
    "white": ["ホワイト", "白", "スターライト"],
    "ホワイト": ["ホワイト", "白", "スターライト"],
    "白": ["ホワイト", "白", "スターライト"],
    "スターライト": ["ホワイト", "白", "スターライト"],
    "gold": ["ゴールド", "金", "ライトゴールド"],
    "ゴールド": ["ゴールド", "金", "ライトゴールド"],
    "ライトゴールド": ["ゴールド", "金", "ライトゴールド"],
    "gray": ["グレー", "グレイ", "灰"],
    "グレー": ["グレー", "グレイ", "灰"],
}

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

@register_cleaner("shop15")
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

# ---------- 颜色词族 / 解析 ----------
# 价格中的首个“￥/円”作为基础价
FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")
# “颜色±金额”片段，例如：'青 -5000'、'青-5000'、'オレンジ：-5000'、'黒/白-1000'
COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*      # 颜色标签
        (?P<sep>[：:\-])?\s*                    # 分隔符，可空
        (?P<sign>[+\-−－])?\s*                  # 正负号，可空
        (?P<amount>\d[\d,]*)\s*(?:円|￥)?       # 金额
    """, re.UNICODE | re.VERBOSE
)
# “颜色￥绝对价”片段，例如：'黒￥86100'、'青￥87100'
COLOR_ABS_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*￥\s*(?P<amount>\d[\d,]*)""",
    re.UNICODE
)
SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s*;\s*)")

# 颜色家族同义词（含英文、日文短标签反向映射）
FAMILY_SYNONYMS_shop16 = {
    # blue
    "blue": ["ブルー", "青", "マリン"],
    "ブルー": ["ブルー", "青", "マリン"],
    "青": ["ブルー", "青", "マリン"],
    "マリン": ["ブルー", "青", "マリン"],
    # black
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],
    # white
    "white": ["ホワイト", "白"],
    "ホワイト": ["ホワイト", "白"],
    "白": ["ホワイト", "白"],
    # green
    "green": ["グリーン", "緑"],
    "グリーン": ["グリーン", "緑"],
    "緑": ["グリーン", "緑"],
    # red
    "red": ["レッド", "赤"],
    "レッド": ["レッド", "赤"],
    "赤": ["レッド", "赤"],
    # yellow
    "yellow": ["イエロー", "黄"],
    "イエロー": ["イエロー", "黄"],
    "黄": ["イエロー", "黄"],
    # orange
    "orange": ["オレンジ", "橙"],
    "オレンジ": ["オレンジ", "橙"],
    "橙": ["オレンジ", "橙"],
    # silver
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],
    # gold
    "gold": ["ゴールド", "金"],
    "ゴールド": ["ゴールド", "金"],
    "金": ["ゴールド", "金"],
    # gray
    "gray": ["グレー", "グレイ", "灰"],
    "グレー": ["グレー", "グレイ", "灰"],
    "グレイ": ["グレー", "グレイ", "灰"],
    "灰": ["グレー", "グレイ", "灰"],
    # natural
    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],
}

def _label_matches_color_shop16(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配：精确(归一) | 原文子串 | 颜色家族关键词命中"""
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop16:
            candidates.update(FAMILY_SYNONYMS_shop16[k])
    if not candidates:
        for _, toks in FAMILY_SYNONYMS_shop16.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break
    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map_shop16(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """(model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }"""
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

def _extract_base_price_shop16(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)  # 兜底
    return to_int_yen(m.group(1))

def _extract_color_deltas_shop16(text: str) -> List[Tuple[str, int]]:
    """从价格串中抽取多段“颜色±金额”，支持 '青/オレンジ -5000' 这类多标签共用金额。"""
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text)
    # 去掉第一个“基础价 N円/￥N”
    m0 = FIRST_YEN_RE.search(s)
    tail = s[m0.end():] if m0 else s

    parts = [p.strip() for p in SPLIT_TOKENS_RE.split(tail) if p and p.strip()]
    if not parts:
        parts = [tail.strip()]

    pending_labels: List[str] = []  # 暂存像 '青/オレンジ -5000' 中的前置标签（如 '青'）

    def _normalize_label(lbl: str) -> str:
        # 去掉各种空白（含全角空格/不间断空格）
        return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

    for part in parts:
        # 该片段是否包含“颜色±金额”
        matches = list(COLOR_DELTA_RE.finditer(part))
        if matches:
            for m in matches:
                label = _normalize_label(m.group("label"))
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

                # 当前标签
                out.append((label, delta))
                # 把之前挂起的标签，也应用同一金额
                for pl in pending_labels:
                    out.append((_normalize_label(pl), delta))
            pending_labels = []  # 清空缓存
            continue

        # 否则，这是“只有标签没有金额”的片段（如 '青'）；缓存它，等待后面的金额
        # 如果是 '青/橙' 没被上层 split 掉，也进一步按斜杠切一下
        for tok in re.split(r"[／/]", part):
            tok = _normalize_label(tok)
            if tok:
                pending_labels.append(tok)

    return out

def _extract_color_abs_prices_shop16(text: str) -> List[Tuple[str, int]]:
    """从价格串中抽取“颜色￥绝对价”，如：'黒￥86100/青￥87100'"""
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    for m in COLOR_ABS_RE.finditer(str(text)):
        label = (m.group("label") or "").strip()
        amt = to_int_yen(m.group("amount"))
        if label and amt is not None:
            out.append((label, int(amt)))
    return out

# ---------- 主清洗器 ----------
MODEL_COL = "iPhone 17 Pro Max"     # 该列承载“机型标题/机型+容量/SIMFREE 開封”等
DESC_COL  = "説明1"                  # ‘SIMFREE 未開封/開封’ 常在此列（未開封才需要）
PRICE_COL = "買取価格"

@register_cleaner("shop16")
def clean_shop16(df: pd.DataFrame) -> pd.DataFrame:
    print("shop16:携帯空間---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop16.csv):
      - MODEL_COL: 既可能是分组标题，也可能是 'iPhone 17 Pro/Max 256GB' 等
      - 説明1: 'SIMFREE 未開封' / 'SIMFREE 開封'（仅取“未開封”）
      - 買取価格: 基础价格；同格或后随文本里可能带“颜色±金额”或“颜色￥绝对价”
      - time-scraped: 抓取时间
    输出:
      - part_number, shop_name(=携帯空間), price_new, recorded_at
    仅输出 _load_iphone17_info_df_for_shop2() 存在的机型/容量/颜色。
    """
    # 必要列
    for c in [MODEL_COL, DESC_COL, PRICE_COL, "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop16 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop16(info_df)

    rows: List[dict] = []

    # 状态机：最近一次出现的“机型+容量”文本（用于容错，但本页未用到；仅按行内解析）
    for _, row in df.iterrows():
        model_cell = str(row.get(MODEL_COL) or "").strip()
        desc_cell  = str(row.get(DESC_COL)  or "").strip()
        price_cell = row.get(PRICE_COL)
        rec_at     = parse_dt_aware(row.get("time-scraped"))

        # 只处理“未開封”行（开封或空都跳过）
        # 备注：有些“開封”行把价格放在 説明1，但我们整体忽略开封价
        is_unopened = ("未開封" in desc_cell) or ("未開封" in model_cell)
        if not is_unopened:
            continue

        # 从 MODEL_COL 抽取机型和容量（可能含换行/空白）
        model_text = model_cell.replace("\u3000", " ").replace("\xa0", " ").replace("\n", " ").strip()
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            # 若该行 MODEL_COL 只是“iPhone 17 Pro Max / 説明 / 買取価格”等标题，cap 解析会失败
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            # 信息表没有该（机型, 容量），跳过
            continue

        # 基础价（在買取価格列；若异常则跳过）
        base_price = _extract_base_price_shop16(str(price_cell) if price_cell is not None else "")
        if base_price is None:
            continue
        base_price = int(base_price)

        # 解析同格里的“颜色±金额”与“颜色￥绝对价”
        deltas = _extract_color_deltas_shop16(str(price_cell))
        absps  = _extract_color_abs_prices_shop16(str(price_cell))

        # 若出现“颜色￥绝对价”，优先使用绝对价；否则使用 base ± delta
        # 把标签映射到具体 color_norm
        color_delta_map: Dict[str, int] = {}
        color_abs_map: Dict[str, int] = {}

        if deltas:
            for col_norm, (_pn, col_raw) in color_map.items():
                for label_raw, delta in deltas:
                    if _label_matches_color_shop16(label_raw, col_raw, col_norm):
                        color_delta_map[col_norm] = delta  # 多命中时以后者为准

        if absps:
            for col_norm, (_pn, col_raw) in color_map.items():
                for label_raw, abs_price in absps:
                    if _label_matches_color_shop16(label_raw, col_raw, col_norm):
                        color_abs_map[col_norm] = abs_price  # 绝对价优先

        # 生成输出
        for col_norm, (pn, _col_raw) in color_map.items():
            if col_norm in color_abs_map:
                price_new = color_abs_map[col_norm]
            else:
                delta = color_delta_map.get(col_norm, 0)
                price_new = base_price + delta

            rows.append({
                "part_number": pn,
                "shop_name": "携帯空間",
                "price_new": int(price_new),
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out

# ---- 常量：可覆盖为正式店名；默认用域名 ----
SHOP_NAME_OVERRIDE: Optional[str] = "ゲストモバイル"  # 例如 "ゲストモバイル"

# ---- 颜色家族同义词（覆盖日英常见写法）----
FAMILY_SYNONYMS_shop17 = {
    # blue 家族
    "blue": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "ブルー": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "青": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "ミッドナイト": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "マリン": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],

    # black
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],

    # white / starlight
    "white": ["ホワイト", "白", "スターライト", "Starlight", "starlight"],
    "ホワイト": ["ホワイト", "白", "スターライト"],
    "白": ["ホワイト", "白", "スターライト"],
    "スターライト": ["ホワイト", "白", "スターライト"],
    "starlight": ["ホワイト", "白", "スターライト"],

    # silver
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],

    # gold / light gold
    "gold": ["ゴールド", "金", "ライトゴールド"],
    "ゴールド": ["ゴールド", "金", "ライトゴールド"],
    "ライトゴールド": ["ゴールド", "金", "ライトゴールド"],

    # orange
    "orange": ["オレンジ", "橙"],
    "オレンジ": ["オレンジ", "橙"],
    "橙": ["オレンジ", "橙"],

    # green / セージ
    "green": ["グリーン", "緑", "セージ"],
    "グリーン": ["グリーン", "緑", "セージ"],
    "緑": ["グリーン", "緑", "セージ"],
    "セージ": ["グリーン", "緑", "セージ"],

    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],

    # yellow
    "yellow": ["イエロー", "黄", "黄色"],
    "イエロー": ["イエロー", "黄", "黄色"],
    "黄": ["イエロー", "黄", "黄色"],
    "黄色": ["イエロー", "黄", "黄色"],

    # purple / lavender
    "purple": ["パープル", "紫", "ラベンダー"],
    "パープル": ["パープル", "紫", "ラベンダー"],
    "紫": ["パープル", "紫", "ラベンダー"],
    "ラベンダー": ["パープル", "紫", "ラベンダー"],

    # natural
    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],

    # space black
    "spaceblack": ["スペースブラック"],
    "スペースブラック": ["スペースブラック"],
}

def _label_matches_color_shop17(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配：精确(归一) | 原文子串 | 颜色家族关键词命中"""
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop17:
            candidates.update(FAMILY_SYNONYMS_shop17[k])
    if not candidates:
        for _, toks in FAMILY_SYNONYMS_shop17.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break
    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map_shop17(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """(model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }"""
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

# ---- 从 色減額 提取“未開封”段落（若存在）----
def _pick_unopened_section(text: str) -> str:
    """若包含【未開封】…，取该段直到下一个 '【' 或行末；否则返回原文。"""
    if not text:
        return ""
    s = str(text)
    m = re.search(r"【\s*未開封\s*】(.*?)(?=【|$)", s, flags=re.DOTALL)
    return m.group(1) if m else s

# ---- 颜色差额解析 ----
# 支持：'シルバー-3000' / 'Blue-3000' / 'Black(黒)-1000' / 'スターライト -1000' / 'ホワイト-なし'
COLOR_DELTA_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# 'ラベル-なし' / 'ラベル：なし'
COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])\s*
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

SPLIT_TOKENS_RE_shop17 = re.compile(r"[／/、，,]|(?:\s*;\s*)|\n")

def _normalize_label_shop17(lbl: str) -> str:
    return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

def _extract_color_deltas_shop17(text: str) -> List[Tuple[str, int]]:
    """
    从 '色減額' 文本提取 [(label_raw, delta_int)]。
    规则：
      - 若存在【未開封】段，仅解析该段；否则解析全文。
      - 'ラベル-なし' 视为 delta=0；单独整段 'なし/減額なし' 才视为无任何差额。
      - 多段、多色均可；当 sign 缺省且 sep='-' 时视为负数。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = _pick_unopened_section(str(text))
    if "色減額" in s:
        s = s.split("色減額", 1)[-1]
        s = s.lstrip(":：")

    # ✅ 仅当整段就是「なし / 減額なし」才早退
    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        return out

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop17.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    for part in parts:
        # 'ラベル-なし' / 'ラベル：なし' -> delta = 0
        m0 = COLOR_NONE_RE_shop17.search(part)
        if m0:
            label = _normalize_label_shop17(m0.group("label"))
            if label:
                out.append((label, 0))
            continue

        # 'ラベル ± 金額'
        for m in COLOR_DELTA_RE_shop17.finditer(part):
            label = _normalize_label_shop17(m.group("label"))
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

@register_cleaner("shop17")
def clean_shop17(df: pd.DataFrame) -> pd.DataFrame:
    print("shop17:ゲストモバイル---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    输入 (shop17.csv):
      - type: 机型（如 'iPhone 17 Pro 256GB'）
      - 新未開封品: 基础价格（如 '180000円'）
      - 色減額: 颜色差额说明（可能含多段与其它文案；仅取颜色差额，优先【未開封】段）
      - time-scraped: 抓取时间
      - web-scraper-start-url: 用于派生 shop_name（或用 SHOP_NAME_OVERRIDE）
    输出：
      - part_number, shop_name, price_new, recorded_at
    """
    # 必要列检查
    for c in ["type", "新未開封品", "色減額", "time-scraped", "web-scraper-start-url"]:
        if c not in df.columns:
            raise ValueError(f"shop20 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop17(info_df)

    rows: List[dict] = []

    for _, row in df.iterrows():
        model_text = str(row.get("type") or "").strip()
        if not model_text:
            continue

        # 机型解析
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            # 信息表没有该（机型, 容量）→ 跳过
            continue

        # 基础价（新未開封品）
        base_price = to_int_yen(row.get("新未開封品"))
        if base_price is None:
            continue
        base_price = int(base_price)

        # 颜色差额
        labels_and_deltas = _extract_color_deltas_shop17(row.get("色減額"))
        color_deltas: Dict[str, int] = {}
        if labels_and_deltas:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in labels_and_deltas:
                    if _label_matches_color_shop17(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta  # 多命中时以后者为准

        # shop_name
        if SHOP_NAME_OVERRIDE:
            shop_name = SHOP_NAME_OVERRIDE
        else:
            start_url = str(row.get("web-scraper-start-url") or "")
            netloc = urlparse(start_url).netloc or "shop17"
            shop_name = netloc

        rec_at = parse_dt_aware(row.get("time-scraped"))

        # 生成输出
        # 若无任何颜色差额命中 → 全部用基础价
        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_deltas.get(col_norm, 0)
            rows.append({
                "part_number": pn,
                "shop_name": "ゲストモバイル",
                "price_new": int(base_price + delta),
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    return out
#
#


# ---- 常量：可在此处覆盖店名；默认用域名当 shop_name ----
SHOP_NAME_OVERRIDE: Optional[str] = "買取オク"  # 例如： "奥…（正式店名）"

# ---- JAN 提取 ----
JAN_DIGITS_RE = re.compile(r"(\d{8,})")  # 抓取连续 8+ 位数字

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

@register_cleaner("shop18")
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
    need_cols = ["jan", "type", "price", "time-scraped", "web-scraper-start-url"]
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



import json
JAN_RE = re.compile(r"(\d{8,})")

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


def _extract_jan_digits_shop20(s: object) -> Optional[str]:
    if s is None:
        return None
    m = JAN_RE.search(str(s))
    return m.group(1) if m else None

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

@register_cleaner("shop20")
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

JAN_RE_shop1 = re.compile(r"(\d{8,})")

def _extract_jan_digits_shop1(v) -> Optional[str]:
    if v is None:
        return None
    m = JAN_RE_shop1.search(str(v))
    return m.group(1) if m else None

def _pick_info_jan_col_shop1(info_df: pd.DataFrame) -> Optional[str]:
    for c in info_df.columns:
        if str(c).strip().lower() in {"jan", "jancode", "jan_code"}:
            return c
    return None

def _build_jan_to_pn_map_shop1(info_df: pd.DataFrame) -> Dict[str, str]:
    jan_map: Dict[str, str] = {}
    jcol = _pick_info_jan_col_shop1(info_df)
    if not jcol:
        return jan_map
    for _, r in info_df.iterrows():
        jan_digits = _extract_jan_digits_shop1(r.get(jcol))
        pn = r.get("part_number")
        if jan_digits and pd.notna(pn):
            jan_map[str(jan_digits)] = str(pn)
    return jan_map

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

@register_cleaner("shop1")
def clean_shop1(df: pd.DataFrame) -> pd.DataFrame:
    """
    以 JAN 映射 part_number；price -> price_new；time-scraped -> recorded_at。
    shop_name 固定为「買取商店」。
    仅输出 _load_iphone17_info_df_for_shop2() 中存在的机型。
    """
    # 准备 JAN->PN 映射
    info_df = _load_iphone17_info_df_for_shop2()
    jan_map = _build_jan_to_pn_map_shop1(info_df)

    rows: List[dict] = []

    for rec in _iter_records(df):
        jan = _extract_jan_digits_shop1(rec.get("JAN"))

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
