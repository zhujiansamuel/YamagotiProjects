# AppleStockChecker/utils/external_ingest/base_cleaners.py
from __future__ import annotations
import pandas as pd
from typing import Protocol, Dict, Callable, Optional,List
from ..external_ingest.helpers import to_int_yen, parse_dt_aware
import os,re
from functools import lru_cache
from pathlib import Path


def pick_first_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """
    在 df 里按顺序返回第一列；都没有则返回空 Series。
    用它替代 df.get('A') or df.get('B') 这种写法，避免 Series 布尔判断歧义。
    """
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(dtype=object)



# 目标统一列（清洗后 DataFrame 必须具备以下字段；多余字段会在服务层忽略）：
# required: part_number, shop_name, price_new, recorded_at
# optional: price_grade_a, price_grade_b, shop_address

_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")

# def _normalize_model_generic(text: str) -> str:
#     """
#     归一型号主体：
#       - 'iPhone17ProMax' / 'iPhone 17 Pro Max' → 'iPhone 17 Pro Max'
#       - 'iPhone Air 256GB' / 'iPhoneエアー 256GB' → 'iPhone Air'
#     """
#     if not text:
#         return ""
#     t = str(text).replace("\u3000", " ")
#     t = re.sub(r"\s+", " ", t)
#
#     # 日文后缀 → 英文词干
#     t = (t.replace("プロマックス", "Pro Max")
#            .replace("プロ", "Pro")
#            .replace("プラス", "Plus")
#            .replace("ミニ", "mini")
#            .replace("エアー", "Air")
#            .replace("エア", "Air"))
#
#     # 在 iPhone 与数字/ Air 之间补空格
#     t = re.sub(r"(iPhone)\s*(\d{2})", r"\1 \2", t, flags=re.I)
#     t = re.sub(r"(iPhone)\s*(Air)", r"\1 \2", t, flags=re.I)
#
#     # 去容量/括号/SIM 标记等噪声
#     t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
#     t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
#     t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
#     t = re.sub(r"\s+", " ", t).strip()
#
#     # 1) 数字代号机型
#     m = _NUM_MODEL_PAT.search(t)
#     if m:
#         base = f"{m.group(1)} {m.group(2)}"
#         suf = m.group(3) or ""
#         suf = re.sub(r"\s+", " ", suf).strip()
#         return f"{base} {suf}".strip()
#
#     # 2) iPhone Air（含后缀容错，当前返回 'iPhone Air'）
#     m2 = _AIR_PAT.search(t)
#     if m2:
#         # 如果将来有 'Air Plus' 等，可以改为返回包含后缀；目前返回主体即可
#         return "iPhone Air"
#
#     return ""


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


# def _normalize_model_17(text: str) -> str:
#     """
#     把 data2 中的机型片段归一化：
#       - 'iPhone17ProMax' / 'iPhone 17 Pro Max' -> 'iPhone 17 Pro Max'
#       - 'iPhone17'       -> 'iPhone 17'
#     """
#     if not text:
#         return ""
#     t = str(text).replace("\u3000", " ")
#     t = re.sub(r"\s+", " ", t)
#     # 日文后缀到英文（万一 data2 带日文后缀）
#     t = (t.replace("プロマックス", "Pro Max")
#            .replace("プロ", "Pro")
#            .replace("プラス", "Plus")
#            .replace("ミニ", "mini"))
#     # iPhone 与数字之间补空格
#     t = re.sub(r"(iPhone)\s*(\d{2})", r"\1 \2", t, flags=re.I)
#     m = _MODEL_PAT.search(t)
#     if not m:
#         return ""
#     base = f"{m.group('iphone')} {m.group('num')}"
#     suf = m.group("suf")
#     suf = re.sub(r"\s+", " ", suf).strip()
#     return f"{base} {suf}".strip()

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


class Cleaner(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...

CLEANERS: Dict[str, Cleaner] = {}

def register_cleaner(name: str):
    def deco(fn: Cleaner):
        CLEANERS[name] = fn
        return fn
    return deco

PN_REGEX = re.compile(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b")

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

# ------------------------------
# 载入 iphone17_info.csv（part_number 映射）
# ------------------------------
# def _load_iphone17_info(path: str) -> pd.DataFrame:
#     """
#     期望列至少包含：part_number, model_name, capacity_gb
#     允许存在其他列（color、JAN 等），会被忽略。
#     """
#     # 兼容 CSV/Excel
#     if re.search(r"\.(xlsx|xlsm|xls|ods)$", path, re.I):
#         df = pd.read_excel(path)
#     else:
#         df = pd.read_csv(path, encoding="utf-8-sig")
#     for col in ("part_number", "model_name", "capacity_gb"):
#         if col not in df.columns:
#             raise ValueError(f"iphone17_info 缺少必要列：{col}")
#     # 归一：model_name 同框架的规范；capacity 转 int
#     df = df.copy()
#     df["model_name_norm"] = df["model_name"].map(_normalize_model_17)
#     df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
#     df = df.dropna(subset=["model_name_norm", "capacity_gb", "part_number"])
#     return df[["part_number", "model_name_norm", "capacity_gb"]]


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



@register_cleaner("shop3")
def clean_shop3(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入列：
      web-scraper-order, web-scraper-start-url, data4, data5, data6, data8, title, time-scraped
    规则：
      - shop_name 固定 '買取一丁目'
      - title 含“机种名 + 容量” → 归一(model_norm) + 解析容量(capacity_gb)
      - 通过 iphone17_info.csv 对应 (model_norm, capacity_gb) 获取“所有颜色”的 part_number 列表并展开
      - data5 为新品 price_new（解析日元/区间）
      - time-scraped 为 recorded_at（为空行直接跳过）
    输出：
      part_number, shop_name, price_new, recorded_at
    """
    info_df = _load_iphone17_info_df()  # -> part_number, model_name_norm, capacity_gb

    # 必要列检查
    need_cols = ["title", "data5", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop3 清洗器缺少必要列：{c}")

    src = df.copy()

    # 过滤：time-scraped 为空的行跳过，避免时间解析问题
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok]
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # 1) 从 title 解析 机型 & 容量
    model_norm = src["title"].map(_normalize_model_generic)
    cap_gb     = src["title"].map(_parse_capacity_gb)

    # 2) 价格 & 时间
    price_new   = src["data5"].map(_price_from_shop3)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    # 3) (model_norm, capacity_gb) -> 所有颜色的 PN 列表
    groups = (
        info_df.groupby(["model_name_norm", "capacity_gb"])["part_number"]
        .apply(list).to_dict()
    )

    # 4) 展开为多行
    rows: List[dict] = []
    for i in range(len(src)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p = price_new.iat[i]
        t = recorded_at.iat[i]

        # 缺关键字段跳过
        if not m or pd.isna(c) or p is None:
            continue

        pn_list = groups.get((m, int(c)), [])
        if not pn_list:
            # info 映射中无该型号/容量，跳过（服务层 unmatched 会统计）
            continue

        for pn in pn_list:
            rows.append({
                "part_number": str(pn),
                "shop_name": "買取一丁目",
                "price_new": int(p),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop5-1")
def clean_shop5_1(df: pd.DataFrame) -> pd.DataFrame:
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
    info_df = _load_iphone17_info_df()  # part_number, model_name_norm, capacity_gb

    # 必要列检查
    need_cols = ["data", "data2", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop7 清洗器缺少必要列：{c}")

    # 先把 time-scraped 为空的行排除，避免时间解析报错
    df = df.copy()
    mask_time_ok = df["time-scraped"].astype(str).str.strip().ne("") & df["time-scraped"].notna()
    df = df[mask_time_ok]
    if df.empty:
        return pd.DataFrame(columns=["part_number","shop_name","price_new","recorded_at"])

    # data -> 机型&容量
    model_norm = df["data"].map(_norm_model_for_shop7)
    cap_gb     = df["data"].map(_parse_capacity_gb)

    # 价格/时间
    price_new  = df["data2"].map(_price_from_shop7)
    recorded_at= df["time-scraped"].map(parse_dt_aware)

    # (model, cap) -> [PN] 映射
    groups = (
        info_df.groupby(["model_name_norm", "capacity_gb"])["part_number"]
        .apply(list).to_dict()
    )

    rows: List[dict] = []
    for i in range(len(df)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p = price_new.iat[i]
        t = recorded_at.iat[i]

        # 关键信息缺失跳过
        if not m or pd.isna(c) or p is None:
            continue

        pnlst = groups.get((m, int(c)), [])
        if not pnlst:
            # 未在 info 映射中找到同型号同容量的 PN 组，跳过
            continue

        for pn in pnlst:
            rows.append({
                "part_number": str(pn),
                "shop_name": "買取ホムラ",
                "price_new": int(p),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop8")
def clean_shop8(df: pd.DataFrame) -> pd.DataFrame:
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

@register_cleaner("shop9")
def clean_shop9(df: pd.DataFrame) -> pd.DataFrame:
    info_df = _load_iphone17_info_df()

    col_model = "機種名"
    col_price = "買取価格"
    col_time  = "time-scraped"
    for need in (col_model, col_price, col_time):
        if need not in df.columns:
            raise ValueError(f"shop9 清洗器缺少必要列：{need}")

    # ★ 改这里：用通用归一（支持 iPhone Air）
    model_norm = df[col_model].map(_normalize_model_generic)
    cap_gb     = df[col_model].map(_parse_capacity_gb)

    def _price_from_shop9(x):
        if x is None: return None
        s = str(x).replace("新品", "").replace("新\u54c1", "")
        return to_int_yen(s)

    price_new   = df[col_price].map(_price_from_shop9)
    recorded_at = df[col_time].map(parse_dt_aware)

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
                "shop_name": "アキモバ",
                "price_new": int(p),
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
    return out

@register_cleaner("shop10")
def clean_shop10(df: pd.DataFrame) -> pd.DataFrame:
    info_df = _load_iphone17_info_df()

    need_cols = ["data2", "price", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop10 清洗器缺少必要列：{c}")

    # ★ 改这里
    model_norm = df["data2"].map(_normalize_model_generic)
    cap_gb     = df["data2"].map(_parse_capacity_gb)

    def _price_from_shop10(x):
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









#
# # ===== 示例清洗器 A =====
# @register_cleaner("shop9")
# def clean_sample_a(df: pd.DataFrame) -> pd.DataFrame:
#     # 假设这个源的列是：pn, store, new_price, a_price, b_price, time
#     out = pd.DataFrame({
#         "part_number": df.get("pn"),
#         "shop_name": df.get("store"),
#         "price_new": df.get("new_price").map(to_int_yen),
#         "price_grade_a": df.get("a_price").map(to_int_yen),
#         "price_grade_b": df.get("b_price").map(to_int_yen),
#         "recorded_at": df.get("time").map(parse_dt_aware),
#         "shop_address": df.get("store_address") if "store_address" in df.columns else None,
#     })
#     # 去掉缺关键值的行
#     out = out.dropna(subset=["part_number", "shop_name", "price_new"])
#     return out
#
# # ===== 示例清洗器 B =====
# @register_cleaner("shop10")
# def clean_sample_b(df: pd.DataFrame) -> pd.DataFrame:
#     # 假设这个源把价格放在 "price" 一列，A/B 没有，我们只填 price_new
#     out = pd.DataFrame({
#         "part_number": df.get("PartNumber") or df.get("part_number"),
#         "shop_name": df.get("Shop") or df.get("shop"),
#         "price_new": (df.get("price") or df.get("Price")).map(to_int_yen),
#         "price_grade_a": None,
#         "price_grade_b": None,
#         "recorded_at": (df.get("timestamp") or df.get("time")).map(parse_dt_aware),
#         "shop_address": df.get("Address") if "Address" in df.columns else None,
#     })
#     out = out.dropna(subset=["part_number", "shop_name", "price_new"])
#     return out
#
# # ===== 预留占位：你将来为每个 API 写一个 cleaner_xxx =====
# # @register_cleaner("your_api_name")
# # def clean_your_api(df: pd.DataFrame) -> pd.DataFrame:
# #     # TODO: 自定义列映射/逻辑
# #     ...
