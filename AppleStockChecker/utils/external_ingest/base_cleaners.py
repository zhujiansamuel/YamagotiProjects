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

def _normalize_model_generic(text: str) -> str:
    """
    归一型号主体：
      - 'iPhone17ProMax' / 'iPhone 17 Pro Max' → 'iPhone 17 Pro Max'
      - 'iPhone Air 256GB' / 'iPhoneエアー 256GB' → 'iPhone Air'
    """
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    # 日文后缀 → 英文词干
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini")
           .replace("エアー", "Air")
           .replace("エア", "Air"))

    # 在 iPhone 与数字/ Air 之间补空格
    t = re.sub(r"(iPhone)\s*(\d{2})", r"\1 \2", t, flags=re.I)
    t = re.sub(r"(iPhone)\s*(Air)", r"\1 \2", t, flags=re.I)

    # 去容量/括号/SIM 标记等噪声
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 1) 数字代号机型
    m = _NUM_MODEL_PAT.search(t)
    if m:
        base = f"{m.group(1)} {m.group(2)}"
        suf = m.group(3) or ""
        suf = re.sub(r"\s+", " ", suf).strip()
        return f"{base} {suf}".strip()

    # 2) iPhone Air（含后缀容错，当前返回 'iPhone Air'）
    m2 = _AIR_PAT.search(t)
    if m2:
        # 如果将来有 'Air Plus' 等，可以改为返回包含后缀；目前返回主体即可
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
def _load_iphone17_info(path: str) -> pd.DataFrame:
    """
    期望列至少包含：part_number, model_name, capacity_gb
    允许存在其他列（color、JAN 等），会被忽略。
    """
    # 兼容 CSV/Excel
    if re.search(r"\.(xlsx|xlsm|xls|ods)$", path, re.I):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("part_number", "model_name", "capacity_gb"):
        if col not in df.columns:
            raise ValueError(f"iphone17_info 缺少必要列：{col}")
    # 归一：model_name 同框架的规范；capacity 转 int
    df = df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_17)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name_norm", "capacity_gb", "part_number"])
    return df[["part_number", "model_name_norm", "capacity_gb"]]


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
