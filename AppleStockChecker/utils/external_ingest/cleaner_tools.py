# AppleStockChecker/utils/external_ingest/cleaner_tools.py
"""
清洗器通用工具模块
提供数据库访问、数据转换等通用功能
"""
from __future__ import annotations
from typing import List, Optional
import pandas as pd
import re


def _load_iphone17_info_df_from_db() -> pd.DataFrame:
    """
    从数据库中读取 iPhone 机型信息，返回 DataFrame

    输出列：part_number, model_name, capacity_gb, color, jan（如果 jan 字段有值）

    Returns:
        pd.DataFrame: 包含 iPhone 机型信息的 DataFrame

    Raises:
        ValueError: 如果数据库中没有 iPhone 数据
    """
    from AppleStockChecker.models import Iphone

    # 查询所有 iPhone 数据，只选择需要的字段
    queryset = Iphone.objects.all().values(
        'part_number',
        'model_name',
        'capacity_gb',
        'color',
        'jan'
    )

    # 转换为 DataFrame
    df = pd.DataFrame.from_records(queryset)

    if df.empty:
        raise ValueError("数据库中没有 iPhone 数据，请先导入 iPhone 机型信息")

    # 确保数据类型正确
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")

    # 删除必要字段为空的行
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number", "color"])

    # 处理 jan 列：如果所有 jan 都是空的，就删除这一列；否则保留
    if df["jan"].isna().all():
        df = df.drop(columns=["jan"])
        cols = ["part_number", "model_name", "capacity_gb", "color"]
    else:
        cols = ["part_number", "model_name", "capacity_gb", "color", "jan"]

    return df[cols].reset_index(drop=True)


# 正则表达式模式用于型号匹配
_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)


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


def _normalize_model_generic(text: str) -> str:
    """
    统一型号主体：
      - iPhone17/16 + 后缀（Pro/Pro Max/Plus/mini）
      - iPhone Air（含"17 air"→ Air）
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

    # 数字后紧跟英文：17pro -> 17 pro
    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)

    # 标准化后缀大小写
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
        return "iPhone Air"

    return ""
