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
