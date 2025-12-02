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

_YEN_RE = re.compile(r"[^\d]+")

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
