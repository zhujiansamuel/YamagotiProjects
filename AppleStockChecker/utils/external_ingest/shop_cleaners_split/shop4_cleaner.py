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

_COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\+\s/、，,・\(\)]+?)    # label：不应包含分隔符或括号
        \s*(?:[:：\s])\s*                      # 分隔符（：或:或冒号或空格）— 宽松匹配
        (?P<sign>[+\-−－])?\s*                 # 可选符号
        (?P<amount>[\d,]+|[０-９，]+)          # 金额（半角或全角，含千位分隔符）
        (?:\s*円|\s*¥|\s*￥)?                  # 可选货币词
    """,
    re.UNICODE | re.VERBOSE,
)

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})
