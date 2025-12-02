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

COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s+\+\s+)|(?:\s*;\s*)")

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

COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])?\s*          # ← 这里改为可选 ?!
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)

_SPLIT_TOKENS_SAFE_RE = re.compile(
    r"""
    [／/、，]                 # 全角/斜杠类分隔符（始终切分）
    |(?<!\d),(?!\d)          # ASCII 逗号：仅当其两侧不是数字时切分（避免拆千位分隔）
    |(?:\s+\+\s+)            # " + " 形式
    |(?:\s*;\s*)             # 分号
    """,
    re.UNICODE | re.VERBOSE,
)

_COLOR_ABS_PRICE_RE = re.compile(
    r"""^\s*
        (?P<label>[^：:\-\s/、／¥円]+?)    # 颜色标签（非贪心，避免包含金额）
        \s*(?:[:：]?\s*)                   # 可选分隔符
        (?:¥|￥)?\s*                       # 可选货币符号
        (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})*|\d+)  # 支持千位逗号（ASCII or fullwidth）或无逗号数字
        \s*(?:円)?\s*$
    """,
    re.UNICODE | re.VERBOSE,
)

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

def _norm_label(lbl: str) -> str:
    """去除空白并统一全角空格/NBSP，保留原文字顺序用作匹配用 key"""
    if lbl is None:
        return ""
    s = str(lbl)
    # 去掉左右空白并规范全角空格为半角
    s = s.strip().replace("\u3000", " ").replace("\xa0", " ").strip()
    # 把中间多空格合并
    s = re.sub(r"\s+", " ", s)
    return s

def _clean_remark_frag(x) -> str:
    """把单列 remark 做清理：去 None/nan，统一空格，去 BOM，去多余标点尾巴等。"""
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # pandas 的 nan/NaN/None 字符串化为 'nan'，把它当空
    if s.lower() == "nan":
        return ""
    # 去 BOM / 不可见空白
    s = s.lstrip("\ufeff").replace("\u3000", " ")
    # 把多空格压成一个
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_color_abs_prices(text: str) -> List[Tuple[str, int]]:
    """
    从 text 中抽取 (label_raw, abs_price) 绝对价。
    修复点：不会在数字千位分隔符处拆分（保留 229,000 完整）。
    支持多标签共用金额：'青/銀327000'、'青 銀 327000' 等。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    pending_labels: List[str] = []

    # 先把非可见 BOM / nan 文本规范化
    s_all = str(text).strip()
    if s_all.lower() == "nan" or s_all == "":
        return out

    # 逐片段处理（使用更安全的切分）
    parts = [p.strip() for p in _SPLIT_TOKENS_SAFE_RE.split(s_all) if p and p.strip()]
    if not parts:
        parts = [s_all]

    for part in parts:
        # 如果片段里同时含有 + 或 - （显式差额），跳过（差额解析会处理）
        if any(ch in part for ch in ("+", "-", "−", "－")):
            # 但也要考虑像 "青 229,000" 这种包含空格和逗号的正常绝对价 -> 上面条件不会触发
            # 所以这里是安全的
            continue

        m = _COLOR_ABS_PRICE_RE.search(part)
        if m:
            label_raw = _norm_label(m.group("label"))
            amt_txt = m.group("amount")
            # 把千分符去掉（支持 ASCII comma 和 全角逗号）
            amt_clean = re.sub(r"[,\uFF0C]", "", amt_txt)
            try:
                amt_val = int(amt_clean)
            except Exception:
                # 额外容错：用 to_int_yen 作为 fallback（如果你有该工具）
                try:
                    amt_val = int(to_int_yen(amt_txt) or 0)
                except Exception:
                    continue

            if label_raw:
                out.append((label_raw, amt_val))
                # 把 pending 的标签也一并赋值（多标签共用金额情形）
                for pl in pending_labels:
                    pln = _norm_label(pl)
                    if pln:
                        out.append((pln, amt_val))
                pending_labels = []
            continue

        # 没有找到金额：这个片段可能只是标签（或多标签连着）
        # 用斜杠或全角/半角逗号/顿号分割出标签候选
        for tok in re.split(r"[／/、，;；,]", part):
            tok = _norm_label(tok)
            if tok:
                pending_labels.append(tok)

    return out

def clean_shop14(df: pd.DataFrame) -> pd.DataFrame:
    print("shop14:買取楽園---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for c in ["name", "data6", "price2", "减价条件2", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop14 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop14(info_df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        status = str(row.get("data6") or "")
        if "未開封" not in status:
            print(f"[{idx}] skip: data6 不包含 未開封 -> '{status}'")
            continue

        model_text = str(row.get("name") or "").strip()
        if not model_text:
            print(f"[{idx}] skip: name 为空")
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            print(f"[{idx}] skip: 无法解析 model/capacity -> name='{model_text}', model_norm='{model_norm}', cap_gb='{cap_gb}'")
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            print(f"[{idx}] skip: info_df 中无该机型容量 -> {key}")
            continue

        base_price = to_int_yen(row.get("price2"))
        if base_price is None:
            print(f"[{idx}] skip: 无法解析基准价 price2='{row.get('price2')}'")
            continue
        base_price = int(base_price)

        # ===== 清洗并合并 remark 列（过滤掉 nan/空）
        part_a = _clean_remark_frag(row.get("减价条件2"))
        part_b = _clean_remark_frag(row.get("23432")) if "23432" in row.index else ""
        # 也支持旧字段名：若某些文件用不同名字，可以按需加
        combined = " ".join([p for p in (part_a, part_b) if p]).strip()

        rec_at = parse_dt_aware(row.get("time-scraped"))
        #rec_at = row.get("time-scraped")

        print(f"[{idx}] model='{model_text}' -> norm='{model_norm}', cap={cap_gb}, base_price={base_price}, combined_remark='{combined}'")

        # 先看“全色”
        all_delta = _has_all_colors(combined)
        if all_delta is not None:
            final_price = base_price + all_delta
            print(f"[{idx}] 全色调整 detected: all_delta={all_delta}, final_price={final_price}")
            for _col_norm, (pn, _raw) in color_map.items():
                rows.append({
                    "part_number": pn,
                    "shop_name": "買取楽園",
                    "price_new": int(final_price),
                    "recorded_at": rec_at,
                })
            continue

        # 关键点：分别在每个单独列上也尝试解析，并将结果合并
        # 这样像你的例子 'nan 青 229,000'，若 '青 229,000' 在单列里可以命中
        abs_list = []
        labels_and_deltas = []

        # parse each source fragment separately (优先解析每个 fragment)
        for frag in (part_a, part_b):
            if not frag:
                continue
            a = _extract_color_abs_prices(frag)
            d = _extract_color_deltas_shop14(frag)
            if a:
                abs_list.extend(a)
            if d:
                labels_and_deltas.extend(d)

        # 兼容再尝试对合并字符串解析（兜底）
        if not abs_list:
            abs_list = _extract_color_abs_prices(combined)
        if not labels_and_deltas:
            labels_and_deltas = _extract_color_deltas_shop14(combined)

        color_abs: Dict[str, int] = {}
        color_deltas: Dict[str, int] = {}

        print(f"[{idx}] parsed abs_list={abs_list}, labels_and_deltas={labels_and_deltas}")

        # 绝对价匹配
        if abs_list:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, abs_price in abs_list:
                    if _label_matches_color_shop14(label_raw, col_raw, col_norm):
                        color_abs[col_norm] = abs_price
                        print(f"[{idx}] abs match -> color_raw='{col_raw}' (norm={col_norm}) abs_price={abs_price}")

        # 差额匹配
        if labels_and_deltas:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in labels_and_deltas:
                    if _label_matches_color_shop14(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta
                        print(f"[{idx}] delta match -> color_raw='{col_raw}' (norm={col_norm}) delta={delta}")

        # 生成 price
        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm in color_abs:
                price_val = color_abs[col_norm]
                reason = "abs"
            else:
                price_val = base_price + color_deltas.get(col_norm, 0)
                reason = f"base+delta({color_deltas.get(col_norm,0)})" if col_norm in color_deltas else "base"
            print(f"[{idx}] -> color='{col_raw}' (norm={col_norm}) pn={pn} price={price_val} reason={reason}")
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

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s*;\s*)")
