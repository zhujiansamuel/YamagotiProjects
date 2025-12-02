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

COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\s]+)\s*
        (?P<sep>[：:\-])\s*           # 新增：捕获分隔符
        (?P<sign>[+\-−－])?\s*        # 显式正负号（可选）
        (?P<amount>\d[\d,]*)\s*円
    """,
    re.UNICODE | re.VERBOSE,
)

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

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s+\+\s+)|(?:\s*;\s*)")

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*      # 颜色标签
        (?P<sep>[：:\-])?\s*                    # 分隔符，可空
        (?P<sign>[+\-−－])?\s*                  # 正负号，可空
        (?P<amount>\d[\d,]*)\s*(?:円|￥)?       # 金额
    """, re.UNICODE | re.VERBOSE
)

COLOR_ABS_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*￥\s*(?P<amount>\d[\d,]*)""",
    re.UNICODE
)
SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s*;\s*)")

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

MODEL_COL = "iPhone 17 Pro Max"     # 该列承载“机型标题/机型+容量/SIMFREE 開封”等
DESC_COL  = "説明1"                  # ‘SIMFREE 未開封/開封’ 常在此列（未開封才需要）
PRICE_COL = "買取価格"

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
