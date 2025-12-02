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

def clean_shop9(df: pd.DataFrame) -> pd.DataFrame:
    import time
    print("shop9:アキモバ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    info_df = _load_iphone17_info_df_for_shop2()

    col_model = "機種名"
    col_price = "買取価格"
    col_color = "色・詳細等"
    col_time  = "time-scraped"

    for need in (col_model, col_price, col_color, col_time):
        if need not in df.columns:
            raise ValueError(f"shop9 清洗器缺少必要列：{need}")

    # 同义表（可扩充）
    FAMILY_SYNONYMS_SHOP9 = {
        "blue": ["ブルー", "青", "ディープブルー", "ディープ ブルー"],
        "ブルー": ["ブルー", "青", "ディープブルー"],
        "青": ["ブルー", "青", "ディープブルー"],
        "ディープブルー": ["ディープブルー", "ブルー", "青"],
        "silver": ["シルバー", "銀"],
        "シルバー": ["シルバー", "銀"],
        "銀": ["シルバー", "銀"],
        "black": ["ブラック", "黒"],
        "ブラック": ["ブラック", "黒"],
        "黒": ["ブラック", "黒"],
        "orange": ["オレンジ", "橙", "コズミックオレンジ"],
        "オレンジ": ["オレンジ", "橙"],
        "橙": ["オレンジ", "橙"],
        "white": ["ホワイト", "白"],
        "ホワイト": ["ホワイト", "白"],
    }
    SYNONYM_LOOKUP = {}
    for k, toks in FAMILY_SYNONYMS_SHOP9.items():
        for t in toks:
            SYNONYM_LOOKUP[_norm(str(t))] = [ _norm(str(x)) for x in toks ]

    # 正则与辅助
    SPLIT_SEPS = r"[／/、，,・\s]+"  # 分隔多个颜色标签的符号集
    # 全局抓取：labels（可以有多个标签） + 金额（允许千分位逗号 & 全角数字）
    GLOBAL_LABEL_AMOUNT_RE = re.compile(
        r"""(?P<labels>(?:[^\d¥￥円/、，,;；]+?(?:[／/、，,・\s]+[^\d¥￥円/、，,;；]+?)*))
            \s*(?:[:：]?\s*)?
            (?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)(?:円)?
        """,
        re.VERBOSE | re.UNICODE,
    )
    # 差额（含符号）
    DELTA_RE = re.compile(
        r"""(?P<labels>[^+\-−－\d¥￥円]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
        re.VERBOSE | re.UNICODE
    )

    def _norm_amount_to_int(s: str) -> Optional[int]:
        if s is None:
            return None
        tt = str(s).replace("　", " ").replace("，", ",").replace("．", ".")
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

    def _is_pure_number_token(tok: str) -> bool:
        tok2 = tok.replace(",", "").replace("，", "").strip()
        return bool(re.fullmatch(r"[0-9,]+", tok2))

    def _extract_abs_prices(text: str) -> List[Tuple[str, int]]:
        """
        使用全局正则抓取 'labels + amount' 的片段（labels 可含多个以 / 、 等分隔）。
        例如：
          '未開 橙230,500/青,銀229,000' -> [('橙',230500), ('青',229000), ('銀',229000)]
        """
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        for m in GLOBAL_LABEL_AMOUNT_RE.finditer(s):
            labels_part = m.group("labels") or ""
            amt_txt = m.group("amount")
            amt = _norm_amount_to_int(amt_txt)
            if amt is None:
                continue
            toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
            for tok in toks:
                if _is_pure_number_token(tok):
                    continue
                out.append((tok, int(amt)))
        # fallback: 若找不到任何 labels+amount，但存在「単独标签」与后面单独金额（少见），
        # 则尝试简单的 "label amount" 的查找（已被 GLOBAL 捕获的大多数会命中）
        if not out:
            # 尝试形式 like '青 229,000'
            m2 = re.finditer(r"(?P<label>[^\d¥￥円/、，,;；]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)", s)
            for m in m2:
                label = m.group("label").strip()
                amt = _norm_amount_to_int(m.group("amount"))
                if label and amt is not None and not _is_pure_number_token(label):
                    out.append((label, int(amt)))
        return out

    def _extract_deltas(text: str) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        for m in DELTA_RE.finditer(s):
            labels_part = m.group("labels") or ""
            sign = m.group("sign") or "+"
            amt_txt = m.group("amount")
            amt = _norm_amount_to_int(amt_txt)
            if amt is None:
                continue
            delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
            toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
            for tok in toks:
                if _is_pure_number_token(tok):
                    continue
                out.append((tok, delta))
        # 全色 fallback
        if not out and "全色" in s:
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

    # build pn map
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

    # process rows
    model_norm = df[col_model].map(_normalize_model_generic)
    cap_gb     = df[col_model].map(_parse_capacity_gb)
    recorded_at = df[col_time].map(lambda x: parse_dt_aware(x))
    # recorded_at = df[col_time]

    rows = []
    for i in range(len(df)):
        raw_model = df[col_model].iat[i]
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        t = recorded_at.iat[i]
        raw_price_cell = df[col_price].iat[i]
        raw_color_cell = df[col_color].iat[i]

        print(f"[DEBUG row={i}] raw_model={raw_model!r} -> norm={m!r}, cap={c!r}, raw_price={raw_price_cell!r}, raw_color={raw_color_cell!r}")

        if not m or pd.isna(c):
            print(f"[DEBUG row={i}] skip: model/cap missing")
            continue
        c = int(c)

        key = (m, c)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            print(f"[DEBUG row={i}] skip: no pn_map for key={key}")
            continue

        s_color = str(raw_color_cell) if raw_color_cell is not None else ""
        s_price = str(raw_price_cell) if raw_price_cell is not None else ""
        # parse from color-col first (优先)
        abs_list = _extract_abs_prices(s_color)
        deltas = _extract_deltas(s_color)
        base_price = to_int_yen(s_price) or to_int_yen(s_color)

        # if not found in color-col, try price-col
        if not abs_list and not deltas:
            abs_list = _extract_abs_prices(s_price)
            deltas = _extract_deltas(s_price)
            if base_price is None:
                base_price = to_int_yen(s_price)

        # final fallback: whole row
        if not abs_list and not deltas:
            full_row_parts = []
            for col in df.columns:
                try:
                    v = df[col].iat[i]
                except Exception:
                    v = df.iloc[i].get(col)
                if v is None:
                    continue
                sv = str(v).strip()
                if sv and sv.lower() != "nan":
                    full_row_parts.append(sv)
            s_full = " ".join(full_row_parts)
            if s_full and s_full != s_color and s_full != s_price:
                print(f"[DEBUG row={i}] fallback parsing from full row: {s_full!r}")
                abs_list = _extract_abs_prices(s_full)
                deltas = _extract_deltas(s_full)
                if base_price is None:
                    base_price = to_int_yen(s_full)

        print(f"[DEBUG row={i}] parsed abs_list={abs_list}, deltas={deltas}, base_price={base_price}")

        # label -> col_norm matching（宽松 + 同义表）
        def _match_label_to_colnorm(tok: str) -> Optional[str]:
            if not tok:
                return None
            tok_norm = _norm(tok)
            # direct equal
            for col_norm in color_to_pn.keys():
                if tok_norm == col_norm:
                    return col_norm
            # synonyms
            candidates = set()
            if tok_norm in SYNONYM_LOOKUP:
                candidates.update(SYNONYM_LOOKUP[tok_norm])
            candidates.add(tok_norm)
            for cand in candidates:
                for col_norm in color_to_pn.keys():
                    if cand == col_norm or cand in col_norm or col_norm in cand:
                        return col_norm
            # fallback substring
            tok_short = re.sub(r"[\s\u3000\-]+", "", tok_norm)
            for col_norm in color_to_pn.keys():
                if tok_short in col_norm or col_norm in tok_short:
                    return col_norm
            return None

        color_abs_map: Dict[str, int] = {}
        color_delta_map: Dict[str, int] = {}

        for label_raw, amt in abs_list:
            toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
            for tok in toks:
                if _is_pure_number_token(tok):
                    print(f"[DEBUG row={i}] abs skip numeric token={tok!r}")
                    continue
                matched = _match_label_to_colnorm(tok)
                if matched:
                    color_abs_map[matched] = int(amt)
                    print(f"[DEBUG row={i}] abs match: token={tok!r} -> color_norm={matched!r} price={amt}")
                else:
                    print(f"[DEBUG row={i}] abs NO-match token={tok!r}")

        for label_raw, delta in deltas:
            if label_raw == "全色":
                color_delta_map["ALL"] = int(delta)
                print(f"[DEBUG row={i}] delta ALL -> {delta}")
                continue
            toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
            for tok in toks:
                if _is_pure_number_token(tok):
                    print(f"[DEBUG row={i}] delta skip numeric token={tok!r}")
                    continue
                matched = _match_label_to_colnorm(tok)
                if matched:
                    color_delta_map[matched] = int(delta)
                    print(f"[DEBUG row={i}] delta match: token={tok!r} -> color_norm={matched!r} delta={delta}")
                else:
                    print(f"[DEBUG row={i}] delta NO-match token={tok!r}")

        # 输出生成逻辑：ALL -> ABS -> delta/base
        if "ALL" in color_delta_map:
            if base_price is None:
                print(f"[DEBUG row={i}] ALL present but base price missing -> skip")
                continue
            final = int(base_price + color_delta_map["ALL"])
            for col_norm, pn in color_to_pn.items():
                rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(final), "recorded_at": t})
                print(f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={final} reason=ALL")
            continue

        if color_abs_map:
            for col_norm, pn in color_to_pn.items():
                if col_norm in color_abs_map:
                    val = color_abs_map[col_norm]
                    rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(val), "recorded_at": t})
                    print(f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={val} reason=ABS")
                else:
                    if base_price is not None:
                        rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(base_price), "recorded_at": t})
                        print(f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={base_price} reason=BASE-FALLBACK")
                    else:
                        print(f"[DEBUG row={i}] -> color={col_norm} pn={pn} skipped (no abs, no base)")
            continue

        if base_price is None:
            print(f"[DEBUG row={i}] no base/abs -> skip")
            continue

        for col_norm, pn in color_to_pn.items():
            delta = color_delta_map.get(col_norm, 0)
            val = int(base_price + delta)
            rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": val, "recorded_at": t})
            print(f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={val} reason={'BASE+DELTA' if delta else 'BASE'}")

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
