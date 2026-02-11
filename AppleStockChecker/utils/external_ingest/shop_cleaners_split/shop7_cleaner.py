from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _normalize_model_generic, _load_iphone17_info_df_from_db, _build_color_map, normalize_text_basic
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

def _norm(s: str) -> str:
    return (s or "").strip()

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

def clean_shop7(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    print("shop7:買取ホムラ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

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
        shop7 的 model 字段宽松归一化：
          - 跳过纯数字行
          - 将 17pro/17promax 等短写扩展
          - 最后调用 _normalize_model_generic
        """
        if s is None:
            return ""
        txt = str(s).strip()
        if not txt:
            return ""
        if re.fullmatch(r'[\d\-\.\s]+', txt):
            return ""

        txt = re.sub(r'[\u3000\s]+', ' ', txt).strip()

        expanded = txt
        for patt, repl in _SHORT_MODEL_REPLACEMENTS:
            expanded = patt.sub(repl, expanded)

        try:
            norm = _normalize_model_generic(expanded)
        except Exception:
            norm = expanded

        if not norm or re.fullmatch(r'[\d\-\.\s]+', str(norm).strip()):
            return ""
        return norm

    info_df = _load_iphone17_info_df_from_db()  # part_number, model_name_norm, capacity_gb, color

    # 必要列检查
    need_cols = ["data", "data2", "data3", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop7 清洗器缺少必要列：{c}")

    # time-scraped 为空的行排除
    df = df.copy().reset_index(drop=True)
    mask_time_ok = df["time-scraped"].astype(str).str.strip().ne("") & df["time-scraped"].notna()
    df = df[mask_time_ok].reset_index(drop=True)
    if df.empty:
        if debug:
            print("[shop7 debug] 输入 df 为空或所有行 time-scraped 缺失，返回空 DataFrame")
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # data2 -> 机型&容量
    model_norm_series = df["data2"].map(_norm_model_for_shop7)
    cap_gb_series = df["data2"].map(_parse_capacity_gb)

    # data3 -> 价格
    price_series = df["data3"].map(_price_from_shop7)
    recorded_at = df["time-scraped"].map(parse_dt_aware)

    pn_map = _build_color_map(info_df)

    # ---- DEBUG：挑选“疑似颜色/减价说明行”，只对这些行及其上一行输出 ----
    debug_pos_set: set[int] = set()
    if debug:
        _COLOR_DISCOUNT_PAT = re.compile(
            r"(ブラック|ホワイト|ブルー|グリーン|ピンク|レッド|イエロー|パープル|オレンジ|"
            r"シルバー|ゴールド|グラファイト|ミッドナイト|スターライト|ナチュラル|"
            r"チタニウム|チタン|ディープブルー|"
            r"Black|White|Blue|Green|Pink|Red|Yellow|Purple|Orange|Silver|Gold|Titanium|"
            r"値下げ|値引|割引|円引|OFF|オフ|[+\-−－]\s*[0-9０-９])",
            re.I,
        )

        s_data2 = df["data2"].fillna("").astype(str)
        is_price_none = price_series.map(lambda x: x is None)
        mask = is_price_none & s_data2.str.contains(_COLOR_DISCOUNT_PAT, na=False)

        picked_groups = 0
        for j in range(len(df)):
            if not bool(mask.iat[j]):
                continue
            if j - 1 >= 0:
                debug_pos_set.add(j - 1)
            debug_pos_set.add(j)
            picked_groups += 1
            if picked_groups >= int(debug_limit):
                break

        if not debug_pos_set:
            picked = 0
            for i in range(len(df)):
                if price_series.iat[i] is None:
                    continue
                debug_pos_set.add(i)
                picked += 1
                if picked >= int(debug_limit):
                    break

        print(f"[shop7 debug] total_rows={len(df)}, debug_rows={len(debug_pos_set)}")

    def _dbg_on(pos: int) -> bool:
        return bool(debug) and (pos in debug_pos_set)

    def _dprint(pos: int, *args, **kwargs):
        if _dbg_on(pos):
            print(*args, **kwargs)

    # ----------------- 颜色减价解析（shop7） -----------------
    DELTA_RE = re.compile(
        r"(?P<labels>[^\d¥￥円\+\-−－]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[0-9０-９,，]+)",
        re.UNICODE,
    )

    def _to_int_amount(s: str) -> Optional[int]:
        """解析金额文本，使用通用规范化函数"""
        if s is None:
            return None
        # 使用通用规范化（全角→半角 + 去换行 + 合并空格）
        t = normalize_text_basic(str(s))
        m = re.search(r"([0-9][0-9,]*)", t)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None

    def _parse_color_deltas_shop7(text: str, ctx_pos: Optional[int] = None) -> Dict[str, int]:
        res: Dict[str, int] = {}
        if not text or not str(text).strip():
            return res
        s = str(text).strip()

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
            for tok in re.split(r"[／/、，,・\s]+", labels_part):
                tok = tok.strip()
                if tok:
                    res[_norm(tok)] = delta

        if not found:
            # 退化：如 "シルバー/ディープブルー-3000"
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

        if ctx_pos is not None and _dbg_on(ctx_pos):
            print(f"[shop7 debug] color_delta_text(pos={ctx_pos}): {text!r}")
            print(f"[shop7 debug] parsed_deltas: {res}")

        return res

    # ----------------- 主循环 -----------------
    rows: List[dict] = []
    n = len(df)

    for i in range(n):
        base_price = price_series.iat[i]
        if base_price is None:
            continue

        if _dbg_on(i):
            print("\n[shop7 debug] base_row pos=", i)
            print("  data(raw):", repr(df["data"].iat[i]))
            print("  data2(raw):", repr(df["data2"].iat[i]))
            print("  data3(raw):", repr(df["data3"].iat[i]))
            print("  base_price:", base_price)

        m = model_norm_series.iat[i]
        c = cap_gb_series.iat[i]
        t = recorded_at.iat[i]

        if _dbg_on(i):
            print("  model_norm:", repr(m))
            print("  capacity_gb:", repr(c))
            print("  recorded_at:", repr(t))

        if not m or pd.isna(c):
            _dprint(i, "  SKIP_REASON: model/cap 缺失")
            continue

        c_int = int(c)
        key = (m, c_int)
        color_to_pn = pn_map.get(key)

        if _dbg_on(i):
            print("  match_key:", repr(key))
            print("  color_to_pn keys:", list(color_to_pn.keys())[:20] if color_to_pn else None)

        if not color_to_pn:
            _dprint(i, f"  SKIP_REASON: info 表中未找到该机型容量 key={key}")
            continue

        # 下一行是否为颜色减价行：data2 非空且 data3 不可解析为价格
        deltas: Dict[str, int] = {}
        j = i + 1
        if j < n:
            nxt_data2 = df["data2"].iat[j] if df["data2"].iat[j] is not None else ""
            nxt_price_cell = df["data3"].iat[j] if df["data3"].iat[j] is not None else ""
            nxt_price_val = _price_from_shop7(nxt_price_cell) if str(nxt_price_cell).strip() else None
            is_color_line = bool(str(nxt_data2).strip()) and (nxt_price_val is None)

            if _dbg_on(j) or _dbg_on(i):
                print("[shop7 debug] next_row pos=", j)
                print("  data2(raw):", repr(nxt_data2))
                print("  data3(raw):", repr(nxt_price_cell))
                print("  parsed_next_price:", nxt_price_val)
                print("  is_color_line:", is_color_line)

            if is_color_line:
                deltas = _parse_color_deltas_shop7(nxt_data2, ctx_pos=j)
            else:
                _dprint(i, "  note: 下一行不是颜色减价行，按 base_price 输出")

        # 输出：对每个颜色做 base + delta
        for col_norm, (pn, _) in color_to_pn.items():
            delta = 0
            used_lbl = None

            if deltas:
                if col_norm in deltas:
                    delta = deltas[col_norm]
                    used_lbl = col_norm
                    _dprint(i, f"  delta_direct_match: col_norm={col_norm!r} delta={delta} pn={pn}")
                else:
                    matches = info2[
                        (info2["model_name_norm"] == m)
                        & (info2["capacity_gb"].astype("Int64") == c_int)
                        & (info2["part_number"].astype(str) == str(pn))
                    ]
                    raw_color = matches["color"].iat[0] if not matches.empty else ""
                    raw_color_norm = _norm(raw_color)

                    for lbl_norm, dval in deltas.items():
                        if not lbl_norm:
                            continue
                        if lbl_norm in raw_color_norm:
                            delta = dval
                            used_lbl = lbl_norm
                            _dprint(i, f"  delta_norm_substring_match: raw_color={raw_color!r} lbl={lbl_norm!r} delta={delta} pn={pn}")
                            break
                        if lbl_norm in str(raw_color):
                            delta = dval
                            used_lbl = lbl_norm
                            _dprint(i, f"  delta_raw_substring_match: raw_color={raw_color!r} lbl={lbl_norm!r} delta={delta} pn={pn}")
                            break

                    if _dbg_on(i) and deltas and used_lbl is None:
                        print(f"  delta_NO_MATCH: pn={pn} raw_color={raw_color!r} deltas_keys={list(deltas.keys())}")

            price_final = int(base_price + delta)

            if _dbg_on(i):
                print("  -> OUT:", {
                    "part_number": str(pn),
                    "color_norm": col_norm,
                    "base": int(base_price),
                    "delta": int(delta),
                    "used_lbl": used_lbl,
                    "final": int(price_final),
                })

            rows.append({
                "part_number": str(pn),
                "shop_name": "買取ホムラ",
                "price_new": price_final,
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    if debug:
        print(f"\n[shop7 debug] out_rows={len(out)} head=\n{out.head(10).to_string(index=False)}")

    return out
