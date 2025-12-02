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

def clean_shop7(df: pd.DataFrame) -> pd.DataFrame:
    print("DEBUG: shop7:買取ホムラ ----------> 进入清洗器 时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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
        针对 shop7 的 model 字段做宽松归一化：
          - 跳过纯数字的行（返回空字符串）
          - 将 '17pro', '17promax', '17 pro max' 等短写扩展为 'iPhone 17 Pro Max'
          - 最后调用 _normalize_model_generic 生成最终的归一字符串（与 info 表匹配）
        返回 '' 表示无法识别（将被跳过）
        """
        if s is None:
            return ""
        txt = str(s).strip()
        if not txt:
            return ""

        # 跳过行号/序号（仅数字或少量标点）
        if re.fullmatch(r'[\d\-\.\s]+', txt):
            # 代表这是序号/编号行，例如 "1", "2" 等
            return ""

        # 统一全角空格 & 多余空白
        txt = re.sub(r'[\u3000\s]+', ' ', txt).strip()

        # 先把常用短写替换成标准形式
        expanded = txt
        for patt, repl in _SHORT_MODEL_REPLACEMENTS:
            expanded = patt.sub(repl, expanded)

        # 可能存在像 "17pro 256GB" 或 "17promax 256GB" 这样的，
        # 上面的替换会把它变成 "iPhone 17 Pro 256GB" 之类，交给 normalize 处理。
        try:
            norm = _normalize_model_generic(expanded)
        except Exception:
            # 若 _normalize_model_generic 出错，退化为简单返回 expanded（再由上级判定是否匹配）
            norm = expanded

        # 如果 normalize 后为空或仅数字，认为无法识别
        if not norm or re.fullmatch(r'[\d\-\.\s]+', str(norm).strip()):
            return ""
        return norm

    info_df = _load_iphone17_info_df_for_shop2()  # part_number, model_name_norm, capacity_gb, color

    # 必要列检查
    need_cols = ["data", "data2", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop7 清洗器缺少必要列：{c}")

    # 先把 time-scraped 为空的行排除，避免时间解析报错
    df = df.copy().reset_index(drop=True)
    mask_time_ok = df["time-scraped"].astype(str).str.strip().ne("") & df["time-scraped"].notna()
    df = df[mask_time_ok].reset_index(drop=True)
    if df.empty:
        print("DEBUG: 输入 df 为空或所有行 time-scraped 缺失，返回空 DataFrame")
        return pd.DataFrame(columns=["part_number","shop_name","price_new","recorded_at"])

    # data -> 机型&容量
    model_norm_series = df["data2"].map(_norm_model_for_shop7)
    cap_gb_series     = df["data2"].map(_parse_capacity_gb)

    # 价格/时间（注意 data2 里是价格，我们仅在行有 price 时才处理该行）
    price_series  = df["data3"].map(_price_from_shop7)
    recorded_at   = df["time-scraped"].map(parse_dt_aware)


    # ------------- 构建 (model_norm, cap) -> { color_norm: part_number } -------------
    info2 = info_df.copy()
    if "color" not in info2.columns:
        raise ValueError("info_df 缺少 'color' 列，无法进行颜色映射")
    info2["model_name_norm"] = info2["model_name"].map(_normalize_model_generic)
    info2["capacity_gb"] = pd.to_numeric(info2["capacity_gb"], errors="coerce").astype("Int64")
    info2["color_norm"] = info2["color"].map(lambda x: _norm(str(x)))

    pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    for _, r in info2.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col = r["color_norm"]
        pn = str(r["part_number"])
        if pd.isna(cap) or not m or not col:
            continue
        key = (m, int(cap))
        pn_map.setdefault(key, {})
        pn_map[key][col] = pn

    print(f"DEBUG: 建立了 pn_map，包含 {len(pn_map)} 个 (model,cap) 条目")

    # ----------------- 颜色减价解析函数（shop7 专用） -----------------
    DELTA_RE = re.compile(
        r"(?P<labels>[^\d¥￥円\+\-−－]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[0-9０-９,，]+)",
        re.UNICODE
    )

    FW_TO_ASC = str.maketrans({
        "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
        "，":",","．":".","－":"-","＋":"+","　":" "
    })

    def _to_int_amount(s: str) -> Optional[int]:
        if s is None:
            return None
        t = str(s).translate(FW_TO_ASC)
        m = re.search(r"([0-9][0-9,]*)", t)
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None

    def _parse_color_deltas_shop7(text: str) -> Dict[str, int]:
        res: Dict[str, int] = {}
        if not text or not str(text).strip():
            return res
        s = str(text).strip()
        parts = []
        # 尝试先用 DELTA_RE 匹配整段里的所有有金额的片段；若没则按分隔符拆
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
            # labels_part 可能包含多个颜色，用常见分隔符拆
            for tok in re.split(r"[／/、，,・\s]+", labels_part):
                tok = tok.strip()
                if not tok:
                    continue
                key = _norm(tok)
                res[key] = delta
        if not found:
            # 退化处理：没有显式金额匹配的情况下，尝试找类似 "シルバー/ディープブルー-3000" 形式
            # 以最后出现的 +/- 数字为金额，前面子串作为标签
            # 例如 "シルバー/ディープブルー-3000"
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
        # Debug print for parsed deltas
        if res:
            print(f"DEBUG: 解析到 color deltas from '{text}': {res}")
        else:
            print(f"DEBUG: 未解析到 color deltas from '{text}'")
        return res

    # ----------------- 主循环：遍历含价格的行 -----------------
    rows: List[dict] = []
    n = len(df)
    for i in range(n):
        base_price = price_series.iat[i]
        if base_price is None:
            # 不是机种行（或者是下行的颜色行），跳过
            continue

        model_text_raw = df["data"].iat[i] if df["data"].iat[i] is not None else ""
        m = model_norm_series.iat[i]
        c = cap_gb_series.iat[i]
        t = recorded_at.iat[i]
        print(f"DEBUG: 处理行 i={i}, model_raw='{model_text_raw}', model_norm='{m}', cap='{c}', base_price={base_price}")

        if not m or pd.isna(c):
            print(f"DEBUG: 跳过 i={i} 因为 model/cap 缺失")
            continue
        c = int(c)
        key = (m, c)
        color_to_pn = pn_map.get(key)
        print(f"DEBUG: 对应的 color->pn 映射: {color_to_pn}")
        if not color_to_pn:
            print(f"DEBUG: info 表中未找到该机型容量 key={key}，跳过 i={i}")
            continue

        # 检查下一行是否为颜色减价行：下一行 data 非空 且 data2 为空
        deltas: Dict[str, int] = {}
        j = i + 1
        if j < n:
            nxt_data = df["data2"].iat[j] if df["data2"].iat[j] is not None else ""
            nxt_data2 = df["data3"].iat[j] if "data3" in df.columns and df["data2"].iat[j] is not None else ""
            if str(nxt_data).strip() and (str(nxt_data2).strip() == "" or pd.isna(nxt_data2)):
                print(f"DEBUG: 发现潜在颜色行 at i+1={j}: '{nxt_data}'")
                deltas = _parse_color_deltas_shop7(nxt_data)
            else:
                print(f"DEBUG: 下一行 i+1={j} 不是颜色行 (data='{nxt_data}' data2='{nxt_data2}')")

        # 生成每个颜色的 price_new
        for col_norm, pn in color_to_pn.items():
            delta = 0
            if deltas:
                if col_norm in deltas:
                    delta = deltas[col_norm]
                    print(f"DEBUG: 直接匹配 col_norm='{col_norm}' 得到 delta={delta} for pn={pn}")
                else:
                    # 进一步尝试在 info2 中查原始 color 文本匹配 deltas 的 key
                    matches = info2[
                        (info2["model_name_norm"] == m) &
                        (info2["capacity_gb"].astype("Int64") == c) &
                        (info2["part_number"].astype(str) == str(pn))
                    ]
                    raw_color = matches["color"].iat[0] if not matches.empty else ""
                    matched = False
                    for lbl_norm, dval in deltas.items():
                        # 尝试在 raw_color 的归一化字符串中查找 lbl_norm
                        if lbl_norm and lbl_norm in _norm(raw_color):
                            delta = dval
                            matched = True
                            print(f"DEBUG: 通过 raw_color='{raw_color}' 的归一化匹配 lbl='{lbl_norm}' -> delta={delta} for pn={pn}")
                            break
                        # 也尝试原文子串匹配
                        if lbl_norm and lbl_norm in raw_color:
                            delta = dval
                            matched = True
                            print(f"DEBUG: 通过 raw_color 原文匹配 lbl='{lbl_norm}' -> delta={delta} for pn={pn}")
                            break
                    if not matched and deltas:
                        print(f"DEBUG: 未匹配到颜色调整 for pn={pn} (raw_color='{raw_color}'), 使用 delta=0")

            price_final = int(base_price + delta)
            print(f"DEBUG: 输出 -> pn={pn}, base={base_price}, delta={delta}, final={price_final}")
            rows.append({
                "part_number": str(pn),
                "shop_name": "買取ホムラ",
                "price_new": price_final,
                "recorded_at": t,
            })

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    print("DEBUG: 完成 clean_shop7, 产出行数:", len(out))
    return out
