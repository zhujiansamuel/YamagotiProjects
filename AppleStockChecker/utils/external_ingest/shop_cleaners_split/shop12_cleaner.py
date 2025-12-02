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

FAMILY_SYNONYMS = {
    "blue": ["ブルー"],
    "black": ["ブラック", "黒"],
    "white": ["ホワイト", "白"],
    "green": ["グリーン", "緑"],
    "red": ["レッド", "赤"],
    "pink": ["ピンク"],
    "purple": ["パープル", "紫"],
    "yellow": ["イエロー", "黄"],
    "gold": ["ゴールド"],
    "silver": ["シルバー"],
    "gray": ["グレー", "グレイ", "灰"],
    "natural": ["ナチュラル"],
}

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

def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    宽松匹配一个 'label_raw' 是否命中某个颜色（color_raw / color_norm）。
    优先：
      - 归一化后完全相等
      - label_raw 子串包含于 color_raw
      - 英文族名（如 Blue）映射到日文家族词，并判断是否是 color_raw 的子串
    """
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    key = label_raw.strip().lower()
    if key in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[key]:
            if jp in str(color_raw):
                return True
    # 也尝试 label_norm 的英文键
    if label_norm in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[label_norm]:
            if jp in str(color_raw):
                return True
    return False

def _build_color_map(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
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

def clean_shop12(df: pd.DataFrame) -> pd.DataFrame:
    import time
    print("shop12:トゥインクル---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    """
    解析规则改进：
      - 若備考1 含多行（有 \n），仅跳过含“開封/開封品/※開封”那一行，不影响其它行的匹配；
      - 支持绝对价 (Silver ¥230,500)、支持差额 (Blue-2000円)、支持 '全色'；
      - 使用宽松的颜色同义表匹配 info 表颜色；
      - debug 输出详细显示每行解析过程和最终结果。
    """
    # 必要列检查
    for c in ["モデルナンバー", "備考1", "買取価格", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop12 清洗器缺少必要列：{c}")

    # 载入 info 表并构建 color map
    info_df = _load_iphone17_info_df_for_shop2()
    def _build_color_map(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
        df2 = info_df.copy()
        df2["model_name_norm"] = df2["model_name"].map(_normalize_model_generic)
        df2["capacity_gb"] = pd.to_numeric(df2["capacity_gb"], errors="coerce").astype("Int64")
        df2["color_norm"] = df2["color"].map(lambda x: _norm(str(x)))
        cmap: Dict[Tuple[str, int], Dict[str, Tuple[str, str]]] = {}
        for _, r in df2.iterrows():
            m = r["model_name_norm"]
            cap = r["capacity_gb"]
            if not m or pd.isna(cap):
                continue
            key = (m, int(cap))
            cmap.setdefault(key, {})
            cmap[key][_norm(str(r["color"]))] = (str(r["part_number"]), str(r["color"]))
        return cmap

    cmap_all = _build_color_map(info_df)

    # 同义表（可按需扩充）
    FAMILY_SYNONYMS = {
        "blue": ["ブルー", "青", "ディープブルー"],
        "ブルー": ["ブルー", "青", "ディープブルー"],
        "青": ["ブルー", "青", "ディープブルー"],
        "ディープブルー": ["ディープブルー", "ブルー", "青"],
        "silver": ["シルバー", "銀", "Silver"],
        "シルバー": ["シルバー", "銀", "Silver"],
        "銀": ["シルバー", "銀"],
        "black": ["ブラック", "黒", "Black"],
        "ブラック": ["ブラック", "黒"],
        "黒": ["ブラック", "黒"],
        "white": ["ホワイト", "白", "White"],
        "ホワイト": ["ホワイト", "白"],
        "orange": ["オレンジ", "橙"],
        "オレンジ": ["オレンジ", "橙"],
    }
    EN_TO_JP = {
    "silver": ["シルバー", "銀"],
    "blue":   ["ブルー", "青", "ディープブルー"],
    "black":  ["ブラック", "黒"],
    "white":  ["ホワイト", "白"],
    "gold":   ["ゴールド", "金"],
    "green":  ["グリーン", "緑"],
    "red":    ["レッド", "赤"],
    "pink":   ["ピンク"],
    "purple": ["パープル", "紫"],
    "yellow": ["イエロー", "黄"],
    "orange": ["オレンジ", "橙"],
    "gray":   ["グレー", "グレイ", "灰"],
    "natural":["ナチュラル"],
}
    # 便于查找的反查表（归一化键 -> 家族关键词列表）
    SYN_LOOKUP: Dict[str, List[str]] = {}
    for k, toks in FAMILY_SYNONYMS.items():
        SYN_LOOKUP[_norm(k)] = [_norm(t) for t in toks]

    # 正则与工具
    SPLIT_SEPS = r"[／/、，,・\s]+"  # 用于拆分多个颜色标签
    ABS_RE = re.compile(
        r"""(?P<labels>[^\d¥￥円:：/、，,;；※]+?)\s*(?:[:：]?\s*)?(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
        re.UNICODE | re.VERBOSE,
    )
    DELTA_RE = re.compile(
        r"""(?P<labels>[^+\-−－\d¥￥円/、，,;；※]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
        re.UNICODE | re.VERBOSE,
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

    def _line_is_opened(ln: str) -> bool:
        """判断该子行是否属于开封价说明（应跳过）"""
        if not ln:
            return False
        s = str(ln)
        return ("開封" in s) or ("※開封" in s) or ("開封品" in s) or ("開封済" in s)

    def _extract_abs_prices(text: str) -> List[Tuple[str, int]]:
        """
        从文本（可能含换行）提取 [(label_raw, absolute_price), ...]。
        仅处理不含“開封”字样的子行。
        """
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        lines = [ln for ln in re.split(r"[\r\n]+", s) if ln is not None]
        for ln in lines:
            if not ln or _line_is_opened(ln):
                # 跳过开封行
                if _line_is_opened(ln):
                    print(f"[DEBUG] skip opened-line for abs: {ln!r}")
                continue
            # 在该行寻找 label+amount 匹配（可能多个）
            for m in ABS_RE.finditer(ln):
                labels_part = m.group("labels") or ""
                amt_txt = m.group("amount")
                amt = _norm_amount_to_int(amt_txt)
                if amt is None:
                    continue
                toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
                for tok in toks:
                    # 忽略纯数字 token
                    if re.fullmatch(r"[0-9,，]+", tok.replace(" ", "")):
                        continue
                    out.append((tok, int(amt)))
        return out

    def _extract_deltas(text: str) -> List[Tuple[str, int]]:
        """
        抽取差额：label +/- amount。跳过含“開封”的子行/片段。
        """
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        lines = [ln for ln in re.split(r"[\r\n]+", s) if ln is not None]
        for ln in lines:
            if not ln:
                continue
            if _line_is_opened(ln):
                print(f"[DEBUG] skip opened-line for delta: {ln!r}")
                continue
            for m in DELTA_RE.finditer(ln):
                labels_part = m.group("labels") or ""
                sign = m.group("sign") or "+"
                amt_txt = m.group("amount")
                amt = _norm_amount_to_int(amt_txt)
                if amt is None:
                    continue
                delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
                toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
                for tok in toks:
                    if re.fullmatch(r"[0-9,，]+", tok.replace(" ", "")):
                        continue
                    out.append((tok, delta))
        # 全色 fallback（如果没有找到差额，但某些非开封行包含 全色）
        if not out:
            # 搜索所有非开封子行是否包含全色
            s_all = str(text)
            for ln in re.split(r"[\r\n]+", s_all):
                if not ln or _line_is_opened(ln):
                    continue
                if "全色" in ln:
                    m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)?", ln)
                    if m:
                        sign = m.group(1) or "+"
                        amt = m.group(2)
                        amt_v = _norm_amount_to_int(amt) if amt else 0
                        out.append(("全色", -amt_v if sign in ("-", "−", "－") else amt_v))
                    else:
                        out.append(("全色", 0))
                    break
        return out

    def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
        """
        更稳健的颜色匹配：
          - 英文标签走纯小写通道（不经 _norm），用 EN_TO_JP 直接映射再比对；
          - 日文/中文标签走 _norm + 同义族（SYN_LOOKUP/FAMILY_SYNONYMS）；
          - 然后做双向原文/归一子串兜底；
          - 带调试打印（去掉/注释即可）。
        """
        if not label_raw or not color_raw:
            return False

        lbl_raw = str(label_raw).strip()
        cr_raw  = str(color_raw).strip()

        # ===== 1) 英文直译通道（不经 _norm，避免 ASCII 被吞/变形）=====
        label_lower = lbl_raw.lower()
        if label_lower in EN_TO_JP:
            for jp in EN_TO_JP[label_lower]:
                if jp in cr_raw:
                    # print(f"[match EN] '{lbl_raw}' -> '{jp}' in color_raw='{cr_raw}'")
                    return True
                if _norm(jp) == color_norm:
                    # print(f"[match EN] '{lbl_raw}' -> norm('{jp}') == color_norm='{color_norm}'")
                    return True
            # 英文直译没中，继续日文流程

        # ===== 2) 日文/中文通道：_norm + 同义族 =====
        ln = _norm(lbl_raw)        # 归一化标签
        cn = color_norm            # 归一化 info 颜色键

        # 2-1 精确归一化等值
        if ln == cn:
            # print(f"[match JP exact] '{lbl_raw}'(norm={ln}) == color_norm={cn}")
            return True

        # 2-2 原文子串/归一子串（双向）
        if lbl_raw in cr_raw or ln in cn or cn in ln:
            # print(f"[match substr] '{lbl_raw}' in '{cr_raw}' or norm-substr")
            return True

        # 2-3 家族同义（先走预构建的 SYN_LOOKUP；没有则走 FAMILY_SYNONYMS）
        if 'SYN_LOOKUP' in globals() and ln in SYN_LOOKUP:
            for cand in SYN_LOOKUP[ln]:
                if cand == cn or cand in _norm(cr_raw) or cand in cr_raw:
                    # print(f"[match SYN] '{lbl_raw}' -> cand='{cand}' matches color_raw/norm")
                    return True
        if ln in FAMILY_SYNONYMS:
            for tok in FAMILY_SYNONYMS[ln]:
                if tok in cr_raw or _norm(tok) == cn:
                    # print(f"[match FAMILY] '{lbl_raw}' -> tok='{tok}' matches color")
                    return True

        # ===== 3) 最后兜底：把空白去掉后做子串（处理 'ディープ ブルー' 之类）=====
        ln_short = re.sub(r"[\s\u3000]+", "", ln)
        cn_short = re.sub(r"[\s\u3000]+", "", cn)
        if ln_short and (ln_short in cn_short or cn_short in ln_short):
            # print(f"[match short-substr] '{ln_short}' ~ '{cn_short}'")
            return True

        print(f"[no match] label_raw='{label_raw}' color_raw='{color_raw}' color_norm='{color_norm}'")
        return False
    rows: List[dict] = []

    for idx, row in df.iterrows():
        price_base = to_int_yen(row.get("買取価格"))
        if price_base is None:
            # 跳过无价行（标题/分隔）
            continue

        model_text = str(row.get("モデルナンバー") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            print(f"[DEBUG row={idx}] 跳过（model/cap 解析失败） model={model_text!r}")
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            print(f"[DEBUG row={idx}] 跳过（info 表无该型号/容量） key={key}")
            continue

        remark_raw = row.get("備考1") or ""
        abs_list = _extract_abs_prices(remark_raw)
        delta_list = _extract_deltas(remark_raw)

        print(f"[DEBUG row={idx}] model={model_text!r} price_base={price_base} remark={remark_raw!r}")
        print(f"[DEBUG row={idx}] parsed abs_list={abs_list}, delta_list={delta_list}")

        # 映射 label -> color_norm
        color_abs_map: Dict[str, int] = {}
        color_delta_map: Dict[str, int] = {}

        for label_raw, amt in abs_list:
            matched = None
            for col_norm, (pn, col_raw) in color_map.items():
                if _label_matches_color(label_raw, col_raw, col_norm):
                    matched = col_norm
                    break
            if matched:
                color_abs_map[matched] = int(amt)
                print(f"[DEBUG row={idx}] abs match: {label_raw!r} -> color_norm={matched}, price={amt}")
            else:
                print(f"[DEBUG row={idx}] abs NO-match: {label_raw!r}")

        for label_raw, delta in delta_list:
            if label_raw == "全色":
                color_delta_map["ALL"] = int(delta)
                print(f"[DEBUG row={idx}] delta ALL = {delta}")
                continue
            matched = None
            for col_norm, (pn, col_raw) in color_map.items():
                if _label_matches_color(label_raw, col_raw, col_norm):
                    matched = col_norm
                    break
            if matched:
                color_delta_map[matched] = int(delta)
                print(f"[DEBUG row={idx}] delta match: {label_raw!r} -> color_norm={matched}, delta={delta}")
            else:
                print(f"[DEBUG row={idx}] delta NO-match: {label_raw!r}")

        # 生成输出：优先绝对价 -> 再考虑 全色差额 -> 否则按基价+差额
        if "ALL" in color_delta_map:
            final_price = int(price_base + color_delta_map["ALL"])
            for col_norm, (pn, col_raw) in color_map.items():
                rows.append({"part_number": pn, "shop_name": "トゥインクル", "price_new": final_price, "recorded_at": row.get("time-scraped")})
                print(f"[DEBUG row={idx}] OUT: color={col_norm} pn={pn} price={final_price} reason=ALL")
            continue

        if color_abs_map:
            # 绝对价覆盖部分颜色，未列出的颜色用 base price
            for col_norm, (pn, col_raw) in color_map.items():
                if col_norm in color_abs_map:
                    val = color_abs_map[col_norm]
                    rows.append({"part_number": pn, "shop_name": "トゥインクル", "price_new": int(val), "recorded_at": row.get("time-scraped")})
                    print(f"[DEBUG row={idx}] OUT: color={col_norm} pn={pn} price={val} reason=ABS")
                else:
                    rows.append({"part_number": pn, "shop_name": "トゥインクル", "price_new": int(price_base), "recorded_at": row.get("time-scraped")})
                    print(f"[DEBUG row={idx}] OUT: color={col_norm} pn={pn} price={price_base} reason=BASE-FALLBACK")
            continue

        # 否则按差额映射（可能部分颜色有 delta；其它用 base）
        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_delta_map.get(col_norm, 0)
            val = int(price_base + delta)
            # rows.append({"part_number": pn, "shop_name": "トゥインクル", "price_new": val, "recorded_at": row.get("time-scraped")})
            rows.append({"part_number": pn, "shop_name": "トゥインクル", "price_new": val, "recorded_at": parse_dt_aware(row.get("time-scraped"))})
            print(f"[DEBUG row={idx}] OUT: color={col_norm} pn={pn} price={val} reason={'BASE+DELTA' if delta else 'BASE'}")

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
