# -*- coding: utf-8 -*-
"""
Django 版二手店 CSV 清洗工具（文件流友好）
来源：会话中用户提供的清洗脚本（四家店：モバイルミックス / モバイル一番 / 森森買取 / 買取ルデヤ）
目标结构：
  店铺名 | iphone型号 | Part_number | JAN | 颜色 | 状态 | 价格 | 价格2 | 价格3 | 其他没有记录的备用信息
"""
from __future__ import annotations
import io, os, re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ========== 基础工具 ==========
def _read_csv_smart_from_bytes(data: bytes, filename: str = "") -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp932", "shift_jis", "utf-8"]
    seps = [",", "\t", ";", "|"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, engine="c", on_bad_lines="skip")
                if df.shape[1] == 1 and any(x in str(df.columns[0]) for x in [",", "\t", ";", "|"]):
                    raise ValueError("likely wrong separator")
                return df
            except Exception:
                pass
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                return df
            except Exception:
                pass
    # 尝试 Excel（极少数场景）
    try:
        return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(f"无法读取文件（{filename}）：{e}")

def _clean_ws(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def _to_int_price(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val)
    if re.search(r"万", s):
        m = re.search(r'([\d\.]+)\s*万', s)
        base = float(m.group(1)) if m else 0.0
        tail = 0
        m2 = re.search(r'万\s*([0-9,]+)', s)
        if m2:
            tail = int(re.sub(r"[^\d]", "", m2.group(1)))
        return int(base * 10000 + tail)
    if not re.search(r"\d", s):
        return None
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

def _normalize_jan(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"[^\d]", "", str(x))

def _extract_part_number(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"\b[A-Z0-9]{4,}J/A\b", text)
    return m.group(0) if m else ""

def _extract_color(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"\b\d{2,4}GB\s+(.+?)\s+SIMフリー", text)
    if m:
        return _clean_ws(m.group(1))
    m2 = re.search(
        r"(desert|black|white|natural|blue|green|pink|yellow|red|gold|silver|gray|grey|"
        r"グリーン|ピンク|ブルー|ブラック|ホワイト|ゴールド|シルバー|ナチュラル|ネイビー|ミッドナイト|スターライト|"
        r"デザートチタニウム)",
        text, re.IGNORECASE,
    )
    return m2.group(1) if m2 else ""

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["店铺名","iphone型号","Part_number","JAN","颜色","状态","价格","价格2","价格3","其他没有记录的备用信息"]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c not in ["价格","价格2","价格3"] else None
    return df[cols]

# ========== 各店解析 ==========
def _parse_rudeya_df(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        title = _clean_ws(r.get("标题",""))
        if not title:
            continue
        m = re.search(r"(iPhone\s*\d+\s*Pro(?:\s*Max)?\s*\d+GB)", title, re.IGNORECASE)
        model = _clean_ws(m.group(1)) if m else _clean_ws(" ".join(title.split()[:5]))
        out.append({
            "店铺名": "買取ルデヤ",
            "iphone型号": model,
            "Part_number": _extract_part_number(title),
            "JAN": _normalize_jan(r.get("编号","")),
            "颜色": _extract_color(title),
            "状态": "未開封" if "未開封" in title else _clean_ws(str(r.get("noru",""))),
            "价格": _to_int_price(r.get("价格")),
            "价格2": None, "价格3": None,
            "其他没有记录的备用信息": title,
        })
    return _ensure_schema(pd.DataFrame(out))

def _parse_mobilemix_df(df: pd.DataFrame, shop_name="モバイルミックス") -> pd.DataFrame:
    rows = df.to_dict("records")
    out = []
    for i in range(0, len(rows), 2):
        r1 = rows[i]
        r2 = rows[i+1] if i+1 < len(rows) else {}
        model = _clean_ws(r1.get("字段",""))
        price = _to_int_price(r1.get("价格"))
        status = ""
        for k in ("字段","字段1"):
            v = r2.get(k,"")
            if isinstance(v,str) and v.strip():
                status = _clean_ws(v); break
        color = _clean_ws(r2.get("价格",""))
        if model and price is not None:
            out.append({
                "店铺名": shop_name, "iphone型号": model, "Part_number": "", "JAN": "",
                "颜色": color, "状态": status,
                "价格": price, "价格2": None, "价格3": None, "其他没有记录的备用信息": "",
            })
    return _ensure_schema(pd.DataFrame(out))

def _parse_morimori_df(df: pd.DataFrame) -> pd.DataFrame:
    def find_prices(row: Dict[str,Any]) -> Tuple[Optional[int],Optional[int],Optional[int]]:
        vals = []
        for c in df.columns:
            v = row.get(c,"")
            if v is None or (isinstance(v,float) and pd.isna(v)): continue
            s = str(v)
            if "円" in s and re.search(r"\d", s):
                p = _to_int_price(s)
                if p is not None: vals.append(p)
        uniq, seen = [], set()
        for p in vals:
            if p not in seen: uniq.append(p); seen.add(p)
        while len(uniq) < 3: uniq.append(None)
        return uniq[0], uniq[1], uniq[2]

    out = []
    for _, r in df.iterrows():
        title0 = _clean_ws(r.get("文本1",""))
        status = _clean_ws(r.get("文本2",""))
        detail = _clean_ws(r.get("文本3",""))
        jan = _normalize_jan(r.get("文本4",""))
        cap_m = re.search(r"(\d{2,4}GB)", detail)
        if cap_m:
            capacity = cap_m.group(1)
            t0 = re.sub(r"iPhone\s*(\d)", r"iPhone \1", title0)
            model = f"{t0} {capacity}".strip()
        else:
            model = re.sub(r"iPhone\s*(\d)", r"iPhone \1", title0).strip()
        color = _extract_color(detail)
        p1,p2,p3 = find_prices(r.to_dict())
        if model:
            out.append({
                "店铺名":"森森買取","iphone型号":model,"Part_number":"",
                "JAN":jan,"颜色":color,"状态":status,
                "价格":p1,"价格2":p2,"价格3":p3,"其他没有记录的备用信息":detail
            })
    return _ensure_schema(pd.DataFrame(out))

def _parse_ichiban_df(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        model = _clean_ws(r.get("cardbody",""))
        extra = _clean_ws(r.get("cardbody1",""))
        status_tag = _clean_ws(r.get("标签",""))
        status = "未開封" if "未開封" in extra else status_tag
        price = _to_int_price(r.get("价格"))
        if model and price is not None:
            out.append({
                "店铺名":"モバイル一番","iphone型号":model,"Part_number":"",
                "JAN":"", "颜色":"", "状态":status,
                "价格":price,"价格2":None,"价格3":None,
                "其他没有记录的备用信息":extra
            })
    return _ensure_schema(pd.DataFrame(out))

# ========== 识别 & 入口 ==========
def parse_tradein_uploaded(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """根据文件名/列名自动识别来源并解析成统一列。"""
    df0 = _read_csv_smart_from_bytes(file_bytes, filename)
    base = os.path.basename(filename)
    # 文件名优先
    if "ルデヤ" in base or "買取ルデヤ" in base:
        return _parse_rudeya_df(df0)
    if "森森" in base or "森森買取" in base:
        return _parse_morimori_df(df0)
    if "モバイル一番" in base:
        return _parse_ichiban_df(df0)
    if "モバイルミックス" in base:
        return _parse_mobilemix_df(df0, shop_name="モバイルミックス")
    # 列头次选
    cols = set(map(str, df0.columns))
    if {"标题","编号","价格"}.issubset(cols):
        return _parse_rudeya_df(df0)
    if {"cardbody","标签","价格"}.issubset(cols):
        return _parse_ichiban_df(df0)
    if {"文本1","文本2","文本3"}.issubset(cols):
        return _parse_morimori_df(df0)
    if {"字段","字段1","价格"}.issubset(cols):
        return _parse_mobilemix_df(df0, shop_name="モバイルミックス")
    raise RuntimeError(f"无法识别文件来源，请检查文件名或列头：{filename}")
