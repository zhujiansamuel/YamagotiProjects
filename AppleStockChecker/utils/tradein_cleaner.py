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
import io, os, re
import pandas as pd
import pandas as pd

from .color_norm import normalize_color              # -> (canon, is_all)
# 若 _normalize_simfree 与本文件同处，直接引用；否则 from .xxx import _normalize_simfree


_SIMFREE_POSITIVE = [
    r"\bSIM\s*フリ[ーｰ–\-]?\b",      # SIMフリー / SIMフリ- / SIMフリ– 等
    r"\bシム\s*フリ[ーｰ–\-]?\b",     # シムフリー
    r"\bＳＩＭ\s*フリ[ーｰ–\-]?\b",   # 全角
    r"\bsim\s*free\b",               # sim free
    r"\bSIM\s*FREE\b",
    r"SIM\s*ロック\s*解除\s*済(?:み)?",  # SIMロック解除済 / 済み
    r"SIMロック\s*解除\s*済(?:み)?",
    r"ロック\s*解除\s*済(?:み)?",       # ロック解除済 / 済み
    r"SIM\s*ロック\s*な[し無]",         # SIMロックなし/無し
    r"SIMロック無し",
    r"SIMロックなし",
    r"ノー\s*ロック",                   # ノーロック / no-lock
]
# 明确排除一些常见运营商词（不写库）
_CARRIER_TOKENS = [
    "au", "ＡＵ", "ソフトバンク", "softbank", "SoftBank",
    "docomo", "ドコモ", "楽天", "rakuten"
]
_SIMFREE_NEGATIVE_HINTS = [
    r"未解除", r"解除\s*必要", r"要解除", r"解除不可",
    r"ロック\s*(有|あり|有り)", r"\bsim\s*lock(ed)?\b", r"\blocked\b",
]

TITLE_CANDIDATES  = ["文本1", "タイトル", "商品名", "品名", "名称", "機種", "型番", "テキスト1"]
DETAIL_CANDIDATES = ["文本3", "詳細", "備考", "説明", "スペック", "コメント", "テキスト3", "テキスト"]
STATUS_CANDIDATES = ["文本2", "状態", "コンディション", "ランク", "タグ"]
JAN_CANDIDATES    = ["文本4", "JAN", "JANコード", "JANｺｰﾄﾞ", "JANコード(13桁)"]



# ========== 基础工具 ==========
def _first_nonempty(row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def _join_texts(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    return " ".join(parts)

def _normalize_simfree(text: str) -> tuple[bool, str]:
    """
    归一化“是否为 SIMフリー”：
      - 命中任一正向模式（含 'SIMロック解除済み' / 'ロック解除済' / 'SIMロックなし' / 'SIMフリー' / 'sim free' 等）
        => (True, 'SIMフリー')
      - 否则返回 (False, '')
    备注：
      - 若同时出现否定提示词（未解除/要解除/ロックあり…），且没有“済/なし/フリー”类正命中，可判为 False。
      - 可按需要扩展上面两个列表。
    """
    if not text:
        return (False, "")
    t = str(text)

    # 先检测“强阳性”：済/なし/フリー类（任何一个命中即为 True）
    for pat in _SIMFREE_POSITIVE:
        if re.search(pat, t, flags=re.IGNORECASE):
            return (True, "SIMフリー")

    # 未命中正向，再看是否出现明显否定上下文（可选）
    for neg in _SIMFREE_NEGATIVE_HINTS:
        if re.search(neg, t, flags=re.IGNORECASE):
            return (False, "")

    return (False, "")

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
    if val is None: return None
    s = str(val)
    # 跳过 12–14 位连续数字（很可能是 JAN/电话）
    if re.fullmatch(r"\d{12,14}", s.strip()):
        return None
    # 处理“万”记法
    if "万" in s:
        m = re.search(r"([\d\.]+)\s*万", s)
        base = float(m.group(1)) if m else 0.0
        tail = 0
        m2 = re.search(r"万\s*([0-9,]+)", s)
        if m2: tail = int(re.sub(r"[^\d]", "", m2.group(1)))
        price = int(base * 10000 + tail)
    else:
        if not re.search(r"\d", s): return None
        price = int(re.sub(r"[^\d]", "", s))

    # 合理区间过滤（避免把库存号/邮编当价格）
    if price < 1000 or price > 5_000_000:
        return None
    return price

def _parse_capacity_gb(text: str) -> int | None:
    if not text: return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", text, re.I)
    if m: return int(round(float(m.group(1)) * 1024))
    m = re.search(r"(\d{2,4})\s*GB", text, re.I)
    if m: return int(m.group(1))
    return None

def _normalize_jan(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"[^\d]", "", str(x))

def _extract_part_number(text: str) -> str:
    if not text: return ""
    # PN 更宽松：字母数字 + J/A（日本区），例如 MTUW3J/A、MW123J/A
    m = re.search(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b", text)
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

def _parse_capacity_gb2(text: str) -> int | None:
    if not text: return None
    # TB 优先（含 1.5TB 等）
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", text, re.I)
    if m:
        return int(round(float(m.group(1)) * 1024))
    m = re.search(r"(\d{2,4})\s*GB", text, re.I)
    if m:
        return int(m.group(1))
    return None

def _normalize_model2(text: str) -> str:
    """统一型号主体：iPhone + 数字 + [Pro/Pro Max/Plus/mini]（含日文别名）"""
    if not text: return ""
    t = _clean_ws(text)
    # 先把日文别名替换成英文词干，便于统一
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini"))
    # 在 iPhone 与数字间补空格
    t = re.sub(r"iPhone\s*(\d{2})", r"iPhone \1", t, flags=re.I)
    # 把“容量/SIM/括号”等噪声移除
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # 捕获 “iPhone 16 Pro Max / iPhone 15 Pro / iPhone 14 Plus / iPhone 13 mini / iPhone 12”
    m = re.search(r"(iPhone\s+\d{2}(?:\s+Pro\s*Max|\s+Pro|\s+Plus|\s+mini)?)", t, flags=re.I)
    return m.group(1) if m else ""

def _normalize_model(text: str) -> str:
    if not text: return ""
    t = _clean_ws(text)
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini"))
    t = re.sub(r"iPhone\s*(\d{2})", r"iPhone \1", t, flags=re.I)
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    m = re.search(r"(iPhone\s+\d{2}(?:\s+Pro\s*Max|\s+Pro|\s+Plus|\s+mini)?)", t, flags=re.I)
    return m.group(1) if m else ""

def _extract_part_number2(text: str) -> str:
    """更宽松的 PN：……J/A（J区），允许 A~Z0-9 混合；如 MTUW3J/A"""
    if not text: return ""
    m = re.search(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b", text)
    return m.group(0) if m else ""

def _scan_jan13(texts: List[str]) -> str:
    s = " ".join([_clean_ws(x) for x in texts if x])
    m = re.search(r"\b(\d{13})\b", s)
    return m.group(1) if m else ""

def _to_int_price2(val) -> int | None:
    if val is None: return None
    s = str(val)
    if "万" in s:
        m = re.search(r"([\d\.]+)\s*万", s)
        base = float(m.group(1)) if m else 0.0
        tail = 0
        m2 = re.search(r"万\s*([0-9,]+)", s)
        if m2: tail = int(re.sub(r"[^\d]","", m2.group(1)))
        return int(base*10000 + tail)
    # 若是纯数值/数值列
    if re.search(r"\d", s):
        return int(re.sub(r"[^\d]","", s))
    return None

def _collect_prices_row(row: pd.Series, df: pd.DataFrame) -> List[int]:
    """
        只从“像金额”的列取值：
          - 单元格包含 '円'；或
          - 列名包含 価格/買取/金額/buy/price；或
          - 列是数值型 且 不像 JAN（非 12–14 位）
        然后去重保序，最多取 3 个。
        """
    vals, seen = [], set()
    for c in df.columns:
        v = row.get(c)
        if v is None or (isinstance(v, float) and pd.isna(v)): continue
        s = str(v)
        header = str(c)
        likely = ("円" in s) or re.search(r"(価格|買取|金額|売価|buy|price)", header, re.I) or isinstance(v,
                                                                                                          (int, float))
        if not likely:
            continue
        p = _to_int_price(v)
        if p is not None and p not in seen:
            vals.append(p);
            seen.add(p)
    return vals[:3]


# ==========
# ==========
# ==========

COL_T3_CANDS = ["文本3", "テキスト3", "詳細", "備考", "説明", "スペック", "コメント", "テキスト", "商品詳細"]
COL_T4_CANDS = ["文本4", "JAN", "JANコード", "JANｺｰﾄﾞ", "JANコード(13桁)"]
COL_T5_CANDS = ["文本5", "テキスト5", "新品買取価格", "新品価格", "未開封買取価格", "未開封価格", "NEW", "新品"]
COL_T6_CANDS = ["文本6", "テキスト6", "A品買取価格", "Aランク", "ランクA", "中古A", "Aランク価格"]
# 状态列（可有可无；若无则留空）
COL_STATUS_CANDS = ["文本2", "状態", "コンディション", "ランク", "タグ", "ステータス", "状態区分"]

# ---- 小工具 ----
def _first_nonempty(row: pd.Series, cands: List[str]) -> str:
    for c in cands:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def _clean_text(s: str) -> str:
    s = str(s)
    s = s.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")  # 全角空格/换行→空格
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _normalize_model(text: str) -> str:
    """统一型号：支持日文别名 → 英文词干，再抽取 iPhone + 2位代号 + 后缀"""
    if not text:
        return ""
    t = _clean_text(text)
    # 日文 → 英文词干
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini"))
    # iPhone 与数字之间加空格
    t = re.sub(r"iPhone\s*(\d{2})", r"iPhone \1", t, flags=re.I)
    # 去容量/SIM/括号噪声
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # 匹配 iPhone 16 / 16 Pro / 16 Pro Max / 16 Plus / 13 mini …
    m = re.search(r"(iPhone\s+\d{2}(?:\s+Pro\s*Max|\s+Pro|\s+Plus|\s+mini)?)", t, flags=re.I)
    return m.group(1) if m else ""

def _parse_capacity_gb(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", text, re.I)
    if m:
        return int(round(float(m.group(1)) * 1024))
    m = re.search(r"(\d{2,4})\s*GB", text, re.I)
    if m:
        return int(m.group(1))
    return None

# 常见颜色（日文为主）粗抽；规范化交给 normalize_color
_COLOR_TOKEN = re.compile(
    r"(ブラックチタニウム|ホワイトチタニウム|ブルーチタニウム|デザートチタニウム|ナチュラルチタニウム|"
    r"ブラック|ホワイト|ブルー|グリーン|ピンク|イエロー|ゴールド|シルバー|グレー|パープル|"
    r"ナチュラル|ミッドナイト|スターライト|グラファイト|スペースグレイ)"
)

def _extract_color_raw(text: str) -> str:
    if not text:
        return ""
    m = _COLOR_TOKEN.search(text)
    return m.group(1) if m else ""

def _normalize_jan(x) -> str:
    if x is None:
        return ""
    s = re.sub(r"[^\d]", "", str(x))
    return s if len(s) == 13 else ""

def _extract_pn(text: str) -> str:
    """Apple PN（日本区）：……J/A。允许大写字母数字前缀 + 可选数字序号 + J/A"""
    if not text:
        return ""
    m = re.search(r"\b[A-Z0-9]{4,6}\d{0,3}J/A\b", text)
    return m.group(0) if m else ""

def _to_int_price(s) -> int | None:
    """
    将字符串/数值转 int（日元）：
      - 先排除 12–14 位纯数字（极可能是 JAN / 电话）；
      - 处理 “万” 记法；
      - 去掉货币/逗号/全角；
      - 过滤不合理区间（<1,000 或 >5,000,000）
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # 先处理范围分隔符（全角/半角 ~ / ～）
    # 留给上层的 range 解析；此处只负责单值
    if re.fullmatch(r"\d{12,14}", s):
        return None
    # 处理“万”
    if "万" in s:
        m = re.search(r"([\d\.]+)\s*万", s)
        base = float(m.group(1)) if m else 0.0
        tail = 0
        m2 = re.search(r"万\s*([0-9,]+)", s)
        if m2:
            tail = int(re.sub(r"[^\d]", "", m2.group(1)))
        v = int(base * 10000 + tail)
    else:
        v = re.sub(r"[^\d]", "", s)
        if not v:
            return None
        v = int(v)
    if v < 1000 or v > 5_000_000:
        return None
    return v

def _parse_price_field(s: str) -> int | None:
    """
    解析价格字段：可能是 '¥105,000' / '105000' / '105,000～110,000円'
    策略：把范围拆开后取 **最大值**。
    """
    if s is None:
        return None
    # 统一分隔：全角/半角 ~ / ～ / —
    parts = re.split(r"[~～\-–—]", str(s))
    candidates = []
    for p in parts:
        val = _to_int_price(p)
        if val is not None:
            candidates.append(val)
    if not candidates:
        # 如果是数值型/纯字符串一个整体
        return _to_int_price(s)
    return max(candidates)

def _find_first_col(df: pd.DataFrame, cands: List[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None

# ================== 主函数（重构） ================== #
def _parse_morimori_df(df: pd.DataFrame, *, filename: str | None = None) -> pd.DataFrame:
    """
    读取森森買取 Excel：
      - 文本3：型号+容量+颜色+SIMフリー（可能含换行）
      - 文本4：JAN（13位数字）
      - 文本5：新品回收价格（可能包含 ～；取最大）
      - 文本6：A品回收价格（可能包含 ～；取最大）
    只在识别到 SIMフリー 时，将 'SIMフリー' 写入 '其他没有记录的备用信息'；否则留空。
    """
    # 找出实际列名
    col_t3 = _find_first_col(df, COL_T3_CANDS)
    col_t4 = _find_first_col(df, COL_T4_CANDS)
    col_t5 = _find_first_col(df, COL_T5_CANDS)
    col_t6 = _find_first_col(df, COL_T6_CANDS)
    col_status = _find_first_col(df, COL_STATUS_CANDS)

    if not col_t3:
        raise RuntimeError("森森買取：未找到 '文本3/詳細' 列（型号+容量+颜色+SIM情報 所在列）")
    if not col_t4:
        raise RuntimeError("森森買取：未找到 '文本4/JAN' 列（JAN 所在列）")
    # 价格列可选，但建议存在
    if not col_t5:
        # 不致命，提示但继续
        col_t5 = None
    if not col_t6:
        col_t6 = None

    out_rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        t3 = _clean_text(r.get(col_t3, ""))
        t4 = str(r.get(col_t4, "")).strip()
        t5 = r.get(col_t5) if col_t5 else None
        t6 = r.get(col_t6) if col_t6 else None
        status = _clean_text(r.get(col_status, "")) if col_status else ""

        if not t3 and not t4:
            # 完全无用行
            continue

        # 从文本3抽取型号/容量/颜色/SIM
        model = _normalize_model(t3) or (_normalize_model(filename) if filename else "")
        cap_gb = _parse_capacity_gb(t3)
        print("Model:"+model)
        # print("cap_gb:" + cap_gb)
        color_raw = _extract_color_raw(t3)
        print("color_raw:" + color_raw)
        # 规范色（供后续 pipeline 匹配；本函数“颜色”列仍写原色）
        color_canon, color_any = normalize_color(t3)

        # PN：尽力从文本3里提取（如有）
        pn = _extract_pn(t3)

        # JAN：直接来自 文本4（清洗 13位）
        jan = _normalize_jan(t4)

        # 价格：文本5=新品；文本6=A品；都支持范围取最大；B品此处无 → 价格3=None
        price_new = _parse_price_field(str(t5)) if t5 is not None else None
        price_a   = _parse_price_field(str(t6)) if t6 is not None else None
        price_b   = None

        # SIMフリー：仅在命中时写 canonical 到“其他…”
        is_sf, simfree_canon = _normalize_simfree(t3)
        other_info = simfree_canon if is_sf else ""

        # 若 model 仍为空，尝试从 文件名 兜底
        if not model and filename:
            model = _normalize_model(filename)

        # 只要 model / pn / jan 有其一，即落一行
        if model or pn or jan:
            out_rows.append({
                "店铺名": "森森買取",
                "iphone型号": model,
                "Part_number": pn,
                "JAN": jan,
                "颜色": color_raw or color_canon,    # 没抓到原色时，用规范色兜底
                "状态": status,
                "价格": price_new,
                "价格2": price_a,
                "价格3": price_b,
                "其他没有记录的备用信息": other_info,
            })

    # 统一列并返回
    cols = ["店铺名","iphone型号","Part_number","JAN","颜色","状态","价格","价格2","价格3","其他没有记录的备用信息"]
    res = pd.DataFrame(out_rows)
    for c in cols:
        if c not in res.columns:
            res[c] = "" if c not in ["价格","价格2","价格3"] else None
    return res[cols]


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

# def _parse_morimori_df(df: pd.DataFrame, *, filename: str | None = None) -> pd.DataFrame:
#     out: List[Dict[str, Any]] = []
#
#     for _, row in df.iterrows():
#
#         title   = _first_nonempty(row, TITLE_CANDIDATES)
#         detail  = _first_nonempty(row, DETAIL_CANDIDATES)
#         status  = _first_nonempty(row, STATUS_CANDIDATES)
#         jan_raw = _first_nonempty(row, JAN_CANDIDATES)
#         all_txt = _join_texts(row, list(set(TITLE_CANDIDATES + DETAIL_CANDIDATES + STATUS_CANDIDATES + JAN_CANDIDATES)))
#         # 型号与容量（多重兜底）
#         model = _normalize_model(title) or _normalize_model(all_txt) or (_normalize_model(filename) if filename else "")
#         cap_gb = _parse_capacity_gb(detail) or _parse_capacity_gb(title) or _parse_capacity_gb(all_txt)
#         print("title:"+title)
#         print("detail:" + detail)
#         print("status:" + status)
#         print("all_txt:" + all_txt)
#         # print("cap_gb:" + cap_gb)
#         # 颜色（raw + 规范名/全色标记）
#         color_raw = ""
#         m_col = re.search(r"(ブラック|ホワイト|ブルー|グリーン|ピンク|イエロー|ゴールド|シルバー|グレー|パープル|"
#                           r"ナチュラルチタニウム|ブラックチタニウム|ホワイトチタニウム|ブルーチタニウム|デザートチタニウム|"
#                           r"ナチュラル|ミッドナイト|スターライト|グラファイト|スペースグレイ)", " ".join([title, detail]))
#         if m_col: color_raw = m_col.group(1)
#         color_canon, color_any = normalize_color(" ".join([title, detail]))
#
#         # PN / JAN
#         pn  = _extract_part_number(" ".join([title, detail, all_txt]))
#         jan = _scan_jan13([jan_raw, title, detail, all_txt])
#
#         # SIMフリー 仅在命中时写出 canonical
#         is_sf, sf_canon = _normalize_simfree(" ".join([title, status, detail]))
#         other_info = sf_canon if is_sf else ""
#
#         # 价格（严格过滤：不再把 13 位数字当价格）
#         prices = _collect_prices_row(row, df)
#         p1 = prices[0] if len(prices) > 0 else None
#         p2 = prices[1] if len(prices) > 1 else None
#         p3 = prices[2] if len(prices) > 2 else None
#
#         # 最后兜底：文件名里若带机型词
#         if not model and filename:
#             model = _normalize_model(filename)
#
#         # 只要“型号 or PN or JAN”任一有值，就落一行
#         if model or pn or jan:
#             out.append({
#                 "店铺名": "森森買取",
#                 "iphone型号": model,
#                 "Part_number": pn,
#                 "JAN": jan,
#                 "颜色": color_raw,
#                 "状态": status,
#                 "价格":  p1,
#                 "价格2": p2,
#                 "价格3": p3,
#                 "其他没有记录的备用信息": other_info,
#                 # 下面两列不是标准输出，供上游调试；pipeline 不会用
#                 # "_color_canon": color_canon,
#                 # "_color_any": color_any,
#             })
#
#     # 统一列
#     cols = ["店铺名","iphone型号","Part_number","JAN","颜色","状态","价格","价格2","价格3","其他没有记录的备用信息"]
#     res = pd.DataFrame(out)
#     for c in cols:
#         if c not in res.columns:
#             res[c] = "" if c not in ["价格","价格2","价格3"] else None
#     return res[cols]


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
# ========== 识别 & 入口 ==========
# ========== 识别 & 入口 ==========
# ========== 识别 & 入口 ==========
# ========== 识别 & 入口 ==========


def _read_excel_from_bytes(data: bytes, filename: str = "", sheet: int | str = 0) -> pd.DataFrame:
    """
    仅读取 Excel（xlsx/xlsm/xls/ods 等），首个工作表默认 sheet=0。
    """
    try:
        # pandas 会自动根据扩展名选择 engine；.xlsx 走 openpyxl
        with io.BytesIO(data) as bio:
            return pd.read_excel(bio, sheet_name=sheet, engine=None)
    except Exception as e:
        raise RuntimeError(f"无法读取 Excel（{filename}）：{e}")

def parse_tradein_uploaded(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    将“各店的 Excel 表”清洗为统一列结构：
      店铺名 | iphone型号 | Part_number | JAN | 颜色 | 状态 | 价格 | 价格2 | 价格3 | 其他没有记录的备用信息
    仅支持 Excel；不再读取 CSV。
    """
    df0 = _read_excel_from_bytes(file_bytes, filename, sheet=0)
    base = os.path.basename(filename)

    # —— 按文件名优先识别来源 —— #
    name = base
    if "ルデヤ" in name or "買取ルデヤ" in name:
        return _parse_rudeya_df(df0)
    if "森森" in name or "森森買取" in name:
        return _parse_morimori_df(df0, filename=base)   # ← 本次重写的函数（见下）
    if "モバイル一番" in name:
        return _parse_ichiban_df(df0)
    if "モバイルミックス" in name:
        return _parse_mobilemix_df(df0, shop_name="モバイルミックス")

    # —— 回退：按列名识别 —— #
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

