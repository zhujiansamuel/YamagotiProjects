from __future__ import annotations

import os
import re
import time
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import textwrap
from functools import lru_cache
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb


# ========== 你的 Ollama 配置 ==========
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "gemma3:1b")


# ========== 你原来就有的通用解析（沿用你 shop16 逻辑） ==========
_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)

def _norm(s: str) -> str:
    return (s or "").strip()

COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\s]+)\s*
        (?P<sep>[：:\-])\s*           # 新增：捕获分隔符
        (?P<sign>[+\-−－])?\s*        # 显式正负号（可选）
        (?P<amount>\d[\d,]*)\s*円
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*￥\s*(?P<amount>\d[\d,]*)""",
    re.UNICODE
)

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

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

def _load_iphone17_info_df_for_shop2() -> pd.DataFrame:
    """
    读取 iphone17_info，并尽量保留 jan 列以供其它 shop 做 JAN→PN 映射。
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

    # 标准化 jan 列
    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))

    cols = ["part_number", "model_name", "capacity_gb", "color"]
    if jan_candidates:
        src = jan_candidates[0]
        df["jan"] = df[src]
        cols.append("jan")

    return df[cols]

def _normalize_model_generic(text: str) -> str:
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini")
           .replace("エアー", "Air")
           .replace("エア", "Air"))

    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)
    t = re.sub(r"(?i)\bpro\s*max\b", "Pro Max", t)
    t = re.sub(r"(?i)\bpro\b", "Pro", t)
    t = re.sub(r"(?i)\bplus\b", "Plus", t)
    t = re.sub(r"(?i)\bmini\b", "mini", t)

    if "iPhone" not in t and re.search(r"\b1[0-9]\b", t):
        t = re.sub(r"\b(1[0-9])\b", r"iPhone \1", t, count=1)

    t = re.sub(r"(?i)\biPhone\s+17\s+Air\b", "iPhone Air", t)
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    m = _NUM_MODEL_PAT.search(t)
    if m:
        base = f"{m.group(1)} {m.group(2)}"
        suf  = (m.group(3) or "").strip()
        return f"{base} {suf}".strip()

    m2 = _AIR_PAT.search(t)
    if m2:
        return "iPhone Air"

    return ""

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s*;\s*)")

# ========== 颜色家族匹配（沿用你 shop16 的宽松逻辑；可按 shop15 特性扩展） ==========
FAMILY_SYNONYMS = {
    "blue": ["ブルー", "青", "マリン"], "ブルー": ["ブルー", "青", "マリン"], "青": ["ブルー", "青", "マリン"], "マリン": ["ブルー", "青", "マリン"],
    "black": ["ブラック", "黒"], "ブラック": ["ブラック", "黒"], "黒": ["ブラック", "黒"],
    "white": ["ホワイト", "白"], "ホワイト": ["ホワイト", "白"], "白": ["ホワイト", "白"],
    "green": ["グリーン", "緑"], "グリーン": ["グリーン", "緑"], "緑": ["グリーン", "緑"],
    "red": ["レッド", "赤"], "レッド": ["レッド", "赤"], "赤": ["レッド", "赤"],
    "yellow": ["イエロー", "黄"], "イエロー": ["イエロー", "黄"], "黄": ["イエロー", "黄"],
    "orange": ["オレンジ", "橙"], "オレンジ": ["オレンジ", "橙"], "橙": ["オレンジ", "橙"],
    "silver": ["シルバー", "銀"], "シルバー": ["シルバー", "銀"], "銀": ["シルバー", "銀"],
    "gold": ["ゴールド", "金"], "ゴールド": ["ゴールド", "金"], "金": ["ゴールド", "金"],
    "gray": ["グレー", "グレイ", "灰"], "グレー": ["グレー", "グレイ", "灰"], "グレイ": ["グレー", "グレイ", "灰"], "灰": ["グレー", "グレイ", "灰"],
    "natural": ["ナチュラル"], "ナチュラル": ["ナチュラル"],
}

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


def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True

    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS:
            candidates.update(FAMILY_SYNONYMS[k])

    if not candidates:
        for _, toks in FAMILY_SYNONYMS.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break

    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
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

_GROUP_SHARED_DELTA_RE = re.compile(
    r"""
    (?P<labels>[^0-9￥円]+?)          # 多颜色标签段（含 /）
    \s*(?P<sign>[+\-−－])\s*         # 显式正负号
    (?P<amount>\d[\d,]*)\s*(?:円)?   # 金额（可无 円）
    """,
    re.UNICODE | re.VERBOSE
)

def _extract_shared_delta_map_shop16(price_text_norm: str) -> Dict[str, int]:
    """
    从原文中抽取： 'オレンジ/青 -1500' 这种共享差价 -> {オレンジ:-1500, 青:-1500}
    这是“纠错用”的确定性证据，不替代你让 LLM 抽取的主流程。
    """
    s = price_text_norm or ""
    out: Dict[str, int] = {}
    # 去掉基础价前缀，减少误匹配（基础价一般在最前）
    m0 = FIRST_YEN_RE.search(s)
    tail = s[m0.end():] if m0 else s

    for m in _GROUP_SHARED_DELTA_RE.finditer(tail):
        labels_raw = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        delta = -int(amt) if sign in ("-", "−", "－") else int(amt)

        # 拆分 labels（/、，等）
        for lb in re.split(r"[／/、，,]", labels_raw):
            lb = _normalize_label_shop16(lb)
            if lb:
                out[lb] = delta
    return out



def _normalize_price_text_shop16(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    # 把换行变成分隔（保留“下一行是颜色差价”的结构）
    s = re.sub(r"[\r\n]+", " / ", s)
    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    # 多个分隔合并
    s = re.sub(r"(?:\s*/\s*){2,}", " / ", s).strip()
    return s


_BASE_ONLY_RE = re.compile(r"^\s*(?:￥|\¥)?\s*\d[\d,]*\s*(?:円)?\s*$")

def _is_base_only_price_text(price_text_norm: str) -> bool:
    return bool(_BASE_ONLY_RE.match(price_text_norm or ""))

# ========== LangExtract + Ollama：替代正则拆价 ==========
def _to_signed_int_yen(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # 优先：找带符号的数（通常是差价）
    signed = list(re.finditer(r"([+\-−－])\s*(\d[\d,]*)", s))
    if signed:
        m = signed[-1]
        sign = m.group(1)
        amt = to_int_yen(m.group(2))
        if amt is None:
            return None
        return -int(amt) if sign in ("-", "−", "－") else int(amt)

    # 其次：取最后一个数字（避免把 base_price 当 delta）
    nums = list(re.finditer(r"(\d[\d,]*)", s))
    if not nums:
        return None
    amt = to_int_yen(nums[-1].group(1))
    return int(amt) if amt is not None else None

_TRAILING_AMOUNT_IN_LABEL_RE = re.compile(
    r"(?:[：:])?\s*(?:￥)?\s*[+\-−－]?\s*\d[\d,]*\s*(?:円)?\s*$",
    re.UNICODE,
)
def _normalize_label_shop16(lbl: str) -> str:
    s = re.sub(r"[\s\u3000\xa0]+", "", lbl or "")
    s = re.sub(r"(カラー|色)$", "", s)
    # 去掉黏在 label 末尾的金额/符号：-1000 / ￥86100 / :-1,000円 等
    s = _TRAILING_AMOUNT_IN_LABEL_RE.sub("", s)
    return s.strip()

def _split_labels_shop16(lbl: str) -> List[str]:
    # 兼容 “青/オレンジ”“黒、白”“blue/black” 等
    raw = _normalize_label_shop16(lbl)
    parts = re.split(r"[／/、，,]", raw)
    return [p for p in (_normalize_label_shop16(x) for x in parts) if p]

def _extract_base_price_shop16(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)  # 兜底
    return to_int_yen(m.group(1))


SHOP16_PRICE_PROMPT = textwrap.dedent("""\
You extract pricing information from Japanese iPhone buyback price strings (買取価格).
Extract ONLY what is explicitly stated in the text; do not guess.

Classes to extract:
1) base_price:
   - the unlabeled base price in yen (e.g. "86,100円", "￥86100").
2) color_delta:
   - per-color adjustment relative to base_price in yen (e.g. "黒:-1000円", "青 +5000円").
   - If multiple colors share a delta (e.g. "青/オレンジ -5000円"), output one color_delta per color label.
3) color_abs:
   - per-color absolute price in yen (e.g. "黒￥86100", "青￥87100").

Rules:
- extraction_text must be an exact span from the input text (no paraphrase).
- Do not invent colors or amounts.
- Attributes schema:
  * base_price: {"amount_yen": "<int>"}
  * color_delta: {"color_label": "<label>", "delta_yen": "<signed int>"}
  * color_abs: {"color_label": "<label>", "amount_yen": "<int>"}
""")

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

def _shop16_price_examples():
    import langextract as lx

    return [
        lx.data.ExampleData(
            text="86,100円 黒:-1,000円 青:+500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="86,100円",
                    attributes={"amount_yen": "86100"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="黒",
                    attributes={"color_label": "黒", "delta_yen": "-1000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="青",
                    attributes={"color_label": "青", "delta_yen": "+500"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="86100円 / 青/オレンジ -5000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="86100円",
                    attributes={"amount_yen": "86100"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="青",
                    attributes={"color_label": "青", "delta_yen": "-5000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="オレンジ",
                    attributes={"color_label": "オレンジ", "delta_yen": "-5000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="黒￥86100/青￥87100",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_abs",
                    extraction_text="黒",
                    attributes={"color_label": "黒", "amount_yen": "86100"},
                ),
                lx.data.Extraction(
                    extraction_class="color_abs",
                    extraction_text="青",
                    attributes={"color_label": "青", "amount_yen": "87100"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="￥90000 ホワイト +0円／ブラック -3000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="￥90000",
                    attributes={"amount_yen": "90000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ホワイト",
                    attributes={"color_label": "ホワイト", "delta_yen": "0"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"color_label": "ブラック", "delta_yen": "-3000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="92,000円 ブルー：+2,000円 グリーン:-1000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="92,000円",
                    attributes={"amount_yen": "92000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"color_label": "ブルー", "delta_yen": "+2000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="グリーン",
                    attributes={"color_label": "グリーン", "delta_yen": "-1000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="￥197000\n\nオレンジ-1000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="￥197000",
                    attributes={"amount_yen": "197000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="オレンジ-1000",
                    attributes={"color_label": "オレンジ", "delta_yen": "-1000"},
                ),
            ],
        ),
    ]
@lru_cache(maxsize=4096)
def _lx_extract_price_parts_shop16(
    price_text: str,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]], List[dict]]:
    """
    返回：
      base_price: Optional[int]
      deltas: [(label, delta_yen)]
      abs_prices: [(label, abs_yen)]
      debug_extractions: [{"class","text","attrs","span"}]
    """
    s = (price_text or "").strip()
    if not s:
        return None, [], [], []

    try:
        import langextract as lx
    except ImportError as e:
        raise ImportError("缺少依赖：pip install langextract") from e

    examples = _shop16_price_examples()

    kwargs = dict(
        text_or_documents=s,
        prompt_description=SHOP16_PRICE_PROMPT,
        examples=examples,
        model_id=OLLAMA_MODEL_ID,
        model_url=OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
        extraction_passes=1,
        max_char_buffer=300,
    )
    # 与官方 release 示例兼容（有些版本需要显式指定）
    try:
        kwargs["language_model_type"] = lx.inference.OllamaLanguageModel
    except Exception:
        pass

    result = lx.extract(**kwargs)

    # 某些情况下可能返回 list；统一成 list 处理
    docs = result if isinstance(result, list) else [result]

    base_price: Optional[int] = None
    deltas: List[Tuple[str, int]] = []
    abs_prices: List[Tuple[str, int]] = []
    debug_extractions: List[dict] = []

    for doc in docs:
        exs = list(getattr(doc, "extractions", None) or [])
        for ex in exs:
            cls = getattr(ex, "extraction_class", "") or ""
            txt = getattr(ex, "extraction_text", "") or ""
            attrs = getattr(ex, "attributes", {}) or {}

            ci = getattr(ex, "char_interval", None)
            span = None
            if ci is not None:
                span = {"start": getattr(ci, "start_pos", None), "end": getattr(ci, "end_pos", None)}
            debug_extractions.append({"class": cls, "text": txt, "attrs": dict(attrs), "span": span})

            if cls == "base_price" and base_price is None:
                v = attrs.get("amount_yen")
                amt = to_int_yen(v) if v is not None else to_int_yen(txt)
                if amt is not None:
                    base_price = int(amt)

            elif cls == "color_delta":
                raw_label = str(attrs.get("color_label") or txt or "").strip()
                dv = attrs.get("delta_yen")
                delta = _to_signed_int_yen(dv if dv is not None else "")
                # 若模型没给 delta_yen，则尝试从原文里兜底
                if delta is None:
                    delta = _to_signed_int_yen(txt)

                if delta is not None:
                    for lb in _split_labels_shop16(raw_label):
                        deltas.append((lb, int(delta)))

            elif cls == "color_abs":
                raw_label = str(attrs.get("color_label") or txt or "").strip()
                av = attrs.get("amount_yen")
                amt = to_int_yen(av) if av is not None else to_int_yen(txt)

                if amt is not None:
                    for lb in _split_labels_shop16(raw_label):
                        abs_prices.append((lb, int(amt)))

    return base_price, deltas, abs_prices, debug_extractions

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

MODEL_COL = "iPhone 17 Pro Max"
DESC_COL  = "説明1"
PRICE_COL = "買取価格"


def clean_shop16(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    # print("shop16:携帯空間---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print("shop4:モバイルミックス---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"shop16:携帯空間---------->进入清洗器时间: {now}")
    for c in [MODEL_COL, DESC_COL, PRICE_COL, "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop16 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop16(info_df)

    rows: List[dict] = []

    def _looks_like_has_color_info(price_text: str) -> bool:
        if not price_text:
            return False
        # 有分隔符/符号/￥等，很可能携带颜色差价/绝对价
        tokens = ("／", "/", "￥", "：", ":", "+", "-", "−", "－")
        return any(t in price_text for t in tokens)

    for idx, row in df.iterrows():
        model_cell = str(row.get(MODEL_COL) or "").strip()
        desc_cell  = str(row.get(DESC_COL)  or "").strip()
        price_cell = row.get(PRICE_COL)
        rec_at     = parse_dt_aware(row.get("time-scraped"))

        is_unopened = ("未開封" in desc_cell) or ("未開封" in model_cell)
        if not is_unopened:
            continue

        model_text = model_cell.replace("\u3000", " ").replace("\xa0", " ").replace("\n", " ").strip()
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        price_raw = "" if price_cell is None else str(price_cell)
        price_text = _normalize_price_text_shop16(price_raw)

        # 1) 先走 LangExtract + Ollama 抽取
        base_llm = None
        deltas: List[Tuple[str, int]] = []
        absps: List[Tuple[str, int]] = []
        dbg_extractions: List[dict] = []
        llm_ok = False

        try:
            base_llm, deltas, absps, dbg_extractions = _lx_extract_price_parts_shop16(price_text)
            llm_ok = True
        except Exception as e:
            # LangExtract/Ollama 失败时，不让整批崩掉；回退到旧逻辑（保持可用性）
            llm_ok = False
            if debug:
                print("\n[shop16][llm] LangExtract/Ollama 解析失败，回退旧逻辑。row=", idx, "err=", repr(e))

        # LangExtract + Ollama 抽取

        base_price = base_llm
        if base_price is None:
            base_price = _extract_base_price_shop16(price_text)
        base_price = int(base_price) if base_price is not None else None

        # -----------------------------
        # Guardrail A: 只有基础价 -> 丢弃所有 color_delta/color_abs（防幻觉）
        # -----------------------------
        if _is_base_only_price_text(price_text):
            deltas = []
            absps = []

        # -----------------------------
        # Guardrail B: 共享差价纠错（修正 “青” 被抽成 +1500）
        # -----------------------------
        shared_delta_map = _extract_shared_delta_map_shop16(price_text)
        if shared_delta_map and deltas:
            corrected: List[Tuple[str, int]] = []
            for label_raw, delta in deltas:
                lb = _normalize_label_shop16(label_raw)
                if not lb:
                    continue
                if lb in shared_delta_map:
                    corrected.append((lb, int(shared_delta_map[lb])))
                else:
                    corrected.append((lb, int(delta)))
            deltas = corrected

        # -----------------------------
        # Guardrail C: 逐条证据过滤 —— label/金额必须在原文出现（进一步防幻觉）
        # -----------------------------
        text_no_commas = price_text.replace(",", "")
        filtered_deltas: List[Tuple[str, int]] = []
        for label_raw, delta in deltas:
            lb = _normalize_label_shop16(label_raw)
            if not lb:
                continue
            # label 必须出现
            if lb not in price_text:
                continue
            # 金额数字必须出现（忽略符号，只校验绝对值数字）
            if str(abs(int(delta))) not in text_no_commas:
                continue
            filtered_deltas.append((lb, int(delta)))
        deltas = filtered_deltas

        filtered_absps: List[Tuple[str, int]] = []
        for label_raw, amt in absps:
            lb = _normalize_label_shop16(label_raw)
            if not lb:
                continue
            if lb not in price_text:
                continue
            if str(int(amt)) not in text_no_commas:
                continue
            filtered_absps.append((lb, int(amt)))
        absps = filtered_absps

        # # 2) base_price：优先 LLM，其次旧 base 提取
        # base_price = base_llm
        # if base_price is None:
        #     base_price = _extract_base_price_shop16(price_text)  # 仅做数值兜底（旧逻辑）
        # if base_price is not None:
        #     base_price = int(base_price)

        # 3) 若 LLM 没抽到任何颜色信息且它失败了，可选回退旧的颜色解析（避免漏数据）
        if (not llm_ok) and (not deltas) and (not absps):
            # 这里仍用你原来的正则函数做容错回退
            deltas = _extract_color_deltas_shop16(price_text)
            absps  = _extract_color_abs_prices_shop16(price_text)

        # 4) 没 base 且没 abs：没法落库
        if base_price is None and not absps:
            continue

        # 5) 标签映射到具体 color_norm
        color_delta_map: Dict[str, int] = {}
        color_abs_map: Dict[str, int] = {}

        if deltas:
            for col_norm, (_pn, col_raw) in color_map.items():
                for label_raw, delta in deltas:
                    label_raw2 = _normalize_label_shop16(label_raw)
                    if _label_matches_color_shop16(label_raw2, col_raw, col_norm):
                        color_delta_map[col_norm] = int(delta)

        if absps:
            for col_norm, (_pn, col_raw) in color_map.items():
                for label_raw, abs_price in absps:
                    label_raw2 = _normalize_label_shop16(label_raw)
                    if _label_matches_color_shop16(label_raw2, col_raw, col_norm):
                        color_abs_map[col_norm] = int(abs_price)

        # 6) debug：只在“确实疑似有颜色信息”时打印
        if debug and (_looks_like_has_color_info(price_text) or deltas or absps):
            print("\n" + "-" * 120)
            print(f"[shop16 debug] row_index={idx} llm_ok={llm_ok}")
            # print(f"  model_text: {model_text!r}")
            print(f"  model_norm/cap: {model_norm!r} / {cap_gb}")
            print(f"  price_raw               : {price_text!r}")
            # print(f"  extracted(base_llm/base_final): {base_llm!r} / {base_price!r}")
            print(f"  extracted(deltas)       : {deltas!r}")
            print(f"  matched color_delta_map : {color_delta_map}")
            print(f"  extracted(absps)        :  {absps!r}")
            print(f"  matched color_abs_map   : {color_abs_map} ")

            # if dbg_extractions:
            #     print("  llm_extractions:")
            #     for it in dbg_extractions:
            #         print("   -", it)

            # print("  color candidates (iphone17_info):")
            # for col_norm, (pn, col_raw) in color_map.items():
            #     print(f"    - color_norm={col_norm!r}, color_raw={col_raw!r}, pn={pn!r}")

            print("  final prices by color:")
            for col_norm, (pn, col_raw) in color_map.items():
                if col_norm in color_abs_map:
                    final_price = color_abs_map[col_norm]
                    src = "abs"
                else:
                    if base_price is None:
                        continue
                    delta = color_delta_map.get(col_norm, 0)
                    final_price = int(base_price) + int(delta)
                    src = "base+delta" if col_norm in color_delta_map else "base"
                print(f"    - {col_norm!r:} ({col_raw:}) : {final_price:<7} [{src}]")

        # 7) 生成输出：绝对价优先，否则 base ± delta
        for col_norm, (pn, _col_raw) in color_map.items():
            if col_norm in color_abs_map:
                price_new = color_abs_map[col_norm]
            else:
                if base_price is None:
                    continue
                price_new = int(base_price) + int(color_delta_map.get(col_norm, 0))

            rows.append({
                "part_number": str(pn),
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