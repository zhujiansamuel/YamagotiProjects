from __future__ import annotations

"""
shop9 清洗器 — アキモバ

  原始文本（買取価格 + 色・詳細等）
    │
    ├─ _coerce_signed_int()                  ← Step 1: 金額解析（全角→半角、符号処理）
    │
    ├─ _bucket_amount()                      ← Step 2: abs/delta 分類（量級・符号ヒント）
    │
    ├─ _extract_price_parts_shop9_dispatch()  ← Step 7: モード調度
    │   │
    │   ├─ regex 路径:
    │   │   ├─ _extract_abs_prices_regex()        ← Step 5a: 正則提取絶対価
    │   │   ├─ _extract_deltas_regex()            ← Step 5b: 正則提取差価
    │   │   └─ _direct_abs_overrides_for_row()    ← Step 5c: テキスト直接覆写
    │   │
    │   └─ llm 路径:
    │       ├─ _llm_extract_rules_cached()        ← Step 6a: LLM 核心提取
    │       └─ _bucket_amount() guardrail         ← Step 6b: abs/delta 防幻觉過濾
    │
    ├─ _map_to_available_color()             ← Step 3: ラベル→カラーマッチング
    │
    └─ clean_shop9()                         ← Step 8: 主函数、出力行生成
"""

import logging
import os
import re
import json
import time
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
)

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

OLLAMA_URL = os.getenv("SHOP9_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
LLM_MODEL_ID = os.getenv("SHOP9_LX_MODEL_ID") or os.getenv("SHOP9_LLM_MODEL_ID") or "gemma3:1b"
LLM_TEMPERATURE = float(os.getenv("SHOP9_LLM_TEMPERATURE", "0.0") or "0.0")

SHOP9_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

ABS_LIKE_MIN = int(os.getenv("SHOP9_ABS_LIKE_MIN", "50000"))  # iPhone17 绝对价量级阈值

COL_MODEL = "機種名"
COL_PRICE = "買取価格"
COL_COLOR = "色・詳細等"
COL_TIME  = "time-scraped"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

def _truncate_for_log(s: str, n: int = 200) -> str:
    """截断长字符串，保留前 n 个字符，用于日志显示"""
    if s is None:
        return ""
    t = str(s)
    if len(t) <= n:
        return t
    return t[:n] + f"... (truncated, total_length={len(t)})"

def _norm(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)
    # 全角数字转半角
    t = t.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return t

def _norm_cls(x: str) -> str:
    # 容错：abs price / abs-price / ABS_PRICE 统一
    s = (x or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    return s

# ----------------------------------------------------------------------
# Step 1: 金額解析
# ----------------------------------------------------------------------

def _coerce_signed_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return int(x)

    s = str(x)
    # 全角数字/符号 -> 半角
    s = s.translate(str.maketrans("０１２３４５６７８９＋－−，", "0123456789+--,"))

    sign = 1
    digits = []
    sign_seen = False
    started = False
    for ch in s:
        if not started and not sign_seen and ch in "+-":
            sign_seen = True
            sign = -1 if ch == "-" else 1
            continue
        if ch.isdigit():
            started = True
            digits.append(ch)
            continue
        if started and ch in {",", " "}:
            # 千分位分隔符忽略
            continue
        if started:
            break

    if not digits:
        return None
    try:
        return sign * int("".join(digits))
    except Exception:
        return None

def _norm_amount_to_int(x: str) -> Optional[int]:
    if not x:
        return None
    s = str(x).strip()
    s = s.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    s = s.replace(",", "")
    if not s.isdigit():
        return None
    return int(s)

# ----------------------------------------------------------------------
# Step 2: abs/delta 分類
# ----------------------------------------------------------------------

DELTA_HINT_RE = re.compile(r"(?:[+\-−－]|値下げ|値引|割引|円引|OFF|オフ|減額)", re.I)

def _bucket_amount(cls_norm: str, ex_text: str, amt: int) -> str:
    """
    返回 "abs" 或 "delta"
    规则：
      - 有负号/折扣词/加减符号 => delta
      - 金额量级很大(>=ABS_LIKE_MIN)且无加减线索 => abs（即使模型标成 delta）
      - 其余按 class；不认识则按金额量级兜底
    """
    tx = ex_text or ""
    if amt is None:
        return "delta"
    if amt < 0:
        return "delta"
    if DELTA_HINT_RE.search(tx):
        return "delta"
    if abs(amt) >= ABS_LIKE_MIN:
        return "abs"
    if cls_norm in {"abs_price", "abs", "absolute"}:
        return "abs"
    if cls_norm in {"delta", "delta_price", "adjust", "adjustment"}:
        return "delta"
    return "delta"

# ----------------------------------------------------------------------
# Step 3: 颜色家族同义词 & マッチング
# ----------------------------------------------------------------------

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
    "オレンジ": ["オレンジ", "橙", "コズミックオレンジ"],
    "橙": ["オレンジ", "橙", "コズミックオレンジ"],
    "コズミックオレンジ": ["コズミックオレンジ", "オレンジ", "橙", "orange"],
    "white": ["ホワイト", "白"],
    "ホワイト": ["ホワイト", "白"],
}

SYNONYM_LOOKUP: Dict[str, List[str]] = {}
for _k, _vs in FAMILY_SYNONYMS_SHOP9.items():
    SYNONYM_LOOKUP[_k] = list(dict.fromkeys(_vs))
    for _v in _vs:
        SYNONYM_LOOKUP.setdefault(_v, [])
        SYNONYM_LOOKUP[_v] = list(dict.fromkeys(SYNONYM_LOOKUP[_v] + _vs + [_k]))

def _build_color_aliases(available_colors: List[str]) -> Dict[str, List[str]]:
    out = {}
    for c in available_colors:
        c0 = str(c).strip()
        if not c0:
            continue
        syns = SYNONYM_LOOKUP.get(c0, [])
        out[c0] = list(dict.fromkeys([c0] + syns))[:20]
    return out

def _map_to_available_color(raw_color: str, available_set: set) -> Optional[str]:
    if not raw_color:
        return None
    rc = str(raw_color).strip()
    if not rc:
        return None

    if rc.upper() == "ALL" or rc == "全色":
        return "ALL"

    if rc in available_set:
        return rc

    # 小写等价
    rcn = _norm(rc)
    for c in available_set:
        if _norm(c) == rcn:
            return c

    # 同义词兜底
    if rc in SYNONYM_LOOKUP:
        for syn in SYNONYM_LOOKUP[rc]:
            if syn in available_set:
                return syn
            synn = _norm(syn)
            for c in available_set:
                if _norm(c) == synn:
                    return c

    # 包含关系兜底
    for c in available_set:
        cn = _norm(c)
        if rcn and (rcn in cn or cn in rcn):
            return c

    return None

# ----------------------------------------------------------------------
# Step 4: 正则模式定义
# ----------------------------------------------------------------------

SPLIT_SEPS = r"[/／、，,;；\s]+"

ABS_PRICE_RE = re.compile(
    r"(?P<labels>[^0-9０-９¥￥円]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
    re.I,
)
DELTA_RE = re.compile(
    r"(?P<labels>[^0-9０-９¥￥円]+?)\s*[：:\-]?\s*(?P<sign>[+\-−－])\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
    re.I,
)

# ----------------------------------------------------------------------
# Step 5: 正則提取関数
# ----------------------------------------------------------------------

def _is_pure_number_token(tok: str) -> bool:
    if not tok:
        return False
    t = _norm(tok)
    t = t.replace(",", "").replace("，", "")
    return t.isdigit()

def _extract_abs_prices_regex(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    s = str(text)
    for m in ABS_PRICE_RE.finditer(s):
        labels_part = (m.group("labels") or "").strip()
        amt = _norm_amount_to_int(m.group("amount"))
        if amt is None:
            continue
        toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            out.append((tok, int(amt)))
    return out

def _extract_deltas_regex(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    s = str(text)
    for m in DELTA_RE.finditer(s):
        labels_part = m.group("labels") or ""
        sign = m.group("sign") or "+"
        amt = _norm_amount_to_int(m.group("amount"))
        if amt is None:
            continue
        delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
        toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            out.append((tok, delta))
    if not out and "全色" in s:
        out.append(("全色", 0))
    return out

def _extract_amount_after_alias(text: str, alias: str) -> Optional[int]:
    """
    在 text 中查找形如 'alias 193,500' / 'alias193,500' / 'alias 193500円' 这种片段，
    只取 alias 后面"最近的那串数字"。
    不吃减价形式 'alias-500円'（中间有 '-'）。
    """
    if not text or not alias:
        return None
    s = str(text)

    # 允许 alias 后有若干空白，再跟可选的货币符号，再跟数字
    pat = re.compile(
        rf"{re.escape(alias)}\s*(?:¥|￥)?\s*([0-9０-９][0-9０-９,，]*)"
    )
    m = pat.search(s)
    if not m:
        return None
    return _norm_amount_to_int(m.group(1))

def _direct_abs_overrides_for_row(
        raw_color_text: str,
        color_to_pn: Dict[str, str],
) -> Dict[str, int]:
    """
    针对当前行，直接在 raw_color_text 里按"每个颜色的别名 -> 紧随其后的数字"扫描，
    得到 per-color 的绝对价覆盖表：{color_norm: amount_yen}。
    只接受金额 >= ABS_LIKE_MIN，避免把 -500 / 500 之类 delta 当成 abs。
    """
    overrides: Dict[str, int] = {}
    if not raw_color_text:
        return overrides

    s = str(raw_color_text)
    for col_norm in color_to_pn.keys():
        # 构建该颜色的别名集合：自身 + 同义词
        aliases = {col_norm}
        for syn in SYNONYM_LOOKUP.get(col_norm, []):
            aliases.add(str(syn).strip())
        amt_for_color: Optional[int] = None
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            val = _extract_amount_after_alias(s, alias)
            if val is not None and val >= ABS_LIKE_MIN:
                amt_for_color = val
                break
        if amt_for_color is not None:
            overrides[col_norm] = int(amt_for_color)

    return overrides

def _extract_price_parts_shop9_regex(
    s_price: str,
    s_color: str,
    color_to_pn: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    纯正則版：从 price/color 文本中提取 abs_map / delta_map。
    """
    abs_map: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}

    abs_list = _extract_abs_prices_regex(s_color) or _extract_abs_prices_regex(s_price)
    deltas = _extract_deltas_regex(s_color) or _extract_deltas_regex(s_price)

    def _match_label_to_colnorm(tok: str) -> Optional[str]:
        if not tok:
            return None
        tok_norm = _norm(tok)
        for col_norm in color_to_pn.keys():
            if tok_norm == col_norm:
                return col_norm
        candidates = set()
        if tok_norm in SYNONYM_LOOKUP:
            candidates.update(SYNONYM_LOOKUP[tok_norm])
        candidates.add(tok_norm)
        for cand in candidates:
            candn = _norm(cand)
            for col_norm in color_to_pn.keys():
                cn = _norm(col_norm)
                if candn == cn or candn in cn or cn in candn:
                    return col_norm
        tok_short = re.sub(r"[\s\u3000\-]+", "", tok_norm)
        for col_norm in color_to_pn.keys():
            cn_short = re.sub(r"[\s\u3000\-]+", "", _norm(col_norm))
            if tok_short and (tok_short in cn_short or cn_short in tok_short):
                return col_norm
        return None

    for label_raw, amt in abs_list:
        toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            matched = _match_label_to_colnorm(tok)
            if matched:
                abs_map[matched] = int(amt)

    for label_raw, delta in deltas:
        if label_raw == "全色":
            delta_map["ALL"] = int(delta)
            continue
        toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            matched = _match_label_to_colnorm(tok)
            if matched:
                delta_map[matched] = int(delta)

    return abs_map, delta_map

# ----------------------------------------------------------------------
# Step 6: LLM 配置 & 核心提取函数
# ----------------------------------------------------------------------

SHOP9_PRICE_PROMPT_TEMPLATE = textwrap.dedent("""\
You are parsing Japanese iPhone buyback pricing notes.

Goal:
- Extract explicit color-scoped absolute prices and signed adjustments from the input.
- Extract ONLY what is explicitly present. Do NOT infer missing prices or colors.

How to interpret the format (VERY IMPORTANT):
- The detail field (色・詳細等) may contain multiple independent groups separated by '/', '／', newline.
- In each group, one amount (e.g. 230,500) applies to the color(s) listed immediately before it in that group.
- Multiple colors in the same group can be separated by ',', '，', '、', or spaces. All those colors share the same amount in that group.
- Example: "橙,銀230,500/青229,000" must produce TWO extractions:
  1) colors=["橙","銀"], amount_yen=230500
  2) colors=["青"], amount_yen=229000
- Condition words are NOT colors: ignore terms like "未開", "未使用", "中古", "美品", etc.
- When several colors and numbers appear in one sequence without separators
  (e.g. "橙193,500青193,500銀195,000"), each color MUST be paired with the closest number immediately following it.

What to output:
- extraction_class MUST be one of: "abs_price", "delta"
- attributes.amount_yen MUST be an integer yen value (no commas). For delta, keep the sign (e.g. -2000).
- attributes.colors MUST be a list of color labels AS THEY APPEAR IN THE INPUT (e.g. "青", "銀", "橙").
  You may also output "ALL" only when the text explicitly indicates all colors (e.g. "全色").
- Do NOT drop a price mention just because it equals the base price shown in 買取価格.

Normalization hints (for your reference):
AVAILABLE_COLORS (system will map your labels to these):
{available_colors}

COLOR_ALIASES (system will map using these aliases):
{aliases}
""")

@lru_cache(maxsize=1)
def _shop9_lx_examples():
    """
    Few-shot 示例：教模型识别
    - "多个颜色共享一个价格"
    - "全色 +/-"
    - "每色 +/-"
    """
    import langextract as lx

    return [
        lx.data.ExampleData(
            text="買取価格: 195,500円\n色・詳細等: 未開 橙194,500/青,銀195,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="橙194,500",
                    attributes={"colors": ["コズミックオレンジ"], "amount_yen": 194500},
                ),
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="青,銀195,500",
                    attributes={"colors": ["ディープブルー", "シルバー"], "amount_yen": 195500},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="買取価格: 200,000円\n色・詳細等: ブラック -2,000円 / シルバー:+1000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="ブラック -2,000円",
                    attributes={"colors": ["ブラック"], "amount_yen": -2000},
                ),
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="シルバー:+1000",
                    attributes={"colors": ["シルバー"], "amount_yen": 1000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="買取価格: 180,000円\n色・詳細等: 全色-500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="全色-500円",
                    attributes={"colors": ["ALL"], "amount_yen": -500},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="買取価格: -\n色・詳細等: ブルー：229,000円 シルバー：230000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="ブルー：229,000円",
                    attributes={"colors": ["ブルー"], "amount_yen": 229000},
                ),
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="シルバー：230000",
                    attributes={"colors": ["シルバー"], "amount_yen": 230000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="買取価格: 230,500円\n色・詳細等: 未開 橙,銀230,500/青229,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="橙,銀230,500",
                    attributes={"colors": ["橙", "銀"], "amount_yen": 230500},
                ),
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="青229,000",
                    attributes={"colors": ["青"], "amount_yen": 229000},
                ),
            ],
        ),
    ]

@lru_cache(maxsize=4096)
def _llm_extract_rules_cached(
    price_text: str,
    detail_text: str,
    avail_colors_key: Tuple[str, ...],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    返回:
      abs_map: {color_norm or 'ALL': amount_yen}
      delta_map: {color_norm or 'ALL': signed_delta_yen}
    """
    try:
        import langextract as lx
    except Exception:
        return {}, {}

    available_colors = list(avail_colors_key)
    aliases = _build_color_aliases(available_colors)

    # 输入拼接：让模型同时看到"基准价"和"详情"
    input_text = f"買取価格: {price_text}\n色・詳細等: {detail_text}"

    prompt = SHOP9_PRICE_PROMPT_TEMPLATE.format(
        available_colors=json.dumps(available_colors, ensure_ascii=False),
        aliases=json.dumps(aliases, ensure_ascii=False),
    )

    kw = dict(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=_shop9_lx_examples(),
        model_id=LLM_MODEL_ID,
        model_url=OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
    )

    # 兼容不同版本参数签名：temperature 可能不被支持
    try:
        result = lx.extract(**kw, temperature=LLM_TEMPERATURE)
    except TypeError:
        result = lx.extract(**kw)
    except Exception:
        return {}, {}

    abs_map: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}

    extractions = getattr(result, "extractions", None) or []
    avail_set = set(available_colors)

    for ex in extractions:
        cls_raw = str(getattr(ex, "extraction_class", "") or "")
        cls_norm = _norm_cls(cls_raw)
        attrs = getattr(ex, "attributes", None) or {}
        ex_text = str(getattr(ex, "extraction_text", "") or "")

        # 取 amount
        amt = attrs.get("amount_yen")
        amt_i = _coerce_signed_int(amt)
        if amt_i is None:
            amt_i = _coerce_signed_int(ex_text)
        if amt_i is None:
            continue

        # colors
        colors = attrs.get("colors") or attrs.get("color") or []
        if isinstance(colors, str):
            colors = [colors]
        if not isinstance(colors, list):
            colors = list(colors) if colors else []

        # Guardrail: _bucket_amount only applies to LLM path
        bucket = _bucket_amount(cls_norm, ex_text, int(amt_i))

        for c_raw in colors:
            mapped = _map_to_available_color(str(c_raw), avail_set)
            if not mapped:
                continue
            if bucket == "abs":
                abs_map[mapped] = int(amt_i)
            else:
                delta_map[mapped] = int(amt_i)

    return abs_map, delta_map

def _extract_price_parts_shop9_llm_with_guardrails(
    s_price: str,
    s_color: str,
    color_to_pn: Dict[str, str],
    row_index: object = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    LLM 提取 + Guardrail (_bucket_amount)（仅 LLM 路径使用）。
    """
    avail_colors_key = tuple(color_to_pn.keys())
    abs_map: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}

    try:
        abs_map, delta_map = _llm_extract_rules_cached(s_price, s_color, avail_colors_key)
    except Exception as e:
        logger.warning(
            "LangExtract extraction failed",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": "アキモバ",
                "cleaner_name": "shop9",
                "error": str(e),
                "error_type": type(e).__name__,
                "model_id": LLM_MODEL_ID,
                "model_url": OLLAMA_URL,
                "row_index": row_index,
                "text_length": len(s_color),
                "text_preview": _truncate_for_log(s_color, 100),
            }
        )

    # 关键新增：用原始 raw_color 文本对 abs_map 做"颜色级别"的覆盖修正
    overrides = _direct_abs_overrides_for_row(
        raw_color_text=s_color,
        color_to_pn=color_to_pn,
    )
    if overrides:
        for col_norm, v in overrides.items():
            abs_map[col_norm] = int(v)

    return abs_map, delta_map

# ----------------------------------------------------------------------
# Step 7: 提取モード調度
# ----------------------------------------------------------------------

def _extract_price_parts_shop9_dispatch(
    s_price: str,
    s_color: str,
    color_to_pn: Dict[str, str],
    row_index: object = None,
) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """
    根据 SHOP9_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrail
      - "auto":  regex 优先，regex 无颜色结果时 LLM + Guardrail 兜底

    返回 (abs_map, delta_map, extraction_method)
    """
    mode = SHOP9_EXTRACTION_MODE

    if mode == "regex":
        abs_map, delta_map = _extract_price_parts_shop9_regex(s_price, s_color, color_to_pn)
        # Apply text-based abs overrides (same as original logic)
        overrides = _direct_abs_overrides_for_row(
            raw_color_text=s_color,
            color_to_pn=color_to_pn,
        )
        if overrides:
            for col_norm, v in overrides.items():
                abs_map[col_norm] = int(v)
        return abs_map, delta_map, "regex"

    if mode == "llm":
        abs_map, delta_map = _extract_price_parts_shop9_llm_with_guardrails(
            s_price, s_color, color_to_pn, row_index=row_index,
        )
        return abs_map, delta_map, "llm"

    # ---- auto: regex 优先，regex 无颜色结果时 LLM 兜底 ----
    abs_map_re, delta_map_re = _extract_price_parts_shop9_regex(s_price, s_color, color_to_pn)
    if abs_map_re or delta_map_re:
        # Apply text-based abs overrides
        overrides = _direct_abs_overrides_for_row(
            raw_color_text=s_color,
            color_to_pn=color_to_pn,
        )
        if overrides:
            for col_norm, v in overrides.items():
                abs_map_re[col_norm] = int(v)
        return abs_map_re, delta_map_re, "regex"

    abs_map_llm, delta_map_llm = _extract_price_parts_shop9_llm_with_guardrails(
        s_price, s_color, color_to_pn, row_index=row_index,
    )
    return abs_map_llm, delta_map_llm, "llm"

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop9(
    df: pd.DataFrame,
    debug: bool = True,
    debug_limit: int = 30,
) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop9 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop9"
    SHOP_NAME = "アキモバ"

    logger.info(
        "Starting shop9 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    for need in (COL_MODEL, COL_PRICE, COL_COLOR, COL_TIME):
        if need not in df.columns:
            logger.error(
                f"Missing required column: {need}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "missing_column": need,
                    "available_columns": list(df.columns),
                }
            )
            raise ValueError(f"shop9 清洗器缺少必要列：{need}")

    info_df = _load_iphone17_info_df_from_db()
    pn_map = _build_color_map(info_df)

    model_norm_ser = df[COL_MODEL].map(_normalize_model_generic)
    cap_gb_ser = df[COL_MODEL].map(_parse_capacity_gb)
    recorded_at_ser = df[COL_TIME].map(lambda x: parse_dt_aware(x))

    rows: List[dict] = []

    for i in range(len(df)):
        current_row_records: List[dict] = []
        raw_model = df[COL_MODEL].iat[i]
        m = model_norm_ser.iat[i]
        c = cap_gb_ser.iat[i]
        t = recorded_at_ser.iat[i]
        raw_price_cell = df[COL_PRICE].iat[i]
        raw_color_cell = df[COL_COLOR].iat[i]

        if not m or pd.isna(c):
            logger.debug(
                f"Row {i}: skip (model/cap missing)",
                extra={
                    "event_type": "row_processing_summary",
                    "log_seq": 0,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": i,
                    "raw_model": str(raw_model),
                    "model_norm": str(m),
                    "skip_reason": "model_or_cap_missing",
                }
            )
            continue
        c = int(c)

        key = (m, c)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            logger.debug(
                f"Row {i}: skip (no pn_map for key)",
                extra={
                    "event_type": "row_processing_summary",
                    "log_seq": 0,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": i,
                    "model_norm": str(m),
                    "capacity_gb": c,
                    "skip_reason": "no_pn_map",
                }
            )
            continue

        s_color = str(raw_color_cell) if raw_color_cell is not None else ""
        s_price = str(raw_price_cell) if raw_price_cell is not None else ""

        # base price：优先 price 列，其次 color 列（保留原逻辑）
        base_price = to_int_yen(s_price) or to_int_yen(s_color)

        # 根据 SHOP9_EXTRACTION_MODE 提取价格信息（regex / llm / auto）
        abs_map, delta_map, extraction_method = _extract_price_parts_shop9_dispatch(
            s_price, s_color, color_to_pn, row_index=i,
        )

        # DEBUG: 记录提取结果
        available_colors_list = [
            {"color_norm": cn, "part_number": pn, "color_raw": cr}
            for cn, (pn, cr) in color_to_pn.items()
        ]

        _log_seq += 1
        logger.debug(
            "Extraction result",
            extra={
                "event_type": "extraction_result",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": str(raw_model),
                "model_norm": str(m),
                "capacity_gb": c,
                "base_price": base_price,
                "price_text_raw": _truncate_for_log(s_price, 200),
                "color_text_raw": _truncate_for_log(s_color, 200),
                "color_text_raw_full": s_color,
                "extraction_method": extraction_method,
                "abs_map": {k: v for k, v in abs_map.items()},
                "delta_map": {k: v for k, v in delta_map.items()},
                "abs_count": len(abs_map),
                "delta_count": len(delta_map),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_to_pn),
            }
        )

        # =============== 输出生成逻辑（扩展：支持 abs_map['ALL']） ===============
        output_records = []

        if "ALL" in delta_map:
            if base_price is None:
                logger.debug(
                    f"Row {i}: skip (ALL delta but base missing)",
                    extra={
                        "event_type": "row_processing_summary",
                        "log_seq": 0,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "skip_reason": "all_delta_no_base",
                    }
                )
                continue
            final = int(base_price + delta_map["ALL"])
            for col_norm, (pn, col_raw) in color_to_pn.items():
                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "model_norm": str(m),
                        "capacity_gb": c,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "color_raw": col_raw,
                        "base_price": base_price,
                        "delta": delta_map["ALL"],
                        "final_price": final,
                        "delta_source": "all_delta",
                        "recorded_at": str(t) if t else None,
                    }
                )
                output_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": delta_map["ALL"], "final_price": final, "delta_source": "all_delta",
                })
                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": int(final), "recorded_at": t})
                current_row_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": delta_map["ALL"], "final_price": final,
                    "recorded_at": t, "delta_source": "all_delta",
                })

        elif "ALL" in abs_map:
            final = int(abs_map["ALL"])
            for col_norm, (pn, col_raw) in color_to_pn.items():
                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "model_norm": str(m),
                        "capacity_gb": c,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "color_raw": col_raw,
                        "base_price": base_price,
                        "delta": 0,
                        "final_price": final,
                        "delta_source": "all_abs",
                        "recorded_at": str(t) if t else None,
                    }
                )
                output_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": 0, "final_price": final, "delta_source": "all_abs",
                })
                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": final, "recorded_at": t})
                current_row_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": 0, "final_price": final,
                    "recorded_at": t, "delta_source": "all_abs",
                })

        elif abs_map:
            for col_norm, (pn, col_raw) in color_to_pn.items():
                if col_norm in abs_map:
                    price_new = int(abs_map[col_norm])
                    delta = 0
                    delta_source = "abs_price"
                else:
                    if base_price is None:
                        continue
                    delta = 0
                    price_new = int(base_price)
                    delta_source = "default_base"

                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "model_norm": str(m),
                        "capacity_gb": c,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "color_raw": col_raw,
                        "base_price": base_price,
                        "delta": delta,
                        "final_price": price_new,
                        "delta_source": delta_source,
                        "recorded_at": str(t) if t else None,
                    }
                )
                output_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": delta, "final_price": price_new, "delta_source": delta_source,
                })
                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": int(price_new), "recorded_at": t})
                current_row_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": delta, "final_price": price_new,
                    "recorded_at": t, "delta_source": delta_source,
                })

        else:
            # delta_map only (or empty)
            if base_price is None:
                logger.debug(
                    f"Row {i}: skip (no base/abs)",
                    extra={
                        "event_type": "row_processing_summary",
                        "log_seq": 0,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "skip_reason": "no_base_no_abs",
                    }
                )
                continue

            for col_norm, (pn, col_raw) in color_to_pn.items():
                delta = int(delta_map.get(col_norm, 0))
                price_new = int(base_price + delta)
                delta_source = "matched_label" if col_norm in delta_map else "default_zero"

                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "model_norm": str(m),
                        "capacity_gb": c,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "color_raw": col_raw,
                        "base_price": base_price,
                        "delta": delta,
                        "final_price": price_new,
                        "delta_source": delta_source,
                        "recorded_at": str(t) if t else None,
                    }
                )
                output_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": delta, "final_price": price_new, "delta_source": delta_source,
                })
                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": int(price_new), "recorded_at": t})
                current_row_records.append({
                    "part_number": pn, "color_norm": col_norm,
                    "delta": delta, "final_price": price_new,
                    "recorded_at": t, "delta_source": delta_source,
                })

        # DEBUG: 行级详细汇总
        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": str(raw_model),
                "model_norm": str(m),
                "capacity_gb": c,
                "base_price": base_price,
                "color_text_raw_full": s_color,
                "current_row_records": [
                    {"pn": r["part_number"], "color": r["color_norm"], "delta": r["delta"], "final_price": r["final_price"], "src": r["delta_source"]}
                    for r in current_row_records
                ],
            }
        )

        # INFO: 行级概览（简洁）
        all_deltas_values = list(delta_map.values())
        colors_matched = len(abs_map) + len(delta_map)

        _log_seq += 1
        logger.info(
            f"Row {i:<3d} | {str(raw_model):<28s} | abs: {len(abs_map):<2d} | deltas: {len(delta_map):<2d} | matched: {colors_matched:<2d} | records: {len(output_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": str(raw_model),
                "model_norm": str(m),
                "capacity_gb": c,
                "base_price": base_price,
                "color_text_raw_preview": _truncate_for_log(s_color, 100),
                "extraction_method": extraction_method,
                "abs_count": len(abs_map),
                "delta_count": len(delta_map),
                "colors_in_catalog": len(color_to_pn),
                "colors_matched_count": colors_matched,
                "output_records_count": len(output_records),
                "has_discounted_colors": any(d != 0 for d in all_deltas_values),
                "min_delta": min(all_deltas_values) if all_deltas_values else 0,
                "max_delta": max(all_deltas_values) if all_deltas_values else 0,
            }
        )

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop9 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "output_records": len(out),
            "elapsed_seconds": round(elapsed_time, 2),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    return out
