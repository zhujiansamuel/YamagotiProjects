from __future__ import annotations

"""
shop16 清洗器 — 携帯空間

  原始文本（買取価格列）
    │
    ├─ _normalize_price_text_shop16()     ← Step 1: 归一化（换行→/、压缩空白）
    │
    ├─ _extract_base_price_shop16()       ← Step 2: 提取基础价
    │
    ├─ _extract_price_parts_shop16_dispatch()  ← Step 9: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   ├─ _extract_color_deltas_shop16()      ← Step 6a: 正则提取差价
    │   │   └─ _extract_color_abs_prices_shop16()   ← Step 6b: 正则提取绝对价
    │   │
    │   └─ llm 路径:
    │       ├─ _lx_extract_price_parts_shop16()     ← Step 7: LLM 核心提取
    │       └─ Guardrail A/B/C                      ← Step 8: 防幻觉过滤
    │
    ├─ _label_matches_color_unified()     ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop16()                     ← Step 10: 主函数，生成输出行
"""

import logging
import os
import re
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
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
)

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "gemma3:1b")

SHOP16_EXTRACTION_MODE = "regex"  # "regex" | "llm" | "auto"

MODEL_COL = "iPhone 17 Pro Max"
DESC_COL  = "説明1"
PRICE_COL = "買取価格"

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

# ----------------------------------------------------------------------
# Step 1: 价格文本归一化
# ----------------------------------------------------------------------

def _normalize_price_text_shop16(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    # 把换行变成分隔（保留"下一行是颜色差价"的结构）
    s = re.sub(r"[\r\n]+", " / ", s)
    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    # 多个分隔合并
    s = re.sub(r"(?:\s*/\s*){2,}", " / ", s).strip()
    return s

# ----------------------------------------------------------------------
# Step 2: 基础价提取 & 判定
# ----------------------------------------------------------------------

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

def _extract_base_price_shop16(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)  # 兜底
    return to_int_yen(m.group(1))

_BASE_ONLY_RE = re.compile(r"^\s*(?:￥|\¥)?\s*\d[\d,]*\s*(?:円)?\s*$")

def _is_base_only_price_text(price_text_norm: str) -> bool:
    """判断文本是否只包含一个基础价，不含任何颜色差价信息。"""
    return bool(_BASE_ONLY_RE.match(price_text_norm or ""))

# ----------------------------------------------------------------------
# Step 3: 标签归一化 & 拆分
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）

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
    # 兼容 "青/オレンジ""黒、白""blue/black" 等
    raw = _normalize_label_shop16(lbl)
    parts = re.split(r"[／/、，,]", raw)
    return [p for p in (_normalize_label_shop16(x) for x in parts) if p]

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop16 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则模式定义
# ----------------------------------------------------------------------

COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\s]+)\s*
        (?P<sep>[：:\-])\s*           # 捕获分隔符
        (?P<sign>[+\-−－])?\s*        # 显式正负号（可选）
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*￥\s*(?P<amount>\d[\d,]*)""",
    re.UNICODE
)

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s*;\s*)")

_GROUP_SHARED_DELTA_RE = re.compile(
    r"""
    (?P<labels>[^0-9￥円]+?)          # 多颜色标签段（含 /）
    \s*(?P<sign>[+\-−－])\s*         # 显式正负号
    (?P<amount>\d[\d,]*)\s*(?:円)?   # 金额（可无 円）
    """,
    re.UNICODE | re.VERBOSE
)

# ----------------------------------------------------------------------
# Step 6: 正则提取函数
# ----------------------------------------------------------------------

def _extract_color_deltas_shop16(text: str) -> List[Tuple[str, int]]:
    """从价格串中抽取多段"颜色±金额"，支持 '青/オレンジ -5000' 这类多标签共用金额。"""
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text)
    # 去掉第一个"基础价 N円/￥N"
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
        # 该片段是否包含"颜色±金额"
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

        # 否则，这是"只有标签没有金额"的片段（如 '青'）；缓存它，等待后面的金额
        # 如果是 '青/橙' 没被上层 split 掉，也进一步按斜杠切一下
        for tok in re.split(r"[／/]", part):
            tok = _normalize_label(tok)
            if tok:
                pending_labels.append(tok)

    return out

def _extract_color_abs_prices_shop16(text: str) -> List[Tuple[str, int]]:
    """从价格串中抽取"颜色￥绝对价"，如：'黒￥86100/青￥87100'"""
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    for m in COLOR_ABS_RE.finditer(str(text)):
        label = (m.group("label") or "").strip()
        amt = to_int_yen(m.group("amount"))
        if label and amt is not None:
            out.append((label, int(amt)))
    return out

def _extract_shared_delta_map_shop16(price_text_norm: str) -> Dict[str, int]:
    """
    从原文中抽取： 'オレンジ/青 -1500' 这种共享差价 -> {オレンジ:-1500, 青:-1500}
    这是"纠错用"的确定性证据，不替代 LLM 抽取的主流程。
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

def _extract_price_parts_shop16_regex(
    price_text: str,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    纯正则版：从 price_text 中提取 (base_price, deltas, abs_prices)。
    """
    base_price = _extract_base_price_shop16(price_text)
    deltas = _extract_color_deltas_shop16(price_text)
    absps = _extract_color_abs_prices_shop16(price_text)
    return base_price, deltas, absps

# ----------------------------------------------------------------------
# Step 7: LLM 配置 & 核心提取函数
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Step 8: LLM + Guardrails（仅 LLM 路径使用）
# ----------------------------------------------------------------------

def _extract_price_parts_shop16_llm_with_guardrails(
    price_text: str, idx: object = None,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    LLM 提取 + Guardrail A/B/C（仅 LLM 路径使用）。
    """
    base_llm = None
    deltas: List[Tuple[str, int]] = []
    absps: List[Tuple[str, int]] = []
    llm_ok = False

    try:
        base_llm, deltas, absps, _dbg = _lx_extract_price_parts_shop16(price_text)
        llm_ok = True
    except Exception as e:
        llm_ok = False
        logger.warning(
            "LangExtract extraction failed",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": "携帯空間",
                "cleaner_name": "shop16",
                "error": str(e),
                "error_type": type(e).__name__,
                "model_id": OLLAMA_MODEL_ID,
                "model_url": OLLAMA_URL,
                "row_index": idx,
                "text_length": len(price_text),
                "text_preview": _truncate_for_log(price_text, 100),
            }
        )

    base_price = base_llm
    if base_price is None:
        base_price = _extract_base_price_shop16(price_text)
    base_price = int(base_price) if base_price is not None else None

    # Guardrail A: 只有基础价 -> 丢弃所有 color_delta/color_abs（防幻觉）
    if _is_base_only_price_text(price_text):
        deltas = []
        absps = []

    # Guardrail B: 共享差价纠错
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

    # Guardrail C: 逐条证据过滤 —— label/金额必须在原文出现
    text_no_commas = price_text.replace(",", "")
    filtered_deltas: List[Tuple[str, int]] = []
    for label_raw, delta in deltas:
        lb = _normalize_label_shop16(label_raw)
        if not lb:
            continue
        if lb not in price_text:
            continue
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

    # LLM 完全失败且无颜色信息时，回退到正则
    if (not llm_ok) and (not deltas) and (not absps):
        deltas = _extract_color_deltas_shop16(price_text)
        absps = _extract_color_abs_prices_shop16(price_text)

    return base_price, deltas, absps

# ----------------------------------------------------------------------
# Step 9: 提取模式调度
# ----------------------------------------------------------------------

def _extract_price_parts_shop16_dispatch(
    price_text: str, idx: object = None,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]], str]:
    """
    根据 SHOP16_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrail A/B/C
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrail 兜底

    返回 (base_price, deltas, abs_prices, extraction_method)
    """
    mode = SHOP16_EXTRACTION_MODE

    if mode == "regex":
        bp, deltas, absps = _extract_price_parts_shop16_regex(price_text)
        return bp, deltas, absps, "regex"

    if mode == "llm":
        bp, deltas, absps = _extract_price_parts_shop16_llm_with_guardrails(
            price_text, idx=idx,
        )
        return bp, deltas, absps, "llm"

    # ---- auto: 正则优先，正则无颜色结果时 LLM 兜底 ----
    bp_re, deltas_re, absps_re = _extract_price_parts_shop16_regex(price_text)
    if deltas_re or absps_re:
        return bp_re, deltas_re, absps_re, "regex"

    bp_llm, deltas_llm, absps_llm = _extract_price_parts_shop16_llm_with_guardrails(
        price_text, idx=idx,
    )
    # LLM 的 base_price 优先，其次正则的
    bp_final = bp_llm if bp_llm is not None else bp_re
    return bp_final, deltas_llm, absps_llm, "llm"

# ----------------------------------------------------------------------
# Step 10: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop16(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop16 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop16"
    SHOP_NAME = "携帯空間"

    logger.info(
        "Starting shop16 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    for c in [MODEL_COL, DESC_COL, PRICE_COL, "time-scraped"]:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "missing_column": c,
                    "available_columns": list(df.columns),
                }
            )
            raise ValueError(f"shop16 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

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

        # 根据 SHOP16_EXTRACTION_MODE 提取价格信息（regex / llm / auto）
        base_price, deltas, absps, extraction_method = _extract_price_parts_shop16_dispatch(
            price_text, idx=idx,
        )

        # 没 base 且没 abs：没法落库
        if base_price is None and not absps:
            continue

        # base_price=None 时不能使用 delta（base+delta 无法计算），且不能输出 default 行
        if base_price is None:
            deltas = []
            emit_default = False
        else:
            emit_default = True

        # 构建 PriceDecomposition
        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=absps,
            extraction_method=extraction_method,
            source_text_raw=price_text,
        )

        # 使用公共函数生成输出行（含完整日志）
        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=emit_default,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop16 cleaner completed",
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
