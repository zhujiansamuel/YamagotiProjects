from __future__ import annotations

"""
llm_shop16 — 携帯空間 LLM 提取模块

从 shop16_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM + Guardrails 封装函数
"""

import logging
import os
import re
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ...external_ingest.cleaner_tools import to_int_yen
from ..cleaner_tools import (
    log_llm_extraction_error,
    llm_guardrail_check,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# LLM prompt
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


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _to_signed_int_yen(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    signed = list(re.finditer(r"([+\-−－])\s*(\d[\d,]*)", s))
    if signed:
        m = signed[-1]
        sign = m.group(1)
        amt = to_int_yen(m.group(2))
        if amt is None:
            return None
        return -int(amt) if sign in ("-", "−", "－") else int(amt)

    nums = list(re.finditer(r"(\d[\d,]*)", s))
    if not nums:
        return None
    amt = to_int_yen(nums[-1].group(1))
    return int(amt) if amt is not None else None


# ----------------------------------------------------------------------
# LLM examples
# ----------------------------------------------------------------------

def _shop16_price_examples():
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


# ----------------------------------------------------------------------
# LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

@lru_cache(maxsize=4096)
def extract_specs_shop16_llm_core(
    price_text: str,
    _split_labels_fn=None,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]], List[dict]]:
    """
    返回：base_price, deltas, abs_prices, debug_extractions
    """
    s = (price_text or "").strip()
    if not s:
        return None, [], [], []

    if not HAS_LANGEXTRACT:
        raise ImportError("缺少依赖：pip install langextract")

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
    try:
        kwargs["language_model_type"] = lx.inference.OllamaLanguageModel
    except Exception:
        pass

    result = lx.extract(**kwargs)

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
                if delta is None:
                    delta = _to_signed_int_yen(txt)

                if delta is not None:
                    labels = _split_labels_fn(raw_label) if _split_labels_fn else [raw_label]
                    for lb in labels:
                        deltas.append((lb, int(delta)))

            elif cls == "color_abs":
                raw_label = str(attrs.get("color_label") or txt or "").strip()
                av = attrs.get("amount_yen")
                amt = to_int_yen(av) if av is not None else to_int_yen(txt)

                if amt is not None:
                    labels = _split_labels_fn(raw_label) if _split_labels_fn else [raw_label]
                    for lb in labels:
                        abs_prices.append((lb, int(amt)))

    return base_price, deltas, abs_prices, debug_extractions


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop16_llm(
    price_text: str,
    idx: object = None,
    extract_base_price_fn=None,
    is_base_only_fn=None,
    extract_shared_delta_map_fn=None,
    normalize_label_fn=None,
    split_labels_fn=None,
    regex_deltas_fn=None,
    regex_abs_fn=None,
    cleaner_name: str = "shop16",
    shop_name: str = "携帯空間",
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    LLM 提取 + Guardrail A/B/C。
    """
    base_llm = None
    deltas: List[Tuple[str, int]] = []
    absps: List[Tuple[str, int]] = []
    llm_ok = False

    try:
        base_llm, deltas, absps, _dbg = extract_specs_shop16_llm_core(price_text, _split_labels_fn=split_labels_fn)
        llm_ok = True
    except Exception as e:
        llm_ok = False
        log_llm_extraction_error(logger, cleaner_name=cleaner_name, shop_name=shop_name, error=e, text=price_text, row_index=idx)

    base_price = base_llm
    if base_price is None and extract_base_price_fn:
        base_price = extract_base_price_fn(price_text)
    base_price = int(base_price) if base_price is not None else None

    # Guardrail A: 只有基础价 -> 丢弃所有 color_delta/color_abs
    if is_base_only_fn and is_base_only_fn(price_text):
        deltas = []
        absps = []

    # Guardrail B: 共享差价纠错
    if extract_shared_delta_map_fn:
        shared_delta_map = extract_shared_delta_map_fn(price_text)
        if shared_delta_map and deltas:
            corrected: List[Tuple[str, int]] = []
            for label_raw, delta in deltas:
                lb = normalize_label_fn(label_raw) if normalize_label_fn else label_raw
                if not lb:
                    continue
                if lb in shared_delta_map:
                    corrected.append((lb, int(shared_delta_map[lb])))
                else:
                    corrected.append((lb, int(delta)))
            deltas = corrected

    # Guardrail C: 逐条证据过滤
    if normalize_label_fn:
        filtered_deltas: List[Tuple[str, int]] = []
        for label_raw, delta in deltas:
            lb = normalize_label_fn(label_raw)
            if not lb:
                continue
            if llm_guardrail_check(lb, delta, price_text):
                filtered_deltas.append((lb, int(delta)))
        deltas = filtered_deltas

        filtered_absps: List[Tuple[str, int]] = []
        for label_raw, amt in absps:
            lb = normalize_label_fn(label_raw)
            if not lb:
                continue
            if llm_guardrail_check(lb, amt, price_text):
                filtered_absps.append((lb, int(amt)))
        absps = filtered_absps

    # LLM 完全失败且无颜色信息时，回退到正则
    if (not llm_ok) and (not deltas) and (not absps):
        if regex_deltas_fn:
            deltas = regex_deltas_fn(price_text)
        if regex_abs_fn:
            absps = regex_abs_fn(price_text)

    return base_price, deltas, absps
