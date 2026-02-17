from __future__ import annotations

"""
llm_shop15 — 買取当番 LLM 提取模块

从 shop15_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM 纠错/增强函数
- LLM + Guardrails 封装函数
"""

import logging
import os
import re
from functools import lru_cache
from typing import List, Optional, Tuple

from ..cleaner_tools import (
    log_llm_extraction_error,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# LLM prompt
# ----------------------------------------------------------------------

SHOP15_PRICE_PROMPT = (
    "You parse Japanese iPhone buyback 'price' strings.\n"
    "Extract:\n"
    "1) base price (the first yen price at the beginning of the string).\n"
    "   Return ONE extraction:\n"
    "   - extraction_class = \"base_price\"\n"
    "   - extraction_text = exact substring including 円 (e.g., \"230,500円\")\n"
    "   - attributes = {\"yen\": \"230500\"}\n"
    "\n"
    "2) color-specific rules. For each color label, return ONE extraction:\n"
    "   - extraction_class = \"color_spec\"\n"
    "   - extraction_text = exact color label substring (e.g., \"ブルー\")\n"
    "   - attributes must include:\n"
    "       kind: \"delta\" or \"abs\"\n"
    "       yen: integer yen string. For delta use signed string like \"-1000\" or \"2000\". For abs use \"229000\".\n"
    "\n"
    "Rules:\n"
    "- If a color label is followed by +/− amount (e.g., ブルー-1000円, シルバー+2,000円) => kind=\"delta\".\n"
    "- If a color label is followed by a price WITHOUT +/− (e.g., ブルー229,000円, ブルー:229,000円, シルバー 229,000円) => kind=\"abs\".\n"
    "- If multiple color labels are listed before the same amount using separators (、/／・,&), apply the same rule to each label.\n"
    "- Return entities in order of appearance. Use exact text; do not paraphrase.\n"
)


# ----------------------------------------------------------------------
# LLM examples
# ----------------------------------------------------------------------

def _shop15_langextract_examples():
    return [
        lx.data.ExampleData(
            text="207,000円　オレンジ、ブルー-1000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="207,000円",
                    attributes={"yen": "207000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="オレンジ",
                    attributes={"kind": "delta", "yen": "-1000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "delta", "yen": "-1000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="230,500円　ブルー229,000円　シルバー　229,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="230,500円",
                    attributes={"yen": "230500"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "abs", "yen": "229000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "abs", "yen": "229000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="213,500円　ブルー-9,000円　シルバー-7,500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="213,500円",
                    attributes={"yen": "213500"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "delta", "yen": "-9000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "delta", "yen": "-7500"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="180,000円 シルバー+2,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="180,000円",
                    attributes={"yen": "180000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "delta", "yen": "2000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="263,000円　ブルー-3,000円　シルバー　-3,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="263,000円",
                    attributes={"yen": "263000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "delta", "yen": "-3000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "delta", "yen": "-3000"},
                ),
            ],
        ),
    ]


# ----------------------------------------------------------------------
# LLM 结果排序辅助
# ----------------------------------------------------------------------

def _iter_extractions_in_order(result) -> List:
    exts = list(getattr(result, "extractions", []) or [])

    def key(e):
        ci = getattr(e, "char_interval", None)
        sp = getattr(ci, "start_pos", None) if ci is not None else None
        if sp is not None:
            return (0, int(sp))
        idx = getattr(e, "extraction_index", None)
        if idx is not None:
            return (1, int(idx))
        return (2, 0)

    return sorted(exts, key=key)


# ----------------------------------------------------------------------
# LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

@lru_cache(maxsize=4096)
def extract_specs_shop15_llm_core(
    price_text: str,
    model_id: str,
    model_url: str,
    _parse_signed_int_yen_fn=None,
    _clean_label_fn=None,
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    """
    返回:
      base_price: Optional[int]
      specs: List[(label, kind, yen_value)]
    """
    if not HAS_LANGEXTRACT:
        return None, []

    examples = _shop15_langextract_examples()

    try:
        result = lx.extract(
            text_or_documents=price_text,
            prompt_description=SHOP15_PRICE_PROMPT,
            examples=examples,
            model_id=model_id,
            model_url=model_url,
            fence_output=False,
            use_schema_constraints=False,
            temperature=0.0,
        )
    except TypeError:
        result = lx.extract(
            text_or_documents=price_text,
            prompt_description=SHOP15_PRICE_PROMPT,
            examples=examples,
            model_id=model_id,
            model_url=model_url,
            fence_output=False,
            use_schema_constraints=False,
        )

    base_price = None
    specs: List[Tuple[str, str, int]] = []

    for ext in _iter_extractions_in_order(result):
        cls = (getattr(ext, "extraction_class", "") or "").strip().lower()
        txt = (getattr(ext, "extraction_text", "") or "").strip()
        attrs = getattr(ext, "attributes", {}) or {}

        if cls == "base_price":
            yen = attrs.get("yen")
            v = _parse_signed_int_yen_fn(yen if yen is not None else txt) if _parse_signed_int_yen_fn else None
            if v is not None:
                base_price = int(v)
            continue

        if cls == "color_spec":
            label = _clean_label_fn(txt) if _clean_label_fn else txt.strip()
            if not label:
                continue
            kind = str(attrs.get("kind", "")).strip().lower()
            yen_raw = attrs.get("yen")
            v = _parse_signed_int_yen_fn(yen_raw) if _parse_signed_int_yen_fn else None
            if v is None and _parse_signed_int_yen_fn:
                v = _parse_signed_int_yen_fn(txt)

            if v is None:
                continue

            if kind not in {"delta", "abs"}:
                ys = str(yen_raw) if yen_raw is not None else ""
                ys = ys.replace("＋", "+").replace("−", "-").replace("－", "-")
                kind = "delta" if ("-" in ys or "+" in ys) else "abs"

            specs.append((label, kind, int(v)))
            continue

    return base_price, specs


# ----------------------------------------------------------------------
# LLM 纠错 & 增强
# ----------------------------------------------------------------------

def _extract_signed_amount_after_label_shop15(
    price_text: str,
    label: str,
    _clean_label_fn=None,
) -> Optional[int]:
    if not price_text or not label:
        return None
    s = str(price_text).replace("\u3000", " ")
    lab = _clean_label_fn(label) if _clean_label_fn else label.strip()
    if not lab:
        return None

    idx = s.find(lab)
    if idx < 0:
        return None
    window = s[idx: idx + 40]

    m = re.match(
        re.escape(lab) + r"\s*(?:[：:])?\s*([+\-−－])\s*(\d[\d,]*)",
        window
    )
    if not m:
        return None

    sign = m.group(1)
    amt = int(m.group(2).replace(",", ""))
    if sign in ("-", "−", "－"):
        amt = -amt
    return amt


def coerce_specs_shop15(
    price_text: str, base_price: Optional[int],
    specs: List[Tuple[str, str, int]],
    _clean_label_fn=None,
) -> List[Tuple[str, str, int]]:
    fixed: List[Tuple[str, str, int]] = []
    for (label, kind, value) in specs:
        kind2, value2 = kind, value

        if kind2 == "abs" and value2 < 0:
            kind2 = "delta"

        signed_ctx = _extract_signed_amount_after_label_shop15(price_text, label, _clean_label_fn)
        if signed_ctx is not None:
            kind2 = "delta"
            value2 = int(signed_ctx)

        fixed.append((label, kind2, value2))
    return fixed


def augment_multi_label_block_specs_shop15(
    price_text: str,
    specs: List[Tuple[str, str, int]],
    multi_label_delta_block_re=None,
    split_color_labels_fn=None,
    clean_label_fn=None,
) -> List[Tuple[str, str, int]]:
    if not price_text:
        return specs

    s = str(price_text)
    new_specs: List[Tuple[str, str, int]] = list(specs)

    if multi_label_delta_block_re is None:
        return new_specs

    for m in multi_label_delta_block_re.finditer(s):
        label_blob = m.group("label_blob") or ""
        sign = m.group("sign")
        amount_str = m.group("amount")

        try:
            amt = int(amount_str.replace(",", ""))
        except Exception:
            continue

        value = -amt if sign in ("-", "−", "－") else amt

        labels = split_color_labels_fn(label_blob) if split_color_labels_fn else [label_blob]

        for lab in labels:
            lab_clean = clean_label_fn(lab) if clean_label_fn else lab.strip()
            if not lab_clean:
                continue

            found = False
            for idx, (lbl_old, kind_old, val_old) in enumerate(new_specs):
                if lbl_old == lab_clean:
                    new_specs[idx] = (lab_clean, "delta", int(value))
                    found = True
                    break

            if not found:
                new_specs.append((lab_clean, "delta", int(value)))

    return new_specs


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop15_llm(
    price_text: str,
    idx: object = None,
    regex_fn=None,
    extract_base_price_fn=None,
    parse_signed_int_yen_fn=None,
    clean_label_fn=None,
    multi_label_delta_block_re=None,
    split_color_labels_fn=None,
    cleaner_name: str = "shop15",
    shop_name: str = "買取当番",
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    """
    LLM 提取 + 纠错（coerce + augment）。仅 LLM 路径使用。
    """
    base_price, specs = None, []
    llm_ok = False

    try:
        base_price, specs = extract_specs_shop15_llm_core(
            str(price_text),
            OLLAMA_MODEL_ID,
            OLLAMA_URL,
            _parse_signed_int_yen_fn=parse_signed_int_yen_fn,
            _clean_label_fn=clean_label_fn,
        )
        llm_ok = True
    except Exception as e:
        log_llm_extraction_error(logger, cleaner_name=cleaner_name, shop_name=shop_name, error=e, text=price_text, row_index=idx)

    if base_price is None and extract_base_price_fn:
        base_price = extract_base_price_fn(price_text)

    specs = coerce_specs_shop15(price_text, base_price, specs, _clean_label_fn=clean_label_fn)

    specs = augment_multi_label_block_specs_shop15(
        price_text, specs,
        multi_label_delta_block_re=multi_label_delta_block_re,
        split_color_labels_fn=split_color_labels_fn,
        clean_label_fn=clean_label_fn,
    )

    if (not llm_ok) and (not specs) and regex_fn:
        raw = regex_fn(price_text)
        if len(raw) >= 3:
            base_regex, deltas, abs_specs = raw[0], raw[1], raw[2]
            if base_price is None and base_regex is not None:
                base_price = base_regex
            specs = [(l, "delta", v) for l, v in (deltas or [])] + [(l, "abs", v) for l, v in (abs_specs or [])]
        elif len(raw) >= 2:
            if base_price is None and raw[0] is not None:
                base_price = raw[0]
            specs = raw[1]  # 兼容旧格式 (base, specs)

    return base_price, specs
