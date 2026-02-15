from __future__ import annotations

"""
llm_shop4 — モバイルミックス LLM 提取模块

从 shop4_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数
- LLM + Guardrails 封装函数
"""

import logging
import os
import re
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..cleaner_tools import (
    _norm_strip,
    coerce_amount_yen,
    log_llm_extraction_error,
    llm_guardrail_check,
    lx,
    HAS_LANGEXTRACT,
    LABEL_SPLIT_RE_shop4 as LABEL_SPLIT_RE,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_norm = _norm_strip
_coerce_int_maybe = coerce_amount_yen

if HAS_LANGEXTRACT:
    from langextract.data import ExampleData, Extraction
else:
    ExampleData = None
    Extraction = None

# ----------------------------------------------------------------------
# LLM prompt & examples
# ----------------------------------------------------------------------

_SHOP4_LE_PROMPT = textwrap.dedent("""\
You are extracting structured information from a Japanese iPhone pricing table.

Input text contains one or more lines. A relevant line expresses:
- a color label (e.g., シルバー, ディープブルー) OR 全色 (means "all colors"),
- optionally followed by a signed yen adjustment amount.

Rules:
- Extract one item per color label.
- extraction_text MUST be the exact color label substring from the input (do not translate).
- attributes MUST include delta_yen as an integer (negative for discounts).
- Sign can be + or - and may include unicode minus characters: '−' or '－'.
- Amount may include commas and/or full-width digits.
- If a line indicates 全色 but has no amount, set delta_yen = 0.
- If a line does not express a color adjustment, output no extractions.
""").strip()


@lru_cache()
def _get_shop4_le_examples():
    if not HAS_LANGEXTRACT:
        return []

    examples = [
        ExampleData(
            text="シルバー-1,000円",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -1000},
                )
            ],
        ),
        ExampleData(
            text="シルバー/ディープブルー-3,000円",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -3000},
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="ディープブルー",
                    attributes={"delta_yen": -3000},
                ),
            ],
        ),
        ExampleData(
            text="全色-2,000円",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="全色",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
        ExampleData(
            text="全色",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="全色",
                    attributes={"delta_yen": 0},
                ),
            ],
        ),
        ExampleData(
            text="ブルー ＋０円",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": 0},
                ),
            ],
        ),
        ExampleData(
            text="全色－２，０００円",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="全色",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
    ]
    return examples


# ----------------------------------------------------------------------
# LLM 核心提取
# ----------------------------------------------------------------------

def _extract_specs_shop4_llm_core(text: str) -> list:
    """
    对 text 做一次 LangExtract 抽取，返回 result.extractions。
    """
    if not (HAS_LANGEXTRACT and isinstance(text, str) and text.strip()):
        return []

    kwargs = dict(
        text_or_documents=text,
        prompt_description=_SHOP4_LE_PROMPT,
        examples=_get_shop4_le_examples(),
        model_id=OLLAMA_MODEL_ID,
        model_url=OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
        extraction_passes=1,
        max_workers=1,
        max_char_buffer=1500,
        temperature=0.0,
        language_model_params={
            "timeout": 60,
            "keep_alive": 10 * 60,
        },
    )

    try:
        if hasattr(lx, "inference") and hasattr(lx.inference, "OllamaLanguageModel"):
            kwargs["language_model_type"] = lx.inference.OllamaLanguageModel
    except Exception:
        pass

    try:
        result = lx.extract(**kwargs)
    except Exception:
        return []

    exts = getattr(result, "extractions", None)
    return list(exts) if exts else []


def _get_start_pos(extraction) -> int:
    ci = getattr(extraction, "char_interval", None)
    if ci is None:
        return 0
    for attr in ("start_pos", "start", "begin"):
        if hasattr(ci, attr):
            try:
                return int(getattr(ci, attr))
            except Exception:
                pass
    if isinstance(ci, dict):
        for k in ("start_pos", "start", "begin"):
            if k in ci:
                try:
                    return int(ci[k])
                except Exception:
                    pass
    return 0


def _split_labels(label: str) -> List[str]:
    return [p.strip() for p in LABEL_SPLIT_RE.split(label or "") if p and p.strip()]


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop4_llm(
    df,
    start_idx: int,
    row_index: object = None,
    cleaner_name: str = "shop4",
    shop_name: str = "モバイルミックス",
    is_next_model_base_price_row_fn=None,
) -> Tuple[Dict[str, int], List[Tuple[str, int]], Dict[str, str]]:
    """
    用 LangExtract 解析 block 里的颜色±金额，并应用 guardrails。
    返回：(adjustments, delta_specs, color_delta_label_map)
    """
    _empty: Tuple[Dict[str, int], List[Tuple[str, int]], Dict[str, str]] = ({}, [], {})
    lines: List[str] = []
    n = len(df)

    for j in range(start_idx, n):
        if j > start_idx:
            nxt_model = ""
            val = df["data11"].iat[j] if "data11" in df.columns else ""
            nxt_model = str(val) if val is not None else ""
            if nxt_model.strip():
                break
            if is_next_model_base_price_row_fn and is_next_model_base_price_row_fn(df, j, n):
                break
        raw = df["data"].iat[j] if "data" in df.columns else ""
        lines.append("" if raw is None else str(raw))

    if not lines:
        return _empty

    block_text = "\n".join(lines)

    line0_start = 0
    line0_end = len(lines[0]) if lines else 0

    try:
        exts = _extract_specs_shop4_llm_core(block_text)
    except Exception as e:
        log_llm_extraction_error(
            logger, cleaner_name=cleaner_name, shop_name=shop_name,
            error=e, text=block_text, row_index=row_index,
        )
        return _empty

    if not exts:
        return _empty

    exts = sorted(exts, key=_get_start_pos)

    result: Dict[str, int] = {}
    delta_specs: List[Tuple[str, int]] = []
    color_delta_label_map: Dict[str, str] = {}
    global_all_delta: Optional[int] = None

    for ex in exts:
        cls = str(getattr(ex, "extraction_class", "") or "").strip()
        if cls and cls.lower() not in {"color_delta", "colordelta", "color"}:
            continue

        label = str(getattr(ex, "extraction_text", "") or "").strip()
        if not label:
            continue

        attrs = getattr(ex, "attributes", None)
        attrs = attrs if isinstance(attrs, dict) else {}
        delta = _coerce_int_maybe(attrs.get("delta_yen"))
        if delta is None:
            if "全色" in label and not re.search(r"[0-9０-９]", block_text):
                delta = 0
            else:
                continue

        # Guardrail A & B
        if not llm_guardrail_check(label, delta, block_text):
            continue

        start_pos = _get_start_pos(ex)

        if "全色" in label and line0_start <= start_pos < max(line0_end, line0_start):
            global_all_delta = int(delta)

        for lbl in _split_labels(label):
            delta_specs.append((lbl, int(delta)))
            if "全色" in lbl:
                result["ALL"] = int(delta)
            else:
                nk = _norm(lbl)
                result[nk] = int(delta)
                color_delta_label_map[nk] = lbl

    if global_all_delta is not None:
        result["ALL"] = int(global_all_delta)

    return result, delta_specs, color_delta_label_map
