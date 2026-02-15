from __future__ import annotations

"""
llm_shop9 — アキモバ LLM 提取模块

从 shop9_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM + Guardrails 封装函数
"""

import json
import logging
import os
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..cleaner_tools import (
    coerce_signed_int,
    log_llm_extraction_error,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_coerce_signed_int = coerce_signed_int

LLM_TEMPERATURE = float(os.getenv("SHOP9_LLM_TEMPERATURE", "0.0") or "0.0")

# ----------------------------------------------------------------------
# LLM prompt & examples
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


# ----------------------------------------------------------------------
# LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

@lru_cache(maxsize=4096)
def extract_specs_shop9_llm_core(
    price_text: str,
    detail_text: str,
    avail_colors_key: Tuple[str, ...],
    _build_color_aliases_fn=None,
    _map_to_available_color_fn=None,
    _bucket_amount_fn=None,
    _norm_cls_fn=None,
) -> Tuple[Dict[str, int], Dict[str, int],
           List[Tuple[str, int]], List[Tuple[str, int]],
           Dict[str, str], Dict[str, str]]:
    """
    返回:
      abs_map, delta_map, abs_specs, delta_specs,
      color_abs_label_map, color_delta_label_map
    """
    _empty = ({}, {}, [], [], {}, {})
    if not HAS_LANGEXTRACT:
        return _empty

    available_colors = list(avail_colors_key)
    aliases = _build_color_aliases_fn(available_colors) if _build_color_aliases_fn else {}

    input_text = f"買取価格: {price_text}\n色・詳細等: {detail_text}"

    prompt = SHOP9_PRICE_PROMPT_TEMPLATE.format(
        available_colors=json.dumps(available_colors, ensure_ascii=False),
        aliases=json.dumps(aliases, ensure_ascii=False),
    )

    kw = dict(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=_shop9_lx_examples(),
        model_id=OLLAMA_MODEL_ID,
        model_url=OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
    )

    try:
        result = lx.extract(**kw, temperature=LLM_TEMPERATURE)
    except TypeError:
        result = lx.extract(**kw)
    except Exception:
        return _empty

    abs_map: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}
    abs_specs: List[Tuple[str, int]] = []
    delta_specs: List[Tuple[str, int]] = []
    color_abs_label_map: Dict[str, str] = {}
    color_delta_label_map: Dict[str, str] = {}

    extractions = getattr(result, "extractions", None) or []
    avail_set = set(available_colors)

    for ex in extractions:
        cls_raw = str(getattr(ex, "extraction_class", "") or "")
        cls_norm = _norm_cls_fn(cls_raw) if _norm_cls_fn else cls_raw.strip().lower().replace("-", "_").replace(" ", "_")
        attrs = getattr(ex, "attributes", None) or {}
        ex_text = str(getattr(ex, "extraction_text", "") or "")

        amt = attrs.get("amount_yen")
        amt_i = _coerce_signed_int(amt)
        if amt_i is None:
            amt_i = _coerce_signed_int(ex_text)
        if amt_i is None:
            continue

        colors = attrs.get("colors") or attrs.get("color") or []
        if isinstance(colors, str):
            colors = [colors]
        if not isinstance(colors, list):
            colors = list(colors) if colors else []

        bucket = _bucket_amount_fn(cls_norm, ex_text, int(amt_i)) if _bucket_amount_fn else "delta"

        for c_raw in colors:
            mapped = _map_to_available_color_fn(str(c_raw), avail_set) if _map_to_available_color_fn else None
            if not mapped:
                continue
            if bucket == "abs":
                abs_map[mapped] = int(amt_i)
                abs_specs.append((str(c_raw), int(amt_i)))
                if mapped != "ALL":
                    color_abs_label_map[mapped] = str(c_raw)
            else:
                delta_map[mapped] = int(amt_i)
                delta_specs.append((str(c_raw), int(amt_i)))
                if mapped != "ALL":
                    color_delta_label_map[mapped] = str(c_raw)

    return abs_map, delta_map, abs_specs, delta_specs, color_abs_label_map, color_delta_label_map


# ----------------------------------------------------------------------
# 模块级依赖注入
# ----------------------------------------------------------------------

_build_color_aliases_ref = None
_map_to_available_color_ref = None
_bucket_amount_ref = None
_norm_cls_ref = None
_direct_abs_overrides_ref = None


def setup_shop9_llm_deps(
    build_color_aliases_fn,
    map_to_available_color_fn,
    bucket_amount_fn,
    norm_cls_fn,
    direct_abs_overrides_fn,
):
    """由 shop9_cleaner 调用，注入非 LLM 依赖。"""
    global _build_color_aliases_ref, _map_to_available_color_ref
    global _bucket_amount_ref, _norm_cls_ref, _direct_abs_overrides_ref
    _build_color_aliases_ref = build_color_aliases_fn
    _map_to_available_color_ref = map_to_available_color_fn
    _bucket_amount_ref = bucket_amount_fn
    _norm_cls_ref = norm_cls_fn
    _direct_abs_overrides_ref = direct_abs_overrides_fn


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop9_llm(
    s_price: str,
    s_color: str,
    color_to_pn: Dict[str, str],
    row_index: object = None,
    cleaner_name: str = "shop9",
    shop_name: str = "アキモバ",
) -> Tuple[Dict[str, int], Dict[str, int],
           List[Tuple[str, int]], List[Tuple[str, int]],
           Dict[str, str], Dict[str, str]]:
    """
    LLM 提取 + Guardrail (_bucket_amount)。
    """
    avail_colors_key = tuple(color_to_pn.keys())
    abs_map: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}
    abs_specs: List[Tuple[str, int]] = []
    delta_specs: List[Tuple[str, int]] = []
    color_abs_label_map: Dict[str, str] = {}
    color_delta_label_map: Dict[str, str] = {}

    try:
        (abs_map, delta_map,
         abs_specs, delta_specs,
         color_abs_label_map, color_delta_label_map) = extract_specs_shop9_llm_core(
            s_price, s_color, avail_colors_key,
            _build_color_aliases_fn=_build_color_aliases_ref,
            _map_to_available_color_fn=_map_to_available_color_ref,
            _bucket_amount_fn=_bucket_amount_ref,
            _norm_cls_fn=_norm_cls_ref,
        )
    except Exception as e:
        log_llm_extraction_error(
            logger, cleaner_name=cleaner_name, shop_name=shop_name,
            error=e, text=s_color, row_index=row_index,
        )

    # abs overrides
    if _direct_abs_overrides_ref:
        overrides = _direct_abs_overrides_ref(
            raw_color_text=s_color,
            color_to_pn=color_to_pn,
        )
        if overrides:
            for col_norm, v in overrides.items():
                abs_map[col_norm] = int(v)
                color_abs_label_map[col_norm] = col_norm
                abs_specs.append((col_norm, int(v)))

    return abs_map, delta_map, abs_specs, delta_specs, color_abs_label_map, color_delta_label_map
