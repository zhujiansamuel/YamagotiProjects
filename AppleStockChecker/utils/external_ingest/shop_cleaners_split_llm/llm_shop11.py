from __future__ import annotations

"""
llm_shop11 — モバステ LLM 提取模块

从 shop11_cleaner.py 提取的 LLM 相关代码：
- ModelConfig / Ollama 封装
- LLM 机型/容量解析（storage_name -> model_norm, cap_gb）
- LLM 颜色差价提取
"""

import logging
import os
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..cleaner_tools import (
    _normalize_model_generic,
    _label_matches_color_unified,
    _norm_strip,
    coerce_int,
    log_llm_extraction_error,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_norm = _norm_strip
_coerce_int = coerce_int

# ----------------------------------------------------------------------
# ModelConfig / Ollama 封装
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _shop11_model_config():
    """
    LangExtract 新版推荐的 ModelConfig 方式。
    """
    if lx is None:
        return None
    provider_kwargs = {
        "model_url": OLLAMA_URL,
        "temperature": float(os.getenv("SHOP11_OLLAMA_TEMPERATURE", "0.0")),
        "timeout": int(os.getenv("SHOP11_OLLAMA_TIMEOUT", "180")),
        "max_tokens": int(os.getenv("SHOP11_OLLAMA_MAX_TOKENS", "512")),
    }
    try:
        provider_kwargs["format_type"] = lx.data.FormatType.JSON
    except Exception:
        pass
    try:
        return lx.factory.ModelConfig(model_id=OLLAMA_MODEL_ID, provider_kwargs=provider_kwargs)
    except Exception:
        return None


def _lx_extract_ollama(text: str, prompt: str, examples: list):
    """
    返回 result 对象；失败返回 None
    """
    if lx is None:
        return None

    cfg = _shop11_model_config()
    try:
        if cfg is not None:
            return lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                config=cfg,
                fence_output=True,
                use_schema_constraints=False,
            )
    except TypeError:
        pass
    except Exception:
        pass

    # 旧参数路径
    try:
        return lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except Exception:
        return None


# ----------------------------------------------------------------------
# LLM 机型/容量解析
# ----------------------------------------------------------------------

@lru_cache(maxsize=8)
def _shop11_lx_storage_materials(valid_models: Tuple[str, ...]):
    """
    storage_name 解析：device_model + storage_capacity
    """
    model_list = "\n".join(f"- {m}" for m in valid_models if m)

    prompt = textwrap.dedent(f"""\
        You are a strict parser.

        Input format:
          STORAGE: <text>

        Extract up to 2 items:
          1) device_model:
             - extraction_text must be an exact substring from STORAGE (do not invent text).
             - attributes must include: {{"model_norm": "<normalized model>"}}
             - model_norm MUST exactly equal one of:
{model_list}

          2) storage_capacity:
             - extraction_text must be an exact substring from STORAGE containing GB or TB (e.g., "256GB", "1TB").
             - attributes must include: {{"capacity_gb": <int>}}
             - Convert TB to GB using 1TB = 1024GB.

        If you cannot determine a field, do not output that extraction.
    """)

    examples = [
        lx.data.ExampleData(
            text="STORAGE: iPhone17 Pro Max 256GB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="iPhone17 Pro Max",
                    attributes={"model_norm": "iPhone 17 Pro Max"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="256GB",
                    attributes={"capacity_gb": 256},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="STORAGE: 17pro 1TB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="17pro",
                    attributes={"model_norm": "iPhone 17 Pro"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="1TB",
                    attributes={"capacity_gb": 1024},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="STORAGE: iPhone17 プロ 512GB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="iPhone17 プロ",
                    attributes={"model_norm": "iPhone 17 Pro"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="512GB",
                    attributes={"capacity_gb": 512},
                ),
            ],
        ),
    ]
    return prompt, examples


@lru_cache(maxsize=4096)
def lx_parse_storage_shop11(storage: str, valid_models: Tuple[str, ...]) -> Tuple[str, Optional[int], Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]]:
    """
    返回 (model_norm, cap_gb, trace)
    """
    if not storage or lx is None:
        return "", None, tuple()

    prompt, examples = _shop11_lx_storage_materials(valid_models)
    txt = f"STORAGE: {storage}"

    res = _lx_extract_ollama(txt, prompt, examples)
    extrs = getattr(res, "extractions", None) or []

    model_norm = ""
    cap_gb: Optional[int] = None
    trace = []

    for e in extrs:
        cls = str(getattr(e, "extraction_class", "") or "")
        et = str(getattr(e, "extraction_text", "") or "")
        attrs = getattr(e, "attributes", None) or {}
        attrs_items = tuple(sorted((str(k), str(v)) for k, v in attrs.items()))
        trace.append((cls, et, attrs_items))

        if cls == "device_model":
            mn = (attrs.get("model_norm") or "").strip()
            if mn:
                model_norm = _normalize_model_generic(mn) or mn
        elif cls == "storage_capacity":
            cap_gb = _coerce_int(attrs.get("capacity_gb"))

    return model_norm, cap_gb, tuple(trace)


# ----------------------------------------------------------------------
# LLM 颜色差价提取
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shop11_lx_color_materials():
    prompt = textwrap.dedent("""\
        You are a strict parser.

        Input format:
          CAUTION: <text>
          AVAILABLE_COLORS: <c1 | c2 | c3 ...>

        Task:
          Extract color price deltas relative to the base unopened price.

        Output extractions:
          - extraction_class: "color_delta"
          - extraction_text: MUST be EXACTLY one color string from AVAILABLE_COLORS (copy it exactly).
          - attributes: {"delta_yen": <int>}

        Parsing rules:
          - "+2000円" => 2000, "-1,000円" => -1000 (JPY).
          - If CAUTION says "全色" or "すべて" or "全カラー", apply the same delta to ALL AVAILABLE_COLORS.
          - If a color has no delta info, do not output it.
          - If multiple deltas exist for same color, the last one wins.
          - Ignore notes in parentheses like "(未開封)".
    """)

    examples = [
        lx.data.ExampleData(
            text="CAUTION: ブルー、ブラック：-2,000円(未開封)\nAVAILABLE_COLORS: ブルー | ブラック | シルバー",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": -2000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="CAUTION: 全色:+1,000円\nAVAILABLE_COLORS: ブルー | ブラック",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": 1000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"delta_yen": 1000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="CAUTION: シルバー・ブルー：-１０００円\nAVAILABLE_COLORS: ブルー | ブラック | シルバー",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -1000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": -1000},
                ),
            ],
        ),
    ]
    return prompt, examples


@lru_cache(maxsize=4096)
def extract_specs_shop11_llm_core(
    caution: str,
    available_colors: Tuple[str, ...],
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]]:
    """
    返回 (deltas_items, trace)
    """
    if lx is None:
        return tuple(), tuple()

    prompt, examples = _shop11_lx_color_materials()
    avail_line = " | ".join([c for c in available_colors if c])
    txt = f"CAUTION: {caution or ''}\nAVAILABLE_COLORS: {avail_line}"

    res = _lx_extract_ollama(txt, prompt, examples)
    extrs = getattr(res, "extractions", None) or []

    tmp: Dict[str, int] = {}
    trace = []

    for e in extrs:
        cls = str(getattr(e, "extraction_class", "") or "")
        et = str(getattr(e, "extraction_text", "") or "").strip()
        attrs = getattr(e, "attributes", None) or {}
        attrs_items = tuple(sorted((str(k), str(v)) for k, v in attrs.items()))
        trace.append((cls, et, attrs_items))

        if cls != "color_delta":
            continue

        delta = _coerce_int(attrs.get("delta_yen"))
        if delta is None or not et:
            continue

        if et in available_colors:
            tmp[et] = int(delta)
            continue

        for c in available_colors:
            if _label_matches_color_unified(et, c, _norm(c)):
                tmp[c] = int(delta)

    return tuple(tmp.items()), tuple(trace)


def extract_specs_shop11_llm(
    caution_txt: str,
    available_colors: Tuple[str, ...],
    color_map: Dict[str, Tuple[str, str]],
    regex_fn=None,
    cleaner_name: str = "shop11",
    shop_name: str = "モバステ",
) -> Dict[str, int]:
    """
    LLM 提取 + Guardrails。
    """
    color_deltas: Dict[str, int] = {}
    llm_ok = False

    try:
        deltas_items, deltas_trace = extract_specs_shop11_llm_core(caution_txt, available_colors)
        color_deltas = dict(deltas_items)
        llm_ok = True
    except Exception as e:
        llm_ok = False
        log_llm_extraction_error(logger, cleaner_name=cleaner_name, shop_name=shop_name, error=e, text=caution_txt, row_index=None)

    # Guardrail A: 过滤不在 available_colors 中的键
    if color_deltas:
        filtered: Dict[str, int] = {}
        for cn, dv in color_deltas.items():
            if cn in available_colors:
                filtered[cn] = dv
        color_deltas = filtered

    # LLM 完全失败且无结果时，回退到正则
    if (not llm_ok) and (not color_deltas) and caution_txt.strip():
        if regex_fn:
            deltas_fb = regex_fn(caution_txt)
            if deltas_fb:
                for col_norm, (pn, col_raw) in color_map.items():
                    for label_raw, delta in deltas_fb:
                        if _label_matches_color_unified(label_raw, col_raw, col_norm):
                            color_deltas[col_norm] = int(delta)

    return color_deltas
