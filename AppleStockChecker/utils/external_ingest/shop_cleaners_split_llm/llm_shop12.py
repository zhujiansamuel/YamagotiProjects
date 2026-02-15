from __future__ import annotations

"""
llm_shop12 — トゥインクル LLM 提取模块

从 shop12_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM + Guardrails 封装函数
"""

import logging
import os
import re
import textwrap
from functools import lru_cache
from typing import List, Optional, Tuple

from ..cleaner_tools import (
    _truncate_for_log,
    _normalize_amount_text,
    log_llm_extraction_error,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_norm_amount_to_int = _normalize_amount_text

# ----------------------------------------------------------------------
# LLM prompt & examples
# ----------------------------------------------------------------------

_LX_PROMPT = textwrap.dedent(r"""
你要从输入文本（備考1）中抽取"颜色对应的价格规则"。只抽取以下两类：

1) delta（差额）
- 形式：<颜色标签><+或-><金额>円
- 例：orange-1000円  => delta_yen=-1000, color_label="orange"
- 例：Blue+2000円    => delta_yen=+2000, color_label="Blue"
- 例：全色-2000円     => delta_yen=-2000, color_label="全色"

2) abs_price（绝对价）
- 形式：<颜色标签> ¥<金額> 或 <颜色标签> <金額>円
- 例：Silver ¥230,500 => price_yen=230500, color_label="Silver"

规则：
- extraction_text 必须是输入里的"原文片段"，不要改写/不要翻译。
- 如果一行里有多种颜色分别给价或给差额，要分别输出多条 extraction。
- 如果文本里出现"開封/開封品/※開封/開封済"，这些内容不参与抽取（可以忽略）。
- 就算文本非常短（例如仅有 'orange-1000円'），只要存在规则也必须抽取出来。
""").strip()


def _lx_examples():
    return [
        lx.data.ExampleData(
            text="orange-1000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="orange-1000円",
                    attributes={"color_label": "orange", "delta_yen": -1000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Orange-2000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="Orange-2000円",
                    attributes={"color_label": "Orange", "delta_yen": -2000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Silver ¥230,500\nBlue ¥229,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="Silver ¥230,500",
                    attributes={"color_label": "Silver", "price_yen": 230500},
                ),
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="Blue ¥229,000",
                    attributes={"color_label": "Blue", "price_yen": 229000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Blue-4000円\nBlack-4000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="Blue-4000円",
                    attributes={"color_label": "Blue", "delta_yen": -4000},
                ),
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="Black-4000円",
                    attributes={"color_label": "Black", "delta_yen": -4000},
                ),
            ],
        ),
    ]


# ----------------------------------------------------------------------
# LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

@lru_cache(maxsize=8192)
def extract_specs_shop12_llm_core(remark_for_llm: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, str, dict]]]:
    """
    返回:
      abs_list, delta_list, llm_dbg
    """
    remark_for_llm = (remark_for_llm or "").strip()
    if not remark_for_llm:
        return [], [], []

    if not HAS_LANGEXTRACT:
        return [], [], []

    try:
        llm_input = "色別価格ルール:\n" + remark_for_llm

        res = lx.extract(
            text_or_documents=llm_input,
            prompt_description=_LX_PROMPT,
            examples=_lx_examples(),
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            temperature=float(os.getenv("SHOP12_LLM_TEMPERATURE", "0.0")),
            fence_output=False,
            use_schema_constraints=False,
            max_char_buffer=2000,
            language_model_params={
                "timeout": int(os.getenv("SHOP12_LLM_TIMEOUT", "120")),
                "num_ctx": int(os.getenv("SHOP12_LLM_NUM_CTX", "4096")),
            },
        )

        exts = getattr(res, "extractions", []) or []
        llm_dbg: List[Tuple[str, str, dict]] = []
        abs_list: List[Tuple[str, int]] = []
        delta_list: List[Tuple[str, int]] = []

        for e in exts:
            cls_raw = (getattr(e, "extraction_class", "") or "").strip()
            txt = getattr(e, "extraction_text", "") or ""
            attrs = dict(getattr(e, "attributes", {}) or {})

            has_sign = bool(re.search(r"[+\-−－]\s*[０-９0-9]", txt))
            has_currency = bool(re.search(r"[¥￥円]", txt))

            if has_sign:
                effective_cls = "delta"
            elif has_currency:
                effective_cls = "abs_price"
            else:
                effective_cls = cls_raw or "delta"

            llm_dbg.append((effective_cls, txt, attrs))

            label = (
                str(attrs.get("color_label") or attrs.get("color") or attrs.get("colour") or "")
                .strip()
            )
            if not label:
                m_lbl = re.match(r"^[^\d0-9¥￥円+\-−－]+", txt)
                if m_lbl:
                    label = m_lbl.group(0).strip()
            if not label:
                continue

            if effective_cls == "abs_price":
                raw_price = attrs.get("price_yen")
                if raw_price is None:
                    raw_price = attrs.get("delta_yen")
                if raw_price is None:
                    raw_price = txt
                price_i = _norm_amount_to_int(raw_price)
                if price_i is None:
                    price_i = _norm_amount_to_int(txt)
                if price_i is None:
                    continue
                price_i = abs(int(price_i))
                abs_list.append((label, price_i))
                continue

            if effective_cls == "delta":
                raw_delta = attrs.get("delta_yen")
                delta_i: Optional[int] = None

                if isinstance(raw_delta, (int,)):
                    delta_i = int(raw_delta)
                else:
                    if raw_delta is not None:
                        delta_i = _norm_amount_to_int(raw_delta)

                if delta_i is None:
                    m = re.search(r"([+\-−－])\s*([０-９0-9][０-９0-9,，]*)", txt)
                    if m:
                        sign = m.group(1)
                        amt = _norm_amount_to_int(m.group(2))
                        if amt is not None:
                            delta_i = -amt if sign in ("-", "−", "－") else amt

                if delta_i is None:
                    continue
                delta_list.append((label, int(delta_i)))
                continue

        # 去重
        if abs_list:
            tmp = {}
            for k, v in abs_list:
                tmp[k] = v
            abs_list = list(tmp.items())
        if delta_list:
            tmp = {}
            for k, v in delta_list:
                tmp[k] = v
            delta_list = list(tmp.items())

        return abs_list, delta_list, llm_dbg

    except Exception as e:
        log_llm_extraction_error(logger, cleaner_name="shop12", shop_name="トゥインクル", error=e, text=remark_for_llm, row_index=None)
        return [], [], []


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop12_llm(
    remark_for_llm: str,
    idx: object = None,
    fallback_parse_rules_fn=None,
    cleaner_name: str = "shop12",
    shop_name: str = "トゥインクル",
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    LLM 提取 + Guardrails。LLM 失败时回退到正则。
    """
    abs_list, delta_list, _llm_dbg = extract_specs_shop12_llm_core(remark_for_llm)

    if not abs_list and not delta_list and remark_for_llm:
        logger.debug(
            "LLM returned empty, falling back to regex",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": shop_name,
                "cleaner_name": cleaner_name,
                "row_index": idx,
                "remark_preview": _truncate_for_log(remark_for_llm, 100),
            }
        )
        if fallback_parse_rules_fn:
            f_abs, f_delta = fallback_parse_rules_fn(remark_for_llm)
            if f_abs or f_delta:
                abs_list, delta_list = f_abs, f_delta

    return abs_list, delta_list
