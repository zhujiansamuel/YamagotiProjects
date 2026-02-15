from __future__ import annotations

"""
llm_shop14 — 買取楽園 LLM 提取模块

从 shop14_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM + Guardrails 封装函数
"""

import logging
import re
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from ..cleaner_tools import (
    _truncate_for_log,
    coerce_amount_yen,
    log_llm_extraction_error,
    lx,
    HAS_LANGEXTRACT,
    LABEL_SPLIT_RE_shop14,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_coerce_amount_yen = coerce_amount_yen


def _clean_remark_frag(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.lstrip("\ufeff").replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_labels(labels: str) -> List[str]:
    s = str(labels or "").strip()
    if not s:
        return []
    parts = LABEL_SPLIT_RE_shop14.split(s)
    return [p.strip() for p in parts if p and p.strip()]


def _labels_from_text_fallback(extraction_text: str) -> str:
    t = str(extraction_text or "")
    t = t.replace("全色", "")
    t = re.sub(r"(?:[+\-−－])?\s*(?:¥|￥)?\s*\d[\d,，]*\s*(?:円)?", "", t)
    t = t.strip()
    return t


# ----------------------------------------------------------------------
# LLM prompt & examples
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _extract_specs_shop14_lx_prompt():
    prompt = textwrap.dedent(
        """\
        你是信息抽取系统。请从输入文本中抽取"按颜色的价格规则（円）"。

        规则类型只有三类：
        1) all_colors：文本出现"全色"，可选跟金额（例如"全色 -3000""全色 3000円"）。
           表示所有颜色统一调整：final = base + amount_yen。若没写金额，amount_yen=0。
        2) abs_group：颜色标签(一个或多个)后面出现一个金额（例如"青 229,500""青/銀 229500円"）。
           表示这些颜色的最终价格等于该绝对金额。
        3) delta_group：颜色标签(一个或多个)后面出现带正负号的金额（例如"橙 -2500""銀+1000"）。
           表示这些颜色在基准价上加上差价（可为负）。

        分隔符可能是空格、换行、"/""／""、"","";"等。多个颜色可能共用同一个金额（例如"青/銀 229,500"），
        这种情况请把 attributes.labels 写成 "青/銀"（原样即可）。

        输出要求（非常重要）：
        - Use exact text for extraction_text（必须是原文连续子串，不要改写）。
        - 只抽取原文明确出现的规则，不要推断/补全。
        - attributes.amount_yen 必须是纯整数（去掉逗号/円/¥），差价允许负数。
        - attributes.labels：颜色标签，字符串（单色就写单个；多色就用原文分隔符，如 "青/銀"）。
        """
    )

    examples = [
        lx.data.ExampleData(
            text="青 229,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青 229,500",
                    attributes={"labels": "青", "amount_yen": "229500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="橙 -2500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙 -2500",
                    attributes={"labels": "橙", "amount_yen": "-2500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="全色 -3,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="all_colors",
                    extraction_text="全色 -3,000円",
                    attributes={"amount_yen": "-3000"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="青/銀 229,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青/銀 229,500",
                    attributes={"labels": "青/銀", "amount_yen": "229500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="橙/銀 -2,500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙/銀 -2,500円",
                    attributes={"labels": "橙/銀", "amount_yen": "-2500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="青 229,500\n橙 -2500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青 229,500",
                    attributes={"labels": "青", "amount_yen": "229500"},
                ),
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙 -2500",
                    attributes={"labels": "橙", "amount_yen": "-2500"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="全色",
            extractions=[
                lx.data.Extraction(
                    extraction_class="all_colors",
                    extraction_text="全色",
                    attributes={"amount_yen": "0"},
                )
            ],
        ),
    ]

    return prompt, examples


# ----------------------------------------------------------------------
# LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

@lru_cache(maxsize=4096)
def extract_specs_shop14_llm_core(
    text: str,
    split_color_amount_pairs_multi_fn=None,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]], List[dict]]]:
    """
    用 LangExtract(Ollama) 抽取规则。
    """
    out: Dict = {"all_delta": None, "abs": [], "delta": [], "raw": []}
    s = _clean_remark_frag(text)
    if not s:
        return out

    prompt, examples = _extract_specs_shop14_lx_prompt()

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except TypeError:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

    all_delta: Optional[int] = None
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []

    for e in (getattr(result, "extractions", None) or []):
        cls = str(getattr(e, "extraction_class", "") or "").strip()
        txt = str(getattr(e, "extraction_text", "") or "")
        attrs = getattr(e, "attributes", {}) or {}

        out["raw"].append({"class": cls, "text": txt, "attributes": attrs})

        cls_l = cls.lower().strip()

        # multi-pair 检测
        if split_color_amount_pairs_multi_fn:
            multi_pairs = split_color_amount_pairs_multi_fn(txt)
        else:
            multi_pairs = []

        if multi_pairs:
            vals_abs = [abs(v) for _, v in multi_pairs]
            kind: Optional[str] = None
            if "abs" in cls_l:
                kind = "abs"
            elif "delta" in cls_l or "diff" in cls_l:
                kind = "delta"
            else:
                if all(v >= 20000 for v in vals_abs):
                    kind = "abs"
                elif all(v <= 20000 for v in vals_abs):
                    kind = "delta"
                else:
                    big = sum(1 for v in vals_abs if v >= 20000)
                    kind = "abs" if big >= len(vals_abs) / 2.0 else "delta"

            for label, amt in multi_pairs:
                if kind == "abs":
                    abs_list.append((label, abs(int(amt))))
                else:
                    delta_list.append((label, int(amt)))

            logger.debug(
                "[LangExtract-multi] multi-pair detected",
                extra={
                    "event_type": "llm_multi_pair",
                    "shop_name": "買取楽園",
                    "cleaner_name": "shop14",
                    "extraction_text": _truncate_for_log(txt),
                    "kind": kind,
                    "pairs": str(multi_pairs),
                },
            )
            continue

        # 全色
        amount = None
        if isinstance(attrs, dict):
            amount = _coerce_amount_yen(attrs.get("amount_yen")) or _coerce_amount_yen(
                attrs.get("amount")
            )
        if amount is None:
            amount = _coerce_amount_yen(txt)

        if ("all" in cls_l) or ("全色" in txt):
            all_delta = int(amount) if amount is not None else 0
            continue

        # 普通 abs/delta
        labels_str = ""
        if isinstance(attrs, dict):
            labels_str = str(attrs.get("labels") or attrs.get("label") or "").strip()
        if not labels_str:
            labels_str = _labels_from_text_fallback(txt)

        labels = _split_labels(labels_str)

        kind = None
        if "abs" in cls_l:
            kind = "abs"
        elif "delta" in cls_l or "diff" in cls_l:
            kind = "delta"
        else:
            if amount is not None and abs(int(amount)) >= 20000:
                kind = "abs"
            elif amount is not None:
                kind = "delta"

        if not kind or amount is None or not labels:
            continue

        if kind == "abs":
            v = abs(int(amount))
            for lb in labels:
                abs_list.append((lb, v))
        else:
            v = int(amount)
            for lb in labels:
                delta_list.append((lb, v))

    out["all_delta"] = all_delta
    out["abs"] = abs_list
    out["delta"] = delta_list
    return out


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop14_llm(
    text: str,
    split_color_amount_pairs_multi_fn=None,
    cleaner_name: str = "shop14",
    shop_name: str = "買取楽園",
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]]]]:
    """LLM抽取 + Guardrails"""
    try:
        parsed = extract_specs_shop14_llm_core(text, split_color_amount_pairs_multi_fn)
    except Exception as exc:
        log_llm_extraction_error(logger, cleaner_name=cleaner_name, shop_name=shop_name, error=exc, text=text)
        return {"all_delta": None, "abs": [], "delta": []}

    return {
        "all_delta": parsed.get("all_delta"),
        "abs": parsed.get("abs", []),
        "delta": parsed.get("delta", []),
    }
