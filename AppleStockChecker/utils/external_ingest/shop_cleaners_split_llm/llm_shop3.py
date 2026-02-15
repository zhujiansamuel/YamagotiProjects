from __future__ import annotations

"""
llm_shop3 — 買取一丁目 LLM 提取模块

从 shop3_cleaner.py 提取的 LLM 相关代码：
- 带符号金额解析辅助函数
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM + Guardrails 封装函数
"""

import logging
import re
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..cleaner_tools import (
    _normalize_amount_text,
    normalize_text_basic,
    clean_label_token,
    log_llm_extraction_error,
    SIGNED_AMOUNT_PATTERN,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_clean_label_token = clean_label_token
_SIGNED_AMOUNT_PAT = SIGNED_AMOUNT_PATTERN

# ----------------------------------------------------------------------
# 带符号金额解析辅助函数
# ----------------------------------------------------------------------


def _extract_signed_amounts_from_text(text: str) -> List[int]:
    """
    从原文中提取所有 "+/- 金额"（单位：JPY），例如 '-1500'、'−3,000'、'＋２,０００'
    """
    s = normalize_text_basic(text or "")
    out: List[int] = []
    for m in _SIGNED_AMOUNT_PAT.finditer(s):
        sign = m.group(1)
        amt = _normalize_amount_text(m.group(2))
        if amt is None:
            continue
        out.append(int(amt) if sign in ("+", "＋") else -int(amt))
    return out


def _single_signed_delta_from_text(text: str) -> Optional[int]:
    """
    若原文只出现一种带符号金额（可能重复出现），返回它；否则返回 None
    """
    vals = _extract_signed_amounts_from_text(text)
    if not vals:
        return None
    return vals[0] if len(set(vals)) == 1 else None


def _infer_default_sign_from_text(text: str) -> Optional[int]:
    """
    若原文只出现"全为负"或"全为正"的带符号金额，返回其符号方向（-1 / +1）；混合则 None
    """
    vals = _extract_signed_amounts_from_text(text)
    if not vals:
        return None
    has_pos = any(v > 0 for v in vals)
    has_neg = any(v < 0 for v in vals)
    if has_neg and not has_pos:
        return -1
    if has_pos and not has_neg:
        return 1
    return None


def _parse_delta_int_llm(x: object, default_sign: Optional[int]) -> Optional[int]:
    """
    解析 LLM 输出的 delta。
    - 若无符号且 default_sign 可推断，则按 default_sign 赋符号
    - 若有符号但与 default_sign 冲突（原文只有一种符号方向），以原文为准修正
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None

    # 数值类型：可能没有"显式符号"，按 default_sign 修正方向
    if isinstance(x, int):
        if default_sign is not None and x != 0 and (x > 0) != (default_sign > 0):
            return int(default_sign * abs(x))
        return int(x)
    if isinstance(x, float):
        v = int(round(x))
        if default_sign is not None and v != 0 and (v > 0) != (default_sign > 0):
            return int(default_sign * abs(v))
        return v

    s = normalize_text_basic(str(x))
    if not s:
        return None

    explicit_sign: Optional[int] = None
    if s[0] in {"+", "-"}:
        explicit_sign = 1 if s[0] == "+" else -1
        s_num = s[1:].strip()
    else:
        s_num = s

    amt = _normalize_amount_text(s_num)
    if amt is None:
        return None

    # 无显式符号：必须借助原文 default_sign，否则不解析
    if explicit_sign is None:
        if default_sign is None:
            return None
        return int(default_sign * amt)

    # 有显式符号但原文只有一种符号方向：以原文为准
    if default_sign is not None and explicit_sign != default_sign:
        explicit_sign = default_sign

    return int(explicit_sign * amt)


# ----------------------------------------------------------------------
# LLM prompt & examples
# ----------------------------------------------------------------------

if HAS_LANGEXTRACT:
    _SHOP3_COLOR_DELTA_PROMPT = textwrap.dedent("""\
        你将看到一个很短的"备注/减价1"字符串，来源于日本二手回收价格表。
        目标：抽取"颜色标签 -> 价格差额(人民币/日元中的日元JPY)"的映射。
        输出要求（非常重要）：
        - 只抽取与"颜色差额/加价/减价"相关的信息；忽略与差额无关的描述（如 新品/未開封/SIMフリー 等）。
        - 每个颜色标签单独输出一条 Extraction。
        - extraction_class 固定为 "color_delta"。
        - extraction_text 必须是输入文本中的"原样子串"，只包含颜色标签本身（不要带分隔符、不要翻译、不要归一化）。
        - attributes 必须包含键 "delta_yen"，值为整数（可为字符串形式的整数也可），单位 JPY：
            * "-3000""−3,000""－３,０００"代表 -3000
            * "+2000""＋２,０００"代表 +2000
        - 如果文本里没有任何明确的 +/- 金额，则返回空的 extractions。
    """)

    _SHOP3_COLOR_DELTA_EXAMPLES = [
        lx.data.ExampleData(
            text="ブルー、シルバー　-1000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "-1000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="シルバー", attributes={"delta_yen": "-1000"}),
            ],
        ),
        lx.data.ExampleData(
            text="シルバー-3,000/ディープブルー-3,000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="シルバー", attributes={"delta_yen": "-3000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ディープブルー", attributes={"delta_yen": "-3000"}),
            ],
        ),
        lx.data.ExampleData(
            text="シルバー　-3,000 ブルー -3,000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="シルバー", attributes={"delta_yen": "-3000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "-3000"}),
            ],
        ),
        lx.data.ExampleData(
            text="ブラック、ブルー　-4000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブラック", attributes={"delta_yen": "-4000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "-4000"}),
            ],
        ),
        lx.data.ExampleData(
            text="ブルー +2000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "+2000"}),
            ],
        ),
        lx.data.ExampleData(
            text="オレンジ、ディープブルー-1500",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="オレンジ",
                                   attributes={"delta_yen": "-1500"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ディープブルー",
                                   attributes={"delta_yen": "-1500"}),
            ],
        ),
    ]
else:
    _SHOP3_COLOR_DELTA_PROMPT = ""
    _SHOP3_COLOR_DELTA_EXAMPLES = []


# ----------------------------------------------------------------------
# LLM 结果解析辅助
# ----------------------------------------------------------------------

def _iter_extractions_from_langextract_result(result) -> List[object]:
    """
    lx.extract(text) 可能返回 AnnotatedDocument，也可能返回 list；
    统一成 list[Extraction]。
    """
    if result is None:
        return []
    if isinstance(result, list):
        out = []
        for doc in result:
            out.extend(getattr(doc, "extractions", []) or [])
        return out
    return list(getattr(result, "extractions", []) or [])


# ----------------------------------------------------------------------
# LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

@lru_cache(maxsize=4096)
def extract_specs_shop3_llm_cached(text: str) -> Tuple[Tuple[str, int], ...]:
    s = (text or "").strip()
    if not s:
        return tuple()

    if not re.search(r"[0-9０-９+\-−－]", s):
        return tuple()

    if lx is None:
        return tuple()

    # 原文全局 delta / 默认符号
    delta_global = _single_signed_delta_from_text(s)
    default_sign = _infer_default_sign_from_text(s)

    result = lx.extract(
        text_or_documents=s,
        prompt_description=_SHOP3_COLOR_DELTA_PROMPT,
        examples=_SHOP3_COLOR_DELTA_EXAMPLES,
        model_id=OLLAMA_MODEL_ID,
        model_url=OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
    )

    mp: Dict[str, int] = {}
    for ext in _iter_extractions_from_langextract_result(result):
        if getattr(ext, "extraction_class", None) != "color_delta":
            continue

        label = _clean_label_token(getattr(ext, "extraction_text", "") or "")
        if not label:
            continue

        # 若原文只有一种带符号金额，直接覆盖所有 label 的 delta
        if delta_global is not None:
            mp[label] = int(delta_global)
            continue

        attrs = getattr(ext, "attributes", None) or {}
        delta_raw = attrs.get("delta_yen")
        delta = _parse_delta_int_llm(delta_raw, default_sign)

        if delta is None:
            delta = _parse_delta_int_llm(attrs.get("delta") or attrs.get("amount"), default_sign)

        if delta is None:
            continue

        mp[label] = int(delta)

    return tuple(mp.items())


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

def extract_specs_shop3_llm(
    text: str,
    row_index: object = None,
    cleaner_name: str = "shop3",
    shop_name: str = "買取一丁目",
) -> List[Tuple[str, int]]:
    """
    LLM 提取 + guardrails。仅 LLM 路径使用。
    """
    s = (text or "").strip()
    if not s:
        return []

    if not HAS_LANGEXTRACT or lx is None:
        return []

    try:
        result = list(extract_specs_shop3_llm_cached(s))
    except Exception as e:
        log_llm_extraction_error(
            logger, cleaner_name=cleaner_name, shop_name=shop_name,
            error=e, text=s, row_index=row_index,
        )
        return []

    return result
