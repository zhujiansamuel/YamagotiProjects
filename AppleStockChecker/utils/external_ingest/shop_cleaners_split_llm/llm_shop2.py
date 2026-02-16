from __future__ import annotations

"""
llm_shop2 — 海峡通信 LLM 提取模块

从 shop2_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数（带缓存）
- LLM + Guardrails 封装函数
"""

import json
import logging
import textwrap
from functools import lru_cache
from typing import List, Optional, Tuple

from ..cleaner_tools import (
    safe_to_text,
    coerce_int,
    log_llm_extraction_error,
    apply_llm_guardrails,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

_coerce_int = coerce_int

# ----------------------------------------------------------------------
# LLM prompt & examples
# ----------------------------------------------------------------------

if HAS_LANGEXTRACT:
    _COLOR_RULE_PROMPT = textwrap.dedent(
        """\
        あなたは中古スマホ買取表の「色ごとの減額条件」を解析するツールです。
        入力は data5 列に入っている短い日本語テキストです。例:
        - "青-1000"
        - "銀-5000+++青-3000"
        - "青-1000円\n※開封品 ¥183,000"
        など、色名と金額（減額/増額）が混在して書かれています。

        タスク:
        - data5 の中から「色グループ」と「基準価格からの差額（円）」をすべて抽出してください。
        - 減額は負の値、増額は正の値とします。
        - 抽出対象は、基準価格(data3)に対する相対額だけです。開封品価格など他の情報は無視してください。

        出力スキーマ:
        - 抽出するエンティティはすべて extraction_class="color_rule" とします。
        - 各 color_rule の attributes には次のキーを入れてください:
          - "group_label": 文字列。元テキスト中の色グループ名（例: "青", "銀", "スペースブラック", "全色"）
          - "delta_yen": 整数。基準価格からの差額（円）。減額は負の値、増額は正の値。

        注意:
        - "青-1000" や "銀-5000" のような書き方は「基準価格から 1000 円/5000 円減額」を意味します。
        - "青+2000" のような表現があれば、それは「基準価格から 2000 円増額」です。
        - テキストの中に色の情報がなく、金額だけの場合は無視してください。
        - 解釈に迷う場合は、その項目を抽出しないでください（安全側）。
        """
    )

    _COLOR_RULE_EXAMPLES: List = [
        lx.data.ExampleData(
            text="青-1000\n※開封品 ¥183,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="青-1000",
                    attributes={"group_label": "青", "delta_yen": -1000},
                )
            ],
        ),
        lx.data.ExampleData(
            text="銀-5000+++青-3000\n※開封品 ¥183,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="銀-5000",
                    attributes={"group_label": "銀", "delta_yen": -5000},
                ),
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="青-3000",
                    attributes={"group_label": "青", "delta_yen": -3000},
                ),
            ],
        ),
    ]
else:
    _COLOR_RULE_PROMPT = ""
    _COLOR_RULE_EXAMPLES = []

# ----------------------------------------------------------------------
# LLM 核心提取（无 guardrails），结果被缓存
# ----------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _extract_specs_shop2_llm_core(
    rule_text: str,
    _parse_rule_token_simple_fn=None,
    _regex_fallback_fn=None,
) -> dict:
    """LLM 核心提取（无 guardrails），结果被缓存。

    注意：_parse_rule_token_simple_fn 和 _regex_fallback_fn 参数不参与缓存 key，
    它们在模块初始化时通过 setup_shop2_llm_deps() 绑定。
    """
    s = (rule_text or "").strip()
    if not s:
        return {}

    if not HAS_LANGEXTRACT:
        if _regex_fallback_fn:
            return _regex_fallback_fn(s)
        return {}

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=_COLOR_RULE_PROMPT,
            examples=_COLOR_RULE_EXAMPLES,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

        doc = result.to_dict() if hasattr(result, "to_dict") else json.loads(
            json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o)))
        )

        rules: dict[str, int] = {}

        for ext in doc.get("extractions", []) or []:
            attrs = ext.get("attributes") or {}
            extraction_text = safe_to_text(ext.get("extraction_text"))

            # 1) 优先从 extraction_text 按行解析（支持复合标签）
            if extraction_text and _parse_rule_token_simple_fn:
                for piece in extraction_text.replace("\r", "\n").split("\n"):
                    parsed_list = _parse_rule_token_simple_fn(piece)
                    # parsed_list 现在是 List[Tuple[str, int]]，支持复合标签
                    for g, d in parsed_list:
                        rules[g] = d

            # 2) 再用 attributes 兜底
            group_label = safe_to_text(attrs.get("group_label"))
            delta = _coerce_int(attrs.get("delta_yen"))
            if group_label and (delta is not None):
                rules[group_label] = int(delta)

        # LLM 一条都没解析出来就回退
        if not rules:
            if _regex_fallback_fn:
                return _regex_fallback_fn(s)
            return {}

        return rules

    except Exception:
        if _regex_fallback_fn:
            return _regex_fallback_fn(s)
        return {}


# ----------------------------------------------------------------------
# LLM + Guardrails 封装
# ----------------------------------------------------------------------

# 模块级变量，由 setup_shop2_llm_deps() 设置
_parse_rule_token_simple_ref = None
_regex_fallback_ref = None


def setup_shop2_llm_deps(parse_rule_token_simple_fn, regex_fallback_fn):
    """由 shop2_cleaner 调用，注入正则相关依赖。"""
    global _parse_rule_token_simple_ref, _regex_fallback_ref
    _parse_rule_token_simple_ref = parse_rule_token_simple_fn
    _regex_fallback_ref = regex_fallback_fn


def extract_specs_shop2_llm_core(rule_text: str) -> dict:
    """对外暴露的 LLM 核心提取（使用已注入的依赖）。"""
    return _extract_specs_shop2_llm_core(
        rule_text,
        _parse_rule_token_simple_fn=_parse_rule_token_simple_ref,
        _regex_fallback_fn=_regex_fallback_ref,
    )


def extract_specs_shop2_llm(
    val,
    row_index: object = None,
    cleaner_name: str = "shop2",
    shop_name: str = "海峡通信",
) -> dict:
    """
    LLM 提取 + Guardrail A/B + 正则补全（仅 LLM 路径使用）。
    """
    s = safe_to_text(val)
    if not s:
        return {}

    llm_ok = False
    llm_rules: dict = {}
    try:
        llm_rules = extract_specs_shop2_llm_core(s)
        llm_ok = True
    except Exception as e:
        log_llm_extraction_error(
            logger, cleaner_name=cleaner_name, shop_name=shop_name,
            error=e, text=s, row_index=row_index,
        )

    # Guardrail A & B
    filtered_rules = apply_llm_guardrails(llm_rules, s)

    # 正则补全
    supplement = _regex_fallback_ref(s) if _regex_fallback_ref else {}
    merged = dict(filtered_rules)
    for k, v in supplement.items():
        merged.setdefault(k, v)

    # LLM 完全失败时，回退到纯正则
    if (not llm_ok) and (not merged):
        if _regex_fallback_ref:
            return _regex_fallback_ref(s)
        return {}

    return merged
