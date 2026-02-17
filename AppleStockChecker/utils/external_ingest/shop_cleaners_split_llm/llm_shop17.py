from __future__ import annotations

"""
llm_shop17 — ゲストモバイル LLM 提取模块

从 shop17_cleaner.py 提取的 LLM 相关代码：
- LLM prompt & few-shot examples
- LLM 核心提取函数
"""

import logging
import re
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..cleaner_tools import (
    log_llm_extraction_error,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
)

logger = logging.getLogger(__name__)

try:
    from langextract.data import ExampleData, Extraction
except Exception:
    ExampleData = None
    Extraction = None


# ----------------------------------------------------------------------
# LLM prompt
# ----------------------------------------------------------------------

COLOR_DELTA_PROMPT_SHOP17 = textwrap.dedent("""
あなたは中古iPhone買取表の「色減額」欄を解析するアシスタントです。
入力は1つのセルのテキストです。この中には色ごとの減額情報のほかに、
「郵送は翌日着のみ保証」「持ち込みのみ保証」「利用制限△-10000」などの
色と関係ない条件も含まれます。

タスク:
- 色名ごとの減額（または増額）だけを抽出してください。
- 色名の例: スカイブルー, スペースブラック, クラウドホワイト, ライトゴールド, シルバー, ブルー など。
- 「利用制限△-10000」や「保証開始3か月未満減額なし」など、色と無関係な金額・文言は無視してください。
- 「色名なし」(例: シルバーなし) はその色の delta=0 として扱います。
- 色名が付いていない「減額なし」(例: △減額なし) は無視します。

出力ポリシー:
- extraction_class は必ず "color_delta" にしてください。
- extraction_text には、表に書かれている「色と金額のフレーズ全体」
  （例: "スカイブルー-3,000", "クラウドホワイト：なし", "シルバーなし"）をそのまま入れてください。
- attributes には必ず次のキーを入れてください:
  - "color": 色名だけ（例: "スカイブルー"）
  - "delta": その色の価格差（整数。値引きは負の数。例: -3000）
  - "raw": 抜き出した元の部分文字列（extraction_text と同じでもよい）

その他ルール:
- 価格は円単位で扱い、「円」「,」などは無視して整数に変換してください。
- 色名が複数ある場合は、それぞれ1つずつ color_delta を出力してください。
- 文章内の改行や空行は無視して構いません。
""").strip()


# ----------------------------------------------------------------------
# LLM examples
# ----------------------------------------------------------------------

@lru_cache()
def _get_color_delta_examples_shop17() -> list:
    if not HAS_LANGEXTRACT:
        return []

    examples = []

    examples.append(
        ExampleData(
            text="色減額:スカイブルー-3,000\n\n郵送は翌日着のみ保証\n\n利用制限△-10000",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー-3,000",
                    attributes={
                        "color": "スカイブルー",
                        "delta": "-3000",
                        "raw": "スカイブルー-3,000",
                    },
                )
            ],
        )
    )

    examples.append(
        ExampleData(
            text="色減額:スカイブルー-4,000/スペースブラック-4,000\n\n持ち込みのみ保証\n\n利用制限△-10000",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー-4,000",
                    attributes={
                        "color": "スカイブルー",
                        "delta": "-4000",
                        "raw": "スカイブルー-4,000",
                    },
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スペースブラック-4,000",
                    attributes={
                        "color": "スペースブラック",
                        "delta": "-4000",
                        "raw": "スペースブラック-4,000",
                    },
                ),
            ],
        )
    )

    examples.append(
        ExampleData(
            text="色減額:シルバーなし/ブルー-1000\n\n郵送は翌日着のみ保証\n\n△減額なし 保証開始3か月未満減額なし",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバーなし",
                    attributes={
                        "color": "シルバー",
                        "delta": "0",
                        "raw": "シルバーなし",
                    },
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー-1000",
                    attributes={
                        "color": "ブルー",
                        "delta": "-1000",
                        "raw": "ブルー-1000",
                    },
                ),
            ],
        )
    )

    return examples


# ----------------------------------------------------------------------
# Delta 解析辅助
# ----------------------------------------------------------------------

def _parse_delta_attr_to_int(val) -> Optional[int]:
    if val is None:
        return None
    s = str(val)
    s = s.replace("円", "").replace(",", "").replace(" ", "").replace("　", "")
    s = s.replace("−", "-").replace("－", "-")
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


# ----------------------------------------------------------------------
# LLM 提取函数
# ----------------------------------------------------------------------

def extract_specs_shop17_llm(
    text: str,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
    normalize_color_text_fn=None,
    pick_unopened_section_fn=None,
    is_plausible_color_label_fn=None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if not HAS_LANGEXTRACT:
        return ([], [])
    if not text or not str(text).strip():
        return ([], [])

    if normalize_color_text_fn and pick_unopened_section_fn:
        s = normalize_color_text_fn(pick_unopened_section_fn(str(text)))
    else:
        s = str(text).strip()

    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        return ([], [])

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=COLOR_DELTA_PROMPT_SHOP17,
            examples=_get_color_delta_examples_shop17(),
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            temperature=0.0,
            fence_output=False,
            use_schema_constraints=False,
            prompt_validation_level="OFF",
            prompt_validation_strict=False,
        )
    except Exception as e:
        log_llm_extraction_error(
            logger, cleaner_name=cleaner_name or "shop17",
            shop_name=shop_name or "ゲストモバイル",
            error=e, text=s,
            row_index=row_context.get("row_index") if row_context else None,
        )
        return ([], [])

    out: List[Tuple[str, int]] = []
    extractions = getattr(result, "extractions", None) or []
    for ext in extractions:
        try:
            if ext.extraction_class != "color_delta":
                continue
            attrs = ext.attributes or {}
            color = (attrs.get("color") or ext.extraction_text or "").strip()
            if is_plausible_color_label_fn and not is_plausible_color_label_fn(color):
                continue
            delta_int = _parse_delta_attr_to_int(attrs.get("delta"))
            if delta_int is None:
                txt = (ext.extraction_text or "").strip()
                m = re.search(r"([+\-−－]?\d[\d,]*)", txt)
                if m:
                    delta_int = _parse_delta_attr_to_int(m.group(1))
            if delta_int is None:
                continue
            out.append((color, delta_int))
        except Exception:
            continue

    return (out, [])  # LLM 仅提取 delta，abs 留空（与 regex 双路径一致）
