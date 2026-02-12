from __future__ import annotations

"""
shop4 清洗器 — モバイルミックス

  原始 DataFrame（data / data11 列）
    │
    ├─ _find_base_price()                    ← Step 1: 回溯查找基准价
    │
    ├─ _normalize_amount_text()              ← Step 2: 全角→半角归一化
    │
    ├─ _parse_color_delta_shop4_regex()      ← Step 3: 正则提取单行色差
    │
    ├─ _collect_adjustments_shop4_dispatch() ← Step 4: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   └─ _collect_adjustments_shop4_regex()   ← Step 5a: 正则逐行收集
    │   │
    │   └─ llm 路径:
    │       ├─ _collect_adjustments_shop4_llm()      ← Step 5b: LLM 核心提取
    │       └─ Guardrails (coerce + validate)        ← Step 6: 防幻觉过滤
    │
    ├─ _label_matches_color_shop4()          ← Step 7: 标签→颜色匹配
    │
    └─ clean_shop4()                         ← Step 8: 主函数，生成输出行
"""

import logging
import os
import re
import textwrap
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    PriceDecomposition,
    resolve_color_prices,
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
    _normalize_amount_text,
    normalize_text_basic,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop4"
SHOP_NAME = "モバイルミックス"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

SHOP4_OLLAMA_URL = os.getenv("SHOP4_OLLAMA_URL", "http://localhost:11434")
SHOP4_OLLAMA_MODEL_ID = os.getenv("SHOP4_OLLAMA_MODEL_ID", "gemma3:1b")

SHOP4_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# ----------------------------------------------------------------------
# Step 1: 全角→半角 & 金额归一化
# ----------------------------------------------------------------------

LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s]+")

def _coerce_int_maybe(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    sign = 1
    if s[0] in ("-", "−", "－"):
        sign = -1
    amt = _normalize_amount_text(s)
    if amt is None:
        return None
    return sign * int(amt)

def _split_labels(label: str) -> List[str]:
    return [p.strip() for p in LABEL_SPLIT_RE.split(label or "") if p and p.strip()]

# ----------------------------------------------------------------------
# Step 2: 基准价回溯查找
# ----------------------------------------------------------------------

def _find_base_price(df: pd.DataFrame, idx: int) -> Optional[int]:
    """
    按规范：机种行(data11非空)的上一行 data 是基准价。
    若上一行取不到，向上最多回溯 3 行找首个含"円"的金额。
    """
    for j in range(idx - 1, max(-1, idx - 4), -1):
        if j < 0:
            break
        s = str(df["data"].iat[j]) if "data" in df.columns else ""
        if s and ("円" in s or re.search(r"\d[\d,]*", s)):
            price = to_int_yen(s)
            if price:
                return int(price)
    return None

# ----------------------------------------------------------------------
# Step 3: 正则模式定义 & 单行解析
# ----------------------------------------------------------------------

_COLOR_DELTA_RE = re.compile(
    r"""^\s*
        (?P<label>全色|[\S　 ]*?[^\s　])
        \s*
        (?P<sign>[+\-−－])?
        \s*
        (?P<amount>\d[\d,]*)\s*円?
        \s*$
    """,
    re.VERBOSE,
)

def _parse_color_delta_shop4_regex(line: str) -> Optional[List[Tuple[str, int]]]:
    if not line or not isinstance(line, str):
        return None
    s = line.strip()
    if s == "全色" or s == "全 色":
        return [("全色", 0)]

    m = _COLOR_DELTA_RE.match(s)
    if not m:
        am = re.search(r"([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)\s*円?", s)
        if not am:
            if "全色" in s:
                return [("全色", 0)]
            return None
        sign = am.group(1) or "+"
        amt_txt = am.group(2)
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            try:
                amt = to_int_yen(amt_txt)
            except Exception:
                amt = None
        if amt is None:
            return None
        if sign in ("-", "−", "－"):
            amt = -int(amt)
        else:
            amt = int(amt)

        label_part = s[:am.start()].strip()
        if not label_part:
            return None
        labels = [p for p in LABEL_SPLIT_RE.split(label_part) if p]
        return [(lbl.strip(), int(amt)) for lbl in labels]

    label_raw = m.group("label").strip()
    sign = m.group("sign") or "+"
    amt_val = None
    try:
        amt_val = to_int_yen(m.group("amount"))
    except Exception:
        amt_val = None
    if amt_val is None:
        amt_val = _normalize_amount_text(m.group("amount"))
    if amt_val is None:
        return None

    amt_val = int(amt_val)
    if sign in ("-", "−", "－"):
        amt_val = -amt_val

    labels = [p for p in LABEL_SPLIT_RE.split(label_raw) if p]
    if not labels:
        return None
    return [(lbl.strip(), int(amt_val)) for lbl in labels]

# ----------------------------------------------------------------------
# Step 4: 颜色家族同义词 & 匹配
# ----------------------------------------------------------------------

FAMILY_SYNONYMS_shop4 = {
    # blue
    "blue": ["ブルー", "青", "ディープブルー"],
    "ブルー": ["ブルー", "青", "ディープブルー"],
    "青": ["ブルー", "青", "ディープブルー"],
    "ディープブルー": ["ブルー", "青", "ディープブルー"],
    # black
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],
    # white / starlight
    "white": ["ホワイト", "白", "スターライト"],
    "ホワイト": ["ホワイト", "白", "スターライト"],
    "白": ["ホワイト", "白", "スターライト"],
    "スターライト": ["ホワイト", "白", "スターライト"],
    # silver
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],
    # gold
    "gold": ["ゴールド", "金"],
    "ゴールド": ["ゴールド", "金"],
    "金": ["ゴールド", "金"],
    # green
    "green": ["グリーン", "緑"],
    "グリーン": ["グリーン", "緑"],
    "緑": ["グリーン", "緑"],
    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],
    # red
    "red": ["レッド", "赤"],
    "レッド": ["レッド", "赤"],
    "赤": ["レッド", "赤"],
    # yellow
    "yellow": ["イエロー", "黄"],
    "イエロー": ["イエロー", "黄"],
    "黄": ["イエロー", "黄"],
    # purple
    "purple": ["パープル", "紫"],
    "パープル": ["パープル", "紫"],
    "紫": ["パープル", "紫"],
    # natural / titanium
    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],
    "チタン": ["チタン", "チタニウム"],
    "チタニウム": ["チタン", "チタニウム"],
    # midnight
    "ミッドナイト": ["ミッドナイト"],
}

def _label_matches_color_shop4(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配：精确(归一) | 原文子串 | 颜色家族关键词命中"""
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop4:
            candidates.update(FAMILY_SYNONYMS_shop4[k])
    if not candidates:
        for _, toks in FAMILY_SYNONYMS_shop4.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break
    return any(tok in str(color_raw) for tok in candidates)

# ----------------------------------------------------------------------
# Step 5a: 正则收集（逐行扫描 block）
# ----------------------------------------------------------------------

def _collect_adjustments_shop4_regex(
    df: pd.DataFrame, start_idx: int,
) -> Tuple[Dict[str, int], List[Tuple[str, int]], Dict[str, str]]:
    """
    纯正则版：逐行扫描 block 收集颜色差额。
    返回：(adjustments, delta_specs, color_delta_label_map)
      - adjustments: { _norm(label) | "ALL" : delta_int }
      - delta_specs: [(label_raw, delta)] 原始标签列表
      - color_delta_label_map: { _norm(label) : label_raw }
    """
    result: Dict[str, int] = {}
    delta_specs: List[Tuple[str, int]] = []
    color_delta_label_map: Dict[str, str] = {}
    n = len(df)
    for j in range(start_idx, n):
        nxt_model = ""
        if "data11" in df.columns:
            val = df["data11"].iat[j]
            nxt_model = str(val) if val is not None else ""
        if j > start_idx and nxt_model.strip():
            break

        line = ""
        if "data" in df.columns:
            val = df["data"].iat[j]
            line = str(val) if val is not None else ""

        parsed = _parse_color_delta_shop4_regex(line)
        if not parsed:
            continue

        for label, delta in parsed:
            delta_specs.append((label, int(delta)))
            if "全色" in label:
                result["ALL"] = int(delta)
            else:
                nk = _norm(label)
                result[nk] = int(delta)
                color_delta_label_map[nk] = label
    return result, delta_specs, color_delta_label_map

# ----------------------------------------------------------------------
# Step 5b: LLM 核心提取
# ----------------------------------------------------------------------

try:
    import langextract as lx
    from langextract.data import ExampleData, Extraction
    _HAS_LANGEXTRACT = True
except Exception:
    lx = None
    ExampleData = None
    Extraction = None
    _HAS_LANGEXTRACT = False

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
    if not _HAS_LANGEXTRACT:
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

def _lx_extract_color_deltas(text: str) -> list:
    """
    对 text 做一次 LangExtract 抽取，返回 result.extractions（若不可用则空列表）。
    """
    if not (_HAS_LANGEXTRACT and isinstance(text, str) and text.strip()):
        return []

    import langextract as lx

    kwargs = dict(
        text_or_documents=text,
        prompt_description=_SHOP4_LE_PROMPT,
        examples=_get_shop4_le_examples(),
        model_id=SHOP4_OLLAMA_MODEL_ID,
        model_url=SHOP4_OLLAMA_URL,
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

    # 兼容不同版本：有的版本建议显式指定 OllamaLanguageModel
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

# ----------------------------------------------------------------------
# Step 6: LLM + Guardrails（仅 LLM 路径使用）
# ----------------------------------------------------------------------

def _collect_adjustments_shop4_llm_with_guardrails(
    df: pd.DataFrame,
    start_idx: int,
    row_index: object = None,
) -> Tuple[Dict[str, int], List[Tuple[str, int]], Dict[str, str]]:
    """
    用 LangExtract 一次性解析"机种段落"(block)里的所有颜色±金额，
    并应用 guardrails 过滤幻觉。
    返回：(adjustments, delta_specs, color_delta_label_map)
    """
    _empty: Tuple[Dict[str, int], List[Tuple[str, int]], Dict[str, str]] = ({}, [], {})
    lines: List[str] = []
    n = len(df)

    # 收集 block 文本：从 start_idx 到下一个 data11 非空前一行
    for j in range(start_idx, n):
        if j > start_idx:
            nxt_model = ""
            val = df["data11"].iat[j] if "data11" in df.columns else ""
            nxt_model = str(val) if val is not None else ""
            if nxt_model.strip():
                break
        raw = df["data"].iat[j] if "data" in df.columns else ""
        lines.append("" if raw is None else str(raw))

    if not lines:
        return _empty

    block_text = "\n".join(lines)

    # 计算每一行在 block_text 的范围，用于识别"同一行(机种行)的全色"
    line0_start = 0
    line0_end = len(lines[0]) if lines else 0

    try:
        exts = _lx_extract_color_deltas(block_text)
    except Exception as e:
        logger.warning(
            "LangExtract extraction failed",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "error": str(e),
                "error_type": type(e).__name__,
                "model_id": SHOP4_OLLAMA_MODEL_ID,
                "model_url": SHOP4_OLLAMA_URL,
                "row_index": row_index,
                "text_length": len(block_text),
                "text_preview": _truncate_for_log(block_text, 100),
            },
        )
        return _empty

    if not exts:
        return _empty

    # 按出现顺序处理，保持覆盖逻辑一致
    exts = sorted(exts, key=_get_start_pos)

    result: Dict[str, int] = {}
    delta_specs: List[Tuple[str, int]] = []
    color_delta_label_map: Dict[str, str] = {}
    global_all_delta: Optional[int] = None

    for ex in exts:
        cls = str(getattr(ex, "extraction_class", "") or "").strip()
        if cls and cls.lower() not in {"color_delta", "colordelta", "color"}:
            # 防止模型乱输出其他类（Guardrail: class filter）
            continue

        label = str(getattr(ex, "extraction_text", "") or "").strip()
        if not label:
            continue

        # Guardrail: label 必须在原文中出现
        if label not in block_text:
            continue

        attrs = getattr(ex, "attributes", None)
        attrs = attrs if isinstance(attrs, dict) else {}
        delta = _coerce_int_maybe(attrs.get("delta_yen"))
        if delta is None:
            # 兜底：如果是"全色"且无金额，按 0
            if "全色" in label and not re.search(r"[0-9０-９]", block_text):
                delta = 0
            else:
                continue

        # Guardrail: delta 金额的绝对值必须在原文中出现（防幻觉金额）
        block_text_no_commas = block_text.replace(",", "").replace("，", "")
        if delta != 0 and str(abs(int(delta))) not in block_text_no_commas:
            continue

        start_pos = _get_start_pos(ex)

        # same-line（机种行同一行 data）里的 全色：作为最高优先级的 ALL
        if "全色" in label and line0_start <= start_pos < max(line0_end, line0_start):
            global_all_delta = int(delta)

        # label 可能是复合项，拆分后分别写入
        for lbl in _split_labels(label):
            delta_specs.append((lbl, int(delta)))
            if "全色" in lbl:
                result["ALL"] = int(delta)
            else:
                nk = _norm(lbl)
                result[nk] = int(delta)
                color_delta_label_map[nk] = lbl

    # 同行全色优先覆盖
    if global_all_delta is not None:
        result["ALL"] = int(global_all_delta)

    return result, delta_specs, color_delta_label_map

# ----------------------------------------------------------------------
# Step 7: 提取模式调度
# ----------------------------------------------------------------------

def _collect_adjustments_shop4_dispatch(
    df: pd.DataFrame,
    start_idx: int,
    row_index: object = None,
) -> Tuple[Dict[str, int], str, List[Tuple[str, int]], Dict[str, str]]:
    """
    根据 SHOP4_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 (adjustments, extraction_method, delta_specs, color_delta_label_map)
    """
    mode = SHOP4_EXTRACTION_MODE

    if mode == "regex":
        result, ds, cdl = _collect_adjustments_shop4_regex(df, start_idx)
        return result, "regex", ds, cdl

    if mode == "llm":
        result, ds, cdl = _collect_adjustments_shop4_llm_with_guardrails(
            df, start_idx, row_index=row_index,
        )
        return result, "llm", ds, cdl

    # ---- auto: 正則優先，正則無颜色结果时 LLM 兜底 ----
    regex_result, ds, cdl = _collect_adjustments_shop4_regex(df, start_idx)
    if regex_result:
        return regex_result, "regex", ds, cdl

    llm_result, ds, cdl = _collect_adjustments_shop4_llm_with_guardrails(
        df, start_idx, row_index=row_index,
    )
    return llm_result, "llm", ds, cdl

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop4(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0

    logger.info(
        "shop4 cleaner started",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "log_seq": _log_seq,
            "input_rows": len(df),
            "extraction_mode": SHOP4_EXTRACTION_MODE,
        },
    )
    _log_seq += 1

    for c in ["data", "data11", "time-scraped"]:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "log_seq": _log_seq,
                    "missing_column": c,
                    "available_columns": list(df.columns),
                },
            )
            _log_seq += 1
            raise ValueError(f"shop4 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_from_db()
    pn_map = _build_color_map(info_df)

    def _next_model_idx(start: int) -> int:
        nn = len(df)
        for k in range(start + 1, nn):
            s = str(df["data11"].iat[k]) if df["data11"].iat[k] is not None else ""
            if s.strip():
                return k
        return nn

    rows: List[dict] = []
    n = len(df)

    for i in range(n):
        model_text = str(df["data11"].iat[i]) if df["data11"].iat[i] is not None else ""
        model_text = model_text.strip()
        if not model_text:
            continue

        block_end = _next_model_idx(i) - 1
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        rec_at = parse_dt_aware(df["time-scraped"].iat[i])

        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            continue

        base_price = _find_base_price(df, i)
        if base_price is None:
            continue

        # ---- 提取 ----
        _, extraction_method, delta_specs_raw, _ = \
            _collect_adjustments_shop4_dispatch(df, i, row_index=i)

        # ---- 收集 block 文本（source_text_raw_full） ----
        block_lines_raw = []
        for j in range(i, block_end + 1):
            raw = str(df["data"].iat[j]) if df["data"].iat[j] is not None else ""
            if raw.strip():
                block_lines_raw.append(raw.strip())
        source_text_raw_full = " | ".join(block_lines_raw)

        # ---- PriceDecomposition + resolve_color_prices ----
        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=list(delta_specs_raw),
            abs_specs=[],
            extraction_method=extraction_method,
            source_text_raw=source_text_raw_full,
        )

        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_to_pn,
            _label_matches_color_shop4,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed = round(time.time() - start_time, 2)
    logger.info(
        "shop4 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "log_seq": _log_seq,
            "input_rows": len(df),
            "output_records": len(out),
            "elapsed_seconds": elapsed,
        },
    )

    return out
