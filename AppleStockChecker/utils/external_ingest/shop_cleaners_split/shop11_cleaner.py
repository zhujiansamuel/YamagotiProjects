from __future__ import annotations

"""
shop11 清洗器 — モバステ

  原始文本（storage_name / price_unopened / caution_empty）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ normalize_text_basic()              ← Step 1: 全角→半角归一化（cleaner_tools）
    │
    ├─ extract_price_yen()                   ← Step 2: 基础价解析（cleaner_tools 统一）
    │
    ├─ Step 3: 机型/容量解析（规则）
    │   └─ _normalize_model_generic + _parse_capacity_gb
    │
    ├─ dispatch_extraction_to_price_decomposition() ← Step 7: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop11_regex()   ← Step 5: 正则提取差价
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop11_llm_core()        ← Step 6a: LLM 核心提取
    │       └─ Guardrails (delta 合理性检查)            ← Step 6b: 防幻觉过滤
    │
    ├─ _label_matches_color_unified()       ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop11()                        ← Step 8: 主函数，生成输出行
"""

import logging
import os
import re
import textwrap
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dateparser

from ...external_ingest.cleaner_tools import parse_dt_aware
from ..cleaner_tools import (
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _truncate_for_log,
    _norm_strip,
    normalize_text_basic,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    coerce_int,
    dispatch_extraction_to_price_decomposition,
    lx,
    HAS_LANGEXTRACT,
    log_llm_extraction_error,
    LABEL_SPLIT_RE_shop11,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
    setup_color_cleaner,
    finalize_color_cleaner,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop11"
SHOP_NAME = "モバステ"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# _coerce_int → cleaner_tools.coerce_int 统一导入
_coerce_int = coerce_int

# ----------------------------------------------------------------------
# Step 3: LLM 机型/容量解析 — 已提取到 shop_cleaners_split_llm/llm_shop11.py
# ----------------------------------------------------------------------
from ..shop_cleaners_split_llm.llm_shop11 import (
    lx_parse_storage_shop11 as _lx_parse_storage_shop11,
    extract_specs_shop11_llm_core as _extract_specs_shop11_llm_core,
    extract_specs_shop11_llm as _extract_specs_shop11_llm_impl,
)

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop11 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则提取颜色差价
# ----------------------------------------------------------------------

_COLOR_GROUP_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # 最多 80 char 的 label group（不会以数字或 +/- 开头）
    [：:]\s*                               # 必须有 冒号（：或:）作为分隔（这是最常见的情形）
    (?P<sign>[+\-−－]?)\s*                 # 可选 +/-
    (?P<amount>[\d０-９,，]+)              # 金额（含全角数字与逗号）
    (?:\s*円|\s*¥|\s*￥)?                  # 可选货币符
    """,
    re.UNICODE | re.VERBOSE,
)

_COLOR_GROUP_FALLBACK_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # label group（保守）
    [\s]*?(?P<sign>[+\-−－])\s*
    (?P<amount>[\d０-９,，]+)
    (?:\s*円|\s*¥|\s*￥)?
    """,
    re.UNICODE | re.VERBOSE,
)

def _extract_specs_shop11_regex(text: str) -> List[Tuple[str, int]]:
    """
    纯正则版颜色差额解析，返回 [(label_raw, delta_int), ...]
    支持：
      - "シルバー・ブルー：-1,000円(未開封)" -> ('シルバー', -1000), ('ブルー', -1000)
      - "ブルー、ブラック：-2,000円(未開封)"
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text).strip()
    # 去掉括号内备注 (未開封) 等
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()
    if not s:
        return out

    s_norm = normalize_text_basic(s)

    # 1) 主匹配：labelGroup：+/-?amount
    for m in _COLOR_GROUP_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = normalize_text_basic(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        # 把 labels 按常见分隔符拆成多个 label
        for lbl in LABEL_SPLIT_RE_shop11.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 2) 回退匹配（如 "ブルー -4000"）
    for m in _COLOR_GROUP_FALLBACK_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = normalize_text_basic(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        for lbl in LABEL_SPLIT_RE_shop11.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 去重：若同 label 多次解析到，以最后一个为准（保留最后出现的 delta）
    if out:
        tmp: Dict[str, int] = {}
        for lbl, d in out:
            tmp[lbl] = d
        return list(tmp.items())

    return out

# ----------------------------------------------------------------------
# Step 6: LLM + Guardrails 颜色差价提取 — 已提取到 shop_cleaners_split_llm/llm_shop11.py
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop11(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["storage_name", "price_unopened", "caution_empty", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    df2 = df.copy().reset_index(drop=True)

    rows: List[dict] = []

    for i, row in df2.iterrows():
        storage_raw = row.get("storage_name")
        price_raw = row.get("price_unopened")
        caution_raw = row.get("caution_empty")
        time_raw = row.get("time-scraped")

        storage = str(storage_raw or "").strip()
        if not storage:
            continue

        model_text = storage

        model_norm = _normalize_model_generic(storage)
        cap_fb = _parse_capacity_gb(storage)
        if not model_norm or cap_fb is None:
            continue
        cap_gb = int(cap_fb)
        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)

        if not color_map:
            model_norm2 = _normalize_model_generic(model_norm) or model_norm
            key2 = (model_norm2, cap_gb)
            color_map = ctx.color_map.get(key2)
            if color_map:
                key = key2
                model_norm = model_norm2

        if not color_map:
            continue

        base_price = extract_price_yen(price_raw)
        if base_price is None:
            continue
        base_price = int(base_price)

        rec_at_raw = time_raw
        try:
            rec_at = dateparser.parse(str(rec_at_raw)) if pd.notna(rec_at_raw) else None
        except Exception:
            rec_at = rec_at_raw

        avail_colors = tuple(color_map.keys())
        caution_txt = normalize_text_basic(str(caution_raw or ""))
        source_text_raw_full = str(caution_raw or "")

        decomp = dispatch_extraction_to_price_decomposition(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_shop11_regex(caution_txt),
            llm_fn=lambda: [
                (cn, dv) for cn, dv in
                _extract_specs_shop11_llm_impl(
                    caution_txt, avail_colors, color_map,
                    regex_fn=_extract_specs_shop11_regex,
                ).items()
            ],
            base_price=base_price,
            source_text_raw=source_text_raw_full,
            result_adapter=lambda r: (r, []),
        )

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=int(i),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
