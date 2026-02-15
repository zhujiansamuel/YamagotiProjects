from __future__ import annotations

"""
shop3 清洗器 — 買取一丁目

  原始文本（title / data5 / 减价1）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _normalize_model_generic()          ← Step 1: 机型归一化（cleaner_tools）
    │
    ├─ _parse_capacity_gb()                ← Step 2: 容量解析（cleaner_tools）
    │
    ├─ extract_price_yen()                 ← Step 3: 基础价提取（cleaner_tools）
    │
    ├─ _extract_specs_shop3_dispatch()  ← Step 6: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop3_regex()   ← Step 4: 正则提取差价
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop3_llm_cached()  ← Step 5a: LLM 核心提取
    │       └─ Guardrails (_parse_delta_int_llm,          ← Step 5b: 防幻觉过滤
    │           _infer_default_sign_from_text)
    │
    ├─ _label_matches_color_unified()      ← Step 7: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop3()                       ← Step 8: 主函数，生成输出行
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
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
    _normalize_amount_text,
    normalize_text_basic,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    log_llm_extraction_error,
    validate_columns,
    clean_label_token,
    dispatch_extraction,
    DELTA_PATTERN_STRICT as _DELTA_PATTERN_STRICT_IMPORTED,
    DELTA_PATTERN_LOOSE as _DELTA_PATTERN_LOOSE_IMPORTED,
    SIGNED_AMOUNT_PATTERN,
    lx,
    HAS_LANGEXTRACT,
    LABEL_SPLIT_RE_shop3 as _LABEL_SPLIT_RE,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop3"
SHOP_NAME = "買取一丁目"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# LangExtract (LLM 抽取)
# lx / HAS_LANGEXTRACT 从 cleaner_tools 统一导入

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# ----------------------------------------------------------------------
# Step 1-2: 文本归一化 & 金额解析
# ----------------------------------------------------------------------

# _clean_label_token → cleaner_tools.clean_label_token 统一导入
_clean_label_token = clean_label_token

# ----------------------------------------------------------------------
# Step 4: 正则提取差价
# ----------------------------------------------------------------------
# _LABEL_SPLIT_RE: 从 cleaner_tools.LABEL_SPLIT_RE_shop3 导入

# 正则模式 → cleaner_tools 统一导入
_SIGNED_AMOUNT_PAT = SIGNED_AMOUNT_PATTERN
_DELTA_PATTERN_STRICT = _DELTA_PATTERN_STRICT_IMPORTED
_DELTA_PATTERN_LOOSE = _DELTA_PATTERN_LOOSE_IMPORTED

def _extract_specs_shop3_regex(text: str) -> List[Tuple[str, int]]:
    """
    从 text 中提取 (label_raw, delta_int) 多条记录，支持多标签共用金额的写法。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = normalize_text_basic(str(text))
    for m in _DELTA_PATTERN_STRICT.finditer(s):
        labels_part = m.group("labels")
        sign = m.group("sign")
        amt_txt = m.group("amount")
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            continue
        if sign in ("-", "−", "－"):
            amt = -amt
        toks = [t for t in _LABEL_SPLIT_RE.split(labels_part) if t and t.strip()]
        for tok in toks:
            lbl = _clean_label_token(tok)
            if lbl:
                out.append((lbl, int(amt)))

    if not out:
        for m in _DELTA_PATTERN_LOOSE.finditer(s):
            labels_part = m.group("labels")
            sign = m.group("sign")
            amt_txt = m.group("amount")
            amt = _normalize_amount_text(amt_txt)
            if amt is None:
                continue
            if sign in ("-", "−", "－"):
                amt = -amt
            toks = [t for t in _LABEL_SPLIT_RE.split(labels_part) if t and t.strip()]
            for tok in toks:
                lbl = _clean_label_token(tok)
                if lbl:
                    out.append((lbl, int(amt)))

    return out

# LLM 相関代码已提取到 shop_cleaners_split_llm/llm_shop3.py
from ..shop_cleaners_split_llm.llm_shop3 import (
    extract_specs_shop3_llm as _extract_specs_shop3_llm,
)

# ----------------------------------------------------------------------
# Step 6: 提取模式调度
# ----------------------------------------------------------------------

def _extract_specs_shop3_dispatch(
    text: str,
    *,
    base_price: int,
    source_text_raw: str,
    row_index: object = None,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 PriceDecomposition
    """
    deltas, method = dispatch_extraction(
        EXTRACTION_MODE,
        regex_fn=lambda: _extract_specs_shop3_regex(text),
        llm_fn=lambda: _extract_specs_shop3_llm(text, row_index=row_index),
    )

    return PriceDecomposition(
        base_price=base_price,
        delta_specs=deltas,
        abs_specs=[],
        extraction_method=method,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 7: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop3 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop3(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq, extraction_mode=EXTRACTION_MODE)
    _log_seq += 1

    _log_seq = validate_columns(df, ["title", "data5", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    src = df.copy()
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok].reset_index(drop=True)
    if src.empty:
        log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=0, start_time=start_time, log_seq=_log_seq)
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_from_db()
    color_maps = _build_color_map(info_df)

    model_norm = src["title"].map(_normalize_model_generic)
    cap_gb     = src["title"].map(_parse_capacity_gb)

    try:
        base_price = src["data5"].map(extract_price_yen)
    except Exception:
        base_price = src["data5"].map(to_int_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    remark = src["减价1"] if "减价1" in src.columns else None

    if remark is not None:
        remark_text = remark.fillna("").astype(str)
    else:
        remark_text = pd.Series(["" for _ in range(len(src))])

    rows: List[dict] = []

    for i in range(len(src)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p0 = base_price.iat[i]
        t  = recorded_at.iat[i]
        model_text = str(src["title"].iat[i])

        rem_text = str(remark.iat[i]) if remark is not None else ""

        if not m:
            continue
        if pd.isna(c):
            continue
        if p0 is None:
            continue

        key = (m, int(c))
        cmap = color_maps.get(key)
        if not cmap:
            continue

        # ---- 提取 ----
        decomp = _extract_specs_shop3_dispatch(
            rem_text,
            base_price=p0,
            source_text_raw=rem_text,
            row_index=i,
        )
        new_rows, _log_seq = resolve_color_prices(
            decomp, cmap, _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=t,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=m,
            capacity_gb=int(c),
        )
        rows.extend(new_rows)

    out = assemble_output_df(rows, coerce_price=False)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=start_time, log_seq=_log_seq)

    return out
