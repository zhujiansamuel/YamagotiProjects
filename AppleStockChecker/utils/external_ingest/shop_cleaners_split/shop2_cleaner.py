from __future__ import annotations

"""
shop2 清洗器 — 海峡通信

  原始文本（data2-1 / data2-2 / data3 / data5）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _is_target()                          ← Step 1: SIMfree+未開封 过滤
    │
    ├─ extract_price_yen()                   ← Step 2: 基础价(data3)解析（cleaner_tools 统一）
    │
    ├─ _normalize_model_generic()            ← Step 3: 机型规范化（cleaner_tools 统一）
    │
    ├─ _parse_capacity_gb()                  ← Step 4: 容量解析
    │
    ├─ dispatch_extraction_to_price_decomposition() ← Step 7: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop2_regex()   ← Step 5: 正则提取规则
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop2_llm_impl()     ← Step 6a: LLM 核心提取 (shop_cleaners_split_llm/llm_shop2.py)
    │       ├─ Guardrail A: label 原文校验         ← Step 6b: 防幻觉过滤
    │       ├─ Guardrail B: amount 原文校验        ← Step 6b: 防幻觉过滤
    │       └─ _extract_specs_shop2_regex()   ← Step 6c: 正则补全
    │
    ├─ _label_matches_color_unified()         ← Step 8: 颜色匹配（cleaner_tools 统一）
    │
    ├─ resolve_color_prices()               ← Step 9: 统一定价流程（cleaner_tools）
    │
    └─ clean_shop2()                         ← Step 10: 主函数，生成输出行
"""

import logging
import os
import re
import time
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.cleaner_tools import parse_dt_aware
from ..cleaner_tools import (
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _truncate_for_log,
    _label_matches_color_unified,
    safe_to_text,
    PriceDecomposition,
    resolve_color_prices,
    setup_color_cleaner,
    finalize_color_cleaner,
    log_row_skip,
    log_llm_extraction_error,
    coerce_int,
    dispatch_extraction_to_price_decomposition,
    apply_llm_guardrails,
    SIGN_MINUS_CHARS,
    SIGN_PLUS_CHARS,
    LABEL_SPLIT_RE_shop2,
    lx,
    HAS_LANGEXTRACT,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop2"
SHOP_NAME = "海峡通信"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

# LangExtract + Ollama (本地 LLM) 集成
# lx / HAS_LANGEXTRACT 从 cleaner_tools 统一导入

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

def _norm(s: str) -> str:
    """shop2 专用：strip 归一化，用于 model/color/part_number 等字段"""
    return (s or "").strip()


# ----------------------------------------------------------------------
# Step 1: SIMfree+未開封 过滤
# ----------------------------------------------------------------------

def _is_target(s: str) -> bool:
    s = (s or "").lower()
    return ("simfree" in s) and ("未開封" in s)

# ----------------------------------------------------------------------
# Step 3: 颜色匹配（使用 cleaner_tools 统一实现）
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Step 4: 规则解析辅助函数
# ----------------------------------------------------------------------

# _coerce_int / _INT_RE / _SIGN_MINUS / _SIGN_PLUS → cleaner_tools 统一导入
_coerce_int = coerce_int
_SIGN_MINUS = SIGN_MINUS_CHARS
_SIGN_PLUS = SIGN_PLUS_CHARS

def _parse_rule_token_simple(token: str) -> List[Tuple[str, int]]:
    """
    解析单条规则 token，支持复合标签，例如：
      '黒-2000' -> [('黒', -2000)]
      '青/オレンジ-2000円' -> [('青', -2000), ('オレンジ', -2000)]
      '銀 +3000' -> [('銀', 3000)]
    """
    s = safe_to_text(token)
    if not s:
        return []

    # 从末尾找数字串
    i = len(s) - 1
    while i >= 0 and not s[i].isdigit():
        i -= 1
    if i < 0:
        return []

    j = i
    while j >= 0 and s[j].isdigit():
        j -= 1
    num_str = s[j + 1 : i + 1]
    if not num_str:
        return []

    # 数字前找 +/- 符号（允许中间有空格）
    k = j
    while k >= 0 and s[k].isspace():
        k -= 1
    if k < 0:
        return []

    sign_ch = s[k]
    if sign_ch in _SIGN_PLUS:
        sign = 1
    elif sign_ch in _SIGN_MINUS:
        sign = -1
    else:
        return []

    group = s[:k].strip().strip(" :：\t")
    if not group:
        return []

    # 使用 LABEL_SPLIT_RE_shop2 分割复合标签
    amt = sign * int(num_str)
    labels = [lbl.strip() for lbl in LABEL_SPLIT_RE_shop2.split(group) if lbl.strip()]
    return [(lbl, amt) for lbl in labels]

# ----------------------------------------------------------------------
# Step 5: 纯正则版规则提取
# ----------------------------------------------------------------------

def _extract_specs_shop2_regex(val) -> dict:
    """
    对原始 data5 做正则解析：
    - 按分隔符拆开（换行/+++ / + / 逗号等），逐段用 _parse_rule_token_simple 解析
    - 支持复合标签（如 "青/オレンジ-2000" → {"青": -2000, "オレンジ": -2000}）
    - 也尝试旧版 regex 模式作为补充
    """
    s = safe_to_text(val)
    if not s:
        return {}

    # 方法 A: _parse_adjust_rule_simple_all 逻辑
    t = s
    for rep in ("+++", "++", "+", "＋＋＋", "＋＋", "＋", "\r"):
        t = t.replace(rep, "\n")
    for sep in ("、", "，", ","):
        t = t.replace(sep, "\n")

    rules: dict[str, int] = {}
    for line in t.splitlines():
        parsed_list = _parse_rule_token_simple(line)
        # parsed_list 现在是 List[Tuple[str, int]]，支持复合标签
        for g, d in parsed_list:
            rules[g] = d

    # 方法 B: 旧版正则（fallback 补充）
    if not rules:
        parts = re.split(r"\+{1,3}|[,、，\s]+", s)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = re.match(r"(.+?)-(\d+)", p)
            if not m:
                continue
            group_raw = m.group(1).strip()
            amt = -int(m.group(2))
            # 支持复合标签
            labels = [lbl.strip() for lbl in LABEL_SPLIT_RE_shop2.split(group_raw) if lbl.strip()]
            for lbl in labels:
                rules[lbl] = amt

    return rules

# ----------------------------------------------------------------------
# Step 6: LLM + Guardrails 版规则提取
# ----------------------------------------------------------------------

# LLM 相关代码已提取到 shop_cleaners_split_llm/llm_shop2.py
from ..shop_cleaners_split_llm.llm_shop2 import (
    setup_shop2_llm_deps,
    extract_specs_shop2_llm as _extract_specs_shop2_llm_impl,
)

# 注入正则依赖到 LLM 模块
setup_shop2_llm_deps(_parse_rule_token_simple, _extract_specs_shop2_regex)


# ----------------------------------------------------------------------
# Step 10: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop2(shop2_df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    # shop2 特殊：列名小写化 + lenient 校验
    df = shop2_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["data2-1", "data2-2", "data3", "data5", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
        lenient=True,
    )
    if ctx is None:
        return early

    # 只保留 simfree 未開封
    df = df[df["data2-2"].apply(_is_target)].copy().reset_index(drop=True)
    if df.empty:
        return finalize_color_cleaner(ctx, [])

    ctx.logger.debug(
        "After filter",
        extra={
            "event_type": "cleaner_start",
            "log_seq": ctx.log_seq,
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "total_rows_after_filter": len(df),
        },
    )
    ctx.log_seq += 1

    out_rows: list[dict] = []

    for pos, row in enumerate(df.to_dict("records")):
        rec_raw = row.get("time-scraped")
        recorded_at = parse_dt_aware(rec_raw)

        raw_modelcap = _norm(row.get("data2-1"))
        raw_targetflag = row.get("data2-2")
        raw_price = row.get("data3")
        raw_rule = row.get("data5")

        if not raw_modelcap:
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="data2-1 empty")
            continue

        cap_gb = _parse_capacity_gb(raw_modelcap)
        if not cap_gb:
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="capacity_gb parse failed",
                         data2_1_raw=_truncate_for_log(raw_modelcap, 100))
            continue

        model_name = _normalize_model_generic(raw_modelcap)
        if not model_name:
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="model_name normalization failed",
                         data2_1_raw=_truncate_for_log(raw_modelcap, 100))
            continue

        key = (model_name, int(cap_gb))
        cmap = ctx.color_map.get(key)
        if not cmap:
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="no info match",
                         model_name=model_name, capacity_gb=cap_gb)
            continue

        base_price = extract_price_yen(raw_price)
        if base_price is None:
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="base_price parse failed",
                         data3_raw=_truncate_for_log(str(raw_price), 100))
            continue

        raw_rule_s = safe_to_text(raw_rule)
        decomp = dispatch_extraction_to_price_decomposition(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_shop2_regex(raw_rule),
            llm_fn=lambda: _extract_specs_shop2_llm_impl(raw_rule, row_index=pos),
            base_price=base_price,
            source_text_raw=raw_rule_s,
            result_adapter=lambda r: (list(r.items()), []),
        )

        cmap_filtered = {cn: (pn, cr) for cn, (pn, cr) in cmap.items() if pn}

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            cmap_filtered,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=recorded_at,
            emit_default_rows=True,
            skip_non_positive=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=pos,
            model_text=raw_modelcap,
            model_norm=model_name,
            capacity_gb=cap_gb,
        )
        out_rows.extend(new_rows)

    ctx.input_rows = len(shop2_df)
    return finalize_color_cleaner(ctx, out_rows)
