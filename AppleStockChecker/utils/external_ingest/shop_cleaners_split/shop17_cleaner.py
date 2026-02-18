from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    _parse_capacity_gb,
    _normalize_model_generic,
    normalize_text_basic,
    coerce_amount_yen,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    LABEL_SPLIT_RE_shop17 as SPLIT_TOKENS_RE_shop17,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    detect_all_delta_unified,
    match_tokens_generic,
    EXTRACTION_MODE,
)
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from datetime import datetime
import pytz
import time
import textwrap
import logging



"""
shop17 清洗器 — ゲストモバイル

  原始文本（type / 新未開封品 / 色減額）
    │ 两阶段流水线：Match → expand_match_tokens → match_tokens_to_specs
    │
    ├─ _normalize_model_generic() / _parse_capacity_gb()  ← Step 1: 机型・容量解析（cleaner_tools）
    │
    ├─ extract_price_yen()         ← Step 2: 基础价提取（cleaner_tools）
    │
    ├─ 阶段 1: _match_shop17()               ← NONE_RE / DELTA_RE(分支) / ABS_RE
    │   ├─ _pick_unopened_section()         ← 提取【未開封】段
    │   ├─ _normalize_color_text_shop17()   ← 归一化
    │   └─ 输出 MatchToken[]（format_hint: signed|sep_minus|after_yen|plain_digits|colon_prefix|none）
    │
    ├─ expand_match_tokens()                 ← 自适应分割（阶段 1 与 2 之间）
    │
    └─ match_tokens_to_specs()               ← 阶段 2 语义映射 + 边界规则 → (deltas, abs_specs)
    ├─ _label_matches_color_unified()  ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop17()              ← Step 5: 主函数，生成输出行

  自适应分割 (shop17 试点功能):
    - 默认启用，支持复合标签如 "青/オレンジ-2000"
    - 日志事件: composite_label_split, composite_label_full_match, no_match
    - 详见: docs/composite_label_split_proposal.md
"""

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop17"
SHOP_NAME = "ゲストモバイル"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 正则表达式与辅助函数（按处理流程排列）
# ----------------------------------------------------------------------

# ── Step 1: 提取【未開封】段落 ──
def _pick_unopened_section(text: str) -> str:
    """若包含【未開封】…，取该段直到下一个 '【' 或行末；否则返回原文。"""
    if not text:
        return ""
    s = str(text)
    m = re.search(r"【\s*未開封\s*】(.*?)(?=【|$)", s, flags=re.DOTALL)
    return m.group(1) if m else s

# ── Step 2: 归一化色減額文本 ──
def _clean_color_text_shop17(s: str) -> str:
    """
    统一色減額文本里的全角数字/逗号/各种 dash，顺便清理空白。
    使用通用规范化函数（全角→半角）。
    保留换行与空白结构（remove_newlines=False, collapse_spaces=False），
    以便 SPLIT_TOKENS_RE 能按 \\n 正确切分多段。
    """
    if s is None:
        return ""
    # 色減額 split 前保留换行，否则「ブルー-1000」与「△減額なし」会合并到同一 part
    return normalize_text_basic(
        str(s), remove_newlines=False, collapse_spaces=False
    )


def _preprocess_color_text_shop17(raw: str) -> str:
    """提取【未開封】段 + 归一化 + 色減額 切分，供 match_tokens_generic preprocessor 使用。"""
    s = _clean_color_text_shop17(_pick_unopened_section(raw))
    if "色減額" in s:
        s = s.split("色減額", 1)[-1].lstrip(":：")
    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s or ""):
        return ""
    return s

# ── Step 3: 归一化颜色标签（清除空白） ──
def _normalize_label_shop17(lbl: str) -> str:
    return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

# ── Step 4: 验证颜色标签合理性 ──
_BAD_LABEL_WORDS_shop17 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")

def _is_plausible_color_label_shop17(label: str) -> bool:
    """过滤掉明显不是"颜色名"的 label（比如 利用制限△ / 保証開始3か月未満 等）。"""
    label = _normalize_label_shop17(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")):
        return False
    if re.search(r"\d", label):
        return False
    if len(label) > 16:
        return False
    if any(w in label for w in _BAD_LABEL_WORDS_shop17):
        return False
    return True

# ── Step 5: 分割多颜色条目 ──
# SPLIT_TOKENS_RE_shop17: 从 cleaner_tools.LABEL_SPLIT_RE_shop17 导入

# ── Step 6: 匹配无减额颜色（なし模式） ──
COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# ── Step 7: 匹配有金额减额的颜色 ──
# label 排除数字，避免 "ブルー229,000円" 中金额被吃进 label
COLOR_DELTA_RE_shop17 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# ── Step 8: 匹配绝对价（label￥amount） ──
COLOR_ABS_RE_shop17 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*￥\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop17 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

# ----------------------------------------------------------------------
# 阶段 1：匹配（输出 MatchToken，不含自适应分割）
# NONE_RE / DELTA_RE(分支→signed|sep_minus|colon_prefix|plain_digits) / ABS_RE
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop17 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------
def clean_shop17(df: pd.DataFrame) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["type", "新未開封品", "色減額", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    rows: List[dict] = []

    for idx, row in df.iterrows():
        model_text = str(row.get("type") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            continue

        base_price = extract_price_yen(row.get("新未開封品"))
        if base_price is None:
            continue
        base_price = int(base_price)

        raw_color = row.get("色減額")
        raw_color_shop17 = normalize_text_stage0("" if raw_color is None else str(raw_color))

        # 1. All Delta (on raw text)
        agg_all_delta = detect_all_delta_unified(raw_color_shop17, _ALL_DELTA_RE_shop17)

        # 2. Match Tokens（统一流程：raw + preprocessor）
        tokens = match_tokens_generic(
            raw_color_shop17,
            split_re=SPLIT_TOKENS_RE_shop17,
            none_re=COLOR_NONE_RE_shop17,
            abs_re=COLOR_ABS_RE_shop17,
            delta_re=COLOR_DELTA_RE_shop17,
            normalize_label_func=_normalize_label_shop17,
            is_plausible_label_func=_is_plausible_color_label_shop17,
            preprocessor=_preprocess_color_text_shop17,
        )
        
        tokens = expand_match_tokens(
            tokens,
            color_map,
            _label_matches_color_unified,
            enable_adaptive=True,
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
        )
        deltas, abs_specs = match_tokens_to_specs(
            tokens,
            context={"base_price": base_price, "has_base_price": True},
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
            row_index=int(idx),
        )

        if agg_all_delta is not None:
            deltas = [("全色", agg_all_delta)] + [
                (lb, v) for lb, v in deltas if str(lb).strip() not in ("全色", "ALL")
            ]

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=raw_color_shop17,
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            skip_non_positive=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
