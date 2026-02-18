from __future__ import annotations

"""
shop3 清洗器 — 買取一丁目

  原始文本（title / data5 / 减价1）
    - 纯正则实现（无 LLM）
    两阶段流水线（与 shop17/16/15/14/12/11/9/7 对齐）:
    ├─ _normalize_model_generic()          ← Step 1: 机型归一化（cleaner_tools）
    ├─ _parse_capacity_gb()                ← Step 2: 容量解析（cleaner_tools）
    ├─ extract_price_yen()                ← Step 3: 基础价提取（cleaner_tools）
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop3()              ← NONE_RE / DELTA_RE / ABS_RE
    ├─ expand_match_tokens()
    ├─ 阶段 2  match_tokens_to_specs()
    └─ resolve_color_prices → 输出
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    normalize_text_basic,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    LABEL_SPLIT_RE_shop3 as LABEL_SPLIT_RE,
    setup_color_cleaner,
    finalize_color_cleaner,
    coerce_amount_yen,
    detect_all_delta_unified,
    match_tokens_generic,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop3"
SHOP_NAME = "買取一丁目"

# ----------------------------------------------------------------------
# 辅助工具
# ----------------------------------------------------------------------

_norm = _norm_strip

# ----------------------------------------------------------------------
# 文本预处理
# ----------------------------------------------------------------------

def _clean_color_text_shop3(text: str) -> str:
    """清理 减价1 文本。"""
    if not text:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s)

# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE）
# ----------------------------------------------------------------------

SPLIT_TOKENS_RE_shop3 = re.compile(r"[／/、，,・]|(?:\s*[;；]\s*)|\n")

COLOR_NONE_RE_shop3 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# 买取一丁目格式：标签 + sign + 金额
COLOR_DELTA_RE_shop3 = re.compile(
    r"""(?P<label>[^+\-−－\d¥￥円\/、，\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop3 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop3 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop3 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop3(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop3(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop3(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop3):
        return False
    return True


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop3(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["title", "data5", "time-scraped"],
        extraction_mode="regex",
    )
    if ctx is None:
        return early

    src = df.copy()
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok].reset_index(drop=True)
    if src.empty:
        return finalize_color_cleaner(ctx, [])

    model_norm = src["title"].map(_normalize_model_generic)
    cap_gb = src["title"].map(_parse_capacity_gb)

    try:
        base_price = src["data5"].map(extract_price_yen)
    except Exception:
        base_price = src["data5"].map(to_int_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    remark = src["减价1"] if "减价1" in src.columns else None

    rows: List[dict] = []

    for i in range(len(src)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p0 = base_price.iat[i]
        t = recorded_at.iat[i]
        model_text = str(src["title"].iat[i])

        raw_rem_shop3 = normalize_text_stage0(str(remark.iat[i]) if remark is not None else "")

        if not m or pd.isna(c) or p0 is None:
            continue

        key = (m, int(c))
        cmap = ctx.color_map.get(key)
        if not cmap:
            continue

        base_price_val = int(p0)
        source_text_raw_full = raw_rem_shop3

        delta_specs: List[Tuple[str, int]] = []
        abs_specs: List[Tuple[str, int]] = []

        if raw_rem_shop3:
            agg_all_delta = detect_all_delta_unified(raw_rem_shop3, _ALL_DELTA_RE_shop3)
            tokens = match_tokens_generic(
                raw_rem_shop3,
                split_re=SPLIT_TOKENS_RE_shop3,
                none_re=COLOR_NONE_RE_shop3,
                abs_re=COLOR_ABS_RE_shop3,
                delta_re=COLOR_DELTA_RE_shop3,
                normalize_label_func=_normalize_label_shop3,
                is_plausible_label_func=_is_plausible_color_label_shop3,
                preprocessor=_clean_color_text_shop3,
            )
            tokens_exp = expand_match_tokens(
                tokens,
                cmap,
                _label_matches_color_unified,
                enable_adaptive=True,
                logger=ctx.logger,
                cleaner_name=CLEANER_NAME,
                shop_name=SHOP_NAME,
            )
            delta_specs, abs_specs = match_tokens_to_specs(
                tokens_exp,
                context={"base_price": base_price_val, "has_base_price": True},
                logger=ctx.logger,
                cleaner_name=CLEANER_NAME,
                shop_name=SHOP_NAME,
                row_index=i,
            )
            if agg_all_delta is not None:
                delta_specs = [("全色", agg_all_delta)] + [
                    (lb, v) for lb, v in delta_specs if str(lb).strip() not in ("全色", "ALL")
                ]

        extraction_method = "regex" if (delta_specs or abs_specs) else "none"

        decomp = PriceDecomposition(
            base_price=base_price_val,
            delta_specs=delta_specs,
            abs_specs=abs_specs,
            extraction_method=extraction_method,
            source_text_raw=source_text_raw_full,
        )
        new_rows, ctx.log_seq = resolve_color_prices(
            decomp, cmap, _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=t,
            skip_non_positive=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=m,
            capacity_gb=int(c),
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
