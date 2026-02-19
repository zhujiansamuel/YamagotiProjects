from __future__ import annotations

"""
shop2 清洗器 — 海峡通信

  原始文本（data2-1 / data2-2 / data3 / data5）
    - 纯正则实现（无 LLM）
    两阶段流水线（与 shop17/16/15/14/12/11/9/7/3 对齐）:
    ├─ _is_target()                          ← Step 1: SIMfree+未開封 过滤
    ├─ extract_price_yen()                   ← Step 2: 基础价(data3)解析（cleaner_tools 统一）
    ├─ _normalize_model_generic()            ← Step 3: 机型规范化（cleaner_tools 统一）
    ├─ _parse_capacity_gb()                  ← Step 4: 容量解析
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop2()                 ← NONE_RE / DELTA_RE / ABS_RE
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
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _truncate_for_log,
    normalize_text_basic,
    _label_matches_color_unified,
    safe_to_text,
    PriceDecomposition,
    resolve_color_prices,
    setup_color_cleaner,
    finalize_color_cleaner,
    log_row_skip,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    LABEL_SPLIT_RE_shop2 as LABEL_SPLIT_RE,
    coerce_amount_yen,
    detect_all_delta_unified,
    match_tokens_generic,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop2"
SHOP_NAME = "海峡通信"

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
# 文本预处理
# ----------------------------------------------------------------------

def _clean_color_text_shop2(text: str) -> str:
    """清理 data5 规则文本。保留与旧版一致的 +++/、 等分隔符替换。"""
    if not text:
        return ""
    s = safe_to_text(text)
    if not s or s.lower() == "nan":
        return ""
    for rep in ("+++", "++", "+", "＋＋＋", "＋＋", "＋", "\r"):
        s = s.replace(rep, "\n")
    for sep in ("、", "，", ","):
        s = s.replace(sep, "\n")
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s)


# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE）
# ----------------------------------------------------------------------

SPLIT_TOKENS_RE_shop2 = re.compile(r"[／/、，,・\s]|(?:\s*[;；]\s*)|\n")

COLOR_NONE_RE_shop2 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop2 = re.compile(
    r"""(?P<label>[^+\-−－\d¥￥円\/、，\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop2 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop2 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop2 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop2(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop2(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop2(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop2):
        return False
    return True


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop2(shop2_df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    # shop2 特殊：列名小写化 + lenient 校验
    df = shop2_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["data2-1", "data2-2", "data3", "data5", "time-scraped"],
        extraction_mode="regex",
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

        raw_rule_shop2 = normalize_text_stage0(safe_to_text(raw_rule))
        base_price_val = int(base_price)

        delta_specs: List[Tuple[str, int]] = []
        abs_specs: List[Tuple[str, int]] = []

        if raw_rule_shop2:
            agg_all_delta = detect_all_delta_unified(raw_rule_shop2, _ALL_DELTA_RE_shop2)
            tokens = match_tokens_generic(
                raw_rule_shop2,
                split_re=SPLIT_TOKENS_RE_shop2,
                none_re=COLOR_NONE_RE_shop2,
                abs_re=COLOR_ABS_RE_shop2,
                delta_re=COLOR_DELTA_RE_shop2,
                normalize_label_func=_normalize_label_shop2,
                is_plausible_label_func=_is_plausible_color_label_shop2,
                preprocessor=_clean_color_text_shop2,
            )
            cmap_filtered = {cn: (pn, cr) for cn, (pn, cr) in cmap.items() if pn}
            tokens_exp = expand_match_tokens(
                tokens,
                cmap_filtered,
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
                row_index=pos,
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
            source_text_raw=raw_rule_shop2,
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
