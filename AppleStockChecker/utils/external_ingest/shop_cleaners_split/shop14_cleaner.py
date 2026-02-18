"""
shop14_cleaner  —  買取楽園

数据处理流程（两阶段流水线，与 shop15/16/17 对齐）:
  raw DataFrame
    ├─ Step 1  列校验 & remark列解析
    ├─ Step 2  行级过滤（未開封 + model/cap/color_map 匹配）
    ├─ Step 3  base_price 提取
    ├─ Step 4  remark文本归一化（3列合并）
    ├─ 前置  all_delta 检测（全色±N）→ 若有则单独分支，与 per-color 合并时 per-color 优先
    ├─ 阶段 1  对每个 frag 跑 _match_shop14()，合并 tokens
    ├─ expand_match_tokens()
    ├─ 阶段 2  match_tokens_to_specs()（阈值与 shop15/16/17 对齐）
    └─ resolve_color_prices()
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    detect_all_delta_unified,
    match_tokens_generic,
    normalize_text_basic,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    coerce_amount_yen,
    extract_price_yen,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    LABEL_SPLIT_RE_shop14,
    EXTRACTION_MODE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop14"
SHOP_NAME = "買取楽園"

# ---------------------------------------------------------------------------
# Step 2: 文本归一化 helpers
# ---------------------------------------------------------------------------

_norm = _norm_strip


def _norm_label(lbl: str) -> str:
    """去除空白并统一全角空格/NBSP，保留原文字顺序用作匹配用 key"""
    if lbl is None:
        return ""
    s = str(lbl)
    s = s.strip().replace("\u3000", " ").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_colname(x) -> str:
    s = str(x or "")
    s = s.lstrip("\ufeff")
    s = s.replace("\u3000", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# _coerce_amount_yen → cleaner_tools.coerce_amount_yen 统一导入
_coerce_amount_yen = coerce_amount_yen


# ---------------------------------------------------------------------------
# Step 3: 正则模式定义（NONE_RE + DELTA_RE + ABS_RE，与 shop15/16/17 对齐）
# ---------------------------------------------------------------------------

# 不含半角逗号，避免 "229,000円" 千位分隔符被误分割
SPLIT_TOKENS_RE_shop14 = re.compile(r"\s*(?:、|，|／|/|;|；)\s*")

COLOR_NONE_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# label 排除数字
COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop14 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*￥\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

# 全色检测（前置步骤）
_ALL_DELTA_RE_shop14 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop14 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _clean_label_shop14(lbl: str) -> str:
    """归一化标签，去除空白与分隔符。"""
    if not lbl:
        return ""
    s = str(lbl).replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[／/、，,;；\s]+", "", s)
    s = re.sub(r"[／/、，,;；\s]+$", "", s)
    return s.strip()


def _is_plausible_color_label_shop14(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _clean_label_shop14(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop14):
        return False
    return True


# ---------------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ---------------------------------------------------------------------------
# 原 shop14 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ---------------------------------------------------------------------------
# Step 5: remark列解析
# ---------------------------------------------------------------------------

def _resolve_remark_cols(df: "pd.DataFrame") -> Dict[str, Optional[str]]:
    want = ["减价条件", "减价条件2", "23432"]
    norm_map = {_norm_colname(c): c for c in df.columns}

    resolved: Dict[str, Optional[str]] = {w: None for w in want}
    for w in want:
        nw = _norm_colname(w)
        if nw in norm_map:
            resolved[w] = norm_map[nw]
            continue
        for nc, ac in norm_map.items():
            if nw and (nw in nc):
                resolved[w] = ac
                break
    return resolved


# ---------------------------------------------------------------------------
# Step 6-7: 不能内移（regex/llm/clean 共用）— 紧贴 regex 组上方
# ---------------------------------------------------------------------------

def _clean_color_text_shop14(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.lstrip("\ufeff").replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s, remove_newlines=False, collapse_spaces=False)


# ---------------------------------------------------------------------------
# 主清洗函数
# ---------------------------------------------------------------------------

def clean_shop14(df: "pd.DataFrame", debug: bool = True) -> "pd.DataFrame":
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["name", "data6", "price2", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    remark_cols_map = _resolve_remark_cols(df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        status = str(row.get("data6") or "")
        if "未開封" not in status:
            continue

        model_text = str(row.get("name") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None:
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            continue

        base_price = extract_price_yen(row.get("price2"))
        if base_price is None:
            continue
        base_price = int(base_price)

        rec_at = parse_dt_aware(row.get("time-scraped"))

        raw_frags_shop14: Dict[str, str] = {}
        for logical in ("减价条件", "减价条件2", "23432"):
            actual = remark_cols_map.get(logical)
            raw_val = row.get(actual) if actual else None
            raw_frags_shop14[logical] = normalize_text_stage0(str(raw_val or ""))

        raw_combined_shop14 = " ".join([v for v in raw_frags_shop14.values() if v.strip()]).strip()

        # 前置：all_delta 检测（全色±N），任一 frag 或 combined 有则采用（后者覆盖）
        agg_all_delta: Optional[int] = None
        for frag in raw_frags_shop14.values():
            if not frag.strip():
                continue
            ad = detect_all_delta_unified(frag, _ALL_DELTA_RE_shop14)
            if ad is not None:
                agg_all_delta = ad
        if raw_combined_shop14:
            ad2 = detect_all_delta_unified(raw_combined_shop14, _ALL_DELTA_RE_shop14)
            if ad2 is not None:
                agg_all_delta = ad2

        # 阶段 1：对每个 frag 跑 _match_shop14，合并 tokens
        all_tokens: List[MatchToken] = []
        for frag in raw_frags_shop14.values():
            if frag.strip():
                all_tokens.extend(match_tokens_generic(
                    frag,
                    split_re=SPLIT_TOKENS_RE_shop14,
                    none_re=COLOR_NONE_RE_shop14,
                    abs_re=COLOR_ABS_RE_shop14,
                    delta_re=COLOR_DELTA_RE_shop14,
                    normalize_label_func=_clean_label_shop14,
                    is_plausible_label_func=_is_plausible_color_label_shop14,
                    preprocessor=_clean_color_text_shop14,
                ))
        if not all_tokens and raw_combined_shop14:
            all_tokens = match_tokens_generic(
                raw_combined_shop14,
                split_re=SPLIT_TOKENS_RE_shop14,
                none_re=COLOR_NONE_RE_shop14,
                abs_re=COLOR_ABS_RE_shop14,
                delta_re=COLOR_DELTA_RE_shop14,
                normalize_label_func=_clean_label_shop14,
                is_plausible_label_func=_is_plausible_color_label_shop14,
                preprocessor=_clean_color_text_shop14,
            )

        # expand + 阶段 2
        tokens_exp = expand_match_tokens(
            all_tokens,
            color_map,
            _label_matches_color_unified,
            enable_adaptive=True,
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
        )
        deltas, abs_specs = match_tokens_to_specs(
            tokens_exp,
            context={"base_price": base_price, "has_base_price": True},
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
            row_index=int(idx),
        )

        # 若有 all_delta，前置到 delta_specs；per-color 在后，resolve_color_prices 中会优先覆盖
        if agg_all_delta is not None:
            deltas = [("全色", agg_all_delta)] + [
                (lb, v) for lb, v in deltas if str(lb).strip() not in ("全色", "ALL")
            ]

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=raw_combined_shop14,
        )

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
