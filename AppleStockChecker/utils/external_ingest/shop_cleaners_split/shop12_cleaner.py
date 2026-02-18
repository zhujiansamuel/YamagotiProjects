from __future__ import annotations

"""
shop12 清洗器 — トゥインクル

  原始文本（備考1 + 買取価格）
    两阶段流水线（与 shop17/16/15/14 对齐）:
    ├─ _clean_color_text_1_shop12()            ← Step 1: 去除開封行，预处理備考1
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop12()                  ← NONE_RE / DELTA_RE(分支) / ABS_RE
    ├─ expand_match_tokens()
    ├─ 阶段 2  match_tokens_to_specs()
    ├─ _label_matches_color_unified()
    └─ clean_shop12()                           ← 主函数
"""

import logging
import re
from typing import List, Optional, Tuple

import pandas as pd

from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    normalize_text_basic,
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
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
    LABEL_SPLIT_RE_shop12,
    EXTRACTION_MODE,
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

CLEANER_NAME = "shop12"
SHOP_NAME = "トゥインクル"

# ----------------------------------------------------------------------
# Step 1: 備考1 文本预处理
# ----------------------------------------------------------------------

def _clean_color_text_1_shop12(remark_raw: str) -> str:
    """
    備考1 预处理（Step 1）：
    - 把与"開封/開封品/※開封/開封済"粘在同一行的内容拆行；
    - 去掉所有"開封"行，只保留可用于新品价规则的行。
    """
    if not remark_raw:
        return ""
    s = str(remark_raw)

    # 关键：把"※開封品"等前面强行插入换行（解决: Orange-2000円※開封品...）
    s = re.sub(r"(※\s*開封品|※\s*開封|開封品|開封済|開封)", r"\n\1", s)

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", s) if ln is not None and ln.strip()]
    keep: List[str] = []
    for ln in lines:
        if ("開封" in ln) or ("開封品" in ln) or ("※開封" in ln) or ("開封済" in ln):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()


def _clean_color_text_shop12(text: str) -> str:
    """清理 remark 片段，供阶段 1 匹配使用。"""
    if not text:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s, remove_newlines=False, collapse_spaces=False)


# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE，与 shop17/16 对齐）
# ----------------------------------------------------------------------

# 单级 split：整块文本按分隔符一次性拆成 parts（与 shop17 一致）
SPLIT_TOKENS_RE_shop12 = re.compile(r"[／/、，]|(?:\s*;\s*)|\n")

COLOR_NONE_RE_shop12 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop12 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop12 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

# 全色检测（前置步骤，与 shop14 一致）
_ALL_DELTA_RE_shop12 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop12 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop12(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop12(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop12(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop12):
        return False
    return True


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop12(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["モデルナンバー", "備考1", "買取価格", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    rows: List[dict] = []

    for idx, row in df.iterrows():
        base_price = extract_price_yen(row.get("買取価格"))
        if base_price is None:
            continue
        base_price = int(base_price)

        model_text = str(row.get("モデルナンバー") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            ctx.log_seq += 1
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=idx, skip_reason="model_or_cap_parse_failed", log_seq=ctx.log_seq,
                         model_text=model_text)
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            ctx.log_seq += 1
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=idx, skip_reason="no_info_key", log_seq=ctx.log_seq,
                         model_text=model_text, model_norm=model_norm, capacity_gb=cap_gb)
            continue

        remark_raw = row.get("備考1") or ""
        raw_remark_shop12 = _clean_color_text_1_shop12(normalize_text_stage0(str(remark_raw)))

        # 前置：all_delta 检测（全色±N）
        agg_all_delta: Optional[int] = None
        if raw_remark_shop12:
            agg_all_delta = detect_all_delta_unified(raw_remark_shop12, _ALL_DELTA_RE_shop12)

        # 阶段 1：_match_shop12
        tokens = match_tokens_generic(
            raw_remark_shop12,
            split_re=SPLIT_TOKENS_RE_shop12,
            none_re=COLOR_NONE_RE_shop12,
            abs_re=COLOR_ABS_RE_shop12,
            delta_re=COLOR_DELTA_RE_shop12,
            normalize_label_func=_normalize_label_shop12,
            is_plausible_label_func=_is_plausible_color_label_shop12,
            preprocessor=_clean_color_text_shop12,
        )

        # expand_match_tokens + 阶段 2
        tokens_exp = expand_match_tokens(
            tokens,
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

        # 若有 all_delta，前置到 delta_specs
        if agg_all_delta is not None:
            deltas = [("全色", agg_all_delta)] + [
                (lb, v) for lb, v in deltas if str(lb).strip() not in ("全色", "ALL")
            ]

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=str(remark_raw),
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp, color_map, _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
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
