from __future__ import annotations
from typing import List, Optional, Tuple
import logging
import re
import pandas as pd

from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    normalize_text_basic,
    safe_to_text,
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
    LABEL_SPLIT_RE_shop7,
    log_row_skip,
    setup_color_cleaner,
    finalize_color_cleaner,
    coerce_amount_yen,
    detect_all_delta_unified,
    match_tokens_generic,
)


"""
shop7 清洗器 — 買取ホムラ

  原始 DataFrame (data, data2, data3, time-scraped)
    - 纯正则实现（无 LLM）
    - 颜色减价来源：当前行下一行（data2）当下一行无价格时视为颜色行
    两阶段流水线（与 shop17/16/15/14/12/11/9 对齐）:
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop7()            ← NONE_RE / DELTA_RE(分支) / ABS_RE
    ├─ expand_match_tokens()
    ├─ 阶段 2  match_tokens_to_specs()
    └─ resolve_color_prices → 输出
"""

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop7"
SHOP_NAME = "買取ホムラ"

# ----------------------------------------------------------------------
# Step 2a: 机型归一化
# ----------------------------------------------------------------------

def _norm_model_for_shop7(s: Optional[str]) -> str:
    """
    shop7 的 model 字段宽松归一化：
      - 跳过纯数字行（shop7 数据中存在纯数字的价格/编号行混入 data2 列）
      - 其余交给公共 _normalize_model_generic 处理
    """
    if s is None:
        return ""
    txt = str(s).strip()
    if not txt:
        return ""
    # shop7 特有：data2 列可能混入纯数字行，提前排除
    if re.fullmatch(r'[\d\-\.\s]+', txt):
        return ""

    return _normalize_model_generic(txt)


# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE，与 shop17/16/14/12/11 对齐）
# ----------------------------------------------------------------------

def _clean_color_text_shop7(text: str) -> str:
    """清理颜色行文本。"""
    if not text:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s)


SPLIT_TOKENS_RE_shop7 = re.compile(r"[／/、，,]|(?:\s*;\s*)|\n")

COLOR_NONE_RE_shop7 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop7 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop7 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop7 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop7 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop7(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop7(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop7(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop7):
        return False
    return True


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop7(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["data", "data2", "data3", "time-scraped"],
        extraction_mode="regex",
    )
    if ctx is None:
        return early

    # time-scraped 为空的行排除
    rows_before = len(df)
    df = df.copy().reset_index(drop=True)
    mask_time_ok = df["time-scraped"].astype(str).str.strip().ne("") & df["time-scraped"].notna()
    df = df[mask_time_ok].reset_index(drop=True)

    if df.empty:
        return finalize_color_cleaner(ctx, [])

    model_norm_series = df["data2"].map(_norm_model_for_shop7)
    cap_gb_series = df["data2"].map(_parse_capacity_gb)
    price_series = df["data3"].map(extract_price_yen)
    recorded_at = df["time-scraped"].map(parse_dt_aware)

    rows: List[dict] = []
    n = len(df)

    for i in range(n):
        base_price = price_series.iat[i]
        if pd.isna(base_price):
            continue
        base_price = int(base_price)

        model_text = safe_to_text(df["data2"].iat[i]).strip()
        model_norm = model_norm_series.iat[i]
        c = cap_gb_series.iat[i]
        rec_at = recorded_at.iat[i]

        if not model_norm or pd.isna(c):
            ctx.log_seq += 1
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="model_or_cap_missing", log_seq=ctx.log_seq,
                         model_text=model_text, model_norm=model_norm or "",
                         capacity_gb=int(c) if pd.notna(c) else None)
            continue

        cap_gb = int(c)
        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            ctx.log_seq += 1
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="no_color_map", log_seq=ctx.log_seq,
                         model_text=model_text, model_norm=model_norm, capacity_gb=cap_gb)
            continue

        source_text_raw_full = ""
        delta_specs: List[Tuple[str, int]] = []
        abs_specs: List[Tuple[str, int]] = []
        j = i + 1
        if j < n:
            raw_nxt_shop7 = normalize_text_stage0(safe_to_text(df["data2"].iat[j]) or "").strip()
            nxt_price_cell = safe_to_text(df["data3"].iat[j]).strip()
            nxt_price_val = extract_price_yen(nxt_price_cell) if nxt_price_cell else None
            is_color_line = bool(raw_nxt_shop7) and (nxt_price_val is None)

            if is_color_line:
                source_text_raw_full = raw_nxt_shop7
                agg_all_delta: Optional[int] = None
                if raw_nxt_shop7:
                    agg_all_delta = detect_all_delta_unified(raw_nxt_shop7, _ALL_DELTA_RE_shop7)
                
                tokens = match_tokens_generic(
                    raw_nxt_shop7,
                    split_re=SPLIT_TOKENS_RE_shop7,
                    none_re=COLOR_NONE_RE_shop7,
                    abs_re=COLOR_ABS_RE_shop7,
                    delta_re=COLOR_DELTA_RE_shop7,
                    normalize_label_func=_normalize_label_shop7,
                    is_plausible_label_func=_is_plausible_color_label_shop7,
                    preprocessor=_clean_color_text_shop7,
                )
                tokens_exp = expand_match_tokens(
                    tokens,
                    color_map,
                    _label_matches_color_unified,
                    enable_adaptive=True,
                    logger=ctx.logger,
                    cleaner_name=CLEANER_NAME,
                    shop_name=SHOP_NAME,
                )
                delta_specs, abs_specs = match_tokens_to_specs(
                    tokens_exp,
                    context={"base_price": base_price, "has_base_price": True},
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
            base_price=base_price,
            delta_specs=delta_specs,
            abs_specs=abs_specs,
            extraction_method=extraction_method,
            source_text_raw=source_text_raw_full,
        )

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp, color_map, _label_matches_color_unified,
            shop_name=SHOP_NAME, cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            skip_non_positive=True,
            logger=ctx.logger, log_seq_start=ctx.log_seq,
            row_index=i, model_text=model_text,
            model_norm=model_norm, capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    ctx.input_rows = rows_before
    return finalize_color_cleaner(ctx, rows)


