from __future__ import annotations

"""
shop9 清洗器 — アキモバ

  原始文本（買取価格 + 色・詳細等）
    两阶段流水线（与 shop17/16/15/14/12/11 对齐）:
    ├─ _clean_color_text_shop9()                 ← 合并 買取価格 + 色・詳細等，归一化
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop9()              ← NONE_RE / DELTA_RE(分支) / ABS_RE
    ├─ expand_match_tokens()
    ├─ 阶段 2  match_tokens_to_specs()
    ├─ _label_matches_color_unified()
    └─ clean_shop9()                        ← 主函数

  [暂未启用] _direct_abs_overrides_for_row：按「alias + 金額」在原文中扫描覆盖 abs，
    已注释。若需恢复，取消注释主流程中的调用与 _merge_abs_overrides 即可。
"""

import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    normalize_text_basic,
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
    LABEL_SPLIT_RE_shop9,
    EXTRACTION_MODE,
    log_row_skip,
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

CLEANER_NAME = "shop9"
SHOP_NAME = "アキモバ"

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

ABS_LIKE_MIN = int(os.getenv("SHOP9_ABS_LIKE_MIN", "50000"))  # 绝对价量级阈值（_direct_abs_overrides 用，目前未启用）
COL_MODEL = "機種名"
COL_PRICE = "買取価格"
COL_COLOR = "色・詳細等"
COL_TIME = "time-scraped"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# 修饰词前缀：strip 后使 "未開 橙227,000" → "橙227,000"，供 DELTA 正确匹配
_SHOP9_MODIFIER_PREFIX_RE = re.compile(
    r"(?:未開|未開封|新品|新品未使用)\s*",
    re.UNICODE,
)


def _clean_color_text_shop9(s_price: str, s_color: str) -> str:
    """
    合并 買取価格 + 色・詳細等，归一化后供阶段 1 匹配。
    """
    parts = [s.strip() for s in (s_price or "", s_color or "") if s and str(s).strip()]
    if not parts:
        return ""
    return _clean_color_text_shop9_single(" ".join(parts))


def _clean_color_text_shop9_single(text: str) -> str:
    """单参数归一化，供 match_tokens_generic preprocessor 使用。"""
    if not text:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = _SHOP9_MODIFIER_PREFIX_RE.sub("", s).strip()
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s)


# ----------------------------------------------------------------------
# [暂未启用] _direct_abs_overrides：按「颜色别名 + 紧随金额」在色・詳細等中扫描，补充 abs。
# 作用：当主正则未覆盖「黒 193,500円」等格式时，按别名+金额兜底。
# 恢复时：1) 取消下方函数注释 2) 导入 SYNONYM_LOOKUP_NORM, _merge_abs_overrides
#         3) 取消主流程中 overrides 与 _merge_abs_overrides 的调用
# ----------------------------------------------------------------------
# def _extract_amount_after_alias(text: str, alias: str) -> Optional[int]:
#     """在 text 中查找 'alias 193,500' / 'alias193,500円' 等，返回 alias 后紧跟的数字。"""
#     if not text or not alias:
#         return None
#     pat = re.compile(rf"{re.escape(alias)}\s*(?:¥|￥)?\s*([0-9０-９][0-9０-９,，]*)")
#     m = pat.search(str(text))
#     if not m:
#         return None
#     from ..cleaner_tools import _normalize_amount_text
#     return _normalize_amount_text(m.group(1))
#
# def _direct_abs_overrides_for_row(raw_color_text: str, color_to_pn: Dict[str, str]) -> Dict[str, int]:
#     """按每个颜色的别名扫描 raw_color_text，得到 {color_norm: amount}。仅接受 >= ABS_LIKE_MIN 的金额。"""
#     overrides: Dict[str, int] = {}
#     if not raw_color_text:
#         return overrides
#     s = str(raw_color_text)
#     for col_norm in color_to_pn.keys():
#         aliases = {col_norm} | {str(syn).strip() for syn in SYNONYM_LOOKUP_NORM.get(col_norm, [])}
#         for alias in aliases:
#             if not alias:
#                 continue
#             val = _extract_amount_after_alias(s, alias)
#             if val is not None and val >= ABS_LIKE_MIN:
#                 overrides[col_norm] = int(val)
#                 break
#     return overrides
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE，与 shop17/16/14/12/11 对齐）
# ----------------------------------------------------------------------

SPLIT_TOKENS_RE_shop9 = re.compile(r"[／/、，]|(?:\s*;\s*)|;|；|\n")

COLOR_NONE_RE_shop9 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop9 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop9 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop9 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop9 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop9(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop9(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop9(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop9):
        return False
    return True


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop9(
    df: pd.DataFrame,
    debug: bool = True,
    debug_limit: int = 30,
) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=[COL_MODEL, COL_PRICE, COL_COLOR, COL_TIME],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    model_norm_ser = df[COL_MODEL].map(_normalize_model_generic)
    cap_gb_ser = df[COL_MODEL].map(_parse_capacity_gb)
    recorded_at_ser = df[COL_TIME].map(lambda x: parse_dt_aware(x))

    rows: List[dict] = []

    for i in range(len(df)):
        raw_model = df[COL_MODEL].iat[i]
        m = model_norm_ser.iat[i]
        c = cap_gb_ser.iat[i]
        t = recorded_at_ser.iat[i]
        raw_price_cell = df[COL_PRICE].iat[i]
        raw_color_cell = df[COL_COLOR].iat[i]

        if not m or pd.isna(c):
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="model_or_cap_missing", log_seq=ctx.log_seq,
                         raw_model=str(raw_model), model_norm=str(m))
            ctx.log_seq += 1
            continue
        c = int(c)

        key = (m, c)
        color_to_pn = ctx.color_map.get(key)
        if not color_to_pn:
            log_row_skip(ctx.logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="no_pn_map", log_seq=ctx.log_seq,
                         model_norm=str(m), capacity_gb=c)
            ctx.log_seq += 1
            continue

        raw_price_shop9 = normalize_text_stage0(str(raw_price_cell) if raw_price_cell is not None else "")
        raw_color_shop9 = normalize_text_stage0(str(raw_color_cell) if raw_color_cell is not None else "")
        raw_combined_shop9 = " ".join(
            [s.strip() for s in (raw_price_shop9, raw_color_shop9) if s and str(s).strip()]
        )

        base_price = extract_price_yen(raw_price_shop9) or extract_price_yen(raw_color_shop9)

        # 前置：all_delta 检测（全色±N），优先从合并文本检测
        agg_all_delta: Optional[int] = None
        if raw_combined_shop9:
            agg_all_delta = detect_all_delta_unified(raw_combined_shop9, _ALL_DELTA_RE_shop9)
        if agg_all_delta is None and raw_color_shop9:
            agg_all_delta = detect_all_delta_unified(raw_color_shop9, _ALL_DELTA_RE_shop9)  # 回退：仅色・詳細等

        # 阶段 1：统一流程 raw + preprocessor
        tokens = match_tokens_generic(
            raw_combined_shop9,
            split_re=SPLIT_TOKENS_RE_shop9,
            none_re=COLOR_NONE_RE_shop9,
            abs_re=COLOR_ABS_RE_shop9,
            delta_re=COLOR_DELTA_RE_shop9,
            normalize_label_func=_normalize_label_shop9,
            is_plausible_label_func=_is_plausible_color_label_shop9,
            preprocessor=_clean_color_text_shop9_single,
        )

        # expand_match_tokens + 阶段 2
        tokens_exp = expand_match_tokens(
            tokens,
            color_to_pn,
            _label_matches_color_unified,
            enable_adaptive=True,
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
        )
        deltas, abs_specs = match_tokens_to_specs(
            tokens_exp,
            context={"base_price": base_price, "has_base_price": base_price is not None},
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
            row_index=i,
        )

        # 若有 all_delta，前置到 delta_specs
        if agg_all_delta is not None:
            deltas = [("全色", agg_all_delta)] + [
                (lb, v) for lb, v in deltas if str(lb).strip() not in ("全色", "ALL")
            ]

        # [暂未启用] _direct_abs_overrides：按「颜色别名+紧随金额」在色・詳細等中扫描，补充 abs。
        # 若需恢复：overrides = _direct_abs_overrides_for_row(raw_color_text=raw_color_shop9, color_to_pn=color_to_pn)
        #            if overrides: abs_specs = _merge_abs_overrides(abs_specs, overrides)
        # overrides = _direct_abs_overrides_for_row(raw_color_text=raw_color_shop9, color_to_pn=color_to_pn)
        # if overrides:
        #     abs_specs = _merge_abs_overrides(abs_specs, overrides)

        decomp = PriceDecomposition(
            base_price=base_price or 0,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=f"{raw_price_shop9} | {raw_color_shop9}" if raw_price_shop9 and raw_color_shop9 else (raw_price_shop9 or raw_color_shop9),
        )
        decomp_emit_default = base_price is not None

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_to_pn,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=t,
            emit_default_rows=decomp_emit_default,
            skip_non_positive=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=i,
            model_text=str(raw_model),
            model_norm=str(m),
            capacity_gb=c,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
