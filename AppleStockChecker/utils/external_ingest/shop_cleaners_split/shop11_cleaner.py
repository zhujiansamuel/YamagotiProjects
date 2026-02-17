from __future__ import annotations

"""
shop11 清洗器 — モバステ

  原始文本（storage_name / price_unopened / caution_empty）
    两阶段流水线（与 shop17/16/15/14/12 对齐）:
    ├─ normalize_text_basic() + _clean_caution_frag()  ← 预处理 caution_empty
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop11()            ← NONE_RE / DELTA_RE(分支) / ABS_RE
    ├─ expand_match_tokens()
    ├─ 阶段 2  match_tokens_to_specs()   ← format_hint 优先级去重（与 12/14/15/16/17 一致）
    ├─ _label_matches_color_unified()
    └─ clean_shop11()                     ← 主函数
"""

import logging
import re
from typing import List, Optional

import pandas as pd

from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
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
    LABEL_SPLIT_RE_shop11,
    EXTRACTION_MODE,
    setup_color_cleaner,
    finalize_color_cleaner,
    coerce_amount_yen,
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


def _clean_caution_frag(text: str) -> str:
    """
    清理 caution_empty 片段：去掉括号内备注、全角归一化。
    """
    if not text:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s)


# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE，与 shop17/16/14/12 对齐）
# ----------------------------------------------------------------------

SPLIT_TOKENS_RE_shop11 = re.compile(r"[／/、，]|(?:\s*;\s*)|\n")

COLOR_NONE_RE_shop11 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop11 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop11 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop11 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop11 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop11(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop11(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop11(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop11):
        return False
    return True


# ----------------------------------------------------------------------
# 阶段 1：匹配（输出 MatchToken，含 pending_labels）
# ----------------------------------------------------------------------

def _match_shop11(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从 caution_empty 文本中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE(分支) / ABS_RE，支持 pending_labels。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = _clean_caution_frag(text)
    if not s:
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop11.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    pending_labels: List[str] = []
    position = 0

    for part in parts:
        m0 = COLOR_NONE_RE_shop11.search(part)
        if m0:
            label_raw = _normalize_label_shop11(m0.group("label"))
            if _is_plausible_color_label_shop11(label_raw):
                tokens.append(MatchToken(
                    label=label_raw,
                    amount_int=0,
                    format_hint=FORMAT_HINT_NONE,
                    position=position,
                ))
                position += 1
            pending_labels = []
            continue

        has_amount_in_part = False
        for m in COLOR_ABS_RE_shop11.finditer(part):
            has_amount_in_part = True
            label_raw = _normalize_label_shop11(m.group("label"))
            if not _is_plausible_color_label_shop11(label_raw):
                continue
            amt = to_int_yen(m.group("amount"))
            if amt is None:
                continue
            tokens.append(MatchToken(
                label=label_raw,
                amount_int=int(amt),
                format_hint=FORMAT_HINT_AFTER_YEN,
                position=position,
            ))
            position += 1
        if has_amount_in_part:
            pending_labels = []
            continue

        has_delta_in_part = False
        for m in COLOR_DELTA_RE_shop11.finditer(part):
            has_delta_in_part = True
            label_raw = _normalize_label_shop11(m.group("label"))
            if not _is_plausible_color_label_shop11(label_raw):
                continue
            sep = m.group("sep")
            sign = m.group("sign")
            amt = to_int_yen(m.group("amount"))
            if amt is None:
                continue
            amt_val = int(amt)
            if sign:
                negative = sign in ("-", "−", "－")
                amount_int = -amt_val if negative else amt_val
                hint = FORMAT_HINT_SIGNED
            elif sep and sep in ("-", "−", "－"):
                amount_int = -amt_val
                hint = FORMAT_HINT_SEP_MINUS
            elif sep and sep in ("：", ":"):
                amount_int = amt_val
                hint = FORMAT_HINT_COLON_PREFIX
            else:
                amount_int = amt_val
                hint = FORMAT_HINT_PLAIN_DIGITS

            tok = MatchToken(label=label_raw, amount_int=amount_int, format_hint=hint, position=position)
            tokens.append(tok)
            position += 1
            for pl in pending_labels:
                pl_norm = _normalize_label_shop11(pl)
                if pl_norm and _is_plausible_color_label_shop11(pl_norm):
                    tokens.append(MatchToken(
                        label=pl_norm,
                        amount_int=amount_int,
                        format_hint=hint,
                        position=position,
                    ))
                    position += 1
            pending_labels = []
        if has_delta_in_part:
            continue

        # 仅标签无金额：挂起等待下一 part
        for tok in LABEL_SPLIT_RE_shop11.split(part):
            tok = _normalize_label_shop11(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


def _detect_all_delta(text: str) -> Optional[int]:
    """前置步骤：检测全色统一减额。"""
    s = _clean_caution_frag(text)
    if not s:
        return None
    m = _ALL_DELTA_RE_shop11.search(s)
    if m:
        return coerce_amount_yen(m.group(0).replace("全色", "").strip()) or 0
    if "全色" in s:
        return 0
    return None


# ----------------------------------------------------------------------
# 清洗主函数
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

        rec_at = parse_dt_aware(time_raw)

        caution_raw_s = str(caution_raw or "")

        # 前置：all_delta 检测（全色±N）
        agg_all_delta: Optional[int] = None
        if caution_raw_s:
            agg_all_delta = _detect_all_delta(caution_raw_s)

        # 阶段 1：_match_shop11（内部会 _clean_caution_frag）
        tokens = _match_shop11(caution_raw_s)

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
            row_index=int(i),
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
            source_text_raw=str(caution_raw or ""),
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
