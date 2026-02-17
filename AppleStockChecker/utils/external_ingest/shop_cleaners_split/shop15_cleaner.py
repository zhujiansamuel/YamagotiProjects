from __future__ import annotations

"""
shop15 清洗器 — 買取当番

  原始文本（price 列）
    │ 两阶段流水线：Match → expand_match_tokens → match_tokens_to_specs
    │ SHOP15_ADAPTIVE_SPLIT (环境变量，默认 true)
    │
    ├─ _extract_base_price_at_start()     ← Step 1: 提取基础价
    │
    ├─ 阶段 1: _match_shop15()            ← NONE_RE / DELTA_RE(分支) / ABS_RE
    │
    ├─ expand_match_tokens()              ← 自适应分割（阶段 1 与 2 之间）
    │
    └─ match_tokens_to_specs()            ← 阶段 2 语义映射 + 边界规则 → (deltas, abs_specs)
    └─ clean_shop15()                     ← 主函数，生成输出行
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    LABEL_SPLIT_RE_shop15 as SPLIT_TOKENS_RE_shop15,
    LABEL_SPLIT_RE_shop15 as _LABEL_LIST_SPLIT_RE_shop15,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    EXTRACTION_MODE,
)

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop15"
SHOP_NAME = "買取当番"

# 自适应分割开关（与 shop16/17 同策略，同时作用 delta 与 abs 链路）
ENABLE_ADAPTIVE_SPLIT_SHOP15 = os.getenv("SHOP15_ADAPTIVE_SPLIT", "true").lower() == "true"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

MODEL_COL = "data2"
PRICE_COL = "price"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop15 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则模式定义（NONE_RE + DELTA_RE + ABS_RE，与 shop16/17 一致）
# ----------------------------------------------------------------------

BASE_YEN_AT_START_RE_shop15 = re.compile(r"^\s*(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

COLOR_NONE_RE_shop15 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# label 排除数字，避免 "ブルー229,000円" 中金额被吃进 label
COLOR_DELTA_RE_shop15 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop15 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*￥\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_BAD_LABEL_WORDS_shop15 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _is_plausible_color_label_shop15(label: str) -> bool:
    """过滤明显非颜色名的 label。"""
    label = _clean_label_shop15(label)
    if not label or label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop15):
        return False
    return True


# ----------------------------------------------------------------------
# 阶段 1：匹配（输出 MatchToken）
# ----------------------------------------------------------------------

def _match_shop15(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从 price 文本中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE(分支) / ABS_RE 三正则，支持 pending_labels。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = str(text).replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()

    m0 = BASE_YEN_AT_START_RE_shop15.search(s)
    tail = s[m0.end():].strip() if m0 else s
    if not tail:
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop15.split(tail) if p and p.strip()]
    if not parts:
        parts = [tail]

    pending_labels: List[str] = []
    position = 0

    for part in parts:
        m0 = COLOR_NONE_RE_shop15.search(part)
        if m0:
            label_raw = _clean_label_shop15(m0.group("label"))
            if _is_plausible_color_label_shop15(label_raw):
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
        for m in COLOR_ABS_RE_shop15.finditer(part):
            has_amount_in_part = True
            label_raw = _clean_label_shop15(m.group("label"))
            if not _is_plausible_color_label_shop15(label_raw):
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
        for m in COLOR_DELTA_RE_shop15.finditer(part):
            has_delta_in_part = True
            label_raw = _clean_label_shop15(m.group("label"))
            if not _is_plausible_color_label_shop15(label_raw):
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

            tokens.append(MatchToken(
                label=label_raw,
                amount_int=amount_int,
                format_hint=hint,
                position=position,
            ))
            position += 1
            for pl in pending_labels:
                pl_clean = _clean_label_shop15(pl)
                if pl_clean and _is_plausible_color_label_shop15(pl_clean):
                    tokens.append(MatchToken(
                        label=pl_clean,
                        amount_int=amount_int,
                        format_hint=hint,
                        position=position,
                    ))
                    position += 1
            pending_labels = []
        if has_delta_in_part:
            continue

        for tok in _LABEL_LIST_SPLIT_RE_shop15.split(part):
            tok = _clean_label_shop15(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def _extract_base_price_at_start(text: object) -> Optional[int]:
    if text is None:
        return None
    s = str(text)
    m = BASE_YEN_AT_START_RE_shop15.search(s)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None


def _clean_label_shop15(label: str) -> str:
    if not label:
        return ""
    s = str(label).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # 去掉可能粘着的分隔符
    s = s.strip(" 　:：-‐‑–—/／、,，・")
    return s


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop15(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=[PRICE_COL, MODEL_COL, "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    rows: List[dict] = []

    for i, row in df.iterrows():
        model_text = str(row.get(MODEL_COL) or "").strip()
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

        price_text = row.get(PRICE_COL)
        price_text_s = "" if price_text is None else str(price_text)

        base_price = _extract_base_price_at_start(price_text_s)
        tokens = _match_shop15(price_text_s)
        tokens = expand_match_tokens(
            tokens,
            color_map,
            _label_matches_color_unified,
            enable_adaptive=ENABLE_ADAPTIVE_SPLIT_SHOP15,
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
        )
        deltas, abs_specs = match_tokens_to_specs(
            tokens,
            context={"base_price": base_price, "has_base_price": base_price is not None},
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
            row_index=int(i),
        )

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=price_text_s,
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=False,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=int(i),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
