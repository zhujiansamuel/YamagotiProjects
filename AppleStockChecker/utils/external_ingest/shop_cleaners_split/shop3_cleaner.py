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

# 买取一丁目格式：标签 + sign + 金额（与 DELTA_PATTERN_STRICT/LOOSE 对齐）
COLOR_DELTA_RE_shop3 = re.compile(
    r"""(?P<label>[^+\-−－\d¥￥円\/、，\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# 宽松 fallback：允许日文等字符
COLOR_DELTA_RE_shop3_LOOSE = re.compile(
    r"""(?P<label>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)\s*
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
# 阶段 1：匹配（输出 MatchToken）
# ----------------------------------------------------------------------

def _match_shop3(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从 减价1 文本中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE(STRICT→LOOSE) / ABS_RE，支持 pending_labels。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = _clean_color_text_shop3(text)
    if not s:
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop3.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    pending_labels: List[str] = []
    position = 0

    def _try_delta_patterns(part: str) -> bool:
        nonlocal position
        for pat in (COLOR_DELTA_RE_shop3, COLOR_DELTA_RE_shop3_LOOSE):
            for m in pat.finditer(part):
                label_raw = _normalize_label_shop3(m.group("label"))
                if not _is_plausible_color_label_shop3(label_raw):
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
                    pl_norm = _normalize_label_shop3(pl)
                    if pl_norm and _is_plausible_color_label_shop3(pl_norm):
                        tokens.append(MatchToken(
                            label=pl_norm,
                            amount_int=amount_int,
                            format_hint=hint,
                            position=position,
                        ))
                        position += 1
                pending_labels.clear()
                return True
        return False

    for part in parts:
        m0 = COLOR_NONE_RE_shop3.search(part)
        if m0:
            label_raw = _normalize_label_shop3(m0.group("label"))
            if _is_plausible_color_label_shop3(label_raw):
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
        for m in COLOR_ABS_RE_shop3.finditer(part):
            has_amount_in_part = True
            label_raw = _normalize_label_shop3(m.group("label"))
            if not _is_plausible_color_label_shop3(label_raw):
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

        if _try_delta_patterns(part):
            continue

        for tok in LABEL_SPLIT_RE.split(part):
            tok = _normalize_label_shop3(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


def _detect_all_delta(text: str) -> Optional[int]:
    """前置步骤：检测全色统一减额。"""
    s = _clean_color_text_shop3(text)
    if not s:
        return None
    m = _ALL_DELTA_RE_shop3.search(s)
    if m:
        return coerce_amount_yen(m.group(0).replace("全色", "").strip()) or 0
    if "全色" in s:
        return 0
    return None


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

        rem_text = str(remark.iat[i]) if remark is not None else ""

        if not m or pd.isna(c) or p0 is None:
            continue

        key = (m, int(c))
        cmap = ctx.color_map.get(key)
        if not cmap:
            continue

        base_price_val = int(p0)
        source_text_raw_full = rem_text

        delta_specs: List[Tuple[str, int]] = []
        abs_specs: List[Tuple[str, int]] = []

        if rem_text:
            agg_all_delta = _detect_all_delta(rem_text)
            tokens = _match_shop3(rem_text)
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
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=m,
            capacity_gb=int(c),
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
