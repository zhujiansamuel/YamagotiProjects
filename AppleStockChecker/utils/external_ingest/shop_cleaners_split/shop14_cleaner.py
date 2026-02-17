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
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    coerce_amount_yen,
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


def _match_shop14(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从 remark  fragment 中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE(分支) / ABS_RE，不包含全色（全色由前置步骤处理）。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = _clean_remark_frag(str(text))
    if not s:
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop14.split(s) if p and p.strip()]
    if not parts:
        parts = [s]

    pending_labels: List[str] = []
    position = 0

    for part in parts:
        m0 = COLOR_NONE_RE_shop14.search(part)
        if m0:
            label_raw = _clean_label_shop14(m0.group("label"))
            if _is_plausible_color_label_shop14(label_raw):
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
        for m in COLOR_ABS_RE_shop14.finditer(part):
            has_amount_in_part = True
            label_raw = _clean_label_shop14(m.group("label"))
            if not _is_plausible_color_label_shop14(label_raw):
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
        for m in COLOR_DELTA_RE_shop14.finditer(part):
            has_delta_in_part = True
            label_raw = _clean_label_shop14(m.group("label"))
            if not _is_plausible_color_label_shop14(label_raw):
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
                pl_clean = _clean_label_shop14(pl)
                if pl_clean and _is_plausible_color_label_shop14(pl_clean):
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

        for tok in LABEL_SPLIT_RE_shop14.split(part):
            tok = _clean_label_shop14(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


def _detect_all_delta(text: str) -> Optional[int]:
    """前置步骤：检测全色统一减额。"""
    s = _clean_remark_frag(text)
    if not s:
        return None
    m = _ALL_DELTA_RE_shop14.search(s)
    if m:
        return _coerce_amount_yen(m.group(0).replace("全色", "").strip()) or 0
    if "全色" in s:
        return 0
    return None

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

def _clean_remark_frag(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.lstrip("\ufeff").replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


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

        base_price = to_int_yen(row.get("price2"))
        if base_price is None:
            continue
        base_price = int(base_price)

        rec_at = parse_dt_aware(row.get("time-scraped"))

        frags: Dict[str, str] = {}
        for logical in ("减价条件", "减价条件2", "23432"):
            actual = remark_cols_map.get(logical)
            raw_val = row.get(actual) if actual else None
            frags[logical] = _clean_remark_frag(raw_val)

        combined = " ".join([v for v in frags.values() if v]).strip()

        # 前置：all_delta 检测（全色±N），任一 frag 或 combined 有则采用（后者覆盖）
        agg_all_delta: Optional[int] = None
        for frag in frags.values():
            if not frag:
                continue
            ad = _detect_all_delta(frag)
            if ad is not None:
                agg_all_delta = ad
        if combined:
            ad2 = _detect_all_delta(combined)
            if ad2 is not None:
                agg_all_delta = ad2

        # 阶段 1：对每个 frag 跑 _match_shop14，合并 tokens
        all_tokens: List[MatchToken] = []
        for frag in frags.values():
            if frag:
                all_tokens.extend(_match_shop14(frag))
        if not all_tokens and combined:
            all_tokens = _match_shop14(combined)

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
            source_text_raw=combined,
        )

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
