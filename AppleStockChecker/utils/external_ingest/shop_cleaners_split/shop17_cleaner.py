from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    normalize_text_basic,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    LABEL_SPLIT_RE_shop17 as SPLIT_TOKENS_RE_shop17,
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
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from datetime import datetime
import pytz
import time
import textwrap
import logging



"""
shop17 清洗器 — ゲストモバイル

  原始文本（type / 新未開封品 / 色減額）
    │ 两阶段流水线：Match → expand_match_tokens → match_tokens_to_specs
    │ SHOP17_ADAPTIVE_SPLIT (环境变量，默认 true)
    │
    ├─ _normalize_model_generic() / _parse_capacity_gb()  ← Step 1: 机型・容量解析（cleaner_tools）
    │
    ├─ extract_price_yen()         ← Step 2: 基础价提取（cleaner_tools）
    │
    ├─ 阶段 1: _match_shop17()               ← NONE_RE / DELTA_RE(分支) / ABS_RE
    │   ├─ _pick_unopened_section()         ← 提取【未開封】段
    │   ├─ _normalize_color_text_shop17()   ← 归一化
    │   └─ 输出 MatchToken[]（format_hint: signed|sep_minus|after_yen|plain_digits|colon_prefix|none）
    │
    ├─ expand_match_tokens()                 ← 自适应分割（阶段 1 与 2 之间）
    │
    └─ match_tokens_to_specs()               ← 阶段 2 语义映射 + 边界规则 → (deltas, abs_specs)
    ├─ _label_matches_color_unified()  ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop17()              ← Step 5: 主函数，生成输出行

  自适应分割 (shop17 试点功能):
    - 环境变量: SHOP17_ADAPTIVE_SPLIT=true/false
    - 默认启用，支持复合标签如 "青/オレンジ-2000"
    - 日志事件: composite_label_split, composite_label_full_match, no_match
    - 详见: docs/composite_label_split_proposal.md
"""

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop17"
SHOP_NAME = "ゲストモバイル"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# 自适应分割开关（shop17 试点）
ENABLE_ADAPTIVE_SPLIT_SHOP17 = os.getenv("SHOP17_ADAPTIVE_SPLIT", "true").lower() == "true"

# ----------------------------------------------------------------------
# 正则表达式与辅助函数（按处理流程排列）
# ----------------------------------------------------------------------

# ── Step 1: 提取【未開封】段落 ──
def _pick_unopened_section(text: str) -> str:
    """若包含【未開封】…，取该段直到下一个 '【' 或行末；否则返回原文。"""
    if not text:
        return ""
    s = str(text)
    m = re.search(r"【\s*未開封\s*】(.*?)(?=【|$)", s, flags=re.DOTALL)
    return m.group(1) if m else s

# ── Step 2: 归一化色減額文本 ──
def _normalize_color_text_shop17(s: str) -> str:
    """
    统一色減額文本里的全角数字/逗号/各种 dash，顺便清理空白。
    使用通用规范化函数（全角→半角）。
    保留换行与空白结构（remove_newlines=False, collapse_spaces=False），
    以便 SPLIT_TOKENS_RE 能按 \\n 正确切分多段。
    """
    if s is None:
        return ""
    # 色減額 split 前保留换行，否则「ブルー-1000」与「△減額なし」会合并到同一 part
    return normalize_text_basic(
        str(s), remove_newlines=False, collapse_spaces=False
    )

# ── Step 3: 归一化颜色标签（清除空白） ──
def _normalize_label_shop17(lbl: str) -> str:
    return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

# ── Step 4: 验证颜色标签合理性 ──
_BAD_LABEL_WORDS_shop17 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")

def _is_plausible_color_label_shop17(label: str) -> bool:
    """过滤掉明显不是"颜色名"的 label（比如 利用制限△ / 保証開始3か月未満 等）。"""
    label = _normalize_label_shop17(label)
    if not label:
        return False
    if label.startswith(("△", "▲")):
        return False
    if re.search(r"\d", label):
        return False
    if len(label) > 16:
        return False
    if any(w in label for w in _BAD_LABEL_WORDS_shop17):
        return False
    return True

# ── Step 5: 分割多颜色条目 ──
# SPLIT_TOKENS_RE_shop17: 从 cleaner_tools.LABEL_SPLIT_RE_shop17 导入

# ── Step 6: 匹配无减额颜色（なし模式） ──
COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# ── Step 7: 匹配有金额减额的颜色 ──
# label 排除数字，避免 "ブルー229,000円" 中金额被吃进 label
COLOR_DELTA_RE_shop17 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# ── Step 8: 匹配绝对价（label￥amount） ──
COLOR_ABS_RE_shop17 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*￥\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

# ----------------------------------------------------------------------
# 阶段 1：匹配（输出 MatchToken，不含自适应分割）
# NONE_RE / DELTA_RE(分支→signed|sep_minus|colon_prefix|plain_digits) / ABS_RE
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop17 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

def _match_shop17(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从色減額文本中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE / ABS_RE 三正则，按 sep/sign 分支设置 format_hint。
    支持 pending_labels（「赤、青 -500」等多标签共用一个金额）。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = _normalize_color_text_shop17(_pick_unopened_section(str(text)))
    if "色減額" in s:
        s = s.split("色減額", 1)[-1].lstrip(":：")

    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop17.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    pending_labels: List[str] = []
    position = 0

    for part in parts:
        m0 = COLOR_NONE_RE_shop17.search(part)
        if m0:
            label_raw = _normalize_label_shop17(m0.group("label"))
            if _is_plausible_color_label_shop17(label_raw):
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
        for m in COLOR_ABS_RE_shop17.finditer(part):
            has_amount_in_part = True
            label_raw = _normalize_label_shop17(m.group("label"))
            if not _is_plausible_color_label_shop17(label_raw):
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
        for m in COLOR_DELTA_RE_shop17.finditer(part):
            has_delta_in_part = True
            label_raw = _normalize_label_shop17(m.group("label"))
            if not _is_plausible_color_label_shop17(label_raw):
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
                pl_norm = _normalize_label_shop17(pl)
                if pl_norm and _is_plausible_color_label_shop17(pl_norm):
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
        for tok in SPLIT_TOKENS_RE_shop17.split(part):
            tok = _normalize_label_shop17(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------
def clean_shop17(df: pd.DataFrame) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["type", "新未開封品", "色減額", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    rows: List[dict] = []

    for idx, row in df.iterrows():
        model_text = str(row.get("type") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            continue

        base_price = extract_price_yen(row.get("新未開封品"))
        if base_price is None:
            continue
        base_price = int(base_price)

        raw_color = row.get("色減額")
        raw_color_s = "" if raw_color is None else str(raw_color)

        tokens = _match_shop17(raw_color_s)
        tokens = expand_match_tokens(
            tokens,
            color_map,
            _label_matches_color_unified,
            enable_adaptive=ENABLE_ADAPTIVE_SPLIT_SHOP17,
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
        )
        deltas, abs_specs = match_tokens_to_specs(
            tokens,
            context={"base_price": base_price, "has_base_price": True},
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
            row_index=int(idx),
        )

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=raw_color_s,
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

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
