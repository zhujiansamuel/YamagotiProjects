from __future__ import annotations

"""
shop4 清洗器 — モバイルミックス

  原始 DataFrame（data / data11 列）
    - 纯正则实现（无 LLM）
    两阶段流水线（与 shop17/16/15/14/12/11/9/7 对齐）:
    ├─ _find_base_price()                    ← 回溯查找基准价
    ├─ _collect_block_segments()             ← 收集 block 内行/段（按 円/ 分割）
    ├─ 前置  all_delta 检测（全色±N）
    ├─ 阶段 1  _match_shop4()                ← NONE_RE / DELTA_RE(分支) / ABS_RE
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
    PriceDecomposition,
    resolve_color_prices,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    normalize_text_basic,
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
    LABEL_SPLIT_RE_shop4 as LABEL_SPLIT_RE,
    EXTRACTION_MODE,
    setup_color_cleaner,
    finalize_color_cleaner,
    coerce_amount_yen,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop4"
SHOP_NAME = "モバイルミックス"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# ----------------------------------------------------------------------
# Step 1: 全角→半角 & 金额归一化
# ----------------------------------------------------------------------
# LABEL_SPLIT_RE: 从 cleaner_tools.LABEL_SPLIT_RE_shop4 导入

# ----------------------------------------------------------------------
# 基准价回溯查找
# ----------------------------------------------------------------------

def _find_base_price(df: pd.DataFrame, idx: int) -> Optional[int]:
    """
    按规范：机种行(data11非空)的上一行 data 是基准价。
    若上一行取不到，向上最多回溯 3 行找首个含"円"的金额。
    """
    for j in range(idx - 1, max(-1, idx - 4), -1):
        if j < 0:
            break
        s = str(df["data"].iat[j]) if "data" in df.columns else ""
        if s and ("円" in s or re.search(r"\d[\d,]*", s)):
            price = to_int_yen(s)
            if price:
                return int(price)
    return None

# ----------------------------------------------------------------------
# 纯金额行判断
# ----------------------------------------------------------------------

_PURE_PRICE_CHARS = re.compile(r"[０-９0-9,，\s円+\-−－]")

def _is_pure_price_only_row(df: pd.DataFrame, idx: int) -> bool:
    """
    判断该行 data 是否仅为纯金额（无颜色标签）。
    若仅为金额（如 "230,500円"）且下一行是机型行，则属于下一 block 的基准价，不应纳入当前 block。
    """
    if idx < 0 or "data" not in df.columns or idx >= len(df):
        return False
    line = str(df["data"].iat[idx]) if df["data"].iat[idx] is not None else ""
    stripped = line.strip()
    if not stripped:
        return False
    # 移除价格相关字符后若为空，则为纯金额
    remains = _PURE_PRICE_CHARS.sub("", stripped)
    if remains:
        return False
    return to_int_yen(line) is not None


def _is_next_model_base_price_row(df: pd.DataFrame, idx: int, n: int) -> bool:
    """
    判断该行是否为下一机型的基准价行。
    条件：纯金额行 + 下一行 data11 非空（下一机型行）。
    """
    if idx < 0 or idx >= n - 1:
        return False
    if not _is_pure_price_only_row(df, idx):
        return False
    val = df["data11"].iat[idx + 1] if "data11" in df.columns else None
    return val is not None and str(val).strip() != ""


# ----------------------------------------------------------------------
# block 内行/段收集（按 円/ 分割）
# ----------------------------------------------------------------------
_SHOP4_LINE_SPLIT_BY_YEN_SLASH = re.compile(r"円\s*[／/]\s*")


def _collect_block_segments(df: pd.DataFrame, start_idx: int) -> List[str]:
    """
    逐行扫描 block，按 円/ 分割，收集段列表供阶段 1 匹配。
    """
    segments: List[str] = []
    n = len(df)
    for j in range(start_idx, n):
        nxt_model = ""
        if "data11" in df.columns:
            val = df["data11"].iat[j]
            nxt_model = str(val) if val is not None else ""
        if j > start_idx and nxt_model.strip():
            break
        if j > start_idx and _is_next_model_base_price_row(df, j, n):
            break

        line = ""
        if "data" in df.columns:
            val = df["data"].iat[j]
            line = str(val) if val is not None else ""

        for seg in _SHOP4_LINE_SPLIT_BY_YEN_SLASH.split(line):
            seg = seg.strip()
            if seg:
                segments.append(seg)
    return segments


# ----------------------------------------------------------------------
# 正则模式（NONE_RE + DELTA_RE + ABS_RE，两阶段）
# ----------------------------------------------------------------------

def _clean_block_text(text: str) -> str:
    """清理 block 文本。"""
    if not text:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return normalize_text_basic(s)


SPLIT_TOKENS_RE_shop4 = re.compile(r"[／/、，]|(?:\s*;\s*)|\n")

COLOR_NONE_RE_shop4 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop4 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop4 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop4 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_BAD_LABEL_WORDS_shop4 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _normalize_label_shop4(lbl: str) -> str:
    """归一化颜色标签。"""
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


def _is_plausible_color_label_shop4(label: str) -> bool:
    """过滤非颜色标签。全色由前置步骤处理，此处排除。"""
    label = _normalize_label_shop4(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop4):
        return False
    return True


# ----------------------------------------------------------------------
# 阶段 1：匹配（输出 MatchToken，含 pending_labels）
# ----------------------------------------------------------------------

def _match_shop4(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从 block 合并文本中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE(分支) / ABS_RE，支持 pending_labels。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = _clean_block_text(text)
    if not s:
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop4.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    pending_labels: List[str] = []
    position = 0

    for part in parts:
        m0 = COLOR_NONE_RE_shop4.search(part)
        if m0:
            label_raw = _normalize_label_shop4(m0.group("label"))
            if _is_plausible_color_label_shop4(label_raw):
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
        for m in COLOR_ABS_RE_shop4.finditer(part):
            has_amount_in_part = True
            label_raw = _normalize_label_shop4(m.group("label"))
            if not _is_plausible_color_label_shop4(label_raw):
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
        for m in COLOR_DELTA_RE_shop4.finditer(part):
            has_delta_in_part = True
            label_raw = _normalize_label_shop4(m.group("label"))
            if not _is_plausible_color_label_shop4(label_raw):
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
                pl_norm = _normalize_label_shop4(pl)
                if pl_norm and _is_plausible_color_label_shop4(pl_norm):
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

        for tok in LABEL_SPLIT_RE.split(part):
            tok = _normalize_label_shop4(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


def _detect_all_delta(text: str) -> Optional[int]:
    """前置步骤：检测全色统一减额。"""
    s = _clean_block_text(text)
    if not s:
        return None
    m = _ALL_DELTA_RE_shop4.search(s)
    if m:
        return coerce_amount_yen(m.group(0).replace("全色", "").strip()) or 0
    if "全色" in s or s.strip() in ("全色", "全 色"):
        return 0
    return None

# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop4(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=["data", "data11", "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    def _next_model_idx(start: int) -> int:
        nn = len(df)
        for k in range(start + 1, nn):
            s = str(df["data11"].iat[k]) if df["data11"].iat[k] is not None else ""
            if s.strip():
                return k
        return nn

    rows: List[dict] = []
    n = len(df)

    for i in range(n):
        model_text = str(df["data11"].iat[i]) if df["data11"].iat[i] is not None else ""
        model_text = model_text.strip()
        if not model_text:
            continue

        block_end = _next_model_idx(i) - 1
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        rec_at = parse_dt_aware(df["time-scraped"].iat[i])

        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_to_pn = ctx.color_map.get(key)
        if not color_to_pn:
            continue

        base_price = _find_base_price(df, i)
        if base_price is None:
            continue

        segments = _collect_block_segments(df, i)
        combined_text = " / ".join(segments)
        block_lines_raw = []
        for j in range(i, block_end + 1):
            if j > i and _is_next_model_base_price_row(df, j, n):
                break
            raw = str(df["data"].iat[j]) if df["data"].iat[j] is not None else ""
            if raw.strip():
                block_lines_raw.append(raw.strip())
        source_text_raw_full = " | ".join(block_lines_raw)

        agg_all_delta: Optional[int] = None
        if combined_text:
            agg_all_delta = _detect_all_delta(combined_text)

        tokens = _match_shop4(combined_text)
        tokens_exp = expand_match_tokens(
            tokens,
            color_to_pn,
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

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=delta_specs,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=source_text_raw_full,
        )

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_to_pn,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)


