from __future__ import annotations

"""
shop4 清洗器 — モバイルミックス

  原始 DataFrame（data / data11 列）
    - 纯正则实现（无 LLM）
    两阶段流水线（与 shop17/16/15/14/12/11/9/7 对齐）:
    ├─ _find_base_price()                    ← 回溯查找基准价
    ├─ _collect_block_segments()             ← 收集 block 内行/段（按 円/ 分割）
    ├─ 前置  all_delta 检測（全色±N）
    ├─ 前置  detect_color_only_filter()      ← 颜色限定モード検出
    │    检测 3 种模式: 括号 / のみ / 裸色名
    │    → 命中时跳过阶段 1~2，emit_default_rows=False
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
    normalize_text_stage0,
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
    detect_all_delta_unified,
    match_tokens_generic,
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

def _clean_color_text_shop4(text: str) -> str:
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

# ----------------------------------------------------------------------
# 颜色限定モード検出（のみ / 括号 / 裸色名）
# ----------------------------------------------------------------------

# 括号パターン: "シルバー(コズミックオレンジ-2,500円)"
_COLOR_PAREN_RE_shop4 = re.compile(
    r"([^(\d\s/、,;]+)\s*\(\s*(.+?)\s*\)",
    re.UNICODE,
)

# 括号内アイテム: "コズミックオレンジ-2,500円"
_PAREN_INNER_ITEM_RE_shop4 = re.compile(
    r"([^\d+\-,、\s]+)\s*([+\-])\s*(\d[\d,]*)\s*(?:円)?",
    re.UNICODE,
)

# のみ 接尾辞
_NOMI_SUFFIX_RE_shop4 = re.compile(r"のみ\s*$")

# 全色パターン除去（分析用）
_ALL_COLOR_REMOVAL_RE_shop4 = re.compile(r"全色\s*[+\-]?\s*\d[\d,]*\s*(?:円)?")

# 価格関連パターン検出（裸色名判定用）
_HAS_PRICE_INDICATOR_RE_shop4 = re.compile(r"\d|円|なし")


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


def _matches_any_color_shop4(
    label: str,
    color_to_pn: Dict[str, Tuple[str, str]],
    label_matcher,
) -> bool:
    """label が color_to_pn 内のいずれかの色に一致するか判定。"""
    for col_norm, (_, col_raw) in color_to_pn.items():
        if label_matcher(label, col_raw, col_norm):
            return True
    return False


def detect_color_only_filter(
    text: str,
    color_to_pn: Dict[str, Tuple[str, str]],
    label_matcher,
) -> Tuple[bool, List[Tuple[str, int, bool]]]:
    """
    颜色限定モードの前置検出。

    3 種のパターンを検出:
      1. 括号パターン: "シルバー(コズミックオレンジ-2,500円)"
      2. のみ接尾辞:   "シルバー/ディープブルーのみ"
      3. 裸色名:       "コズミックオレンジ" (価格情報なし、色名のみ)

    戻り値:
      (color_only_mode, specs)
      specs: [(label, delta, has_explicit_delta)]
        has_explicit_delta=True  → 括号内の明示的 delta（全色と重畳しない）
        has_explicit_delta=False → 裸色名/のみ（全色 delta と重畳する）
    """
    if not text or not text.strip():
        return False, []

    # normalize_text_basic で全角→半角統一
    s = normalize_text_basic(text)
    if not s:
        return False, []

    # 全色パターンを除去して残りを分析
    s_no_all = _ALL_COLOR_REMOVAL_RE_shop4.sub("", s).strip()
    s_no_all = re.sub(r"^\s*[/、,;]\s*|\s*[/、,;]\s*$", "", s_no_all).strip()

    if not s_no_all:
        return False, []

    specs: List[Tuple[str, int, bool]] = []

    # --- 1. 括号パターン ---
    paren_m = _COLOR_PAREN_RE_shop4.search(s_no_all)
    if paren_m:
        outer = _normalize_label_shop4(paren_m.group(1))
        inner = paren_m.group(2)

        if (outer
                and _is_plausible_color_label_shop4(outer)
                and _matches_any_color_shop4(outer, color_to_pn, label_matcher)):
            specs.append((outer, 0, False))

        for im in _PAREN_INNER_ITEM_RE_shop4.finditer(inner):
            lbl = _normalize_label_shop4(im.group(1))
            sign = im.group(2)
            amt = int(im.group(3).replace(",", ""))
            delta = -amt if sign == "-" else amt
            if (lbl
                    and _is_plausible_color_label_shop4(lbl)
                    and _matches_any_color_shop4(lbl, color_to_pn, label_matcher)):
                specs.append((lbl, delta, True))

        if specs:
            return True, specs

    # --- 2. のみ接尾辞 ---
    if "のみ" in s_no_all:
        parts = [p.strip() for p in SPLIT_TOKENS_RE_shop4.split(s_no_all)
                 if p and p.strip()]
        for p in parts:
            lbl = _NOMI_SUFFIX_RE_shop4.sub("", p).strip()
            lbl = _normalize_label_shop4(lbl)
            if (lbl
                    and _is_plausible_color_label_shop4(lbl)
                    and _matches_any_color_shop4(lbl, color_to_pn, label_matcher)):
                specs.append((lbl, 0, False))
        if specs:
            return True, specs

    # --- 3. 裸色名 ---
    if _HAS_PRICE_INDICATOR_RE_shop4.search(s_no_all):
        return False, []

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop4.split(s_no_all)
             if p and p.strip()]
    if not parts:
        return False, []

    for p in parts:
        lbl = _normalize_label_shop4(p)
        if not lbl or not _is_plausible_color_label_shop4(lbl):
            return False, []
        if not _matches_any_color_shop4(lbl, color_to_pn, label_matcher):
            return False, []
        specs.append((lbl, 0, False))

    return True, specs


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
        raw_combined_shop4 = normalize_text_stage0(" / ".join(segments))
        block_lines_raw = []
        for j in range(i, block_end + 1):
            if j > i and _is_next_model_base_price_row(df, j, n):
                break
            raw = str(df["data"].iat[j]) if df["data"].iat[j] is not None else ""
            if raw.strip():
                block_lines_raw.append(raw.strip())
        source_text_raw_full = " | ".join(block_lines_raw)

        agg_all_delta: Optional[int] = None
        if raw_combined_shop4:
            agg_all_delta = detect_all_delta_unified(raw_combined_shop4, _ALL_DELTA_RE_shop4)

        # ── 颜色限定モード検出 (Step 3.5) ─────────────────────────
        color_only_mode, color_only_specs = detect_color_only_filter(
            raw_combined_shop4, color_to_pn, _label_matches_color_unified,
        )

        if color_only_mode:
            # 全色 delta との重畳処理:
            #   has_explicit_delta=True  → 括号内の明示値をそのまま使用
            #   has_explicit_delta=False → 全色 delta を重畳（なければ 0）
            co_delta_specs: List[Tuple[str, int]] = []
            for lbl, delta, has_explicit in color_only_specs:
                if has_explicit:
                    co_delta_specs.append((lbl, delta))
                else:
                    effective = agg_all_delta if agg_all_delta is not None else delta
                    co_delta_specs.append((lbl, effective))

            decomp = PriceDecomposition(
                base_price=base_price,
                delta_specs=co_delta_specs,
                abs_specs=[],
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
                emit_default_rows=False,
                skip_non_positive=True,
                logger=ctx.logger,
                log_seq_start=ctx.log_seq,
                row_index=i,
                model_text=model_text,
                model_norm=model_norm,
                capacity_gb=cap_gb,
            )
            rows.extend(new_rows)
            continue

        # ── 通常フロー: Stage 1 → resolve ────────────────────────
        tokens = match_tokens_generic(
            raw_combined_shop4,
            split_re=SPLIT_TOKENS_RE_shop4,
            none_re=COLOR_NONE_RE_shop4,
            abs_re=COLOR_ABS_RE_shop4,
            delta_re=COLOR_DELTA_RE_shop4,
            normalize_label_func=_normalize_label_shop4,
            is_plausible_label_func=_is_plausible_color_label_shop4,
            preprocessor=_clean_color_text_shop4,
        )
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
            skip_non_positive=True,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)


