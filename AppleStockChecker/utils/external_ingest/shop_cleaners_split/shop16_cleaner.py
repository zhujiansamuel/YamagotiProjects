from __future__ import annotations

"""
shop16 清洗器 — 携帯空間

  原始文本（買取価格列）
    │ 两阶段流水线：Match → expand_match_tokens → match_tokens_to_specs
    │
    ├─ _normalize_price_text_shop16()     ← Step 1: 归一化（换行→/、压缩空白）
    │
    ├─ _extract_base_price_shop16()       ← Step 2: 提取基础价
    │
    ├─ 阶段 1: _match_shop16()            ← NONE_RE / DELTA_RE(分支) / ABS_RE
    │
    ├─ expand_match_tokens()              ← 自适应分割（阶段 1 与 2 之间）
    │
    └─ match_tokens_to_specs()            ← 阶段 2 语义映射 + 边界规则 → (deltas, abs_specs)
    ├─ _label_matches_color_unified()     ← 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop16()                     ← 主函数，生成输出行

  自适应分割 (与 shop17 同策略):
    - 默认启用，支持复合标签如 "青/オレンジ -5000"
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    normalize_text_stage0,
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    normalize_text_basic,
    coerce_amount_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    LABEL_SPLIT_RE_shop16 as SPLIT_TOKENS_RE,
    LABEL_SPLIT_RE_shop16_SIMPLE,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    detect_all_delta_unified,
    match_tokens_generic,
    EXTRACTION_MODE,
)

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop16"
SHOP_NAME = "携帯空間"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置 (EXTRACTION_MODE 见 cleaner_tools)
# ----------------------------------------------------------------------

MODEL_COL = "iPhone 17 Pro Max"
DESC_COL  = "説明1"
PRICE_COL = "買取価格"

# ----------------------------------------------------------------------
# Step 1-3: 常量与 _norm
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")
_BASE_ONLY_RE = re.compile(r"^\s*(?:￥|\¥)?\s*\d[\d,]*\s*(?:円)?\s*$")
_TRAILING_AMOUNT_IN_LABEL_RE = re.compile(
    r"(?:[：:])?\s*(?:￥)?\s*[+\-−－]?\s*\d[\d,]*\s*(?:円)?\s*$",
    re.UNICODE,
)

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop16 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则模式定义（NONE_RE + DELTA_RE + ABS_RE，与 shop17 三正则模式一致）
# ----------------------------------------------------------------------

COLOR_NONE_RE_shop16 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# label 排除数字，避免 "ブルー229,000円" 中金额被吃进 label
COLOR_DELTA_RE_shop16 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop16 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*￥\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_ALL_DELTA_RE_shop16 = re.compile(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?")

_GROUP_SHARED_DELTA_RE = re.compile(
    r"""
    (?P<labels>[^0-9￥円]+?)          # 多颜色标签段（含 /）
    \s*(?P<sign>[+\-−－])\s*         # 显式正负号
    (?P<amount>\d[\d,]*)\s*(?:円)?   # 金额（可无 円）
    """,
    re.UNICODE | re.VERBOSE
)

# 过滤非颜色标签（参考 shop17）
_BAD_LABEL_WORDS_shop16 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _is_plausible_color_label_shop16(label: str) -> bool:
    """过滤明显非颜色名的 label。"""
    label = _normalize_label_shop16(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop16):
        return False
    return True


# ----------------------------------------------------------------------
# 辅助：归一化与基础价提取
# ----------------------------------------------------------------------

def _clean_color_text_shop16(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    # 把换行变成分隔（保留"下一行是颜色差价"的结构）
    s = re.sub(r"[\r\n]+", " / ", s)
    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    # 多个分隔合并
    s = re.sub(r"(?:\s*/\s*){2,}", " / ", s).strip()
    return s


def _preprocess_color_text_shop16(text: str) -> str:
    """归一化 + 去掉行首基准价，供 match_tokens_generic preprocessor 使用。"""
    s = _clean_color_text_shop16(text)
    m0 = FIRST_YEN_RE.search(s)
    return s[m0.end():].strip() if m0 else s


def _extract_base_price(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)  # 兜底
    return to_int_yen(m.group(1))


def _is_base_only_price_text(price_text_norm: str) -> bool:
    """判断文本是否只包含一个基础价，不含任何颜色差价信息。"""
    return bool(_BASE_ONLY_RE.match(price_text_norm or ""))


def _normalize_label_shop16(lbl: str) -> str:
    s = re.sub(r"[\s\u3000\xa0]+", "", lbl or "")
    s = re.sub(r"(カラー|色)$", "", s)
    # 去掉黏在 label 末尾的金额/符号：-1000 / ￥86100 / :-1,000円 等
    s = _TRAILING_AMOUNT_IN_LABEL_RE.sub("", s)
    return s.strip()


def _split_labels_shop16(lbl: str) -> List[str]:
    # 兼容 "青/オレンジ""黒、白""blue/black" 等
    raw = _normalize_label_shop16(lbl)
    parts = LABEL_SPLIT_RE_shop16_SIMPLE.split(raw)
    return [p for p in (_normalize_label_shop16(x) for x in parts) if p]


def _extract_shared_delta_map_shop16(price_text_norm: str) -> Dict[str, int]:
    """
    从原文中抽取： 'オレンジ/青 -1500' 这种共享差价 -> {オレンジ:-1500, 青:-1500}
    这是"纠错用"的确定性证据，不替代 LLM 抽取的主流程。
    """
    s = price_text_norm or ""
    out: Dict[str, int] = {}
    # 去掉基础价前缀，减少误匹配（基础价一般在最前）
    m0 = FIRST_YEN_RE.search(s)
    tail = s[m0.end():] if m0 else s

    for m in _GROUP_SHARED_DELTA_RE.finditer(tail):
        labels_raw = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        delta = -int(amt) if sign in ("-", "−", "－") else int(amt)

        # 拆分 labels（/、，等）
        for lb in LABEL_SPLIT_RE_shop16_SIMPLE.split(labels_raw):
            lb = _normalize_label_shop16(lb)
            if lb:
                out[lb] = delta
    return out


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop16(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=[MODEL_COL, DESC_COL, PRICE_COL, "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    rows: List[dict] = []

    for idx, row in df.iterrows():
        model_cell = str(row.get(MODEL_COL) or "").strip()
        desc_cell  = str(row.get(DESC_COL)  or "").strip()
        price_cell = row.get(PRICE_COL)
        rec_at     = parse_dt_aware(row.get("time-scraped"))

        is_unopened = ("未開封" in desc_cell) or ("未開封" in model_cell)
        if not is_unopened:
            continue

        model_text = model_cell.replace("\u3000", " ").replace("\xa0", " ").replace("\n", " ").strip()
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            continue

        raw_price_shop16 = normalize_text_stage0("" if price_cell is None else str(price_cell))
        
        # 1. Base Price（需先归一化再提取）
        price_text_norm = _clean_color_text_shop16(raw_price_shop16)
        base_price = _extract_base_price(price_text_norm)
        
        # 2. All Delta
        agg_all_delta = detect_all_delta_unified(raw_price_shop16, _ALL_DELTA_RE_shop16)
        
        # 3. Match Tokens（统一流程：raw + preprocessor）
        tokens = match_tokens_generic(
            raw_price_shop16,
            split_re=SPLIT_TOKENS_RE,
            none_re=COLOR_NONE_RE_shop16,
            abs_re=COLOR_ABS_RE_shop16,
            delta_re=COLOR_DELTA_RE_shop16,
            normalize_label_func=_normalize_label_shop16,
            is_plausible_label_func=_is_plausible_color_label_shop16,
            preprocessor=_preprocess_color_text_shop16,
        )
        
        tokens = expand_match_tokens(
            tokens,
            color_map,
            _label_matches_color_unified,
            enable_adaptive=True,
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
            row_index=int(idx),
        )

        if agg_all_delta is not None:
            deltas = [("全色", agg_all_delta)] + [
                (lb, v) for lb, v in deltas if str(lb).strip() not in ("全色", "ALL")
            ]

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=raw_price_shop16,
        )

        if decomp.base_price is None and not decomp.abs_specs:
            continue

        emit_default = decomp.base_price is not None

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=emit_default,
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
