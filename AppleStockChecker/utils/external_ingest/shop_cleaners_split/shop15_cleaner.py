from __future__ import annotations

"""
shop15 清洗器 — 買取当番

  原始文本（price 列）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _extract_base_price_at_start()             ← Step 2: 提取基础价
    │
    ├─ dispatch_extraction_to_price_decomposition() ← Step 9: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop15_regex()       ← Step 6: 正则提取 specs
    │   │
    │   └─ llm 路径:
    │       ├─ _parse_shop15_price_via_langextract()     ← Step 7: LLM 核心提取
    │       └─ _coerce_specs / _augment_multi_label      ← Step 8: 纠错/增强
    │
    ├─ _build_color_prices_from_specs_shop15()     ← Step 10: specs → 最终价格
    │
    ├─ _label_matches_color_unified()               ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop15()                              ← Step 11: 主函数，生成输出行
"""

import logging
import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _truncate_for_log,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    LABEL_SPLIT_RE_shop15 as _LABEL_LIST_SPLIT_RE_shop15,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
    dispatch_extraction_to_price_decomposition,
    lx,
    HAS_LANGEXTRACT,
    log_llm_extraction_error,
)

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop15"
SHOP_NAME = "買取当番"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

# lx / HAS_LANGEXTRACT 从 cleaner_tools 统一导入

MODEL_COL = "data2"
PRICE_COL = "price"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）

# ----------------------------------------------------------------------
# Step 2: 基础价提取
# ----------------------------------------------------------------------

# 基准价只从开头抓（避免把"ブルー229,000円"的229,000误当 base）
_BASE_YEN_AT_START_RE = re.compile(r"^\s*(?:￥|¥|\u00a5)?\s*(\d[\d,]*)\s*円?")

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop15 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则模式定义
# ----------------------------------------------------------------------

BASE_YEN_AT_START_RE_shop15 = re.compile(r"^\s*(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

# 允许 label 里包含 "、/／・" 等，整体抓出来后再拆分
COLOR_ENTRY_RE_shop15 = re.compile(
    r"""(?P<label>[^\d円¥]+?)\s*             # 标签或标签列表（可含 、/／・）
        (?P<sep>[：:\-])? \s*                # 可选分隔符
        (?P<sign>[+\-−－])? \s*              # 可选正负号
        (?P<amount>\d[\d,]*) \s* (?:円)?     # 金额
    """,
    re.UNICODE | re.VERBOSE,
)

# 颜色列表 + 差额 的 block，例如:
# "オレンジ、ブルー-1000円", "シルバー、ブルー-3000円"
MULTI_LABEL_DELTA_BLOCK_RE_shop15 = re.compile(
    r"""
    (?P<label_blob>[^\d円¥]+?)     # 一个或多个颜色标签（可包含 、／/・ 等）
    \s*
    (?P<sign>[+\-−－])            # + 或 -
    \s*
    (?P<amount>\d[\d,]*)          # 金额
    \s*円?
    """,
    re.UNICODE | re.VERBOSE,
)

# ----------------------------------------------------------------------
# Step 6: 正则提取函数（不能内移：regex 与 llm 共用 — 紧贴 regex 组上方）
# ----------------------------------------------------------------------

def _parse_signed_int_yen(s: object) -> Optional[int]:
    """
    解析：'229,000' / '229,000円' / '-1000' / '-1,000円' / '+2000円'
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None

    # 统一符号
    t = t.replace("＋", "+").replace("−", "-").replace("－", "-")
    sign = 1
    if t.startswith("+"):
        t = t[1:].strip()
    elif t.startswith("-"):
        sign = -1
        t = t[1:].strip()

    # 去掉非数字/逗号
    t = re.sub(r"[^\d,]", "", t)
    if not t:
        return None
    try:
        return sign * int(t.replace(",", ""))
    except Exception:
        return None


def _extract_base_price_at_start(text: object) -> Optional[int]:
    if text is None:
        return None
    s = str(text)
    m = _BASE_YEN_AT_START_RE.search(s)
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


def _split_color_labels_shop15(label_blob: str) -> List[str]:
    if not label_blob:
        return []
    s = str(label_blob).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" 　:：-‐‑–—/／、,，・")
    parts = [p.strip() for p in _LABEL_LIST_SPLIT_RE_shop15.split(s) if p.strip()]
    return parts or [s]


def _extract_specs_shop15_regex(
    price_text: str,
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    """
    纯正则版：从 price_text 中提取 (base_price, specs)。
    specs = [(label, kind, value)]  kind ∈ {"delta", "abs"}
    """
    if not price_text:
        return None, []

    s = str(price_text).replace("\u3000", " ")
    base_price = _extract_base_price_at_start(s)

    # 跳过开头的基础价部分
    m0 = _BASE_YEN_AT_START_RE.search(s)
    tail = s[m0.end():] if m0 else s

    if not tail.strip():
        return base_price, []

    specs: List[Tuple[str, str, int]] = []

    for m in COLOR_ENTRY_RE_shop15.finditer(tail):
        label_blob = m.group("label") or ""
        sep = m.group("sep")
        sign = m.group("sign")
        amount_str = m.group("amount")

        amt = _parse_signed_int_yen(amount_str)
        if amt is None:
            continue
        amt = abs(amt)

        # 判断 delta vs abs
        if sign:
            negative = sign in ("-", "−", "－")
            kind = "delta"
            value = -amt if negative else amt
        elif sep and sep in ("-", "−", "－"):
            kind = "delta"
            value = -amt
        else:
            # 无符号 → abs（如 "ブルー229,000円"）
            kind = "abs"
            value = amt

        # 拆分 label blob（如 "オレンジ、ブルー"）
        labels = _split_color_labels_shop15(label_blob)
        for lab in labels:
            lab_clean = _clean_label_shop15(lab)
            if lab_clean:
                specs.append((lab_clean, kind, value))

    return base_price, specs

# Step 7-8: LLM 提取 — 已提取到 shop_cleaners_split_llm/llm_shop15.py
from ..shop_cleaners_split_llm.llm_shop15 import (
    extract_specs_shop15_llm as _extract_specs_shop15_llm_impl,
)

# ----------------------------------------------------------------------
# Step 11: 清洗主函数
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

        decomp = dispatch_extraction_to_price_decomposition(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_shop15_regex(price_text_s),
            llm_fn=lambda: _extract_specs_shop15_llm_impl(
                price_text_s, idx=i,
                regex_fn=_extract_specs_shop15_regex,
                extract_base_price_fn=_extract_base_price_at_start,
                parse_signed_int_yen_fn=_parse_signed_int_yen,
                clean_label_fn=_clean_label_shop15,
                multi_label_delta_block_re=MULTI_LABEL_DELTA_BLOCK_RE_shop15,
                split_color_labels_fn=_split_color_labels_shop15,
            ),
            base_price=None,
            source_text_raw=price_text_s,
            result_adapter=lambda r: (
                r[0],
                [(l, v) for l, k, v in r[1] if k == "delta"],
                [(l, v) for l, k, v in r[1] if k == "abs"],
            ),
            has_result_fn=lambda r: bool(r[1]),
            extract_base_from_result=lambda r: r[0],
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
