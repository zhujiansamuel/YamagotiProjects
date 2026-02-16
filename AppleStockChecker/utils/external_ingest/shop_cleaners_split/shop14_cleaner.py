"""
shop14_cleaner  —  買取楽園

数据处理流程:
  raw DataFrame
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ Step 1  列校验 & remark列解析
    ├─ Step 2  行级过滤（未開封 + model/cap/color_map 匹配）
    ├─ Step 3  base_price 提取
    ├─ Step 4  remark文本归一化（3列合并）
    ├─ Step 5  价格规则抽取 dispatch（EXTRACTION_MODE: regex / llm / auto）
    │           ├─ regex路径: _extract_specs_shop14_regex()
    │           └─ llm路径:   _extract_specs_shop14_llm_impl()
    ├─ Step 6  全色处理（all_delta 快捷路径）
    ├─ Step 7  label → color 匹配（家族同义词）
    ├─ Step 8  价格计算（abs优先 > base+delta > base）
    └─ Step 9  输出 DataFrame 组装
"""
from __future__ import annotations

import logging
import os
import re
import time
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

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
    coerce_amount_yen,
    dispatch_extraction_to_price_decomposition,
    lx,
    HAS_LANGEXTRACT,
    log_llm_extraction_error,
    LABEL_SPLIT_RE_shop14,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
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


def _labels_from_text_fallback(extraction_text: str) -> str:
    t = str(extraction_text or "")
    t = t.replace("全色", "")
    t = re.sub(r"(?:[+\-−－])?\s*(?:¥|￥)?\s*\d[\d,，]*\s*(?:円)?", "", t)
    t = t.strip()
    return t


# ---------------------------------------------------------------------------
# Step 3: 正则模式定义
# ---------------------------------------------------------------------------

COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)

_SPLIT_TOKENS_SAFE_RE = re.compile(
    r"""
    [／/、，]
    |(?<!\d),(?!\d)
    |(?:\s+\+\s+)
    |(?:\s*;\s*)
    """,
    re.UNICODE | re.VERBOSE,
)

_COLOR_ABS_PRICE_RE = re.compile(
    r"""^\s*
        (?P<label>[^：:\-\s/、／¥円]+?)
        \s*(?:[:：]?\s*)
        (?:¥|￥)?\s*
        (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})*|\d+)
        \s*(?:円)?\s*$
    """,
    re.UNICODE | re.VERBOSE,
)

_PAIR_GROUP_RE_shop14 = re.compile(
    r"""
    (?P<labels>[^\d¥￥円:+\-−－＋]+?)
    \s*(?:[:：]\s*)?
    (?P<sign>[+\-−－＋])?
    \s*(?:¥|￥)?\s*
    (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})+|\d+)
    \s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

PAIR_RE_MULTI = re.compile(
    r"([^\d¥円,，＋+－\-−\s]+)\s*([+\-−－]?\s*\d[\d,，]*)"
)

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


def _split_color_amount_pairs_multi(txt: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not txt:
        return out
    s = str(txt)

    for label, amt_s in PAIR_RE_MULTI.findall(s):
        label = label.lstrip("、/,／，,;；").strip()
        if not label:
            continue
        amt = _coerce_amount_yen(amt_s)
        if amt is None:
            continue
        out.append((label, amt))

    if len(out) >= 2:
        return out
    return []


# ---------------------------------------------------------------------------
# Step 7-A: 纯正则抽取路径
# ---------------------------------------------------------------------------

def _extract_specs_shop14_regex(
    text: str,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]]]]:
    """
    纯正则从 remark 文本中抽取颜色价格规则。
    返回: {"all_delta": Optional[int], "abs": [...], "delta": [...]}
    """

    def _strip_label_delims(s: str) -> str:
        s = str(s or "").strip()
        s = re.sub(r"^[／/、，,;；\s]+", "", s)
        s = re.sub(r"[／/、，,;；\s]+$", "", s)
        return s.strip()

    def _split_labels(labels: str) -> List[str]:
        s = str(labels or "").strip()
        if not s:
            return []
        parts = LABEL_SPLIT_RE_shop14.split(s)
        return [p.strip() for p in parts if p and p.strip()]

    out: Dict[str, Union[Optional[int], List[Tuple[str, int]]]] = {
        "all_delta": None, "abs": [], "delta": [],
    }
    s = _clean_remark_frag(text)
    if not s:
        return out

    # 全色检测
    m_all = re.search(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?", s)
    if m_all:
        out["all_delta"] = _coerce_amount_yen(m_all.group(0).replace("全色", "").strip()) or 0
        return out
    if "全色" in s:
        out["all_delta"] = 0
        return out

    # 先尝试 multi-pair 解析
    multi = _split_color_amount_pairs_multi(s)
    if multi:
        vals_abs = [abs(v) for _, v in multi]
        if all(v >= 20000 for v in vals_abs):
            out["abs"] = [(lb, abs(v)) for lb, v in multi]
        else:
            out["delta"] = list(multi)
        return out

    # PAIR_GROUP 正则
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []

    for m in _PAIR_GROUP_RE_shop14.finditer(s):
        labels_raw = _strip_label_delims(m.group("labels"))
        sign_str = m.group("sign") or ""
        amt_str = m.group("amount")
        amt = _coerce_amount_yen(amt_str)
        if amt is None:
            continue

        has_sign = sign_str in {"+", "-", "−", "－", "＋"}
        if has_sign:
            if sign_str in {"-", "−", "－"}:
                amt = -abs(amt)
            for lb in _split_labels(labels_raw):
                delta_list.append((lb, amt))
        else:
            if abs(amt) >= 20000:
                for lb in _split_labels(labels_raw):
                    abs_list.append((lb, abs(amt)))
            else:
                for lb in _split_labels(labels_raw):
                    delta_list.append((lb, amt))

    out["abs"] = abs_list
    out["delta"] = delta_list
    return out


# Step 7-B: LLM 路径 — 已提取到 shop_cleaners_split_llm/llm_shop14.py
from ..shop_cleaners_split_llm.llm_shop14 import (
    extract_specs_shop14_llm as _extract_specs_shop14_llm_impl,
)

# ---------------------------------------------------------------------------
# Step 7-C: Dispatch（三模式路由，经 dispatch_extraction_to_price_decomposition 单 fragment 语义）
# ---------------------------------------------------------------------------

def _extract_specs_shop14_parse(
    text: str,
    mode: str = EXTRACTION_MODE,
) -> Tuple[Dict[str, Union[Optional[int], List[Tuple[str, int]]]], str]:
    """
    单 fragment 三模式路由。
    返回: (parsed_dict, extraction_method)
    parsed_dict = {"all_delta": ..., "abs": [...], "delta": [...]}
    """
    return dispatch_extraction_to_price_decomposition(
        mode,
        regex_fn=lambda: _extract_specs_shop14_regex(text),
        llm_fn=lambda: _extract_specs_shop14_llm_impl(
            text, split_color_amount_pairs_multi_fn=_split_color_amount_pairs_multi,
        ),
        has_result_fn=lambda r: (
            r.get("all_delta") is not None or r.get("abs") or r.get("delta")
        ),
        as_parse_fn=True,
    )


# ---------------------------------------------------------------------------
# Step 8: 主清洗函数
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

        decomp = dispatch_extraction_to_price_decomposition(
            base_price=base_price,
            source_text_raw=combined,
            frags=frags,
            combined=combined,
            parse_fn=lambda t: _extract_specs_shop14_parse(t),
            all_delta_key="all_delta",
            abs_key="abs",
            delta_key="delta",
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
