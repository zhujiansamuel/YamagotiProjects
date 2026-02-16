from __future__ import annotations

"""
shop12 清洗器 — トゥインクル

  原始文本（備考1 + 買取価格）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _normalize_remark_for_llm()              ← Step 1: 去除開封行，预处理備考1
    │
    ├─ _norm_amount_to_int()                    ← Step 2: 统一全角数字→int
    │
    ├─ _extract_specs_shop12_dispatch()   ← Step 5: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop12_regex()    ← Step 3: 正则提取 (abs + delta)
    │   │       └─ _fallback_parse_rules()            ← 核心正则: _FALLBACK_ABS_RE / _FALLBACK_DELTA_RE
    │   │
    │   └─ llm 路径:
    │       └─ _extract_specs_shop12_llm()  ← Step 4: LLM 提取 + 防幻觉 (shop_cleaners_split_llm/llm_shop12.py)
    │           └─ _extract_specs_shop12_llm_core()    ← LLM 核心: effective_class 修正 + 去重
    │
    ├─ _label_matches_color_unified()           ← Step 6: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop12()                           ← Step 7: 主函数，生成输出行
"""

import logging
import re
import time
from typing import List, Optional, Tuple

import pandas as pd

from ...external_ingest.cleaner_tools import parse_dt_aware
from ..cleaner_tools import (
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _norm_strip,
    _normalize_amount_text,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    validate_columns,
    log_row_skip,
    dispatch_extraction,
    DELTA_PATTERN_STRICT,
    ABS_PRICE_PATTERN,
    LABEL_SPLIT_RE_shop12,
    EXTRACTION_MODE,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop12"
SHOP_NAME = "トゥインクル"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# ----------------------------------------------------------------------
# Step 1: 備考1 文本预处理
# ----------------------------------------------------------------------

def _normalize_remark_for_llm(remark_raw: str) -> str:
    """
    - 把与"開封/開封品/※開封/開封済"粘在同一行的内容拆行；
    - 去掉所有"開封"行，只保留可用于新品价规则的行；
    - 最终返回喂给 LLM 的文本（可能是多行）。
    """
    if not remark_raw:
        return ""
    s = str(remark_raw)

    # 关键：把"※開封品"等前面强行插入换行（解决: Orange-2000円※開封品...）
    s = re.sub(r"(※\s*開封品|※\s*開封|開封品|開封済|開封)", r"\n\1", s)

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", s) if ln is not None and ln.strip()]
    keep: List[str] = []
    for ln in lines:
        if ("開封" in ln) or ("開封品" in ln) or ("※開封" in ln) or ("開封済" in ln):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()

# ----------------------------------------------------------------------
# Step 2: 数字归一化（含全角）
# ----------------------------------------------------------------------

# _norm_amount_to_int は cleaner_tools._normalize_amount_text に統一
_norm_amount_to_int = _normalize_amount_text

# ----------------------------------------------------------------------
# Step 3: 正则提取函数
# ----------------------------------------------------------------------

# 正则模式：使用 cleaner_tools 统一的基础模式
# shop12 的 _FALLBACK_ABS_RE 与 ABS_PRICE_PATTERN 基本一致
_FALLBACK_ABS_RE = ABS_PRICE_PATTERN
# shop12 的 _FALLBACK_DELTA_RE 与 DELTA_PATTERN_STRICT 基本一致
_FALLBACK_DELTA_RE = DELTA_PATTERN_STRICT
# LABEL_SPLIT_RE_shop12: 从 cleaner_tools 导入

def _fallback_parse_rules(text: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []
    if not text:
        return abs_list, delta_list

    for ln in re.split(r"[\r\n]+", str(text)):
        ln = (ln or "").strip()
        if not ln:
            continue

        # 全色
        if "全色" in ln:
            m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)?", ln)
            if m:
                sign = m.group(1) or "+"
                amt = _norm_amount_to_int(m.group(2) or "0") or 0
                delta_list.append(("全色", -amt if sign in ("-", "−", "－") else amt))
            else:
                delta_list.append(("全色", 0))
            continue

        for m in _FALLBACK_ABS_RE.finditer(ln):
            amt = _norm_amount_to_int(m.group("amount"))
            if amt is None:
                continue
            labels_part = m.group("labels") or ""
            toks = [t.strip() for t in LABEL_SPLIT_RE_shop12.split(labels_part) if t.strip()]
            for tok in toks:
                if tok:
                    abs_list.append((tok, int(amt)))

        for m in _FALLBACK_DELTA_RE.finditer(ln):
            amt = _norm_amount_to_int(m.group("amount"))
            if amt is None:
                continue
            sign = m.group("sign") or "+"
            delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
            labels_part = m.group("labels") or ""
            toks = [t.strip() for t in LABEL_SPLIT_RE_shop12.split(labels_part) if t.strip()]
            for tok in toks:
                if tok:
                    delta_list.append((tok, delta))

    return abs_list, delta_list

def _extract_specs_shop12_regex(
    remark_for_llm: str,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    纯正则版：从预处理后的備考1文本中提取 (abs_list, delta_list)。
    """
    return _fallback_parse_rules(remark_for_llm)

# ----------------------------------------------------------------------
# Step 4: LLM 提取 — 已提取到 shop_cleaners_split_llm/llm_shop12.py
# ----------------------------------------------------------------------

from ..shop_cleaners_split_llm.llm_shop12 import (
    extract_specs_shop12_llm as _extract_specs_shop12_llm_impl,
)


def _extract_specs_shop12_llm(
    remark_for_llm: str,
    idx: object = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    return _extract_specs_shop12_llm_impl(
        remark_for_llm, idx=idx,
        fallback_parse_rules_fn=_fallback_parse_rules,
    )

# ----------------------------------------------------------------------
# Step 5: 提取模式调度
# ----------------------------------------------------------------------

def _extract_specs_shop12_dispatch(
    remark_for_llm: str,
    *,
    base_price: int,
    source_text_raw: str,
    idx: object = None,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 PriceDecomposition
    """
    (abs_list, delta_list), method = dispatch_extraction(
        EXTRACTION_MODE,
        regex_fn=lambda: _extract_specs_shop12_regex(remark_for_llm),
        llm_fn=lambda: _extract_specs_shop12_llm(remark_for_llm, idx=idx),
        has_result_fn=lambda r: bool(r[0] or r[1]),  # r = (abs_list, delta_list)
    )

    # ---- "全色" delta → 覆盖全部，清空 abs ----
    delta_specs: List[Tuple[str, int]] = []
    abs_specs: List[Tuple[str, int]] = list(abs_list)

    for label_raw, delta_val in delta_list:
        if str(label_raw).strip() in {"全色", "ALL"}:
            delta_specs = [("全色", int(delta_val))]
            abs_specs = []
            break
        delta_specs.append((label_raw, int(delta_val)))
    else:
        delta_specs = [(lb, int(d)) for lb, d in delta_list]

    return PriceDecomposition(
        base_price=base_price,
        delta_specs=delta_specs,
        abs_specs=abs_specs,
        extraction_method=method,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 6: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop12 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 7: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop12(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    t_start = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq, extraction_mode=EXTRACTION_MODE)
    _log_seq += 1

    _log_seq = validate_columns(df, ["モデルナンバー", "備考1", "買取価格", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        base_price = extract_price_yen(row.get("買取価格"))
        if base_price is None:
            continue
        base_price = int(base_price)

        model_text = str(row.get("モデルナンバー") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            _log_seq += 1
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=idx, skip_reason="model_or_cap_parse_failed", log_seq=_log_seq,
                         model_text=model_text)
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            _log_seq += 1
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=idx, skip_reason="no_info_key", log_seq=_log_seq,
                         model_text=model_text, model_norm=model_norm, capacity_gb=cap_gb)
            continue

        remark_raw = row.get("備考1") or ""
        remark_for_llm = _normalize_remark_for_llm(remark_raw)
        source_text_raw_full = str(remark_raw)

        decomp = _extract_specs_shop12_dispatch(
            remark_for_llm,
            base_price=base_price,
            source_text_raw=source_text_raw_full,
            idx=idx,
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        new_rows, _log_seq = resolve_color_prices(
            decomp, color_map, _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    # ---- 输出 DataFrame 组装 ----
    out = assemble_output_df(rows)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=t_start, log_seq=_log_seq)

    return out
