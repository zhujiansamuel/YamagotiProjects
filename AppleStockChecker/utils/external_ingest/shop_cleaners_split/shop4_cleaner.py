from __future__ import annotations

"""
shop4 清洗器 — モバイルミックス

  原始 DataFrame（data / data11 列）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _find_base_price()                    ← Step 1: 回溯查找基准价
    │
    ├─ _normalize_amount_text()              ← Step 2: 全角→半角归一化
    │
    ├─ _extract_specs_shop4_regex_line()      ← Step 3: 正则提取单行色差
    │
    ├─ _extract_specs_shop4_dispatch() ← Step 4: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop4_regex_block()   ← Step 5a: 正则逐行收集
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop4_llm()             ← Step 5b: LLM 核心提取
    │       └─ Guardrails (coerce + validate)        ← Step 6: 防幻觉过滤
    │
    ├─ _label_matches_color_unified()       ← Step 7: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop4()                         ← Step 8: 主函数，生成输出行
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    PriceDecomposition,
    resolve_color_prices,
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _norm_strip,
    _normalize_amount_text,
    _label_matches_color_unified,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    validate_columns,
    dispatch_extraction,
    LABEL_SPLIT_RE_shop4 as LABEL_SPLIT_RE,
    EXTRACTION_MODE,
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

def _split_labels(label: str) -> List[str]:
    return [p.strip() for p in LABEL_SPLIT_RE.split(label or "") if p and p.strip()]

# ----------------------------------------------------------------------
# Step 2: 基准价回溯查找
# ----------------------------------------------------------------------

# 纯金额行：仅含数字、逗号、円、空格，无数颜色/全色等文字
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
# Step 3: 正则模式定义 & 单行解析
# ----------------------------------------------------------------------

_COLOR_DELTA_RE = re.compile(
    r"""^\s*
        (?P<label>全色|[\S　 ]*?[^\s　])
        \s*
        (?P<sign>[+\-−－])?
        \s*
        (?P<amount>\d[\d,]*)\s*円?
        \s*$
    """,
    re.VERBOSE,
)

def _extract_specs_shop4_regex_line(line: str) -> Optional[List[Tuple[str, int]]]:
    if not line or not isinstance(line, str):
        return None
    s = line.strip()
    if s == "全色" or s == "全 色":
        return [("全色", 0)]

    m = _COLOR_DELTA_RE.match(s)
    if not m:
        am = re.search(r"([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)\s*円?", s)
        if not am:
            if "全色" in s:
                return [("全色", 0)]
            return None
        sign = am.group(1) or "+"
        amt_txt = am.group(2)
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            try:
                amt = to_int_yen(amt_txt)
            except Exception:
                amt = None
        if amt is None:
            return None
        if sign in ("-", "−", "－"):
            amt = -int(amt)
        else:
            amt = int(amt)

        label_part = s[:am.start()].strip()
        if not label_part:
            return None
        labels = [p for p in LABEL_SPLIT_RE.split(label_part) if p]
        return [(lbl.strip(), int(amt)) for lbl in labels]

    label_raw = m.group("label").strip()
    sign = m.group("sign") or "+"
    amt_val = None
    try:
        amt_val = to_int_yen(m.group("amount"))
    except Exception:
        amt_val = None
    if amt_val is None:
        amt_val = _normalize_amount_text(m.group("amount"))
    if amt_val is None:
        return None

    amt_val = int(amt_val)
    if sign in ("-", "−", "－"):
        amt_val = -amt_val

    labels = [p for p in LABEL_SPLIT_RE.split(label_raw) if p]
    if not labels:
        return None
    return [(lbl.strip(), int(amt_val)) for lbl in labels]

# ----------------------------------------------------------------------
# Step 4: 颜色家族同义词 & 匹配
# ----------------------------------------------------------------------
# Step 4b: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop4 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5a: 正则收集（逐行扫描 block）
# ----------------------------------------------------------------------
# 按 円/ 或 円／ 分割，支持 "ディープブルー-2,000円/コズミックオレンジ-6,500円" 这类
# 同一行多组「颜色±金额」的格式（2025-02 修复漏解析第二项及后续项）
_SHOP4_LINE_SPLIT_BY_YEN_SLASH = re.compile(r"円\s*[／/]\s*")


def _extract_specs_shop4_regex_block(
    df: pd.DataFrame, start_idx: int,
) -> Tuple[Dict[str, int], List[Tuple[str, int]], Dict[str, str]]:
    """
    纯正则版：逐行扫描 block 收集颜色差额。
    若行内含 "円/色名±金额" 或 "円／色名±金额"，先按此模式分割再逐段解析。
    返回：(adjustments, delta_specs, color_delta_label_map)
      - adjustments: { _norm(label) | "ALL" : delta_int }
      - delta_specs: [(label_raw, delta)] 原始标签列表
      - color_delta_label_map: { _norm(label) : label_raw }
    """
    result: Dict[str, int] = {}
    delta_specs: List[Tuple[str, int]] = []
    color_delta_label_map: Dict[str, str] = {}
    n = len(df)
    for j in range(start_idx, n):
        nxt_model = ""
        if "data11" in df.columns:
            val = df["data11"].iat[j]
            nxt_model = str(val) if val is not None else ""
        if j > start_idx and nxt_model.strip():
            break
        # 若该行为纯金额且下一行为机型行，则属于下一 block 基准价，不纳入当前 block
        if j > start_idx and _is_next_model_base_price_row(df, j, n):
            break

        line = ""
        if "data" in df.columns:
            val = df["data"].iat[j]
            line = str(val) if val is not None else ""

        # 按 円/ 或 円／ 分割，逐段解析（支持 "A-2000円/B-6500円" 格式）
        segments = _SHOP4_LINE_SPLIT_BY_YEN_SLASH.split(line)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            parsed = _extract_specs_shop4_regex_line(seg)
            if not parsed:
                continue
            for label, delta in parsed:
                delta_specs.append((label, int(delta)))
                if "全色" in label:
                    result["ALL"] = int(delta)
                else:
                    nk = _norm(label)
                    result[nk] = int(delta)
                    color_delta_label_map[nk] = label
    return result, delta_specs, color_delta_label_map

# ----------------------------------------------------------------------
# Step 5b: LLM 核心提取
# ----------------------------------------------------------------------

# LLM 相关代码已提取到 shop_cleaners_split_llm/llm_shop4.py
from ..shop_cleaners_split_llm.llm_shop4 import (
    extract_specs_shop4_llm as _extract_specs_shop4_llm,
)

# ----------------------------------------------------------------------
# Step 7: 提取模式调度
# ----------------------------------------------------------------------

def _extract_specs_shop4_dispatch(
    df: pd.DataFrame,
    start_idx: int,
    *,
    base_price: int,
    source_text_raw: str,
    row_index: object = None,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 PriceDecomposition
    """
    (_, ds, _), method = dispatch_extraction(
        EXTRACTION_MODE,
        regex_fn=lambda: _extract_specs_shop4_regex_block(df, start_idx),
        llm_fn=lambda: _extract_specs_shop4_llm(df, start_idx, row_index=row_index),
        has_result_fn=lambda r: bool(r[0]),  # r = (adjustments, delta_specs, label_map)
    )

    return PriceDecomposition(
        delta_specs=list(ds),
        abs_specs=[],
        extraction_method=method,
        base_price=base_price,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop4(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq, extraction_mode=EXTRACTION_MODE)
    _log_seq += 1

    _log_seq = validate_columns(df, ["data", "data11", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    info_df = _load_iphone17_info_df_from_db()
    pn_map = _build_color_map(info_df)

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
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            continue

        base_price = _find_base_price(df, i)
        if base_price is None:
            continue

        # ---- 收集 block 文本（source_text_raw_full） ----
        # 排除「纯金额行且下一行为机型行」的下一 block 基准价行，避免日志误导
        block_lines_raw = []
        for j in range(i, block_end + 1):
            if j > i and _is_next_model_base_price_row(df, j, n):
                break
            raw = str(df["data"].iat[j]) if df["data"].iat[j] is not None else ""
            if raw.strip():
                block_lines_raw.append(raw.strip())
        source_text_raw_full = " | ".join(block_lines_raw)

        # ---- 提取 ----
        decomp = _extract_specs_shop4_dispatch(
            df, i,
            base_price=base_price,
            source_text_raw=source_text_raw_full,
            row_index=i,
        )

        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_to_pn,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=i,
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    out = assemble_output_df(rows)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=start_time, log_seq=_log_seq)

    return out
