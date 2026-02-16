from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _load_iphone17_info_df_from_db,
    _parse_capacity_gb,
    _truncate_for_log,
    _normalize_model_generic,
    _build_color_map,
    normalize_text_basic,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    LABEL_SPLIT_RE_shop17 as SPLIT_TOKENS_RE_shop17,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    validate_columns,
    dispatch_extraction_to_price_decomposition,
    lx,
    HAS_LANGEXTRACT,
    log_llm_extraction_error,
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
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _normalize_model_generic() / _parse_capacity_gb()  ← Step 1: 机型・容量解析（cleaner_tools）
    │
    ├─ extract_price_yen()         ← Step 2: 基础价提取（cleaner_tools）
    │
    ├─ dispatch_extraction_to_price_decomposition() ← Step 3: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   ├─ _pick_unopened_section()     ← 提取【未開封】段
    │   │   ├─ _normalize_color_text_shop17()  ← 归一化
    │   │   ├─ SPLIT_TOKENS_RE 拆分          ← 分割多条目
    │   │   └─ COLOR_NONE_RE / COLOR_DELTA_RE  ← なし模式・金额模式
    │   │
    │   └─ llm 路径:
    │       └─ _extract_specs_shop17_llm()  ← LangExtract 核心提取
    │
    ├─ _label_matches_color_unified()  ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop17()              ← Step 5: 主函数，生成输出行
"""

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop17"
SHOP_NAME = "ゲストモバイル"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

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
COLOR_DELTA_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# ----------------------------------------------------------------------
# 颜色匹配函数
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop17 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

def _extract_specs_shop17_regex(text: str) -> List[Tuple[str, int]]:
    """
    正则版提取 [(label_raw, delta_int)]，作为 LLM 的 fallback，也可以单独使用。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = _normalize_color_text_shop17(_pick_unopened_section(str(text)))

    if "色減額" in s:
        s = s.split("色減額", 1)[-1].lstrip(":：")

    # 整段就是「なし/減額なし」-> 无色差额
    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        return out

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop17.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    for part in parts:
        # 「シルバーなし」/「クラウドホワイト：なし」
        m0 = COLOR_NONE_RE_shop17.search(part)
        if m0:
            label = _normalize_label_shop17(m0.group("label"))
            if _is_plausible_color_label_shop17(label):
                out.append((label, 0))
            continue

        # 「ブルー-1000」「スカイブルー: -3,000」 等
        for m in COLOR_DELTA_RE_shop17.finditer(part):
            label = _normalize_label_shop17(m.group("label"))
            if not _is_plausible_color_label_shop17(label):
                continue
            sep = m.group("sep")
            sign = m.group("sign")
            amt = to_int_yen(m.group("amount"))
            if amt is None:
                continue
            if sign:
                negative = sign in ("-", "−", "－")
            else:
                negative = sep in ("-", "−", "－") if sep else False
            delta = -int(amt) if negative else int(amt)
            out.append((label, delta))

    return out

# LLM 提取 — 已提取到 shop_cleaners_split_llm/llm_shop17.py
from ..shop_cleaners_split_llm.llm_shop17 import (
    extract_specs_shop17_llm as _extract_specs_shop17_llm_impl,
)


def _extract_specs_shop17_llm(
    text: str,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None
) -> List[Tuple[str, int]]:
    return _extract_specs_shop17_llm_impl(
        text, shop_name=shop_name, cleaner_name=cleaner_name,
        row_context=row_context,
        normalize_color_text_fn=_normalize_color_text_shop17,
        pick_unopened_section_fn=_pick_unopened_section,
        is_plausible_color_label_fn=_is_plausible_color_label_shop17,
    )

# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------
def clean_shop17(df: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop17 调用内单调递增，用于 ELK 排序

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq)

    _log_seq = validate_columns(df, ["type", "新未開封品", "色減額", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    if df.empty:
        log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=0, start_time=start_time, log_seq=_log_seq)
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)
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
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        base_price = extract_price_yen(row.get("新未開封品"))
        if base_price is None:
            continue
        base_price = int(base_price)

        raw_color = row.get("色減額")
        raw_color_s = "" if raw_color is None else str(raw_color)

        # 构建行级上下文，用于传递给下级函数和日志
        row_context = {
            "row_index": int(idx),
            "model_text": model_text,
            "model_norm": model_norm,
            "capacity_gb": cap_gb,
            "base_price": base_price,
        }

        # 提取颜色差额
        decomp = dispatch_extraction_to_price_decomposition(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_shop17_regex(raw_color_s),
            llm_fn=lambda: _extract_specs_shop17_llm(raw_color_s, SHOP_NAME, CLEANER_NAME, row_context),
            base_price=base_price,
            source_text_raw=raw_color_s,
            result_adapter=lambda r: (r, []),
        )
        if not decomp.delta_specs:
            decomp = PriceDecomposition(
                base_price=decomp.base_price,
                delta_specs=[],
                abs_specs=[],
                extraction_method="none",
                source_text_raw=decomp.source_text_raw,
            )

        shop_name = SHOP_NAME
        rec_at = parse_dt_aware(row.get("time-scraped"))

        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=shop_name,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    out = assemble_output_df(rows)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=start_time, log_seq=_log_seq)

    return out
