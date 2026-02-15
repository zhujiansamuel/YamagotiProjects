from __future__ import annotations

"""
shop16 清洗器 — 携帯空間

  原始文本（買取価格列）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _normalize_price_text_shop16()     ← Step 1: 归一化（换行→/、压缩空白）
    │
    ├─ _extract_base_price_shop16()       ← Step 2: 提取基础价
    │
    ├─ _extract_specs_shop16_dispatch()  ← Step 9: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   ├─ _extract_specs_shop16_regex_deltas()      ← Step 6a: 正则提取差价
    │   │   └─ _extract_specs_shop16_regex_abs()   ← Step 6b: 正则提取绝对价
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop16_llm_core()     ← Step 7: LLM 核心提取
    │       └─ Guardrail A/B/C                      ← Step 8: 防幻觉过滤
    │
    ├─ _label_matches_color_unified()     ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop16()                     ← Step 10: 主函数，生成输出行
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    LABEL_SPLIT_RE_shop16 as SPLIT_TOKENS_RE,
    LABEL_SPLIT_RE_shop16_SIMPLE,
    EXTRACTION_MODE,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    validate_columns,
    dispatch_extraction,
)

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置 (OLLAMA / EXTRACTION_MODE 见 cleaner_tools)
# ----------------------------------------------------------------------

MODEL_COL = "iPhone 17 Pro Max"
DESC_COL  = "説明1"
PRICE_COL = "買取価格"

# ----------------------------------------------------------------------
# Step 1: 价格文本归一化
# ----------------------------------------------------------------------

def _normalize_price_text_shop16(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    # 把换行变成分隔（保留"下一行是颜色差价"的结构）
    s = re.sub(r"[\r\n]+", " / ", s)
    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    # 多个分隔合并
    s = re.sub(r"(?:\s*/\s*){2,}", " / ", s).strip()
    return s

# ----------------------------------------------------------------------
# Step 2: 基础价提取 & 判定
# ----------------------------------------------------------------------

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

def _extract_base_price_shop16(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)  # 兜底
    return to_int_yen(m.group(1))

_BASE_ONLY_RE = re.compile(r"^\s*(?:￥|\¥)?\s*\d[\d,]*\s*(?:円)?\s*$")

def _is_base_only_price_text(price_text_norm: str) -> bool:
    """判断文本是否只包含一个基础价，不含任何颜色差价信息。"""
    return bool(_BASE_ONLY_RE.match(price_text_norm or ""))

# ----------------------------------------------------------------------
# Step 3: 标签归一化 & 拆分
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）

_TRAILING_AMOUNT_IN_LABEL_RE = re.compile(
    r"(?:[：:])?\s*(?:￥)?\s*[+\-−－]?\s*\d[\d,]*\s*(?:円)?\s*$",
    re.UNICODE,
)

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

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop16 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则模式定义
# ----------------------------------------------------------------------

COLOR_DELTA_RE = re.compile(
    r"""(?P<label>[^：:\-\s]+)\s*
        (?P<sep>[：:\-])\s*           # 捕获分隔符
        (?P<sign>[+\-−－])?\s*        # 显式正负号（可选）
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円]+)\s*￥\s*(?P<amount>\d[\d,]*)""",
    re.UNICODE
)
# SPLIT_TOKENS_RE: 从 cleaner_tools.LABEL_SPLIT_RE_shop16 导入

_GROUP_SHARED_DELTA_RE = re.compile(
    r"""
    (?P<labels>[^0-9￥円]+?)          # 多颜色标签段（含 /）
    \s*(?P<sign>[+\-−－])\s*         # 显式正负号
    (?P<amount>\d[\d,]*)\s*(?:円)?   # 金额（可无 円）
    """,
    re.UNICODE | re.VERBOSE
)

# ----------------------------------------------------------------------
# Step 6: 正则提取函数
# ----------------------------------------------------------------------

def _extract_specs_shop16_regex_deltas(text: str) -> List[Tuple[str, int]]:
    """从价格串中抽取多段"颜色±金额"，支持 '青/オレンジ -5000' 这类多标签共用金额。"""
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text)
    # 去掉第一个"基础价 N円/￥N"
    m0 = FIRST_YEN_RE.search(s)
    tail = s[m0.end():] if m0 else s

    parts = [p.strip() for p in SPLIT_TOKENS_RE.split(tail) if p and p.strip()]
    if not parts:
        parts = [tail.strip()]

    pending_labels: List[str] = []  # 暂存像 '青/オレンジ -5000' 中的前置标签（如 '青'）

    def _normalize_label(lbl: str) -> str:
        # 去掉各种空白（含全角空格/不间断空格）
        return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

    for part in parts:
        # 该片段是否包含"颜色±金额"
        matches = list(COLOR_DELTA_RE.finditer(part))
        if matches:
            for m in matches:
                label = _normalize_label(m.group("label"))
                if not label:
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

                # 当前标签
                out.append((label, delta))
                # 把之前挂起的标签，也应用同一金额
                for pl in pending_labels:
                    out.append((_normalize_label(pl), delta))
            pending_labels = []  # 清空缓存
            continue

        # 否则，这是"只有标签没有金额"的片段（如 '青'）；缓存它，等待后面的金额
        # 如果是 '青/橙' 没被上层 split 掉，也进一步按斜杠切一下
        for tok in LABEL_SPLIT_RE_shop16_SIMPLE.split(part):
            tok = _normalize_label(tok)
            if tok:
                pending_labels.append(tok)

    return out

def _extract_specs_shop16_regex_abs(text: str) -> List[Tuple[str, int]]:
    """从价格串中抽取"颜色￥绝对价"，如：'黒￥86100/青￥87100'"""
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    for m in COLOR_ABS_RE.finditer(str(text)):
        label = (m.group("label") or "").strip()
        amt = to_int_yen(m.group("amount"))
        if label and amt is not None:
            out.append((label, int(amt)))
    return out

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

def _extract_specs_shop16_regex(
    price_text: str,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    纯正则版：从 price_text 中提取 (base_price, deltas, abs_prices)。
    """
    base_price = _extract_base_price_shop16(price_text)
    deltas = _extract_specs_shop16_regex_deltas(price_text)
    absps = _extract_specs_shop16_regex_abs(price_text)
    return base_price, deltas, absps

# Step 7-8: LLM 提取 — 已提取到 shop_cleaners_split_llm/llm_shop16.py
from ..shop_cleaners_split_llm.llm_shop16 import (
    extract_specs_shop16_llm as _extract_specs_shop16_llm_impl,
)


def _extract_specs_shop16_llm(
    price_text: str, idx: object = None,
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]]]:
    return _extract_specs_shop16_llm_impl(
        price_text, idx=idx,
        extract_base_price_fn=_extract_base_price_shop16,
        is_base_only_fn=_is_base_only_price_text,
        extract_shared_delta_map_fn=_extract_shared_delta_map_shop16,
        normalize_label_fn=_normalize_label_shop16,
        split_labels_fn=_split_labels_shop16,
        regex_deltas_fn=_extract_specs_shop16_regex_deltas,
        regex_abs_fn=_extract_specs_shop16_regex_abs,
    )

# ----------------------------------------------------------------------
# Step 9: 提取模式调度
# ----------------------------------------------------------------------

def _extract_specs_shop16_dispatch(
    price_text: str, idx: object = None, *, source_text_raw: str,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrail A/B/C
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrail 兜底

    返回 PriceDecomposition
    """
    (bp, deltas, absps), method = dispatch_extraction(
        EXTRACTION_MODE,
        regex_fn=lambda: _extract_specs_shop16_regex(price_text),
        llm_fn=lambda: _extract_specs_shop16_llm(price_text, idx=idx),
        has_result_fn=lambda r: bool(r[1] or r[2]),  # r = (bp, deltas, absps)
    )
    # auto 模式下 LLM 的 base_price 为 None 时，回退到正则的 base_price
    if method == "llm" and bp is None:
        bp_re = _extract_specs_shop16_regex(price_text)[0]
        if bp_re is not None:
            bp = bp_re

    # base_price=None 时不能使用 delta（base+delta 无法计算）
    if bp is None:
        deltas = []

    return PriceDecomposition(
        base_price=bp,
        delta_specs=deltas,
        abs_specs=absps,
        extraction_method=method,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 10: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop16(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop16 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop16"
    SHOP_NAME = "携帯空間"

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq)

    _log_seq = validate_columns(df, [MODEL_COL, DESC_COL, PRICE_COL, "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

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
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        price_raw = "" if price_cell is None else str(price_cell)
        price_text = _normalize_price_text_shop16(price_raw)

        # 根据 EXTRACTION_MODE 提取价格信息（regex / llm / auto）
        decomp = _extract_specs_shop16_dispatch(
            price_text, idx=idx, source_text_raw=price_text,
        )

        # 没 base 且没 abs：没法落库
        if decomp.base_price is None and not decomp.abs_specs:
            continue

        emit_default = decomp.base_price is not None

        # 使用公共函数生成输出行（含完整日志）
        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=emit_default,
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
