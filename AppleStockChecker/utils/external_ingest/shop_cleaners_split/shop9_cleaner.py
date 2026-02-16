from __future__ import annotations

"""
shop9 清洗器 — アキモバ

  原始文本（買取価格 + 色・詳細等）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _coerce_signed_int()                  ← Step 1: 金額解析（全角→半角、符号処理）
    │
    ├─ _bucket_amount()                      ← Step 2: abs/delta 分類（量級・符号ヒント）
    │
    ├─ _extract_specs_shop9_dispatch()  ← Step 7: モード調度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   ├─ _extract_abs_prices_regex()        ← Step 5a: 正則提取絶対価
    │   │   ├─ _extract_deltas_regex()            ← Step 5b: 正則提取差価
    │   │   └─ _direct_abs_overrides_for_row()    ← Step 5c: テキスト直接覆写
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop9_llm_core()        ← Step 6a: LLM 核心提取
    │       └─ _bucket_amount() guardrail         ← Step 6b: abs/delta 防幻觉過濾
    │
    ├─ _map_to_available_color()             ← Step 3: ラベル→カラーマッチング（cleaner_tools 统一）
    │
    └─ clean_shop9()                         ← Step 8: 主函数、出力行生成
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.cleaner_tools import parse_dt_aware
from ..cleaner_tools import (
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
    _normalize_amount_text,
    normalize_text_basic,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    LABEL_SPLIT_RE_shop9,
    EXTRACTION_MODE,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    log_row_skip,
    validate_columns,
    coerce_signed_int,
    dispatch_extraction,
)

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop9"
SHOP_NAME = "アキモバ"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详細）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

ABS_LIKE_MIN = int(os.getenv("SHOP9_ABS_LIKE_MIN", "50000"))  # iPhone17 绝对价量级阈值

COL_MODEL = "機種名"
COL_PRICE = "買取価格"
COL_COLOR = "色・詳細等"
COL_TIME  = "time-scraped"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）


def _norm_cls(x: str) -> str:
    # 容错：abs price / abs-price / ABS_PRICE 统一
    s = (x or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    return s

# ----------------------------------------------------------------------
# Step 1: 金額解析
# ----------------------------------------------------------------------

# _coerce_signed_int → cleaner_tools.coerce_signed_int 统一导入
_coerce_signed_int = coerce_signed_int

# _norm_amount_to_int は cleaner_tools._normalize_amount_text に統一
_norm_amount_to_int = _normalize_amount_text

# ----------------------------------------------------------------------
# Step 2: abs/delta 分類
# ----------------------------------------------------------------------

DELTA_HINT_RE = re.compile(r"(?:[+\-−－]|値下げ|値引|割引|円引|OFF|オフ|減額)", re.I)

def _bucket_amount(cls_norm: str, ex_text: str, amt: int) -> str:
    """
    返回 "abs" 或 "delta"
    规则：
      - 有负号/折扣词/加减符号 => delta
      - 金额量级很大(>=ABS_LIKE_MIN)且无加减线索 => abs（即使模型标成 delta）
      - 其余按 class；不认识则按金额量级兜底
    """
    tx = ex_text or ""
    if amt is None:
        return "delta"
    if amt < 0:
        return "delta"
    if DELTA_HINT_RE.search(tx):
        return "delta"
    if abs(amt) >= ABS_LIKE_MIN:
        return "abs"
    if cls_norm in {"abs_price", "abs", "absolute"}:
        return "abs"
    if cls_norm in {"delta", "delta_price", "adjust", "adjustment"}:
        return "delta"
    return "delta"

# ----------------------------------------------------------------------
# Step 3: 颜色家族同义词 & マッチング
# ----------------------------------------------------------------------
# 原逻辑：FAMILY_SYNONYMS_SHOP9 = {...}; SYNONYM_LOOKUP 由 for _k,_vs 循环构建
#   每个 key/val 映射到同族列表。现改用 cleaner_tools.SYNONYM_LOOKUP_NORM（去除空格版本）

def _build_color_aliases(available_colors: List[str]) -> Dict[str, List[str]]:
    # 原逻辑：syns = SYNONYM_LOOKUP.get(c0, []); out[c0]=[c0]+syns
    out = {}
    for c in available_colors:
        c0 = str(c).strip()
        if not c0:
            continue
        c0_norm = _norm(c0)
        syns = SYNONYM_LOOKUP_NORM.get(c0_norm, [])
        out[c0] = list(dict.fromkeys([c0] + syns))[:20]
    return out

def _map_to_available_color(raw_color: str, available_set: set) -> Optional[str]:
    if not raw_color:
        return None
    rc = str(raw_color).strip()
    if not rc:
        return None

    if rc.upper() == "ALL" or rc == "全色":
        return "ALL"

    if rc in available_set:
        return rc

    # 小写等价
    rcn = _norm(rc)
    for c in available_set:
        if _norm(c) == rcn:
            return c

    # 同义词兜底（原逻辑：if rc in SYNONYM_LOOKUP: for syn in SYNONYM_LOOKUP[rc]: ...）
    if rcn in SYNONYM_LOOKUP_NORM:
        for syn_norm in SYNONYM_LOOKUP_NORM[rcn]:
            for c in available_set:
                if _norm(c) == syn_norm:
                    return c

    # 包含关系兜底
    for c in available_set:
        cn = _norm(c)
        if rcn and (rcn in cn or cn in rcn):
            return c

    return None

# ----------------------------------------------------------------------
# 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop9 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑。全色/ALL 由 resolve_color_prices 的 is_all 处理。

# ----------------------------------------------------------------------
# Step 4: 正则模式定义
# ----------------------------------------------------------------------
# LABEL_SPLIT_RE_shop9: 从 cleaner_tools 导入

ABS_PRICE_RE = re.compile(
    r"(?P<labels>[^0-9０-９¥￥円]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
    re.I,
)
DELTA_RE = re.compile(
    r"(?P<labels>[^0-9０-９¥￥円]+?)\s*[：:\-]?\s*(?P<sign>[+\-−－])\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
    re.I,
)

# ----------------------------------------------------------------------
# Step 5: 正則提取関数
# ----------------------------------------------------------------------

def _is_pure_number_token(tok: str) -> bool:
    if not tok:
        return False
    t = _norm(tok)
    t = t.replace(",", "").replace("，", "")
    return t.isdigit()

def _extract_abs_prices_regex(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    s = str(text)
    for m in ABS_PRICE_RE.finditer(s):
        labels_part = (m.group("labels") or "").strip()
        amt = _norm_amount_to_int(m.group("amount"))
        if amt is None:
            continue
        toks = [t.strip() for t in LABEL_SPLIT_RE_shop9.split(labels_part) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            out.append((tok, int(amt)))
    return out

def _extract_deltas_regex(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not text:
        return out
    s = str(text)
    for m in DELTA_RE.finditer(s):
        labels_part = m.group("labels") or ""
        sign = m.group("sign") or "+"
        amt = _norm_amount_to_int(m.group("amount"))
        if amt is None:
            continue
        delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
        toks = [t.strip() for t in LABEL_SPLIT_RE_shop9.split(labels_part) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            out.append((tok, delta))
    if not out and "全色" in s:
        out.append(("全色", 0))
    return out

def _extract_amount_after_alias(text: str, alias: str) -> Optional[int]:
    """
    在 text 中查找形如 'alias 193,500' / 'alias193,500' / 'alias 193500円' 这种片段，
    只取 alias 后面"最近的那串数字"。
    不吃减价形式 'alias-500円'（中间有 '-'）。
    """
    if not text or not alias:
        return None
    s = str(text)

    # 允许 alias 后有若干空白，再跟可选的货币符号，再跟数字
    pat = re.compile(
        rf"{re.escape(alias)}\s*(?:¥|￥)?\s*([0-9０-９][0-9０-９,，]*)"
    )
    m = pat.search(s)
    if not m:
        return None
    return _norm_amount_to_int(m.group(1))

def _direct_abs_overrides_for_row(
        raw_color_text: str,
        color_to_pn: Dict[str, str],
) -> Dict[str, int]:
    """
    针对当前行，直接在 raw_color_text 里按"每个颜色的别名 -> 紧随其后的数字"扫描，
    得到 per-color 的绝对价覆盖表：{color_norm: amount_yen}。
    只接受金额 >= ABS_LIKE_MIN，避免把 -500 / 500 之类 delta 当成 abs。
    """
    overrides: Dict[str, int] = {}
    if not raw_color_text:
        return overrides

    s = str(raw_color_text)
    for col_norm in color_to_pn.keys():
        # 构建该颜色的别名集合：自身 + 同义词（原逻辑：SYNONYM_LOOKUP.get(col_norm, [])）
        aliases = {col_norm}
        for syn in SYNONYM_LOOKUP_NORM.get(col_norm, []):
            aliases.add(str(syn).strip())
        amt_for_color: Optional[int] = None
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            val = _extract_amount_after_alias(s, alias)
            if val is not None and val >= ABS_LIKE_MIN:
                amt_for_color = val
                break
        if amt_for_color is not None:
            overrides[col_norm] = int(amt_for_color)

    return overrides

def _extract_specs_shop9_regex(
    s_price: str,
    s_color: str,
    color_to_pn: Dict[str, str],
) -> Tuple[Dict[str, int], Dict[str, int],
           List[Tuple[str, int]], List[Tuple[str, int]],
           Dict[str, str], Dict[str, str]]:
    """
    纯正則版：从 price/color 文本中提取。
    返回: abs_map, delta_map, abs_specs, delta_specs,
          color_abs_label_map, color_delta_label_map
    """
    abs_map: Dict[str, int] = {}
    delta_map: Dict[str, int] = {}
    color_abs_label_map: Dict[str, str] = {}
    color_delta_label_map: Dict[str, str] = {}

    abs_list = _extract_abs_prices_regex(s_color) or _extract_abs_prices_regex(s_price)
    deltas = _extract_deltas_regex(s_color) or _extract_deltas_regex(s_price)

    # raw specs（提取阶段的原始标签列表）
    abs_specs: List[Tuple[str, int]] = list(abs_list)
    delta_specs: List[Tuple[str, int]] = list(deltas)

    def _match_label_to_colnorm(tok: str) -> Optional[str]:
        # 原逻辑：candidates = SYNONYM_LOOKUP.get(tok_norm,[])|{tok_norm}; for cand: candn=_norm(cand); if candn==cn or ...
        if not tok:
            return None
        tok_norm = _norm(tok)
        for col_norm in color_to_pn.keys():
            if tok_norm == col_norm:
                return col_norm
        candidates = set(SYNONYM_LOOKUP_NORM.get(tok_norm, []))
        candidates.add(tok_norm)
        for cand in candidates:
            candn = _norm(cand)  # cand 来自 SYNONYM_LOOKUP_NORM 已归一化，_norm 幂等
            for col_norm in color_to_pn.keys():
                cn = _norm(col_norm)
                if candn == cn or candn in cn or cn in candn:
                    return col_norm
        tok_short = re.sub(r"[\s\u3000\-]+", "", tok_norm)
        for col_norm in color_to_pn.keys():
            cn_short = re.sub(r"[\s\u3000\-]+", "", _norm(col_norm))
            if tok_short and (tok_short in cn_short or cn_short in tok_short):
                return col_norm
        return None

    for label_raw, amt in abs_list:
        toks = [t.strip() for t in LABEL_SPLIT_RE_shop9.split(label_raw) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            matched = _match_label_to_colnorm(tok)
            if matched:
                abs_map[matched] = int(amt)
                color_abs_label_map[matched] = tok

    for label_raw, delta in deltas:
        if label_raw == "全色":
            delta_map["ALL"] = int(delta)
            continue
        toks = [t.strip() for t in LABEL_SPLIT_RE_shop9.split(label_raw) if t.strip()]
        for tok in toks:
            if _is_pure_number_token(tok):
                continue
            matched = _match_label_to_colnorm(tok)
            if matched:
                delta_map[matched] = int(delta)
                color_delta_label_map[matched] = tok

    return abs_map, delta_map, abs_specs, delta_specs, color_abs_label_map, color_delta_label_map

# ----------------------------------------------------------------------
# Step 6: LLM 配置 & 核心提取函数
# ----------------------------------------------------------------------

# LLM 相関代码已提取到 shop_cleaners_split_llm/llm_shop9.py
from ..shop_cleaners_split_llm.llm_shop9 import (
    setup_shop9_llm_deps,
    extract_specs_shop9_llm as _extract_specs_shop9_llm,
)

# 注入非 LLM 依赖到 LLM 模块
setup_shop9_llm_deps(
    build_color_aliases_fn=_build_color_aliases,
    map_to_available_color_fn=_map_to_available_color,
    bucket_amount_fn=_bucket_amount,
    norm_cls_fn=_norm_cls,
    direct_abs_overrides_fn=_direct_abs_overrides_for_row,
)

# ----------------------------------------------------------------------
# Step 7: 提取モード調度
# ----------------------------------------------------------------------

def _extract_specs_shop9_dispatch(
    s_price: str,
    s_color: str,
    color_to_pn: Dict[str, str],
    *,
    base_price: Optional[int],
    source_text_raw: str,
    row_index: object = None,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrail
      - "auto":  regex 优先，regex 无颜色结果时 LLM + Guardrail 兜底

    返回 PriceDecomposition
    """
    (abs_map, delta_map, abs_specs, delta_specs, cal, cdl), method = dispatch_extraction(
        EXTRACTION_MODE,
        regex_fn=lambda: _extract_specs_shop9_regex(s_price, s_color, color_to_pn),
        llm_fn=lambda: _extract_specs_shop9_llm(s_price, s_color, color_to_pn, row_index=row_index),
        has_result_fn=lambda r: bool(r[0] or r[1]),  # r = (abs_map, delta_map, ...)
    )

    # regex 路径追加 abs overrides（仅 regex/auto-regex 需要）
    if method == "regex":
        overrides = _direct_abs_overrides_for_row(
            raw_color_text=s_color, color_to_pn=color_to_pn,
        )
        if overrides:
            for col_norm, v in overrides.items():
                abs_map[col_norm] = int(v)
                abs_specs.append((col_norm, int(v)))

    # ---- "ALL" 归一化 ----
    final_delta_specs: List[Tuple[str, int]] = list(delta_specs)
    final_abs_specs: List[Tuple[str, int]] = list(abs_specs)

    if "ALL" in delta_map:
        final_delta_specs = [("全色", delta_map["ALL"])]
        final_abs_specs = []
    elif "ALL" in abs_map:
        final_abs_specs = [("全色", abs_map["ALL"])]
        final_delta_specs = []

    # ---- base_price=None 时仅 abs 路径有效 ----
    if base_price is None:
        decomp_delta: List[Tuple[str, int]] = []
        decomp_base = 0
    else:
        decomp_delta = final_delta_specs
        decomp_base = base_price

    return PriceDecomposition(
        base_price=decomp_base,
        delta_specs=decomp_delta,
        abs_specs=final_abs_specs,
        extraction_method=method,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop9(
    df: pd.DataFrame,
    debug: bool = True,
    debug_limit: int = 30,
) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq, extraction_mode=EXTRACTION_MODE)
    _log_seq += 1

    _log_seq = validate_columns(df, [COL_MODEL, COL_PRICE, COL_COLOR, COL_TIME],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    if df.empty:
        log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=0, start_time=start_time, log_seq=_log_seq)
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_from_db()
    pn_map = _build_color_map(info_df)

    model_norm_ser = df[COL_MODEL].map(_normalize_model_generic)
    cap_gb_ser = df[COL_MODEL].map(_parse_capacity_gb)
    recorded_at_ser = df[COL_TIME].map(lambda x: parse_dt_aware(x))

    rows: List[dict] = []

    for i in range(len(df)):
        raw_model = df[COL_MODEL].iat[i]
        m = model_norm_ser.iat[i]
        c = cap_gb_ser.iat[i]
        t = recorded_at_ser.iat[i]
        raw_price_cell = df[COL_PRICE].iat[i]
        raw_color_cell = df[COL_COLOR].iat[i]

        if not m or pd.isna(c):
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="model_or_cap_missing", log_seq=_log_seq,
                         raw_model=str(raw_model), model_norm=str(m))
            _log_seq += 1
            continue
        c = int(c)

        key = (m, c)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="no_pn_map", log_seq=_log_seq,
                         model_norm=str(m), capacity_gb=c)
            _log_seq += 1
            continue

        s_color = str(raw_color_cell) if raw_color_cell is not None else ""
        s_price = str(raw_price_cell) if raw_price_cell is not None else ""

        # base price：优先 price 列，其次 color 列（保留原逻辑）
        base_price = extract_price_yen(s_price) or extract_price_yen(s_color)

        # source_text_raw_full：两列合并
        source_text_raw_full = f"{s_price} | {s_color}" if s_price and s_color else (s_price or s_color)

        # ---- 提取 ----
        decomp = _extract_specs_shop9_dispatch(
            s_price, s_color, color_to_pn,
            base_price=base_price,
            source_text_raw=source_text_raw_full,
            row_index=i,
        )
        decomp_emit_default = base_price is not None

        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_to_pn,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=t,
            emit_default_rows=decomp_emit_default,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=i,
            model_text=str(raw_model),
            model_norm=str(m),
            capacity_gb=c,
        )
        rows.extend(new_rows)

    out = assemble_output_df(rows)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=start_time, log_seq=_log_seq)

    return out
