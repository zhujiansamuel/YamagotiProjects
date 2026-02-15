from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _norm_strip,
    _normalize_amount_text,
    normalize_text_basic,
    safe_to_text,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    LABEL_SPLIT_RE_shop7,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    log_row_skip,
    validate_columns,
)
import re
import time
import logging
import pandas as pd


"""
shop7 清洗器 — 買取ホムラ

  原始 DataFrame (data, data2, data3, time-scraped)
    │
    ├─ Step 1: 输入验证 & 过滤
    │   └─ 必要列检查、time-scraped 非空过滤
    │
    ├─ Step 2: 批量解析字段
    │   ├─ _norm_model_for_shop7()    ← 短写扩展 + _normalize_model_generic (cleaner_tools)
    │   ├─ _parse_capacity_gb()       ← 容量解析（cleaner_tools）
    │   ├─ extract_price_yen()         ← 基础价提取（cleaner_tools）
    │   └─ parse_dt_aware()           ← 时间解析
    │
    ├─ Step 3: 颜色减价解析（下一行检测）
    │   └─ _extract_specs_shop7_regex()
    │       ├─ DELTA_RE               ← 核心正则: 标签+金额
    │       └─ _normalize_amount_text()  ← 金额文本 → int（cleaner_tools）
    │
    ├─ Step 4: label → color 匹配
    │   └─ _label_matches_color_unified()  ← 统一匹配（cleaner_tools）
    │
    └─ Step 5: part_number 输出
        └─ base_price + color delta → final price
"""

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop7"
SHOP_NAME = "買取ホムラ"

# ----------------------------------------------------------------------
# Step 2a: 机型归一化
# ----------------------------------------------------------------------

def _norm_model_for_shop7(s: Optional[str]) -> str:
    """
    shop7 的 model 字段宽松归一化：
      - 跳过纯数字行（shop7 数据中存在纯数字的价格/编号行混入 data2 列）
      - 其余交给公共 _normalize_model_generic 处理
    """
    if s is None:
        return ""
    txt = str(s).strip()
    if not txt:
        return ""
    # shop7 特有：data2 列可能混入纯数字行，提前排除
    if re.fullmatch(r'[\d\-\.\s]+', txt):
        return ""

    return _normalize_model_generic(txt)


# ----------------------------------------------------------------------
# Step 3: 颜色减价解析
# ----------------------------------------------------------------------

DELTA_RE = re.compile(
    r"(?P<labels>[^\d¥￥円\+\-−－]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[0-9０-９,，]+)",
    re.UNICODE,
)


def _extract_specs_shop7_regex(text: str) -> List[Tuple[str, int]]:
    """
    解析颜色减价文本，返回 [(颜色标签, delta金额)] 列表。
    例如: "シルバー/ディープブルー-3000" → [("シルバー", -3000), ("ディープブルー", -3000)]
    """
    res: List[Tuple[str, int]] = []
    if not text or not str(text).strip():
        return res
    s = str(text).strip()

    found = False
    for m in DELTA_RE.finditer(s):
        found = True
        labels_part = m.group("labels") or ""
        sign = m.group("sign") or "+"
        amt_txt = m.group("amount")
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            continue
        delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
        for tok in LABEL_SPLIT_RE_shop7.split(labels_part):
            tok = tok.strip()
            if tok:
                res.append((_norm_strip(tok), delta))

    if not found:
        # 退化：如 "シルバー/ディープブルー-3000"
        m2 = re.search(r"(?P<labels>.+?)[\s]*([+\-−－])\s*(?P<amount>[0-9０-９,，]+)", s)
        if m2:
            labels_part = m2.group("labels") or ""
            sign = m2.group(2) or "+"
            amt_txt = m2.group("amount")
            amt = _normalize_amount_text(amt_txt)
            if amt is not None:
                delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
                for tok in LABEL_SPLIT_RE_shop7.split(labels_part):
                    tok = tok.strip()
                    if tok:
                        res.append((_norm_strip(tok), delta))

    return res


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop7(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    t_start = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq, extraction_mode="regex")
    _log_seq += 1

    # ── Step 1: 加载参考数据 & 输入验证 ──────────────────────────────
    _log_seq = validate_columns(df, ["data", "data2", "data3", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    info_df = _load_iphone17_info_df_from_db()

    # ── Step 1b: 行级过滤 ────────────────────────────────────────────
    # time-scraped 为空的行排除
    rows_before = len(df)
    df = df.copy().reset_index(drop=True)
    mask_time_ok = df["time-scraped"].astype(str).str.strip().ne("") & df["time-scraped"].notna()
    df = df[mask_time_ok].reset_index(drop=True)
    rows_dropped_time = rows_before - len(df)

    if df.empty:
        logger.info(
            "shop7 cleaner completed (empty input)",
            extra={
                "event_type": "cleaner_complete",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "log_seq": _log_seq,
                "output_rows": 0,
                "rows_dropped_no_time": rows_dropped_time,
                "elapsed_seconds": round(time.time() - t_start, 2),
            },
        )
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    # ── Step 2: 批量解析 model / capacity / price / recorded_at ──────
    model_norm_series = df["data2"].map(_norm_model_for_shop7)
    cap_gb_series = df["data2"].map(_parse_capacity_gb)
    price_series = df["data3"].map(extract_price_yen)
    recorded_at = df["time-scraped"].map(parse_dt_aware)

    # 构建 (model, cap) → {color_norm: (pn, color_raw)} 映射
    pn_map = _build_color_map(info_df)

    # ── Step 2b: 数据质量摘要 ────────────────────────────────────────
    n_has_price = int(price_series.map(lambda x: x is not None).sum())
    n_has_model = int(model_norm_series.astype(bool).sum())
    n_has_cap = int(cap_gb_series.notna().sum())

    _log_seq += 1
    logger.info(
        f"Data quality: rows={len(df)} has_price={n_has_price} has_model={n_has_model} has_cap={n_has_cap} dropped_no_time={rows_dropped_time}",
        extra={
            "event_type": "data_quality_summary",
            "log_seq": _log_seq,
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "total_rows": len(df),
            "rows_dropped_no_time": rows_dropped_time,
            "rows_with_price": n_has_price,
            "rows_with_model": n_has_model,
            "rows_with_capacity": n_has_cap,
        },
    )

    # ── Step 3~5: 主循环 — 逐行解析颜色减价 & 匹配输出 ──────────────
    rows: List[dict] = []
    n = len(df)

    for i in range(n):
        # ---- 取 base_price ----
        base_price = price_series.iat[i]
        if base_price is None:
            continue
        base_price = int(base_price)

        # ---- 取 model / capacity / recorded_at ----
        model_text = safe_to_text(df["data2"].iat[i]).strip()
        model_norm = model_norm_series.iat[i]
        c = cap_gb_series.iat[i]
        rec_at = recorded_at.iat[i]

        if not model_norm or pd.isna(c):
            _log_seq += 1
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="model_or_cap_missing", log_seq=_log_seq,
                         model_text=model_text, model_norm=model_norm or "",
                         capacity_gb=int(c) if pd.notna(c) else None)
            continue

        cap_gb = int(c)
        key = (model_norm, cap_gb)
        color_map = pn_map.get(key)
        if not color_map:
            _log_seq += 1
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=i, skip_reason="no_color_map", log_seq=_log_seq,
                         model_text=model_text, model_norm=model_norm, capacity_gb=cap_gb)
            continue

        # ---- Step 3: 下一行是否为颜色减价行 ----
        source_text_raw_full = ""
        labels_and_deltas: List[Tuple[str, int]] = []
        j = i + 1
        if j < n:
            nxt_data2 = safe_to_text(df["data2"].iat[j]).strip()
            nxt_price_cell = safe_to_text(df["data3"].iat[j]).strip()
            nxt_price_val = extract_price_yen(nxt_price_cell) if nxt_price_cell else None
            is_color_line = bool(nxt_data2) and (nxt_price_val is None)

            if is_color_line:
                source_text_raw_full = nxt_data2
                labels_and_deltas = _extract_specs_shop7_regex(nxt_data2)

        extraction_method = "regex" if labels_and_deltas else "none"

        # ---- Step 4~5: 匹配 + 定价 + 输出（公共函数） ----
        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=labels_and_deltas,
            abs_specs=[],
            extraction_method=extraction_method,
            source_text_raw=source_text_raw_full,
        )

        new_rows, _log_seq = resolve_color_prices(
            decomp, color_map, _label_matches_color_unified,
            shop_name=SHOP_NAME, cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            logger=logger, log_seq_start=_log_seq,
            row_index=i, model_text=model_text,
            model_norm=model_norm, capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    # ── 构建输出 DataFrame ───────────────────────────────────────────
    out = assemble_output_df(rows)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=t_start, log_seq=_log_seq)

    return out
