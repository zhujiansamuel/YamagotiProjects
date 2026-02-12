from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
    _normalize_amount_text,
    normalize_text_basic,
    safe_to_text,
    extract_price_yen,
)
import re
import time
import logging
import pandas as pd


"""
  shop7（買取ホムラ）清洗器 — 数据处理流程
  ============================================

    原始 DataFrame (data, data2, data3, time-scraped)
      │
      ├─ Step 1: 输入验证 & 过滤
      │   └─ 必要列检查、time-scraped 非空过滤
      │
      ├─ Step 2: 批量解析字段
      │   ├─ _norm_model_for_shop7()    ← 短写扩展 + _normalize_model_generic
      │   ├─ _parse_capacity_gb()       ← 容量解析
      │   ├─ extract_price_yen()         ← 去掉"新品/未開封"后取价格（公共函数）
      │   └─ parse_dt_aware()           ← 时间解析
      │
      ├─ Step 3: 颜色减价解析（下一行检测）
      │   └─ _parse_color_deltas_shop7()
      │       ├─ DELTA_RE               ← 核心正则: 标签+金额
      │       └─ _normalize_amount_text()  ← 金额文本 → int（公共函数）
      │
      ├─ Step 4: label → color 匹配
      │   └─ _label_matches_color_shop7()  ← 精确 | 子串匹配
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


def _parse_color_deltas_shop7(text: str) -> List[Tuple[str, int]]:
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
        for tok in re.split(r"[／/、，,・\s]+", labels_part):
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
                for tok in re.split(r"[／/、，,・\s]+", labels_part):
                    tok = tok.strip()
                    if tok:
                        res.append((_norm_strip(tok), delta))

    return res


# ----------------------------------------------------------------------
# Step 4: 颜色匹配
# ----------------------------------------------------------------------

def _label_matches_color_shop7(label_raw: str, col_raw: str, col_norm: str) -> bool:
    """
    判断提取的颜色标签是否匹配目标颜色。
    匹配策略：精确(归一) | 原文子串。
    """
    label_norm = _norm_strip(label_raw)
    # 精确匹配归一化颜色
    if label_norm == col_norm:
        return True
    # 标签是原文颜色的子串
    if label_norm and label_norm in _norm_strip(col_raw):
        return True
    if label_raw and label_raw in str(col_raw):
        return True
    return False


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop7(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    t_start = time.time()
    _log_seq = 0

    logger.info(
        "shop7 cleaner started",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "log_seq": _log_seq,
            "input_rows": len(df),
            "extraction_mode": "regex",
        },
    )
    _log_seq += 1

    # ── Step 1: 加载参考数据 & 输入验证 ──────────────────────────────
    need_cols = ["data", "data2", "data3", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "log_seq": _log_seq,
                    "missing_column": c,
                    "available_columns": list(df.columns),
                },
            )
            _log_seq += 1
            raise ValueError(f"shop7 清洗器缺少必要列：{c}")

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
            logger.debug(
                f"Row {i}: skip (model/cap missing)",
                extra={
                    "event_type": "row_skip",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": i,
                    "model_text": model_text,
                    "model_norm": model_norm or "",
                    "capacity_gb": int(c) if pd.notna(c) else None,
                    "skip_reason": "model_or_cap_missing",
                },
            )
            continue

        cap_gb = int(c)
        key = (model_norm, cap_gb)
        color_map = pn_map.get(key)
        if not color_map:
            _log_seq += 1
            logger.debug(
                f"Row {i}: skip (no color_map for key={key})",
                extra={
                    "event_type": "row_skip",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": i,
                    "model_text": model_text,
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "skip_reason": "no_color_map",
                },
            )
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
                labels_and_deltas = _parse_color_deltas_shop7(nxt_data2)

        extraction_method = "regex" if labels_and_deltas else "none"

        # ---- extraction_result (DEBUG) ----
        available_colors_list = [
            {"color_norm": cn, "part_number": pn, "color_raw": cr}
            for cn, (pn, cr) in color_map.items()
        ]

        _log_seq += 1
        logger.debug(
            "Extraction result",
            extra={
                "event_type": "extraction_result",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "source_text_raw": _truncate_for_log(source_text_raw_full, 200),
                "source_text_raw_full": source_text_raw_full,
                "source_text_normalized": _truncate_for_log(
                    normalize_text_basic(source_text_raw_full) if source_text_raw_full else "", 200
                ),
                "extraction_method": extraction_method,
                "labels_and_deltas": [
                    {"label": label, "delta": delta}
                    for label, delta in labels_and_deltas
                ],
                "abs_prices": [],
                "labels_extracted_count": len(labels_and_deltas),
                "abs_prices_count": 0,
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_map),
            },
        )

        # ---- Step 4: label → color 匹配 ----
        color_deltas: Dict[str, int] = {}
        color_delta_label_map: Dict[str, str] = {}

        if labels_and_deltas:
            for label_raw, delta_val in labels_and_deltas:
                matched_colors: List[str] = []
                matched_pns: List[str] = []

                for col_norm, (pn, col_raw) in color_map.items():
                    if _label_matches_color_shop7(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta_val
                        color_delta_label_map[col_norm] = label_raw
                        matched_colors.append(col_norm)
                        matched_pns.append(pn)

                # label_matching (DEBUG) / label_no_match (WARNING)
                if matched_colors:
                    _log_seq += 1
                    logger.debug(
                        f"Label matching (delta): {label_raw}",
                        extra={
                            "event_type": "label_matching",
                            "log_seq": _log_seq,
                            "shop_name": SHOP_NAME,
                            "cleaner_name": CLEANER_NAME,
                            "row_index": i,
                            "model_text": model_text,
                            "model_norm": model_norm,
                            "capacity_gb": cap_gb,
                            "base_price": base_price,
                            "label": label_raw,
                            "delta": delta_val,
                            "match_type": "delta",
                            "matched_colors": matched_colors,
                            "matched_part_numbers": matched_pns,
                            "match_count": len(matched_colors),
                            "source_text_raw_full": source_text_raw_full,
                            "labels_and_deltas": [
                                {"label": label, "delta": delta}
                                for label, delta in labels_and_deltas
                            ],
                        },
                    )
                else:
                    _log_seq += 1
                    logger.warning(
                        f"Label not matched (delta): {label_raw}",
                        extra={
                            "event_type": "label_no_match",
                            "log_seq": _log_seq,
                            "shop_name": SHOP_NAME,
                            "cleaner_name": CLEANER_NAME,
                            "row_index": i,
                            "model_text": model_text,
                            "model_norm": model_norm,
                            "capacity_gb": cap_gb,
                            "base_price": base_price,
                            "label": label_raw,
                            "delta": delta_val,
                            "match_type": "delta",
                            "available_colors": [cn for cn in color_map.keys()],
                            "source_text_raw_full": source_text_raw_full,
                            "labels_and_deltas": [
                                {"label": label, "delta": delta}
                                for label, delta in labels_and_deltas
                            ],
                        },
                    )

        # ---- Step 5: 为每个颜色生成输出记录 ----
        current_row_records: List[dict] = []
        colors_matched = 0

        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm in color_deltas:
                effective_source = "matched_label"
                matched_label = color_delta_label_map.get(col_norm, col_norm)
                spec_value = int(color_deltas[col_norm])
                final_price = base_price + spec_value
            else:
                effective_source = "default_zero"
                matched_label = None
                spec_value = None
                final_price = base_price

            if effective_source != "default_zero":
                colors_matched += 1

            rows.append({
                "part_number": pn,
                "shop_name": SHOP_NAME,
                "price_new": int(final_price),
                "recorded_at": rec_at,
            })

            current_row_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "final_price": int(final_price),
                "recorded_at": rec_at,
                "effective_source": effective_source,
                "matched_label": matched_label,
                "spec_value": spec_value,
            })

            # output_record (DEBUG)
            _log_seq += 1
            logger.debug(
                f"Output record: {pn}",
                extra={
                    "event_type": "output_record",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": i,
                    "model_text": model_text,
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "part_number": pn,
                    "color_norm": col_norm,
                    "color_raw": col_raw,
                    "base_price": base_price,
                    "final_price": int(final_price),
                    "effective_source": effective_source,
                    "matched_label": matched_label,
                    "spec_value": spec_value,
                    "recorded_at": str(rec_at) if rec_at else None,
                    "source_text_raw_full": source_text_raw_full,
                    "labels_and_deltas": [
                        {"label": label, "delta": delta}
                        for label, delta in labels_and_deltas
                    ],
                },
            )

        # ---- row_processing_summary (DEBUG) ----
        all_spec_values = [
            r["spec_value"] for r in current_row_records
            if r["spec_value"] is not None
        ]

        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "source_text_raw_full": source_text_raw_full,
                "abs_applied_details": [],
                "delta_applied_details": [
                    {
                        "pn": r["part_number"],
                        "color": r["color_norm"],
                        "final_price": r["final_price"],
                        "matched_label": r["matched_label"],
                        "spec_value": r["spec_value"],
                    }
                    for r in current_row_records
                    if r["effective_source"] == "matched_label"
                ],
                "default_applied_pns": [
                    r["part_number"]
                    for r in current_row_records
                    if r["effective_source"] == "default_zero"
                ],
            },
        )

        # ---- row_processing_summary (INFO) ----
        _log_seq += 1
        _model_display = f"{model_text[:28]}" if len(model_text) > 28 else model_text
        logger.info(
            f"Row {i:<3d} | {_model_display:<28s} | deltas: {len(labels_and_deltas):<2d} | abs: 0  | matched: {colors_matched:<2d} | records: {len(current_row_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "source_text_raw_preview": _truncate_for_log(source_text_raw_full, 100),
                "extraction_method": extraction_method,
                "labels_extracted_count": len(labels_and_deltas),
                "abs_prices_extracted_count": 0,
                "colors_in_catalog": len(color_map),
                "colors_matched_count": colors_matched,
                "output_records_count": len(current_row_records),
                "has_discounted_colors": any(v != 0 for v in all_spec_values),
                "min_delta": min(all_spec_values) if all_spec_values else 0,
                "max_delta": max(all_spec_values) if all_spec_values else 0,
            },
        )

    # ── 构建输出 DataFrame ───────────────────────────────────────────
    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed = round(time.time() - t_start, 2)
    logger.info(
        "shop7 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "log_seq": _log_seq,
            "output_rows": len(out),
            "elapsed_seconds": elapsed,
        },
    )

    return out
