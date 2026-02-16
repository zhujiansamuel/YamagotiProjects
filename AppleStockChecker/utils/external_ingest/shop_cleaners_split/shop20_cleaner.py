from __future__ import annotations
"""
shop20 清洗器

  原始 DataFrame
    │
    ├─ _load_iphone17_info_df_from_db()    ← Step 1: 机型信息（cleaner_tools 共用）
    ├─ _extract_jan_digits()               ← Step 2: JAN 提取（cleaner_tools 共用）
    ├─ _build_jan_map()                    ← Step 2b: JAN→PN 映射（cleaner_tools 共用）
    └─ clean_shop20()                      ← Step 3: 主函数，输出 part_number / price_new / recorded_at
"""
from typing import Dict, Optional, List
import json
import logging
import time

import pandas as pd

from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import assemble_output_df, validate_columns, _load_iphone17_info_df_from_db, _extract_jan_digits, _build_jan_map, log_cleaner_start, log_cleaner_complete

logger = logging.getLogger(__name__)

def _coerce_price(v) -> Optional[int]:
    """goodsPrice 既可能是数字也可能是字符串，统一转 int（日元）"""
    if v is None:
        return None
    if isinstance(v, (int, float)) and pd.notna(v):
        return int(round(float(v)))
    return to_int_yen(v)

def clean_shop20(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 (shop20.csv):
      - json: 形如 {""success"":true,""data"":[...]} 的 JSON 文本（需先把 "" → "）
      - time-scraped: 抓取时间
    输出:
      - part_number, shop_name(=買取当番), price_new, recorded_at
    规则:
      - 对 json['data'] 的每个项，取 jancode → 在信息表中匹配 PN；取 goodsPrice → price_new
      - 无法解析/缺少 jancode 或 goodsPrice 的条目跳过
      - recorded_at 使用该行的 time-scraped
    """
    start_time = time.time()
    log_cleaner_start(logger, cleaner_name="shop20", shop_name="毎日買取", input_rows=len(df))

    # 必要列检查
    validate_columns(df, ["json", "time-scraped"],
                     cleaner_name="shop20", shop_name="毎日買取")

    if df.empty:
        log_cleaner_complete(logger, cleaner_name="shop20", shop_name="毎日買取", input_rows=len(df), output_records=0, start_time=start_time)
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_from_db()
    jan_map = _build_jan_map(info_df)

    rows: List[dict] = []

    for _, row in df.iterrows():
        raw_json = row.get("json")
        if not isinstance(raw_json, str) or not raw_json.strip():
            continue

        # 将 CSV 内部双引号转为标准 JSON 引号
        # 例如 {""success"":true} -> {"success":true}
        s = raw_json.replace('""', '"').strip()

        try:
            payload = json.loads(s)
        except Exception:
            # 解析失败，尝试去掉可能的 BOM/不可见字符后再试
            s2 = s.lstrip("\ufeff").strip()
            try:
                payload = json.loads(s2)
            except Exception:
                continue

        data = payload.get("data")
        if not isinstance(data, list):
            continue

        rec_at = parse_dt_aware(row.get("time-scraped"))

        for item in data:
            if not isinstance(item, dict):
                continue

            jan_digits = _extract_jan_digits(item.get("jancode") or item.get("jan"))
            if not jan_digits:
                # 一些接口把 JAN 也写进 keywords，如 "... 4549995xxxxxxx"
                jan_digits = _extract_jan_digits(item.get("keywords"))

            if not jan_digits:
                continue

            pn = jan_map.get(jan_digits)
            if not pn:
                # 信息表里找不到该 JAN → 跳过（只输出已知机型）
                continue

            price = _coerce_price(item.get("goodsPrice"))
            if price is None:
                # 无价格（或无法解析）→ 跳过
                continue

            rows.append({
                "part_number": pn,
                "shop_name": "毎日買取",
                "price_new": int(price),
                "recorded_at": rec_at,
            })

    out = assemble_output_df(rows)
    log_cleaner_complete(logger, cleaner_name="shop20", shop_name="毎日買取", input_rows=len(df), output_records=len(out), start_time=start_time)
    return out
