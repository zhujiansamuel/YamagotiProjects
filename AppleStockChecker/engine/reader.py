"""
PG 批量读取 PurchasingShopPriceRecord → pandas DataFrame。
参考: docs/REFACTOR_PLAN_V1.md §6.2 Step 1
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
from django.db import connection

logger = logging.getLogger(__name__)

_SQL = """\
SELECT
    shop_id,
    iphone_id,
    price_new,
    price_grade_a,
    price_grade_b,
    recorded_at
FROM "AppleStockChecker_purchasingshoppricerecord"
WHERE recorded_at >= %s
  AND recorded_at <  %s
  {shop_clause}
  {iphone_clause}
ORDER BY recorded_at
"""


def read_price_records(
    date_from: date | datetime,
    date_to: date | datetime,
    *,
    shop_ids: list[int] | None = None,
    iphone_ids: list[int] | None = None,
) -> pd.DataFrame:
    """从 PG 批量读取原始价格记录。

    Returns
    -------
    DataFrame  columns: shop_id, iphone_id, price_new, price_a, price_b, recorded_at
    """
    params: list = [date_from, date_to]
    shop_clause = ""
    iphone_clause = ""

    if shop_ids:
        shop_clause = f"AND shop_id IN ({','.join('%s' for _ in shop_ids)})"
        params.extend(shop_ids)
    if iphone_ids:
        iphone_clause = f"AND iphone_id IN ({','.join('%s' for _ in iphone_ids)})"
        params.extend(iphone_ids)

    sql = _SQL.format(shop_clause=shop_clause, iphone_clause=iphone_clause)
    logger.info("read_price_records  %s → %s  shops=%s iphones=%s",
                date_from, date_to, shop_ids, iphone_ids)

    df = pd.read_sql(sql, connection, params=params)

    # 统一列名
    df.rename(columns={
        "price_grade_a": "price_a",
        "price_grade_b": "price_b",
    }, inplace=True)

    logger.info("read_price_records  rows=%d", len(df))
    return df
