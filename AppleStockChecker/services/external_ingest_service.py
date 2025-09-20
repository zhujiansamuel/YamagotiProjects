# AppleStockChecker/services/external_ingest_service.py
from __future__ import annotations
import asyncio
from typing import List, Dict, Any
import pandas as pd

from django.db import transaction
from django.utils import timezone

from AppleStockChecker.utils.external_ingest.helpers import async_http_get_bytes, read_csv_smart
from AppleStockChecker.utils.external_ingest.registry import run_cleaner
from AppleStockChecker.models import Iphone, SecondHandShop, PurchasingShopPriceRecord

async def fetch_sources_async(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 源结构：{"name": "sample_a", "url": "...", "headers": {...}}
    tasks = [async_http_get_bytes(src["url"], headers=src.get("headers")) for src in sources]
    contents = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for src, content in zip(sources, contents):
        if isinstance(content, Exception):
            results.append({"source": src, "error": str(content), "df": None})
        else:
            try:
                df = read_csv_smart(content)
                results.append({"source": src, "error": None, "df": df})
            except Exception as e:
                results.append({"source": src, "error": str(e), "df": None})
    return results

def ingest_external_sources(sources: List[Dict[str, Any]], *, dry_run: bool = False) -> Dict[str, Any]:
    """
    sources: [{"name": "sample_a", "url": "...", "headers": {...}}, ...]
    return: 统计与预览
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pulled = loop.run_until_complete(fetch_sources_async(sources))
    loop.close()

    cleaned_tables: List[pd.DataFrame] = []
    pull_errors = []
    for item in pulled:
        src = item["source"]
        if item["error"]:
            pull_errors.append({"source": src, "error": item["error"]})
            continue
        try:
            df_clean = run_cleaner(src["name"], item["df"])
            if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
                pull_errors.append({"source": src, "error": "清洗后为空"})
                continue
            cleaned_tables.append(df_clean)
        except Exception as e:
            pull_errors.append({"source": src, "error": f"清洗失败: {e}"})

    if not cleaned_tables:
        return {"inserted": 0, "unmatched": [], "errors": pull_errors, "preview": []}

    # 合并所有清洗后的 df（规范列）
    df_all = pd.concat(cleaned_tables, ignore_index=True)
    # 规范列缺失填空
    for col in ["part_number","shop_name","price_new","price_grade_a","price_grade_b","recorded_at","shop_address"]:
        if col not in df_all.columns:
            df_all[col] = None

    inserted = 0
    unmatched = []
    preview_rows = []

    # 逐行处理
    for idx, row in df_all.iterrows():
        pn = str(row.get("part_number") or "").strip()
        shop_name = str(row.get("shop_name") or "").strip()
        if not pn or not shop_name:
            unmatched.append({"row": int(idx), "reason": "缺少 part_number 或 shop_name", "data": row.to_dict()})
            continue

        iphone = Iphone.objects.filter(part_number=pn).first()
        if not iphone:
            unmatched.append({"row": int(idx), "reason": f"未找到 iPhone(PN={pn})", "data": row.to_dict()})
            continue

        # 门店；用 name+address 判唯一；address 可空
        shop_address = (row.get("shop_address") or "").strip() if isinstance(row.get("shop_address"), str) else ""
        shop = SecondHandShop.objects.filter(name=shop_name, address=shop_address).first()
        if not shop:
            shop = SecondHandShop.objects.create(name=shop_name, address=shop_address)

        price_new = row.get("price_new")
        price_a   = row.get("price_grade_a")
        price_b   = row.get("price_grade_b")
        recorded_at = row.get("recorded_at") or timezone.now()

        if dry_run:
            inserted += 1
            if len(preview_rows) < 10:
                preview_rows.append({
                    "shop_name": shop_name, "part_number": pn,
                    "price_new": price_new, "price_grade_a": price_a, "price_grade_b": price_b,
                    "recorded_at": str(recorded_at)
                })
            continue

        with transaction.atomic():
            rec = PurchasingShopPriceRecord.objects.create(
                shop=shop,
                iphone=iphone,
                price_new=int(price_new) if pd.notna(price_new) else 0,
                price_grade_a=int(price_a) if pd.notna(price_a) and price_a is not None else None,
                price_grade_b=int(price_b) if pd.notna(price_b) and price_b is not None else None,
            )
            # 覆盖 recorded_at
            PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=recorded_at)
            inserted += 1

            if len(preview_rows) < 10:
                preview_rows.append({
                    "shop_name": shop_name, "part_number": pn,
                    "price_new": price_new, "price_grade_a": price_a, "price_grade_b": price_b,
                    "recorded_at": str(recorded_at)
                })

    return {
        "inserted": inserted,
        "unmatched": unmatched[:50],
        "errors": pull_errors,
        "preview": preview_rows,
        "rows_total": int(df_all.shape[0])
    }
