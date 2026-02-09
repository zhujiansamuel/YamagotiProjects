# AppleStockChecker/services/external_ingest_service.py
from __future__ import annotations
import asyncio
from typing import List, Dict, Any
import pandas as pd
from typing import Optional
import uuid

from django.db import transaction
from django.utils import timezone

from AppleStockChecker.utils.external_ingest.registry import run_cleaner
from AppleStockChecker.models import Iphone, SecondHandShop, PurchasingShopPriceRecord

def ingest_external_dataframe(
    source_name: str,
    df: pd.DataFrame,
    *,
    dry_run: bool = False,
    pn_only: bool = True,
    create_shop: bool = True,
    dedupe: bool = True,
    upsert: bool = False,
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - dedupe=True: 遇到 (shop, iphone, recorded_at) 已存在则跳过（dedup_skipped++)
    - upsert=True: 在 dedupe 为真时，遇到已存在则更新价格（updated++）
    - batch_id: 本批次标识（uuid4）。未传则自动生成；dry_run 模式仅返回，不写库。
    """
    if batch_id:
        try:
            batch_uuid = uuid.UUID(str(batch_id))
        except Exception:
            batch_uuid = uuid.uuid4()
    else:
        batch_uuid = uuid.uuid4()

    # 1) 清洗（与原来一致）...
    df_clean = run_cleaner(source_name, df)
    if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
        return {"inserted": 0, "unmatched": [], "errors": [{"source": source_name, "error": "清洗后为空"}], "preview": [], "rows_total": 0, "batch_id": str(batch_uuid)}

    # 2) 规范列（与原来一致）...
    for col in ["part_number","shop_name","price_new","price_grade_a","price_grade_b","recorded_at","shop_address"]:
        if col not in df_clean.columns:
            df_clean[col] = None

    inserted = 0
    updated = 0
    dedup_skipped = 0
    unmatched = []
    preview_rows = []

    for idx, row in df_clean.iterrows():
        pn = str(row.get("part_number") or "").strip()
        shop_name = str(row.get("shop_name") or "").strip()
        if not pn or not shop_name:
            unmatched.append({"row": int(idx), "reason": "缺少 part_number 或 shop_name", "data": row.to_dict()})
            continue

        iphone = Iphone.objects.filter(part_number=pn).first()
        if not iphone:
            unmatched.append({"row": int(idx), "reason": f"未找到 iPhone(PN={pn})", "data": row.to_dict()})
            continue

        shop_address = (row.get("shop_address") or "").strip() if isinstance(row.get("shop_address"), str) else ""
        shop = SecondHandShop.objects.filter(name=shop_name, address=shop_address).first()
        if not shop:
            if not create_shop:
                unmatched.append({"row": int(idx), "reason": f"门店不存在且 create_shop=0: {shop_name}/{shop_address}", "data": row.to_dict()})
                continue
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
                    "recorded_at": str(recorded_at), "batch_id": str(batch_uuid),
                })
            continue

        with transaction.atomic():
            existed = None
            if dedupe:
                existed = PurchasingShopPriceRecord.objects.filter(
                    shop=shop, iphone=iphone, recorded_at=recorded_at
                ).first()

            if existed:
                if upsert:
                    changed = False
                    if price_new is not None and existed.price_new != int(price_new):
                        existed.price_new = int(price_new); changed = True
                    if price_a is not None and (existed.price_grade_a or None) != (None if pd.isna(price_a) else int(price_a)):
                        existed.price_grade_a = None if pd.isna(price_a) else int(price_a); changed = True
                    if price_b is not None and (existed.price_grade_b or None) != (None if pd.isna(price_b) else int(price_b)):
                        existed.price_grade_b = None if pd.isna(price_b) else int(price_b); changed = True
                    if changed:
                        # upsert 不改 recorded_at；可以写上本次 batch_id 做追踪
                        existed.batch_id = batch_uuid
                        existed.save(update_fields=["price_new","price_grade_a","price_grade_b","batch_id"])
                        updated += 1
                    else:
                        dedup_skipped += 1
                else:
                    dedup_skipped += 1
            else:
                rec = PurchasingShopPriceRecord.objects.create(
                    shop=shop,
                    iphone=iphone,
                    price_new=int(price_new) if pd.notna(price_new) else 0,
                    price_grade_a=int(price_a) if pd.notna(price_a) and price_a is not None else None,
                    price_grade_b=int(price_b) if pd.notna(price_b) and price_b is not None else None,
                    batch_id=batch_uuid,
                )
                PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=recorded_at)
                inserted += 1
                if len(preview_rows) < 10:
                    preview_rows.append({
                        "shop_name": shop_name, "part_number": pn,
                        "price_new": price_new, "price_grade_a": price_a, "price_grade_b": price_b,
                        "recorded_at": str(recorded_at), "batch_id": str(batch_uuid),
                    })

    return {
        "inserted": inserted,
        "updated": updated,
        "dedup_skipped": dedup_skipped,
        "unmatched": unmatched[:50],
        "errors": [],
        "preview": preview_rows,
        "rows_total": int(df_clean.shape[0]),
        "batch_id": str(batch_uuid),
    }