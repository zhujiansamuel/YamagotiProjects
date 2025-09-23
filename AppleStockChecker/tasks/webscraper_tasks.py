# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import pandas as pd
from celery import shared_task

from AppleStockChecker.utils.external_ingest.webscraper import fetch_webscraper_export_sync
from AppleStockChecker.services.external_ingest_service import ingest_external_dataframe

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,          # 30s,60s,120s...
    retry_kwargs={"max_retries": 5},
)
def task_process_webscraper_job(self, job_id: str, source_name: str, *,
                                dry_run: bool = False, create_shop: bool = True,
                                dedupe: bool = True, upsert: bool = False,
                                batch_id: str | None = None) -> dict:
    content = fetch_webscraper_export_sync(job_id, format="csv")
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    result = ingest_external_dataframe(
        source_name, df,
        dry_run=dry_run, pn_only=True, create_shop=create_shop,
        dedupe=dedupe, upsert=upsert, batch_id=batch_id
    )
    return result
