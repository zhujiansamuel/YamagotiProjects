# -*- coding: utf-8 -*-
"""
WebScraper 任务模块

任务分为两类：
1. 数据接收任务（webscraper 队列）：接收数据 → 存 Redis → 触发清洗任务
2. 数据清洗任务（shop_* 队列）：从 Redis 读取 → 清洗 → 写库

架构说明：
- task_process_xlsx: 解析 xlsx/csv 文件 → 存 Redis → 触发 task_clean_shop_data
- task_process_webscraper_job: 拉取 WebScraper 数据 → 存 Redis → 触发 task_clean_shop_data
- task_ingest_json_shop1: 接收 JSON → 存 Redis → 触发 task_clean_shop1_json
- task_clean_shop_data: 通用清洗任务（动态路由到各店铺队列）
- task_clean_shop1_json: shop1 JSON 专用清洗任务
"""
from __future__ import annotations

import io
import re
import uuid
from typing import Optional, Dict, Any

import pandas as pd
from celery import shared_task
from django.db import transaction
from django.db.utils import OperationalError
from django.utils import timezone

from AppleStockChecker.models import Iphone, SecondHandShop, PurchasingShopPriceRecord
from AppleStockChecker.utils.external_ingest.registry import run_cleaner
from AppleStockChecker.utils.external_ingest.webscraper import fetch_webscraper_export_sync
from AppleStockChecker.utils.redis_temp_storage import (
    store_dataframe,
    retrieve_dataframe,
    make_redis_key,
)
from AppleStockChecker.utils.shop_queue_mapping import (
    get_shop_queue,
    get_cleaner_name,
)


# =============================================================================
# 文件解析工具函数
# =============================================================================

_ENGINE_HINT = {
    "xlsx": ("openpyxl", "pip install openpyxl"),
    "xlsm": ("openpyxl", "pip install openpyxl"),
    "xls": ("xlrd", "pip install 'xlrd<2.0'"),
    "ods": ("odf", "pip install odfpy"),
    "xlsb": ("pyxlsb", "pip install pyxlsb"),
    "csv": (None, None),
}


def _suffix(filename: str) -> str:
    m = re.search(r"\.([A-Za-z0-9]+)$", (filename or "").strip())
    return m.group(1).lower() if m else ""


def _read_tabular(filename: str, raw: bytes) -> pd.DataFrame:
    """
    根据后缀与可用引擎读取为 DataFrame；缺依赖时给出明确提示。
    """
    suf = _suffix(filename)
    buf = io.BytesIO(raw or b"")

    if suf == "csv":
        # 允许 UTF-8 / UTF-8-SIG / Shift-JIS 常见编码
        for enc in ("utf-8-sig", "utf-8", "cp932"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except Exception:
                continue
        buf.seek(0)
        return pd.read_csv(buf)  # 最后一次由 pandas 猜

    if suf in ("xlsx", "xlsm", "xls", "ods", "xlsb"):
        engine, hint = _ENGINE_HINT[suf]
        # 优先尝试推荐引擎
        if engine:
            try:
                buf.seek(0)
                return pd.read_excel(buf, engine=engine)
            except ImportError:
                raise RuntimeError(f"缺少依赖：{engine}。请先安装：{hint}")
            except Exception as e:
                # 再给一次"自动引擎"机会（pandas 自探测）
                try:
                    buf.seek(0)
                    return pd.read_excel(buf)
                except Exception:
                    raise RuntimeError(f"读取 {suf} 失败：{e}")
        else:
            # 理论不会走到这里（csv 上面已处理）
            buf.seek(0)
            return pd.read_excel(buf)

    # 兜底：尝试当 CSV
    try:
        buf.seek(0)
        return pd.read_csv(buf, encoding="utf-8-sig")
    except Exception:
        raise RuntimeError(f"无法识别的文件类型：{filename or '(未命名)'}")


# =============================================================================
# 数据写库工具函数（供清洗任务使用）
# =============================================================================

def _to_int_or_none(v):
    """安全转换为整数或 None"""
    if pd.isna(v) or v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _write_records_to_db(
    df_clean: pd.DataFrame,
    source_name: str,
    batch_id: str,
    dry_run: bool = False,
    dedupe: bool = True,
    upsert: bool = False,
) -> Dict[str, Any]:
    """
    将清洗后的 DataFrame 写入数据库

    Returns:
        统计结果字典
    """
    # 规范列（防御）
    for col in ["part_number", "shop_name", "price_new", "price_grade_a", "price_grade_b", "recorded_at", "shop_address"]:
        if col not in df_clean.columns:
            df_clean[col] = None

    inserted = 0
    updated = 0
    dedup_skipped = 0
    unmatched = []
    preview_rows = []

    try:
        batch_uuid = uuid.UUID(str(batch_id))
    except Exception:
        batch_uuid = uuid.uuid4()

    for idx, row in df_clean.iterrows():
        pn = str(row.get("part_number") or "").strip()
        shop_name = str(row.get("shop_name") or "").strip()
        if not pn or not shop_name:
            unmatched.append({
                "row": int(idx),
                "reason": "缺少 part_number 或 shop_name",
                "data": row.to_dict()
            })
            continue

        iphone = Iphone.objects.filter(part_number=pn).first()
        if not iphone:
            unmatched.append({
                "row": int(idx),
                "reason": f"未找到 iPhone(PN={pn})",
                "data": row.to_dict()
            })
            continue

        shop_address = row.get("shop_address")
        shop_address = (shop_address or "").strip() if isinstance(shop_address, str) else ""
        shop = SecondHandShop.objects.filter(name=shop_name, address=shop_address).first()
        if not shop:
            shop = SecondHandShop.objects.create(name=shop_name, address=shop_address)

        price_new = _to_int_or_none(row.get("price_new"))
        price_a = _to_int_or_none(row.get("price_grade_a"))
        price_b = _to_int_or_none(row.get("price_grade_b"))

        rec_at = row.get("recorded_at")
        if not rec_at:
            recorded_at = timezone.now()
        else:
            try:
                recorded_at = pd.to_datetime(rec_at, utc=True, errors="coerce")
                if pd.isna(recorded_at):
                    recorded_at = timezone.now()
                else:
                    recorded_at = recorded_at.to_pydatetime()
            except Exception:
                recorded_at = timezone.now()

        if dry_run:
            inserted += 1
            if len(preview_rows) < 10:
                preview_rows.append({
                    "shop_name": shop_name,
                    "part_number": pn,
                    "price_new": price_new,
                    "price_grade_a": price_a,
                    "price_grade_b": price_b,
                    "recorded_at": str(recorded_at),
                    "batch_id": str(batch_uuid),
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
                    if price_new is not None and existed.price_new != price_new:
                        existed.price_new = price_new
                        changed = True
                    if price_a is not None and existed.price_grade_a != price_a:
                        existed.price_grade_a = price_a
                        changed = True
                    if price_b is not None and existed.price_grade_b != price_b:
                        existed.price_grade_b = price_b
                        changed = True
                    if changed:
                        existed.batch_id = batch_uuid
                        existed.save(update_fields=["price_new", "price_grade_a", "price_grade_b", "batch_id"])
                        updated += 1
                    else:
                        dedup_skipped += 1
                else:
                    dedup_skipped += 1
            else:
                rec = PurchasingShopPriceRecord.objects.create(
                    shop=shop,
                    iphone=iphone,
                    price_new=price_new or 0,
                    price_grade_a=price_a,
                    price_grade_b=price_b,
                    batch_id=batch_uuid,
                )
                PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=recorded_at)
                inserted += 1

                if len(preview_rows) < 10:
                    preview_rows.append({
                        "shop_name": shop_name,
                        "part_number": pn,
                        "price_new": price_new,
                        "price_grade_a": price_a,
                        "price_grade_b": price_b,
                        "recorded_at": str(recorded_at),
                        "batch_id": str(batch_uuid),
                    })

    return {
        "source": source_name,
        "rows_total": int(df_clean.shape[0]),
        "inserted": inserted,
        "updated": updated,
        "dedup_skipped": dedup_skipped,
        "unmatched": unmatched[:50],
        "errors": [],
        "preview": preview_rows,
        "batch_id": str(batch_uuid),
        "dry_run": dry_run,
        "dedupe": dedupe,
        "upsert": upsert,
    }


# =============================================================================
# 数据接收任务（webscraper 队列）
# =============================================================================

@shared_task(
    bind=True,
    autoretry_for=(OperationalError,),
    retry_backoff=5,
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=300,
    time_limit=360,
    name="AppleStockChecker.tasks.task_process_xlsx",
)
def task_process_xlsx(
    self,
    *,
    file_bytes: bytes,
    filename: str,
    source_name: str,
    dry_run: bool = False,
    dedupe: bool = True,
    upsert: bool = False,
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    数据接收任务：解析 xlsx/csv 文件 → 存 Redis → 触发清洗任务

    此任务只负责解析文件和分发，不执行数据清洗。
    """
    # 生成 batch_id
    if not batch_id:
        batch_id = str(uuid.uuid4())

    # 1. 解析文件
    try:
        df = _read_tabular(filename, file_bytes)
    except Exception as e:
        return {
            "accepted": False,
            "file": filename,
            "source": source_name,
            "error": f"读取表格失败: {e}",
            "hint": _ENGINE_HINT.get(_suffix(filename), (None, None))[1],
            "batch_id": batch_id,
        }

    # 2. 存入 Redis
    try:
        redis_key = store_dataframe(batch_id, source_name, df)
    except Exception as e:
        return {
            "accepted": False,
            "file": filename,
            "source": source_name,
            "error": f"存储到 Redis 失败: {e}",
            "batch_id": batch_id,
        }

    # 3. 触发清洗任务（动态路由到对应店铺队列）
    queue_name = get_shop_queue(source_name)

    # 延迟导入避免循环依赖
    cleaning_task = task_clean_shop_data.apply_async(
        kwargs={
            "source_name": source_name,
            "batch_id": batch_id,
            "redis_key": redis_key,
            "dry_run": dry_run,
            "dedupe": dedupe,
            "upsert": upsert,
        },
        queue=queue_name,
    )

    return {
        "accepted": True,
        "file": filename,
        "source": source_name,
        "batch_id": batch_id,
        "redis_key": redis_key,
        "cleaning_task_id": cleaning_task.id,
        "cleaning_queue": queue_name,
        "dry_run": dry_run,
        "dedupe": dedupe,
        "upsert": upsert,
    }


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=300,
    time_limit=360,
    name="AppleStockChecker.tasks.webscraper_tasks.task_process_webscraper_job",
)
def task_process_webscraper_job(
    self,
    job_id: str,
    source_name: str,
    *,
    dry_run: bool = False,
    create_shop: bool = True,
    dedupe: bool = True,
    upsert: bool = False,
    batch_id: str | None = None,
) -> dict:
    """
    数据接收任务：拉取 WebScraper 数据 → 存 Redis → 触发清洗任务

    此任务只负责拉取数据和分发，不执行数据清洗。
    """
    # 生成 batch_id
    if not batch_id:
        batch_id = str(uuid.uuid4())

    # 1. 拉取 WebScraper 数据
    try:
        content = fetch_webscraper_export_sync(job_id, format="csv")
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as e:
        return {
            "accepted": False,
            "job_id": job_id,
            "source": source_name,
            "error": f"拉取 WebScraper 数据失败: {e}",
            "batch_id": batch_id,
        }

    # 2. 存入 Redis
    try:
        redis_key = store_dataframe(batch_id, source_name, df)
    except Exception as e:
        return {
            "accepted": False,
            "job_id": job_id,
            "source": source_name,
            "error": f"存储到 Redis 失败: {e}",
            "batch_id": batch_id,
        }

    # 3. 触发清洗任务（动态路由到对应店铺队列）
    queue_name = get_shop_queue(source_name)

    cleaning_task = task_clean_shop_data.apply_async(
        kwargs={
            "source_name": source_name,
            "batch_id": batch_id,
            "redis_key": redis_key,
            "dry_run": dry_run,
            "dedupe": dedupe,
            "upsert": upsert,
        },
        queue=queue_name,
    )

    return {
        "accepted": True,
        "job_id": job_id,
        "source": source_name,
        "batch_id": batch_id,
        "redis_key": redis_key,
        "cleaning_task_id": cleaning_task.id,
        "cleaning_queue": queue_name,
        "dry_run": dry_run,
        "dedupe": dedupe,
        "upsert": upsert,
    }


@shared_task(
    bind=True,
    soft_time_limit=120,
    time_limit=180,
    name="AppleStockChecker.tasks.webscraper_tasks.task_ingest_json_shop1",
)
def task_ingest_json_shop1(self, records: list, opts: dict):
    """
    数据接收任务：接收 shop1 JSON 数据 → 存 Redis → 触发清洗任务

    此任务只负责接收数据和分发，不执行数据清洗。
    opts: {"dry_run": bool, "dedupe": bool, "upsert": bool, "batch_id": str, "source": "shop1"}
    """
    dry_run = bool(opts.get("dry_run"))
    dedupe = bool(opts.get("dedupe", True))
    upsert = bool(opts.get("upsert", False))
    batch_id = opts.get("batch_id") or str(uuid.uuid4())
    source = opts.get("source") or "shop1"

    # 1. JSON → DataFrame
    try:
        df = pd.DataFrame(records)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        return {
            "accepted": False,
            "source": source,
            "error": f"JSON 转换失败: {e}",
            "batch_id": batch_id,
        }

    # 2. 存入 Redis
    try:
        redis_key = store_dataframe(batch_id, source, df)
    except Exception as e:
        return {
            "accepted": False,
            "source": source,
            "error": f"存储到 Redis 失败: {e}",
            "batch_id": batch_id,
        }

    # 3. 触发 shop1 专用清洗任务
    cleaning_task = task_clean_shop1_json.apply_async(
        kwargs={
            "batch_id": batch_id,
            "redis_key": redis_key,
            "dry_run": dry_run,
            "dedupe": dedupe,
            "upsert": upsert,
        },
        queue="shop_shop1",
    )

    return {
        "accepted": True,
        "mode": "json",
        "source": source,
        "batch_id": batch_id,
        "redis_key": redis_key,
        "cleaning_task_id": cleaning_task.id,
        "cleaning_queue": "shop_shop1",
        "rows_received": len(records),
        "dry_run": dry_run,
        "dedupe": dedupe,
        "upsert": upsert,
    }


# =============================================================================
# 数据清洗任务（shop_* 队列）
# =============================================================================

@shared_task(
    bind=True,
    autoretry_for=(OperationalError,),
    retry_backoff=5,
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=9000,
    time_limit=9000,
    name="AppleStockChecker.tasks.task_clean_shop_data",
)
def task_clean_shop_data(
    self,
    *,
    source_name: str,
    batch_id: str,
    redis_key: str,
    dry_run: bool = False,
    dedupe: bool = True,
    upsert: bool = False,
) -> Dict[str, Any]:
    """
    通用数据清洗任务：从 Redis 读取 → 调用对应清洗器 → 写入数据库

    此任务通过 apply_async 动态路由到各店铺专用队列。
    """
    # 1. 从 Redis 读取原始数据
    df = retrieve_dataframe(redis_key)
    if df is None:
        return {
            "source": source_name,
            "batch_id": batch_id,
            "error": f"Redis 数据不存在或已过期: {redis_key}",
            "inserted": 0,
            "rows_total": 0,
        }

    # 2. 获取规范化的清洗器名称并执行清洗
    cleaner_name = get_cleaner_name(source_name)
    try:
        df_clean = run_cleaner(cleaner_name, df)
    except Exception as e:
        return {
            "source": source_name,
            "batch_id": batch_id,
            "error": f"清洗失败: {e}",
            "inserted": 0,
            "rows_total": int(df.shape[0]),
        }

    if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
        return {
            "source": source_name,
            "batch_id": batch_id,
            "error": "清洗后为空",
            "inserted": 0,
            "rows_total": int(df.shape[0]),
        }

    # 3. 写入数据库
    result = _write_records_to_db(
        df_clean=df_clean,
        source_name=source_name,
        batch_id=batch_id,
        dry_run=dry_run,
        dedupe=dedupe,
        upsert=upsert,
    )

    # 注：不删除 Redis 数据，依赖 TTL 自动过期（方便排查问题）
    return result


@shared_task(
    bind=True,
    autoretry_for=(OperationalError,),
    retry_backoff=5,
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=600,
    time_limit=900,
    name="AppleStockChecker.tasks.task_clean_shop1_json",
)
def task_clean_shop1_json(
    self,
    *,
    batch_id: str,
    redis_key: str,
    dry_run: bool = False,
    dedupe: bool = True,
    upsert: bool = False,
) -> Dict[str, Any]:
    """
    shop1 JSON 数据专用清洗任务

    与通用清洗任务类似，但固定使用 shop1 清洗器。
    """
    source_name = "shop1"

    # 1. 从 Redis 读取原始数据
    df = retrieve_dataframe(redis_key)
    if df is None:
        return {
            "mode": "json",
            "source": source_name,
            "batch_id": batch_id,
            "error": f"Redis 数据不存在或已过期: {redis_key}",
            "inserted": 0,
            "rows_total": 0,
        }

    # 2. 执行清洗
    try:
        df_clean = run_cleaner("shop1", df)
    except Exception as e:
        return {
            "mode": "json",
            "source": source_name,
            "batch_id": batch_id,
            "error": f"清洗失败: {e}",
            "inserted": 0,
            "rows_total": int(df.shape[0]),
        }

    if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
        return {
            "mode": "json",
            "source": source_name,
            "batch_id": batch_id,
            "error": "清洗后为空",
            "inserted": 0,
            "rows_total": int(df.shape[0]),
        }

    # 3. 写入数据库
    result = _write_records_to_db(
        df_clean=df_clean,
        source_name=source_name,
        batch_id=batch_id,
        dry_run=dry_run,
        dedupe=dedupe,
        upsert=upsert,
    )

    result["mode"] = "json"
    return result
