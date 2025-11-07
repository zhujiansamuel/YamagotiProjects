# -*- coding: utf-8 -*-
from __future__ import annotations
import io, re
from celery import shared_task
from django.db import transaction
from django.utils import timezone
import pandas as pd
from typing import Optional, Dict, Any
from AppleStockChecker.utils.external_ingest.registry import run_cleaner  # 已有注册器
from AppleStockChecker.models import Iphone, SecondHandShop, PurchasingShopPriceRecord
from AppleStockChecker.utils.external_ingest.webscraper import fetch_webscraper_export_sync
from AppleStockChecker.services.external_ingest_service import ingest_external_dataframe
from django.db.utils import OperationalError

_ENGINE_HINT = {
    "xlsx": ("openpyxl", "pip install openpyxl"),
    "xlsm": ("openpyxl", "pip install openpyxl"),
    "xls":  ("xlrd",     "pip install 'xlrd<2.0'"),
    "ods":  ("odf",      "pip install odfpy"),
    "xlsb": ("pyxlsb",   "pip install pyxlsb"),
    "csv":  (None,       None),
}

def _suffix(filename: str) -> str:
    m = re.search(r"\.([A-Za-z0-9]+)$", (filename or "").strip())
    return (m.group(1).lower() if m else "")

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
                # 再给一次“自动引擎”机会（pandas 自探测）
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


@shared_task(
    bind=True,
    autoretry_for=(OperationalError,),
    retry_backoff=5,
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
    soft_time_limit=9000, time_limit=9000,
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
    try:
        df = _read_tabular(filename, file_bytes)
    except Exception as e:
        # 不抛出，直接把错误返回给调用方（避免无限重试）
        return {
            "file": filename,
            "source": source_name,
            "error": f"读取表格失败: {e}",
            "hint": _ENGINE_HINT.get(_suffix(filename), (None, None))[1],
            "dry_run": dry_run,
            "dedupe": dedupe,
            "upsert": upsert,
            "batch_id": batch_id,
        }

    result = ingest_external_dataframe(
        source_name=source_name,
        df=df,
        dry_run=dry_run,
        pn_only=True,
        create_shop=True,
        dedupe=dedupe,
        upsert=upsert,
        batch_id=batch_id,
    )
    result.update({
        "file": filename,
        "source": source_name,
        "dry_run": dry_run,
        "dedupe": dedupe,
        "upsert": upsert,
        "batch_id": batch_id,
    })
    return result





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




@shared_task(bind=True, soft_time_limit=600, time_limit=900)
def task_ingest_json_shop1(self, records: list, opts: dict):
    """
    Celery 任务：records(JSON 数组) -> DataFrame -> 清洗器 shop1 -> （dry_run 或落库）
    opts: {"dry_run":bool, "dedupe":bool, "upsert":bool, "batch_id":str, "source":"shop1"}
    返回与 ingest_external_dataframe 类似的统计结构（插入数/未匹配/预览等）
    """
    dry_run = bool(opts.get("dry_run"))
    dedupe  = bool(opts.get("dedupe", True))
    upsert  = bool(opts.get("upsert", False))
    batch_id = opts.get("batch_id") or ""
    source   = opts.get("source") or "shop1"

    # 1) JSON -> DataFrame
    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]

    # 2) 清洗：调用注册的 cleaner("shop1")
    try:
        df_clean = run_cleaner(source, df)
        # print("df_clean--------------------->",df_clean)
    except Exception as e:
        return {"inserted": 0, "errors": [{"source": source, "error": f"清洗失败: {e}"}], "preview": [], "rows_total": 0}

    if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
        return {"inserted": 0, "errors": [{"source": source, "error": "清洗后为空"}], "preview": [], "rows_total": 0}

    # 3) 规范列（防御）
    for col in ["part_number","shop_name","price_new","price_grade_a","price_grade_b","recorded_at","shop_address"]:
        if col not in df_clean.columns:
            df_clean[col] = None

    # 4) 逐行处理（dry_run=预览；否则写库；支持 dedupe/upsert）
    inserted = 0
    updated  = 0
    skipped  = 0
    unmatched = []
    preview_rows = []

    def to_int_or_none(v):
        if pd.isna(v) or v is None: return None
        try: return int(v)
        except Exception: return None

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

        shop_address = row.get("shop_address")
        shop_address = (shop_address or "").strip() if isinstance(shop_address, str) else ""
        shop = SecondHandShop.objects.filter(name=shop_name, address=shop_address).first()
        if not shop:
            shop = SecondHandShop.objects.create(name=shop_name, address=shop_address)

        price_new = to_int_or_none(row.get("price_new"))
        price_a   = to_int_or_none(row.get("price_grade_a"))
        price_b   = to_int_or_none(row.get("price_grade_b"))

        rec_at = row.get("recorded_at")
        if not rec_at:
            recorded_at = timezone.now()
        else:
            # 让 cleaner 输出 ISO8601 / aware datetime 更稳；此处兜底 parse
            try:
                # 若你项目前已有 parse_datetime 工具可用就换成它
                recorded_at = pd.to_datetime(rec_at, utc=True, errors="coerce")
                if pd.isna(recorded_at):
                    recorded_at = timezone.now()
                else:
                    recorded_at = recorded_at.to_pydatetime()
            except Exception:
                recorded_at = timezone.now()

        if dry_run:
            inserted += 1   # 预估写入条数
            if len(preview_rows) < 10:
                preview_rows.append({
                    "shop_name": shop_name, "part_number": pn,
                    "price_new": price_new, "price_grade_a": price_a, "price_grade_b": price_b,
                    "recorded_at": str(recorded_at)
                })
            continue

        # 实际落库
        with transaction.atomic():
            if dedupe:
                existed = PurchasingShopPriceRecord.objects.filter(
                    shop=shop, iphone=iphone, recorded_at=recorded_at
                ).first()
            else:
                existed = None

            if existed:
                changed = False
                if price_new is not None and existed.price_new != price_new:
                    existed.price_new = price_new; changed = True
                if price_a is not None and existed.price_grade_a != price_a:
                    existed.price_grade_a = price_a; changed = True
                if price_b is not None and existed.price_grade_b != price_b:
                    existed.price_grade_b = price_b; changed = True
                if changed:
                    existed.save(update_fields=["price_new", "price_grade_a", "price_grade_b"])
                    updated += 1
                else:
                    skipped += 1
            else:
                rec = PurchasingShopPriceRecord.objects.create(
                    shop=shop, iphone=iphone,
                    price_new=price_new or 0,
                    price_grade_a=price_a,
                    price_grade_b=price_b,
                )
                PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=recorded_at)
                inserted += 1

            # 可选：批次标识（若你模型加了 batch_id 字段）
            if hasattr(PurchasingShopPriceRecord, "batch_id"):
                PurchasingShopPriceRecord.objects.filter(
                    shop=shop, iphone=iphone, recorded_at=recorded_at
                ).update(batch_id=batch_id)

            if len(preview_rows) < 10:
                preview_rows.append({
                    "shop_name": shop_name, "part_number": pn,
                    "price_new": price_new, "price_grade_a": price_a, "price_grade_b": price_b,
                    "recorded_at": str(recorded_at)
                })

    return {
        "mode": "json",
        "source": source,
        "rows_total": int(df_clean.shape[0]),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "unmatched": unmatched[:50],
        "errors": [],
        "preview": preview_rows,
        "batch_id": batch_id,
        "dry_run": dry_run,
        "dedupe": dedupe,
        "upsert": upsert,
    }