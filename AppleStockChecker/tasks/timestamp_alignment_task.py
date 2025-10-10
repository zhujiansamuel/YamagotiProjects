
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from typing import List, Dict, Any, Optional
from ..serializers import PurchasingShopTimeAnalysisSerializer
from ..collectors import collect_items_for_psta  # ← 新增：查询模式的收集器
from celery import shared_task
from django.utils.dateparse import parse_datetime
from ..services.time_analysis_services import upsert_purchasing_time_analysis
from ..serializers import PSTACompactSerializer
from ..ws_notify import notify_progress, notify_batch_items
import time
import logging
from typing import List, Dict, Any, Optional
from celery import shared_task
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from AppleStockChecker.utils.timebox import nearest_past_minute_iso  # ← 新增

from AppleStockChecker.services.time_analysis_services import upsert_purchasing_time_analysis
from AppleStockChecker.serializers import PSTACompactSerializer
from AppleStockChecker.ws_notify import (
    notify_progress_all,
    notify_batch_items_all,
    notify_batch_done_all,
)

logger = logging.getLogger(__name__)

def _normalize_ts(ts_iso: str) -> str:
    """
    把任意 ISO8601 字符串 -> 规范的 ISO8601（含时区，秒精度）。
    """
    dt = parse_datetime(ts_iso)
    if dt is None:
        raise ValueError(f"timestamp_iso 无法解析: {ts_iso!r}")
    if timezone.is_naive(dt):
        # 默认按项目时区补齐（一般是 Asia/Tokyo）
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt.isoformat(timespec="seconds")

def _group_send(job_id: str, payload: dict):
    # 可选：把结果推到 WebSocket（Channels）
    try:
        ch = get_channel_layer()
        async_to_sync(ch.group_send)(
            f"task_{job_id}",
            {"type": "progress_message", "data": payload},
        )
    except Exception:
        pass




def _iter_chunks(items: List[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i:i+size]


@shared_task(bind=True, name="AppleStockChecker.tasks.timestamp_alignment_task.batch_generate_psta")
def batch_generate_psta(
        self,
        *,
        # —— 通用控制项 —— #
        job_id: Optional[str] = None,
        chunk_size: int = 100,
        max_items: Optional[int] = None,

        # —— 两种模式的入参 —— #
        # A) payload 模式：直接给 items（每条就是 upsert 所需 payload）
        items: Optional[List[Dict[str, Any]]] = None,

        # B) query 模式：传查询条件，任务启动后自动查询出 items
        mode: str = "query",  # "payload" | "query"
        query_window_minutes: int = 15,  # 最近N分钟
        shop_ids: Optional[List[int]] = None,  # 限定某些店
        iphone_ids: Optional[List[int]] = None,  # 限定某些机型
) -> dict:
    """
        一次 Celery 任务处理“一批” PSTA 记录。
        - payload 模式：items 已经在 kwargs 里给出
        - query 模式（默认）：根据 kwargs 的查询条件，任务启动时先生成 items
        """
    job = job_id or self.request.id

    # 1) 先得到 items
    if mode == "payload":
        assert items is not None and isinstance(items, list), "payload 模式必须提供 items:list"
        source_items = items
    else:
        source_items = collect_items_for_psta(
            window_minutes=query_window_minutes,
            shop_ids=shop_ids,
            iphone_ids=iphone_ids,
            max_items=max_items,
        )

    total = len(source_items)
    processed = ok = failed = 0
    summary_errors = []

    # 仅用于稳定主题推送的维度（如果整批不唯一，就用0；实际前端更多订阅“稳定主题”）
    stable_shop = shop_ids[0] if shop_ids else 0
    stable_iphone = iphone_ids[0] if iphone_ids else 0

    notify_progress(job_id=job, shop_id=stable_shop, iphone_id=stable_iphone,
                    data={"status": "running", "step": "start", "progress": 0, "total": total})

    # 2) 批处理（每 chunk 汇报一次进度）
    for chunk in _iter_chunks(source_items[: (max_items or total)], chunk_size):
        for item in chunk:
            try:
                inst = upsert_purchasing_time_analysis(item)  # ← 每条“一次事务”
                # 可选：只回传关键字段，避免结果过大
                _ = PurchasingShopTimeAnalysisSerializer(inst).data
                ok += 1
            except Exception as e:
                failed += 1
                summary_errors.append({"msg": str(e)})
            finally:
                processed += 1

        progress = int(processed * 100 / max(1, total))
        notify_progress(job_id=job, shop_id=stable_shop, iphone_id=stable_iphone,
                        data={"status": "running", "step": "chunk", "progress": progress,
                              "processed": processed, "ok": ok, "failed": failed})

    summary = {"total": total, "ok": ok, "failed": failed}
    if summary_errors:
        summary["errors"] = summary_errors[:5]  # 只截前5条，防炸 payload

    notify_progress(job_id=job, shop_id=stable_shop, iphone_id=stable_iphone,
                    data={"status": "done", "progress": 100, "summary": summary})

    return summary


@shared_task(bind=True, name="AppleStockChecker.tasks.timestamp_alignment_task.batch_generate_psta_same_ts")
def batch_generate_psta_same_ts(
    self,
    *,
    job_id: Optional[str] = None,               # 仍可用于日志/追踪，但不用于分组
    items: List[Dict[str, Any]],
    timestamp_iso: Optional[str] = None,        # ← 可省略：自动取最近过去整分钟
    chunk_size: int = 200
) -> dict:
    # 1) 统一时间戳：最近过去整分钟（Asia/Tokyo 等项目时区）
    ts_iso = timestamp_iso or nearest_past_minute_iso()

    total = len(items)
    ok = failed = processed = 0

    # 2) 开始：发一次 ALL 频道状态
    notify_progress_all(data={"status": "running", "step": "start", "progress": 0, "total": total, "timestamp": ts_iso})

    # 3) 边处理边分片推送（ALL 频道）
    chunk_index = 0
    total_chunks = (total + chunk_size - 1) // chunk_size if total else 0
    errors = []  # ← 新增：收集错误
    for chunk in _iter_chunks(items, chunk_size):
        chunk_index += 1
        compact_rows = []
        for item in chunk:
            # 强制统一时间戳
            item["Timestamp_Time"] = ts_iso
            try:
                inst = upsert_purchasing_time_analysis(item)  # 每条条目一次事务/行锁，幂等
                compact_rows.append(PSTACompactSerializer(inst).data)
                ok += 1
            except Exception as e:
                failed += 1
                err = {
                    "exc": e.__class__.__name__,
                    "msg": str(e),
                    "item": {k: item.get(k) for k in ["shop_id", "iphone_id", "Timestamp_Time", "New_Product_Price"]},
                }
                errors.append(err)
                logger.warning("PSTA upsert failed", extra={"job": job_id, "err": str(e), "item": item})
            finally:
                processed += 1

        # 本片产生的实例列表 -> ALL 频道
        if compact_rows:
            notify_batch_items_all(
                timestamp_iso=ts_iso,
                items=compact_rows,
                index=chunk_index,
                total=total_chunks or 1,
            )

        # 进度 -> ALL 频道
        progress = int(processed * 100 / max(1, total))
        notify_progress_all(data={
            "status":"running", "step":"chunk", "progress":progress,
            "processed": processed, "ok": ok, "failed": failed,
            "timestamp": ts_iso
        })
    summary = {
        "timestamp": ts_iso,
        "total": total,
        "ok": ok,
        "failed": failed,
        "errors": errors[:5],
    }

    # 4) 结束：批量完成标记 + 摘要
    notify_batch_done_all(timestamp_iso=ts_iso, total_chunks=total_chunks, total_items=total)
    notify_progress_all(data={"status":"done","progress":100,"summary":{"timestamp": ts_iso, "total": total, "ok": ok, "failed": failed}})

    return summary



@shared_task(bind=True, name="AppleStockChecker.tasks.timestamp_alignment_task.generate_purchasing_time_analysis")
def generate_purchasing_time_analysis(self, *, payload: dict) -> dict:
    job_id = payload.get("Job_ID") or self.request.id
    shop_id = payload["shop_id"]
    iphone_id = payload["iphone_id"]

    notify_progress(job_id=job_id, shop_id=shop_id, iphone_id=iphone_id,
                    data={"status": "running", "step": "upsert", "progress": 5})

    inst = upsert_purchasing_time_analysis(payload)
    data = PurchasingShopTimeAnalysisSerializer(inst).data

    notify_progress(job_id=job_id, shop_id=shop_id, iphone_id=iphone_id,
                    data={"status": "done", "result": data, "progress": 100})
    return data
