from __future__ import annotations
from ..serializers import PurchasingShopTimeAnalysisSerializer
from collections import Counter
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import IntegrityError
from django.utils.dateparse import parse_datetime
from AppleStockChecker.ws_notify import notify_progress_all
from AppleStockChecker.models import SecondHandShop, Iphone
from ..ws_notify import notify_progress, notify_batch_items
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta
from celery import shared_task, group, chord
from celery import shared_task, group
from django.db import transaction, IntegrityError
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from typing import Optional, List, Dict, Any
from AppleStockChecker.collectors import collect_items_for_psta
from AppleStockChecker.utils.timebox import nearest_past_minute_iso
from AppleStockChecker.models import PurchasingShopTimeAnalysis, SecondHandShop, Iphone
from AppleStockChecker.services.time_analysis_services import upsert_purchasing_time_analysis
from AppleStockChecker.serializers import PSTACompactSerializer
from AppleStockChecker.ws_notify import (
    notify_progress_all,
    notify_batch_items_all,
    notify_batch_done_all,
)

logger = logging.getLogger(__name__)



# ---------- 工具：ISO ↔ aware datetime ----------
def _to_aware(dt_or_iso):
    """接受 datetime 或 ISO 字符串，返回 aware datetime（项目时区）。"""
    if isinstance(dt_or_iso, str):
        dt = parse_datetime(dt_or_iso)
        if dt is None:
            raise ValueError(f"Invalid datetime: {dt_or_iso!r}")
    else:
        dt = dt_or_iso
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


def _tz_offset_str(dt: timezone.datetime) -> str:
    """把 datetime 的 tzinfo 转成 '+09:00' 这种偏移字符串。"""
    offset = dt.utcoffset() or timedelta(0)
    sign = "+" if offset.total_seconds() >= 0 else "-"
    total = abs(int(offset.total_seconds()))
    hh, mm = divmod(total // 60, 60)
    return f"{sign}{hh:02d}:{mm:02d}"

# AppleStockChecker/tasks.py
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from celery import shared_task, chord
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import transaction, IntegrityError

# === 可调参数（根据你们前后端链路容量调节） ===
MAX_BUCKET_ERROR_SAMPLES = 50       # 单桶保留的 error 明细条数上限
MAX_BUCKET_CHART_POINTS = 1000      # 单桶打包给回调聚合用的 chart point 上限
MAX_PUSH_POINTS = 20000             # 本次广播给前端的 point 总上限（超过则裁剪到最近 N 条）

# -----------------------------------------------
# 子任务：处理“分钟桶”并返回桶级摘要 + 图表增量
# -----------------------------------------------
@shared_task(name="AppleStockChecker.tasks.psta_process_minute_bucket")
def psta_process_minute_bucket(
    *,
    ts_iso: str,
    rows: List[Dict[str, Any]],
    job_id: str
) -> Dict[str, Any]:
    """
    为某分钟桶生成/更新 PSTA。
    返回：
      - 错误详情与直方图（便于定位 why ok=0 failed>0）
      - chart_points：给 finalize 汇总后推送到前端的图表增量（去重由前端做）
    """
    from django.utils import timezone  # 若你有用到
    ok = 0
    failed = 0
    errors: List[Dict[str, Any]] = []
    err_counter = Counter()

    ts_dt = _to_aware(ts_iso)
    ts_tz = _tz_offset_str(ts_dt)
    orig_tz = "+09:00"

    chart_points: List[Dict[str, Any]] = []

    for r in rows:
        try:
            # —— 轻量校验/预取 —— #
            shop_id = r.get("shop_id")
            iphone_id = r.get("iphone_id")
            if not shop_id or not iphone_id:
                raise ValueError("missing shop_id/iphone_id")

            # 外键存在性（报更友好的 DoesNotExist）
            SecondHandShop.objects.only("id").get(pk=shop_id)
            Iphone.objects.only("id").get(pk=iphone_id)

            rec_dt = _to_aware(r.get("recorded_at"))
            new_price = r.get("price_new") or r.get("New_Product_Price")
            if new_price is None:
                raise ValueError("missing New_Product_Price")

            align_diff = int((rec_dt - ts_dt).total_seconds())

            with transaction.atomic():
                inst = PurchasingShopTimeAnalysis.objects.select_for_update().filter(
                    shop_id=shop_id, iphone_id=iphone_id, Timestamp_Time=ts_dt
                ).first()

                if inst:
                    inst.Job_ID = job_id
                    inst.Original_Record_Time_Zone = orig_tz
                    inst.Timestamp_Time_Zone = ts_tz
                    inst.Record_Time = rec_dt
                    inst.Alignment_Time_Difference = align_diff
                    inst.New_Product_Price = int(new_price)
                    inst.Update_Count = (inst.Update_Count or 0) + 1
                    inst.save()
                else:
                    inst = PurchasingShopTimeAnalysis.objects.create(
                        Batch_ID=None,
                        Job_ID=job_id,
                        Original_Record_Time_Zone=orig_tz,
                        Timestamp_Time_Zone=ts_tz,
                        Record_Time=rec_dt,
                        Timestamp_Time=ts_dt,
                        Alignment_Time_Difference=align_diff,
                        Update_Count=0,
                        shop_id=shop_id,
                        iphone_id=iphone_id,
                        New_Product_Price=int(new_price),
                    )

            ok += 1

            # === 收集图表增量：每条成功写库就记录一个点 ===
            if len(chart_points) < MAX_BUCKET_CHART_POINTS:
                chart_points.append({
                    "id": inst.pk,                      # 用于前端幂等去重（可选）
                    "t": ts_iso,                        # 分钟桶时间（x 轴）
                    "iphone_id": iphone_id,
                    "shop_id": shop_id,
                    "price": int(new_price),
                    "recorded_at": rec_dt.isoformat(),  # 原始记录时间（辅助展示）
                })

        except (ObjectDoesNotExist, ValidationError, IntegrityError, TypeError, ValueError) as e:
            failed += 1
            err_counter[e.__class__.__name__] += 1
            if len(errors) < MAX_BUCKET_ERROR_SAMPLES:
                errors.append({
                    "exc": e.__class__.__name__,
                    "msg": str(e),
                    "item": {
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                        "New_Product_Price": r.get("price_new") or r.get("New_Product_Price"),
                    }
                })
        except Exception as e:
            failed += 1
            err_counter[e.__class__.__name__] += 1
            if len(errors) < MAX_BUCKET_ERROR_SAMPLES:
                errors.append({
                    "exc": e.__class__.__name__,
                    "msg": str(e),
                    "item": {
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                    }
                })

    # 有错误就推一条“桶级摘要”调试消息（ALL 频道）
    if failed:
        try:
            notify_progress_all(data={
                "type": "bucket_errors",
                "ts_iso": ts_iso,
                "total": ok + failed,
                "ok": ok,
                "failed": failed,
                "error_hist": dict(err_counter),
                "sample": errors[:5],  # 只展示前 5 条样例
            })
        except Exception:
            pass

    return {
        "ts_iso": ts_iso,
        "ok": ok,
        "failed": failed,
        "total": ok + failed,
        "error_hist": dict(err_counter),
        "errors": errors[:MAX_BUCKET_ERROR_SAMPLES],
        "chart_points": chart_points,  # ✅ 传给回调聚合后做图表更新
    }


# -----------------------------------------------
# 回调：聚合所有分钟桶，广播最终“done + 图表增量”
# -----------------------------------------------
@shared_task(name="AppleStockChecker.tasks.psta_finalize_buckets")
def psta_finalize_buckets(
    results: List[Dict[str, Any]],
    job_id: str,
    ts_iso: str
) -> Dict[str, Any]:
    """汇总所有分钟桶的错误直方图 + 打包本次入库的图表增量，并推送最终 done。"""
    total_buckets = len(results or [])
    total_ok = sum(int(r.get("ok", 0)) for r in results or [])
    total_failed = sum(int(r.get("failed", 0)) for r in results or [])

    # 汇总错误直方图
    agg_err = Counter()
    for r in results or []:
        for k, v in (r.get("error_hist") or {}).items():
            agg_err[k] += v

    # === 聚合图表增量：分组为 (iphone_id, shop_id) 的独立序列 ===
    series_map = defaultdict(list)  # key: (iphone_id, shop_id) -> List[point]
    total_points = 0

    for r in results or []:
        for p in (r.get("chart_points") or []):
            key = (p.get("iphone_id"), p.get("shop_id"))
            # 仅保留必要字段，减小 payload
            series_map[key].append({
                "id": p.get("id"),
                "t": p.get("t"),
                "price": p.get("price"),
                "recorded_at": p.get("recorded_at"),
            })
            total_points += 1

    # 按时间排序，并做全局裁剪（保留最近 MAX_PUSH_POINTS 条）
    # 思路：先合并后排序，再裁剪；时间为 ISO 字符串，可直接字典序
    # 同时把结果转成前端友好的 list 结构
    # 统计全量点数，必要时截断
    clipped = False
    if total_points > MAX_PUSH_POINTS:
        clipped = True
        # 计算需要保留的最近点的全局阈值：先把所有点扁平化排序，再筛回各组
        flat = []
        for (iphone_id, shop_id), pts in series_map.items():
            for item in pts:
                flat.append((item["t"], iphone_id, shop_id, item))
        flat.sort(key=lambda x: x[0])  # 按时间升序
        flat = flat[-MAX_PUSH_POINTS:]  # 仅保留最近 N 条

        # 重建 series_map（仅保留被截断后的点）
        series_map = defaultdict(list)
        for _, iphone_id, shop_id, item in flat:
            series_map[(iphone_id, shop_id)].append(item)

    # 每个序列内再做一次排序，以免截断后顺序被打乱
    series_delta = []
    for (iphone_id, shop_id), pts in series_map.items():
        pts.sort(key=lambda x: x["t"])
        series_delta.append({
            "iphone_id": iphone_id,
            "shop_id": shop_id,
            "points": pts,  # [{id, t, price, recorded_at}, ...]
        })

    # 构建 summary
    summary = {
        "timestamp": ts_iso,
        "job_id": job_id,
        "total_buckets": total_buckets,
        "ok": total_ok,
        "failed": total_failed,
        "error_hist": dict(agg_err),      # 全局错误直方图
        "by_bucket": [
            {k: r.get(k) for k in ("ts_iso", "ok", "failed", "total", "error_hist")}
            for r in (results or [])
        ][:100],  # 明细最多 100 桶
    }

    payload = {
        "status": "done",
        "step": "finalize",
        "progress": 100,
        "summary": summary,
        # ✅ 前端图表直接应用的增量
        "chart_delta": {
            "job_id": job_id,
            "timestamp": ts_iso,
            "series_delta": series_delta,
            "meta": {
                "total_points": min(total_points, MAX_PUSH_POINTS),
                "clipped": clipped,
            }
        }
    }

    # 广播
    try:
        notify_progress_all(data=payload)
    except Exception:
        pass

    # 任务返回值也带上（便于后端排查）
    return payload


# -----------------------------------------------
# 父任务：chord 并行 + 回调（保持你已有写法）
# -----------------------------------------------
@shared_task(bind=True, name="AppleStockChecker.tasks.batch_generate_psta_same_ts")
def batch_generate_psta_same_ts(
    self,
    *,
    job_id: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,   # 兼容
    timestamp_iso: Optional[str] = None,
    chunk_size: int = 200,                          # 兼容
    query_window_minutes: int = 15,
    shop_ids: Optional[List[int]] = None,
    iphone_ids: Optional[List[int]] = None,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    task_job_id = job_id or self.request.id
    ts_iso = timestamp_iso or nearest_past_minute_iso()

    pack = collect_items_for_psta(
        window_minutes=query_window_minutes,
        timestamp_iso=ts_iso,
        shop_ids=shop_ids,
        iphone_ids=iphone_ids,
        max_items=max_items,
    )[0]
    rows = pack["rows"]
    bucket_minute_key = pack.get("bucket_minute_key") or {}

    # 按“分钟桶”构建子任务列表
    subtasks: List = []
    for minute_iso, key_map in bucket_minute_key.items():
        minute_rows: List[Dict[str, Any]] = []
        for _, idx_list in key_map.items():
            for i in idx_list:
                if 0 <= i < len(rows):
                    r = rows[i]
                    minute_rows.append({
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                        "price_new": r.get("price_new", r.get("New_Product_Price")),
                    })
        if minute_rows:
            # 一分钟桶一个子任务
            subtasks.append(
                psta_process_minute_bucket.s(ts_iso=minute_iso, rows=minute_rows, job_id=task_job_id)
            )

    total_buckets = len(subtasks)

    # 开始广播（可选）
    try:
        notify_progress_all(data={
            "status": "running",
            "step": "dispatch_buckets",
            "progress": 0,
            "buckets": total_buckets,
            "timestamp": ts_iso
        })
    except Exception:
        pass

    if not subtasks:
        # 无桶可处理，直接返回空摘要
        empty = {"timestamp": ts_iso, "total_buckets": 0, "ok": 0, "failed": 0, "by_bucket": []}
        try:
            notify_progress_all(data={
                "status": "done",
                "progress": 100,
                "summary": empty,
                "chart_delta": {"job_id": task_job_id, "timestamp": ts_iso, "series_delta": [], "meta": {"total_points": 0, "clipped": False}}
            })
        except Exception:
            pass
        return empty

    # ★ 使用 chord 并行执行 + 汇总回调，不要 .get()
    callback = psta_finalize_buckets.s(task_job_id, ts_iso)
    chord_result = chord(subtasks)(callback)

    # 返回“编排任务”信息；前端通过广播拿最终 summary+chart_delta
    return {
        "timestamp": ts_iso,
        "total_buckets": total_buckets,
        "job_id": task_job_id,
        "chord_id": chord_result.id,
    }


#-----------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------------
#---------------------------------------------------------------
#-----------------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------



# @shared_task(bind=True, name="AppleStockChecker.tasks.timestamp_alignment_task.batch_generate_psta")
# def batch_generate_psta(
#         self,
#         *,
#         job_id: Optional[str] = None,
#         chunk_size: int = 100,
#         max_items: Optional[int] = None,
#         items: Optional[List[Dict[str, Any]]] = None,
#         mode: str = "query",  # "payload" | "query"
#         query_window_minutes: int = 15,  # 最近N分钟
#         shop_ids: Optional[List[int]] = None,  # 限定某些店
#         iphone_ids: Optional[List[int]] = None,  # 限定某些机型
# ) -> dict:
#     """
#         一次 Celery 任务处理“一批” PSTA 记录。
#         - payload 模式：items 已经在 kwargs 里给出
#         - query 模式（默认）：根据 kwargs 的查询条件，任务启动时先生成 items
#         """
#     job = job_id or self.request.id
#
#     # 1) 先得到 items
#     if mode == "payload":
#         assert items is not None and isinstance(items, list), "payload 模式必须提供 items:list"
#         source_items = items
#     else:
#         source_items = collect_items_for_psta(
#             window_minutes=query_window_minutes,
#             shop_ids=shop_ids,
#             iphone_ids=iphone_ids,
#             max_items=max_items,
#         )
#
#     total = len(source_items)
#     processed = ok = failed = 0
#     summary_errors = []
#
#     # 仅用于稳定主题推送的维度（如果整批不唯一，就用0；实际前端更多订阅“稳定主题”）
#     stable_shop = shop_ids[0] if shop_ids else 0
#     stable_iphone = iphone_ids[0] if iphone_ids else 0
#
#     notify_progress(job_id=job, shop_id=stable_shop, iphone_id=stable_iphone,
#                     data={"status": "running", "step": "start", "progress": 0, "total": total})
#
#     # 2) 批处理（每 chunk 汇报一次进度）
#     for chunk in _iter_chunks(source_items[: (max_items or total)], chunk_size):
#         for item in chunk:
#             try:
#                 inst = upsert_purchasing_time_analysis(item)  # ← 每条“一次事务”
#                 # 可选：只回传关键字段，避免结果过大
#                 _ = PurchasingShopTimeAnalysisSerializer(inst).data
#                 ok += 1
#             except Exception as e:
#                 failed += 1
#                 summary_errors.append({"msg": str(e)})
#             finally:
#                 processed += 1
#
#         progress = int(processed * 100 / max(1, total))
#         notify_progress(job_id=job, shop_id=stable_shop, iphone_id=stable_iphone,
#                         data={"status": "running", "step": "chunk", "progress": progress,
#                               "processed": processed, "ok": ok, "failed": failed})
#
#     summary = {"total": total, "ok": ok, "failed": failed}
#     if summary_errors:
#         summary["errors"] = summary_errors[:5]  # 只截前5条，防炸 payload
#
#     notify_progress(job_id=job, shop_id=stable_shop, iphone_id=stable_iphone,
#                     data={"status": "done", "progress": 100, "summary": summary})
#
#     return summary


# @shared_task(bind=True, name="AppleStockChecker.tasks.timestamp_alignment_task.generate_purchasing_time_analysis")
# def generate_purchasing_time_analysis(self, *, payload: dict) -> dict:
#     job_id = payload.get("Job_ID") or self.request.id
#     shop_id = payload["shop_id"]
#     iphone_id = payload["iphone_id"]
#
#     notify_progress(job_id=job_id, shop_id=shop_id, iphone_id=iphone_id,
#                     data={"status": "running", "step": "upsert", "progress": 5})
#
#     inst = upsert_purchasing_time_analysis(payload)
#     data = PurchasingShopTimeAnalysisSerializer(inst).data
#
#     notify_progress(job_id=job_id, shop_id=shop_id, iphone_id=iphone_id,
#                     data={"status": "done", "result": data, "progress": 100})
#     return data
