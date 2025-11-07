import uuid
from uuid import uuid4
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticatedOrReadOnly, AllowAny
from rest_framework.response import Response
from rest_framework import status
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from celery import shared_task, chord
from .tasks.timestamp_alignment_task import psta_finalize_buckets

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

@api_view(["POST"])
@permission_classes([AllowAny])  # 本地调试也可 AllowAny
def dispatch_psta_batch_same_ts(request):
    """
    调用方法的一个例子
    （作用是发起为历史数据做分桶计算，用于增加或者修改统计指标时重算历史记录）
    ---------------------------------------------
    from datetime import datetime, timedelta, timezone
    import subprocess, json

    JST = timezone(timedelta(hours=9))

    start = datetime(2025,10,23, 7, 0, 0, tzinfo=JST)
    end   = datetime(2025,10,23, 20, 25, 0, tzinfo=JST)

    minutes = int((end - start).total_seconds() // 60)  # 59040
    timestamps = [(start + timedelta(minutes=i)).isoformat(timespec="seconds")
                  for i in range(minutes - 1, -1, -1)]

    # 注意必须使用域名
    url = "http://yamaguti.ngrok.io/AppleStockChecker/purchasing-time-analyses/dispatch_ts/"
    # url = "https://yamaguti.ngrok.io/AppleStockChecker/purchasing-time-analyses/dispatch_ts/"

    for i, ts in enumerate(timestamps, 1):
        payload = json.dumps({"timestamp_iso": ts})
        out = subprocess.check_output(
            ["curl", "-sS", "-X", "POST", url,
             "-H", "Content-Type: application/json",
             "-d", payload],
            stderr=subprocess.STDOUT
        ).decode("utf-8", "replace")
        print(f"[{i}/{len(timestamps)}] {ts}\n{out}\n")
        if i%100 == 0:
            time.sleep(60)
        time.sleep(5)

    ---------------------------------------------
    :param request:
    :return:
    """
    '''
    触发示例（无需 body）
    最简单：空 POST（JWT 头按需加）
    # 空 body 触发，任务内默认收集“最近15分钟”的一批
    curl -X POST "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/"

    指定收集窗口/限流（可选）
    curl -X POST "http://127.0.0.1:8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/" \
         -H "Content-Type: application/json" \
         -d '{"query_window_minutes": 10, "max_items": 100}'

    带 JWT（推荐生产）
    ACCESS=<你的access>
    http POST :8000/AppleStockChecker/purchasing-time-analyses/dispatch_ts/ \
      "Authorization: Bearer $ACCESS"

    '''
    body = request.data or {}
    job_id = body.get("job_id") or uuid4().hex
    async_res = batch_generate_psta_same_ts.apply_async(
        kwargs={
            "job_id": job_id,                                 # ← 任务参数
            "items": body.get("items"),                       # 可为 None：任务内 collect_items_for_psta
            "timestamp_iso": body.get("timestamp_iso"),       # 可省略：任务内取最近过去整分钟
            "chunk_size": body.get("chunk_size", 200),        # 兼容旧签名（并行桶版不会用到）
            "query_window_minutes": body.get("query_window_minutes", 15),
            "shop_ids": body.get("shop_ids"),
            "iphone_ids": body.get("iphone_ids"),
            "max_items": body.get("max_items"),
        },
        task_id=job_id,                                       # ← Celery 的 task_id 与 job_id 一致
    )

    # 3) 返回：task_id 与 job_id 相同，便于前端订阅 /ws/task/<job_id>/ 或记录追踪
    return Response(
        {"task_id": async_res.id, "job_id": job_id},
        status=status.HTTP_202_ACCEPTED,
    )


@api_view(["POST"])
@permission_classes([AllowAny])
def dispatch_psta_range(request):
    """
    POST body:
      {
        "start": "2025-10-01T00:00:00+09:00",
        "end":   "2025-11-01T00:00:00+09:00",
        "shop_ids": [...],         # 可选
        "iphone_ids": [...],       # 可选
        "step_minutes": 1,         # 可选；15分钟桶可以传15
        "query_window_minutes": 15,# 拉取窗口
        "max_items": 1000          # 限流
      }
    """
    body = request.data or {}
    job_id = body.get("job_id") or uuid4().hex
    start = _to_aware(body["start"])
    end   = _to_aware(body["end"])
    step  = int(body.get("step_minutes", 1))
    # 生成时间序列（建议倒序），并发控制交给 Celery
    ts_list = []
    cur = end
    while cur >= start:
        ts_list.append(cur.isoformat(timespec="seconds"))
        cur = cur - timezone.timedelta(minutes=step)

    # 组装并发子任务（或分批次多次 apply_async）
    subtasks = [
        batch_generate_psta_same_ts.s(
            job_id=job_id,
            timestamp_iso=ts_iso,
            query_window_minutes=body.get("query_window_minutes", 15),
            shop_ids=body.get("shop_ids"),
            iphone_ids=body.get("iphone_ids"),
            max_items=body.get("max_items"),
        ) for ts_iso in ts_list
    ]
    # 并发数可用 chord+chunks 控制；或者逐批次 apply_async
    chord_res = chord(subtasks)(psta_finalize_buckets.s(job_id, ts_list[0]))
    return Response({"job_id": job_id, "count": len(ts_list), "chord_id": chord_res.id}, status=202)
