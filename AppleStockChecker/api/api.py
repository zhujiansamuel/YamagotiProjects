import uuid
from uuid import uuid4
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from AppleStockChecker.tasks.timestamp_alignment_task import (
    batch_generate_psta_same_ts,
    psta_finalize_buckets,
)
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticatedOrReadOnly, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from celery import shared_task, chord

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
    # 空 body 触发，任务内默认收集"最近15分钟"的一批
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
            "job_id": job_id,
            "items": body.get("items"),
            "timestamp_iso": body.get("timestamp_iso"),
            "chunk_size": body.get("chunk_size", 200),
            "query_window_minutes": body.get("query_window_minutes", 15),
            "shop_ids": body.get("shop_ids"),
            "iphone_ids": body.get("iphone_ids"),
            "max_items": body.get("max_items"),

            # ✅ 新参数（统一用这四个）
            "agg_minutes": int(body.get("agg_minutes", 15)),  # 聚合步长（1/5/15）
            "agg_mode": (body.get("agg_mode") or "boundary").lower(),  # 'boundary' | 'rolling' | 'off'
            "force_agg": bool(body.get("force_agg", False)),  # 强制本轮聚合
            "sequential": bool(body.get("sequential", False)),  # 是否顺序执行（默认并发）
        },
        task_id=job_id,
    )
    return Response({"task_id": async_res.id, "job_id": job_id}, status=status.HTTP_202_ACCEPTED)
