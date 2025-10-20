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



#@permission_classes([IsAuthenticatedOrReadOnly])

@api_view(["POST"])
@permission_classes([AllowAny])  # 本地调试也可 AllowAny
def dispatch_psta_batch_same_ts(request):
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