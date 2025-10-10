import uuid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .tasks.timestamp_alignment_task import generate_purchasing_time_analysis
from .tasks.timestamp_alignment_task import batch_generate_psta_same_ts


@api_view(["POST"])
def dispatch_generate_analysis(request):
    payload = request.data or {}
    job_id = payload.get("Job_ID") or uuid.uuid4().hex
    payload["Job_ID"] = job_id
    # 异步发起
    async_result = generate_purchasing_time_analysis.delay(payload=payload)
    return Response({"task_id": async_result.id, "Job_ID": job_id}, status=status.HTTP_202_ACCEPTED)




@api_view(["POST"])
def dispatch_psta_batch_same_ts(request):
    body = request.data
    async_res = batch_generate_psta_same_ts.delay(
        job_id=body.get("job_id"),
        timestamp_iso=None,              # ← 不传/传 None 都行
        items=body["items"],
        chunk_size=body.get("chunk_size", 200),
    )
    return Response({"task_id": async_res.id}, status=status.HTTP_202_ACCEPTED)
