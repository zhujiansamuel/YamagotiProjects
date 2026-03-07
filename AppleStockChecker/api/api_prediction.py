# -*- coding: utf-8 -*-
"""
Price Prediction API Endpoints
价格预测模型的训练触发、推理触发、预测结果查询。
"""
from __future__ import annotations

import logging

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status as http_status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from AppleStockChecker.models import ForecastSnapshot, ModelArtifact, Iphone
from AppleStockChecker.tasks.prediction_tasks import (
    train_price_models_task,
    predict_prices_task,
)

logger = logging.getLogger(__name__)


# ── 触发训练 ───────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class TriggerTrainingView(APIView):
    """POST 触发异步 LightGBM 训练任务。

    Body:
        iphone_ids: list[int]  (必须)
        days: int              (默认 14)
        run_id: str            (默认 "live")
        version: str           (默认 "v1")
    """
    permission_classes = [AllowAny]

    def post(self, request):
        iphone_ids = request.data.get("iphone_ids")
        if not iphone_ids:
            return Response(
                {"error": "iphone_ids is required"},
                status=http_status.HTTP_400_BAD_REQUEST,
            )
        if not isinstance(iphone_ids, list):
            return Response(
                {"error": "iphone_ids must be a list of integers"},
                status=http_status.HTTP_400_BAD_REQUEST,
            )

        days = int(request.data.get("days", 14))
        run_id = request.data.get("run_id", "live")
        version = request.data.get("version", "v1")

        task = train_price_models_task.delay(iphone_ids, days=days, run_id=run_id, version=version)
        logger.info("Training task triggered: %s for iphones=%s", task.id, iphone_ids)

        return Response({
            "status": "success",
            "message": f"Training task triggered for {len(iphone_ids)} iPhones",
            "task_id": task.id,
        })


# ── 触发推理 ───────────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class TriggerPredictionView(APIView):
    """POST 触发异步推理任务。

    Body:
        iphone_ids: list[int]  (必须)
        run_id: str            (默认 "live")
        version: str           (默认 "v1")
    """
    permission_classes = [AllowAny]

    def post(self, request):
        iphone_ids = request.data.get("iphone_ids")
        if not iphone_ids:
            return Response(
                {"error": "iphone_ids is required"},
                status=http_status.HTTP_400_BAD_REQUEST,
            )
        if not isinstance(iphone_ids, list):
            return Response(
                {"error": "iphone_ids must be a list of integers"},
                status=http_status.HTTP_400_BAD_REQUEST,
            )

        run_id = request.data.get("run_id", "live")
        version = request.data.get("version", "v1")

        task = predict_prices_task.delay(iphone_ids, run_id=run_id, version=version)
        logger.info("Prediction task triggered: %s for iphones=%s", task.id, iphone_ids)

        return Response({
            "status": "success",
            "message": f"Prediction task triggered for {len(iphone_ids)} iPhones",
            "task_id": task.id,
        })


# ── 预测结果查询 ──────────────────────────────────────────────────────

class ForecastListView(APIView):
    """GET 查询预测结果。

    Query params:
        iphone_id: int         (可选, 筛选机型)
        model_name: str        (可选)
        horizon_min: int       (可选, 15/30/45/60)
        limit: int             (默认 100)
        offset: int            (默认 0)
    """
    permission_classes = [AllowAny]

    def get(self, request):
        qs = ForecastSnapshot.objects.all().order_by("-bucket")

        iphone_id = request.query_params.get("iphone_id")
        if iphone_id:
            qs = qs.filter(iphone_id=int(iphone_id))

        model_name = request.query_params.get("model_name")
        if model_name:
            qs = qs.filter(model_name=model_name)

        horizon_min = request.query_params.get("horizon_min")
        if horizon_min:
            qs = qs.filter(horizon_min=int(horizon_min))

        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
        total = qs.count()
        rows = qs[offset:offset + limit]

        data = [
            {
                "id": r.id,
                "bucket": r.bucket.isoformat() if r.bucket else None,
                "model_name": r.model_name,
                "version": r.version,
                "horizon_min": r.horizon_min,
                "iphone_id": r.iphone_id,
                "yhat": r.yhat,
                "yhat_var": r.yhat_var,
                "is_final": r.is_final,
            }
            for r in rows
        ]

        return Response({"count": total, "results": data})


# ── 模型产物查询 ──────────────────────────────────────────────────────

class ModelArtifactListView(APIView):
    """GET 查询已训练模型列表 (不含二进制 blob)。

    Query params:
        iphone_id: int    (可选, 按 model_name 中的 iphone_id 过滤)
        version: str      (可选)
    """
    permission_classes = [AllowAny]

    def get(self, request):
        qs = ModelArtifact.objects.filter(
            model_name__startswith="lgbm_price"
        ).order_by("-trained_at")

        iphone_id = request.query_params.get("iphone_id")
        if iphone_id:
            qs = qs.filter(model_name__contains=f"_{iphone_id}_")

        version = request.query_params.get("version")
        if version:
            qs = qs.filter(version=version)

        data = [
            {
                "id": a.id,
                "model_name": a.model_name,
                "version": a.version,
                "trained_at": a.trained_at.isoformat() if a.trained_at else None,
                "metrics": a.metrics_json,
            }
            for a in qs[:200]
        ]

        return Response({"count": len(data), "results": data})
