# -*- coding: utf-8 -*-
"""
AutoML API Endpoints
用于触发 AutoML 任务的 API 端点
"""
from __future__ import annotations
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import logging

from AppleStockChecker.tasks.automl_tasks import (
    preprocessing_rapid,
    cause_and_effect_testing,
    quantification_of_impact
)

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class TriggerPreprocessingRapidView(APIView):
    """
    Trigger Preprocessing-Rapid Task
    触发快速预处理任务
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            # 异步触发任务
            task = preprocessing_rapid.delay()
            logger.info(f"Preprocessing-Rapid task triggered: {task.id}")

            return Response({
                "status": "success",
                "message": "Preprocessing-Rapid task triggered successfully",
                "task_id": task.id
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error triggering Preprocessing-Rapid task: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class TriggerCauseAndEffectTestingView(APIView):
    """
    Trigger Cause-and-Effect-Testing Task
    触发因果关系测试任务
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            # 异步触发任务
            task = cause_and_effect_testing.delay()
            logger.info(f"Cause-and-Effect-Testing task triggered: {task.id}")

            return Response({
                "status": "success",
                "message": "Cause-and-Effect-Testing task triggered successfully",
                "task_id": task.id
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error triggering Cause-and-Effect-Testing task: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class TriggerQuantificationOfImpactView(APIView):
    """
    Trigger Quantification-of-Impact Task
    触发影响量化任务
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            # 异步触发任务
            task = quantification_of_impact.delay()
            logger.info(f"Quantification-of-Impact task triggered: {task.id}")

            return Response({
                "status": "success",
                "message": "Quantification-of-Impact task triggered successfully",
                "task_id": task.id
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error triggering Quantification-of-Impact task: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
