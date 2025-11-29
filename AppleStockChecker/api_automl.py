# -*- coding: utf-8 -*-
"""
AutoML API Endpoints
用于触发 AutoML 任务的 API 端点
"""
from __future__ import annotations
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status as http_status
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.utils import timezone
import logging

from AppleStockChecker.tasks.automl_tasks import (
    preprocessing_rapid,
    cause_and_effect_testing,
    quantification_of_impact,
    schedule_automl_jobs,
    run_preprocessing_for_job,
    run_var_for_job,
    run_impact_for_job,
)
from AppleStockChecker.models import AutomlCausalJob, Iphone

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
            }, status=http_status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error triggering Preprocessing-Rapid task: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


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
            }, status=http_status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error triggering Cause-and-Effect-Testing task: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


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
            }, status=http_status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error triggering Quantification-of-Impact task: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================================
# 完整 Pipeline API 端点
# ============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class ScheduleAutoMLJobsView(APIView):
    """
    调度 AutoML 因果分析任务
    为所有活跃机种创建分析任务
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            # 触发调度任务
            task = schedule_automl_jobs.delay()
            logger.info(f"AutoML job scheduling triggered: {task.id}")

            return Response({
                "status": "success",
                "message": "AutoML job scheduling triggered successfully",
                "task_id": task.id
            }, status=http_status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error scheduling AutoML jobs: {str(e)}")
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class CreateAutoMLJobView(APIView):
    """
    为指定机种创建 AutoML 分析任务
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            iphone_id = request.data.get('iphone_id')
            days = request.data.get('days', 7)

            if not iphone_id:
                return Response({
                    "status": "error",
                    "error": "iphone_id is required"
                }, status=http_status.HTTP_400_BAD_REQUEST)

            iphone = Iphone.objects.get(pk=iphone_id)

            now = timezone.now()
            window_end = now
            window_start = now - timezone.timedelta(days=days)

            # 创建或获取 Job
            job, created = AutomlCausalJob.objects.get_or_create(
                iphone=iphone,
                window_start=window_start,
                window_end=window_end,
                bucket_freq="10min",
                defaults={"priority": 100},
            )

            # 触发预处理任务
            if job.preprocessing_status in [
                AutomlCausalJob.StageStatus.PENDING,
                AutomlCausalJob.StageStatus.FAILED,
            ]:
                task = run_preprocessing_for_job.apply_async(
                    args=[job.id],
                    queue="automl_preprocessing"
                )
                logger.info(f"Created job {job.id} for iphone {iphone_id}, triggered preprocessing task {task.id}")
            else:
                logger.info(f"Job {job.id} already exists with status {job.preprocessing_status}")

            return Response({
                "status": "success",
                "job_id": job.id,
                "created": created,
                "iphone": iphone.part_number,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "preprocessing_status": job.preprocessing_status,
                "cause_effect_status": job.cause_effect_status,
                "impact_status": job.impact_status,
            }, status=http_status.HTTP_201_CREATED if created else http_status.HTTP_200_OK)

        except Iphone.DoesNotExist:
            return Response({
                "status": "error",
                "error": f"iPhone with id {iphone_id} not found"
            }, status=http_status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error creating AutoML job: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class AutoMLJobStatusView(APIView):
    """
    查询 AutoML 任务状态
    """
    permission_classes = [AllowAny]

    def get(self, request, job_id=None, *args, **kwargs):
        try:
            if job_id:
                # 查询单个任务
                job = AutomlCausalJob.objects.select_related('iphone').get(pk=job_id)
                return Response({
                    "status": "success",
                    "job": self._serialize_job(job),
                }, status=http_status.HTTP_200_OK)
            else:
                # 查询所有任务（可以添加过滤参数）
                limit = int(request.GET.get('limit', 20))
                jobs = AutomlCausalJob.objects.select_related('iphone').order_by('-created_at')[:limit]

                return Response({
                    "status": "success",
                    "count": jobs.count(),
                    "jobs": [self._serialize_job(job) for job in jobs],
                }, status=http_status.HTTP_200_OK)

        except AutomlCausalJob.DoesNotExist:
            return Response({
                "status": "error",
                "error": f"Job with id {job_id} not found"
            }, status=http_status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error querying AutoML job status: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _serialize_job(self, job):
        """序列化Job对象"""
        return {
            "id": job.id,
            "iphone_id": job.iphone_id,
            "iphone": job.iphone.part_number,
            "window_start": job.window_start.isoformat(),
            "window_end": job.window_end.isoformat(),
            "bucket_freq": job.bucket_freq,
            "preprocessing": {
                "status": job.preprocessing_status,
                "started_at": job.preprocessing_started_at.isoformat() if job.preprocessing_started_at else None,
                "finished_at": job.preprocessing_finished_at.isoformat() if job.preprocessing_finished_at else None,
            },
            "cause_effect": {
                "status": job.cause_effect_status,
                "started_at": job.cause_effect_started_at.isoformat() if job.cause_effect_started_at else None,
                "finished_at": job.cause_effect_finished_at.isoformat() if job.cause_effect_finished_at else None,
            },
            "impact": {
                "status": job.impact_status,
                "started_at": job.impact_started_at.isoformat() if job.impact_started_at else None,
                "finished_at": job.impact_finished_at.isoformat() if job.impact_finished_at else None,
            },
            "priority": job.priority,
            "retry_count": job.retry_count,
            "last_error": job.last_error,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }
