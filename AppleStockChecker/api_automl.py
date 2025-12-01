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
from AppleStockChecker.models import (
    AutomlCausalJob,
    AutomlCausalEdge,
    AutomlGrangerResult,
    AutomlVarModel,
    AutomlPreprocessedSeries,
    Iphone,
    SecondHandShop,
)
from django.db.models import Count, Q
import json

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


# ============================================================================
# 机型管理 API
# ============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class IphoneListView(APIView):
    """
    获取所有机型列表
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        try:
            # 获取所有有数据的机型
            iphones = Iphone.objects.filter(
                purchasing_time_analysis__isnull=False
            ).distinct().order_by('part_number')

            return Response({
                "status": "success",
                "count": iphones.count(),
                "iphones": [
                    {
                        "id": iphone.id,
                        "part_number": iphone.part_number,
                        "model_name": iphone.model_name or iphone.part_number,
                    }
                    for iphone in iphones
                ]
            }, status=http_status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching iPhone list: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


@method_decorator(csrf_exempt, name='dispatch')
class BatchCreateAutoMLJobsView(APIView):
    """
    批量创建 AutoML 任务（所有机型或指定机型列表）
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            days = request.data.get('days', 7)
            iphone_ids = request.data.get('iphone_ids', None)  # 如果为 None，则分析所有机型

            now = timezone.now()
            window_end = now
            window_start = now - timezone.timedelta(days=days)

            if iphone_ids:
                # 指定机型
                iphones = Iphone.objects.filter(id__in=iphone_ids)
            else:
                # 所有有数据的机型
                iphones = Iphone.objects.filter(
                    purchasing_time_analysis__Timestamp_Time__gte=window_start
                ).distinct()

            created_jobs = []
            existing_jobs = []

            for iphone in iphones:
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
                    logger.info(f"Created job {job.id} for {iphone.part_number}, task {task.id}")

                if created:
                    created_jobs.append({
                        "job_id": job.id,
                        "iphone": iphone.part_number,
                    })
                else:
                    existing_jobs.append({
                        "job_id": job.id,
                        "iphone": iphone.part_number,
                        "status": job.preprocessing_status,
                    })

            return Response({
                "status": "success",
                "created_count": len(created_jobs),
                "existing_count": len(existing_jobs),
                "created_jobs": created_jobs,
                "existing_jobs": existing_jobs,
            }, status=http_status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error batch creating AutoML jobs: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================================
# 结果查询 API
# ============================================================================

@method_decorator(csrf_exempt, name='dispatch')
class AutoMLJobResultView(APIView):
    """
    获取 AutoML 任务的详细结果
    包括 VAR 系数、Granger 结果、因果边
    """
    permission_classes = [AllowAny]

    def get(self, request, job_id, *args, **kwargs):
        try:
            job = AutomlCausalJob.objects.select_related('iphone').get(pk=job_id)

            # 检查任务是否完成
            if job.impact_status != AutomlCausalJob.StageStatus.SUCCESS:
                return Response({
                    "status": "incomplete",
                    "message": "Analysis not yet complete",
                    "preprocessing_status": job.preprocessing_status,
                    "cause_effect_status": job.cause_effect_status,
                    "impact_status": job.impact_status,
                }, status=http_status.HTTP_200_OK)

            # 获取 VAR 模型
            var_model = AutomlVarModel.objects.filter(job=job).first()

            # 获取因果边（显著的因果关系）
            causal_edges = AutomlCausalEdge.objects.filter(
                job=job,
                enabled=True
            ).select_related('cause_shop', 'effect_shop').order_by('-weight')

            # 获取 Granger 结果
            granger_results = AutomlGrangerResult.objects.filter(
                job=job,
                is_significant=True
            ).select_related('cause_shop', 'effect_shop').order_by('min_pvalue')

            # 计算领头店排名
            leader_shops = self._calculate_leader_shops(causal_edges)

            # 构建因果邻接矩阵
            adjacency_matrix = self._build_adjacency_matrix(causal_edges, var_model)

            return Response({
                "status": "success",
                "job": {
                    "id": job.id,
                    "iphone": job.iphone.part_number,
                    "window_start": job.window_start.isoformat(),
                    "window_end": job.window_end.isoformat(),
                },
                "var_model": {
                    "shop_ids": var_model.shop_ids if var_model else [],
                    "lag_order": var_model.lag_order if var_model else 0,
                    "aic": var_model.aic if var_model else None,
                    "bic": var_model.bic if var_model else None,
                    "sample_size": var_model.sample_size if var_model else 0,
                } if var_model else None,
                "causal_edges": [
                    {
                        "cause_shop_id": edge.cause_shop_id,
                        "cause_shop_name": edge.cause_shop.name,
                        "effect_shop_id": edge.effect_shop_id,
                        "effect_shop_name": edge.effect_shop.name,
                        "weight": edge.weight,
                        "min_pvalue": edge.min_pvalue,
                        "confidence": edge.confidence,
                        "main_lag": edge.main_lag,
                    }
                    for edge in causal_edges
                ],
                "granger_results": [
                    {
                        "cause_shop_id": gr.cause_shop_id,
                        "cause_shop_name": gr.cause_shop.name,
                        "effect_shop_id": gr.effect_shop_id,
                        "effect_shop_name": gr.effect_shop.name,
                        "min_pvalue": gr.min_pvalue,
                        "best_lag": gr.best_lag,
                        "pvalues_by_lag": gr.pvalues_by_lag,
                    }
                    for gr in granger_results[:20]  # 限制返回数量
                ],
                "leader_shops": leader_shops,
                "adjacency_matrix": adjacency_matrix,
            }, status=http_status.HTTP_200_OK)

        except AutomlCausalJob.DoesNotExist:
            return Response({
                "status": "error",
                "error": f"Job with id {job_id} not found"
            }, status=http_status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error fetching AutoML job result: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _calculate_leader_shops(self, causal_edges):
        """
        计算领头店排名
        领头店 = 对其他店铺有最多/最强因果影响的店铺
        """
        shop_influence = {}

        for edge in causal_edges:
            cause_id = edge.cause_shop_id
            if cause_id not in shop_influence:
                shop_influence[cause_id] = {
                    "shop_id": cause_id,
                    "shop_name": edge.cause_shop.name,
                    "outgoing_count": 0,
                    "total_weight": 0.0,
                    "avg_weight": 0.0,
                    "incoming_count": 0,
                }

            shop_influence[cause_id]["outgoing_count"] += 1
            shop_influence[cause_id]["total_weight"] += edge.weight

            # 同时统计被影响的次数
            effect_id = edge.effect_shop_id
            if effect_id not in shop_influence:
                shop_influence[effect_id] = {
                    "shop_id": effect_id,
                    "shop_name": edge.effect_shop.name,
                    "outgoing_count": 0,
                    "total_weight": 0.0,
                    "avg_weight": 0.0,
                    "incoming_count": 0,
                }
            shop_influence[effect_id]["incoming_count"] += 1

        # 计算平均权重
        for shop_id, data in shop_influence.items():
            if data["outgoing_count"] > 0:
                data["avg_weight"] = data["total_weight"] / data["outgoing_count"]

        # 按影响力排序（出度 * 平均权重）
        ranked_shops = sorted(
            shop_influence.values(),
            key=lambda x: x["outgoing_count"] * x["avg_weight"],
            reverse=True
        )

        return ranked_shops[:10]  # 返回前10名

    def _build_adjacency_matrix(self, causal_edges, var_model):
        """
        构建因果邻接矩阵
        矩阵元素 A[i][j] = 店铺 j 对店铺 i 的因果影响强度
        """
        if not var_model:
            return None

        shop_ids = var_model.shop_ids
        n_shops = len(shop_ids)

        # 初始化邻接矩阵
        matrix = [[0.0 for _ in range(n_shops)] for _ in range(n_shops)]

        # 填充因果边的权重
        for edge in causal_edges:
            try:
                cause_idx = shop_ids.index(edge.cause_shop_id)
                effect_idx = shop_ids.index(edge.effect_shop_id)
                matrix[effect_idx][cause_idx] = edge.weight
            except ValueError:
                continue

        # 获取店铺名称
        shop_names = []
        for shop_id in shop_ids:
            try:
                shop = SecondHandShop.objects.get(pk=shop_id)
                shop_names.append(shop.name)
            except SecondHandShop.DoesNotExist:
                shop_names.append(f"Shop {shop_id}")

        return {
            "shop_ids": shop_ids,
            "shop_names": shop_names,
            "matrix": matrix,
            "size": n_shops,
        }


@method_decorator(csrf_exempt, name='dispatch')
class CompletedJobsListView(APIView):
    """
    获取已完成的 AutoML 任务列表（按机型分组）
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        try:
            # 只获取已完成的任务
            completed_jobs = AutomlCausalJob.objects.filter(
                impact_status=AutomlCausalJob.StageStatus.SUCCESS
            ).select_related('iphone').order_by('-created_at')

            # 按机型分组
            jobs_by_iphone = {}
            for job in completed_jobs:
                iphone_id = job.iphone_id
                if iphone_id not in jobs_by_iphone:
                    jobs_by_iphone[iphone_id] = {
                        "iphone_id": iphone_id,
                        "iphone_name": job.iphone.part_number,
                        "jobs": []
                    }

                jobs_by_iphone[iphone_id]["jobs"].append({
                    "job_id": job.id,
                    "window_start": job.window_start.isoformat(),
                    "window_end": job.window_end.isoformat(),
                    "created_at": job.created_at.isoformat(),
                })

            return Response({
                "status": "success",
                "count": len(jobs_by_iphone),
                "jobs_by_iphone": list(jobs_by_iphone.values()),
            }, status=http_status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error fetching completed jobs: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "error": str(e)
            }, status=http_status.HTTP_500_INTERNAL_SERVER_ERROR)
