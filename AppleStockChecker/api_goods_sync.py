"""
外部商品价格同步 API
提供商品映射、价格同步等功能
"""
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from drf_spectacular.types import OpenApiTypes
from AppleStockChecker.services import ExternalGoodsSyncService
from AppleStockChecker.models import Iphone

logger = logging.getLogger(__name__)


class GoodsSyncFetchView(APIView):
    """
    从外部项目获取商品列表并生成映射关系
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="同步外部商品映射",
        description="从外部项目API获取所有商品信息,并自动映射到本项目的Iphone实例",
        parameters=[
            OpenApiParameter(
                name='api_url',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description='外部API URL (可选,默认使用配置)',
                required=False
            ),
            OpenApiParameter(
                name='category_filter',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description='商品类别过滤 (如 "iPhone")，只同步指定类别的商品',
                required=False
            ),
        ],
        responses={
            200: OpenApiResponse(
                description="同步成功",
                response={
                    'type': 'object',
                    'properties': {
                        'success': {'type': 'boolean'},
                        'message': {'type': 'string'},
                        'statistics': {
                            'type': 'object',
                            'properties': {
                                'total_items': {'type': 'integer'},
                                'matched_items': {'type': 'integer'},
                                'unmatched_items': {'type': 'integer'},
                                'error_items': {'type': 'integer'},
                            }
                        }
                    }
                }
            ),
            400: OpenApiResponse(description="请求参数错误"),
            500: OpenApiResponse(description="同步失败"),
        }
    )
    def post(self, request):
        """执行同步操作"""
        try:
            # 获取可选参数
            api_url = request.query_params.get('api_url')
            api_token = request.query_params.get('api_token')
            category_filter = request.query_params.get('category_filter')

            # 创建同步服务
            sync_service = ExternalGoodsSyncService(
                api_url=api_url,
                api_token=api_token,
                category_filter=category_filter
            )

            # 执行同步
            stats = sync_service.sync_goods_mappings()

            return Response({
                'success': True,
                'message': '商品映射同步完成',
                'statistics': stats
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Sync failed: {e}", exc_info=True)
            return Response({
                'success': False,
                'message': f'同步失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GoodsMappingsView(APIView):
    """
    查看当前的商品映射关系
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="获取商品映射列表",
        description="查询当前所有的外部商品与本项目Iphone的映射关系",
        parameters=[
            OpenApiParameter(
                name='status',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description='过滤状态: matched(已匹配), unmatched(未匹配), pending(待处理), error(错误)',
                required=False,
                enum=['matched', 'unmatched', 'pending', 'error']
            ),
            OpenApiParameter(
                name='limit',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='返回结果数量限制',
                required=False
            ),
        ],
        responses={
            200: OpenApiResponse(
                description="查询成功",
                response={
                    'type': 'object',
                    'properties': {
                        'success': {'type': 'boolean'},
                        'total': {'type': 'integer'},
                        'mappings': {'type': 'array'}
                    }
                }
            ),
        }
    )
    def get(self, request):
        """获取映射列表"""
        try:
            # 获取过滤参数
            filter_status = request.query_params.get('status')
            limit = request.query_params.get('limit')

            # 创建同步服务
            sync_service = ExternalGoodsSyncService()

            # 获取映射列表
            mappings = sync_service.get_all_mappings(status=filter_status)

            # 应用限制
            if limit:
                try:
                    limit = int(limit)
                    mappings = mappings[:limit]
                except ValueError:
                    pass

            return Response({
                'success': True,
                'total': len(mappings),
                'mappings': mappings
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Failed to get mappings: {e}", exc_info=True)
            return Response({
                'success': False,
                'message': f'获取映射失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GoodsMappingStatisticsView(APIView):
    """
    获取映射统计信息
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="获取映射统计信息",
        description="获取商品映射的汇总统计数据",
        responses={
            200: OpenApiResponse(
                description="统计信息",
                response={
                    'type': 'object',
                    'properties': {
                        'success': {'type': 'boolean'},
                        'statistics': {
                            'type': 'object',
                            'properties': {
                                'total': {'type': 'integer', 'description': '总映射数'},
                                'matched': {'type': 'integer', 'description': '已匹配数'},
                                'unmatched': {'type': 'integer', 'description': '未匹配数'},
                                'pending': {'type': 'integer', 'description': '待处理数'},
                                'error': {'type': 'integer', 'description': '错误数'},
                                'last_sync_at': {'type': 'string', 'description': '最后同步时间'},
                            }
                        }
                    }
                }
            ),
        }
    )
    def get(self, request):
        """获取统计信息"""
        try:
            sync_service = ExternalGoodsSyncService()
            stats = sync_service.get_mapping_statistics()

            return Response({
                'success': True,
                'statistics': stats
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return Response({
                'success': False,
                'message': f'获取统计信息失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UpdateExternalPriceView(APIView):
    """
    根据本项目的 Iphone ID 更新外部项目的商品价格
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="更新外部项目商品价格",
        description="根据本项目的Iphone ID,更新外部项目中对应商品的价格",
        request={
            'type': 'object',
            'properties': {
                'iphone_id': {
                    'type': 'integer',
                    'description': '本项目的Iphone ID'
                },
                'new_price': {
                    'type': 'integer',
                    'description': '新价格(日元)'
                }
            },
            'required': ['iphone_id', 'new_price']
        },
        responses={
            200: OpenApiResponse(
                description="更新成功",
                response={
                    'type': 'object',
                    'properties': {
                        'success': {'type': 'boolean'},
                        'message': {'type': 'string'},
                        'results': {
                            'type': 'object',
                            'properties': {
                                'total': {'type': 'integer'},
                                'success': {'type': 'integer'},
                                'failed': {'type': 'integer'},
                                'details': {'type': 'array'}
                            }
                        }
                    }
                }
            ),
            400: OpenApiResponse(description="请求参数错误"),
            404: OpenApiResponse(description="Iphone不存在"),
        }
    )
    def post(self, request):
        """更新价格"""
        try:
            # 获取参数
            iphone_id = request.data.get('iphone_id')
            new_price = request.data.get('new_price')

            # 参数验证
            if not iphone_id or not new_price:
                return Response({
                    'success': False,
                    'message': '缺少必需参数: iphone_id 和 new_price'
                }, status=status.HTTP_400_BAD_REQUEST)

            # 验证 Iphone 是否存在
            try:
                iphone = Iphone.objects.get(id=iphone_id)
            except Iphone.DoesNotExist:
                return Response({
                    'success': False,
                    'message': f'Iphone ID {iphone_id} 不存在'
                }, status=status.HTTP_404_NOT_FOUND)

            # 执行价格更新
            sync_service = ExternalGoodsSyncService()
            results = sync_service.update_external_price(iphone_id, int(new_price))

            return Response({
                'success': True,
                'message': f'已更新 {results["success"]}/{results["total"]} 个商品价格',
                'iphone_info': {
                    'id': iphone.id,
                    'part_number': iphone.part_number,
                    'model_name': iphone.model_name,
                    'capacity_gb': iphone.capacity_gb,
                    'color': iphone.color,
                },
                'results': results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Failed to update price: {e}", exc_info=True)
            return Response({
                'success': False,
                'message': f'更新价格失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchUpdateExternalPricesView(APIView):
    """
    批量更新外部项目商品价格
    """
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="批量更新外部项目商品价格",
        description="批量更新多个Iphone对应的外部商品价格",
        request={
            'type': 'object',
            'properties': {
                'updates': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'iphone_id': {'type': 'integer'},
                            'new_price': {'type': 'integer'}
                        }
                    }
                }
            },
            'required': ['updates']
        },
        responses={
            200: OpenApiResponse(description="批量更新完成"),
            400: OpenApiResponse(description="请求参数错误"),
        }
    )
    def post(self, request):
        """批量更新价格"""
        try:
            updates = request.data.get('updates', [])

            if not updates:
                return Response({
                    'success': False,
                    'message': '缺少更新列表'
                }, status=status.HTTP_400_BAD_REQUEST)

            sync_service = ExternalGoodsSyncService()
            all_results = []

            for update in updates:
                iphone_id = update.get('iphone_id')
                new_price = update.get('new_price')

                if not iphone_id or not new_price:
                    all_results.append({
                        'iphone_id': iphone_id,
                        'success': False,
                        'message': '缺少必需参数'
                    })
                    continue

                try:
                    iphone = Iphone.objects.get(id=iphone_id)
                    results = sync_service.update_external_price(iphone_id, int(new_price))

                    all_results.append({
                        'iphone_id': iphone_id,
                        'iphone_info': str(iphone),
                        'success': True,
                        'results': results
                    })
                except Iphone.DoesNotExist:
                    all_results.append({
                        'iphone_id': iphone_id,
                        'success': False,
                        'message': f'Iphone ID {iphone_id} 不存在'
                    })
                except Exception as e:
                    all_results.append({
                        'iphone_id': iphone_id,
                        'success': False,
                        'message': str(e)
                    })

            return Response({
                'success': True,
                'message': f'批量更新完成',
                'results': all_results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Batch update failed: {e}", exc_info=True)
            return Response({
                'success': False,
                'message': f'批量更新失败: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
