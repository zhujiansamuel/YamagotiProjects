from django.shortcuts import render
from django.utils import timezone
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework import status
from django.utils.dateparse import parse_date
from rest_framework import viewsets, permissions, filters
from drf_spectacular.utils import (
    extend_schema, extend_schema_view, OpenApiParameter, OpenApiTypes
)
from django.utils.dateparse import parse_datetime

from .models import Iphone, OfficialStore, InventoryRecord
from .serializers import OfficialStoreSerializer, InventoryRecordSerializer, IphoneSerializer
from .serializers import UserSerializer
# Create your views here.

class HealthView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response(
            {
                "status": "ok",
                "server_time": timezone.now().isoformat(),
                "app": "api",
                "version": "1.0.0",
            },
            status=status.HTTP_200_OK,
        )

class ApiRoot(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response(
            {
                "health": "/api/health/",
                "docs": "/api/docs/",
                "schema": "/api/schema/",
                "auth_token_obtain": "/api/auth/token/",
                "auth_token_refresh": "/api/auth/token/refresh/",
                "auth_token_verify": "/api/auth/token/verify/",
                "me": "/api/me/",
            }
        )

class MeView(APIView):
    def get(self, request):
        data = UserSerializer(request.user).data
        return Response(data)
#

@extend_schema_view(
    list=extend_schema(tags=["Apple / Store"], summary="门店列表", auth=[]),
    retrieve=extend_schema(tags=["Apple / Store"], summary="门店详情", auth=[]),
    create=extend_schema(tags=["Apple / Store"], summary="创建门店"),
    update=extend_schema(tags=["Apple / Store"], summary="更新门店（整体）"),
    partial_update=extend_schema(tags=["Apple / Store"], summary="更新门店（部分）"),
    destroy=extend_schema(tags=["Apple / Store"], summary="删除门店"),
)

class IphoneViewSet(viewsets.ModelViewSet):
    """
    /api/iphones/ 列表、创建
    /api/iphones/{part_number}/ 详情、更新、删除
    支持查询参数：
      - model: 按型号包含匹配
      - color: 按颜色包含匹配
      - capacity: 容量精确匹配（单位GB）
      - min_capacity / max_capacity: 容量范围（GB）
      - released_after / released_before: 上市日期区间(YYYY-MM-DD)
      - search: 在 part_number / model_name / color 上全文搜索
      - ordering: 排序字段（release_date, capacity_gb, model_name, color；默认 -release_date）
    """
    queryset = Iphone.objects.all()
    serializer_class = IphoneSerializer

    # 以 Apple Part Number 作为资源定位字段
    lookup_field = "part_number"

    # 搜索与排序（无需安装 django-filter）
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["part_number", "model_name", "color"]
    ordering_fields = ["release_date", "capacity_gb", "model_name", "color"]
    ordering = ["-release_date"]

    def get_permissions(self):
        if self.action in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        # 写操作仅管理员
        return [permissions.IsAdminUser()]

    def get_queryset(self):
        qs = super().get_queryset()

        model_name = self.request.query_params.get("model")
        color = self.request.query_params.get("color")
        capacity = self.request.query_params.get("capacity")
        min_capacity = self.request.query_params.get("min_capacity")
        max_capacity = self.request.query_params.get("max_capacity")
        released_after = self.request.query_params.get("released_after")
        released_before = self.request.query_params.get("released_before")

        if model_name:
            qs = qs.filter(model_name__icontains=model_name)
        if color:
            qs = qs.filter(color__icontains=color)

        if capacity and capacity.isdigit():
            qs = qs.filter(capacity_gb=int(capacity))
        if min_capacity and min_capacity.isdigit():
            qs = qs.filter(capacity_gb__gte=int(min_capacity))
        if max_capacity and max_capacity.isdigit():
            qs = qs.filter(capacity_gb__lte=int(max_capacity))

        if released_after:
            d = parse_date(released_after)
            if d:
                qs = qs.filter(release_date__gte=d)
        if released_before:
            d = parse_date(released_before)
            if d:
                qs = qs.filter(release_date__lte=d)

        return qs


class OfficialStoreViewSet(viewsets.ModelViewSet):
    queryset = OfficialStore.objects.all()
    serializer_class = OfficialStoreSerializer

    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "address"]
    ordering_fields = ["name"]
    ordering = ["name"]

    def get_permissions(self):
        if self.action in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]


def _to_bool(val: str | None):
    if val is None:
        return None
    s = val.strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


@extend_schema_view(
    list=extend_schema(
        tags=["Apple / Inventory"],
        summary="库存记录列表（支持过滤/搜索/排序）",
        auth=[],
        parameters=[
            OpenApiParameter("store", OpenApiTypes.INT, description="按门店ID筛选", required=False),
            OpenApiParameter("store_name", OpenApiTypes.STR, description="按门店名模糊匹配", required=False),
            OpenApiParameter("iphone", OpenApiTypes.INT, description="按 iPhone ID 筛选", required=False),
            OpenApiParameter("iphone_part_number", OpenApiTypes.STR, description="按 iPhone Part Number 精确匹配", required=False),
            OpenApiParameter("has_stock", OpenApiTypes.BOOL, description="是否有库存（true/false）", required=False),
            OpenApiParameter("recorded_after", OpenApiTypes.DATETIME, description="记录时间不早于(ISO8601)", required=False),
            OpenApiParameter("recorded_before", OpenApiTypes.DATETIME, description="记录时间不晚于(ISO8601)", required=False),
            OpenApiParameter("arrival_earliest_after", OpenApiTypes.DATETIME, description="最早到达不早于(ISO8601)", required=False),
            OpenApiParameter("arrival_earliest_before", OpenApiTypes.DATETIME, description="最早到达不晚于(ISO8601)", required=False),
            OpenApiParameter("arrival_latest_after", OpenApiTypes.DATETIME, description="最晚到达不早于(ISO8601)", required=False),
            OpenApiParameter("arrival_latest_before", OpenApiTypes.DATETIME, description="最晚到达不晚于(ISO8601)", required=False),
            OpenApiParameter(
                "ordering",
                OpenApiTypes.STR,
                enum=[
                    "recorded_at", "-recorded_at",
                    "has_stock", "-has_stock",
                    "estimated_arrival_earliest", "-estimated_arrival_earliest",
                    "estimated_arrival_latest", "-estimated_arrival_latest",
                ],
                description="排序字段（默认 -recorded_at）",
                required=False,
            ),
            OpenApiParameter("search", OpenApiTypes.STR, description="在 门店名/PN/型号/颜色 上搜索", required=False),
        ],
    ),
    retrieve=extend_schema(tags=["Apple / Inventory"], summary="库存记录详情", auth=[]),
    create=extend_schema(tags=["Apple / Inventory"], summary="新增库存记录"),
    update=extend_schema(tags=["Apple / Inventory"], summary="更新库存记录（整体）"),
    partial_update=extend_schema(tags=["Apple / Inventory"], summary="更新库存记录（部分）"),
    destroy=extend_schema(tags=["Apple / Inventory"], summary="删除库存记录"),
)
class InventoryRecordViewSet(viewsets.ModelViewSet):
    queryset = InventoryRecord.objects.select_related("store", "iphone").all()
    serializer_class = InventoryRecordSerializer

    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["store__name", "iphone__part_number", "iphone__model_name", "iphone__color"]
    ordering_fields = ["recorded_at", "has_stock", "estimated_arrival_earliest", "estimated_arrival_latest"]
    ordering = ["-recorded_at"]

    def get_permissions(self):
        if self.action in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]

    def get_queryset(self):
        qs = super().get_queryset()
        qp = self.request.query_params

        store = qp.get("store")
        store_name = qp.get("store_name")
        iphone = qp.get("iphone")
        iphone_pn = qp.get("iphone_part_number")
        has_stock = qp.get("has_stock")

        recorded_after = parse_datetime(qp.get("recorded_after") or "")
        recorded_before = parse_datetime(qp.get("recorded_before") or "")
        ae_after = parse_datetime(qp.get("arrival_earliest_after") or "")
        ae_before = parse_datetime(qp.get("arrival_earliest_before") or "")
        al_after = parse_datetime(qp.get("arrival_latest_after") or "")
        al_before = parse_datetime(qp.get("arrival_latest_before") or "")

        if store and store.isdigit():
            qs = qs.filter(store_id=int(store))
        if store_name:
            qs = qs.filter(store__name__icontains=store_name)
        if iphone and iphone.isdigit():
            qs = qs.filter(iphone_id=int(iphone))
        if iphone_pn:
            qs = qs.filter(iphone__part_number=iphone_pn)

        hb = _to_bool(has_stock)
        if hb is not None:
            qs = qs.filter(has_stock=hb)

        if recorded_after:
            qs = qs.filter(recorded_at__gte=recorded_after)
        if recorded_before:
            qs = qs.filter(recorded_at__lte=recorded_before)
        if ae_after:
            qs = qs.filter(estimated_arrival_earliest__gte=ae_after)
        if ae_before:
            qs = qs.filter(estimated_arrival_earliest__lte=ae_before)
        if al_after:
            qs = qs.filter(estimated_arrival_latest__gte=al_after)
        if al_before:
            qs = qs.filter(estimated_arrival_latest__lte=al_before)

        return qs