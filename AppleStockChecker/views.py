from __future__ import annotations

from uuid import uuid4
from rest_framework_simplejwt.authentication import JWTAuthentication
from .models import Iphone, OfficialStore, InventoryRecord
from .serializers import OfficialStoreSerializer, InventoryRecordSerializer, IphoneSerializer
from .serializers import UserSerializer
from .serializers import TrendResponseByPNSerializer
from rest_framework import viewsets, permissions, filters
from drf_spectacular.utils import (
    extend_schema, extend_schema_view, OpenApiParameter, OpenApiTypes
)
from datetime import datetime, date
from .serializers import SecondHandShopSerializer, PurchasingShopPriceRecordSerializer
from math import ceil
import csv, io, re
from django.http import HttpResponse
from django.db import transaction, IntegrityError
from datetime import datetime
from django.db import transaction
from django.utils.dateparse import parse_datetime, parse_date
from .utils.tradein_pipeline import clean_and_aggregate_tradein
from .utils.color_norm import synonyms_for_query
from .services.external_ingest_service import ingest_external_sources
from django.db.models import Q
from celery.result import AsyncResult
from AppleStockChecker.utils.external_ingest.webscraper import fetch_webscraper_export_sync, to_dataframe_from_request
from AppleStockChecker.tasks.webscraper_tasks import task_process_webscraper_job,task_process_xlsx
from django.conf import settings
from datetime import timedelta
from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, FormParser, MultiPartParser, FileUploadParser
from rest_framework.parsers import BaseParser
from rest_framework import permissions, status
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes
from rest_framework.decorators import action, parser_classes, authentication_classes, permission_classes
from rest_framework.parsers import JSONParser
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from AppleStockChecker.tasks.webscraper_tasks import task_ingest_json_shop1
from rest_framework import viewsets, mixins
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import OrderingFilter
from .models import PurchasingShopTimeAnalysis
from .serializers import PurchasingShopTimeAnalysisSerializer, PSTACompactSerializer
from .filters import PurchasingShopTimeAnalysisFilter
import io
import re
import uuid
from typing import Optional, Dict, Any

from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import action, authentication_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework_simplejwt.authentication import JWTAuthentication
from AppleStockChecker.utils.external_ingest.registry import get_cleaner, run_cleaner
from AppleStockChecker.models import SecondHandShop, Iphone, PurchasingShopPriceRecord

# 复用你先前实现过的入库主流程（此处假设已存在；若文件不同模块，请调整导入）
from AppleStockChecker.services.external_ingest_service import ingest_external_dataframe


class PlainTextParser(BaseParser):
    media_type = 'text/plain'
    def parse(self, stream, media_type=None, parser_context=None):
        return stream.read()  # 交给你的视图里 to_dataframe_from_request 自己处理

class TextCsvParser(BaseParser):
    media_type = 'text/csv'
    def parse(self, stream, media_type=None, parser_context=None):
        return stream.read()


def _get_bool_param(request, name: str, default: bool = False) -> bool:
    # query 优先；body 仅当 data 是 dict 才读取
    val = request.query_params.get(name, None)
    if val is None and isinstance(getattr(request, "data", None), dict):
        val = request.data.get(name, None)
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    if isinstance(val, bool): return val
    s = str(val).strip().lower()
    if s in {"1","true","t","yes","y","on"}: return True
    if s in {"0","false","f","no","n","off"}: return False
    return default

def _classify_mode(request):
    """
    根据 Content-Type + JSON 形态判断 'direct' or 'webhook'。
    返回: (mode, body_bytes, effective_ct)
    直传: 我们需要 body_bytes + effective_ct 交给 to_dataframe_from_request
    Webhook: 返回 (mode, None, None)
    """
    ct = (request.content_type or "").lower()
    data_obj = getattr(request, "data", None)

    # 1) CSV 直传
    if "csv" in ct:
        return "direct", (request.body or b""), ct

    # 2) JSON
    if "json" in ct:
        # JSON 数组 -> 直传
        if isinstance(data_obj, list):
            return "direct", (request.body or b""), "application/json"
        # JSON 对象 -> 可能是 webhook
        if isinstance(data_obj, dict):
            keys = set(data_obj.keys())
            if (
                ("scrapingjob_id" in keys) or ("job_id" in keys)
                or (("status" in keys) and (("sitemap_id" in keys) or ("sitemap_name" in keys)))
            ):
                return "webhook", None, None
            # 不是 webhook 特征，当作直传 JSON 对象（看你的使用场景也可拒绝）
            return "direct", (request.body or b""), "application/json"

    # 3) multipart/form-data -> 直传（从 request.FILES 取）
    if ct.startswith("multipart/form-data"):
        up = next(iter(request.FILES.values()), None)
        if up:
            return "direct", up.read(), (up.content_type or "text/csv").lower()
        return "direct", (request.body or b""), "application/octet-stream"

    # 4) text/plain -> 直传
    if ct.startswith("text/plain"):
        return "direct", (request.body or b""), "text/plain"

    # 5) fallback: 看 query 上有没有 job_id
    if request.query_params.get("scrapingjob_id") or request.query_params.get("job_id"):
        return "webhook", None, None

    # 默认按直传处理（也可改为 400）
    return "direct", (request.body or b""), ct

def _check_token(request, path_token=None):
    shared = settings.WEB_SCRAPER_WEBHOOK_TOKEN
    incoming = request.headers.get("X-Webhook-Token") \
               or request.query_params.get("token") \
               or request.query_params.get("t") \
               or (path_token or "") \
               or ""
    return (not shared) or (incoming == shared)


def _resolve_source(request) -> str | None:
    """优先 body/query 的 source；否则用 sitemap_name/custom_id → WEB_SCRAPER_SOURCE_MAP"""
    source_name = request.query_params.get("source")
    if not source_name and isinstance(request.data, dict):
        source_name = request.data.get("source")
    if source_name:
        return source_name
    # 映射
    sitemap_name = (request.data.get("sitemap_name") or request.data.get("sitemap") or "") if isinstance(
        request.data, dict) else ""
    custom_id = (request.data.get("custom_id") or "") if isinstance(request.data, dict) else ""
    mp = getattr(settings, "WEB_SCRAPER_SOURCE_MAP", {})
    return mp.get(sitemap_name) or mp.get(custom_id)


def _as_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

def _is_nan_like(v):
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"", "na", "nan", "null", "none", "undefined"}

def _parse_recorded_at(val):
    if not val:
        return timezone.now()
    dt = parse_datetime(val)
    if dt:
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_current_timezone())
        return dt
    d = parse_date(val)
    if isinstance(d, date):
        dt = datetime(d.year, d.month, d.day)
        return timezone.make_aware(dt, timezone.get_current_timezone())
    return timezone.now()

def _extract_source_name(filename: str) -> str | None:
    """
    从文件名提取清洗器名：
      shop2.xlsx -> shop2
      shop_foo.csv -> shop_foo
    允许的后缀：xlsx/xlsm/xls/ods/xlsb/csv
    """
    import re, os
    base = os.path.basename(filename or "")
    m = re.match(r"^([A-Za-z0-9_\-]+)\.(xlsx|xlsm|xls|ods|xlsb|csv)$", base, flags=re.IGNORECASE)
    return m.group(1) if m else None

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
    search_fields = ["part_number", "jan", "model_name", "color"]
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
        qs = super().get_queryset()
        qp = self.request.query_params
        if qp.get("jan"):
            jan = re.sub(r"\D", "", qp.get("jan"))
            qs = qs.filter(jan=jan)
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

    # ==================== CSV 模板（GET） ====================
    @extend_schema(
        tags=["Apple / iPhone"], summary="下载 iPhone CSV 模板（仅表头）",
        auth=[], responses={200: OpenApiTypes.BINARY}
    )
    @action(detail=False, methods=["get"], url_path="csv-template", permission_classes=[permissions.AllowAny])
    def csv_template(self, request):
        header = "part_number,model_name,capacity_gb,color,release_date,jan\n"
        resp = HttpResponse(header, content_type="text/csv; charset=utf-8")
        resp["Content-Disposition"] = 'attachment; filename="iphone_template.csv"'
        return resp

    # ==================== CSV 导入（POST） ====================
    @extend_schema(
        tags=["Apple / iPhone"],
        summary="导入 iPhone（CSV 批量录入）",
        description=(
                "必需列：part_number, model_name, capacity_gb, color, release_date\n"
                "参数：update=1 允许按 part_number 更新已存在的记录；dry_run=1 仅校验不落库。\n"
                "编码自动尝试 UTF-8/UTF-8-BOM/CP932/Shift-JIS；分隔符自动尝试 , / \\t / ; / |。"
        ),
        parameters=[
            OpenApiParameter("update", OpenApiTypes.BOOL, description="存在则更新（默认 true）", required=False),
            OpenApiParameter("dry_run", OpenApiTypes.BOOL, description="仅校验不落库（默认 false）", required=False),
        ],
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "format": "binary", "description": "包含表头的 CSV 文件"},
                },
                "required": ["file"],
            }
        },
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT},
    )
    @action(
        detail=False, methods=["post"], url_path="import-csv",
        parser_classes=[MultiPartParser, FormParser],
        authentication_classes=[JWTAuthentication],
        permission_classes=[IsAuthenticated],
    )
    def import_csv(self, request):
        from .models import Iphone  # 避免循环导入

        f = request.FILES.get("file")
        if not f:
            return Response({"detail": "缺少文件字段 file"}, status=status.HTTP_400_BAD_REQUEST)

        def as_bool(val, default=False):
            if val is None: return default
            return str(val).strip().lower() in {"1", "true", "t", "yes", "y"}

        allow_update = as_bool(request.query_params.get("update"), True)
        dry_run = as_bool(request.query_params.get("dry_run"), False)

        # 读取 CSV：多编码/多分隔符尝试
        raw = f.read()
        if hasattr(f, "seek"):
            try:
                f.seek(0)
            except Exception:
                pass

        text = None
        for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            return Response({"detail": "无法读取CSV：不支持的编码（推荐 UTF-8 或 Shift-JIS）"},
                            status=status.HTTP_400_BAD_REQUEST)

        reader = None
        for sep in (",", "\t", ";", "|"):
            try:
                reader = csv.DictReader(io.StringIO(text), delimiter=sep)
                # 简单判断：必须包含至少一个需要的列
                cols_lower = {(c or "").strip().lower() for c in reader.fieldnames or []}
                if {"part_number", "model_name", "capacity_gb", "color", "release_date"}.issubset(cols_lower):
                    break
            except Exception:
                reader = None
        if reader is None:
            # 最后再尝试默认逗号
            reader = csv.DictReader(io.StringIO(text))

        def norm_keys(d: dict) -> dict:
            return {(k or "").strip().lower(): v for k, v in d.items()}

        total = inserted = updated = skipped = 0
        errors = []
        preview = []

        for lineno, row in enumerate(reader, start=2):
            total += 1
            data = norm_keys(row)

            pn = (data.get("part_number") or data.get("pn") or "").strip()
            name = (data.get("model_name") or "").strip()
            cap = (data.get("capacity_gb") or "").strip()
            color = (data.get("color") or "").strip()
            rls = (data.get("release_date") or "").strip()
            jan_raw = (data.get("jan") or "").strip()
            jan = re.sub(r"\D", "", jan_raw) if jan_raw else None

            line_err = []
            if not pn:   line_err.append("缺少 part_number")
            if not name: line_err.append("缺少 model_name")
            if jan and len(jan) != 13:
                line_err.append("jan 必须是 13 位数字")
            try:
                cap_int = int(re.sub(r"[^\d]", "", str(cap)))
            except Exception:
                cap_int = None
            if not cap_int: line_err.append("capacity_gb 非法")
            # 允许 2024/09/20 → 2024-09-20
            rls_norm = rls.replace("/", "-")
            rdate = parse_date(rls_norm)
            if not rdate: line_err.append("release_date 非法（YYYY-MM-DD）")

            if line_err:
                errors.append({"line": lineno, "errors": line_err, "row": row})
                skipped += 1
                continue

            if len(preview) < 5:
                preview.append({
                    "part_number": pn, "model_name": name, "capacity_gb": cap_int,
                    "color": color, "release_date": rdate.isoformat(),
                })

            if dry_run:
                continue

            try:
                with transaction.atomic():
                    obj = Iphone.objects.filter(part_number=pn).first()
                    if obj:
                        if allow_update:
                            changed = False
                            new_jan = jan or None
                            if obj.jan != new_jan:
                                obj.jan = new_jan;
                                changed = True
                            if obj.model_name != name: obj.model_name = name; changed = True
                            if obj.capacity_gb != cap_int: obj.capacity_gb = cap_int; changed = True
                            if obj.color != color: obj.color = color; changed = True
                            if obj.release_date != rdate: obj.release_date = rdate; changed = True
                            if changed:
                                obj.save(update_fields=["model_name", "capacity_gb", "color", "release_date", "jan"])
                                updated += 1
                            else:
                                skipped += 1
                        else:
                            skipped += 1
                    else:
                        Iphone.objects.create(
                            part_number=pn, model_name=name, capacity_gb=cap_int, jan=(jan or None),
                            color=color, release_date=rdate
                        )
                        inserted += 1
            except IntegrityError as e:
                errors.append({"line": lineno, "errors": [f"数据库约束错误：{str(e)}"], "row": row})
                skipped += 1

        return Response({
            "headers": ["part_number", "model_name", "capacity_gb", "color", "release_date"],
            "rows_total": total, "inserted": inserted, "updated": updated, "skipped": skipped,
            "errors_count": len(errors), "errors": errors[:50], "preview": preview,
            "options": {"update": allow_update, "dry_run": dry_run},
        }, status=status.HTTP_200_OK)


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
            OpenApiParameter("iphone_part_number", OpenApiTypes.STR, description="按 iPhone Part Number 精确匹配",
                             required=False),
            OpenApiParameter("has_stock", OpenApiTypes.BOOL, description="是否有库存（true/false）", required=False),
            OpenApiParameter("recorded_after", OpenApiTypes.DATETIME, description="记录时间不早于(ISO8601)",
                             required=False),
            OpenApiParameter("recorded_before", OpenApiTypes.DATETIME, description="记录时间不晚于(ISO8601)",
                             required=False),
            OpenApiParameter("arrival_earliest_after", OpenApiTypes.DATETIME, description="最早到达不早于(ISO8601)",
                             required=False),
            OpenApiParameter("arrival_earliest_before", OpenApiTypes.DATETIME, description="最早到达不晚于(ISO8601)",
                             required=False),
            OpenApiParameter("arrival_latest_after", OpenApiTypes.DATETIME, description="最晚到达不早于(ISO8601)",
                             required=False),
            OpenApiParameter("arrival_latest_before", OpenApiTypes.DATETIME, description="最晚到达不晚于(ISO8601)",
                             required=False),
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

    @extend_schema(
        tags=["Apple / Inventory"],
        summary="送达天数趋势（按 part_number → 门店，日聚合）",
        description=(
                "以唯一编码 part_number 为一级分类，门店为二级分类。"
                "同一门店同一天：最早天数取最小、最晚天数取最大；中位数 = round((最早+最晚)/2)。"
                "天数 = ceil(送达日期 - 记录日期)，若无库存或缺少送达时间则记 0。"
        ),
        parameters=[
            OpenApiParameter("pn", OpenApiTypes.STR, description="必填：iPhone 唯一编码 Part Number（精确匹配）",
                             required=True),
            OpenApiParameter("recorded_after", OpenApiTypes.DATETIME, description="记录时间不早于（ISO8601）",
                             required=False),
            OpenApiParameter("recorded_before", OpenApiTypes.DATETIME, description="记录时间不晚于（ISO8601）",
                             required=False),
            OpenApiParameter("days", OpenApiTypes.INT,
                             description="最近 N 天（默认 14；若提供 recorded_after/recorded_before 则忽略）",
                             required=False),
            OpenApiParameter("store", OpenApiTypes.INT, description="可选：仅该门店 ID", required=False),
        ],
        responses=TrendResponseByPNSerializer,
    )
    @action(detail=False, methods=["GET"], url_path="trend")
    def trend(self, request):
        """
        GET /api/inventory-records/trend/?pn=MTUW3J%2FA&days=14
        返回：
        {
          "part_number": "MTUW3J/A",
          "iphone": {...},
          "recorded_after": "...", "recorded_before": null,
          "stores": [
            { "id": 1, "name": "...", "address": "...",
              "dates": ["2025-09-01", ...],
              "earliest": [0,2,...], "median": [0,2,...], "latest": [0,4,...]
            }
          ]
        }
        """
        qp = request.query_params
        pn = (qp.get("pn") or qp.get("part_number") or "").strip()
        if not pn:
            return Response({"detail": "缺少参数 pn（part_number）"}, status=status.HTTP_400_BAD_REQUEST)

        # 时间范围
        recorded_after = parse_datetime(qp.get("recorded_after") or "") or None
        recorded_before = parse_datetime(qp.get("recorded_before") or "") or None
        if not recorded_after and not recorded_before:
            try:
                days = max(1, int(qp.get("days", 14)))
            except ValueError:
                days = 14
            recorded_after = timezone.now() - timedelta(days=days)

        qs = (
            InventoryRecord.objects.select_related("store", "iphone")
            .filter(iphone__part_number=pn)
            .order_by("recorded_at")
        )
        store_id = qp.get("store")
        if store_id and store_id.isdigit():
            qs = qs.filter(store_id=int(store_id))
        if recorded_after:
            qs = qs.filter(recorded_at__gte=recorded_after)
        if recorded_before:
            qs = qs.filter(recorded_at__lte=recorded_before)

        rows = qs.values(
            "store_id",
            "store__name",
            "store__address",
            "recorded_at",
            "has_stock",
            "estimated_arrival_earliest",
            "estimated_arrival_latest",
        )

        # store -> date -> {e, l}
        by_store: dict[int, dict] = {}

        def to_days(rec):
            ra = rec["recorded_at"]
            if not rec["has_stock"]:
                return 0, 0

            def diff_days(target):
                if not target:
                    return 0
                d = ceil((target - ra).total_seconds() / 86400.0)
                return d if d > 0 else 0

            e = diff_days(rec["estimated_arrival_earliest"])
            l = diff_days(rec["estimated_arrival_latest"])
            return e, l

        for r in rows:
            sid = r["store_id"]
            if sid not in by_store:
                by_store[sid] = {
                    "store": {"id": sid, "name": r["store__name"], "address": r["store__address"]},
                    "dates": {},
                }
            key = r["recorded_at"].date().isoformat()
            e_days, l_days = to_days(r)
            slot = by_store[sid]["dates"].get(key)
            if slot is None:
                by_store[sid]["dates"][key] = {"e": e_days, "l": l_days}
            else:
                slot["e"] = min(slot["e"], e_days)
                slot["l"] = max(slot["l"], l_days)

        stores_out = []
        for sid, payload in by_store.items():
            dmap = payload["dates"]
            dates = sorted(dmap.keys())
            e_list, m_list, l_list = [], [], []
            for d in dates:
                e = int(dmap[d]["e"] or 0)
                l = int(dmap[d]["l"] or 0)
                m = int(round((e + l) / 2.0))
                e_list.append(e)
                l_list.append(l)
                m_list.append(m)
            stores_out.append({
                **payload["store"],
                "dates": dates,
                "earliest": e_list,
                "median": m_list,
                "latest": l_list,
            })
        stores_out.sort(key=lambda x: x["name"] or "")

        # 附带 iPhone 基本信息（若存在）
        iphone_info = None
        ip = Iphone.objects.filter(part_number=pn).values(
            "part_number", "model_name", "capacity_gb", "color", "release_date"
        ).first()
        if ip:
            iphone_info = {
                **ip,
                "capacity_label": (
                    f"{ip['capacity_gb'] // 1024}TB" if ip["capacity_gb"] % 1024 == 0 else f"{ip['capacity_gb']}GB")
            }

        data = {
            "part_number": pn,
            "iphone": iphone_info,
            "recorded_after": recorded_after,
            "recorded_before": recorded_before,
            "stores": stores_out,
        }
        # 用序列化器规范输出
        ser = TrendResponseByPNSerializer(data)
        return Response(ser.data)


@extend_schema_view(
    list=extend_schema(tags=["Resale / Shop"], summary="二手店列表", auth=[]),
    retrieve=extend_schema(tags=["Resale / Shop"], summary="二手店详情", auth=[]),
    create=extend_schema(tags=["Resale / Shop"], summary="创建二手店"),
    update=extend_schema(tags=["Resale / Shop"], summary="更新二手店（整体）"),
    partial_update=extend_schema(tags=["Resale / Shop"], summary="更新二手店（部分）"),
    destroy=extend_schema(tags=["Resale / Shop"], summary="删除二手店"),
)
class SecondHandShopViewSet(viewsets.ModelViewSet):
    queryset = SecondHandShop.objects.all()
    serializer_class = SecondHandShopSerializer

    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "address", "website"]
    ordering_fields = ["name"]
    ordering = ["name"]

    def get_permissions(self):
        if self.action in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]


# —— 回收价格记录 —— #
@extend_schema_view(
    list=extend_schema(
        tags=["Resale / Price"],
        summary="回收价格记录列表（支持过滤/搜索/排序）",
        # auth=[],
        parameters=[
            OpenApiParameter("shop", OpenApiTypes.INT, description="按二手店ID", required=False),
            OpenApiParameter("shop_name", OpenApiTypes.STR, description="按二手店名（模糊）", required=False),
            OpenApiParameter("iphone", OpenApiTypes.INT, description="按 iPhone ID", required=False),
            OpenApiParameter("iphone_part_number", OpenApiTypes.STR, description="按 iPhone PN 精确匹配",
                             required=False),
            OpenApiParameter("recorded_after", OpenApiTypes.DATETIME, description="记录时间不早于(ISO8601)",
                             required=False),
            OpenApiParameter("recorded_before", OpenApiTypes.DATETIME, description="记录时间不晚于(ISO8601)",
                             required=False),
            OpenApiParameter("min_price_new", OpenApiTypes.INT, description="新品卖取价格 ≥", required=False),
            OpenApiParameter("max_price_new", OpenApiTypes.INT, description="新品卖取价格 ≤", required=False),
            OpenApiParameter("search", OpenApiTypes.STR, description="在 店名/PN/型号/颜色 上搜索", required=False),
            OpenApiParameter(
                "ordering", OpenApiTypes.STR, required=False,
                enum=[
                    "recorded_at", "-recorded_at",
                    "price_new", "-price_new",
                    "price_grade_a", "-price_grade_a",
                    "price_grade_b", "-price_grade_b",
                ],
                description="排序字段（默认 -recorded_at）",
            ),
        ],
    ),
    retrieve=extend_schema(tags=["Resale / Price"], summary="回收价格记录详情"),
    create=extend_schema(tags=["Resale / Price"], summary="新增回收价格记录"),
    update=extend_schema(tags=["Resale / Price"], summary="更新回收价格记录（整体）"),
    partial_update=extend_schema(tags=["Resale / Price"], summary="更新回收价格记录（部分）"),
    destroy=extend_schema(tags=["Resale / Price"], summary="删除回收价格记录"),
)
class PurchasingShopPriceRecordViewSet(viewsets.ModelViewSet):
    queryset = PurchasingShopPriceRecord.objects.select_related("shop", "iphone").all()
    serializer_class = PurchasingShopPriceRecordSerializer

    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["shop__name", "iphone__part_number", "iphone__model_name", "iphone__color"]
    ordering_fields = ["recorded_at", "price_new", "price_grade_a", "price_grade_b"]
    ordering = ["-recorded_at"]

    def get_permissions(self):
        # 优先使用装饰器上的 permission_classes（如果某 action 单独声明了）
        if hasattr(getattr(self, self.action, None), 'permission_classes'):
            return [perm() for perm in getattr(self, self.action).permission_classes]
        # 仅放行 Webhook & 任务结果查询（我们用共享密钥校验 token）
        if self.action in ["ingest_webscraper", "ingest_webscraper_with_path_token", "ingest_webscraper_result"]:
            return [AllowAny()]
        # 其余动作（list/retrieve/import_csv/import_tradein/ingest-external/...）全部需要登录
        return [IsAuthenticated()]

    def get_queryset(self):
        qs = super().get_queryset()
        qp = self.request.query_params

        shop = qp.get("shop")
        shop_name = qp.get("shop_name")
        iphone = qp.get("iphone")
        iphone_pn = qp.get("iphone_part_number")

        recorded_after = parse_datetime(qp.get("recorded_after") or "")
        recorded_before = parse_datetime(qp.get("recorded_before") or "")
        min_price_new = qp.get("min_price_new")
        max_price_new = qp.get("max_price_new")

        if shop and shop.isdigit():
            qs = qs.filter(shop_id=int(shop))
        if shop_name:
            qs = qs.filter(shop__name__icontains=shop_name)
        if iphone and iphone.isdigit():
            qs = qs.filter(iphone_id=int(iphone))
        if iphone_pn:
            qs = qs.filter(iphone__part_number=iphone_pn)

        if recorded_after:
            qs = qs.filter(recorded_at__gte=recorded_after)
        if recorded_before:
            qs = qs.filter(recorded_at__lte=recorded_before)

        if min_price_new and min_price_new.isdigit():
            qs = qs.filter(price_new__gte=int(min_price_new))
        if max_price_new and max_price_new.isdigit():
            qs = qs.filter(price_new__lte=int(max_price_new))

        return qs

    def _to_int_yen(val):
        """
        将 '¥105,000' / '105000' / '105,000.0' / '' / None → int 或 None
        规则：
          - 去掉货币符号与逗号、空格
          - 有小数点则取整数部分
          - 空字符串 / 无数字 → None
        """
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        # 去除常见符号
        s = s.replace("¥", "").replace(",", "").replace("円", "").replace(" ", "")
        # 仅保留 0-9 和小数点
        s = re.sub(r"[^0-9.]", "", s)
        if not s:
            return None
        if "." in s:
            s = s.split(".", 1)[0] or "0"
        try:
            return int(s)
        except ValueError:
            return None

    def _parse_recorded_at(val):
        """
        解析 recorded_at：
          - 支持 ISO8601 字符串，如 '2025-09-06T10:00:00+09:00'
          - 支持 'YYYY-MM-DD'（视为本地时区 00:00）
          - 为空则返回 None（上层以 now() 填充）
        返回：timezone-aware datetime
        """
        if not val:
            return None
        s = str(val).strip()
        dt = parse_datetime(s)
        if dt is None:
            d = parse_date(s)
            if d is not None:
                dt = datetime(d.year, d.month, d.day)
        if dt is None:
            return None
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_current_timezone())
        return dt

    def _norm_key(d):
        """将 DictReader 的列名统一为小写去空格"""
        return {(k or "").strip().lower(): v for k, v in d.items()}


    @action(
        detail=False,
        methods=["post"],
        url_path="import-tradein-xlsx",
        parser_classes=[MultiPartParser, FormParser, FileUploadParser],
    )
    @authentication_classes([JWTAuthentication])  # ✅ 仅 JWT
    @permission_classes([IsAuthenticated])  # ✅ 需要 Bearer Token
    def import_tradein_xlsx(self, request):
        """
        POST /AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?dry_run=1&dedupe=1&upsert=0
        Header: Authorization: Bearer <access>
        Form: files=<shopX.xlsx>[, <shopY.xlsm>...]
        行为：每个文件各起一个 Celery 任务。
        """

        def _as_bool(v, default=False):
            return str(v).strip().lower() in {"1", "true", "t", "yes", "y"} if v is not None else default

        dry_run = _as_bool(request.query_params.get("dry_run"), False)
        dedupe = _as_bool(request.query_params.get("dedupe"), True)
        upsert = _as_bool(request.query_params.get("upsert"), False)

        # 批次
        import uuid
        bid = request.headers.get("X-Batch-Id") or request.query_params.get("batch_id")
        try:
            batch_uuid = uuid.UUID(str(bid)) if bid else uuid.uuid4()
        except Exception:
            batch_uuid = uuid.uuid4()

        # 兼容 files / file / 纯文件流
        files = request.FILES.getlist("files") or ([request.FILES["file"]] if request.FILES.get("file") else [])
        if not files:
            return Response({"detail": "请上传至少一个表格文件（Excel/CSV），字段名 files 或 file"},
                            status=status.HTTP_400_BAD_REQUEST)

        tasks = []
        for f in files:
            fname = getattr(f, "name", "")
            source_name = _extract_source_name(fname)
            if not source_name:
                return Response(
                    {"detail": f"无法从文件名提取清洗器名，或不支持的后缀：{fname}（仅支持 xlsx/xlsm/xls/ods/xlsb/csv）"},
                    status=status.HTTP_400_BAD_REQUEST)

            # 校验清洗器是否存在
            try:
                get_cleaner(source_name)
            except Exception:
                return Response({"detail": f"未知清洗器: {source_name}"}, status=status.HTTP_400_BAD_REQUEST)

            content = f.read()  # 交给任务自行解析（task_process_xlsx 已支持 csv）
            t = task_process_xlsx.delay(
                file_bytes=content,
                filename=fname,
                source_name=source_name,
                dry_run=dry_run,
                dedupe=dedupe,
                upsert=upsert,
                batch_id=str(batch_uuid),
            )
            tasks.append({"file": fname, "task_id": t.id, "source": source_name})

        return Response(
            {
                "accepted": True,
                "dry_run": dry_run,
                "dedupe": dedupe,
                "upsert": upsert,
                "batch_id": str(batch_uuid),
                "tasks": tasks,
            },
            status=status.HTTP_202_ACCEPTED,
        )


    @extend_schema(
            tags=["Resale / Price"],
            summary="导入二手店回收价格（CSV）",
            description=(
                    "上传 CSV 批量写入二手店回收价格记录。\n\n"
                    "必需列：`pn`(或 `part_number`)、`shop_name`、`price_new`。\n"
                    "可选列：`shop_address`、`shop_website`、`price_grade_a`、`price_grade_b`、`recorded_at`。\n"
                    "参数：`create_shop`=1 允许自动创建新店；`dedupe`=1 同店+PN+记录时间相同则更新而非新建；`dry_run`=1 仅校验不写库。"
            ),
            parameters=[
                OpenApiParameter("create_shop", OpenApiTypes.BOOL, description="若店铺不存在则创建（默认 true）",
                                 required=False),
                OpenApiParameter("dedupe", OpenApiTypes.BOOL, description="同店+PN+recorded_at 去重并更新（默认 true）",
                                 required=False),
                OpenApiParameter("dry_run", OpenApiTypes.BOOL, description="仅校验不落库（默认 false）", required=False),
            ],
            request={
                "multipart/form-data": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "format": "binary", "description": "CSV 文件"},
                        "create_shop": {"type": "boolean"},
                        "dedupe": {"type": "boolean"},
                        "dry_run": {"type": "boolean"},
                    },
                    "required": ["file"],
                }
            },
            responses={
                200: OpenApiTypes.OBJECT,
                400: OpenApiTypes.OBJECT,
            },
        )
    @action(
        detail=False,
        methods=["POST"],
        url_path="import-csv",
        parser_classes=[MultiPartParser, FormParser],
        permission_classes=[permissions.IsAdminUser],
    )
    def import_csv(self, request):
        # —— 参数解析 —— #
        def as_bool(v, default=False):
            if v is None:
                return default
            return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

        create_shop = as_bool(request.data.get("create_shop"), True)
        dedupe = as_bool(request.data.get("dedupe"), True)
        upsert = as_bool(request.data.get("upsert"), False)
        dry_run = as_bool(request.data.get("dry_run"), False)

        # 批次 ID：Header 优先，其次 Query/Body；非法则自动生成
        bid = request.headers.get("X-Batch-Id") or request.query_params.get("batch_id") or request.data.get("batch_id")
        try:
            batch_uuid = uuid.UUID(str(bid)) if bid else uuid.uuid4()
        except Exception:
            batch_uuid = uuid.uuid4()

        # —— 读取 CSV —— #
        f = request.FILES.get("file")
        if not f:
            return Response({"detail": "缺少文件字段 file"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            text = io.TextIOWrapper(f.file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
        except Exception as e:
            return Response({"detail": f"无法读取CSV: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        # —— 计数器 —— #
        total = inserted = updated = skipped = dedup_skipped = 0
        errors = []
        preview = []

        # —— 主循环 —— #
        for lineno, row in enumerate(reader, start=2):
            total += 1
            data = self._norm_key(row)

            # 映射列名
            pn = data.get("pn") or data.get("part_number") or data.get("iphone_pn")
            shop_name = data.get("shop_name") or data.get("shop")
            shop_addr = data.get("shop_address") or data.get("address") or ""
            shop_site = data.get("shop_website") or data.get("website") or ""

            price_new = self._to_int_yen(data.get("price_new"))
            price_a = self._to_int_yen(data.get("price_grade_a") or data.get("a") or data.get("grade_a"))
            price_b = self._to_int_yen(data.get("price_grade_b") or data.get("b") or data.get("grade_b"))

            rec_at = self._parse_recorded_at(data.get("recorded_at") or data.get("time") or data.get("date"))
            if rec_at is None:
                rec_at = timezone.now()

            # 基础校验
            line_errors = []
            if not pn:
                line_errors.append("缺少 pn/part_number")
            if not shop_name:
                line_errors.append("缺少 shop_name")
            if price_new is None or price_new <= 0:
                line_errors.append("price_new 非法或缺失")
            if line_errors:
                errors.append({"line": lineno, "errors": line_errors, "row": row})
                skipped += 1
                continue

            # 匹配 iPhone（只用 PN）
            iphone = Iphone.objects.filter(part_number=str(pn).strip()).first()
            if not iphone:
                errors.append({"line": lineno, "errors": [f"未找到 iPhone PN: {pn}"], "row": row})
                skipped += 1
                continue

            # 匹配/创建 shop（按 name+address）
            shop = SecondHandShop.objects.filter(name=shop_name.strip(), address=shop_addr.strip()).first()
            if not shop:
                if create_shop and not dry_run:
                    shop = SecondHandShop.objects.create(
                        name=shop_name.strip(), address=shop_addr.strip(), website=shop_site.strip()
                    )
                elif create_shop and dry_run:
                    shop = None  # 仅模拟
                else:
                    errors.append({"line": lineno, "errors": [f"未找到店铺: {shop_name} / {shop_addr}"], "row": row})
                    skipped += 1
                    continue

            # 预览（最多 5 条）
            if len(preview) < 5:
                preview.append({
                    "pn": str(pn).strip(),
                    "shop_name": shop_name.strip(),
                    "shop_address": shop_addr.strip(),
                    "price_new": price_new,
                    "price_grade_a": price_a,
                    "price_grade_b": price_b,
                    "recorded_at": rec_at.isoformat(),
                    "batch_id": str(batch_uuid),
                })

            if dry_run:
                # 仅校验，不落库
                continue

            # —— 幂等/更新/新建 —— #
            with transaction.atomic():
                existed = None
                if dedupe and shop is not None:
                    existed = PurchasingShopPriceRecord.objects.filter(
                        shop=shop, iphone=iphone, recorded_at=rec_at
                    ).first()

                if existed:
                    if upsert:
                        changed = False
                        if existed.price_new != price_new:
                            existed.price_new = price_new;
                            changed = True
                        if price_a is not None and existed.price_grade_a != price_a:
                            existed.price_grade_a = price_a;
                            changed = True
                        if price_b is not None and existed.price_grade_b != price_b:
                            existed.price_grade_b = price_b;
                            changed = True
                        if changed:
                            existed.batch_id = batch_uuid
                            existed.save(update_fields=["price_new", "price_grade_a", "price_grade_b", "batch_id"])
                            updated += 1
                        else:
                            dedup_skipped += 1
                    else:
                        dedup_skipped += 1
                else:
                    if shop is None:
                        errors.append({"line": lineno, "errors": ["内部错误：shop 为空"], "row": row})
                        skipped += 1
                        continue
                    rec = PurchasingShopPriceRecord.objects.create(
                        shop=shop, iphone=iphone,
                        price_new=price_new,
                        price_grade_a=price_a, price_grade_b=price_b,
                        batch_id=batch_uuid,
                    )
                    PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=rec_at)
                    inserted += 1

        resp = {
            "rows_total": total,
            "inserted": inserted,
            "updated": updated,
            "dedup_skipped": dedup_skipped,  # 幂等跳过的行数
            "skipped": skipped,  # 其它原因（校验失败/店铺缺失等）
            "errors_count": len(errors),
            "errors": errors[:50],  # 防止回包过大
            "preview": preview,  # 前 5 条预览
            "options": {
                "create_shop": create_shop,
                "dedupe": dedupe,
                "upsert": upsert,
                "dry_run": dry_run,
                "batch_id": str(batch_uuid),
            },
        }
        return Response(resp, status=status.HTTP_200_OK)



    # @csrf_exempt
    # @authentication_classes([JWTAuthentication])             # ← 用 JWT，不触发 CSRF
    # @permission_classes([permissions.IsAdminUser])

    @extend_schema(  # ← 你的原注解可保留
        tags=["Resale / Price"],
        summary="导入二手店回收价（清洗逻辑已抽到 utils）——每次上传新增记录",
        parameters=[
            OpenApiParameter("create_shop", OpenApiTypes.BOOL, description="店铺不存在时自动创建（默认 true）",
                             required=False),
            OpenApiParameter("dry_run", OpenApiTypes.BOOL, description="只清洗/聚合不落库（默认 false）", required=False),
            OpenApiParameter("recorded_at", OpenApiTypes.DATETIME, description="统一记录时间（ISO8601/日期），默认现在",
                             required=False),
            OpenApiParameter("dedupe", OpenApiTypes.BOOL,
                             description="同一 (shop, iphone, recorded_at) 去重（默认 true）", required=False),
            OpenApiParameter("upsert", OpenApiTypes.BOOL, description="去重命中时是否覆盖（默认 false）", required=False),
        ],
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
                    "file": {"type": "string", "format": "binary"}
                },
            },
            "application/json": {
                "type": "object",
                "properties": {
                    "blobs": {  # 可选：支持 JSON 直接传 {name, content(base64 or text)}
                        "type": "array",
                        "items": {"type": "object",
                                  "properties": {"name": {"type": "string"}, "content": {"type": "string"}}}
                    }
                }
            }
        }
    )
    @method_decorator(csrf_exempt)
    @action(
        detail=False, methods=["post"], url_path="import-tradein",
        parser_classes=[MultiPartParser, FormParser, JSONParser],
    )
    @authentication_classes([JWTAuthentication])  # ✅ 仅 JWT 认证；若需要匿名上传可改为 AllowAny
    @permission_classes([IsAuthenticated])
    def import_tradein(self, request):
        # ---- 读取 query/header 选项 ----
        bid = request.query_params.get("batch_id") or request.headers.get("X-Batch-Id")
        try:
            batch_uuid = uuid.UUID(str(bid)) if bid else uuid4()
        except Exception:
            batch_uuid = uuid4()

        dedupe = _as_bool(request.query_params.get("dedupe"), True)
        upsert = _as_bool(request.query_params.get("upsert"), False)
        create_shop = _as_bool(request.query_params.get("create_shop"), True)
        dry_run = _as_bool(request.query_params.get("dry_run"), False)
        recorded_at = _parse_recorded_at(request.query_params.get("recorded_at"))

        # ---- 读取文件 / JSON blobs ----
        files = request.FILES.getlist("files") or ([request.FILES["file"]] if request.FILES.get("file") else [])
        blobs = []

        if files:
            for f in files:
                data = f.read()
                blobs.append((data, getattr(f, "name", "upload.csv")))
                try:
                    if hasattr(f, "seek"):
                        f.seek(0)
                except Exception:
                    pass
        else:
            # JSON body 支持：{"blobs":[{"name":"a.csv","content":"...原文或base64..."}]}
            body = request.data or {}
            jb = body.get("blobs") or []
            for b in jb:
                name = b.get("name") or "payload.csv"
                content = b.get("content") or ""
                # 这里假设 content 是纯文本 CSV；若是 base64，可在此处解码
                blobs.append((content.encode("utf-8"), name))

        if not blobs:
            return Response({"detail": "请上传至少一个 CSV（files/file 或 JSON.blobs）"},
                            status=status.HTTP_400_BAD_REQUEST)

        # ---- 清洗/聚合 ----
        result = clean_and_aggregate_tradein(blobs)
        rows_total = result.get("rows_total", 0)
        errors = result.get("errors", [])
        records = result.get("records", [])
        preview = result.get("preview", [])

        if rows_total == 0 and not records:
            return Response({"detail": "清洗失败或无有效数据", "errors": errors}, status=status.HTTP_400_BAD_REQUEST)

        # ---- 写库（始终新增；仅当 dedupe/upsert 指定时覆盖） ----
        inserted = 0
        updated = 0
        skipped_no_match = 0

        for rec in records:
            shop_name = (rec.get("shop_name") or "").strip()
            price_new = rec.get("price_new")
            price_a = rec.get("price_grade_a")
            price_b = rec.get("price_grade_b")
            meta = rec.get("meta") or {}

            pn = (meta.get("pn") or "").strip()
            jan_digits = (meta.get("jan") or "").strip()
            model_name = (meta.get("model_name") or "").strip()
            capacity_gb = meta.get("capacity_gb")
            color_raw = (meta.get("color_raw") or "").strip()
            color_canon = (meta.get("color_canon") or "").strip()
            color_any = bool(meta.get("color_any"))

            # iPhone 匹配：PN → JAN → 型号+容量(+颜色/全色)
            iphones = []
            if pn:
                ip = Iphone.objects.filter(part_number=pn).first()
                if ip:
                    iphones = [ip]
            if not iphones and len(jan_digits) == 13:
                ip = Iphone.objects.filter(jan=jan_digits).first()
                if ip:
                    iphones = [ip]
            if not iphones and model_name and capacity_gb:
                base_qs = Iphone.objects.filter(model_name__iexact=model_name, capacity_gb=capacity_gb)
                if color_any:
                    iphones = list(base_qs)
                else:
                    if color_canon:
                        q = Q()
                        for s in synonyms_for_query(color_canon):
                            q |= Q(color__iexact=s) | Q(color__icontains=s)
                        iphones = list(base_qs.filter(q).distinct())
                    else:
                        iphones = list(base_qs.filter(color__icontains=color_raw))

            if not iphones:
                skipped_no_match += 1
                continue

            # 店铺：name+address 唯一（地址未知用空字符串）
            shop = SecondHandShop.objects.filter(name=shop_name, address="").first()
            if not shop and create_shop:
                shop = SecondHandShop.objects.create(name=shop_name, address="", website="")
            if not shop:
                skipped_no_match += 1
                continue

            if dry_run:
                inserted += len(iphones)
                continue

            for iphone in iphones:
                with transaction.atomic():
                    existed = None
                    if dedupe:
                        existed = PurchasingShopPriceRecord.objects.filter(
                            shop=shop, iphone=iphone, recorded_at=recorded_at
                        ).first()

                    if existed:
                        if upsert:
                            changed = False

                            # 归一化数值
                            def _int_or_none(v):
                                if _is_nan_like(v):
                                    return None
                                try:
                                    return int(v)
                                except Exception:
                                    return None

                            nv = _int_or_none(price_new)
                            av = _int_or_none(price_a)
                            bv = _int_or_none(price_b)

                            if nv is not None and existed.price_new != nv:
                                existed.price_new = nv;
                                changed = True
                            if (existed.price_grade_a or None) != av:
                                existed.price_grade_a = av;
                                changed = True
                            if (existed.price_grade_b or None) != bv:
                                existed.price_grade_b = bv;
                                changed = True

                            if changed:
                                existed.batch_id = batch_uuid
                                existed.save(update_fields=["price_new", "price_grade_a", "price_grade_b", "batch_id"])
                                updated += 1
                            else:
                                skipped_no_match += 1
                        else:
                            skipped_no_match += 1
                        continue

                    # 始终新增
                    PurchasingShopPriceRecord.objects.create(
                        shop=shop, iphone=iphone,
                        price_new=0 if _is_nan_like(price_new) else int(price_new),
                        price_grade_a=None if _is_nan_like(price_a) else int(price_a),
                        price_grade_b=None if _is_nan_like(price_b) else int(price_b),
                        batch_id=batch_uuid,
                        recorded_at=recorded_at,
                    )
                    inserted += 1

        return Response({
            "rows_total": rows_total,
            "aggregated": len(records),
            "inserted": inserted,
            "updated": updated,
            "skipped_no_match": skipped_no_match,
            "preview": preview[:10],  # 前端表格预览只需前 10
            "errors": errors[:50],
            "options": {
                "create_shop": create_shop,
                "dry_run": dry_run,
                "recorded_at": recorded_at.isoformat(),
                "dedupe": dedupe,
                "upsert": upsert,
            },
        }, status=status.HTTP_200_OK)






    @extend_schema(
        tags=["Resale / Price"],
        summary="从外部平台 GET 一组 CSV → 清洗 → 以 PN 定位 iPhone → 新增回收价格记录",
        description=(
                "请求体传入 sources 列表（每个包含 name/url/headers），或使用 settings 中预置的 EXTERNAL_TRADEIN_SOURCES 并用 names 指定子集。\n"
                "清洗规则在 utils.external_ingest.base_cleaners 中按 name 注册（独立脚本框架）。\n"
                "支持 ?dry_run=1 只预览不落库。"
        ),
        request=OpenApiTypes.OBJECT,
        parameters=[
            OpenApiParameter("dry_run", OpenApiTypes.BOOL, required=False, description="1=仅预览"),
        ],
        responses={200: OpenApiTypes.OBJECT},
    )
    @action(detail=False, methods=["post"], url_path="ingest-external", permission_classes=[permissions.IsAdminUser])
    def ingest_external(self, request):

        dry_run = str(request.query_params.get("dry_run") or "").lower() in {"1", "true", "t", "yes", "y"}

        payload = request.data or {}
        sources = payload.get("sources") or []
        names = payload.get("names") or []

        # 允许从 settings 读取预置
        from django.conf import settings
        preset = getattr(settings, "EXTERNAL_TRADEIN_SOURCES", [])
        # EXTERNAL_TRADEIN_SOURCES 形式举例： [{"name":"sample_a","url":"https://...","headers":{"Authorization":"..."}}]

        if not sources and names:
            # 从 preset 里筛选
            by_name = {it["name"]: it for it in preset}
            sources = [by_name[n] for n in names if n in by_name]

        if not sources and not names:
            # 默认用预置全部
            sources = preset

        if not sources:
            return Response({"detail": "未提供 sources，且 settings.EXTERNAL_TRADEIN_SOURCES 为空"},
                            status=status.HTTP_400_BAD_REQUEST)

        result = ingest_external_sources(sources, dry_run=dry_run)
        return Response({
            "dry_run": dry_run,
            **result
        })





    #
    # @extend_schema(
    #     tags=["Resale / Price"],
    #     summary="WebScraper Webhook/直传（非 Celery，同步清洗与落库）",
    #     description=(
    #             "两种用法：\n"
    #             "1) Webhook 通知：POST 里带 scrapingjob_id/job_id；我们同步用 API 拉取导出 → 清洗器 → 落库；\n"
    #             "   - URL 可用 ?t=<token>（短参），或 Header: X-Webhook-Token: <token>，或 Path Token（见下方变体）\n"
    #             "   - source：优先 body/query；否则 sitemap_name/custom_id 走 settings.WEB_SCRAPER_SOURCE_MAP 映射\n"
    #             "2) 直传数据：CSV/JSON 正文 + source=shopX，同步清洗并落库（用于调试或其它来源直接推送）。\n"
    #             "通用：?dry_run=1 仅预览不落库。"
    #     ),
    #     parameters=[
    #         OpenApiParameter("dry_run", OpenApiTypes.BOOL, required=False),
    #         OpenApiParameter("t", OpenApiTypes.STR, required=False, description="短 token（可替代 token）"),
    #         OpenApiParameter("source", OpenApiTypes.STR, required=False),
    #     ],
    #     request=OpenApiTypes.BYTE,
    #     responses={200: OpenApiTypes.OBJECT}
    # )
    # @action(
    #     detail=False, methods=["post"],
    #     url_path="ingest-webscraper",
    #     permission_classes=[AllowAny],  # 外部回调需匿名可达，用 token 校验
    # )
    # @parser_classes([JSONParser, FormParser, MultiPartParser, FileUploadParser, PlainTextParser, TextCsvParser])
    # def ingest_webscraper_csv(self, request):
    #     dry_run = str(request.query_params.get("dry_run") or "").lower() in {"1", "true", "t", "yes", "y"}
    #     if not _check_token(request, path_token=None):
    #         return Response({"detail": "Webhook token 不匹配"}, status=status.HTTP_403_FORBIDDEN)
    #
    #     ct = (request.content_type or "").lower()
    #     is_direct = ("csv" in ct) or ("json" in ct) or ct.startswith("text/plain") or ct.startswith(
    #         "multipart/form-data")
    #     # A) 直传 CSV/JSON → 同步处理
    #     if ("csv" in ct) or ("json" in ct) or ct.startswith("text/plain"):
    #         source_name = _resolve_source(request)
    #         if not source_name:
    #             return Response({"detail": "直传数据必须提供 source（或通过 sitemap/custom_id 映射）"},
    #                             status=status.HTTP_400_BAD_REQUEST)
    #         # 校验清洗器存在
    #         try:
    #             get_cleaner(source_name)
    #         except Exception:
    #             return Response({"detail": f"未知清洗器: {source_name}"}, status=status.HTTP_400_BAD_REQUEST)
    #
    #         try:
    #             df = to_dataframe_from_request(request.content_type, request.body or b"")
    #         except Exception as e:
    #             return Response({"detail": f"载入数据失败: {e}"}, status=status.HTTP_400_BAD_REQUEST)
    #
    #         result = ingest_external_dataframe(source_name, df, dry_run=dry_run, pn_only=True, create_shop=True)
    #         return Response({"mode": "direct", "dry_run": dry_run, "source": source_name, **result},
    #                         status=status.HTTP_200_OK)
    #
    #     # B) Webhook 通知 → job_id + source → 同步拉取导出并处理
    #
    #     job_id = request.data.get("scrapingjob_id") or request.data.get("job_id")
    #     source_name = _resolve_source(request)
    #     if not job_id or not source_name:
    #         return Response({"detail": "Webhook 需要 job_id(scrapingjob_id) 与 source（或提供映射）"},
    #                         status=status.HTTP_400_BAD_REQUEST)
    #
    #     try:
    #         get_cleaner(source_name)
    #     except Exception:
    #         return Response({"detail": f"未知清洗器: {source_name}"}, status=status.HTTP_400_BAD_REQUEST)
    #
    #     try:
    #         content = fetch_webscraper_export_sync(str(job_id), format="csv")
    #         df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    #     except Exception as e:
    #         return Response({"detail": f"拉取导出失败: {e}"}, status=status.HTTP_502_BAD_GATEWAY)
    #
    #     result = ingest_external_dataframe(source_name, df, dry_run=dry_run, pn_only=True, create_shop=True)
    #     return Response({"mode": "webhook", "dry_run": dry_run, "job_id": job_id, "source": source_name, **result},
    #                     status=status.HTTP_200_OK)
    #
    # # —— Path Token 版（更短的 URL）：/ingest-webscraper/<token>/ —— #
    # # 最短路径：/.../ingest-webscraper/<token>/
    # @action(detail=False, methods=["post"],
    #         url_path=r"ingest-webscraper/(?P<ptoken>[-A-Za-z0-9_]+)",
    #         permission_classes=[AllowAny])
    # @parser_classes([JSONParser, FormParser, MultiPartParser, FileUploadParser, PlainTextParser, TextCsvParser])
    # def ingest_webscraper_csv_with_path_token(self, request, ptoken=""):
    #     if not _check_token(request, path_token=ptoken):
    #         return Response({"detail": "Webhook token 不匹配"}, status=403)
    #     return self.ingest_webscraper_csv(request)  # 复用主体逻辑

    #
    # —— Webhook/直传入口（异步：Webhook 入队；直传：仍同步） —— #
    @extend_schema(
        tags=["Resale / Price"],
        summary="(Celery) WebScraper Webhook/直传：Webhook 入队，直传同步",
        parameters=[
            OpenApiParameter("dry_run", OpenApiTypes.BOOL, required=False),
            OpenApiParameter("t", OpenApiTypes.STR, required=False, description="短 token（可替代 token）"),
            OpenApiParameter("source", OpenApiTypes.STR, required=False),
        ],
        request=OpenApiTypes.BYTE,
        responses={202: OpenApiTypes.OBJECT, 200: OpenApiTypes.OBJECT}
    )
    @action(detail=False, methods=["post"], url_path="ingest-webscraper", permission_classes=[AllowAny])
    # @parser_classes([JSONParser, FormParser, MultiPartParser, FileUploadParser, PlainTextParser, TextCsvParser])
    def ingest_webscraper(self, request):
        dry_run = str(request.query_params.get("dry_run") or "").lower() in {"1", "true", "t", "yes", "y"}
        dedupe = _get_bool_param(request, "dedupe", True)
        upsert = _get_bool_param(request, "upsert", False)
        # batch_id: Header 优先，其次 query，再次 body；合法 uuid4，否则自动生成
        bid = request.headers.get("X-Batch-Id") or request.query_params.get("batch_id") or request.data.get("batch_id")
        try:
            batch_uuid = uuid.UUID(str(bid)) if bid else uuid.uuid4()
        except Exception:
            batch_uuid = uuid.uuid4()

        if not _check_token(request, path_token=None):
            return Response({"detail": "Webhook token 不匹配"}, status=status.HTTP_403_FORBIDDEN)

        # mode, body_bytes, eff_ct = _classify_mode(request)

        # 只在 finished 时处理（可选优化）

        ct = (request.content_type or "").lower()

        # if mode == "direct":
        #     source_name = _resolve_source(request)
        #     dedupe = _get_bool_param(request, "dedupe", True)
        #     upsert = _get_bool_param(request, "upsert", False)

        # A) 直传 CSV/JSON：同步处理（便于调试或第三方直推）
        if ("csv" in ct) or ("json" in ct) or ct.startswith("text/plain"):
            source_name = _resolve_source(request)
            if not source_name:
                return Response({"detail": "直传数据必须提供 source（或映射）"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                get_cleaner(source_name)
            except Exception:
                return Response({"detail": f"未知清洗器: {source_name}"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                df = to_dataframe_from_request(request.content_type, request.body or b"")
            except Exception as e:
                return Response({"detail": f"载入数据失败: {e}"}, status=status.HTTP_400_BAD_REQUEST)

            result = ingest_external_dataframe(source_name, df, dry_run=dry_run, pn_only=True, create_shop=True,
                                               dedupe=dedupe, upsert=upsert, batch_id=str(batch_uuid))
            return Response(
                {"mode": "direct", "dry_run": dry_run, "source": source_name, "batch_id": str(batch_uuid), **result},
                status=200)

        status_str = (request.data.get("status") or request.query_params.get("status") or "").lower()
        if status_str and status_str != "finished":
            return Response({"accepted": True, "reason": f"skip status={status_str}"}, status=status.HTTP_202_ACCEPTED)

        # B) Webhook：job_id + source → 入队 Celery，立即 202
        job_id = request.data.get("scrapingjob_id") or request.data.get("job_id") \
                 or request.query_params.get("scrapingjob_id") or request.query_params.get("job_id")
        source_name = _resolve_source(request) or request.query_params.get("source")
        if not job_id or not source_name:
            return Response({"detail": "Webhook 需要 job_id(scrapingjob_id) 与 source（或提供映射）"},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            get_cleaner(source_name)
        except Exception:
            return Response({"detail": f"未知清洗器: {source_name}"}, status=status.HTTP_400_BAD_REQUEST)

        task = task_process_webscraper_job.delay(str(job_id), source_name,
                                                 dry_run=dry_run, create_shop=True,
                                                 dedupe=dedupe, upsert=upsert, batch_id=str(batch_uuid))
        return Response({"mode": "webhook", "accepted": True, "task_id": task.id, "job_id": job_id,
                         "source": source_name, "dry_run": dry_run, "dedupe": dedupe, "upsert": upsert,
                         "batch_id": str(batch_uuid)}, status=202)









    # —— 极短 URL：/ingest-webscraper/<token>/ —— #
    @extend_schema(tags=["Resale / Price"], summary="Webhook（Path Token 版）")
    @action(detail=False, methods=["post"], url_path=r"ingest-webscraper/(?P<ptoken>[-A-Za-z0-9_]+)",
            permission_classes=[AllowAny])
    @parser_classes([JSONParser, FormParser, MultiPartParser, FileUploadParser, PlainTextParser, TextCsvParser])
    def ingest_webscraper_with_path_token(self, request, ptoken: str = ""):
        if not _check_token(request, path_token=ptoken):
            return Response({"detail": "Webhook token 不匹配"}, status=status.HTTP_403_FORBIDDEN)
        return self.ingest_webscraper(request)






    # —— 任务查询 —— #
    @extend_schema(tags=["Resale / Price"], summary="查询 Celery 任务结果",
                   parameters=[OpenApiParameter("task_id", OpenApiTypes.STR, required=True)])
    @action(detail=False, methods=["get"], url_path="ingest-webscraper/result", permission_classes=[AllowAny])
    def ingest_webscraper_result(self, request):
        task_id = request.query_params.get("task_id")
        if not task_id:
            return Response({"detail": "缺少 task_id"}, status=status.HTTP_400_BAD_REQUEST)
        res = AsyncResult(task_id)
        data = {"task_id": task_id, "state": res.state}
        if res.state == "SUCCESS":
            data["result"] = res.result
        elif res.state == "FAILURE":
            data["error"] = str(res.result)
        return Response(data, status=status.HTTP_200_OK)











    @extend_schema(
        tags=["Resale / Price"],
        summary="直传 JSON → 清洗器 shop1 → （预览或落库）",
        description=(
                "接收 JSON 正文，走清洗器 `shop1` 做清洗与写库。\n\n"
                "入参：application/json；正文可以是 **数组**（每个元素一行）或 **对象**（键值对集合）。\n"
                "参数：`?dry_run=1` 仅预览不落库；`?dedupe=1|0`；`?upsert=1|0`；Header: `X-Batch-Id`（可选 UUID）。\n"
                "安全：默认免登录，用 `_check_token` 校验。若只给管理员用，请把 AllowAny 改回 IsAdminUser 并按会话/CSRF 调用。"
        ),
        parameters=[
            OpenApiParameter("t", OpenApiTypes.STR, required=False, description="短 token（或 Header: X-Webhook-Token）"),
            OpenApiParameter("dry_run", OpenApiTypes.BOOL, required=False, description="1=仅预览"),
            OpenApiParameter("dedupe", OpenApiTypes.BOOL, required=False,
                             description="同店+PN+recorded_at 去重更新（默认 true）"),
            OpenApiParameter("upsert", OpenApiTypes.BOOL, required=False, description="按业务键 upsert（默认 false）"),
        ],
        request=OpenApiTypes.OBJECT,
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT, 403: OpenApiTypes.OBJECT}
    )
    @action(
        detail=False, methods=["post"], url_path="ingest-json",
    )
    @authentication_classes([])  # ★ 禁用 Session/JWT 认证 → 不触发 CSRF
    @permission_classes([AllowAny])  # ★ 允许匿名（配合我们自己的 token 校验）
    @parser_classes([JSONParser])  # 仅接收 application/json
    @method_decorator(csrf_exempt)  # ★ 再保险：对该视图禁用 CSRF
    def ingest_json(self, request):
        if not _check_token(request, path_token=None):
            return Response({"detail": "Webhook token 不匹配"}, status=status.HTTP_403_FORBIDDEN)

        dry_run = _get_bool_param(request, "dry_run", False)
        dedupe = _get_bool_param(request, "dedupe", True)
        upsert = _get_bool_param(request, "upsert", False)

        bid = request.headers.get("X-Batch-Id") \
              or request.query_params.get("batch_id") \
              or (request.data.get("batch_id") if isinstance(request.data, dict) else None)
        try:
            batch_uuid = uuid.UUID(str(bid)) if bid else uuid.uuid4()
        except Exception:
            batch_uuid = uuid.uuid4()

        # 解析 JSON -> records（避免 DataFrame 在 Celery 序列化时过大/复杂）
        try:
            payload = request.data
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = [payload]
            else:
                return Response({"detail": "JSON 结构不支持（应为数组或对象）"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"detail": f"解析 JSON 失败: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        # 校验清洗器是否存在
        try:
            get_cleaner("shop1")
        except Exception:
            return Response({"detail": "未知清洗器: shop1"}, status=status.HTTP_400_BAD_REQUEST)

        # 入队 Celery
        task = task_ingest_json_shop1.delay(
            records,  # list[dict] 原始 JSON
            {
                "dry_run": bool(dry_run),
                "dedupe": bool(dedupe),
                "upsert": bool(upsert),
                "batch_id": str(batch_uuid),
                "source": "shop1",
            }
        )
        return Response({"accepted": True, "task_id": task.id, "batch_id": str(batch_uuid)},
                        status=status.HTTP_202_ACCEPTED)








class PurchasingShopTimeAnalysisViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    queryset = (PurchasingShopTimeAnalysis.objects
                .select_related("shop", "iphone")
                .all())
    serializer_class = PurchasingShopTimeAnalysisSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]

    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_class = PurchasingShopTimeAnalysisFilter
    ordering_fields = [
        "Timestamp_Time",
        "Warehouse_Receipt_Time",
        "New_Product_Price",
        "Price_A",
        "Price_B",
        "Update_Count",
    ]
    ordering = ["-Timestamp_Time"]



class PurchasingShopTimeAnalysisPSTACompactViewSet(
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    queryset = (PurchasingShopTimeAnalysis.objects
                .select_related("shop", "iphone")
                .all())
    serializer_class = PSTACompactSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]

    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_class = PurchasingShopTimeAnalysisFilter
    ordering_fields = [
        "Timestamp_Time",
        "Warehouse_Receipt_Time",
        "New_Product_Price",
        "Price_A",
        "Price_B",
        "Update_Count",
    ]
    ordering = ["-Timestamp_Time"]