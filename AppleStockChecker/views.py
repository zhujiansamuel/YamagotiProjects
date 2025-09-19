from django.shortcuts import render

from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from django.utils.dateparse import parse_date
from collections import defaultdict
from django.db import transaction


from .models import Iphone, OfficialStore, InventoryRecord
from .serializers import OfficialStoreSerializer, InventoryRecordSerializer, IphoneSerializer
from .serializers import UserSerializer
from django.db.models import Q, F
from math import ceil
from datetime import timedelta
from django.utils import timezone

from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status

from .serializers import TrendResponseByPNSerializer
from .utils.color_norm import normalize_color, synonyms_for_query, is_all_color
from django.utils.dateparse import parse_datetime
from rest_framework import viewsets, permissions, filters
from drf_spectacular.utils import (
    extend_schema, extend_schema_view, OpenApiParameter, OpenApiTypes
)
from .models import SecondHandShop, PurchasingShopPriceRecord
from .serializers import SecondHandShopSerializer, PurchasingShopPriceRecordSerializer

import csv
import io
import re
from datetime import datetime
from math import ceil

from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date

from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status, permissions
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes

from .models import Iphone, SecondHandShop, PurchasingShopPriceRecord


import re, csv, io
from collections import defaultdict
from datetime import datetime
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date
from django.db import transaction
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import permissions, status
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes

from .utils.tradein_cleaner import parse_tradein_uploaded  # ← 新增导入
from .models import Iphone, SecondHandShop, PurchasingShopPriceRecord
import pandas as pd

import csv, io, re
from django.http import HttpResponse
from django.utils.dateparse import parse_date
from django.db import transaction, IntegrityError
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import permissions, status
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes

from datetime import datetime
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import permissions, status
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes

from .utils.tradein_pipeline import clean_and_aggregate_tradein
from .utils.color_norm import synonyms_for_query
from .models import Iphone, SecondHandShop, PurchasingShopPriceRecord




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
        permission_classes=[permissions.IsAdminUser],
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
                                obj.jan = new_jan; changed = True
                            if obj.model_name != name: obj.model_name = name; changed = True
                            if obj.capacity_gb != cap_int: obj.capacity_gb = cap_int; changed = True
                            if obj.color != color: obj.color = color; changed = True
                            if obj.release_date != rdate: obj.release_date = rdate; changed = True
                            if changed:
                                obj.save(update_fields=["model_name", "capacity_gb", "color", "release_date","jan"])
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
        auth=[],
        parameters=[
            OpenApiParameter("shop", OpenApiTypes.INT, description="按二手店ID", required=False),
            OpenApiParameter("shop_name", OpenApiTypes.STR, description="按二手店名（模糊）", required=False),
            OpenApiParameter("iphone", OpenApiTypes.INT, description="按 iPhone ID", required=False),
            OpenApiParameter("iphone_part_number", OpenApiTypes.STR, description="按 iPhone PN 精确匹配", required=False),
            OpenApiParameter("recorded_after", OpenApiTypes.DATETIME, description="记录时间不早于(ISO8601)", required=False),
            OpenApiParameter("recorded_before", OpenApiTypes.DATETIME, description="记录时间不晚于(ISO8601)", required=False),
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
    retrieve=extend_schema(tags=["Resale / Price"], summary="回收价格记录详情", auth=[]),
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
        if self.action in ["list", "retrieve"]:
            return [permissions.AllowAny()]
        return [permissions.IsAdminUser()]

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
        return { (k or "").strip().lower(): v for k, v in d.items() }


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
            OpenApiParameter("create_shop", OpenApiTypes.BOOL, description="若店铺不存在则创建（默认 true）", required=False),
            OpenApiParameter("dedupe", OpenApiTypes.BOOL, description="同店+PN+recorded_at 去重并更新（默认 true）", required=False),
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

        def _norm_key(d):
            """将 DictReader 的列名统一为小写去空格"""
            return {(k or "").strip().lower(): v for k, v in d.items()}
        """
        导入规则：
          - 必需：pn/part_number、shop_name、price_new
          - 若 shop 不存在且 create_shop=true -> 自动创建（按 name+address 判定唯一）
          - recorded_at 为空则使用服务器当前时间
          - dedupe: (shop, iphone, recorded_at) 相同则 update，否则 create
          - 每行单独原子性；允许部分成功
        """
        f = request.FILES.get("file")
        if not f:
            return Response({"detail": "缺少文件字段 file"}, status=status.HTTP_400_BAD_REQUEST)

        def as_bool(v, default=False):
            if v is None:
                return default
            s = str(v).strip().lower()
            return s in {"1", "true", "t", "yes", "y"}

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

        create_shop = as_bool(request.data.get("create_shop"), True)
        dedupe = as_bool(request.data.get("dedupe"), True)
        dry_run = as_bool(request.data.get("dry_run"), False)

        # 读取 CSV（自动处理 UTF-8 BOM）
        try:
            text = io.TextIOWrapper(f.file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
        except Exception as e:
            return Response({"detail": f"无法读取CSV: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        total = 0
        inserted = 0
        updated = 0
        skipped = 0
        errors = []
        preview = []

        for lineno, row in enumerate(reader, start=2):  # 从第2行起（跳过表头）
            total += 1
            data = _norm_key(row)

            # 列名映射
            pn = data.get("pn") or data.get("part_number") or data.get("iphone_pn")
            shop_name = data.get("shop_name") or data.get("shop")
            shop_addr = data.get("shop_address") or data.get("address") or ""
            shop_site = data.get("shop_website") or data.get("website") or ""

            price_new = _to_int_yen(data.get("price_new"))
            price_a = _to_int_yen(data.get("price_grade_a") or data.get("a") or data.get("grade_a"))
            price_b = _to_int_yen(data.get("price_grade_b") or data.get("b") or data.get("grade_b"))

            rec_at = _parse_recorded_at(data.get("recorded_at") or data.get("time") or data.get("date"))
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

            # 早发现错误
            if line_errors:
                errors.append({"line": lineno, "errors": line_errors, "row": row})
                skipped += 1
                continue

            # 匹配 iPhone
            iphone = Iphone.objects.filter(part_number=str(pn).strip()).first()
            if not iphone:
                errors.append({"line": lineno, "errors": [f"未找到 iPhone PN: {pn}"], "row": row})
                skipped += 1
                continue

            # 匹配/创建 shop（按 name+address）
            shop = SecondHandShop.objects.filter(name=shop_name.strip(), address=shop_addr.strip()).first()
            if not shop:
                if create_shop and not dry_run:
                    shop = SecondHandShop.objects.create(name=shop_name.strip(), address=shop_addr.strip(), website=shop_site.strip())
                elif create_shop and dry_run:
                    # dry_run 时仅模拟
                    shop = None  # 不创建实体
                else:
                    errors.append({"line": lineno, "errors": [f"未找到店铺: {shop_name} / {shop_addr}"], "row": row})
                    skipped += 1
                    continue

            # 预览（最多记录前 5 条）
            if len(preview) < 5:
                preview.append({
                    "pn": str(pn).strip(),
                    "shop_name": shop_name.strip(),
                    "shop_address": shop_addr.strip(),
                    "price_new": price_new,
                    "price_grade_a": price_a,
                    "price_grade_b": price_b,
                    "recorded_at": rec_at.isoformat(),
                })

            if dry_run:
                # 仅校验不落库
                continue

            # 单行原子操作
            with transaction.atomic():
                # 去重键：同店 + 同 iPhone + 相同 recorded_at（精确到秒）
                if dedupe and shop is not None:
                    existed = PurchasingShopPriceRecord.objects.filter(
                        shop=shop, iphone=iphone, recorded_at=rec_at
                    ).first()
                else:
                    existed = None

                if existed:
                    # 更新
                    changed = False
                    if existed.price_new != price_new:
                        existed.price_new = price_new; changed = True
                    if price_a is not None and existed.price_grade_a != price_a:
                        existed.price_grade_a = price_a; changed = True
                    if price_b is not None and existed.price_grade_b != price_b:
                        existed.price_grade_b = price_b; changed = True
                    if changed:
                        existed.save(update_fields=["price_new", "price_grade_a", "price_grade_b"])
                        updated += 1
                    else:
                        skipped += 1
                else:
                    # 创建（注意覆盖 auto_now_add 的 recorded_at）
                    if shop is None:
                        # dry_run + create_shop 情况不会到这里
                        errors.append({"line": lineno, "errors": ["内部错误：shop 为空"], "row": row})
                        skipped += 1
                        continue
                    rec = PurchasingShopPriceRecord.objects.create(
                        shop=shop,
                        iphone=iphone,
                        price_new=price_new,
                        price_grade_a=price_a,
                        price_grade_b=price_b,
                    )
                    # 覆盖 recorded_at
                    PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=rec_at)
                    inserted += 1

        resp = {
            "rows_total": total,
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "errors_count": len(errors),
            "errors": errors[:50],   # 防止回包过大，最多返回前 50 条错误
            "preview": preview,      # 前 5 条预览
            "options": {"create_shop": create_shop, "dedupe": dedupe, "dry_run": dry_run},
        }
        return Response(resp, status=status.HTTP_200_OK)

    @extend_schema(
        tags=["Resale / Price"],
        summary="导入二手店回收价（清洗逻辑已抽到 utils）——每次上传新增记录",
        parameters=[
            OpenApiParameter("create_shop", OpenApiTypes.BOOL, description="店铺不存在时自动创建（默认 true）",
                             required=False),
            OpenApiParameter("dry_run", OpenApiTypes.BOOL, description="只清洗/聚合不落库（默认 false）", required=False),
            OpenApiParameter("recorded_at", OpenApiTypes.DATETIME, description="统一记录时间（ISO8601/日期），默认现在",
                             required=False),
        ],
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {"files": {"type": "array", "items": {"type": "string", "format": "binary"}},
                               "file": {"type": "string", "format": "binary"}},
                "required": ["files"],
            }
        },
    )
    @action(
        detail=False, methods=["post"], url_path="import-tradein",
        parser_classes=[MultiPartParser, FormParser],
        permission_classes=[permissions.IsAdminUser],
    )
    def import_tradein(self, request):
        # ---- 参数 ----
        def as_bool(v, default=False):
            if v is None: return default
            return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

        create_shop = as_bool(request.query_params.get("create_shop"), True)
        dry_run = as_bool(request.query_params.get("dry_run"), False)

        ra_param = request.query_params.get("recorded_at")
        if ra_param:
            ra = parse_datetime(ra_param) or parse_date(ra_param)
            if isinstance(ra, datetime) and timezone.is_naive(ra):
                recorded_at = timezone.make_aware(ra, timezone.get_current_timezone())
            elif isinstance(ra, datetime):
                recorded_at = ra
            elif hasattr(ra, "year"):
                recorded_at = timezone.make_aware(datetime(ra.year, ra.month, ra.day), timezone.get_current_timezone())
            else:
                recorded_at = timezone.now()
        else:
            recorded_at = timezone.now()

        # ---- 读文件为 (bytes, name) 列表，交由 utils 清洗/聚合 ----
        files = request.FILES.getlist("files") or ([request.FILES["file"]] if request.FILES.get("file") else [])
        if not files:
            return Response({"detail": "请上传至少一个 CSV（字段 files 或 file）"}, status=status.HTTP_400_BAD_REQUEST)

        blobs = []
        for f in files:
            data = f.read()
            blobs.append((data, f.name))
            try:
                if hasattr(f, "seek"): f.seek(0)
            except Exception:
                pass

        result = clean_and_aggregate_tradein(blobs)
        rows_total = result["rows_total"]
        errors = result["errors"]
        records = result["records"]
        preview = result["preview"]

        if rows_total == 0 and not records:
            return Response({"detail": "清洗失败或无有效数据", "errors": errors}, status=status.HTTP_400_BAD_REQUEST)

        # ---- 写库：匹配 iPhone 并始终新增记录（不覆盖历史） ----
        inserted = 0
        skipped_no_match = 0

        for rec in records:
            shop_name = rec["shop_name"]
            price_new = rec["price_new"]
            price_a = rec["price_grade_a"]
            price_b = rec["price_grade_b"]
            meta = rec["meta"]

            pn = (meta.get("pn") or "").strip()
            jan_digits = (meta.get("jan") or "").strip()
            model_name = (meta.get("model_name") or "").strip()
            capacity_gb = meta.get("capacity_gb")
            color_raw = (meta.get("color_raw") or "").strip()
            color_canon = (meta.get("color_canon") or "").strip()
            color_any = bool(meta.get("color_any"))

            # iPhone 匹配：PN → JAN → 型号+容量(+颜色/全色)
            iphones: list[Iphone] = []
            if pn:
                ip = Iphone.objects.filter(part_number=pn).first()
                if ip: iphones = [ip]
            if not iphones and len(jan_digits) == 13:
                ip = Iphone.objects.filter(jan=jan_digits).first()
                if ip: iphones = [ip]
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

            # 店铺（name+address 唯一；此处地址未知置空）
            shop = SecondHandShop.objects.filter(name=shop_name, address="").first()
            if not shop and create_shop:
                shop = SecondHandShop.objects.create(name=shop_name, address="", website="")
            if not shop:
                continue

            # 始终新增：对匹配到的每个 iPhone 新建一条记录
            if dry_run:
                inserted += len(iphones)
                continue

            for iphone in iphones:
                with transaction.atomic():
                    rec_obj = PurchasingShopPriceRecord.objects.create(
                        shop=shop, iphone=iphone,
                        price_new=(price_new or 0),
                        price_grade_a=price_a,
                        price_grade_b=price_b,
                    )
                    PurchasingShopPriceRecord.objects.filter(pk=rec_obj.pk).update(recorded_at=recorded_at)
                    inserted += 1

        return Response({
            "rows_total": rows_total,
            "aggregated": len(records),
            "inserted": inserted,
            "skipped_no_match": skipped_no_match,
            "preview": preview,
            "errors": errors[:50],
            "options": {
                "create_shop": create_shop,
                "dry_run": dry_run,
                "recorded_at": recorded_at.isoformat(),
            },
        }, status=status.HTTP_200_OK)