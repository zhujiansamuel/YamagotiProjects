# AppleStockChecker/views_options.py
import re
from typing import Set
from django.db.models import Q
from typing import List, Dict
from django.db.models import Prefetch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

from AppleStockChecker.models import Iphone
from AppleStockChecker.serializers import (
    IphoneOptionSerializer,
    ShopOptionSerializer,
    ShopWeightProfileOptionSerializer,
    CohortOptionSerializer,
    _cap_label_from_gb,
)
from AppleStockChecker.models import (
    SecondHandShop,
    Iphone,
    Cohort, CohortMember,
    ShopWeightProfile, ShopWeightItem
)

def _parse_capacity_q(q: str):
    """
    从用户 q 中解析容量数字（支持 256 / 256g / 256GB / " 256 GB "）
    返回 int 或 None
    """
    if not q:
        return None
    m = re.match(r"^\s*(\d{2,5})\s*(g|gb)?\s*$", q, flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _capacity_label(i: Iphone) -> str | None:
    gb = getattr(i, "capacity_gb", None)
    return f"{gb}GB" if gb not in (None, "") else None

def _iphone_label(i: Iphone) -> str:
    # 组合一个人类可读标签（不依赖 DB 存在的 capacity_label 字段）
    parts = [
        getattr(i, "part_number", None),
        getattr(i, "model_name", None),
        _capacity_label(i),
        getattr(i, "color", None),
    ]
    return " ｜ ".join([p for p in parts if p])

def _shop_item_to_dict(it: ShopWeightItem) -> dict:
    return {
        "id": it.id,
        "shop_id": it.shop_id,
        "shop_name": getattr(it.shop, "name", str(it.shop_id)),
        "weight": float(it.weight or 0.0),
        "display_index": int(it.display_index or 0),
    }

class ScopeOptionsView(APIView):
    """
    GET /AppleStockChecker/options/scopes/?limit=
    返回：
      - shop_profiles: [{id, slug, title, label, items:[{shop_id, shop_name, weight, display_index}]}]
      - cohorts:       [{id, slug, title, label, members:[{iphone_id, ... , weight, label}]}]

    仍然保持“轻量列表”接口在其它 endpoint（如 /options/iphones/），
    这里专注返回“组合可视配置”给前端下拉使用。
    """
    permission_classes = [AllowAny]

    def get(self, request):
        # 限制顶层数量（嵌套 items/members 不限）
        try:
            limit = int(request.query_params.get("limit") or 200)
        except Exception:
            limit = 200

        # === Shop Profiles（带 items->shop） ===
        prof_qs = (
            ShopWeightProfile.objects
            .prefetch_related(
                Prefetch(
                    "items",
                    queryset=ShopWeightItem.objects.select_related("shop").order_by("display_index", "shop__name")
                )
            )
            .order_by("slug")[:limit]
        )
        shop_profiles = ShopWeightProfileOptionSerializer(prof_qs, many=True).data

        # === Cohorts（带 members->iphone） ===
        cohort_qs = (
            Cohort.objects
            .prefetch_related(
                Prefetch(
                    "members",
                    queryset=(CohortMember.objects
                              .select_related("iphone")
                              .order_by("iphone__model_name", "iphone__capacity_gb", "iphone__part_number"))
                )
            )
            .order_by("slug")[:limit]
        )
        cohorts = CohortOptionSerializer(cohort_qs, many=True).data

        return Response({
            "shop_profiles": shop_profiles,
            "cohorts": cohorts,
        })
# -------- 细分接口：便于单独请求/缓存/调试 --------

@api_view(["GET"])
@permission_classes([AllowAny])
def options_shops(request):
    qs = SecondHandShop.objects.all().order_by("id").values("id", "name")
    items = [{"id": s["id"], "name": s.get("name") or f"#{s['id']}"} for s in qs]
    return Response({"shops": items})

@api_view(["GET"])
@permission_classes([AllowAny])
def options_iphones(request):
    """
    GET /AppleStockChecker/options/iphones/?q=...&limit=...
    支持按 part_number / model_name / color 模糊，
    若 q 看起来像容量（256/256gb），则转成 capacity_gb=256 精确匹配
    """
    q = (request.query_params.get("q") or "").strip()
    limit = int(request.query_params.get("limit") or 200)

    qs = Iphone.objects.all()
    if q:
        cond = Q(part_number__icontains=q) | Q(model_name__icontains=q) | Q(color__icontains=q)
        cap = _parse_capacity_q(q)
        if cap is not None:
            cond = cond | Q(capacity_gb=cap)
        qs = qs.filter(cond)

    qs = qs.order_by("id")[:limit]
    data = IphoneOptionSerializer(qs, many=True).data
    return Response(data)

@api_view(["GET"])
@permission_classes([AllowAny])
def options_cohorts(request):
    qs = (Cohort.objects
          .prefetch_related(Prefetch("members", queryset=CohortMember.objects.only("iphone_id")))
          .order_by("slug"))
    items = []
    for c in qs:
        ids = [m.iphone_id for m in c.members.all()]
        items.append({"slug": c.slug, "label": c.slug, "n_members": len(ids), "iphone_ids": ids})
    return Response({"cohorts": items})

@api_view(["GET"])
@permission_classes([AllowAny])
def options_shop_profiles(request):
    qs = (ShopWeightProfile.objects
          .prefetch_related(Prefetch(
              "items",
              queryset=ShopWeightItem.objects.select_related("shop").order_by("display_index", "shop_id")
          ))
          .order_by("slug"))
    items = []
    for p in qs:
        arr = [_shop_item_to_dict(it) for it in p.items.all()]
        items.append({"slug": p.slug, "title": p.title or p.slug, "n_shops": len(arr), "items": arr})
    return Response({"profiles": items})
