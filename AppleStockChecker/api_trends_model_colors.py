from .trends_core import compute_trends_for_model_capacity
from .models import Iphone, PurchasingShopPriceRecord
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

def _norm_name(s: str) -> str:
    """统一店名/字符串比较：去首尾空白"""
    return (s or "").strip()


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def trends_model_colors(request):
    """
    请求 JSON:
    {
      "model_name": "iPhone 17",
      "capacity_gb": 256,
      "days": 30,
      "shops": ["買取一丁目","森森買取", ...],   // 勾选店；空或不传=ALL
      "avg": {
        "A": {"bucketMinutes": 30},
        "B": {"windowMinutes": 60,  "lineWidth": 2, "color":"#ff0077", "dash":"dash"},
        "C": {"windowMinutes": 240, "lineWidth": 2, "color":"#00bcd4", "dash":"dot"}
      },
      "grid": { "stepMinutes": 15, "offsetMinute": 0 }  // 0时N分开始、每 step 分钟一个点
    }
    响应 JSON：见 compute_trends_for_model_capacity()
    """
    payload = request.data or {}
    model_name = (payload.get("model_name") or "").strip()
    capacity_gb = int(payload.get("capacity_gb") or 0)
    days = int(payload.get("days") or 30)
    shops = payload.get("shops") or []
    avg_cfg = payload.get("avg") or {}
    grid_cfg = payload.get("grid") or {}

    if not model_name or not capacity_gb:
        return Response({"detail": "model_name/capacity_gb 不能为空"}, status=400)

    if shops:
        selected_shops = set(_norm_name(str(s)) for s in shops)
    else:
        # ALL：从库中去重
        selected_shops = set(_norm_name(n) for n in
                             PurchasingShopPriceRecord.objects.values_list("shop__name", flat=True).distinct())

    data = compute_trends_for_model_capacity(
        model_name, capacity_gb, days,
        selected_shops=selected_shops,
        avg_cfg=avg_cfg,
        grid_cfg=grid_cfg
    )
    return Response(data)
