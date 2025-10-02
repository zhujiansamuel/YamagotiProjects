from .trends_core import compute_trends_for_model_capacity

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

class TrendsAvgOnlyApiView(APIView):
    """
    POST /AppleStockChecker/api/trends/model-colors/avg-only/
    请求体与 /model-colors/ 完全一致，但仅返回平均线（A/B/C），不带店铺明细，以加速“只重算平均线”的场景。
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        p = request.data or {}
        model_name = (p.get("model_name") or "").strip()
        capacity_gb = int(p.get("capacity_gb") or 0)
        days = int(p.get("days") or 30)
        shops = set((p.get("shops") or []))
        avg_cfg = p.get("avg") or {}
        grid_cfg = p.get("grid") or {}
        if not model_name or not capacity_gb:
            return Response({"detail": "model_name/capacity_gb 不能为空"}, status=400)

        data = compute_trends_for_model_capacity(
            model_name=model_name,
            capacity_gb=capacity_gb,
            days=days,
            selected_shops=shops,
            avg_cfg=avg_cfg,
            grid_cfg=grid_cfg,
        )
        # 只返回平均线，丢弃 stores，以缩小体积
        resp = {
            "merged": {"avg": data["merged"]["avg"]},
            "per_color": [{"color": it["color"], "avg": it["avg"]} for it in data["per_color"]],
        }
        return Response(resp, status=200)