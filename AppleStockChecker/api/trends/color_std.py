"""
单色或合并（跨颜色）的均值±标准差、上下界曲线。
"""
from .core import (
    compute_trends_for_model_capacity,
    _norm_name,
    _build_time_grid,
    _moving_average_time,
    TREND_MAX_LOOKBACK_DAYS,
)
from ...models import Iphone, PurchasingShopPriceRecord
from django.utils import timezone
from math import sqrt
from datetime import timedelta
from typing import Dict, List
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView


def _std(values: List[float]) -> float:
    """总体标准差（样本数量<=1 时记 0）"""
    n = len(values)
    if n <= 1:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    return sqrt(var)


def _moving_std_time(points: List[Dict], window_minutes: int) -> List[Dict]:
    """对 A 线做"时间窗(分钟)"移动标准差（窗口取 points 中 t-w..t 的 y 值）"""
    if not points:
        return []
    wms = max(1, int(window_minutes)) * 60 * 1000
    pts = sorted(points, key=lambda p: p["x"])
    out = []
    for i, pt in enumerate(pts):
        t = pt["x"]
        bucket = [p["y"] for p in pts if (t - p["x"]) <= wms and p["x"] <= t and p["y"] is not None]
        sd = _std(bucket) if bucket else 0.0
        out.append({"x": t, "y": sd})
    return out


class TrendsColorStdApiView(APIView):
    """POST /AppleStockChecker/api/trends/model-color/std/"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """
        请求 JSON:
        {
          "model_name": "iPhone 17",
          "capacity_gb": 256,
          "color": "ミストブルー",         # 必填：单色；color="__MERGED__" 为跨颜色
          "days": 30,
          "shops": ["買取一丁目","森森買取", ...],   # 勾选店；空=ALL
          "grid": { "stepMinutes": 15, "offsetMinute": 0 },
          "avg": { "A": {"bucketMinutes": 30}, "B": {"windowMinutes": 60}, "C": {"windowMinutes": 240} }
        }
        响应 JSON:
        {
          "A": {"mean":[{x,y}...], "std":[{x,y}...], "upper":[{x,y}], "lower":[{x,y}]},
          "B": {...},
          "C": {...}
        }
        """
        p = request.data or {}
        model_name = (p.get("model_name") or "").strip()
        capacity_gb = int(p.get("capacity_gb") or 0)
        color = _norm_name(p.get("color") or "")
        days = int(p.get("days") or 30)
        shops = p.get("shops") or []
        grid = p.get("grid") or {}
        avg = p.get("avg") or {}

        if not (model_name and capacity_gb and color):
            return Response({"detail": "model_name/capacity_gb/color 不能为空"}, status=400)

        step_minutes = int(grid.get("stepMinutes", 15))
        offset_minute = int(grid.get("offsetMinute", 0))

        A_cfg = avg.get("A", {}) if avg else {}
        B_cfg = avg.get("B", {}) if avg else {}
        C_cfg = avg.get("C", {}) if avg else {}
        b_win = int(B_cfg.get("windowMinutes", 60))
        c_win = int(C_cfg.get("windowMinutes", 240))

        now = timezone.now()
        window_start = now - timedelta(days=days)
        history_after = min(window_start, now - timedelta(days=TREND_MAX_LOOKBACK_DAYS))
        grid_ms = _build_time_grid(window_start, now, step_minutes=step_minutes, offset_minute=offset_minute)

        # ---------- 合并图（跨颜色）分支 ----------
        if color == "__MERGED__":
            data = compute_trends_for_model_capacity(
                model_name=model_name,
                capacity_gb=capacity_gb,
                days=days,
                selected_shops={_norm_name(s) for s in shops} if shops else set(),
                avg_cfg=avg,
                grid_cfg=grid
            )
            stores = data.get("merged", {}).get("stores", [])
            if not stores:
                return Response({"detail": "无店铺数据"}, status=200)

            sel = {_norm_name(s) for s in shops} if shops else {_norm_name(s["label"]) for s in stores}

            any_series = stores[0]["data"]
            A_mean, A_std = [], []
            for idx in range(len(any_series)):
                x = any_series[idx]["x"]
                bucket = []
                for s in stores:
                    nm = _norm_name(s["label"])
                    if nm not in sel:
                        continue
                    v = s["data"][idx].get("y") if idx < len(s["data"]) else None
                    if v is not None:
                        bucket.append(float(v))
                if bucket:
                    mu = sum(bucket) / len(bucket)
                    sd = _std(bucket)
                    A_mean.append({"x": x, "y": mu})
                    A_std.append({"x": x, "y": sd})

            A_upper = [{"x": d["x"], "y": d["y"] + (A_std[i]["y"] if i < len(A_std) else 0.0)} for i, d in enumerate(A_mean)]
            A_lower = [{"x": d["x"], "y": d["y"] - (A_std[i]["y"] if i < len(A_std) else 0.0)} for i, d in enumerate(A_mean)]

            B_mean = _moving_average_time(A_mean, b_win)
            B_std = _moving_std_time(A_mean, b_win)
            B_upper = [{"x": d["x"], "y": d["y"] + (B_std[i]["y"] if i < len(B_std) else 0.0)} for i, d in enumerate(B_mean)]
            B_lower = [{"x": d["x"], "y": d["y"] - (B_std[i]["y"] if i < len(B_std) else 0.0)} for i, d in enumerate(B_mean)]

            C_mean = _moving_average_time(A_mean, c_win)
            C_std = _moving_std_time(A_mean, c_win)
            C_upper = [{"x": d["x"], "y": d["y"] + (C_std[i]["y"] if i < len(C_std) else 0.0)} for i, d in enumerate(C_mean)]
            C_lower = [{"x": d["x"], "y": d["y"] - (C_std[i]["y"] if i < len(C_std) else 0.0)} for i, d in enumerate(C_mean)]

            return Response({
                "A": {"mean": A_mean, "std": A_std, "upper": A_upper, "lower": A_lower},
                "B": {"mean": B_mean, "std": B_std, "upper": B_upper, "lower": B_lower},
                "C": {"mean": C_mean, "std": C_std, "upper": C_upper, "lower": C_lower},
            }, status=200)

        # ---------- 单色分支 ----------
        pns = list(Iphone.objects.filter(model_name=model_name, capacity_gb=capacity_gb, color=color)
                   .values_list("part_number", flat=True))
        if not pns:
            return Response({"detail": f"该颜色无机型: {color}"}, status=400)

        tz = timezone.get_current_timezone()
        store_raw = {}
        qs = PurchasingShopPriceRecord.objects.filter(
            iphone__part_number__in=pns, recorded_at__gte=history_after
        ).select_related("shop").only("recorded_at", "price_new", "shop__name").order_by("recorded_at")
        for r in qs.iterator():
            shop = _norm_name(r.shop.name)
            t = int(timezone.localtime(r.recorded_at, tz).timestamp() * 1000)
            store_raw.setdefault(shop, []).append({"x": t, "y": r.price_new})

        def resample(pts):
            if not pts:
                return [{"x": t, "y": None} for t in grid_ms]
            i = 0
            n = len(pts)
            out = []
            for t in grid_ms:
                while i + 1 < n and abs(pts[i + 1]["x"] - t) < abs(pts[i]["x"] - t):
                    i += 1
                out.append({"x": t, "y": pts[i]["y"]})
            return out

        store_rs = {shop: resample(seq) for shop, seq in store_raw.items()}
        sel = {_norm_name(s) for s in shops} if shops else set(store_rs.keys())

        any_series = next(iter(store_rs.values()), [])
        A_mean, A_std_list = [], []
        for idx in range(len(any_series)):
            x = any_series[idx]["x"] if any_series else None
            bucket = [seq[idx]["y"] for shop, seq in store_rs.items() if shop in sel and seq[idx]["y"] is not None]
            if bucket:
                mu = sum(bucket) / len(bucket)
                sd = _std(bucket)
                A_mean.append({"x": x, "y": mu})
                A_std_list.append({"x": x, "y": sd})
        A_upper = [{"x": d["x"], "y": d["y"] + (A_std_list[i]["y"] if i < len(A_std_list) else 0.0)} for i, d in enumerate(A_mean)]
        A_lower = [{"x": d["x"], "y": d["y"] - (A_std_list[i]["y"] if i < len(A_std_list) else 0.0)} for i, d in enumerate(A_mean)]

        B_mean = _moving_average_time(A_mean, b_win)
        B_std_list = _moving_std_time(A_mean, b_win)
        B_upper = [{"x": d["x"], "y": d["y"] + (B_std_list[i]["y"] if i < len(B_std_list) else 0.0)} for i, d in enumerate(B_mean)]
        B_lower = [{"x": d["x"], "y": d["y"] - (B_std_list[i]["y"] if i < len(B_std_list) else 0.0)} for i, d in enumerate(B_mean)]

        C_mean = _moving_average_time(A_mean, c_win)
        C_std_list = _moving_std_time(A_mean, c_win)
        C_upper = [{"x": d["x"], "y": d["y"] + (C_std_list[i]["y"] if i < len(C_std_list) else 0.0)} for i, d in enumerate(C_mean)]
        C_lower = [{"x": d["x"], "y": d["y"] - (C_std_list[i]["y"] if i < len(C_std_list) else 0.0)} for i, d in enumerate(C_mean)]

        return Response({
            "A": {"mean": A_mean, "std": A_std_list, "upper": A_upper, "lower": A_lower},
            "B": {"mean": B_mean, "std": B_std_list, "upper": B_upper, "lower": B_lower},
            "C": {"mean": C_mean, "std": C_std_list, "upper": C_upper, "lower": C_lower},
        }, status=200)
