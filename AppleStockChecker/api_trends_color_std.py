from .trends_core import compute_trends_for_model_capacity
from .models import Iphone, PurchasingShopPriceRecord
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from django.utils import timezone
from .models import Iphone, PurchasingShopPriceRecord
from typing import Dict, List, Tuple, Iterable, Any
from math import sqrt
from datetime import timedelta
from django.conf import settings
import csv, io, re
import math
import os
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView



TREND_MAX_LOOKBACK_DAYS = int(getattr(settings, "TREND_MAX_LOOKBACK_DAYS", 90))
TREND_DB_MAX_WORKERS    = int(getattr(settings, "TREND_DB_MAX_WORKERS", 6))
TREND_CPU_MAX_WORKERS   = int(getattr(settings, "TREND_CPU_MAX_WORKERS", 0)) or (os.cpu_count() or 2)
TREND_DOWNSAMPLE_TARGET = int(getattr(settings, "TREND_DOWNSAMPLE_TARGET", 0))  # 0=关闭，>0=每条最多点数


def _aware_local(dt):
    tz = timezone.get_current_timezone()
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, tz)
    return dt.astimezone(tz)

def _moving_average_time(points: List[Dict], window_minutes: int) -> List[Dict]:
    """时间窗(分钟)移动平均（包含当前点），在 A 线结果上做平滑。"""
    if not points:
        return []
    wms = max(1, int(window_minutes)) * 60 * 1000
    pts = sorted(points, key=lambda p: p["x"])
    out = []
    head = 0
    s = 0.0
    c = 0
    for i, pt in enumerate(pts):
        s += float(pt["y"]); c += 1
        while head <= i and (pt["x"] - pts[head]["x"]) > wms:
            s -= float(pts[head]["y"]); c -= 1; head += 1
        out.append({"x": pt["x"], "y": s / c if c else None})
    return out

def _build_time_grid(start_dt, end_dt, step_minutes: int = 15, offset_minute: int = 0) -> List[int]:
    """生成本地时区网格（ms）：从 start_dt~end_dt，对齐到“0时offset分+步长 step 分钟”"""
    step_minutes = max(1, int(step_minutes))
    offset_minute = int(offset_minute) % step_minutes

    start_local = _aware_local(start_dt)
    end_local = _aware_local(end_dt)

    day0 = start_local.replace(hour=0, minute=0, second=0, microsecond=0)
    first = day0 + timedelta(minutes=offset_minute)
    if first < start_local:
        delta_min = (start_local - first).total_seconds() / 60.0
        steps = int(delta_min // step_minutes) + 1
        first = first + timedelta(minutes=steps * step_minutes)

    grid = []
    cur = first
    while cur <= end_local:
        grid.append(int(cur.timestamp() * 1000))
        cur = cur + timedelta(minutes=step_minutes)
    return grid


def _norm_name(s: str) -> str:
    """统一店名/字符串比较：去首尾空白"""
    return (s or "").strip()



def _std(values: List[float]) -> float:
    """总体标准差（样本数量<=1 时记 0）"""
    n = len(values)
    if n <= 1:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    return sqrt(var)


def _moving_std_time(points: List[Dict], window_minutes: int) -> List[Dict]:
    """对 A 线做“时间窗(分钟)”移动标准差（窗口取 points 中 t-w..t 的 y 值）"""
    if not points:
        return []
    wms = max(1, int(window_minutes)) * 60 * 1000
    pts = sorted(points, key=lambda p: p["x"])
    out = []
    head = 0
    acc = []  # 为了简洁直接切片计算，若点很密可用双端队列维护
    for i, pt in enumerate(pts):
        # 这里简单每次切片；若性能瓶颈，再改成滑窗累加器
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
          "color": "ミストブルー",         # 必填：单色；若想支持 merged 可另加 flag
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
            # 直接复用核心计算，拿到 merged.stores（每店一条、已网格化）
            data = compute_trends_for_model_capacity(
                model_name=model_name,
                capacity_gb=capacity_gb,
                days=days,
                selected_shops={_norm_name(s) for s in shops} if shops else set(),
                avg_cfg=avg,
                grid_cfg=grid
            )
            stores = data.get("merged", {}).get("stores", [])  # [{label, data:[{x,y}...]}]
            if not stores:
                return Response({"detail": "无店铺数据"}, status=200)

            # 参与店集合（空则全部）
            sel = {_norm_name(s) for s in shops} if shops else {_norm_name(s["label"]) for s in stores}

            # A：对同一 grid 点收集参与店的样本，做均值与标准差
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
                    if v is not None: bucket.append(float(v))
                if bucket:
                    mu = sum(bucket) / len(bucket)
                    sd = _std(bucket)  # 总体标准差；若取样本标准差可用 (n-1)
                    A_mean.append({"x": x, "y": mu})
                    A_std.append({"x": x, "y": sd})

            A_upper = [{"x": d["x"], "y": d["y"] + (A_std[i]["y"] if i < len(A_std) else 0.0)} for i, d in
                       enumerate(A_mean)]
            A_lower = [{"x": d["x"], "y": d["y"] - (A_std[i]["y"] if i < len(A_std) else 0.0)} for i, d in
                       enumerate(A_mean)]

            # B/C：在 A 线的均值上做时间窗移动均值；std 取“窗口内 A_mean 的波动”
            B_mean = _moving_average_time(A_mean, b_win)
            B_std = _moving_std_time(A_mean, b_win)
            B_upper = [{"x": d["x"], "y": d["y"] + (B_std[i]["y"] if i < len(B_std) else 0.0)} for i, d in
                       enumerate(B_mean)]
            B_lower = [{"x": d["x"], "y": d["y"] - (B_std[i]["y"] if i < len(B_std) else 0.0)} for i, d in
                       enumerate(B_mean)]

            C_mean = _moving_average_time(A_mean, c_win)
            C_std = _moving_std_time(A_mean, c_win)
            C_upper = [{"x": d["x"], "y": d["y"] + (C_std[i]["y"] if i < len(C_std) else 0.0)} for i, d in
                       enumerate(C_mean)]
            C_lower = [{"x": d["x"], "y": d["y"] - (C_std[i]["y"] if i < len(C_std) else 0.0)} for i, d in
                       enumerate(C_mean)]

            return Response({
                "A": {"mean": A_mean, "std": A_std, "upper": A_upper, "lower": A_lower},
                "B": {"mean": B_mean, "std": B_std, "upper": B_upper, "lower": B_lower},
                "C": {"mean": C_mean, "std": C_std, "upper": C_upper, "lower": C_lower},
            }, status=200)

        # ---------- 单色分支：与你之前版本一致 ----------
        # 取该颜色下所有 PN → 拉数据 → 最近邻重采样至 grid → A/B/C
        from .models import Iphone, PurchasingShopPriceRecord
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

        # 最近邻重采样到 grid
        def resample(points):
            if not points:
                return [{"x": t, "y": None} for t in grid_ms]
            i = 0;
            n = len(points);
            out = []
            for t in grid_ms:
                while i + 1 < n and abs(points[i + 1]["x"] - t) < abs(points[i]["x"] - t):
                    i += 1
                out.append({"x": t, "y": points[i]["y"]})
            return out

        store_rs = {shop: resample(seq) for shop, seq in store_raw.items()}

        # 参与店铺集合
        sel = {_norm_name(s) for s in shops} if shops else set(store_rs.keys())

        # A：同一 grid 点对参与店的样本做均值+std
        any_series = next(iter(store_rs.values()), [])
        A_mean, A_std = [], []
        for idx in range(len(any_series)):
            x = any_series[idx]["x"] if any_series else None
            bucket = [seq[idx]["y"] for shop, seq in store_rs.items() if shop in sel and seq[idx]["y"] is not None]
            if bucket:
                mu = sum(bucket) / len(bucket)
                sd = _std(bucket)
                A_mean.append({"x": x, "y": mu})
                A_std.append({"x": x, "y": sd})
        A_upper = [{"x": d["x"], "y": d["y"] + (A_std[i]["y"] if i < len(A_std) else 0.0)} for i, d in
                   enumerate(A_mean)]
        A_lower = [{"x": d["x"], "y": d["y"] - (A_std[i]["y"] if i < len(A_std) else 0.0)} for i, d in
                   enumerate(A_mean)]

        # B/C：在 A_mean 上做时间窗移动均值 + 移动标准差
        B_mean = _moving_average_time(A_mean, b_win)
        B_std = _moving_std_time(A_mean, b_win)
        B_upper = [{"x": d["x"], "y": d["y"] + (B_std[i]["y"] if i < len(B_std) else 0.0)} for i, d in
                   enumerate(B_mean)]
        B_lower = [{"x": d["x"], "y": d["y"] - (B_std[i]["y"] if i < len(B_std) else 0.0)} for i, d in
                   enumerate(B_mean)]

        C_mean = _moving_average_time(A_mean, c_win)
        C_std = _moving_std_time(A_mean, c_win)
        C_upper = [{"x": d["x"], "y": d["y"] + (C_std[i]["y"] if i < len(C_std) else 0.0)} for i, d in
                   enumerate(C_mean)]
        C_lower = [{"x": d["x"], "y": d["y"] - (C_std[i]["y"] if i < len(C_std) else 0.0)} for i, d in
                   enumerate(C_mean)]

        return Response({
            "A": {"mean": A_mean, "std": A_std, "upper": A_upper, "lower": A_lower},
            "B": {"mean": B_mean, "std": B_std, "upper": B_upper, "lower": B_lower},
            "C": {"mean": C_mean, "std": C_std, "upper": C_upper, "lower": C_lower},
        }, status=200)


