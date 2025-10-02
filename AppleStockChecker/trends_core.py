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
from concurrent.futures import ProcessPoolExecutor
# =========================
# 基础工具
# =========================

TREND_MAX_LOOKBACK_DAYS = int(getattr(settings, "TREND_MAX_LOOKBACK_DAYS", 90))
TREND_DB_MAX_WORKERS    = int(getattr(settings, "TREND_DB_MAX_WORKERS", 6))
TREND_CPU_MAX_WORKERS   = int(getattr(settings, "TREND_CPU_MAX_WORKERS", 0)) or (os.cpu_count() or 2)
TREND_DOWNSAMPLE_TARGET = int(getattr(settings, "TREND_DOWNSAMPLE_TARGET", 0))  # 0=关闭，>0=每条最多点数


def _aware_local(dt):
    tz = timezone.get_current_timezone()
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, tz)
    return dt.astimezone(tz)


def _norm_name(s: str) -> str:
    """统一店名/字符串比较：去首尾空白"""
    return (s or "").strip()


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


def _resample_nearest(points: List[Dict], grid_ms: List[int]) -> List[Dict]:
    """
    最近邻重采样：对每个网格 t，挑绝对时间差最小的历史点（允许“未来”点），保证 15min 连续。
    points: [{x(ms), y}...]（升序）
    返回与 grid 等长的 [{x,y}]
    """
    out = []
    n = len(points)
    if n == 0:
        return [{"x": t, "y": None} for t in grid_ms]
    i = 0
    for t in grid_ms:
        while i + 1 < n and abs(points[i + 1]["x"] - t) < abs(points[i]["x"] - t):
            i += 1
        out.append({"x": t, "y": points[i].get("y")})
    return out


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


def _order_shops(all_names: List[str]) -> List[str]:
    """按 settings.SHOP_DISPLAY_ORDER 排序；未在该列表的放末尾（按名称字母序）"""
    preferred = list(getattr(settings, "SHOP_DISPLAY_ORDER", []))
    pref_index = {name: i for i, name in enumerate(preferred)}
    def key(name: str):
        return (0, pref_index[name]) if name in pref_index else (1, name)
    return sorted(all_names, key=key)


def _downsample_minmax(points: List[Dict], target: int) -> List[Dict]:
    """简易 min/max 桶降采样，保留趋势峰谷；target<=0 时不降采样"""
    if target <= 0:
        return points
    n = len(points)
    if n <= target:
        return points
    bucket = n // target
    out: List[Dict] = []
    for i in range(0, n, bucket):
        chunk = points[i:i + bucket]
        if not chunk:
            continue
        # 取时间顺序中的最小/最大 y
        ymin = min(chunk, key=lambda p: (p["y"] if p["y"] is not None else math.inf))
        ymax = max(chunk, key=lambda p: (p["y"] if p["y"] is not None else -math.inf))
        uniq = {id(ymin): ymin}
        uniq[id(ymax)] = ymax
        out.extend(sorted(uniq.values(), key=lambda p: p["x"]))
    return out

# =========================
# 并行计算（Map / Reduce）
# =========================
def _fetch_points_for_color(model_name: str, capacity_gb: int, color: str, history_after, fields=("recorded_at", "price_new", "shop__name")) -> Dict[str, List[Dict]]:
    """
    在线程池中调用：读取某颜色下所有 PN 的记录（限制最早时间 history_after），仅取必要字段，
    输出：{shop -> [{x,y}...升序]}
    """
    tz = timezone.get_current_timezone()
    pns = list(Iphone.objects.filter(model_name=model_name, capacity_gb=capacity_gb, color=color)
               .values_list("part_number", flat=True))
    store_map: Dict[str, List[Dict]] = defaultdict(list)
    if not pns:
        return store_map

    qs = PurchasingShopPriceRecord.objects.filter(
        iphone__part_number__in=pns,
        recorded_at__gte=history_after
    ).select_related("shop").only(*fields).order_by("recorded_at")

    # 组装轻量点集
    for r in qs.iterator():
        shop_name = _norm_name(r.shop.name)
        t = int(timezone.localtime(r.recorded_at, tz).timestamp() * 1000)
        store_map[shop_name].append({"x": t, "y": r.price_new})

    # 排序
    for shop in store_map:
        store_map[shop].sort(key=lambda p: p["x"])
    return store_map





def _resample_task(args: Tuple[str, str, List[Dict], List[int]]) -> Tuple[str, str, List[Dict]]:
    """
    在进程池中调用：对 (shop,color) 的原始点做最近邻重采样
    入参：color, shop, points, grid_ms
    出参：(color, shop, resampled_points)
    """
    color, shop, points, grid_ms = args
    seq = _resample_nearest(points, grid_ms)
    return color, shop, seq

# =========================
# 核心计算
# =========================

def compute_trends_for_model_capacity(model_name: str,
                                      capacity_gb: int,
                                      days: int,
                                      selected_shops: set[str],
                                      avg_cfg: dict,
                                      grid_cfg: dict | None = None) -> dict:
    """
    见模块顶部说明
    """
    tz = timezone.get_current_timezone()
    now = timezone.now()
    start_window = now - timedelta(days=max(1, int(days)))  # 展示窗口/网格范围
    # 最近邻最大回溯窗口（避免全库扫描）
    history_after = min(start_window, now - timedelta(days=TREND_MAX_LOOKBACK_DAYS))

    # 配置
    A_cfg = (avg_cfg or {}).get("A", {})
    B_cfg = (avg_cfg or {}).get("B", {})
    C_cfg = (avg_cfg or {}).get("C", {})
    b_win = max(1, int(B_cfg.get("windowMinutes", 60)))
    c_win = max(1, int(C_cfg.get("windowMinutes", 240)))
    step_minutes = int((grid_cfg or {}).get("stepMinutes", 15))
    offset_minute = int((grid_cfg or {}).get("offsetMinute", 0))  # 0 时 N 分

    # 机型+容量 → 颜色列表（规范化）
    colors = list(Iphone.objects.filter(model_name=model_name, capacity_gb=int(capacity_gb))
                  .values_list("color", flat=True).distinct())
    colors = sorted({_norm_name(c) for c in colors if _norm_name(c)})

    # 1) DB I/O 并行：按颜色并行取数 → per_color_store_raw[color][shop] = points
    per_color_store_raw: Dict[str, Dict[str, List[Dict]]] = {}
    with ThreadPoolExecutor(max_workers=TREND_DB_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_points_for_color, model_name, capacity_gb, color, history_after): color
            for color in colors
        }
        for fut in as_completed(futures):
            color = futures[fut]
            try:
                per_color_store_raw[color] = fut.result()
            except Exception as e:
                # 某色失败不应影响其它，记空
                per_color_store_raw[color] = {}
                print(f"[warn] fetch color={color} failed: {e}")

    # 2) 生成网格
    grid_ms = _build_time_grid(start_window, now, step_minutes=step_minutes, offset_minute=offset_minute)
    grid_len = len(grid_ms)

    # 3) CPU 并行：对每个 (店,色) 做最近邻重采样
    # 汇总参数
    tasks: List[Tuple[str, str, List[Dict], List[int]]] = []
    all_shops_set: set[str] = set()
    for color, store_map in per_color_store_raw.items():
        for shop, pts in store_map.items():
            tasks.append((color, shop, pts, grid_ms))
            all_shops_set.add(shop)

    per_color_resampled: Dict[str, Dict[str, List[Dict]]] = {c: {} for c in colors}
    if tasks:
        with ThreadPoolExecutor(max_workers=TREND_CPU_MAX_WORKERS or 8) as tpool:
            for color, shop, seq in tpool.map(_resample_task, tasks, chunksize=max(1, len(tasks) // 8 or 1)):
                per_color_resampled[color][shop] = seq

    # 4) 整理顺序（全量 & 本次窗口）
    # 全库店名 → 规范化 → 唯一化 → 排序
    all_names_db = list(PurchasingShopPriceRecord.objects.values_list("shop__name", flat=True).distinct())
    names_norm = [_norm_name(n) for n in all_names_db if _norm_name(n)]
    names_unique = list(dict.fromkeys(names_norm))           # 稳定去重
    shop_order_all = _order_shops(names_unique)

    shop_order_present = _order_shops(sorted(all_shops_set))

    # 5) merged（跨颜色、每店）= 同一网格点，对所有颜色做横向均值（连续 15min）
    merged_store_resampled_avg: Dict[str, List[Dict]] = {}
    for shop in shop_order_present:
        series = []
        for idx in range(grid_len):
            x = grid_ms[idx]
            ys = []
            for color, store_map in per_color_resampled.items():
                seq = store_map.get(shop)
                if not seq:
                    continue
                y = seq[idx].get("y")
                if y is not None:
                    ys.append(float(y))
            series.append({"x": x, "y": (sum(ys)/len(ys)) if ys else None})
        # 可选后端降采样
        series = _downsample_minmax(series, TREND_DOWNSAMPLE_TARGET)
        merged_store_resampled_avg[shop] = series

    # 6) 横向均值（在网格上对勾选店做均值）
    sel_shops = {_norm_name(s) for s in selected_shops or set(shop_order_all)}
    any_series = next(iter(merged_store_resampled_avg.values()), [])
    seriesA_merged: List[Dict] = []
    for idx in range(len(any_series)):
        x = any_series[idx]["x"]
        ys = []
        for shop, seq in merged_store_resampled_avg.items():
            if shop not in sel_shops:
                continue
            y = seq[idx].get("y") if idx < len(seq) else None
            if y is not None:
                ys.append(float(y))
        if ys:
            seriesA_merged.append({"x": x, "y": sum(ys)/len(ys)})
    # B/C
    seriesB_merged = _moving_average_time(seriesA_merged, b_win)
    seriesC_merged = _moving_average_time(seriesA_merged, c_win)
    # 降采样（可选）
    if TREND_DOWNSAMPLE_TARGET > 0:
        seriesA_merged = _downsample_minmax(seriesA_merged, TREND_DOWNSAMPLE_TARGET)
        seriesB_merged = _downsample_minmax(seriesB_merged, TREND_DOWNSAMPLE_TARGET)
        seriesC_merged = _downsample_minmax(seriesC_merged, TREND_DOWNSAMPLE_TARGET)

    merged = {
        "stores": [
            {"label": shop, "data": merged_store_resampled_avg[shop]}
            for shop in shop_order_present
        ],
        "avg": {"A": seriesA_merged, "B": seriesB_merged, "C": seriesC_merged}
    }

    # 7) per_color：每色也返回“重采样后的每店曲线”及其 A/B/C（店顺序按 settings）
    per_color = []
    for color in colors:
        stores_rs = per_color_resampled.get(color, {})
        stores_list = [{"label": shop, "data": _downsample_minmax(stores_rs[shop], TREND_DOWNSAMPLE_TARGET)}
                       for shop in shop_order_present if shop in stores_rs]

        # 该颜色的 A/B/C（对该颜色所有店的横向均值→A，再MA→B/C）
        any_seq = next(iter(stores_rs.values()), [])
        seriesA: List[Dict] = []
        for idx in range(len(any_seq)):
            x = any_seq[idx]["x"]
            ys = []
            for shop, seq in stores_rs.items():
                if shop not in sel_shops:
                    continue
                y = seq[idx].get("y") if idx < len(seq) else None
                if y is not None:
                    ys.append(float(y))
            if ys:
                seriesA.append({"x": x, "y": sum(ys)/len(ys)})
        seriesB = _moving_average_time(seriesA, b_win)
        seriesC = _moving_average_time(seriesA, c_win)

        if TREND_DOWNSAMPLE_TARGET > 0:
            seriesA = _downsample_minmax(seriesA, TREND_DOWNSAMPLE_TARGET)
            seriesB = _downsample_minmax(seriesB, TREND_DOWNSAMPLE_TARGET)
            seriesC = _downsample_minmax(seriesC, TREND_DOWNSAMPLE_TARGET)

        per_color.append({
            "color": color,
            "stores": stores_list,
            "avg": {"A": seriesA, "B": seriesB, "C": seriesC}
        })

    return {
        "shop_order_all": shop_order_all,
        "shop_order_present": shop_order_present,
        "colors": colors,
        "merged": merged,
        "per_color": per_color,
    }
