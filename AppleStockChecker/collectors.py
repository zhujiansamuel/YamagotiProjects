# app/collectors.py
from __future__ import annotations
import json
from pathlib import Path
from django.conf import settings
from typing import List, Dict, Any, Optional, Tuple
from django.db.models import QuerySet
from AppleStockChecker.models import PurchasingShopPriceRecord
from datetime import timedelta
from django.utils import timezone
from typing import List, Dict, Any, Optional
from AppleStockChecker.utils.timebox import nearest_past_minute_iso


def _floor_to_minute(dt):
    """把 aware datetime 截断到分钟（秒/微秒=0）。"""
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt.astimezone(timezone.get_current_timezone()).replace(second=0, microsecond=0)


def _iso(dt) -> str:
    return dt.astimezone(timezone.get_current_timezone()).isoformat(timespec="seconds")


def _compact_row(obj) -> Dict[str, Any]:
    """
    将 PurchasingShopPriceRecord 实体压缩成轻量 dict。
    价格字段名在不同版本可能不同，这里做容错提取。
    """
    # 价格字段容错提取
    def _pick(*names, default=None):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return default

    return {
        "id": getattr(obj, "id", None),
        "shop_id": getattr(obj, "shop_id", None),
        "iphone_id": getattr(obj, "iphone_id", None),
        "recorded_at": _iso(getattr(obj, "recorded_at")),
        "price_new": getattr(obj, "price_new", None),
    }



def collect_items_for_psta(
    *,
    window_minutes: int = 15,
    timestamp_iso: Optional[str] = None,
    shop_ids: Optional[List[int]] = None,
    iphone_ids: Optional[List[int]] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ts_iso_final = timestamp_iso or nearest_past_minute_iso()
    ts_dt = timezone.make_aware(timezone.datetime.fromisoformat(ts_iso_final)) \
        if timezone.is_naive(timezone.datetime.fromisoformat(ts_iso_final)) \
        else timezone.datetime.fromisoformat(ts_iso_final)


    # 2) 向前生成 15 个整分钟刻度（含 ts_dt 自身，共 15 个）
    ticks_dt: List = []
    cur = ts_dt
    for _ in range(15):
        ticks_dt.append(_floor_to_minute(cur))
        cur = cur - timedelta(minutes=1)
    # 保持从新到旧
    ticks_iso: List[str] = [_iso(x) for x in ticks_dt]


    # 3) 查询窗口：从最旧刻度到 ts_dt（含边界）
    window_start = ticks_dt[-1]  # 第 15 个（最旧）
    window_end = ticks_dt[0]  # ts_dt（最新）

    qs: QuerySet = PurchasingShopPriceRecord.objects.filter(
        recorded_at__gte=window_start,
        recorded_at__lte=window_end,
    )
    if shop_ids:
        qs = qs.filter(shop_id__in=shop_ids)
    if iphone_ids:
        qs = qs.filter(iphone_id__in=iphone_ids)

    # 可按时间升序拿出，便于后面直接构建索引
    qs = qs.select_related("shop", "iphone").order_by("recorded_at")

    # 4) 取出对象并压缩为轻量 rows
    rows: List[Dict[str, Any]] = []
    for obj in qs.iterator():
        rows.append(_compact_row(obj))
        if max_items and len(rows) >= max_items:
            break

    # 5) 建立两个索引：index_by_key（(shop,iphone)→升序时间索引） 与 bucket_by_minute（分钟桶）
    index_by_key: Dict[str, Dict[str, List]] = {}
    bucket_by_minute: Dict[str, List[int]] = {}

    for idx, r in enumerate(rows):
        sid = r.get("shop_id")
        iid = r.get("iphone_id")
        rec_iso = r.get("recorded_at")
        if sid is None or iid is None or not rec_iso:
            continue
        try:
            rec_dt = timezone.datetime.fromisoformat(rec_iso)
            if timezone.is_naive(rec_dt):
                rec_dt = timezone.make_aware(rec_dt, timezone.get_current_timezone())
        except Exception:
            continue

        minute_iso = _iso(_floor_to_minute(rec_dt))
        if minute_iso not in bucket_by_minute:
            # 不在 15 分钟窗口内的行，跳过
            continue

        # —— 维度索引 —— #
        key = f"{sid}:{iid}"
        buf = index_by_key.setdefault(key, {"order": [], "times": []})
        buf["order"].append(idx)
        buf["times"].append(rec_iso)

        # —— 全局分钟桶 —— #
        bucket_by_minute[minute_iso].append(idx)


    # 确保每个 key 的 order 与 times 已按时间升序（qs 已 order_by，可省；此处稳妥再排一次）
    for key, buf in index_by_key.items():
        order = buf["order"]
        times = buf["times"]
        new_price = buf["new_price"]
        # 根据 times 进行稳定排序
        paired: List[Tuple[str, int]] = list(zip(times, order,new_price))
        paired.sort(key=lambda x: x[0])
        buf["times"] = [p[0] for p in paired]
        buf["order"] = [p[1] for p in paired]
        buf["new_price"] = [p[2] for p in paired]

    # 6) minute × key 的桶：minute_iso -> { "shop:iphone": [row_idx, ...], ... }
    bucket_minute_key: Dict[str, Dict[str, List[int]]] = {}

    for idx, r in enumerate(rows):
        sid = r.get("shop_id")
        iid = r.get("iphone_id")
        rec_iso = r.get("recorded_at")
        if sid is None or iid is None or not rec_iso:
            continue
        try:
            rec_dt = timezone.datetime.fromisoformat(rec_iso)
            if timezone.is_naive(rec_dt):
                rec_dt = timezone.make_aware(rec_dt, timezone.get_current_timezone())
        except Exception:
            continue

        minute_iso = _iso(_floor_to_minute(rec_dt))
        key = f"{sid}:{iid}"
        bucket_minute_key.setdefault(minute_iso, {}).setdefault(key, []).append(idx)

    # 7) 组织返回（在原有 result 的基础上，新增 "bucket_minute_key" 字段）
    result = {
        "ts_iso": ts_iso_final,
        "ticks": ticks_iso,  # 15 个刻度（新→旧）
        "rows": rows,  # 轻量原始行
        "index_by_key": index_by_key,  # (shop:iphone) → {"order":[idx...], "times":[iso...]}
        "bucket_by_minute": bucket_by_minute,  # minute_iso → [idx...]
        "bucket_minute_key": bucket_minute_key,  # ★ 新增：minute_iso → {"shop:iphone":[idx,...]}
        "window": {
            "start": _iso(window_start),
            "end": _iso(window_end),
        },
    }
    return [result]

    # # 2) 选择保存位置：项目根目录 /AppleStockChecker/_debug/ 下
    # base_dir = Path(getattr(settings, "BASE_DIR", "."))  # Django 项目根
    # out_dir = base_dir / "AppleStockChecker" / "_debug"
    # out_dir.mkdir(parents=True, exist_ok=True)
    #
    # # 3) 生成文件名（含时间戳）
    # from datetime import datetime
    # fname = f"collect_psta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # out_path = out_dir / fname
    #
    # # 4) 保存为 JSON（UTF-8，漂亮缩进）
    # with out_path.open("w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=2)
    #
    # print(f"已保存: {out_path}")

