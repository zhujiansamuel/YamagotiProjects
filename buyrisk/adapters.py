from django.apps import apps
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from typing import List, Tuple


def _get_model(model_path: str):
    """根据 'app.Model' 字符串获取模型类"""
    app_label, model_name = model_path.split(".")
    return apps.get_model(app_label, model_name)


def fetch_price_series(sku: str) -> List[Tuple[int, float]]:
    """
    返回近 N 天的 (ts_ms, bid) 序列

    **重要变更**：现在读取市场层 30m 指数（MarketIphoneAgg30m）
    而不是原始价格表，这样得到的是跨门店稳健聚合的收购价指数。

    这里的 bid = market 30m 指数的 bid_pref（优先 A，其次 B 再次 New）

    Args:
        sku: SKU 标识符

    Returns:
        列表，每个元素为 (时间戳毫秒, 市场收购价指数)
    """
    from .models import MarketIphoneAgg30m

    days = getattr(settings, "BUY_RISK_PRICE_WINDOW_DAYS", 7)
    since = timezone.now() - timedelta(days=days)

    # 直接从市场层30分钟聚合表读取
    qs = (MarketIphoneAgg30m.objects
          .filter(sku=sku, bin_start__gte=since)
          .order_by("bin_start")
          .values_list("bin_start", "bid_pref"))

    out = []
    for ts, v in qs:
        if v is None:
            # 保障序列连续：若缺失，可做 LOCF（Last Observation Carried Forward）
            # 也可直接跳过，这里选择跳过
            continue
        out.append((int(ts.timestamp() * 1000), float(v)))

    return out


def fetch_inventory_costs(sku: str) -> List[float]:
    """
    返回可售库存的成本列表。如果没有映射，则读 buyrisk.InventoryLot

    Args:
        sku: SKU 标识符

    Returns:
        成本列表
    """
    model_path = getattr(settings, "BUY_RISK_INVENTORY_MODEL", None)

    if model_path:
        Model = _get_model(model_path)
        sf = getattr(settings, "BUY_RISK_INVENTORY_SKU_FIELD", "sku")
        cf = getattr(settings, "BUY_RISK_INVENTORY_COST_FIELD", "cost")
        stf = getattr(settings, "BUY_RISK_INVENTORY_STATUS_FIELD", "status")
        ok = getattr(settings, "BUY_RISK_INVENTORY_STATUS_VALUES", ["in_stock", "ready"])

        qs = (Model.objects
              .filter(**{sf: sku, f"{stf}__in": ok})
              .values_list(cf, flat=True))
        return [float(x) for x in qs if x is not None]
    else:
        # 使用默认的 InventoryLot 模型
        from .models import InventoryLot
        qs = InventoryLot.objects.filter(
            sku=sku,
            status__in=["in_stock", "ready"]
        ).values_list("cost", flat=True)
        return [float(x) for x in qs if x is not None]


def list_skus() -> List[str]:
    """
    优先从 settings.BUY_RISK_SKUS；否则从价格表 distinct 取

    Returns:
        SKU 列表
    """
    skus = getattr(settings, "BUY_RISK_SKUS", None)
    if skus:
        return list(skus)

    model_path = getattr(settings, "BUY_RISK_PRICE_MODEL", None)
    if not model_path:
        return []

    Model = _get_model(model_path)
    sf = getattr(settings, "BUY_RISK_PRICE_SKU_FIELD", "sku")
    return list(Model.objects.values_list(sf, flat=True).distinct())
