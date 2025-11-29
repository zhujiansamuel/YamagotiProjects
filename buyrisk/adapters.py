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
    返回近 N 天的[(ts_ms, bid), ...]，按 settings 中的映射读取你的表

    Args:
        sku: SKU 标识符

    Returns:
        列表，每个元素为 (时间戳毫秒, 收购价)
    """
    model_path = getattr(settings, "BUY_RISK_PRICE_MODEL", None)
    if not model_path:
        return []

    Model = _get_model(model_path)
    tf = getattr(settings, "BUY_RISK_PRICE_TIME_FIELD", "ts")
    vf = getattr(settings, "BUY_RISK_PRICE_VALUE_FIELD", "bid")
    sf = getattr(settings, "BUY_RISK_PRICE_SKU_FIELD", "sku")
    days = getattr(settings, "BUY_RISK_PRICE_WINDOW_DAYS", 7)

    since = timezone.now() - timedelta(days=days)
    qs = (Model.objects
          .filter(**{sf: sku, f"{tf}__gte": since})
          .order_by(tf)
          .values_list(tf, vf))

    out = []
    for t, v in qs:
        if v is None:
            continue
        # 处理可能的时区问题
        if hasattr(t, 'timestamp'):
            ts_ms = int(t.timestamp() * 1000)
        else:
            ts_ms = int(t.total_seconds() * 1000)
        out.append((ts_ms, float(v)))
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
