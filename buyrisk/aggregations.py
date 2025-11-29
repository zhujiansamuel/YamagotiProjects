"""
时间桶算法和辅助函数
用于门店层和市场层的 30 分钟聚合
"""

from datetime import datetime


def floor_to_30m(dt):
    """
    把任意 datetime 落到所在 30 分钟桶起点（保留 tzinfo）

    Args:
        dt: aware datetime 对象

    Returns:
        对齐到 30 分钟桶起点的 datetime

    Examples:
        >>> from django.utils import timezone
        >>> dt = timezone.now()  # 2024-01-15 14:23:45
        >>> floor_to_30m(dt)     # 2024-01-15 14:00:00
        >>> dt = timezone.now()  # 2024-01-15 14:47:12
        >>> floor_to_30m(dt)     # 2024-01-15 14:30:00
    """
    # 假设 dt 为 aware datetime
    minute = (dt.minute // 30) * 30
    return dt.replace(minute=minute, second=0, microsecond=0)


def sku_from_iphone(iphone_obj):
    """
    把 iPhone 实体转成 '机型-容量-颜色' 的逻辑 SKU 字符串

    Args:
        iphone_obj: iPhone 模型实例

    Returns:
        SKU 字符串，格式：model-capacity-color

    Examples:
        >>> iphone = Iphone(model_name="iPhone 15 Pro", storage_gb=256, color="Natural Titanium")
        >>> sku_from_iphone(iphone)
        'iphone-15-pro-256g-natural-titanium'
    """
    # 尝试多种可能的字段名
    model = getattr(iphone_obj, "model", None) or \
            getattr(iphone_obj, "model_name", None) or \
            getattr(iphone_obj, "name", "unknown")

    # 容量字段
    cap = getattr(iphone_obj, "capacity_gb", None) or \
          getattr(iphone_obj, "storage_gb", None) or \
          getattr(iphone_obj, "storage", "na")

    # 颜色字段
    color = getattr(iphone_obj, "color", None) or \
            getattr(iphone_obj, "colour", "na")

    # 生成 SKU（小写，替换空格为短横线）
    sku = f"{model}-{cap}g-{color}"
    sku = sku.lower().replace(" ", "-").replace("_", "-")

    return sku
