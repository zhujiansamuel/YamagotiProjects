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
    **简化版**：sku 直接等于 iphone_id

    把 iPhone 实体转成 SKU 字符串（就是 iphone_id）

    Args:
        iphone_obj: iPhone 模型实例

    Returns:
        SKU 字符串（= str(iphone_id)）

    Examples:
        >>> iphone = Iphone(id=123)
        >>> sku_from_iphone(iphone)
        '123'
    """
    return str(iphone_obj.id if hasattr(iphone_obj, 'id') else iphone_obj.pk)
