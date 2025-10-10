# app/collectors.py
from datetime import timedelta
from django.utils import timezone
from typing import List, Dict, Any, Optional
from .models import PurchasingShopPriceRecord as Raw
# 字段示例（自行按实际替换）：
#   Raw.shop_id, Raw.iphone_id, Raw.Timestamp_Time, Raw.Record_Time,
#   Raw.Original_Record_Time_Zone, Raw.Timestamp_Time_Zone,
#   Raw.New_Product_Price, Raw.Price_A, Raw.Price_B

def collect_items_for_psta(
    *,
    window_minutes: int = 15,
    shop_ids: Optional[List[int]] = None,
    iphone_ids: Optional[List[int]] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    查询最近 N 分钟内的“原始价格记录”，组装成 upsert PSTA 所需的 payload 列表。
    注意：把下面 Raw.* 字段替换为你实际的“原表”字段。
    """
    now = timezone.now()
    since = now - timedelta(minutes=window_minutes)

    # —— 这里写你的查询 —— #
    # qs = Raw.objects.filter(Timestamp_Time__gte=since)
    # if shop_ids:
    #     qs = qs.filter(shop_id__in=shop_ids)
    # if iphone_ids:
    #     qs = qs.filter(iphone_id__in=iphone_ids)
    # qs = qs.order_by("Timestamp_Time")  # 或按需要排序

    # 这里给一个“伪代码”结构，说明应当如何生成 items：
    items: List[Dict[str, Any]] = []

    # for r in qs.iterator():
    #     items.append({
    #         "Batch_ID": None,
    #         "Job_ID": None,  # 由外层任务填写；也可直接不填
    #         "Original_Record_Time_Zone": r.Original_Record_Time_Zone or "+09:00",
    #         "Timestamp_Time_Zone": r.Timestamp_Time_Zone or "+09:00",
    #         "Record_Time": r.Record_Time,
    #         "Timestamp_Time": r.Timestamp_Time,
    #         "Alignment_Time_Difference": int((r.Record_Time - r.Timestamp_Time).total_seconds()),
    #         "shop_id": r.shop_id,
    #         "iphone_id": r.iphone_id,
    #         "New_Product_Price": r.New_Product_Price,
    #         "Price_A": r.Price_A,
    #         "Price_B": r.Price_B,
    #     })

    # —— 若没有原表可参考，这里给一个“空列表”容错 —— #
    if max_items:
        items = items[:max_items]
    return items
