from django.db import transaction
from ..models import PurchasingShopTimeAnalysis, SecondHandShop, Iphone

def upsert_purchasing_time_analysis(data: dict) -> PurchasingShopTimeAnalysis:
    """基于 (shop, iphone, Timestamp_Time) 幂等写入，并维护 Update_Count。"""
    with transaction.atomic():
        shop = SecondHandShop.objects.get(pk=data["shop_id"])
        iphone = Iphone.objects.get(pk=data["iphone_id"])

        inst = PurchasingShopTimeAnalysis.objects.select_for_update().filter(
            shop=shop, iphone=iphone, Timestamp_Time=data["Timestamp_Time"]
        ).first()

        if inst:
            # 更新字段 + 自增
            fields = [
              "Batch_ID","Job_ID","Original_Record_Time_Zone","Timestamp_Time_Zone",
              "Record_Time","Alignment_Time_Difference","New_Product_Price","Price_A","Price_B"
            ]
            for f in fields:
                if f in data:
                    setattr(inst, f, data[f])
            inst.Update_Count = (inst.Update_Count or 0) + 1
            inst.save()
        else:
            inst = PurchasingShopTimeAnalysis.objects.create(
                Batch_ID=data.get("Batch_ID"),
                Job_ID=data.get("Job_ID"),
                Original_Record_Time_Zone=data["Original_Record_Time_Zone"],
                Timestamp_Time_Zone=data["Timestamp_Time_Zone"],
                Record_Time=data["Record_Time"],
                Timestamp_Time=data["Timestamp_Time"],
                Alignment_Time_Difference=data["Alignment_Time_Difference"],
                Update_Count=0,
                shop=shop,
                iphone=iphone,
                New_Product_Price=data["New_Product_Price"],
                Price_A=data.get("Price_A"),
                Price_B=data.get("Price_B"),
            )
    return inst