import re
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Iphone, OfficialStore, InventoryRecord
from .models import SecondHandShop, PurchasingShopPriceRecord
from .models import PurchasingShopTimeAnalysis
User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id", "username", "email",
            "first_name", "last_name",
            "is_staff", "date_joined", "last_login",
        ]
        read_only_fields = fields


class IphoneSerializer(serializers.ModelSerializer):
    capacity_label = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Iphone
        fields = ["part_number", "jan", "model_name", "capacity_gb", "capacity_label", "color", "release_date"]
        read_only_fields = ["capacity_label"]

    def get_capacity_label(self, obj):
        return f"{obj.capacity_gb // 1024}TB" if obj.capacity_gb % 1024 == 0 else f"{obj.capacity_gb}GB"

    def validate(self, attrs):
        """在数据库唯一约束之外，提前友好报错：同 型号+容量+颜色 不可重复"""
        model_name = attrs.get("model_name") or getattr(self.instance, "model_name", None)
        capacity_gb = attrs.get("capacity_gb") or getattr(self.instance, "capacity_gb", None)
        color = attrs.get("color") or getattr(self.instance, "color", None)

        if model_name and capacity_gb and color:
            qs = Iphone.objects.filter(model_name=model_name, capacity_gb=capacity_gb, color=color)
            if self.instance:
                qs = qs.exclude(pk=self.instance.pk)
            if qs.exists():
                raise serializers.ValidationError("已存在相同『型号/容量/颜色』的 iPhone 记录。")
        return attrs

    def validate_jan(self, v):
        if v in (None, ""):
            return None
        s = re.sub(r"\D", "", str(v))
        if len(s) != 13:
            raise serializers.ValidationError("JAN 必须是 13 位数字")
        return s

class OfficialStoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = OfficialStore
        fields = ["id", "name", "address"]

class InventoryRecordSerializer(serializers.ModelSerializer):
    # 便于前端显示的只读衍生字段
    store_name = serializers.CharField(source="store.name", read_only=True)
    iphone_part_number = serializers.CharField(source="iphone.part_number", read_only=True)
    iphone_label = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = InventoryRecord
        fields = [
            "id",
            "store", "store_name",
            "iphone", "iphone_part_number", "iphone_label",
            "has_stock",
            "estimated_arrival_earliest",
            "estimated_arrival_latest",
            "recorded_at",
        ]
        read_only_fields = ["store_name", "iphone_part_number", "iphone_label", "recorded_at"]

    def get_iphone_label(self, obj):
        cap = f"{obj.iphone.capacity_gb // 1024}TB" if obj.iphone.capacity_gb % 1024 == 0 else f"{obj.iphone.capacity_gb}GB"
        return f"{obj.iphone.model_name} {cap} {obj.iphone.color}"

    def validate(self, attrs):
        e = attrs.get("estimated_arrival_earliest", getattr(self.instance, "estimated_arrival_earliest", None))
        l = attrs.get("estimated_arrival_latest", getattr(self.instance, "estimated_arrival_latest", None))
        if e and l and e > l:
            raise serializers.ValidationError("配送到达最早时间不能晚于最晚时间。")
        return attrs


class TrendStoreSeriesSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    address = serializers.CharField(allow_blank=True, allow_null=True)
    dates = serializers.ListField(child=serializers.DateField(format="%Y-%m-%d"))
    earliest = serializers.ListField(child=serializers.IntegerField())
    median = serializers.ListField(child=serializers.IntegerField())
    latest = serializers.ListField(child=serializers.IntegerField())

class BasicIphoneInfoSerializer(serializers.Serializer):
    part_number = serializers.CharField()
    model_name = serializers.CharField()
    capacity_gb = serializers.IntegerField()
    color = serializers.CharField()
    release_date = serializers.DateField()
    capacity_label = serializers.SerializerMethodField()

    def get_capacity_label(self, obj):
        gb = obj.get("capacity_gb")
        return f"{gb // 1024}TB" if gb and gb % 1024 == 0 else f"{gb}GB"

class TrendResponseByPNSerializer(serializers.Serializer):
    part_number = serializers.CharField()
    iphone = BasicIphoneInfoSerializer(required=False)
    recorded_after = serializers.DateTimeField(allow_null=True, required=False)
    recorded_before = serializers.DateTimeField(allow_null=True, required=False)
    stores = TrendStoreSeriesSerializer(many=True)




class SecondHandShopSerializer(serializers.ModelSerializer):
    class Meta:
        model = SecondHandShop
        fields = ["id", "name", "website", "address"]


class PurchasingShopPriceRecordSerializer(serializers.ModelSerializer):
    # 便于前端展示的只读字段
    shop_name = serializers.CharField(source="shop.name", read_only=True)
    iphone_part_number = serializers.CharField(source="iphone.part_number", read_only=True)
    iphone_label = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = PurchasingShopPriceRecord
        fields = [
            "id",
            "shop", "shop_name",
            "iphone", "iphone_part_number", "iphone_label",
            "price_new", "price_grade_a", "price_grade_b",
            "recorded_at",
        ]
        read_only_fields = ["shop_name", "iphone_part_number", "iphone_label", "recorded_at"]

    def get_iphone_label(self, obj):
        cap = f"{obj.iphone.capacity_gb // 1024}TB" if obj.iphone.capacity_gb % 1024 == 0 else f"{obj.iphone.capacity_gb}GB"
        return f"{obj.iphone.model_name} {cap} {obj.iphone.color}"

    def validate_price_new(self, v):
        if v is None or v <= 0:
            raise serializers.ValidationError("新品卖取价格必须为正整数。")
        return v

    def validate(self, attrs):
        for fld in ("price_grade_a", "price_grade_b"):
            val = attrs.get(fld)
            if val is not None and val <= 0:
                raise serializers.ValidationError({fld: "必须为正整数或留空。"})
        return attrs


class PurchasingShopTimeAnalysisSerializer(serializers.ModelSerializer):
    # 写入用：外键 id
    shop_id = serializers.PrimaryKeyRelatedField(
        source="shop", queryset=SecondHandShop.objects.all(), write_only=True
    )
    iphone_id = serializers.PrimaryKeyRelatedField(
        source="iphone", queryset=Iphone.objects.all(), write_only=True
    )

    # 只读：店铺快照（给前端直接显示）
    shop = serializers.SerializerMethodField(read_only=True)
    # 只读：iPhone 关键规格（给前端直接显示/筛选标签）
    iphone = serializers.SerializerMethodField(read_only=True)

    id = serializers.IntegerField(read_only=True)
    created_at = serializers.DateTimeField(source="Warehouse_Receipt_Time", read_only=True)

    class Meta:
        model = PurchasingShopTimeAnalysis
        fields = [
            "id",
            "Batch_ID",
            "Job_ID",
            "Original_Record_Time_Zone",
            "Timestamp_Time_Zone",
            "Record_Time",
            "Timestamp_Time",
            "Alignment_Time_Difference",
            "Update_Count",
            "New_Product_Price",
            "Price_A",
            "Price_B",
            "created_at",
            # 外键：写入 id、读取展开对象
            "shop_id", "iphone_id",
            "shop", "iphone",
        ]
        read_only_fields = ["Update_Count", "created_at", "shop", "iphone"]

    # —— 展开读取用的快照字段 —— #
    def get_shop(self, obj):
        s = obj.shop
        return {
            "id": s.id,
            "name": s.name,
            "website": s.website,
            "address": s.address,
        }

    def get_iphone(self, obj):
        p = obj.iphone
        return {
            "id": p.id,
            "part_number": p.part_number,   # 唯一编码
            "jan": p.jan,
            "model_name": p.model_name,
            "capacity_gb": p.capacity_gb,
            "color": p.color,
            "release_date": p.release_date,
            # 便于前端直接显示 “256GB/1TB”
            "capacity_label": (f"{p.capacity_gb // 1024}TB"
                               if p.capacity_gb % 1024 == 0 else f"{p.capacity_gb}GB"),
            "label": f"{p.model_name} {p.color}",
        }

    # —— 可选一致性校验（与你之前思路一致） —— #
    def validate(self, attrs):
        rt = attrs.get("Record_Time")
        ts = attrs.get("Timestamp_Time")
        diff = attrs.get("Alignment_Time_Difference")
        if rt and ts and diff is not None:
            actual = int((rt - ts).total_seconds())
            if abs(actual - diff) > 1:
                raise serializers.ValidationError(
                    {"Alignment_Time_Difference": f"应与 Record_Time - Timestamp_Time 的秒差匹配，实际为 {actual}。"}
                )
        return attrs


class PSTACompactSerializer(serializers.ModelSerializer):
    # 注意：只发前端立刻要用的关键字段，避免消息太大
    shop = serializers.CharField(source="shop.name", read_only=True)
    iphone = serializers.CharField(source="iphone.part_number", read_only=True)

    class Meta:
        model = PurchasingShopTimeAnalysis
        fields = [
            "id",
            "Timestamp_Time",           # 同一批的共同时间戳
            "shop", "iphone",           # 读友好
            "shop_id", "iphone_id",     # 写/查友好
            "New_Product_Price",
            "Alignment_Time_Difference",
        ]