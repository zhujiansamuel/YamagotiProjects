from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import Iphone, OfficialStore, InventoryRecord

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
        fields = [
            "part_number",
            "model_name",
            "capacity_gb",
            "capacity_label",
            "color",
            "release_date",
        ]
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