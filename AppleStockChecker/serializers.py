import re
from django.contrib.auth import get_user_model
from rest_framework import serializers
from .models import (
    SecondHandShop,
    PurchasingShopPriceRecord,
    Iphone,
    OfficialStore,
    InventoryRecord,
    PurchasingShopTimeAnalysis,
    OverallBar,
    FeatureSnapshot,
    ModelArtifact,
    ForecastSnapshot,
    Cohort,
    CohortMember,
    CohortBar,
    ShopWeightProfile,
    ShopWeightItem,
)

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
        fields = ["pk","part_number", "jan", "model_name", "capacity_gb", "capacity_label", "color", "release_date"]
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
        fields = ["pk", "name", "website", "address"]


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
            # "Alignment_Time_Difference",
        ]


# ======== 基础工具 ========
class D4DecimalField(serializers.DecimalField):
    """
    保留1位小数的Decimal输出，结合 settings.REST_FRAMEWORK["COERCE_DECIMAL_TO_STRING"]=False
    可直接输出为数字；否则为字符串。
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("max_digits", 12)
        kwargs.setdefault("decimal_places", 1)
        kwargs.setdefault("coerce_to_string", False)
        super().__init__(**kwargs)

# ======== OverallBar ========
class OverallBarSerializer(serializers.ModelSerializer):
    """
    前端友好：统一输出 iphone_id（即使模型里字段名为 iphone FK 或不存在该字段）
    """
    iphone_id = serializers.SerializerMethodField()

    mean = D4DecimalField()
    median = D4DecimalField()
    std = D4DecimalField(allow_null=True)
    dispersion = D4DecimalField(allow_null=True)

    class Meta:
        model = OverallBar
        fields = [
            "bucket",
            "iphone_id",    # 统一返回id（不存在则为null）
            "mean", "median", "std", "shop_count", "dispersion",
            "is_final", "updated_at",
        ]

    def get_iphone_id(self, obj):
        # 兼容：若模型有 iphone 外键则返回其id；没有则返回 None
        return getattr(obj, "iphone_id", None)


class OverallBarPointSerializer(serializers.ModelSerializer):
    """
    专供图表：只返回 t(时间) + v(值) + 维度 id
    """
    t = serializers.DateTimeField(source="bucket")
    v = D4DecimalField(source="mean")
    iphone_id = serializers.SerializerMethodField()

    class Meta:
        model = OverallBar
        fields = ["t", "v", "iphone_id", "shop_count", "is_final"]

    def get_iphone_id(self, obj):
        return getattr(obj, "iphone_id", None)


# ======== CohortBar ========
class CohortBarSerializer(serializers.ModelSerializer):
    cohort_id = serializers.IntegerField(source="cohort.id", read_only=True)
    cohort_slug = serializers.CharField(source="cohort.slug", read_only=True)

    mean = D4DecimalField()
    median = D4DecimalField(allow_null=True)
    std = D4DecimalField(allow_null=True)
    dispersion = D4DecimalField(allow_null=True)

    class Meta:
        model = CohortBar
        fields = [
            "bucket",
            "cohort_id", "cohort_slug",
            "mean", "median", "std", "dispersion",
            "n_models", "shop_count_agg",
            "is_final", "updated_at",
        ]


class CohortBarPointSerializer(serializers.ModelSerializer):
    t = serializers.DateTimeField(source="bucket")
    v = D4DecimalField(source="mean")
    cohort_id = serializers.IntegerField(source="cohort.id", read_only=True)
    cohort_slug = serializers.CharField(source="cohort.slug", read_only=True)

    class Meta:
        model = CohortBar
        fields = ["t", "v", "cohort_id", "cohort_slug", "n_models", "is_final"]


# ======== FeatureSnapshot（四种组合）=======
class FeatureSnapshotSerializer(serializers.ModelSerializer):
    """
    通用：用于列表/调试；scope 形如：
      shop:17|iphone:21
      shopcohort:trusted|iphone:21
      shop:17|cohort:17PM_set
      shopcohort:trusted|cohort:17PM_set
    """
    value = serializers.FloatField()

    class Meta:
        model = FeatureSnapshot
        fields = ["bucket", "scope", "name", "value", "version", "is_final"]


class FeaturePointSerializer(serializers.ModelSerializer):
    """
    图表点：t(时间) + v(值) + scope + name（比如 mean/median/std/dispersion/count）
    """
    t = serializers.DateTimeField(source="bucket")
    v = serializers.FloatField(source="value")

    class Meta:
        model = FeatureSnapshot
        fields = ["t", "v", "scope", "name", "is_final"]


# ======== 可选：Cohort、店铺组合定义 ========
# 方便前端拉组合列表
class CohortSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    slug = serializers.CharField()
    title = serializers.CharField()


class ShopWeightProfileSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    slug = serializers.CharField()
    title = serializers.CharField()


# —— 一些展示用的小工具 —— #
def _cap_label_from_gb(cap_gb) -> str:
    try:
        return f"{int(cap_gb)}GB"
    except Exception:
        return ""

def _slug_to_title(slug: str) -> str:
    if not slug:
        return ""
    # 例：iphone_17_pro_max_256 -> Iphone 17 Pro Max 256
    return slug.replace("_", " ").replace("-", " ").title()


# # ---- iPhone 选项 ----
# class IphoneOptionSerializer(serializers.ModelSerializer):
#     capacity_label = serializers.SerializerMethodField()
#     label = serializers.SerializerMethodField()
#
#     class Meta:
#         model = Iphone
#         fields = ("id", "part_number", "model_name", "capacity_gb", "capacity_label", "color", "label")
#
#     def get_capacity_label(self, obj):
#         return f"{obj.capacity_gb}GB" if obj.capacity_gb is not None else None
#
#     def get_label(self, obj):
#         parts = [obj.part_number, obj.model_name, self.get_capacity_label(obj), obj.color]
#        return " ｜ ".join([p for p in parts if p])

# ---- 单店选项 ----
class ShopOptionSerializer(serializers.ModelSerializer):
    label = serializers.SerializerMethodField()

    class Meta:
        model = SecondHandShop
        fields = ("id", "name", "label")

    def get_label(self, obj):
        return obj.name or f"Shop#{obj.id}"

# ---- 店铺组合（Profile）及条目 ----
class ShopWeightItemOptionSerializer(serializers.ModelSerializer):
    shop_name = serializers.SerializerMethodField()

    class Meta:
        model = ShopWeightItem
        fields = ("shop_id", "weight", "display_index", "shop_name")

    def get_shop_name(self, obj):
        return getattr(obj.shop, "name", None)

# ---- Cohort（iPhone 组合）及成员 ----
class CohortMemberOptionSerializer(serializers.ModelSerializer):
    iphone_label = serializers.SerializerMethodField()

    class Meta:
        model = CohortMember
        fields = ("iphone_id", "weight", "iphone_label")

    def get_iphone_label(self, obj):
        ip = obj.iphone
        if not ip:
            return None
        cap = f"{ip.capacity_gb}GB" if ip.capacity_gb is not None else None
        parts = [ip.part_number, ip.model_name, cap, ip.color]
        return " ｜ ".join([p for p in parts if p])

# ===== iPhone 下拉项 =====
class IphoneOptionSerializer(serializers.ModelSerializer):
    label = serializers.SerializerMethodField()

    class Meta:
        model = Iphone
        fields = ("id", "part_number", "model_name", "capacity_gb", "color", "label")

    def get_label(self, obj: Iphone) -> str:
        # 例：iPhone 17 Pro Max 256GB ｜ MG864J/A ｜ Natural
        model = obj.model_name or ""
        cap   = _cap_label_from_gb(obj.capacity_gb)
        pn    = obj.part_number or ""
        color = obj.color or ""
        pieces = [p for p in [model, cap] if p]
        left = " ".join(pieces) if pieces else pn
        if pn and pn not in left:
            left = f"{left} ｜ {pn}"
        if color:
            left = f"{left} ｜ {color}"
        return left or f"#{obj.id}"

# ===== ShopWeightProfile.items 的嵌套项 =====
class ShopWeightItemInlineSerializer(serializers.ModelSerializer):
    shop_id    = serializers.IntegerField(source="shop.id", read_only=True)
    shop_name  = serializers.CharField(source="shop.name", read_only=True)

    class Meta:
        model = ShopWeightItem
        fields = ("shop_id", "shop_name", "weight", "display_index")

# ===== ShopWeightProfile 顶层（带 items[]） =====
class ShopWeightProfileOptionSerializer(serializers.ModelSerializer):
    label = serializers.SerializerMethodField()
    items = ShopWeightItemInlineSerializer(many=True)

    class Meta:
        model  = ShopWeightProfile
        fields = ("id", "slug", "title", "label", "items")

    def get_label(self, obj: ShopWeightProfile) -> str:
        return (obj.title or obj.slug)

# ===== Cohort.members 的嵌套项 =====
class CohortMemberInlineSerializer(serializers.ModelSerializer):
    iphone_id    = serializers.IntegerField(source="iphone.id", read_only=True)
    part_number  = serializers.CharField(source="iphone.part_number", read_only=True)
    model_name   = serializers.CharField(source="iphone.model_name", read_only=True)
    capacity_gb  = serializers.IntegerField(source="iphone.capacity_gb", read_only=True)
    color        = serializers.CharField(source="iphone.color", read_only=True)
    label        = serializers.SerializerMethodField()

    class Meta:
        model  = CohortMember
        fields = ("iphone_id", "part_number", "model_name", "capacity_gb", "color", "weight", "label")

    def get_label(self, obj: CohortMember) -> str:
        cap = _cap_label_from_gb(getattr(obj.iphone, "capacity_gb", None))
        pn  = getattr(obj.iphone, "part_number", "") or ""
        clr = getattr(obj.iphone, "color", "") or ""
        base = " ".join([p for p in [obj.iphone.model_name, cap] if p])
        if pn:
            base = f"{base} ｜ {pn}"
        if clr:
            base = f"{base} ｜ {clr}"
        return base.strip() or f"iphone:{obj.iphone_id}"

# ===== Cohort 顶层（带 members[]） =====
class CohortOptionSerializer(serializers.ModelSerializer):
    # 有些项目里 Cohort 可能没有 title 字段，这里统一兜底
    title = serializers.SerializerMethodField()
    label = serializers.SerializerMethodField()
    members = CohortMemberInlineSerializer(many=True)

    class Meta:
        model  = Cohort
        fields = ("id", "slug", "title", "label", "members")

    def get_title(self, obj: Cohort) -> str:
        return getattr(obj, "title", None) or _slug_to_title(obj.slug)

    def get_label(self, obj: Cohort) -> str:
        # 例：iPhone 17 Pro Max 256GB（根据你的 title 或 slug 生成）
        return self.get_title(obj)





