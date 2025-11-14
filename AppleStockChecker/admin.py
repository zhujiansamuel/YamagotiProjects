from django.contrib import admin
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
    FeatureSpec,
)


@admin.register(Iphone)
class IphoneAdmin(admin.ModelAdmin):
    list_display = ("part_number", "jan", "model_name", "capacity_gb", "color", "release_date")
    search_fields = ("part_number", "jan", "model_name", "color")
    list_filter = ("color", "release_date")
    ordering = ("-release_date", "model_name", "capacity_gb")


@admin.register(OfficialStore)
class OfficialStoreAdmin(admin.ModelAdmin):
    list_display = ("name", "address")
    search_fields = ("name", "address")
    ordering = ("name",)


@admin.register(InventoryRecord)
class InventoryRecordAdmin(admin.ModelAdmin):
    list_display = (
        "recorded_at",
        "store",
        "iphone",
        "has_stock",
        "estimated_arrival_earliest",
        "estimated_arrival_latest",
    )
    list_filter = ("has_stock", "store", "iphone")
    search_fields = ("store__name", "iphone__part_number", "iphone__model_name", "iphone__color")
    date_hierarchy = "recorded_at"
    ordering = ("-recorded_at",)
    autocomplete_fields = ("store", "iphone")  # 门店/机型很多时更友好


@admin.register(SecondHandShop)
class SecondHandShopAdmin(admin.ModelAdmin):
    list_display = ("name", "website", "address")
    search_fields = ("name", "address", "website")
    ordering = ("name",)


@admin.register(PurchasingShopPriceRecord)
class PurchasingShopPriceRecordAdmin(admin.ModelAdmin):
    list_display = (
        "recorded_at",
        "shop",
        "iphone",
        "price_new",
        "price_grade_a",
        "price_grade_b",
    )
    list_filter = ("shop", "iphone")
    search_fields = (
        "shop__name",
        "shop__address",
        "iphone__part_number",
        "iphone__model_name",
        "iphone__color",
    )
    date_hierarchy = "recorded_at"
    ordering = ("-recorded_at",)
    autocomplete_fields = ("shop", "iphone")


@admin.register(PurchasingShopTimeAnalysis)
class PurchasingShopTimeAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        "shop",
        "iphone",
        "New_Product_Price",
        "Timestamp_Time",
        "Record_Time",
    )
    list_filter = ("shop", "iphone","Timestamp_Time")
    search_fields = (
        "shop__name",
        "shop__address",
        "iphone__part_number",
        "iphone__model_name",
        "iphone__color",
        "Timestamp_Time"
    )
    date_hierarchy = "Timestamp_Time"
    ordering = ("-Timestamp_Time",)
    autocomplete_fields = ("shop", "iphone")

#
@admin.register(OverallBar)
class OverallBarAdmin(admin.ModelAdmin):
    list_display = (
        "bucket",
        "iphone",
        "mean",
        "median",
        "std",
        "shop_count",
        "dispersion",
        "is_final",
        "updated_at"
    )
    list_filter = ("bucket","iphone", "is_final", "updated_at")



@admin.register(FeatureSnapshot)
class FeatureSnapshotAdmin(admin.ModelAdmin):
    list_display = ("bucket", "scope", "name", "value", "version", "is_final")
    # 1) 明确使用合适的过滤器类：
    list_filter = (
        ("bucket", admin.DateFieldListFilter),          # DateTime -> DateFieldListFilter
        ("scope", admin.AllValuesFieldListFilter),      # 精确匹配 scope
        ("name", admin.AllValuesFieldListFilter),       # 'mean' / 'ema' / 'wma_linear' / ...
        ("version", admin.AllValuesFieldListFilter),    # 使用你在 FeatureSpec.slug 的版本
        "is_final",
    )
    # 2) 搜索（常用）
    search_fields = ("scope", "name", "version")
    # 3) 日期层级（支持按日跳转）
    date_hierarchy = "bucket"

    # 4) 如需修正“今天/过去7天”与 UTC 的错位，可重写 queryset 或增加一个后台时区提示
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # 如果你的 TIME_ZONE='UTC' 但想按 JST 理解 bucket 的“日”，可以在前端过滤里用绝对时间。
        # 这里不做强制转换，保留默认（Django 会按当前 TIME_ZONE 做 __date 过滤）
        return qs


@admin.register(ModelArtifact)
class ModelArtifactAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "version",
        "trained_at",
        "params_blob",
        "metrics_json",
    )
    list_filter = ("model_name", "version", "trained_at")

@admin.register(ForecastSnapshot)
class ForecastSnapshotAdmin(admin.ModelAdmin):
    list_display = (
        "bucket",
        "model_name",
        "version",
        "horizon_min",
        "yhat",
        "yhat_var",
        "is_final"
    )
    list_filter = ("bucket", "model_name", "version","is_final")


@admin.register(Cohort)
class CohortAdmin(admin.ModelAdmin):
    list_display = ("id", "slug", "title","display_index", "note")
    search_fields = ("slug", "title")

@admin.register(CohortMember)
class CohortMemberAdmin(admin.ModelAdmin):
    list_display = ("id", "cohort", "iphone", "weight")
    list_filter = ("cohort",)
    search_fields = ("cohort__slug", "iphone__part_number")

@admin.register(CohortBar)
class CohortBarAdmin(admin.ModelAdmin):
    list_display = (
        "bucket",
        "cohort",
        "mean",
        "median",
        "std",
        "n_models",
        "shop_count_agg",
        "dispersion",
        "is_final",
        "updated_at"
    )
    list_filter = ("bucket","n_models","shop_count_agg","is_final","updated_at")


@admin.register(ShopWeightProfile)
class ShopWeightProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "slug","display_index", "title")
    list_filter = ("slug","title")

@admin.register(ShopWeightItem)
class ShopWeightItemAdmin(admin.ModelAdmin):
    list_display = ("id", "profile","shop","display_index", "weight")
    list_filter = ("profile","shop","weight")

@admin.register(FeatureSpec)
class FeatureSpecAdmin(admin.ModelAdmin):
    list_display = ("id","slug","family","base_name","active","version","created_at")
    list_filter = ("family","base_name","active","version")
    search_fields = ("slug","note")