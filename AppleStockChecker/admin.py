from django.contrib import admin
from .models import SecondHandShop, PurchasingShopPriceRecord, Iphone, OfficialStore, InventoryRecord,PurchasingShopTimeAnalysis


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