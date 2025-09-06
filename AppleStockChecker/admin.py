from django.contrib import admin
from .models import Iphone, OfficialStore, InventoryRecord


@admin.register(Iphone)
class IphoneAdmin(admin.ModelAdmin):
    list_display = ("part_number", "model_name", "capacity_gb", "color", "release_date")
    list_filter = ("color", "release_date")
    search_fields = ("part_number", "model_name", "color")
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
