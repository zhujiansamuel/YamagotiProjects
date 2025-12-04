from django.contrib import admin
from django.contrib.contenttypes.admin import GenericTabularInline
from .models import InboundInventory, InventoryStatusHistory


class InventoryStatusHistoryInline(admin.TabularInline):
    """¶†ò…T>:"""
    model = InventoryStatusHistory
    extra = 0
    can_delete = False
    readonly_fields = (
        'old_status', 'new_status', 'changed_at',
        'change_reason', 'changed_by'
    )
    fields = (
        'changed_at', 'old_status', 'new_status',
        'change_reason', 'changed_by'
    )
    ordering = ['-changed_at']

    def has_add_permission(self, request, obj=None):
        """bK¨û †ò°U"""
        return False


@admin.register(InboundInventory)
class InboundInventoryAdmin(admin.ModelAdmin):
    """e“FÁ¡"""

    list_display = (
        'unique_code',
        'get_product_display',
        'status',
        'reserved_arrival_time',
        'created_at',
        'updated_at',
    )

    list_filter = (
        'status',
        'product_content_type',
        'created_at',
        'reserved_arrival_time',
    )

    search_fields = (
        'unique_code',
        'special_description',
        'abnormal_remark',
    )

    readonly_fields = (
        'created_at',
        'updated_at',
    )

    fieldsets = (
        ('ú,áo', {
            'fields': (
                'unique_code',
                ('product_content_type', 'product_object_id'),
                'status',
            )
        }),
        ('æÆáo', {
            'fields': (
                'special_description',
                'reserved_arrival_time',
                'abnormal_remark',
            )
        }),
        ('öôáo', {
            'fields': (
                'created_at',
                'updated_at',
            ),
            'classes': ('collapse',),
        }),
    )

    inlines = [InventoryStatusHistoryInline]

    date_hierarchy = 'created_at'

    def get_product_display(self, obj):
        """>:sT„FÁ"""
        if obj.product:
            return str(obj.product)
        return "-"
    get_product_display.short_description = "sTFÁ"

    def get_readonly_fields(self, request, obj=None):
        """‘öĞ›Wµêû"""
        readonly = list(super().get_readonly_fields(request, obj))
        if obj:  # ‘°	ùaö
            readonly.extend(['unique_code'])
        return readonly


@admin.register(InventoryStatusHistory)
class InventoryStatusHistoryAdmin(admin.ModelAdmin):
    """¶Øô†ò¡êû	"""

    list_display = (
        'get_unique_code',
        'old_status',
        'new_status',
        'changed_at',
        'change_reason',
        'changed_by',
    )

    list_filter = (
        'old_status',
        'new_status',
        'changed_at',
    )

    search_fields = (
        'inventory__unique_code',
        'change_reason',
        'changed_by',
    )

    readonly_fields = (
        'inventory',
        'old_status',
        'new_status',
        'changed_at',
        'change_reason',
        'changed_by',
    )

    date_hierarchy = 'changed_at'

    def get_unique_code(self, obj):
        """>:FÁ"""
        return obj.inventory.unique_code
    get_unique_code.short_description = "FÁ"
    get_unique_code.admin_order_field = 'inventory__unique_code'

    def has_add_permission(self, request):
        """bK¨û †ò°U"""
        return False

    def has_delete_permission(self, request, obj=None):
        """b d†ò°U"""
        return False
