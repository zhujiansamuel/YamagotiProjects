from django.contrib import admin
from .models import (
    SupplyCurveParam, ShadowParam, PurchaseLog, SellProxy,
    ClearanceEvent, FxDaily, InventoryLot, DecisionRun, CoverageReport,
    ShopIphoneAgg30m, MarketIphoneAgg30m
)


@admin.register(SupplyCurveParam)
class SupplyCurveParamAdmin(admin.ModelAdmin):
    list_display = ['sku', 'b_ref', 'lambda_ref', 'b_elastic', 'trained_at']
    list_filter = ['trained_at']
    search_fields = ['sku']
    ordering = ['-trained_at']
    readonly_fields = ['trained_at']


@admin.register(ShadowParam)
class ShadowParamAdmin(admin.ModelAdmin):
    list_display = ['sku', 'alpha', 'beta', 'd_liq', 'q', 'tau_hours', 'fx_sigma_daily', 'trained_at']
    list_filter = ['trained_at']
    search_fields = ['sku']
    ordering = ['-trained_at']
    readonly_fields = ['trained_at']


@admin.register(PurchaseLog)
class PurchaseLogAdmin(admin.ModelAdmin):
    list_display = ['sku', 'ts', 'offer_price', 'acquired', 'channel', 'store']
    list_filter = ['acquired', 'channel', 'store', 'ts']
    search_fields = ['sku', 'channel', 'store']
    ordering = ['-ts']
    date_hierarchy = 'ts'


@admin.register(SellProxy)
class SellProxyAdmin(admin.ModelAdmin):
    list_display = ['sku', 'ts', 'sell_proxy', 'source']
    list_filter = ['source', 'ts']
    search_fields = ['sku', 'source']
    ordering = ['-ts']
    date_hierarchy = 'ts'


@admin.register(ClearanceEvent)
class ClearanceEventAdmin(admin.ModelAdmin):
    list_display = ['sku', 'ts', 'extra_discount', 'note']
    list_filter = ['ts']
    search_fields = ['sku', 'note']
    ordering = ['-ts']
    date_hierarchy = 'ts'


@admin.register(FxDaily)
class FxDailyAdmin(admin.ModelAdmin):
    list_display = ['ts', 'fx']
    ordering = ['-ts']
    date_hierarchy = 'ts'


@admin.register(InventoryLot)
class InventoryLotAdmin(admin.ModelAdmin):
    list_display = ['sku', 'cost', 'received_at', 'condition', 'storage', 'status']
    list_filter = ['status', 'condition', 'storage', 'received_at']
    search_fields = ['sku']
    ordering = ['-received_at']
    date_hierarchy = 'received_at'


@admin.register(DecisionRun)
class DecisionRunAdmin(admin.ModelAdmin):
    list_display = [
        'sku', 'ts_calc', 'gate', 'b_final', 's_shadow',
        'lam_final_per_hour', 'doh_days'
    ]
    list_filter = ['gate', 'ts_calc']
    search_fields = ['sku']
    ordering = ['-ts_calc']
    date_hierarchy = 'ts_calc'
    readonly_fields = [
        'ts_calc', 'created_at', 'decomposition_json', 'params_json'
    ]

    fieldsets = (
        ('基本信息', {
            'fields': ('sku', 'ts_calc', 'created_at')
        }),
        ('影子卖价', {
            'fields': ('s_shadow', 'e_b_tau')
        }),
        ('漂移与波动', {
            'fields': ('mu_step', 'sigma_step', 'mu_tau', 'sigma_tau')
        }),
        ('决策价格', {
            'fields': ('b_max', 'b_fill', 'b_final')
        }),
        ('闸门状态', {
            'fields': ('gate', 'gate_gap_ratio')
        }),
        ('供给与库存', {
            'fields': ('lam_final_per_hour', 'wac', 'unit_shadow_pnl',
                      'mvar_total', 'doh_days', 'daily_outflow')
        }),
        ('风险控制', {
            'fields': ('fx_buffer_reco', 'inv_penalty')
        }),
        ('详细数据', {
            'fields': ('decomposition_json', 'params_json'),
            'classes': ('collapse',)
        }),
    )


@admin.register(CoverageReport)
class CoverageReportAdmin(admin.ModelAdmin):
    list_display = ['sku', 'window_days', 'q_target', 'coverage_realized', 'n', 'created_at']
    list_filter = ['window_days', 'created_at']
    search_fields = ['sku']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    readonly_fields = ['created_at']


@admin.register(ShopIphoneAgg30m)
class ShopIphoneAgg30mAdmin(admin.ModelAdmin):
    list_display = [
        'shop', 'iphone', 'bin_start',
        'avg_new', 'avg_a', 'avg_b', 'rec_cnt'
    ]
    list_filter = ['bin_start', 'shop']
    search_fields = ['shop__name', 'iphone__model_name']
    ordering = ['-bin_start']
    date_hierarchy = 'bin_start'
    readonly_fields = ['updated_at', 'min_src_ts', 'max_src_ts']

    fieldsets = (
        ('基本信息', {
            'fields': ('shop', 'iphone', 'bin_start')
        }),
        ('聚合价格', {
            'fields': ('avg_new', 'avg_a', 'avg_b')
        }),
        ('统计信息', {
            'fields': ('rec_cnt', 'min_src_ts', 'max_src_ts', 'updated_at')
        }),
    )


@admin.register(MarketIphoneAgg30m)
class MarketIphoneAgg30mAdmin(admin.ModelAdmin):
    list_display = [
        'sku', 'bin_start', 'bid_pref',
        'med_a', 'med_b', 'shops_included'
    ]
    list_filter = ['bin_start']
    search_fields = ['sku', 'iphone__model_name']
    ordering = ['-bin_start']
    date_hierarchy = 'bin_start'
    readonly_fields = ['updated_at']

    fieldsets = (
        ('基本信息', {
            'fields': ('sku', 'iphone', 'bin_start', 'bid_pref')
        }),
        ('中位数', {
            'fields': ('med_new', 'med_a', 'med_b')
        }),
        ('均值', {
            'fields': ('mean_new', 'mean_a', 'mean_b')
        }),
        ('截断均值', {
            'fields': ('tmean_a', 'tmean_b')
        }),
        ('统计信息', {
            'fields': ('shops_included', 'updated_at')
        }),
    )
