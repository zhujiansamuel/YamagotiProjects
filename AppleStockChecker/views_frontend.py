from django.views.generic import TemplateView
from django.conf import settings
import json
from django.utils.safestring import mark_safe
class StockDashboardView(TemplateView):
    template_name = "apple_stock/dashboard.html"


class StoreLatestStockView(TemplateView):
    template_name = "apple_stock/store_latest.html"

class DeliveryTrendView(TemplateView):
    template_name = "apple_stock/delivery_trend.html"

class ResaleTrendPNView(TemplateView):
    template_name = "apple_stock/resale_trend_pn_merged.html"


class ResaleTrendPNMergedView(TemplateView):
    template_name = "apple_stock/resale_trend_pn_merged.html"


class ImportResaleCSVView(TemplateView):
    template_name = "apple_stock/import_price_csv.html"


class ImportTradeinCSVView(TemplateView):
    template_name = "apple_stock/import_tradein_csv.html"

class ImportIphoneCSVView(TemplateView):
    template_name = "apple_stock/import_iphone_csv.html"

class ExternalIngestView(TemplateView):
    template_name = "apple_stock/external_ingest.html"

class PriceMatrixView(TemplateView):
    template_name = "apple_stock/price_matrix.html"

class ResaleTrendColorsMergedView(TemplateView):
    template_name = "apple_stock/model_capacity_colors_trend.html"
    def get_context_data(self, **kwargs):
            ctx = super().get_context_data(**kwargs)
            ctx["fx_api_keys"] = mark_safe(json.dumps(getattr(settings, "FX_API_KEYS", {})))
            return ctx


class TemplateChartjsView(TemplateView):
    template_name = "apple_stock/TemplateChartjs.html"