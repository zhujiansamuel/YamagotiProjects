from django.views.generic import TemplateView

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