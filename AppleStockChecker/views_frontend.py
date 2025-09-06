from django.views.generic import TemplateView

class StockDashboardView(TemplateView):
    template_name = "apple_stock/dashboard.html"


class StoreLatestStockView(TemplateView):
    template_name = "apple_stock/store_latest.html"