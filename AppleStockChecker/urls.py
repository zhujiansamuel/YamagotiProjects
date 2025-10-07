from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from .views import ApiRoot, HealthView, MeView

from rest_framework.routers import DefaultRouter
from .views import IphoneViewSet, OfficialStoreViewSet, InventoryRecordViewSet
from .views import SecondHandShopViewSet, PurchasingShopPriceRecordViewSet
from .views_frontend import StockDashboardView
from .views_frontend import StoreLatestStockView
from .views_frontend import DeliveryTrendView
from .views_frontend import ResaleTrendPNView
from .views_frontend import ResaleTrendPNMergedView
from .views_frontend import ImportResaleCSVView
from .views_frontend import ImportTradeinCSVView, ImportIphoneCSVView, ExternalIngestView
from .views_frontend import PriceMatrixView, ResaleTrendColorsMergedView, TemplateChartjsView, AnalysisDashboardView
from .api_trends_TrendsAvgOnly import TrendsAvgOnlyApiView
from .api_trends_model_colors import trends_model_colors
from .api_trends_color_std import TrendsColorStdApiView

router = DefaultRouter()
router.register(r"iphones", IphoneViewSet, basename="iphone")
router.register(r"stores", OfficialStoreViewSet, basename="store")
router.register(r"inventory-records", InventoryRecordViewSet, basename="inventoryrecord")
router.register(r"secondhand-shops", SecondHandShopViewSet, basename="secondhandshop")
router.register(r"purchasing-price-records", PurchasingShopPriceRecordViewSet, basename="purchasingpricerecord")

urlpatterns = [
    path("dashboard/", StockDashboardView.as_view(), name="stock-dashboard"),  # 前端展示页
    path("", ApiRoot.as_view(), name="api-root"),
    path("health/", HealthView.as_view(), name="health"),
    path("me/", MeView.as_view(), name="me"),
    # JWT
    path("auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/token/verify/", TokenVerifyView.as_view(), name="token_verify"),
]

urlpatterns = [
                  path("store-latest/", StoreLatestStockView.as_view(), name="store-latest"),
                  path("resale-trend-pn/", ResaleTrendPNView.as_view(), name="resale-trend-pn"),
                  path("import-resale-csv/", ImportResaleCSVView.as_view(), name="import-resale-csv"),
                  path("resale-trend-pn-merged/", ResaleTrendPNMergedView.as_view(), name="resale-trend-pn-merged"),
                  path("delivery-trend/", DeliveryTrendView.as_view(), name="delivery-trend"),

                  path("import-tradein-csv/", ImportTradeinCSVView.as_view(), name="import-tradein-csv"),
                  path("import-iphone-csv/", ImportIphoneCSVView.as_view(), name="import-iphone-csv"),
                  path("external-ingest/", ExternalIngestView.as_view(), name="external-ingest"),
                  path("price-matrix/", PriceMatrixView.as_view(), name="price-matrix"),
                  path("resale-trend-colors-merged/", ResaleTrendColorsMergedView.as_view(),
                       name="resale-trend-colors-merged"),
                  path("template-chartjs/", TemplateChartjsView.as_view(), name="template-chartjs"),
                  path("api/trends/model-colors/", trends_model_colors, name="trends-model-colors"),
                  path("api/trends/model-color/std/", TrendsColorStdApiView.as_view(), name="trends-color-std"),
                  path("api/trends/model-colors/avg-only/", TrendsAvgOnlyApiView.as_view(), name="trends-avg-only"),
                  path("analysis-dashboard/", AnalysisDashboardView.as_view(), name="analysis-dashboard/"),

              ] + urlpatterns

urlpatterns += router.urls
