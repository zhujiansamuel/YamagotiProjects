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
from .views_frontend import PriceMatrixView
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
              ] + urlpatterns
urlpatterns = [
                  path("delivery-trend/", DeliveryTrendView.as_view(), name="delivery-trend"),
              ] + urlpatterns
urlpatterns = [ path("resale-trend-pn/", ResaleTrendPNView.as_view(), name="resale-trend-pn"), ] + urlpatterns
urlpatterns = ([path("resale-trend-pn-merged/", ResaleTrendPNMergedView.as_view(), name="resale-trend-pn-merged"),
                ] + urlpatterns)

urlpatterns = [
                  path("import-resale-csv/", ImportResaleCSVView.as_view(), name="import-resale-csv"),
              ] + urlpatterns
urlpatterns = [
                  path("import-tradein-csv/", ImportTradeinCSVView.as_view(), name="import-tradein-csv"),
                  path("import-iphone-csv/", ImportIphoneCSVView.as_view(), name="import-iphone-csv"),
path("external-ingest/", ExternalIngestView.as_view(), name="external-ingest"),
path("price-matrix/", PriceMatrixView.as_view(), name="price-matrix"),
              ] + urlpatterns

urlpatterns += router.urls
