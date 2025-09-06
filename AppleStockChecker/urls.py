from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from .views import ApiRoot, HealthView, MeView


from rest_framework.routers import DefaultRouter
from .views import IphoneViewSet,OfficialStoreViewSet, InventoryRecordViewSet
from .views_frontend import StockDashboardView
from .views_frontend import StoreLatestStockView
from .views_frontend import DeliveryTrendView
router = DefaultRouter()
router.register(r"iphones", IphoneViewSet, basename="iphone")
router.register(r"stores", OfficialStoreViewSet, basename="store")
router.register(r"inventory-records", InventoryRecordViewSet, basename="inventoryrecord")
# urlpatterns = router.urls

# urlpatterns = [
#     path("", ApiRoot.as_view(), name="api-root"),
#     path("health/", HealthView.as_view(), name="health"),
#     path("me/", MeView.as_view(), name="me"),
#     # JWT
#     path("auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
#     path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
#     path("auth/token/verify/", TokenVerifyView.as_view(), name="token_verify"),
# ]
urlpatterns = [
    path("dashboard/", StockDashboardView.as_view(), name="stock-dashboard"),  # 前端展示页
]

urlpatterns = [
    path("store-latest/", StoreLatestStockView.as_view(), name="store-latest"),
] + urlpatterns
urlpatterns = [
    path("delivery-trend/", DeliveryTrendView.as_view(), name="delivery-trend"),
] + urlpatterns

urlpatterns += router.urls

