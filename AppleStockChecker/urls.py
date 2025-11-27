from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from .views import ApiRoot, HealthView, MeView

from rest_framework.routers import DefaultRouter
from .views import IphoneViewSet, OfficialStoreViewSet, InventoryRecordViewSet, post_to_x
from .views import SecondHandShopViewSet, PurchasingShopPriceRecordViewSet
from .views import (
    PurchasingShopTimeAnalysisViewSet,
    PurchasingShopTimeAnalysisPSTACompactViewSet,
    OverallBarViewSet,
    OverallBarPointsViewSet,
    CohortBarViewSet,
    CohortBarPointsViewSet,
    FeatureSnapshotViewSet,
    FeaturePointsViewSet,
    CohortListViewSet,
    ShopWeightProfileListViewSet,
)
from AppleStockChecker.views_options import (
    ScopeOptionsView,
    options_shops,
    options_iphones,
    options_cohorts,
    options_shop_profiles,
)
from .views_frontend import (
    PriceMatrixView,
    ResaleTrendColorsMergedView,
    TemplateChartjsView,
    AnalysisDashboardView,
    ImportTradeinCSVView,
    ImportIphoneCSVView,
    ExternalIngestView,
    ImportResaleCSVView,
    ResaleTrendPNMergedView,
    ResaleTrendPNView,
    DeliveryTrendView,
    StoreLatestStockView,
    StockDashboardView,
    StatisticalDataSummaryView,
    AutoMLView,
)
from .api_trends_TrendsAvgOnly import TrendsAvgOnlyApiView
from .api_trends_model_colors import trends_model_colors
from .api_trends_color_std import TrendsColorStdApiView
from .api import dispatch_psta_batch_same_ts
from .api_automl import (
    TriggerPreprocessingRapidView,
    TriggerCauseAndEffectTestingView,
    TriggerQuantificationOfImpactView
)

router = DefaultRouter()
router.register(r"iphones", IphoneViewSet, basename="iphone")
router.register(r"stores", OfficialStoreViewSet, basename="store")
router.register(r"inventory-records", InventoryRecordViewSet, basename="inventoryrecord")
router.register(r"secondhand-shops", SecondHandShopViewSet, basename="secondhandshop")
router.register(r"purchasing-price-records", PurchasingShopPriceRecordViewSet, basename="purchasingpricerecord")
router.register(r"purchasing-time-analyses", PurchasingShopTimeAnalysisViewSet, basename="purchasing-time-analyses")
router.register(r"purchasing-time-analyses-psta-compact", PurchasingShopTimeAnalysisPSTACompactViewSet,
                basename="purchasing-time-analyses-psta-compact")
router.register(r'overall-bars', OverallBarViewSet, basename='overallbar')
"""

"""
router.register(r'overall-bars/points', OverallBarPointsViewSet, basename='overallbar-points')
"""

"""
router.register(r'cohort-bars', CohortBarViewSet, basename='cohortbar')
"""

"""
router.register(r'cohort-bars/points', CohortBarPointsViewSet, basename='cohortbar-points')
"""

"""
router.register(r'features', FeatureSnapshotViewSet, basename='feature')
"""

"""
router.register(r'features/points', FeaturePointsViewSet, basename='feature-points')
"""

"""
router.register(r'cohorts', CohortListViewSet, basename='cohort-list')
"""
# [
#     {
#         "id": 1,
#         "slug": "iphone_17_256",
#         "title": "iPhone 17 256GB"
#     },
#     {
#         "id": 2,
#         "slug": "iphone_17_512",
#         "title": "iPhone 17 512GB"
#     },
#     {
#         "id": 3,
#         "slug": "iphone_17_pro_256",
#         "title": "iPhone 17 Pro 256GB"
#     },
#     {
#         "id": 4,
#         "slug": "iphone_17_pro_512",
#         "title": "iPhone 17 Pro 512GB"
#     },
#     {
#         "id": 5,
#         "slug": "iphone_17_pro_1024",
#         "title": "iPhone 17 Pro 1TB"
#     },
#     {
#         "id": 6,
#         "slug": "iphone_17_pro_max_256",
#         "title": "iPhone 17 Pro Max 256GB"
#     },
#     {
#         "id": 7,
#         "slug": "iphone_17_pro_max_512",
#         "title": "iPhone 17 Pro Max 512GB"
#     },
#     {
#         "id": 8,
#         "slug": "iphone_17_pro_max_1024",
#         "title": "iPhone 17 Pro Max 1TB"
#     },
#     {
#         "id": 9,
#         "slug": "iphone_17_pro_max_2048",
#         "title": "iPhone 17 Pro Max 2TB"
#     },
#     {
#         "id": 10,
#         "slug": "iphone_air_256",
#         "title": "iPhone Air 256GB"
#     },
#     {
#         "id": 11,
#         "slug": "iphone_air_512",
#         "title": "iPhone Air 512GB"
#     },
#     {
#         "id": 12,
#         "slug": "iphone_air_1024",
#         "title": "iPhone Air 1TB"
#     }
# ]
"""
router.register(r'shop-profiles', ShopWeightProfileListViewSet, basename='shop-profile-list')
"""
# [
#     {
#         "id": 1,
#         "slug": "core_store",
#         "title": "核心店铺组合"
#     },
#     {
#         "id": 2,
#         "slug": "full_store",
#         "title": "全店铺组合"
#     },
#     {
#         "id": 3,
#         "slug": "premium_store",
#         "title": "优质店铺组合"
#     },
#     {
#         "id": 4,
#         "slug": "hub_stores",
#         "title": "枢纽店铺"
#     }
# ]
"""

urlpatterns = [

    path("", ApiRoot.as_view(), name="api-root"),
    path("health/", HealthView.as_view(), name="health"),
    path("me/", MeView.as_view(), name="me"),
    # JWT
    path("auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/token/verify/", TokenVerifyView.as_view(), name="token_verify"),
]

urlpatterns = [
                  path("dashboard/", StockDashboardView.as_view(), name="stock-dashboard"),  # Apple 库存看板
                  path("store-latest/", StoreLatestStockView.as_view(), name="store-latest"),
                  # 按门店分组：所有 iPhone 最新记录（（这个是关于收货时间延迟的记录显示））
                  path("resale-trend-pn/", ResaleTrendPNView.as_view(), name="resale-trend-pn"),  # 二手店回收价历史（按 PN → 各店）
                  path("import-resale-csv/", ImportResaleCSVView.as_view(), name="import-resale-csv"),
                  # CSV 导入：二手店回收价格((好像这里没有清洗数据))
                  path("resale-trend-pn-merged/", ResaleTrendPNMergedView.as_view(), name="resale-trend-pn-merged"),
                  # 二手店回收价历史（按 PN → 各店）
                  path("delivery-trend/", DeliveryTrendView.as_view(), name="delivery-trend"),  # 送达天数趋势（按 PN → 门店）

                  path("import-tradein-csv/", ImportTradeinCSVView.as_view(), name="import-tradein-csv"),
                  # Excel 上传 & 预览：四家二手店清洗导入
                  path("import-iphone-csv/", ImportIphoneCSVView.as_view(), name="import-iphone-csv"),  # iPhone 导入（CSV）
                  path("external-ingest/", ExternalIngestView.as_view(), name="external-ingest"),  # 外部平台拉取 & 预览 & 入库
                  path("price-matrix/", PriceMatrixView.as_view(), name="price-matrix"),
                  path("resale-trend-colors-merged/", ResaleTrendColorsMergedView.as_view(),
                       name="resale-trend-colors-merged"),  # 机型 + 容量 → 各颜色新品价格趋势
                  path("template-chartjs/", TemplateChartjsView.as_view(), name="template-chartjs"),

                  path("api/trends/model-colors/", trends_model_colors, name="trends-model-colors"),
                  path("api/trends/model-color/std/", TrendsColorStdApiView.as_view(), name="trends-color-std"),
                  path("api/trends/model-colors/avg-only/", TrendsAvgOnlyApiView.as_view(), name="trends-avg-only"),

                  path("analysis-dashboard/", AnalysisDashboardView.as_view(), name="analysis-dashboard/"),
                  # 二手店价格分析（前端只显示 · 后端做分析）
                  path("purchasing-time-analyses/dispatch_ts/", dispatch_psta_batch_same_ts),
                  path('post-to-x/', post_to_x, name='post_to_x'),
                  path('statistical-data-summary/', StatisticalDataSummaryView.as_view(),
                       name='statistical-data-summary'),
                  path('automl/', AutoMLView.as_view(), name='automl'),
                  path('automl/trigger/preprocessing-rapid/', TriggerPreprocessingRapidView.as_view(),
                       name='trigger-preprocessing-rapid'),
                  path('automl/trigger/cause-and-effect-testing/', TriggerCauseAndEffectTestingView.as_view(),
                       name='trigger-cause-and-effect-testing'),
                  path('automl/trigger/quantification-of-impact/', TriggerQuantificationOfImpactView.as_view(),
                       name='trigger-quantification-of-impact'),

                  path("options/scopes/", ScopeOptionsView.as_view(), name="options-scopes"),
                  path("options/shops/", options_shops, name="options-shops"),
                  path("options/iphones/", options_iphones, name="options-iphones"),
                  path("options/cohorts/", options_cohorts, name="options-cohorts"),
                  path("options/shop-profiles/", options_shop_profiles, name="options-shop-profiles"),

              ] + urlpatterns

urlpatterns += router.urls

# https://verbless-sadistically-jayceon.ngrok-free.dev/AppleStockChecker/options/cohorts/
"""
{
    "cohorts": [
        {
            "slug": "iphone_17_256",
            "label": "iphone_17_256",
            "n_members": 5,
            "iphone_ids": [
                1,
                2,
                3,
                4,
                5
            ]
        },
        {
            "slug": "iphone_17_512",
            "label": "iphone_17_512",
            "n_members": 5,
            "iphone_ids": [
                6,
                7,
                8,
                10,
                9
            ]
        },
        {
            "slug": "iphone_17_pro_1024",
            "label": "iphone_17_pro_1024",
            "n_members": 3,
            "iphone_ids": [
                17,
                18,
                19
            ]
        },
        {
            "slug": "iphone_17_pro_256",
            "label": "iphone_17_pro_256",
            "n_members": 3,
            "iphone_ids": [
                11,
                12,
                13
            ]
        },
        {
            "slug": "iphone_17_pro_512",
            "label": "iphone_17_pro_512",
            "n_members": 3,
            "iphone_ids": [
                14,
                15,
                16
            ]
        },
        {
            "slug": "iphone_17_pro_max_1024",
            "label": "iphone_17_pro_max_1024",
            "n_members": 3,
            "iphone_ids": [
                26,
                27,
                28
            ]
        },
        {
            "slug": "iphone_17_pro_max_2048",
            "label": "iphone_17_pro_max_2048",
            "n_members": 3,
            "iphone_ids": [
                29,
                30,
                31
            ]
        },
        {
            "slug": "iphone_17_pro_max_256",
            "label": "iphone_17_pro_max_256",
            "n_members": 3,
            "iphone_ids": [
                20,
                21,
                22
            ]
        },
        {
            "slug": "iphone_17_pro_max_512",
            "label": "iphone_17_pro_max_512",
            "n_members": 3,
            "iphone_ids": [
                23,
                24,
                25
            ]
        },
        {
            "slug": "iphone_air_1024",
            "label": "iphone_air_1024",
            "n_members": 4,
            "iphone_ids": [
                40,
                41,
                42,
                43
            ]
        },
        {
            "slug": "iphone_air_256",
            "label": "iphone_air_256",
            "n_members": 4,
            "iphone_ids": [
                32,
                33,
                34,
                35
            ]
        },
        {
            "slug": "iphone_air_512",
            "label": "iphone_air_512",
            "n_members": 4,
            "iphone_ids": [
                36,
                37,
                38,
                39
            ]
        }
    ]
}
"""
# https://verbless-sadistically-jayceon.ngrok-free.dev/AppleStockChecker/options/shop-profiles/
"""
{
    "profiles": [
        {
            "slug": "core_store",
            "title": "核心店铺组合",
            "n_shops": 17,
            "items": [
                {
                    "id": 3,
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "id": 7,
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "id": 11,
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "id": 15,
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                },
                {
                    "id": 19,
                    "shop_id": 7,
                    "shop_name": "森森買取",
                    "weight": 1.0,
                    "display_index": 5
                },
                {
                    "id": 22,
                    "shop_id": 9,
                    "shop_name": "買取ルデヤ",
                    "weight": 1.0,
                    "display_index": 6
                },
                {
                    "id": 25,
                    "shop_id": 10,
                    "shop_name": "買取wiki",
                    "weight": 1.0,
                    "display_index": 7
                },
                {
                    "id": 28,
                    "shop_id": 21,
                    "shop_name": "買取ホムラ",
                    "weight": 1.0,
                    "display_index": 8
                },
                {
                    "id": 31,
                    "shop_id": 16,
                    "shop_name": "ドラゴンモバイル",
                    "weight": 1.0,
                    "display_index": 9
                },
                {
                    "id": 34,
                    "shop_id": 18,
                    "shop_name": "モバステ",
                    "weight": 1.0,
                    "display_index": 10
                },
                {
                    "id": 37,
                    "shop_id": 12,
                    "shop_name": "アキモバ",
                    "weight": 1.0,
                    "display_index": 11
                },
                {
                    "id": 40,
                    "shop_id": 17,
                    "shop_name": "トゥインクル",
                    "weight": 1.0,
                    "display_index": 12
                },
                {
                    "id": 42,
                    "shop_id": 3,
                    "shop_name": "家電市場",
                    "weight": 1.0,
                    "display_index": 13
                },
                {
                    "id": 44,
                    "shop_id": 4,
                    "shop_name": "買取楽園",
                    "weight": 1.0,
                    "display_index": 14
                },
                {
                    "id": 46,
                    "shop_id": 5,
                    "shop_name": "携帯空間",
                    "weight": 1.0,
                    "display_index": 15
                },
                {
                    "id": 48,
                    "shop_id": 11,
                    "shop_name": "ゲストモバイル",
                    "weight": 1.0,
                    "display_index": 16
                },
                {
                    "id": 50,
                    "shop_id": 15,
                    "shop_name": "毎日買取",
                    "weight": 1.0,
                    "display_index": 17
                }
            ]
        },
        {
            "slug": "full_store",
            "title": "全店铺组合",
            "n_shops": 19,
            "items": [
                {
                    "id": 4,
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "id": 8,
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "id": 12,
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "id": 16,
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                },
                {
                    "id": 20,
                    "shop_id": 7,
                    "shop_name": "森森買取",
                    "weight": 1.0,
                    "display_index": 5
                },
                {
                    "id": 23,
                    "shop_id": 9,
                    "shop_name": "買取ルデヤ",
                    "weight": 1.0,
                    "display_index": 6
                },
                {
                    "id": 26,
                    "shop_id": 10,
                    "shop_name": "買取wiki",
                    "weight": 1.0,
                    "display_index": 7
                },
                {
                    "id": 29,
                    "shop_id": 21,
                    "shop_name": "買取ホムラ",
                    "weight": 1.0,
                    "display_index": 8
                },
                {
                    "id": 32,
                    "shop_id": 16,
                    "shop_name": "ドラゴンモバイル",
                    "weight": 1.0,
                    "display_index": 9
                },
                {
                    "id": 35,
                    "shop_id": 18,
                    "shop_name": "モバステ",
                    "weight": 1.0,
                    "display_index": 10
                },
                {
                    "id": 38,
                    "shop_id": 12,
                    "shop_name": "アキモバ",
                    "weight": 1.0,
                    "display_index": 11
                },
                {
                    "id": 41,
                    "shop_id": 17,
                    "shop_name": "トゥインクル",
                    "weight": 1.0,
                    "display_index": 12
                },
                {
                    "id": 43,
                    "shop_id": 3,
                    "shop_name": "家電市場",
                    "weight": 1.0,
                    "display_index": 13
                },
                {
                    "id": 45,
                    "shop_id": 4,
                    "shop_name": "買取楽園",
                    "weight": 1.0,
                    "display_index": 14
                },
                {
                    "id": 47,
                    "shop_id": 5,
                    "shop_name": "携帯空間",
                    "weight": 1.0,
                    "display_index": 15
                },
                {
                    "id": 49,
                    "shop_id": 11,
                    "shop_name": "ゲストモバイル",
                    "weight": 1.0,
                    "display_index": 16
                },
                {
                    "id": 51,
                    "shop_id": 15,
                    "shop_name": "毎日買取",
                    "weight": 1.0,
                    "display_index": 17
                },
                {
                    "id": 52,
                    "shop_id": 2,
                    "shop_name": "買取当番",
                    "weight": 1.0,
                    "display_index": 18
                },
                {
                    "id": 53,
                    "shop_id": 6,
                    "shop_name": "買取オク",
                    "weight": 1.0,
                    "display_index": 19
                }
            ]
        },
        {
            "slug": "hub_stores",
            "title": "枢纽店铺",
            "n_shops": 12,
            "items": [
                {
                    "id": 2,
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "id": 6,
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "id": 10,
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "id": 14,
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                },
                {
                    "id": 18,
                    "shop_id": 7,
                    "shop_name": "森森買取",
                    "weight": 1.0,
                    "display_index": 5
                },
                {
                    "id": 21,
                    "shop_id": 9,
                    "shop_name": "買取ルデヤ",
                    "weight": 1.0,
                    "display_index": 6
                },
                {
                    "id": 24,
                    "shop_id": 10,
                    "shop_name": "買取wiki",
                    "weight": 1.0,
                    "display_index": 7
                },
                {
                    "id": 27,
                    "shop_id": 21,
                    "shop_name": "買取ホムラ",
                    "weight": 1.0,
                    "display_index": 8
                },
                {
                    "id": 30,
                    "shop_id": 16,
                    "shop_name": "ドラゴンモバイル",
                    "weight": 1.0,
                    "display_index": 9
                },
                {
                    "id": 33,
                    "shop_id": 18,
                    "shop_name": "モバステ",
                    "weight": 1.0,
                    "display_index": 10
                },
                {
                    "id": 36,
                    "shop_id": 12,
                    "shop_name": "アキモバ",
                    "weight": 1.0,
                    "display_index": 11
                },
                {
                    "id": 39,
                    "shop_id": 17,
                    "shop_name": "トゥインクル",
                    "weight": 1.0,
                    "display_index": 12
                }
            ]
        },
        {
            "slug": "premium_store",
            "title": "优质店铺组合",
            "n_shops": 4,
            "items": [
                {
                    "id": 1,
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "id": 5,
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "id": 9,
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "id": 13,
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                }
            ]
        }
    ]
}
"""
# https://verbless-sadistically-jayceon.ngrok-free.dev/AppleStockChecker/options/iphones/
"""
[
    {
        "id": 1,
        "part_number": "MG674J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ブラック",
        "label": "MG674J/A ｜ iPhone 17 ｜ 256GB ｜ ブラック"
    },
    {
        "id": 2,
        "part_number": "MG684J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ホワイト",
        "label": "MG684J/A ｜ iPhone 17 ｜ 256GB ｜ ホワイト"
    },
    {
        "id": 3,
        "part_number": "MG694J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ミストブルー",
        "label": "MG694J/A ｜ iPhone 17 ｜ 256GB ｜ ミストブルー"
    },
    {
        "id": 4,
        "part_number": "MG6A4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ラベンダー",
        "label": "MG6A4J/A ｜ iPhone 17 ｜ 256GB ｜ ラベンダー"
    },
    {
        "id": 5,
        "part_number": "MG6C4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "セージ",
        "label": "MG6C4J/A ｜ iPhone 17 ｜ 256GB ｜ セージ"
    },
    {
        "id": 6,
        "part_number": "MG6D4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ブラック",
        "label": "MG6D4J/A ｜ iPhone 17 ｜ 512GB ｜ ブラック"
    },
    {
        "id": 7,
        "part_number": "MG6E4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ホワイト",
        "label": "MG6E4J/A ｜ iPhone 17 ｜ 512GB ｜ ホワイト"
    },
    {
        "id": 8,
        "part_number": "MG6F4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ミストブルー",
        "label": "MG6F4J/A ｜ iPhone 17 ｜ 512GB ｜ ミストブルー"
    },
    {
        "id": 9,
        "part_number": "MG6G4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ラベンダー",
        "label": "MG6G4J/A ｜ iPhone 17 ｜ 512GB ｜ ラベンダー"
    },
    {
        "id": 10,
        "part_number": "MG6H4J/A",
        "model_name": "iPhone 17",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "セージ",
        "label": "MG6H4J/A ｜ iPhone 17 ｜ 512GB ｜ セージ"
    },
    {
        "id": 11,
        "part_number": "MG854J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "シルバー",
        "label": "MG854J/A ｜ iPhone 17 Pro ｜ 256GB ｜ シルバー"
    },
    {
        "id": 12,
        "part_number": "MG864J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "コズミックオレンジ",
        "label": "MG864J/A ｜ iPhone 17 Pro ｜ 256GB ｜ コズミックオレンジ"
    },
    {
        "id": 13,
        "part_number": "MG874J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ディープブルー",
        "label": "MG874J/A ｜ iPhone 17 Pro ｜ 256GB ｜ ディープブルー"
    },
    {
        "id": 14,
        "part_number": "MG894J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "シルバー",
        "label": "MG894J/A ｜ iPhone 17 Pro ｜ 512GB ｜ シルバー"
    },
    {
        "id": 15,
        "part_number": "MG8A4J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "コズミックオレンジ",
        "label": "MG8A4J/A ｜ iPhone 17 Pro ｜ 512GB ｜ コズミックオレンジ"
    },
    {
        "id": 16,
        "part_number": "MG8C4J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ディープブルー",
        "label": "MG8C4J/A ｜ iPhone 17 Pro ｜ 512GB ｜ ディープブルー"
    },
    {
        "id": 17,
        "part_number": "MG8D4J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "シルバー",
        "label": "MG8D4J/A ｜ iPhone 17 Pro ｜ 1024GB ｜ シルバー"
    },
    {
        "id": 18,
        "part_number": "MG8E4J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "コズミックオレンジ",
        "label": "MG8E4J/A ｜ iPhone 17 Pro ｜ 1024GB ｜ コズミックオレンジ"
    },
    {
        "id": 19,
        "part_number": "MG8F4J/A",
        "model_name": "iPhone 17 Pro",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "ディープブルー",
        "label": "MG8F4J/A ｜ iPhone 17 Pro ｜ 1024GB ｜ ディープブルー"
    },
    {
        "id": 20,
        "part_number": "MFY84J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "シルバー",
        "label": "MFY84J/A ｜ iPhone 17 Pro Max ｜ 256GB ｜ シルバー"
    },
    {
        "id": 21,
        "part_number": "MFY94J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "コズミックオレンジ",
        "label": "MFY94J/A ｜ iPhone 17 Pro Max ｜ 256GB ｜ コズミックオレンジ"
    },
    {
        "id": 22,
        "part_number": "MFYA4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ディープブルー",
        "label": "MFYA4J/A ｜ iPhone 17 Pro Max ｜ 256GB ｜ ディープブルー"
    },
    {
        "id": 23,
        "part_number": "MFYC4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "シルバー",
        "label": "MFYC4J/A ｜ iPhone 17 Pro Max ｜ 512GB ｜ シルバー"
    },
    {
        "id": 24,
        "part_number": "MFYD4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "コズミックオレンジ",
        "label": "MFYD4J/A ｜ iPhone 17 Pro Max ｜ 512GB ｜ コズミックオレンジ"
    },
    {
        "id": 25,
        "part_number": "MFYE4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ディープブルー",
        "label": "MFYE4J/A ｜ iPhone 17 Pro Max ｜ 512GB ｜ ディープブルー"
    },
    {
        "id": 26,
        "part_number": "MFYF4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "シルバー",
        "label": "MFYF4J/A ｜ iPhone 17 Pro Max ｜ 1024GB ｜ シルバー"
    },
    {
        "id": 27,
        "part_number": "MFYG4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "コズミックオレンジ",
        "label": "MFYG4J/A ｜ iPhone 17 Pro Max ｜ 1024GB ｜ コズミックオレンジ"
    },
    {
        "id": 28,
        "part_number": "MFYH4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "ディープブルー",
        "label": "MFYH4J/A ｜ iPhone 17 Pro Max ｜ 1024GB ｜ ディープブルー"
    },
    {
        "id": 29,
        "part_number": "MFYJ4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 2048,
        "capacity_label": "2048GB",
        "color": "シルバー",
        "label": "MFYJ4J/A ｜ iPhone 17 Pro Max ｜ 2048GB ｜ シルバー"
    },
    {
        "id": 30,
        "part_number": "MFYK4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 2048,
        "capacity_label": "2048GB",
        "color": "コズミックオレンジ",
        "label": "MFYK4J/A ｜ iPhone 17 Pro Max ｜ 2048GB ｜ コズミックオレンジ"
    },
    {
        "id": 31,
        "part_number": "MFYL4J/A",
        "model_name": "iPhone 17 Pro Max",
        "capacity_gb": 2048,
        "capacity_label": "2048GB",
        "color": "ディープブルー",
        "label": "MFYL4J/A ｜ iPhone 17 Pro Max ｜ 2048GB ｜ ディープブルー"
    },
    {
        "id": 32,
        "part_number": "MG274J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "スペースブラック",
        "label": "MG274J/A ｜ iPhone Air ｜ 256GB ｜ スペースブラック"
    },
    {
        "id": 33,
        "part_number": "MG284J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "クラウドホワイト",
        "label": "MG284J/A ｜ iPhone Air ｜ 256GB ｜ クラウドホワイト"
    },
    {
        "id": 34,
        "part_number": "MG294J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "ライトゴールド",
        "label": "MG294J/A ｜ iPhone Air ｜ 256GB ｜ ライトゴールド"
    },
    {
        "id": 35,
        "part_number": "MG2A4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 256,
        "capacity_label": "256GB",
        "color": "スカイブルー",
        "label": "MG2A4J/A ｜ iPhone Air ｜ 256GB ｜ スカイブルー"
    },
    {
        "id": 36,
        "part_number": "MG2C4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "スペースブラック",
        "label": "MG2C4J/A ｜ iPhone Air ｜ 512GB ｜ スペースブラック"
    },
    {
        "id": 37,
        "part_number": "MG2D4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "クラウドホワイト",
        "label": "MG2D4J/A ｜ iPhone Air ｜ 512GB ｜ クラウドホワイト"
    },
    {
        "id": 38,
        "part_number": "MG2E4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "ライトゴールド",
        "label": "MG2E4J/A ｜ iPhone Air ｜ 512GB ｜ ライトゴールド"
    },
    {
        "id": 39,
        "part_number": "MG2F4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 512,
        "capacity_label": "512GB",
        "color": "スカイブルー",
        "label": "MG2F4J/A ｜ iPhone Air ｜ 512GB ｜ スカイブルー"
    },
    {
        "id": 40,
        "part_number": "MG2G4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "スペースブラック",
        "label": "MG2G4J/A ｜ iPhone Air ｜ 1024GB ｜ スペースブラック"
    },
    {
        "id": 41,
        "part_number": "MG2H4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "クラウドホワイト",
        "label": "MG2H4J/A ｜ iPhone Air ｜ 1024GB ｜ クラウドホワイト"
    },
    {
        "id": 42,
        "part_number": "MG2J4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "ライトゴールド",
        "label": "MG2J4J/A ｜ iPhone Air ｜ 1024GB ｜ ライトゴールド"
    },
    {
        "id": 43,
        "part_number": "MG2K4J/A",
        "model_name": "iPhone Air",
        "capacity_gb": 1024,
        "capacity_label": "1024GB",
        "color": "スカイブルー",
        "label": "MG2K4J/A ｜ iPhone Air ｜ 1024GB ｜ スカイブルー"
    }
]
"""
# https://verbless-sadistically-jayceon.ngrok-free.dev/AppleStockChecker/options/scopes/
"""
{
    "shop_profiles": [
        {
            "id": 1,
            "slug": "core_store",
            "title": "核心店铺组合",
            "label": "核心店铺组合",
            "items": [
                {
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                },
                {
                    "shop_id": 7,
                    "shop_name": "森森買取",
                    "weight": 1.0,
                    "display_index": 5
                },
                {
                    "shop_id": 9,
                    "shop_name": "買取ルデヤ",
                    "weight": 1.0,
                    "display_index": 6
                },
                {
                    "shop_id": 10,
                    "shop_name": "買取wiki",
                    "weight": 1.0,
                    "display_index": 7
                },
                {
                    "shop_id": 21,
                    "shop_name": "買取ホムラ",
                    "weight": 1.0,
                    "display_index": 8
                },
                {
                    "shop_id": 16,
                    "shop_name": "ドラゴンモバイル",
                    "weight": 1.0,
                    "display_index": 9
                },
                {
                    "shop_id": 18,
                    "shop_name": "モバステ",
                    "weight": 1.0,
                    "display_index": 10
                },
                {
                    "shop_id": 12,
                    "shop_name": "アキモバ",
                    "weight": 1.0,
                    "display_index": 11
                },
                {
                    "shop_id": 17,
                    "shop_name": "トゥインクル",
                    "weight": 1.0,
                    "display_index": 12
                },
                {
                    "shop_id": 3,
                    "shop_name": "家電市場",
                    "weight": 1.0,
                    "display_index": 13
                },
                {
                    "shop_id": 4,
                    "shop_name": "買取楽園",
                    "weight": 1.0,
                    "display_index": 14
                },
                {
                    "shop_id": 5,
                    "shop_name": "携帯空間",
                    "weight": 1.0,
                    "display_index": 15
                },
                {
                    "shop_id": 11,
                    "shop_name": "ゲストモバイル",
                    "weight": 1.0,
                    "display_index": 16
                },
                {
                    "shop_id": 15,
                    "shop_name": "毎日買取",
                    "weight": 1.0,
                    "display_index": 17
                }
            ]
        },
        {
            "id": 2,
            "slug": "full_store",
            "title": "全店铺组合",
            "label": "全店铺组合",
            "items": [
                {
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                },
                {
                    "shop_id": 7,
                    "shop_name": "森森買取",
                    "weight": 1.0,
                    "display_index": 5
                },
                {
                    "shop_id": 9,
                    "shop_name": "買取ルデヤ",
                    "weight": 1.0,
                    "display_index": 6
                },
                {
                    "shop_id": 10,
                    "shop_name": "買取wiki",
                    "weight": 1.0,
                    "display_index": 7
                },
                {
                    "shop_id": 21,
                    "shop_name": "買取ホムラ",
                    "weight": 1.0,
                    "display_index": 8
                },
                {
                    "shop_id": 16,
                    "shop_name": "ドラゴンモバイル",
                    "weight": 1.0,
                    "display_index": 9
                },
                {
                    "shop_id": 18,
                    "shop_name": "モバステ",
                    "weight": 1.0,
                    "display_index": 10
                },
                {
                    "shop_id": 12,
                    "shop_name": "アキモバ",
                    "weight": 1.0,
                    "display_index": 11
                },
                {
                    "shop_id": 17,
                    "shop_name": "トゥインクル",
                    "weight": 1.0,
                    "display_index": 12
                },
                {
                    "shop_id": 3,
                    "shop_name": "家電市場",
                    "weight": 1.0,
                    "display_index": 13
                },
                {
                    "shop_id": 4,
                    "shop_name": "買取楽園",
                    "weight": 1.0,
                    "display_index": 14
                },
                {
                    "shop_id": 5,
                    "shop_name": "携帯空間",
                    "weight": 1.0,
                    "display_index": 15
                },
                {
                    "shop_id": 11,
                    "shop_name": "ゲストモバイル",
                    "weight": 1.0,
                    "display_index": 16
                },
                {
                    "shop_id": 15,
                    "shop_name": "毎日買取",
                    "weight": 1.0,
                    "display_index": 17
                },
                {
                    "shop_id": 2,
                    "shop_name": "買取当番",
                    "weight": 1.0,
                    "display_index": 18
                },
                {
                    "shop_id": 6,
                    "shop_name": "買取オク",
                    "weight": 1.0,
                    "display_index": 19
                }
            ]
        },
        {
            "id": 4,
            "slug": "hub_stores",
            "title": "枢纽店铺",
            "label": "枢纽店铺",
            "items": [
                {
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                },
                {
                    "shop_id": 7,
                    "shop_name": "森森買取",
                    "weight": 1.0,
                    "display_index": 5
                },
                {
                    "shop_id": 9,
                    "shop_name": "買取ルデヤ",
                    "weight": 1.0,
                    "display_index": 6
                },
                {
                    "shop_id": 10,
                    "shop_name": "買取wiki",
                    "weight": 1.0,
                    "display_index": 7
                },
                {
                    "shop_id": 21,
                    "shop_name": "買取ホムラ",
                    "weight": 1.0,
                    "display_index": 8
                },
                {
                    "shop_id": 16,
                    "shop_name": "ドラゴンモバイル",
                    "weight": 1.0,
                    "display_index": 9
                },
                {
                    "shop_id": 18,
                    "shop_name": "モバステ",
                    "weight": 1.0,
                    "display_index": 10
                },
                {
                    "shop_id": 12,
                    "shop_name": "アキモバ",
                    "weight": 1.0,
                    "display_index": 11
                },
                {
                    "shop_id": 17,
                    "shop_name": "トゥインクル",
                    "weight": 1.0,
                    "display_index": 12
                }
            ]
        },
        {
            "id": 3,
            "slug": "premium_store",
            "title": "优质店铺组合",
            "label": "优质店铺组合",
            "items": [
                {
                    "shop_id": 14,
                    "shop_name": "買取商店",
                    "weight": 1.0,
                    "display_index": 1
                },
                {
                    "shop_id": 1,
                    "shop_name": "海峡通信",
                    "weight": 1.0,
                    "display_index": 2
                },
                {
                    "shop_id": 8,
                    "shop_name": "買取一丁目",
                    "weight": 1.0,
                    "display_index": 3
                },
                {
                    "shop_id": 13,
                    "shop_name": "モバイルミックス",
                    "weight": 1.0,
                    "display_index": 4
                }
            ]
        }
    ],
    "cohorts": [
        {
            "id": 1,
            "slug": "iphone_17_256",
            "title": "iPhone 17 256GB",
            "label": "iPhone 17 256GB",
            "members": [
                {
                    "iphone_id": 1,
                    "part_number": "MG674J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 256,
                    "color": "ブラック",
                    "weight": 1.0,
                    "label": "iPhone 17 256GB ｜ MG674J/A ｜ ブラック"
                },
                {
                    "iphone_id": 2,
                    "part_number": "MG684J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 256,
                    "color": "ホワイト",
                    "weight": 1.0,
                    "label": "iPhone 17 256GB ｜ MG684J/A ｜ ホワイト"
                },
                {
                    "iphone_id": 3,
                    "part_number": "MG694J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 256,
                    "color": "ミストブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 256GB ｜ MG694J/A ｜ ミストブルー"
                },
                {
                    "iphone_id": 4,
                    "part_number": "MG6A4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 256,
                    "color": "ラベンダー",
                    "weight": 1.0,
                    "label": "iPhone 17 256GB ｜ MG6A4J/A ｜ ラベンダー"
                },
                {
                    "iphone_id": 5,
                    "part_number": "MG6C4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 256,
                    "color": "セージ",
                    "weight": 1.0,
                    "label": "iPhone 17 256GB ｜ MG6C4J/A ｜ セージ"
                }
            ]
        },
        {
            "id": 2,
            "slug": "iphone_17_512",
            "title": "iPhone 17 512GB",
            "label": "iPhone 17 512GB",
            "members": [
                {
                    "iphone_id": 6,
                    "part_number": "MG6D4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 512,
                    "color": "ブラック",
                    "weight": 1.0,
                    "label": "iPhone 17 512GB ｜ MG6D4J/A ｜ ブラック"
                },
                {
                    "iphone_id": 7,
                    "part_number": "MG6E4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 512,
                    "color": "ホワイト",
                    "weight": 1.0,
                    "label": "iPhone 17 512GB ｜ MG6E4J/A ｜ ホワイト"
                },
                {
                    "iphone_id": 8,
                    "part_number": "MG6F4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 512,
                    "color": "ミストブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 512GB ｜ MG6F4J/A ｜ ミストブルー"
                },
                {
                    "iphone_id": 9,
                    "part_number": "MG6G4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 512,
                    "color": "ラベンダー",
                    "weight": 1.0,
                    "label": "iPhone 17 512GB ｜ MG6G4J/A ｜ ラベンダー"
                },
                {
                    "iphone_id": 10,
                    "part_number": "MG6H4J/A",
                    "model_name": "iPhone 17",
                    "capacity_gb": 512,
                    "color": "セージ",
                    "weight": 1.0,
                    "label": "iPhone 17 512GB ｜ MG6H4J/A ｜ セージ"
                }
            ]
        },
        {
            "id": 5,
            "slug": "iphone_17_pro_1024",
            "title": "iPhone 17 Pro 1TB",
            "label": "iPhone 17 Pro 1TB",
            "members": [
                {
                    "iphone_id": 17,
                    "part_number": "MG8D4J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 1024,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 1024GB ｜ MG8D4J/A ｜ シルバー"
                },
                {
                    "iphone_id": 18,
                    "part_number": "MG8E4J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 1024,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 1024GB ｜ MG8E4J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 19,
                    "part_number": "MG8F4J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 1024,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 1024GB ｜ MG8F4J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 3,
            "slug": "iphone_17_pro_256",
            "title": "iPhone 17 Pro 256GB",
            "label": "iPhone 17 Pro 256GB",
            "members": [
                {
                    "iphone_id": 11,
                    "part_number": "MG854J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 256,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 256GB ｜ MG854J/A ｜ シルバー"
                },
                {
                    "iphone_id": 12,
                    "part_number": "MG864J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 256,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 256GB ｜ MG864J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 13,
                    "part_number": "MG874J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 256,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 256GB ｜ MG874J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 4,
            "slug": "iphone_17_pro_512",
            "title": "iPhone 17 Pro 512GB",
            "label": "iPhone 17 Pro 512GB",
            "members": [
                {
                    "iphone_id": 14,
                    "part_number": "MG894J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 512,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 512GB ｜ MG894J/A ｜ シルバー"
                },
                {
                    "iphone_id": 15,
                    "part_number": "MG8A4J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 512,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 512GB ｜ MG8A4J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 16,
                    "part_number": "MG8C4J/A",
                    "model_name": "iPhone 17 Pro",
                    "capacity_gb": 512,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro 512GB ｜ MG8C4J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 8,
            "slug": "iphone_17_pro_max_1024",
            "title": "iPhone 17 Pro Max 1TB",
            "label": "iPhone 17 Pro Max 1TB",
            "members": [
                {
                    "iphone_id": 26,
                    "part_number": "MFYF4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 1024,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 1024GB ｜ MFYF4J/A ｜ シルバー"
                },
                {
                    "iphone_id": 27,
                    "part_number": "MFYG4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 1024,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 1024GB ｜ MFYG4J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 28,
                    "part_number": "MFYH4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 1024,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 1024GB ｜ MFYH4J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 9,
            "slug": "iphone_17_pro_max_2048",
            "title": "iPhone 17 Pro Max 2TB",
            "label": "iPhone 17 Pro Max 2TB",
            "members": [
                {
                    "iphone_id": 29,
                    "part_number": "MFYJ4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 2048,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 2048GB ｜ MFYJ4J/A ｜ シルバー"
                },
                {
                    "iphone_id": 30,
                    "part_number": "MFYK4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 2048,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 2048GB ｜ MFYK4J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 31,
                    "part_number": "MFYL4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 2048,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 2048GB ｜ MFYL4J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 6,
            "slug": "iphone_17_pro_max_256",
            "title": "iPhone 17 Pro Max 256GB",
            "label": "iPhone 17 Pro Max 256GB",
            "members": [
                {
                    "iphone_id": 20,
                    "part_number": "MFY84J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 256,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 256GB ｜ MFY84J/A ｜ シルバー"
                },
                {
                    "iphone_id": 21,
                    "part_number": "MFY94J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 256,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 256GB ｜ MFY94J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 22,
                    "part_number": "MFYA4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 256,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 256GB ｜ MFYA4J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 7,
            "slug": "iphone_17_pro_max_512",
            "title": "iPhone 17 Pro Max 512GB",
            "label": "iPhone 17 Pro Max 512GB",
            "members": [
                {
                    "iphone_id": 23,
                    "part_number": "MFYC4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 512,
                    "color": "シルバー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 512GB ｜ MFYC4J/A ｜ シルバー"
                },
                {
                    "iphone_id": 24,
                    "part_number": "MFYD4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 512,
                    "color": "コズミックオレンジ",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 512GB ｜ MFYD4J/A ｜ コズミックオレンジ"
                },
                {
                    "iphone_id": 25,
                    "part_number": "MFYE4J/A",
                    "model_name": "iPhone 17 Pro Max",
                    "capacity_gb": 512,
                    "color": "ディープブルー",
                    "weight": 1.0,
                    "label": "iPhone 17 Pro Max 512GB ｜ MFYE4J/A ｜ ディープブルー"
                }
            ]
        },
        {
            "id": 12,
            "slug": "iphone_air_1024",
            "title": "iPhone Air 1TB",
            "label": "iPhone Air 1TB",
            "members": [
                {
                    "iphone_id": 40,
                    "part_number": "MG2G4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 1024,
                    "color": "スペースブラック",
                    "weight": 1.0,
                    "label": "iPhone Air 1024GB ｜ MG2G4J/A ｜ スペースブラック"
                },
                {
                    "iphone_id": 41,
                    "part_number": "MG2H4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 1024,
                    "color": "クラウドホワイト",
                    "weight": 1.0,
                    "label": "iPhone Air 1024GB ｜ MG2H4J/A ｜ クラウドホワイト"
                },
                {
                    "iphone_id": 42,
                    "part_number": "MG2J4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 1024,
                    "color": "ライトゴールド",
                    "weight": 1.0,
                    "label": "iPhone Air 1024GB ｜ MG2J4J/A ｜ ライトゴールド"
                },
                {
                    "iphone_id": 43,
                    "part_number": "MG2K4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 1024,
                    "color": "スカイブルー",
                    "weight": 1.0,
                    "label": "iPhone Air 1024GB ｜ MG2K4J/A ｜ スカイブルー"
                }
            ]
        },
        {
            "id": 10,
            "slug": "iphone_air_256",
            "title": "iPhone Air 256GB",
            "label": "iPhone Air 256GB",
            "members": [
                {
                    "iphone_id": 32,
                    "part_number": "MG274J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 256,
                    "color": "スペースブラック",
                    "weight": 1.0,
                    "label": "iPhone Air 256GB ｜ MG274J/A ｜ スペースブラック"
                },
                {
                    "iphone_id": 33,
                    "part_number": "MG284J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 256,
                    "color": "クラウドホワイト",
                    "weight": 1.0,
                    "label": "iPhone Air 256GB ｜ MG284J/A ｜ クラウドホワイト"
                },
                {
                    "iphone_id": 34,
                    "part_number": "MG294J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 256,
                    "color": "ライトゴールド",
                    "weight": 1.0,
                    "label": "iPhone Air 256GB ｜ MG294J/A ｜ ライトゴールド"
                },
                {
                    "iphone_id": 35,
                    "part_number": "MG2A4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 256,
                    "color": "スカイブルー",
                    "weight": 1.0,
                    "label": "iPhone Air 256GB ｜ MG2A4J/A ｜ スカイブルー"
                }
            ]
        },
        {
            "id": 11,
            "slug": "iphone_air_512",
            "title": "iPhone Air 512GB",
            "label": "iPhone Air 512GB",
            "members": [
                {
                    "iphone_id": 36,
                    "part_number": "MG2C4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 512,
                    "color": "スペースブラック",
                    "weight": 1.0,
                    "label": "iPhone Air 512GB ｜ MG2C4J/A ｜ スペースブラック"
                },
                {
                    "iphone_id": 37,
                    "part_number": "MG2D4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 512,
                    "color": "クラウドホワイト",
                    "weight": 1.0,
                    "label": "iPhone Air 512GB ｜ MG2D4J/A ｜ クラウドホワイト"
                },
                {
                    "iphone_id": 38,
                    "part_number": "MG2E4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 512,
                    "color": "ライトゴールド",
                    "weight": 1.0,
                    "label": "iPhone Air 512GB ｜ MG2E4J/A ｜ ライトゴールド"
                },
                {
                    "iphone_id": 39,
                    "part_number": "MG2F4J/A",
                    "model_name": "iPhone Air",
                    "capacity_gb": 512,
                    "color": "スカイブルー",
                    "weight": 1.0,
                    "label": "iPhone Air 512GB ｜ MG2F4J/A ｜ スカイブルー"
                }
            ]
        }
    ]
}
"""
#
"""

"""



