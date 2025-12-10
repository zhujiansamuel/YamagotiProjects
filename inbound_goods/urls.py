# -*- coding: utf-8 -*-
from django.urls import path
from . import views

app_name = 'inbound_goods'

urlpatterns = [
    # 主页面
    path('', views.inventory_management, name='inventory_management'),

    # API 端点
    path('api/products/', views.api_get_products, name='api_get_products'),
    path('api/inventory/create/', views.api_create_inventory, name='api_create_inventory'),
    path('api/inventory/bulk-update/', views.api_bulk_update_status, name='api_bulk_update_status'),
    path('api/inventory/<int:inventory_id>/', views.api_get_inventory_detail, name='api_get_inventory_detail'),
]
