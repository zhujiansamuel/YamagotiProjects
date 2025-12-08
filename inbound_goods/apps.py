# -*- coding: utf-8 -*-
from django.apps import AppConfig


class InboundGoodsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'inbound_goods'
    verbose_name = '入库商品管理'
