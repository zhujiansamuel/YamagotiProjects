# -*- coding: utf-8 -*-
"""
webscraper_tasks 专用工具模块

包含仅由 webscraper_tasks 使用的：
- redis_temp_storage: Redis 临时存储（接收任务与清洗任务间传递 DataFrame）
- shop_queue_mapping: source_name → Celery 队列名映射
"""
