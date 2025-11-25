from __future__ import annotations
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")
app = Celery("YamagotiProjects")
app.config_from_object("django.conf:settings", namespace="CELERY")

# 定义队列路由：将特定任务路由到专用队列
app.conf.task_routes = {
    # WebScraper 相关任务路由到 webscraper 队列（2个专用worker）
    "AppleStockChecker.tasks.webscraper_tasks.task_process_webscraper_job": {
        "queue": "webscraper",
        "routing_key": "webscraper.process_job",
    },
    "AppleStockChecker.tasks.webscraper_tasks.task_process_xlsx": {
        "queue": "webscraper",
        "routing_key": "webscraper.process_xlsx",
    },
    "AppleStockChecker.tasks.webscraper_tasks.task_ingest_json_shop1": {
        "queue": "webscraper",
        "routing_key": "webscraper.ingest_json",
    },
    # 其他所有任务默认路由到 default 队列
}

# 配置队列优先级
app.conf.task_queue_max_priority = 10
app.conf.task_default_priority = 5

# 配置默认队列
app.conf.task_default_queue = "default"
app.conf.task_default_exchange = "default"
app.conf.task_default_routing_key = "default"

# 连接池配置（适配 PgBouncer）
app.conf.broker_pool_limit = 10  # Redis 连接池大小
app.conf.broker_connection_retry_on_startup = True

app.autodiscover_tasks()

