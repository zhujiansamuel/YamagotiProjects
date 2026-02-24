from __future__ import annotations
import os
import logging.config
from celery import Celery
from celery.signals import setup_logging, worker_process_init

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")
app = Celery("YamagotiProjects")
app.config_from_object("django.conf:settings", namespace="CELERY")

app.conf.task_track_started = True


@setup_logging.connect
def config_loggers(*args, **kwargs):
    """
    阻止 Celery 劫持 Django 的日志配置。

    连接此信号后，Celery 会跳过自身的日志初始化（不劫持 root logger，
    不重定向 stdout），完全使用 Django settings.LOGGING 中的配置。
    """
    from django.conf import settings
    logging.config.dictConfig(settings.LOGGING)


@worker_process_init.connect
def setup_logging_in_worker(**kwargs):
    """
    在每个 Celery worker 子进程中重新初始化日志配置。

    fork 后 TimedRotatingFileHandler 等文件处理器会失去文件描述符，
    需要重新初始化。
    """
    from django.conf import settings
    logging.config.dictConfig(settings.LOGGING)

# =============================================================================
# 队列路由配置
# =============================================================================
#
# 架构说明：
# 1. webscraper 队列：数据接收任务（解析/拉取数据 → 存 Redis → 触发清洗任务）
# 2. shop_* 队列：数据清洗任务（从 Redis 读取 → 清洗 → 写库）
#    - 清洗任务通过 apply_async(queue=...) 动态路由，不在此配置
#
# 队列列表：
# - webscraper: 数据接收（task_process_xlsx, task_process_webscraper_job, task_ingest_json_shop1）
# - shop_shop1 ~ shop_shop20: 各店铺数据清洗（task_clean_shop_data, task_clean_shop1_json）
#   - shop_shop5: 包含 shop5_1, shop5_2, shop5_3, shop5_4
#   - shop_shop6: 包含 shop6_1, shop6_2, shop6_3, shop6_4
# - default: 其他任务
# - psta_aggregation, psta_finalize: PSTA 相关任务
# - automl_*: AutoML 相关任务
#
# =============================================================================

app.conf.task_routes = {
    # -------------------------------------------------------------------------
    # 数据接收任务 → webscraper 队列
    # -------------------------------------------------------------------------

    # task_process_webscraper_job: 拉取 WebScraper 数据 → 存 Redis → 触发清洗
    "AppleStockChecker.tasks.webscraper_tasks.task_process_webscraper_job": {
        "queue": "webscraper",
        "routing_key": "webscraper.process_job",
    },

    # task_process_xlsx: 解析 xlsx/csv 文件 → 存 Redis → 触发清洗
    "AppleStockChecker.tasks.task_process_xlsx": {
        "queue": "webscraper",
        "routing_key": "webscraper.process_xlsx",
    },

    # task_ingest_json_shop1: 接收 shop1 JSON 数据 → 存 Redis → 触发清洗
    "AppleStockChecker.tasks.webscraper_tasks.task_ingest_json_shop1": {
        "queue": "webscraper",
        "routing_key": "webscraper.ingest_json",
    },

    # -------------------------------------------------------------------------
    # 数据清洗任务 → 动态路由到 shop_* 队列
    # -------------------------------------------------------------------------
    # task_clean_shop_data 和 task_clean_shop1_json 通过 apply_async(queue=...)
    # 动态路由到各店铺专用队列（shop_shop1 ~ shop_shop20），不在此静态配置。
    #
    # 示例：
    #   task_clean_shop_data.apply_async(kwargs={...}, queue="shop_shop1")
    #   task_clean_shop1_json.apply_async(kwargs={...}, queue="shop_shop1")
    # -------------------------------------------------------------------------

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
