from django.apps import AppConfig


class ApplestockcheckerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'AppleStockChecker'
    verbose_name = "Iphone价格监控"

    def ready(self):
        # Celery worker 连接安全（fork 后关闭继承连接、任务前清理过期连接）
        from . import celery_db_connection_safety  # noqa: F401


