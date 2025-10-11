from django.apps import AppConfig


class ApplestockcheckerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'AppleStockChecker'
    verbose_name = "Iphone价格监控"

    def ready(self):
        # 激活 SQLite PRAGMA 与 Celery 连接防腐
        from . import db_sqlite_tweaks  # noqa: F401
        from . import celery_sqlite_safety  # noqa: F401


