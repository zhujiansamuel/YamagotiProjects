# 文件：AppleStockChecker/db_sqlite_tweaks.py
# 作用：在 SQLite 下启用 WAL 模式并增加 busy_timeout，缓解 “database is locked”。
# 用法：创建本文件后，在 AppConfig.ready() 中 import 一次（见下方注释）。

from django.db.backends.signals import connection_created
from django.dispatch import receiver

@receiver(connection_created)
def set_sqlite_pragmas(sender, connection, **kwargs):
    """
    对 SQLite 连接设置更适合并发写入的 PRAGMA：
      - journal_mode=WAL：写入与读取可以并发进行
      - busy_timeout=5000ms：遇到锁时等待一会儿而不是立刻报错
      - synchronous=NORMAL：配合 WAL，降低 fsync 频率以减少锁竞争
    """
    if connection.vendor != "sqlite":
        return

    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout = 5000;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
    finally:
        cursor.close()


# 另外：在你的 AppConfig 中确保导入本模块一次即可生效。
# 文件：AppleStockChecker/apps.py
# from django.apps import AppConfig
#
# class AppleStockCheckerConfig(AppConfig):
#     default_auto_field = "django.db.models.BigAutoField"
#     name = "AppleStockChecker"
#
#     def ready(self):
#         # 激活 SQLite PRAGMA 调整
#         from . import db_sqlite_tweaks  # noqa: F401  (仅为触发 import)
