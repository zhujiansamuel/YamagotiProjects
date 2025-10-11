from celery import signals
from django.db import connections, close_old_connections

@signals.worker_process_init.connect
def _close_conns_on_fork(**kwargs):
    # 子进程启动时关闭继承的连接
    connections.close_all()

@signals.task_prerun.connect
def _task_prerun_close_stale(**kwargs):
    # 每个任务开始前清理可能的陈旧连接
    close_old_connections()

@signals.task_postrun.connect
def _task_postrun_close_all(**kwargs):
    # 每个任务结束后尽快释放连接（SQLite 单写器特性）
    connections.close_all()