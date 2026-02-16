# -*- coding: utf-8 -*-
"""
Celery + Django DB 连接安全

在 Celery prefork worker 场景下管理 Django 数据库连接：
- worker_process_init: fork 后关闭子进程继承的父进程连接，避免多进程共享连接
- task_prerun: 任务开始前清理过期/无效连接

适用 PostgreSQL 等数据库。不再包含 SQLite 特有的 task_postrun close_all。
"""
from celery import signals
from django.db import connections, close_old_connections


@signals.worker_process_init.connect
def _close_conns_on_fork(**kwargs):
    """子进程启动时关闭继承的连接，避免 fork 后多进程共享同一连接"""
    connections.close_all()


@signals.task_prerun.connect
def _task_prerun_close_stale(**kwargs):
    """每个任务开始前清理可能的陈旧连接"""
    close_old_connections()
