from __future__ import annotations
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")  # ←改成你的项目包名
app = Celery("YamagotiProjects")  # ←改成你的项目包名
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()