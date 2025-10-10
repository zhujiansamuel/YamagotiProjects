"""
ASGI config for YamagotiProjects project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()  # ✅ 先让 AppRegistry 就绪，再导入后续依赖

from django.core.asgi import get_asgi_application

django_asgi_app = get_asgi_application()

import os
from django.core.asgi import get_asgi_application
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from AppleStockChecker.consumers import TaskProgressConsumer
from channels.auth import AuthMiddlewareStack
from AppleStockChecker.middlewares import TokenAuthMiddleware
from channels.security.websocket import AllowedHostsOriginValidator, OriginValidator

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'YamagotiProjects.settings')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket":
        TokenAuthMiddleware(  # 先尝试 JWT
            AuthMiddlewareStack(  # 再退回 Session
                URLRouter([
                    path("ws/stream/psta/", TaskProgressConsumer.as_asgi()),
                    path("ws/task/<str:job_id>/", TaskProgressConsumer.as_asgi()),
                ])
            )
        )
})
