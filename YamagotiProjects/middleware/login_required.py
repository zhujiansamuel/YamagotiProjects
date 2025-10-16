import re
from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin

import re
from django.conf import settings
from django.shortcuts import redirect

EXEMPT_PATTERNS = [
    r"^/accounts/login/?$",
    r"^/accounts/logout/?$",
    r"^/admin/.*$",                        # Django Admin
    r"^/static/.*$",
    r"^/media/.*$",
    # —— 你的应用 API 白名单（根据实际挂载前缀调整）——
    r"^/AppleStockChecker/auth/.*$",       # JWT 获取/刷新/验证
    r"^/AppleStockChecker/docs/.*$",       # 文档（可选）
    r"^/AppleStockChecker/health/?$",
    r"^/AppleStockChecker/purchasing-time-analyses/dispatch/?$",
    r"^/AppleStockChecker/purchasing-time-analyses/dispatch_ts/?$",
    # 如果你把 API 都挂在 /api/ 前缀，可以直接放行整个前缀：
    r"^/api/.*$",
    # WebSocket 必须放行，否则握手会被重定向
    r"^/ws/.*$",
    r"^/AppleStockChecker/purchasing-price-records/import-tradein/?$",
    r"^/AppleStockChecker/purchasing-price-records/import-tradein-xlsx/?$",
    r"^/AppleStockChecker/purchasing-time-analyses-psta-compact/?$",
    r"^/AppleStockChecker/purchasing-price-records/?$",
    r"^/AppleStockChecker/purchasing-price-records/ingest-webscraper/?$",
    r"^/AppleStockChecker/iphones/import-csv/?$",

]

class LoginRequiredMiddleware:
    """新风格 HTTP 中间件：未登录则跳转登录；白名单 & /ws/ 放行。
    放置顺序：在 AuthenticationMiddleware 之后，CommonMiddleware 之前更稳妥。
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self._compiled = [re.compile(p) for p in EXEMPT_PATTERNS]

    def __call__(self, request):
        path = request.path

        # 1) 放行白名单
        for pat in self._compiled:
            if pat.match(path):
                return self.get_response(request)

        # 2) 已登录放行
        user = getattr(request, "user", None)
        if user and user.is_authenticated:
            return self.get_response(request)

        # 3) 其余页面跳转登录（带 next）
        login_url = settings.LOGIN_URL or "/accounts/login/"
        return redirect(f"{login_url}?next={path}")
