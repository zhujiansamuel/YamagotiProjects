import re
from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin

EXEMPT_URLS = [
    r"^/accounts/login/$",
    r"^/accounts/logout/$",
    r"^/admin/.*$",
    r"^/static/.*$",
    r"^/AppleStockChecker/auth/.*$",         # JWT 获取/刷新
    r"^/AppleStockChecker/docs/.*$",         # Swagger 文档（按需放行）
    r"^/healthz$",                           # 健康检查（如果有）
]

class LoginRequiredMiddleware(MiddlewareMixin):
    def process_request(self, request):
        path = request.path
        # 已登录直接放行
        if request.user.is_authenticated:
            return None
        # 白名单
        for pattern in EXEMPT_URLS:
            if re.match(pattern, path):
                return None
        # 跳转登录页，带 next
        login_url = settings.LOGIN_URL
        return redirect(f"{login_url}?next={path}")
