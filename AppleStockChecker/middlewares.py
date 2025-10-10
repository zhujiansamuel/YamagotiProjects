# app/middlewares.py
from urllib.parse import parse_qs
from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.contrib.auth import get_user_model

User = get_user_model()
jwt_auth = JWTAuthentication()

@database_sync_to_async
def get_user_from_token(raw_token: str):
    validated = jwt_auth.get_validated_token(raw_token)
    user = jwt_auth.get_user(validated)
    return user

class TokenAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        token = None

        # 1) Authorization: Bearer xxx
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()

        # 2) ?token=xxx
        if not token:
            query = parse_qs(scope.get("query_string", b"").decode())
            token = (query.get("token") or [None])[0]

        user = AnonymousUser()
        if token:
            try:
                user = await get_user_from_token(token)
            except Exception:
                user = AnonymousUser()

        scope["user"] = user
        return await super().__call__(scope, receive, send)
