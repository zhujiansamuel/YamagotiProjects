"""
Simple token authentication for Dashboard service-to-service calls.
"""
from django.conf import settings
from rest_framework import authentication, exceptions


class SimpleTokenAuthentication(authentication.BaseAuthentication):
    """
    Static token authentication for Dashboard backend.

    Token should be passed in the Authorization header:
        Authorization: Token <token>
    """

    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'token':
            return None

        token = parts[1]
        expected = getattr(settings, 'DASHBOARD_SERVICE_TOKEN', '')

        if not expected:
            raise exceptions.AuthenticationFailed('DASHBOARD_SERVICE_TOKEN not configured')

        if token != expected:
            raise exceptions.AuthenticationFailed('Invalid token')

        return (None, {'token': token})

    def authenticate_header(self, request):
        return 'Token'
