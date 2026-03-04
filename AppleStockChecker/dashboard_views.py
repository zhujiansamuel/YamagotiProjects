"""
Dashboard API views for AppleStockChecker app.

Provides read-only endpoint for the Dashboard to display
scraper/ingestion event history.
"""
import re
from collections import defaultdict
from datetime import timedelta

from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from .dashboard_auth import SimpleTokenAuthentication
from .models import DataIngestionLog
from .dashboard_serializers import ScraperEventSerializer

# Regex to extract parent shop name: shop5_1 → shop5, shop6_3 → shop6
_PARENT_SHOP_RE = re.compile(r'^(shop\d+)')


class ScraperEventDashboardView(APIView):
    """
    GET /api/dashboard/scraper-events/?days=2

    Returns DataIngestionLog entries grouped by parent source_name.
    shop5_1~5_4 are merged into "shop5", shop6_1~6_4 into "shop6", etc.
    """
    authentication_classes = [SimpleTokenAuthentication]
    permission_classes = [AllowAny]

    def get(self, request):
        days = int(request.query_params.get('days', 2))
        since = timezone.now() - timedelta(days=days)

        logs = DataIngestionLog.objects.filter(
            created_at__gte=since,
        ).order_by('-completed_at', '-created_at')

        # Group by parent source_name
        groups = defaultdict(list)
        for log in logs:
            m = _PARENT_SHOP_RE.match(log.source_name)
            parent = m.group(1) if m else log.source_name
            groups[parent].append(log)

        result = []
        for parent_name, log_list in sorted(groups.items()):
            events = ScraperEventSerializer(log_list, many=True).data
            result.append({
                'source_name': parent_name,
                'events': events,
            })

        return Response(result)
