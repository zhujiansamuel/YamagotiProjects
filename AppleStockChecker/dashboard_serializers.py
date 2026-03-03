"""
Dashboard serializers for AppleStockChecker app.
"""
from rest_framework import serializers


class ScraperEventSerializer(serializers.Serializer):
    id = serializers.UUIDField(source='batch_id')
    source_name = serializers.CharField()
    task_type = serializers.CharField()
    timestamp = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    rows_received = serializers.IntegerField()
    rows_inserted = serializers.IntegerField()
    rows_updated = serializers.IntegerField()
    rows_skipped = serializers.IntegerField()
    rows_unmatched = serializers.IntegerField()
    error_message = serializers.SerializerMethodField()
    created_at = serializers.DateTimeField()
    received_at = serializers.DateTimeField()
    cleaning_started_at = serializers.DateTimeField()
    completed_at = serializers.DateTimeField()

    STATUS_MAP = {
        'pending': 'pending',
        'receiving': 'running',
        'cleaning': 'running',
        'completed': 'success',
        'failed': 'error',
    }

    def get_timestamp(self, obj):
        return obj.completed_at or obj.created_at

    def get_status(self, obj):
        return self.STATUS_MAP.get(obj.status, obj.status)

    def get_error_message(self, obj):
        return obj.error_message or None


class ScraperEventGroupSerializer(serializers.Serializer):
    source_name = serializers.CharField()
    events = ScraperEventSerializer(many=True)
