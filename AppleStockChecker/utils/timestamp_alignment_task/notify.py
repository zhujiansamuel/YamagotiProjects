# AppleStockChecker/utils/timestamp_alignment_task/notify.py
"""PSTA 任务进度推送（WebSocket Channels）"""
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


def _group_for_all():
    return "stream_psta_all"


def notify_progress_all(*, data: dict) -> None:
    ch = get_channel_layer()
    async_to_sync(ch.group_send)(
        _group_for_all(),
        {"type": "progress_message", "data": data},
    )


def notify_batch_items_all(*, timestamp_iso: str, items: list, index: int, total: int) -> None:
    ch = get_channel_layer()
    async_to_sync(ch.group_send)(
        _group_for_all(),
        {"type": "progress_message",
         "data": {"type": "batch_chunk", "timestamp": timestamp_iso, "index": index, "total": total,
                  "count": len(items), "items": items}}
    )


def notify_batch_done_all(*, timestamp_iso: str, total_chunks: int, total_items: int) -> None:
    ch = get_channel_layer()
    async_to_sync(ch.group_send)(
        _group_for_all(),
        {"type": "progress_message",
         "data": {"type": "batch_done", "timestamp": timestamp_iso,
                  "total_chunks": total_chunks, "total_items": total_items}}
    )
