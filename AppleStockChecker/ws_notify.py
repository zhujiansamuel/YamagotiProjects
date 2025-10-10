# app/ws_notify.py
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from math import ceil
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

def group_for_all():
    return "stream_psta_all"


def group_for_job(job_id: str) -> str:
    return f"task_{job_id}"

def group_for_stream(shop_id: int, iphone_id: int) -> str:
    return f"stream_psta_shop_{shop_id}_iphone_{iphone_id}"


def notify_progress(*, job_id: str|None, shop_id: int|None, iphone_id: int|None, data: dict):
    ch = get_channel_layer()
    payload = {"type": "progress_message", "data": data}
    # 单次任务频道（若还保留）
    if job_id:
        async_to_sync(ch.group_send)(f"task_{job_id}", payload)
    # 维度频道
    if shop_id is not None and iphone_id is not None:
        async_to_sync(ch.group_send)(group_for_stream(shop_id, iphone_id), payload)
    # 全部频道（可选）
    async_to_sync(ch.group_send)(group_for_all(), payload)

def notify_batch_items(*, job_id: str, shop_id: int, iphone_id: int,
                       timestamp_iso: str, items: list, chunk_size: int = 200):
    """
    把一批实例（同一 Timestamp_Time）分片推送。
    每片消息结构:
    {
      "type": "batch_chunk",
      "timestamp": "...",
      "index": 1, "total": 5,
      "count": 200,
      "items": [ {...}, ... ]
    }
    最后一条发送 "batch_done" 汇总。
    """
    if not items:
        return
    ch = get_channel_layer()
    total_chunks = ceil(len(items) / chunk_size)
    stream_group = group_for_stream(shop_id, iphone_id)
    job_group = group_for_job(job_id) if job_id else None

    for i in range(total_chunks):
        batch = items[i * chunk_size : (i + 1) * chunk_size]
        msg = {
            "type": "progress_message",
            "data": {
                "type": "batch_chunk",
                "timestamp": timestamp_iso,
                "index": i + 1,
                "total": total_chunks,
                "count": len(batch),
                "items": batch,
            },
        }
        # 稳定主题
        async_to_sync(ch.group_send)(stream_group, msg)
        # 单次任务频道（若有）
        if job_group:
            async_to_sync(ch.group_send)(job_group, msg)

    # 结束标记
    done_msg = {
        "type": "progress_message",
        "data": {
            "type": "batch_done",
            "timestamp": timestamp_iso,
            "total_chunks": total_chunks,
            "total_items": len(items),
        },
    }
    async_to_sync(ch.group_send)(stream_group, done_msg)
    if job_group:
        async_to_sync(ch.group_send)(job_group, done_msg)





def notify_progress_all(*, data: dict) -> None:
    ch = get_channel_layer()
    async_to_sync(ch.group_send)(
        group_for_all(),
        {"type": "progress_message", "data": data},
    )

def notify_batch_items_all(*, timestamp_iso: str, items: list, index: int, total: int) -> None:
    ch = get_channel_layer()
    async_to_sync(ch.group_send)(
        group_for_all(),
        {"type": "progress_message",
         "data": {"type":"batch_chunk","timestamp":timestamp_iso,"index":index,"total":total,
                  "count":len(items),"items":items}}
    )

def notify_batch_done_all(*, timestamp_iso: str, total_chunks: int, total_items: int) -> None:
    ch = get_channel_layer()
    async_to_sync(ch.group_send)(
        group_for_all(),
        {"type": "progress_message",
         "data": {"type":"batch_done","timestamp":timestamp_iso,
                  "total_chunks":total_chunks,"total_items":total_items}}
    )