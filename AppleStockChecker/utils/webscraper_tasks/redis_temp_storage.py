# -*- coding: utf-8 -*-
"""
Redis 临时存储工具模块

用于在数据接收任务和清洗任务之间传递 DataFrame 数据。
"""
from __future__ import annotations

import json
from typing import Optional
import pandas as pd
import redis
from django.conf import settings
from io import StringIO


# 默认 TTL: 1 小时
DEFAULT_TTL = 3600

# Redis key 前缀
KEY_PREFIX = "ingest:temp"


def _get_redis_client() -> redis.Redis:
    """获取 Redis 客户端实例"""
    redis_url = getattr(settings, "CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
    return redis.from_url(redis_url, decode_responses=True)


def make_redis_key(batch_id: str, source_name: str) -> str:
    """生成 Redis key"""
    return f"{KEY_PREFIX}:{batch_id}:{source_name}"


def store_dataframe(
    batch_id: str,
    source_name: str,
    df: pd.DataFrame,
    ttl: int = DEFAULT_TTL,
) -> str:
    """
    将 DataFrame 存储到 Redis

    Args:
        batch_id: 批次 ID
        source_name: 数据源名称
        df: 要存储的 DataFrame
        ttl: 过期时间（秒），默认 1 小时

    Returns:
        Redis key
    """
    client = _get_redis_client()
    key = make_redis_key(batch_id, source_name)

    # 将 DataFrame 序列化为 JSON
    # 使用 date_format="iso" 确保日期时间格式正确
    data = df.to_json(orient="records", date_format="iso", force_ascii=False)

    client.setex(key, ttl, data)
    return key


def retrieve_dataframe(redis_key: str) -> Optional[pd.DataFrame]:
    """
    从 Redis 读取 DataFrame

    Args:
        redis_key: Redis key

    Returns:
        DataFrame 或 None（如果 key 不存在或已过期）
    """
    client = _get_redis_client()
    data = client.get(redis_key)

    if data is None:
        return None

    # 反序列化为 DataFrame
    return pd.read_json(StringIO(data), orient="records")


def delete_key(redis_key: str) -> bool:
    """
    删除 Redis key

    Args:
        redis_key: 要删除的 key

    Returns:
        是否删除成功
    """
    client = _get_redis_client()
    return client.delete(redis_key) > 0


def key_exists(redis_key: str) -> bool:
    """
    检查 Redis key 是否存在

    Args:
        redis_key: 要检查的 key

    Returns:
        是否存在
    """
    client = _get_redis_client()
    return client.exists(redis_key) > 0
