# -*- coding: utf-8 -*-
"""
Shop 队列映射工具模块

将 source_name 映射到对应的 Celery 队列名称。
"""
from __future__ import annotations


def normalize_source_name(source_name: str) -> str:
    """
    规范化 source_name，将连字符转换为下划线

    Args:
        source_name: 原始 source 名称（如 "shop5-1" 或 "shop5_1"）

    Returns:
        规范化后的名称（如 "shop5_1"）
    """
    return source_name.replace("-", "_")


def get_shop_queue(source_name: str) -> str:
    """
    将 source_name 映射到 Celery 队列名称

    规则：
    - shop5_1, shop5_2, shop5_3, shop5_4 → shop_shop5
    - shop6_1, shop6_2, shop6_3, shop6_4 → shop_shop6
    - 其他：shop1 → shop_shop1, shop2 → shop_shop2, ...

    Args:
        source_name: 数据源名称（如 "shop1", "shop5-1", "shop5_1"）

    Returns:
        队列名称（如 "shop_shop1", "shop_shop5"）
    """
    # 规范化：连字符 → 下划线
    normalized = normalize_source_name(source_name)

    # shop5_1~4 → shop_shop5（合并到同一队列）
    if normalized.startswith("shop5_"):
        return "shop_shop5"

    # shop6_1~4 → shop_shop6（合并到同一队列）
    if normalized.startswith("shop6_"):
        return "shop_shop6"

    # 其他店铺：直接映射
    # shop1 → shop_shop1
    # shop10 → shop_shop10
    return f"shop_{normalized}"


def get_cleaner_name(source_name: str) -> str:
    """
    获取清洗器名称（用于 run_cleaner 调用）

    Args:
        source_name: 数据源名称（如 "shop1", "shop5-1"）

    Returns:
        清洗器名称（如 "shop1", "shop5_1"）
    """
    return normalize_source_name(source_name)


# 所有支持的队列列表（用于启动脚本和配置）
SHOP_QUEUES = [
    "shop_shop1",
    "shop_shop2",
    "shop_shop3",
    "shop_shop4",
    "shop_shop5",   # 包含 shop5_1, shop5_2, shop5_3, shop5_4
    "shop_shop6",   # 包含 shop6_1, shop6_2, shop6_3, shop6_4
    "shop_shop7",
    "shop_shop8",
    "shop_shop9",
    "shop_shop10",
    "shop_shop11",
    "shop_shop12",
    "shop_shop13",
    "shop_shop14",
    "shop_shop15",
    "shop_shop16",
    "shop_shop17",
    "shop_shop18",
    "shop_shop20",
]

# 队列到店铺的反向映射（用于文档和调试）
QUEUE_TO_SHOPS = {
    "shop_shop1": ["shop1"],
    "shop_shop2": ["shop2"],
    "shop_shop3": ["shop3"],
    "shop_shop4": ["shop4"],
    "shop_shop5": ["shop5_1", "shop5_2", "shop5_3", "shop5_4"],
    "shop_shop6": ["shop6_1", "shop6_2", "shop6_3", "shop6_4"],
    "shop_shop7": ["shop7"],
    "shop_shop8": ["shop8"],
    "shop_shop9": ["shop9"],
    "shop_shop10": ["shop10"],
    "shop_shop11": ["shop11"],
    "shop_shop12": ["shop12"],
    "shop_shop13": ["shop13"],
    "shop_shop14": ["shop14"],
    "shop_shop15": ["shop15"],
    "shop_shop16": ["shop16"],
    "shop_shop17": ["shop17"],
    "shop_shop18": ["shop18"],
    "shop_shop20": ["shop20"],
}
