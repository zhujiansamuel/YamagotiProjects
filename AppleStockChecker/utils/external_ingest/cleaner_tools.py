# AppleStockChecker/utils/external_ingest/cleaner_tools.py
"""
清洗器通用工具模块
提供数据库访问、数据转换等通用功能
"""
from __future__ import annotations
from typing import List
import pandas as pd


def _load_iphone17_info_df_from_db() -> pd.DataFrame:
    """
    从数据库中读取 iPhone 机型信息，返回 DataFrame

    输出列：part_number, model_name, capacity_gb, color, jan（如果 jan 字段有值）

    Returns:
        pd.DataFrame: 包含 iPhone 机型信息的 DataFrame

    Raises:
        ValueError: 如果数据库中没有 iPhone 数据
    """
    from AppleStockChecker.models import Iphone

    # 查询所有 iPhone 数据，只选择需要的字段
    queryset = Iphone.objects.all().values(
        'part_number',
        'model_name',
        'capacity_gb',
        'color',
        'jan'
    )

    # 转换为 DataFrame
    df = pd.DataFrame.from_records(queryset)

    if df.empty:
        raise ValueError("数据库中没有 iPhone 数据，请先导入 iPhone 机型信息")

    # 确保数据类型正确
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")

    # 删除必要字段为空的行
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number", "color"])

    # 处理 jan 列：如果所有 jan 都是空的，就删除这一列；否则保留
    if df["jan"].isna().all():
        df = df.drop(columns=["jan"])
        cols = ["part_number", "model_name", "capacity_gb", "color"]
    else:
        cols = ["part_number", "model_name", "capacity_gb", "color", "jan"]

    return df[cols].reset_index(drop=True)
