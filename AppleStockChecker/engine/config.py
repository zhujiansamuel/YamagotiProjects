"""
Pipeline 配置常量。
参考: docs/REFACTOR_PLAN_V1.md §4
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BucketConfig:
    """15 分钟桶对齐参数"""
    interval_min: int = 15
    lookback_min: int = 15
    min_quorum: int = 16
    quorum_policy: str = "use_anyway"   # 不管几个店都算，记录 shop_count


# 特征计算窗口 (分钟)
FEATURE_WINDOWS: list[int] = [120, 900, 1800]

# 每个窗口对应的桶数 (÷15min)
WINDOW_TO_BUCKETS: dict[int, int] = {w: w // 15 for w in FEATURE_WINDOWS}

# pipeline 默认步骤名
ALL_STEPS: list[str] = ["align", "aggregate", "features", "cohorts"]
