"""价格趋势 API：机型+容量下的回收价曲线、平均线、标准差"""
from .model_colors import trends_model_colors
from .avg_only import TrendsAvgOnlyApiView
from .color_std import TrendsColorStdApiView
from .core import compute_trends_for_model_capacity

__all__ = [
    "trends_model_colors",
    "TrendsAvgOnlyApiView",
    "TrendsColorStdApiView",
    "compute_trends_for_model_capacity",
]
