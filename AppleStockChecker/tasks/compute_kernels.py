"""
GPU 加速的计算核心函数

提供可以在 GPU 或 CPU 上运行的统计和数组计算函数。
所有函数都支持通过 use_gpu 参数控制使用 GPU 还是 CPU。
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple
from AppleStockChecker.utils.gpu_utils import (
    get_array_module,
    gpu_fallback,
    to_numpy,
)

logger = logging.getLogger(__name__)


@gpu_fallback
def compute_std(vals: List[float], use_gpu: Optional[bool] = None) -> float:
    """
    计算总体标准差（GPU 加速版本）

    参数:
        vals: 数值列表
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        总体标准差；N<=1 返回 0.0
    """
    n = len(vals)
    if n <= 1:
        return 0.0

    xp = get_array_module(use_gpu)
    arr = xp.asarray(vals, dtype=xp.float64)

    mu = xp.mean(arr)
    s2 = xp.mean((arr - mu) ** 2)
    result = float(xp.sqrt(s2))

    return result


@gpu_fallback
def compute_quantile(
    sorted_vals: List[float],
    p: float,
    use_gpu: Optional[bool] = None
) -> Optional[float]:
    """
    计算分位数（GPU 加速版本）

    参数:
        sorted_vals: 已排序的数值列表（升序）
        p: 分位数 (0.0 ~ 1.0)
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        分位数值；空列表返回 None
    """
    if not sorted_vals:
        return None

    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    xp = get_array_module(use_gpu)
    arr = xp.asarray(sorted_vals, dtype=xp.float64)

    # 使用最近邻方法
    k = int(round((n - 1) * p))
    k = max(0, min(k, n - 1))

    result = float(arr[k])
    return result


@gpu_fallback
def filter_outliers_by_mean_band(
    vals: List[float],
    lower_factor: float = 0.5,
    upper_factor: float = 1.5,
    use_gpu: Optional[bool] = None
) -> Tuple[List[float], Optional[float], Optional[float], Optional[float]]:
    """
    按"相对平均值"过滤异常值（GPU 加速版本）

    参数:
        vals: 数值列表
        lower_factor: 下界因子（默认 0.5 表示均值的 50%）
        upper_factor: 上界因子（默认 1.5 表示均值的 150%）
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        (filtered_vals, mean, low, high)
        - filtered_vals: 过滤后的值列表
        - mean: 原始均值
        - low: 下界阈值
        - high: 上界阈值
    """
    if not vals:
        return [], None, None, None

    xp = get_array_module(use_gpu)
    arr = xp.asarray(vals, dtype=xp.float64)

    m = float(xp.mean(arr))
    if m <= 0:
        # 极端情况，直接不过滤
        return list(vals), m, None, None

    low = m * lower_factor
    high = m * upper_factor

    # 使用 GPU 进行过滤
    mask = (arr >= low) & (arr <= high)
    filtered_arr = arr[mask]

    if len(filtered_arr) == 0:
        # 全被判成异常，用原始值
        return list(vals), m, low, high

    # 转换回 Python 列表
    filtered = to_numpy(filtered_arr).tolist()
    return filtered, m, low, high


@gpu_fallback
def compute_sma(
    series_old_to_new: List[float],
    window: int,
    use_gpu: Optional[bool] = None
) -> Optional[float]:
    """
    简单移动平均（GPU 加速版本）

    参数:
        series_old_to_new: 从旧到新的时间序列
        window: 窗口大小
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        移动平均值；无数据返回 None
    """
    if not series_old_to_new:
        return None

    w = max(1, int(window))
    xp = get_array_module(use_gpu)
    arr = xp.asarray(series_old_to_new, dtype=xp.float64)

    # 取最后 w 个元素
    s = arr[-w:] if w < len(arr) else arr
    result = float(xp.mean(s))

    return result


@gpu_fallback
def compute_ema(
    series_old_to_new: List[float],
    alpha: float,
    use_gpu: Optional[bool] = None
) -> float:
    """
    指数移动平均（GPU 加速版本）

    参数:
        series_old_to_new: 从旧到新的时间序列
        alpha: 平滑因子 (0 < alpha <= 1)
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        指数移动平均值
    """
    if not series_old_to_new:
        return 0.0

    xp = get_array_module(use_gpu)
    arr = xp.asarray(series_old_to_new, dtype=xp.float64)

    # EMA 计算：ema[t] = alpha * x[t] + (1-alpha) * ema[t-1]
    ema = float(arr[0])
    for i in range(1, len(arr)):
        ema = alpha * float(arr[i]) + (1.0 - alpha) * ema

    return ema


@gpu_fallback
def compute_wma_linear(
    series_old_to_new: List[float],
    window: int,
    use_gpu: Optional[bool] = None
) -> Optional[float]:
    """
    线性权重移动平均（GPU 加速版本）

    越新的数据权重越大。

    参数:
        series_old_to_new: 从旧到新的时间序列
        window: 窗口大小
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        加权移动平均值；无数据返回 None
    """
    if not series_old_to_new:
        return None

    w = max(1, int(window))
    xp = get_array_module(use_gpu)
    arr = xp.asarray(series_old_to_new, dtype=xp.float64)

    # 取最后 w 个元素
    s = arr[-w:] if w < len(arr) else arr
    n = len(s)

    # 权重: 1, 2, 3, ..., n
    weights = xp.arange(1, n + 1, dtype=xp.float64)
    denom = float(xp.sum(weights))

    if denom > 0:
        result = float(xp.sum(s * weights) / denom)
        return result
    else:
        return None


@gpu_fallback
def filter_price_range(
    prices: List[float],
    price_min: float,
    price_max: float,
    use_gpu: Optional[bool] = None
) -> List[float]:
    """
    过滤价格区间（GPU 加速版本）

    参数:
        prices: 价格列表
        price_min: 最小价格
        price_max: 最大价格
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        过滤后的价格列表
    """
    if not prices:
        return []

    xp = get_array_module(use_gpu)
    arr = xp.asarray(prices, dtype=xp.float64)

    mask = (arr >= price_min) & (arr <= price_max)
    filtered_arr = arr[mask]

    # 转换回 Python 列表
    filtered = to_numpy(filtered_arr).tolist()
    return filtered


@gpu_fallback
def compute_mean_median_sorted(
    sorted_vals: List[float],
    use_gpu: Optional[bool] = None
) -> Tuple[float, float]:
    """
    计算均值和中位数（输入已排序）

    参数:
        sorted_vals: 已排序的数值列表
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        (mean, median)
    """
    if not sorted_vals:
        return 0.0, 0.0

    xp = get_array_module(use_gpu)
    arr = xp.asarray(sorted_vals, dtype=xp.float64)

    mean = float(xp.mean(arr))
    median = float(xp.median(arr))

    return mean, median


# ===== 批量操作函数 =====

@gpu_fallback
def batch_compute_stats(
    data_groups: List[List[float]],
    use_gpu: Optional[bool] = None
) -> List[dict]:
    """
    批量计算统计量（GPU 加速版本）

    对多组数据并行计算均值、中位数、标准差等统计量。

    参数:
        data_groups: 多组数据的列表
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        统计结果列表，每个元素包含 mean, median, std, min, max, p10, p90
    """
    if not data_groups:
        return []

    xp = get_array_module(use_gpu)
    results = []

    for vals in data_groups:
        if not vals:
            results.append({
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'p10': None,
                'p90': None,
            })
            continue

        arr = xp.asarray(sorted(vals), dtype=xp.float64)  # 排序以便计算分位数

        mean = float(xp.mean(arr))
        median = float(xp.median(arr))
        std = float(xp.std(arr))
        min_val = float(xp.min(arr))
        max_val = float(xp.max(arr))

        # 计算分位数
        n = len(arr)
        p10_idx = max(0, min(int(round((n - 1) * 0.10)), n - 1))
        p90_idx = max(0, min(int(round((n - 1) * 0.90)), n - 1))

        p10 = float(arr[p10_idx])
        p90 = float(arr[p90_idx])

        results.append({
            'mean': mean,
            'median': median,
            'std': std,
            'min': min_val,
            'max': max_val,
            'p10': p10,
            'p90': p90,
        })

    return results
