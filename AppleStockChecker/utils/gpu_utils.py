# -*- coding: utf-8 -*-
"""
GPU Utilities for AutoML Pipeline
GPU 加速工具 - 支持 cupy/numpy 自动切换与优雅降级
"""
import logging
import numpy as np
from typing import Union, Any

logger = logging.getLogger(__name__)

# 全局 GPU 可用性标志
_GPU_AVAILABLE = False
_GPU_CHECKED = False
_GPU_ERROR = None

# 数组模块（默认 numpy）
xp = np


def check_gpu_availability() -> bool:
    """
    检查 GPU 和 CuPy 可用性

    Returns:
        bool: GPU 是否可用
    """
    global _GPU_AVAILABLE, _GPU_CHECKED, _GPU_ERROR, xp

    if _GPU_CHECKED:
        return _GPU_AVAILABLE

    _GPU_CHECKED = True

    try:
        import cupy as cp

        # 尝试创建一个简单的 GPU 数组来验证
        test_array = cp.array([1, 2, 3])
        _ = test_array.sum()

        # GPU 可用
        xp = cp
        _GPU_AVAILABLE = True

        # 记录 GPU 信息
        device_id = cp.cuda.Device().id
        device_name = cp.cuda.Device().name.decode() if hasattr(cp.cuda.Device().name, 'decode') else str(cp.cuda.Device().name)
        memory_info = cp.cuda.Device().mem_info
        total_memory_gb = memory_info[1] / (1024**3)

        logger.info(
            f"✓ GPU acceleration enabled: "
            f"Device {device_id} ({device_name}), "
            f"Total memory: {total_memory_gb:.2f} GB"
        )

        return True

    except ImportError as e:
        _GPU_ERROR = f"CuPy not installed: {e}"
        logger.warning(f"GPU not available: {_GPU_ERROR}")
        logger.info("Falling back to CPU (numpy)")
        xp = np
        _GPU_AVAILABLE = False
        return False

    except Exception as e:
        _GPU_ERROR = f"GPU initialization failed: {e}"
        logger.warning(f"GPU not available: {_GPU_ERROR}")
        logger.info("Falling back to CPU (numpy)")
        xp = np
        _GPU_AVAILABLE = False
        return False


def get_array_module(arr: Any = None):
    """
    获取当前使用的数组模块 (numpy or cupy)

    Args:
        arr: 可选的数组对象，用于推断模块

    Returns:
        numpy or cupy 模块
    """
    if arr is not None:
        # 如果提供了数组，使用 get_array_module 推断
        try:
            import cupy as cp
            return cp.get_array_module(arr)
        except (ImportError, AttributeError):
            return np

    # 否则返回全局模块
    return xp


def to_cpu(arr):
    """
    将数组转换到 CPU (numpy)

    Args:
        arr: numpy 或 cupy 数组

    Returns:
        numpy 数组
    """
    if isinstance(arr, np.ndarray):
        return arr

    # CuPy 数组
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ImportError:
        pass

    # 其他类型（如列表）
    return np.asarray(arr)


def to_gpu(arr):
    """
    将数组转换到 GPU (cupy)，如果 GPU 可用

    Args:
        arr: numpy 数组或类似对象

    Returns:
        cupy 数组（GPU 可用时）或 numpy 数组（GPU 不可用时）
    """
    if not _GPU_AVAILABLE:
        return np.asarray(arr)

    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)
    except Exception as e:
        logger.warning(f"Failed to transfer to GPU: {e}, using CPU")
        return np.asarray(arr)


def ensure_cpu_for_pandas(arr):
    """
    确保数组是 numpy 格式（用于 pandas 操作）

    Args:
        arr: 任何数组类型

    Returns:
        numpy 数组
    """
    return to_cpu(arr)


class GPUContext:
    """
    GPU 上下文管理器
    用于临时切换到 GPU 或 CPU 模式
    """

    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: 是否尝试使用 GPU
        """
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.original_xp = None

    def __enter__(self):
        global xp
        self.original_xp = xp

        if self.use_gpu:
            try:
                import cupy as cp
                xp = cp
                logger.debug("Switched to GPU mode")
            except ImportError:
                logger.debug("GPU requested but CuPy not available, using CPU")
                xp = np
        else:
            xp = np
            logger.debug("Using CPU mode")

        return xp

    def __exit__(self, exc_type, exc_val, exc_tb):
        global xp
        xp = self.original_xp
        logger.debug("Restored original array module")


def gpu_accelerated(fallback_to_cpu: bool = True):
    """
    装饰器：使函数支持 GPU 加速，失败时自动回退到 CPU

    Args:
        fallback_to_cpu: 是否在 GPU 失败时回退到 CPU
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if _GPU_AVAILABLE:
                try:
                    logger.debug(f"Running {func.__name__} on GPU")
                    return func(*args, **kwargs)
                except Exception as e:
                    if fallback_to_cpu:
                        logger.warning(
                            f"GPU execution failed for {func.__name__}: {e}, "
                            f"falling back to CPU"
                        )
                        # 临时禁用 GPU
                        global xp
                        original_xp = xp
                        xp = np
                        try:
                            result = func(*args, **kwargs)
                            return result
                        finally:
                            xp = original_xp
                    else:
                        raise
            else:
                logger.debug(f"Running {func.__name__} on CPU")
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def get_gpu_memory_info() -> dict:
    """
    获取 GPU 内存信息

    Returns:
        dict: {
            'total_gb': 总内存（GB）,
            'used_gb': 已使用（GB）,
            'free_gb': 可用（GB）,
            'utilization': 使用率（0-1）
        }
    """
    if not _GPU_AVAILABLE:
        return {
            'total_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
            'utilization': 0.0,
            'status': 'GPU not available'
        }

    try:
        import cupy as cp
        free, total = cp.cuda.Device().mem_info
        used = total - free

        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'utilization': used / total if total > 0 else 0.0,
            'status': 'GPU available'
        }
    except Exception as e:
        return {
            'total_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
            'utilization': 0.0,
            'status': f'GPU error: {e}'
        }


# 自动检查 GPU 可用性（模块导入时）
check_gpu_availability()
