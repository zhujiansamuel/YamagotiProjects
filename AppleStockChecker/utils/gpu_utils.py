"""
GPU/CPU 自动切换工具模块

提供在 GPU 可用时使用 cupy 加速，GPU 不可用或资源耗尽时自动降级到 numpy 的能力。
"""
from __future__ import annotations
import logging
import threading
from typing import Any, Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# 线程安全的 GPU 状态管理
_gpu_lock = threading.Lock()
_gpu_available: Optional[bool] = None
_xp = None  # 当前使用的数组库 (cupy 或 numpy)


def get_array_module(use_gpu: Optional[bool] = None):
    """
    获取数组计算模块（cupy 或 numpy）

    参数:
        use_gpu: 如果为 None，则自动检测；True 强制使用 GPU；False 强制使用 CPU

    返回:
        cupy 或 numpy 模块
    """
    global _gpu_available, _xp

    # 如果强制指定了 CPU
    if use_gpu is False:
        import numpy
        return numpy

    # 检查 GPU 是否可用（带缓存）
    with _gpu_lock:
        if _gpu_available is None:
            try:
                import cupy
                # 尝试执行一个简单的 GPU 操作以确保 GPU 真正可用
                _ = cupy.array([1.0])
                _gpu_available = True
                _xp = cupy
                logger.info("GPU (cupy) 可用，启用 GPU 加速")
            except Exception as e:
                _gpu_available = False
                import numpy
                _xp = numpy
                logger.info(f"GPU 不可用，使用 CPU 模式: {e}")

        # 如果用户强制要求 GPU 但 GPU 不可用
        if use_gpu is True and not _gpu_available:
            logger.warning("强制要求使用 GPU，但 GPU 不可用，降级到 CPU")
            import numpy
            return numpy

        return _xp


def reset_gpu_state():
    """重置 GPU 状态缓存（主要用于测试）"""
    global _gpu_available, _xp
    with _gpu_lock:
        _gpu_available = None
        _xp = None


def is_gpu_available() -> bool:
    """检查 GPU 是否可用"""
    xp = get_array_module()
    return xp.__name__ == 'cupy'


def to_numpy(arr):
    """
    将数组转换为 numpy 数组

    如果输入是 cupy 数组，则转换为 numpy；否则直接返回
    """
    if hasattr(arr, 'get'):  # cupy 数组有 get() 方法
        return arr.get()
    return arr


def gpu_fallback(func: Callable) -> Callable:
    """
    装饰器：GPU 资源耗尽时自动降级到 CPU

    用法:
        @gpu_fallback
        def my_computation(data, use_gpu=None):
            xp = get_array_module(use_gpu)
            arr = xp.array(data)
            return xp.mean(arr)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 首先尝试使用 GPU
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            # 检查是否是 GPU 相关的错误
            is_gpu_error = any(keyword in error_msg for keyword in [
                'out of memory',
                'cuda',
                'gpu',
                'device',
                'cupy',
            ])

            if is_gpu_error:
                logger.warning(
                    f"GPU 计算失败 ({e.__class__.__name__}: {e})，"
                    f"降级到 CPU 重试: {func.__name__}"
                )
                # 强制使用 CPU 重试
                kwargs['use_gpu'] = False
                try:
                    return func(*args, **kwargs)
                except Exception as cpu_error:
                    logger.error(
                        f"CPU 降级后仍然失败: {func.__name__} - {cpu_error}"
                    )
                    raise
            else:
                # 非 GPU 错误，直接抛出
                raise

    return wrapper


class GPUContext:
    """
    GPU/CPU 上下文管理器

    用法:
        with GPUContext() as xp:
            arr = xp.array([1, 2, 3])
            result = xp.mean(arr)
    """
    def __init__(self, use_gpu: Optional[bool] = None):
        self.use_gpu = use_gpu
        self.xp = None

    def __enter__(self):
        self.xp = get_array_module(self.use_gpu)
        return self.xp

    def __exit__(self, exc_type, exc_val, exc_tb):
        # GPU OOM 等错误时的处理可以在这里添加
        if exc_type is not None and self.use_gpu is not False:
            error_msg = str(exc_val).lower() if exc_val else ''
            is_gpu_error = any(keyword in error_msg for keyword in [
                'out of memory',
                'cuda',
                'gpu',
                'device',
            ])
            if is_gpu_error:
                logger.warning(
                    f"GPU 上下文中发生错误: {exc_type.__name__}: {exc_val}"
                )
        return False  # 不抑制异常


# 便捷函数：常用数组操作
def array(data, use_gpu: Optional[bool] = None):
    """创建数组"""
    xp = get_array_module(use_gpu)
    return xp.array(data)


def asarray(data, use_gpu: Optional[bool] = None):
    """转换为数组"""
    xp = get_array_module(use_gpu)
    return xp.asarray(data)


def zeros(shape, dtype=None, use_gpu: Optional[bool] = None):
    """创建全零数组"""
    xp = get_array_module(use_gpu)
    return xp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None, use_gpu: Optional[bool] = None):
    """创建全一数组"""
    xp = get_array_module(use_gpu)
    return xp.ones(shape, dtype=dtype)
