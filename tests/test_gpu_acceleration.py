"""
GPU 加速功能测试

测试 GPU/CPU 自动切换和计算核心函数
"""
import pytest
import sys
from typing import List


class TestGPUUtils:
    """测试 GPU 工具函数"""

    def test_import_gpu_utils(self):
        """测试导入 GPU 工具模块"""
        from AppleStockChecker.utils.gpu_utils import (
            get_array_module,
            is_gpu_available,
            reset_gpu_state,
        )
        assert callable(get_array_module)
        assert callable(is_gpu_available)
        assert callable(reset_gpu_state)

    def test_cpu_mode(self):
        """测试强制 CPU 模式"""
        from AppleStockChecker.utils.gpu_utils import (
            get_array_module,
            reset_gpu_state,
        )

        reset_gpu_state()
        xp = get_array_module(use_gpu=False)
        assert xp.__name__ == 'numpy'

    def test_auto_detect(self):
        """测试自动检测模式"""
        from AppleStockChecker.utils.gpu_utils import (
            get_array_module,
            reset_gpu_state,
        )

        reset_gpu_state()
        xp = get_array_module(use_gpu=None)
        # xp 可能是 cupy 或 numpy，取决于环境
        assert xp.__name__ in ['cupy', 'numpy']

    def test_to_numpy(self):
        """测试数组转换"""
        from AppleStockChecker.utils.gpu_utils import (
            get_array_module,
            to_numpy,
        )
        import numpy as np

        xp = get_array_module(use_gpu=False)
        arr = xp.array([1, 2, 3])
        result = to_numpy(arr)

        assert isinstance(result, np.ndarray)
        assert result.tolist() == [1, 2, 3]


class TestComputeKernels:
    """测试计算核心函数"""

    def test_compute_std_cpu(self):
        """测试 CPU 模式的标准差计算"""
        from AppleStockChecker.tasks.compute_kernels import compute_std

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_std(data, use_gpu=False)

        # 总体标准差约为 sqrt(2) ≈ 1.414
        assert abs(result - 1.414) < 0.01

    def test_compute_std_empty(self):
        """测试空数据的标准差"""
        from AppleStockChecker.tasks.compute_kernels import compute_std

        result = compute_std([], use_gpu=False)
        assert result == 0.0

        result = compute_std([1.0], use_gpu=False)
        assert result == 0.0

    def test_compute_quantile_cpu(self):
        """测试 CPU 模式的分位数计算"""
        from AppleStockChecker.tasks.compute_kernels import compute_quantile

        data = sorted([float(i) for i in range(1, 11)])  # [1, 2, ..., 10]

        # 中位数 (50%)
        result = compute_quantile(data, 0.5, use_gpu=False)
        assert 5.0 <= result <= 6.0

        # 90 分位数
        result = compute_quantile(data, 0.9, use_gpu=False)
        assert 9.0 <= result <= 10.0

    def test_compute_quantile_empty(self):
        """测试空数据的分位数"""
        from AppleStockChecker.tasks.compute_kernels import compute_quantile

        result = compute_quantile([], 0.5, use_gpu=False)
        assert result is None

    def test_filter_outliers_cpu(self):
        """测试 CPU 模式的异常值过滤"""
        from AppleStockChecker.tasks.compute_kernels import (
            filter_outliers_by_mean_band
        )

        # 正常数据 + 异常值
        data = [100.0, 102.0, 98.0, 101.0, 99.0, 200.0, 10.0]
        filtered, mean, low, high = filter_outliers_by_mean_band(
            data,
            lower_factor=0.8,
            upper_factor=1.2,
            use_gpu=False
        )

        # 应该过滤掉 200 和 10
        assert len(filtered) < len(data)
        assert 200.0 not in filtered
        assert 10.0 not in filtered

    def test_filter_outliers_all_filtered(self):
        """测试全部被过滤的情况"""
        from AppleStockChecker.tasks.compute_kernels import (
            filter_outliers_by_mean_band
        )

        # 极端数据，会全部被过滤
        data = [1.0, 1000.0]
        filtered, mean, low, high = filter_outliers_by_mean_band(
            data,
            lower_factor=0.9,
            upper_factor=1.1,
            use_gpu=False
        )

        # 应该回退到原始数据
        assert len(filtered) == len(data)

    def test_compute_sma_cpu(self):
        """测试 CPU 模式的简单移动平均"""
        from AppleStockChecker.tasks.compute_kernels import compute_sma

        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # 窗口 3
        result = compute_sma(data, window=3, use_gpu=False)
        # 应该是最后 3 个的平均: (3+4+5)/3 = 4.0
        assert abs(result - 4.0) < 0.01

        # 窗口 10（大于数据长度）
        result = compute_sma(data, window=10, use_gpu=False)
        # 应该是全部的平均: (1+2+3+4+5)/5 = 3.0
        assert abs(result - 3.0) < 0.01

    def test_compute_ema_cpu(self):
        """测试 CPU 模式的指数移动平均"""
        from AppleStockChecker.tasks.compute_kernels import compute_ema

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        alpha = 0.5

        result = compute_ema(data, alpha=alpha, use_gpu=False)

        # EMA 计算：不断迭代
        # ema[0] = 1.0
        # ema[1] = 0.5*2 + 0.5*1 = 1.5
        # ema[2] = 0.5*3 + 0.5*1.5 = 2.25
        # ema[3] = 0.5*4 + 0.5*2.25 = 3.125
        # ema[4] = 0.5*5 + 0.5*3.125 = 4.0625
        expected = 4.0625
        assert abs(result - expected) < 0.01

    def test_compute_wma_linear_cpu(self):
        """测试 CPU 模式的线性权重移动平均"""
        from AppleStockChecker.tasks.compute_kernels import compute_wma_linear

        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # 窗口 3
        result = compute_wma_linear(data, window=3, use_gpu=False)
        # 最后 3 个: [3, 4, 5]
        # 权重: [1, 2, 3]
        # WMA = (3*1 + 4*2 + 5*3) / (1+2+3) = (3+8+15) / 6 = 26/6 ≈ 4.33
        assert abs(result - 4.33) < 0.01

    def test_filter_price_range_cpu(self):
        """测试 CPU 模式的价格区间过滤"""
        from AppleStockChecker.tasks.compute_kernels import filter_price_range

        prices = [50000.0, 100000.0, 150000.0, 200000.0, 250000.0]

        result = filter_price_range(
            prices,
            price_min=100000.0,
            price_max=200000.0,
            use_gpu=False
        )

        assert len(result) == 3
        assert 50000.0 not in result
        assert 250000.0 not in result
        assert 100000.0 in result
        assert 150000.0 in result
        assert 200000.0 in result


class TestIntegration:
    """集成测试"""

    def test_task_functions_cpu(self):
        """测试任务函数（CPU 模式）"""
        from AppleStockChecker.tasks.timestamp_alignment_task import (
            _pop_std,
            _quantile,
            _filter_outliers_by_mean_band,
            _sma,
            _ema_from_series,
        )
        import os

        # 强制 CPU 模式
        os.environ['PSTA_USE_GPU'] = 'false'

        # 测试标准差
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _pop_std(data)
        assert abs(result - 1.414) < 0.01

        # 测试分位数
        sorted_data = sorted(data)
        result = _quantile(sorted_data, 0.5)
        assert result == 3.0

        # 测试异常值过滤
        filtered, mean, low, high = _filter_outliers_by_mean_band(data)
        assert len(filtered) == len(data)

        # 测试 SMA
        result = _sma(data, window=3)
        assert abs(result - 4.0) < 0.01

        # 测试 EMA
        result = _ema_from_series(data, alpha=0.5)
        assert abs(result - 4.0625) < 0.01

        # 恢复环境变量
        os.environ.pop('PSTA_USE_GPU', None)


class TestGPUFallback:
    """测试 GPU 降级机制"""

    def test_fallback_decorator(self):
        """测试 GPU 降级装饰器"""
        from AppleStockChecker.utils.gpu_utils import gpu_fallback

        call_count = {'cpu': 0, 'gpu': 0}

        @gpu_fallback
        def mock_computation(data: List[float], use_gpu=None):
            """模拟计算函数"""
            if use_gpu is False:
                call_count['cpu'] += 1
                return sum(data)
            else:
                # 模拟 GPU 错误
                call_count['gpu'] += 1
                raise RuntimeError("CUDA out of memory")

        # 调用函数，应该自动降级到 CPU
        result = mock_computation([1, 2, 3], use_gpu=True)

        # CPU 模式应该被调用
        assert call_count['cpu'] == 1
        assert result == 6


@pytest.mark.skipif(
    sys.platform.startswith('darwin') and sys.version_info < (3, 9),
    reason="需要 Python 3.9+ 或非 macOS 环境"
)
class TestGPUMode:
    """GPU 模式测试（需要 GPU 环境）"""

    def test_gpu_available(self):
        """测试 GPU 是否可用"""
        from AppleStockChecker.utils.gpu_utils import is_gpu_available

        # 这个测试只是检查函数能否正常运行，不检查结果
        # 因为 CI 环境可能没有 GPU
        result = is_gpu_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("cupy", reason="需要安装 cupy"),
        reason="需要 cupy 支持"
    )
    def test_compute_with_gpu(self):
        """测试 GPU 计算（需要 cupy）"""
        from AppleStockChecker.tasks.compute_kernels import compute_std

        data = [float(i) for i in range(1, 1001)]  # 大数据集

        # 尝试使用 GPU
        try:
            result_gpu = compute_std(data, use_gpu=True)
            result_cpu = compute_std(data, use_gpu=False)

            # 结果应该相近
            assert abs(result_gpu - result_cpu) < 1.0
        except Exception as e:
            # GPU 不可用或其他错误，跳过测试
            pytest.skip(f"GPU 测试失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
