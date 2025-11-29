# GPU 加速文档

## 概述

分钟桶任务（`psta_process_minute_bucket`）现已支持 GPU 加速。通过使用 CuPy（CUDA 加速的 NumPy 兼容库），可以显著提升数值计算性能。

系统会自动检测 GPU 可用性，并在 GPU 不可用或资源耗尽时无缝降级到 CPU 模式。

## 架构设计

### 核心组件

1. **gpu_utils.py** - GPU/CPU 自动切换工具
   - `get_array_module()`: 获取 cupy 或 numpy 模块
   - `is_gpu_available()`: 检查 GPU 是否可用
   - `gpu_fallback`: 装饰器，自动处理 GPU 故障降级
   - `GPUContext`: 上下文管理器，便于在代码块中使用 GPU

2. **compute_kernels.py** - GPU 加速的计算核心
   - `compute_std()`: 标准差计算
   - `compute_quantile()`: 分位数计算
   - `filter_outliers_by_mean_band()`: 异常值过滤
   - `compute_sma()`: 简单移动平均
   - `compute_ema()`: 指数移动平均
   - `compute_wma_linear()`: 线性权重移动平均
   - `filter_price_range()`: 价格区间过滤

3. **timestamp_alignment_task.py** - 任务主文件
   - 已改造的计算函数自动调用 GPU 加速版本
   - 通过环境变量控制 GPU 使用策略

### 改造的计算函数

以下函数已支持 GPU 加速（原接口不变）：

- `_pop_std()` - 总体标准差
- `_quantile()` - 分位数
- `_filter_outliers_by_mean_band()` - 异常值过滤
- `_sma()` - 简单移动平均
- `_ema_from_series()` - 指数移动平均
- `_wma_linear()` - 线性权重移动平均

## 环境配置

### 1. GPU 服务器配置

#### 安装 CUDA 依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 GPU 依赖（根据 CUDA 版本选择）
pip install -r requirements-gpu.txt
```

#### 配置环境变量

在 `.env` 文件中添加：

```bash
# GPU 加速配置
PSTA_USE_GPU=auto  # auto|true|false
```

配置选项说明：
- `auto` (默认): 自动检测 GPU，有就用，没有就用 CPU
- `true`: 强制尝试使用 GPU（失败时降级到 CPU）
- `false`: 强制使用 CPU（即使有 GPU 也不用）

### 2. CPU 服务器配置

CPU 服务器**不需要**安装 cupy，只需：

```bash
# 安装基础依赖即可
pip install -r requirements.txt

# 设置环境变量（可选，默认会自动检测）
PSTA_USE_GPU=false
```

系统会自动使用 numpy 进行计算。

## 故障降级机制

### 自动降级触发条件

系统会在以下情况自动降级到 CPU：

1. **GPU 不可用**
   - 未安装 cupy
   - 未安装 CUDA 驱动
   - GPU 硬件故障

2. **GPU 资源耗尽**
   - GPU 显存不足 (Out of Memory)
   - GPU 设备错误

3. **环境配置**
   - `PSTA_USE_GPU=false`

### 降级行为

当 GPU 故障时：

1. 捕获异常并记录日志
2. 自动使用 CPU (numpy) 重试
3. 如果 CPU 也失败，则抛出异常

示例日志：

```
INFO: 分钟桶任务启动 (ts=2025-01-01T10:00:00+09:00): GPU可用=True, GPU模式=None
WARNING: GPU 计算失败 (CudaError: out of memory)，降级到 CPU 重试: compute_std
```

## 性能预期

### 加速效果

对于典型的分钟桶任务（包含数千个价格数据点）：

- **小数据集** (< 100 个数据点): CPU 可能更快（GPU 有初始化开销）
- **中等数据集** (100-1000 个数据点): GPU 加速 **2-5 倍**
- **大数据集** (> 1000 个数据点): GPU 加速 **5-10 倍**

### 监控指标

任务日志中会记录：

```python
logger.info(
    f"分钟桶任务启动 (ts={ts_iso}): "
    f"GPU可用={gpu_available}, GPU模式={use_gpu_mode}"
)
```

通过日志可以监控：
- GPU 使用情况
- 降级事件频率
- 性能对比

## 开发指南

### 添加新的 GPU 加速函数

1. 在 `compute_kernels.py` 中实现函数：

```python
from AppleStockChecker.utils.gpu_utils import get_array_module, gpu_fallback, to_numpy

@gpu_fallback
def my_new_computation(data: List[float], use_gpu: Optional[bool] = None) -> float:
    """
    新的 GPU 加速计算

    参数:
        data: 输入数据
        use_gpu: 是否使用 GPU（None=自动检测）

    返回:
        计算结果
    """
    xp = get_array_module(use_gpu)
    arr = xp.asarray(data, dtype=xp.float64)

    # 使用 xp (cupy 或 numpy) 进行计算
    result = xp.mean(arr)

    # 返回 Python 标量（自动转换）
    return float(result)
```

2. 在 `timestamp_alignment_task.py` 中使用：

```python
from AppleStockChecker.tasks.compute_kernels import my_new_computation

def _my_wrapper(data):
    """包装函数，调用 GPU 版本"""
    use_gpu = _should_use_gpu()
    return my_new_computation(data, use_gpu=use_gpu)
```

### 测试 GPU 功能

#### 单元测试

```python
# tests/test_gpu_utils.py
from AppleStockChecker.utils.gpu_utils import get_array_module, reset_gpu_state

def test_cpu_fallback():
    """测试 CPU 降级"""
    reset_gpu_state()
    xp = get_array_module(use_gpu=False)
    assert xp.__name__ == 'numpy'

def test_auto_detect():
    """测试自动检测"""
    reset_gpu_state()
    xp = get_array_module(use_gpu=None)
    # xp 可能是 cupy 或 numpy，取决于环境
    assert xp.__name__ in ['cupy', 'numpy']
```

#### 集成测试

```python
# tests/test_compute_kernels.py
from AppleStockChecker.tasks.compute_kernels import compute_std, compute_quantile

def test_compute_std_cpu():
    """测试 CPU 模式的标准差计算"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = compute_std(data, use_gpu=False)
    assert abs(result - 1.414) < 0.01  # 约等于 sqrt(2)

def test_compute_quantile_gpu():
    """测试 GPU 模式的分位数计算"""
    data = sorted([float(i) for i in range(1, 101)])
    result = compute_quantile(data, 0.5, use_gpu=True)
    assert abs(result - 50.0) < 1.0
```

## 部署指南

### Docker 部署

#### GPU 容器

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 安装 Python 和依赖
RUN apt-get update && apt-get install -y python3.11 python3-pip

# 安装依赖
COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements-gpu.txt

# 复制代码
COPY . /app
WORKDIR /app

# 启动 Celery worker
CMD ["celery", "-A", "your_project", "worker", "--loglevel=info"]
```

#### docker-compose.yml

```yaml
services:
  # GPU Worker
  celery-worker-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    runtime: nvidia
    environment:
      - PSTA_USE_GPU=auto
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # CPU Worker（备用）
  celery-worker-cpu:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - PSTA_USE_GPU=false
```

### ECS 部署

#### GPU 实例配置

1. 选择 GPU 实例类型（如 `ecs.gn7i-c8g1.2xlarge`）
2. 安装 NVIDIA 驱动和 CUDA
3. 配置环境变量 `PSTA_USE_GPU=auto`

#### CPU 实例配置

1. 使用普通计算实例
2. 配置环境变量 `PSTA_USE_GPU=false`

## 故障排查

### GPU 不可用

**症状**: 日志显示 `GPU可用=False`

**排查步骤**:
1. 检查是否安装 cupy: `pip list | grep cupy`
2. 检查 CUDA 驱动: `nvidia-smi`
3. 检查 cupy 能否导入: `python -c "import cupy; print(cupy.__version__)"`

### GPU OOM（显存不足）

**症状**: 日志显示 `GPU 计算失败 (out of memory)`

**解决方案**:
1. 系统会自动降级到 CPU，无需人工干预
2. 如果频繁出现，考虑：
   - 减少 Celery 并发数
   - 增加 GPU 显存
   - 使用批处理优化

### 性能未提升

**症状**: GPU 模式反而更慢

**可能原因**:
1. 数据量太小，GPU 初始化开销大于收益
2. 频繁的 CPU-GPU 数据传输
3. GPU 资源竞争

**解决方案**:
- 对于小数据集，使用 `PSTA_USE_GPU=false`
- 优化批处理逻辑，减少数据传输

## 未来优化

1. **批量优化**: 对多个 iPhone 型号并行计算
2. **流水线优化**: 异步 GPU 计算 + CPU 数据准备
3. **动态调度**: 根据数据大小自动选择 GPU/CPU
4. **性能监控**: 添加 GPU 使用率、显存占用等指标

## 参考资料

- [CuPy 官方文档](https://docs.cupy.dev/)
- [CUDA 安装指南](https://docs.nvidia.com/cuda/)
- [NumPy 与 CuPy 兼容性](https://docs.cupy.dev/en/stable/reference/comparison.html)
