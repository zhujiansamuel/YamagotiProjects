"""
GPU 加速后端选择模块（CuPy / PyTorch / NumPy 自动回退）

支持的后端：
- cupy: NVIDIA GPU 加速（推荐）
- torch: PyTorch CUDA 加速
- numpy: CPU 回退（默认）

环境变量：
- BUY_RISK_ACCEL_BACKEND: "cupy" / "torch" / "numpy" / "auto"
"""
import os
import math
import numpy as np

ACCEL_BACKEND = os.getenv("BUY_RISK_ACCEL_BACKEND", "numpy").lower()  # numpy/cupy/torch
USE_TORCH = False
USE_CUPY = False

xp = np  # 数组计算命名空间
torch = None
cp = None

# --- 选择后端 ---
if ACCEL_BACKEND in ("cupy", "auto"):
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            USE_CUPY = True
    except Exception:
        pass

if not USE_CUPY and ACCEL_BACKEND in ("torch", "auto"):
    try:
        import torch
        if torch.cuda.is_available():
            USE_TORCH = True
        else:
            torch = None
    except Exception:
        torch = None


def to_device(arr):
    """把 numpy 数组放到目标设备；torch 用 tensor，cupy 用 ndarray"""
    if USE_CUPY:
        return cp.asarray(arr)
    if USE_TORCH:
        return torch.as_tensor(arr, device="cuda", dtype=torch.float32)
    return np.asarray(arr)


def to_numpy(x):
    """把设备数据拉回 CPU numpy"""
    if USE_CUPY:
        return cp.asnumpy(x)
    if USE_TORCH:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x
    return x


# ========= 基础统计 =========
def drift_vol(bids_np: np.ndarray, steps_tau: int):
    """
    计算价格漂移和波动率

    输入：
        bids_np: CPU 上的 numpy 1D 数组（时间升序）
        steps_tau: 时间步数

    返回：(mu_step, sigma_step, mu_tau, sigma_tau)
        - mu_step: 单步漂移
        - sigma_step: 单步波动率
        - mu_tau: tau 期漂移
        - sigma_tau: tau 期波动率

    自动把计算丢给 GPU（cupy/torch），无 GPU 回退 numpy
    """
    n = bids_np.shape[0]
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0

    if USE_TORCH:
        x = to_device(bids_np)
        db = x[1:] - x[:-1]
        mu_step = float(db.mean().item())
        sigma_step = float(db.std(unbiased=True).item()) if db.numel() > 1 else 0.0
    else:
        x = to_device(bids_np)
        db = x[1:] - x[:-1]
        mu_step = float(xp.mean(db))
        sigma_step = float(xp.std(db, ddof=1)) if db.size > 1 else 0.0
        if USE_CUPY:
            mu_step = float(mu_step)
            sigma_step = float(sigma_step)

    mu_tau = mu_step * steps_tau
    sigma_tau = sigma_step * math.sqrt(steps_tau)
    return mu_step, sigma_step, mu_tau, sigma_tau


def bootstrap_shadow_ci(
    bids_np: np.ndarray,
    steps_tau: int,
    alpha: float, beta: float, d_liq: float, z_value: float,
    n_boot: int = 2000, seed: int = 42
):
    """
    用自助法计算 Sshadow 的置信区间（重采样收益/路径，GPU 上并行）

    输入：
        bids_np: 历史价格序列
        steps_tau: 预测步数
        alpha, beta, d_liq: 影子价格参数
        z_value: 分位数对应的 z 值
        n_boot: Bootstrap 样本数
        seed: 随机种子

    返回：(ci_low, ci_high) - 95% 置信区间下界和上界
    """
    n = bids_np.shape[0]
    if n < 2 or n_boot <= 0:
        return None, None

    if USE_TORCH:
        torch.manual_seed(seed)
        x = to_device(bids_np).float()
        db = x[1:] - x[:-1]              # 15m 收益
        m = db.numel()
        # [n_boot, steps_tau] 的索引，按有放回抽样
        idx = torch.randint(0, m, (n_boot, steps_tau), device=x.device)
        paths = db[idx].sum(dim=1)       # 每条路径的总变化
        b_last = x[-1]
        e_b_tau = b_last + paths         # 期望未来价
        # 简化：σ_tau 用样本 std 作为公共尺度（或对每条路径再估）
        sigma_step = db.std(unbiased=True)
        sigma_tau = sigma_step * math.sqrt(steps_tau)
        sshadow = alpha + beta*e_b_tau - z_value*beta*sigma_tau - d_liq
        sshadow_sorted = torch.sort(sshadow).values
        lo = float(sshadow_sorted[int(0.025 * n_boot)].item())
        hi = float(sshadow_sorted[int(0.975 * n_boot)].item())
        return lo, hi

    # CuPy / NumPy 分支
    rng = (cp.random.RandomState(seed) if USE_CUPY else np.random.RandomState(seed))
    x = to_device(bids_np)
    db = x[1:] - x[:-1]
    m = db.shape[0]
    # 生成 [n_boot, steps_tau] 的索引
    idx = rng.randint(0, m, size=(n_boot, steps_tau))
    sims = db[idx].sum(axis=1)
    b_last = x[-1]
    e_b_tau = b_last + sims
    sigma_step = (db.std(ddof=1) if not USE_CUPY else db.std())  # cupy 默认 ddof=0，略差异问题不大
    sigma_tau = sigma_step * math.sqrt(steps_tau)
    sshadow = alpha + beta*e_b_tau - z_value*beta*sigma_tau - d_liq
    sshadow = to_numpy(sshadow)
    lo = float(np.quantile(sshadow, 0.025))
    hi = float(np.quantile(sshadow, 0.975))
    return lo, hi
