"""
GPU 加速的特征计算模块

将原 features.py 中的核心计算迁移到 GPU，支持：
- 价格漂移和波动率计算（GPU 并行）
- 影子卖价置信区间（Bootstrap GPU 加速）
- 自动回退到 CPU（无 GPU 环境）
"""
import math
from .accel import drift_vol as drift_vol_accel, bootstrap_shadow_ci
from .accel import ACCEL_BACKEND


def z_from_quantile(q: float) -> float:
    """分位数对应的 z 值（正态分布）"""
    table = {0.90: 1.281, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}
    return table.get(round(q, 3), 1.645)


def drift_vol(bids_np, steps_tau):
    """
    计算价格漂移和波动率（透传到 GPU/CPU 的加速实现）

    输入：
        bids_np: numpy 数组，历史价格序列
        steps_tau: 时间步数

    返回：(mu_step, sigma_step, mu_tau, sigma_tau)
    """
    return drift_vol_accel(bids_np, steps_tau)


def shadow_sell_price_with_ci(
    b_last: float, mu_tau: float, sigma_tau: float,
    alpha: float, beta: float, d_liq: float, q: float,
    bids_np=None, steps_tau=None, ci_bootstrap_n: int = 0
):
    """
    计算影子卖价及其置信区间（GPU 加速 Bootstrap）

    输入：
        b_last: 最新价格
        mu_tau: tau 期漂移
        sigma_tau: tau 期波动率
        alpha, beta, d_liq: 影子价格参数
        q: 分位数
        bids_np: 原始价格序列（用于 Bootstrap）
        steps_tau: 时间步数（用于 Bootstrap）
        ci_bootstrap_n: Bootstrap 样本数（0 = 不计算 CI）

    返回：dict
        - s_shadow: 影子卖价
        - e_b_tau: 期望未来价
        - z: 分位数 z 值
        - ci: 置信区间 (low, high) 或 (None, None)
        - backend: 使用的后端（cupy/torch/numpy）
    """
    z = z_from_quantile(q)
    e_b_tau = b_last + mu_tau
    sshadow = alpha + beta * e_b_tau - z * beta * sigma_tau - d_liq

    # 置信区间：如果给了序列 & 步长，并开启 bootstrap，就用 GPU 自助法；否则返回 None
    ci = (None, None)
    if bids_np is not None and steps_tau is not None and ci_bootstrap_n and ci_bootstrap_n > 0:
        lo, hi = bootstrap_shadow_ci(
            bids_np=bids_np, steps_tau=steps_tau,
            alpha=alpha, beta=beta, d_liq=d_liq, z_value=z,
            n_boot=ci_bootstrap_n
        )
        if lo is not None:
            ci = (lo, hi)

    return dict(
        s_shadow=float(sshadow),
        e_b_tau=float(e_b_tau),
        z=z,
        ci=ci,
        backend=ACCEL_BACKEND
    )


# ========= 以下保留原 features.py 中不需 GPU 加速的函数 =========

def b_max(s_shadow: float, d_op: float, d_fx: float, inv_qty: float, h: float) -> float:
    """
    安全买价上限（考虑库存惩罚）

    输入：
        s_shadow: 影子卖价
        d_op: 运营折扣
        d_fx: 汇率缓冲
        inv_qty: 当前库存数量
        h: 库存成本率

    返回：安全买价上限
    """
    penalty = h * inv_qty
    return s_shadow - d_op - d_fx - penalty


def hour_factor(hour: int) -> float:
    """
    小时效应因子（9-21 正常，其他时段降低）

    输入：
        hour: 小时 (0-23)

    返回：供给因子
    """
    return 1.0 if 9 <= hour <= 21 else 0.5


def lambda_of_B(
    b_offer: float, b_ref: float, lambda_ref: float,
    b_elastic: float, hour: int
) -> float:
    """
    供给曲线 λ(B)

    输入：
        b_offer: 出价
        b_ref: 参考价格
        lambda_ref: 参考供给率
        b_elastic: 价格弹性
        hour: 当前小时

    返回：预期成交率
    """
    db = (b_offer - b_ref) / 100.0
    hf = hour_factor(hour)
    return lambda_ref * math.exp(b_elastic * db) * hf


def invert_Bfill(
    target_qty: float, b_ref: float, lambda_ref: float,
    b_elastic: float, hour: int, eps=1e-6
) -> float:
    """
    反推成交价（牛顿法求解 λ(B) = target_qty）

    输入：
        target_qty: 目标成交率
        b_ref, lambda_ref, b_elastic: 供给曲线参数
        hour: 当前小时
        eps: 收敛精度

    返回：成交价格
    """
    hf = hour_factor(hour)
    if target_qty <= 0.0 or lambda_ref <= 0.0 or hf <= 0.0:
        return b_ref

    # λ(B) = lambda_ref * exp(b_elastic * (B - b_ref)/100) * hf = target_qty
    # => (B - b_ref)/100 = ln(target_qty / (lambda_ref * hf)) / b_elastic
    if abs(b_elastic) < 1e-9:
        return b_ref

    db_100 = math.log(target_qty / (lambda_ref * hf)) / b_elastic
    return b_ref + db_100 * 100.0


def wac_and_risk(lots, s_shadow: float):
    """
    加权平均成本和风险敞口

    输入：
        lots: 库存批次列表 [(qty, cost), ...]
        s_shadow: 影子卖价

    返回：(wac, margin_at_risk)
        - wac: 加权平均成本
        - margin_at_risk: 风险敞口 (s_shadow - wac) * total_qty
    """
    total_qty = sum(qty for qty, _ in lots)
    if total_qty <= 0:
        return 0.0, 0.0

    wac = sum(qty * cost for qty, cost in lots) / total_qty
    mar = (s_shadow - wac) * total_qty
    return wac, mar


def fx_buffer_reco(fx_sigma_daily: float, days_hold: float = 7.0, q: float = 0.95) -> float:
    """
    汇率缓冲建议

    输入：
        fx_sigma_daily: 日汇率波动率
        days_hold: 持仓天数
        q: 分位数

    返回：汇率缓冲金额
    """
    z = z_from_quantile(q)
    return z * fx_sigma_daily * math.sqrt(days_hold)
