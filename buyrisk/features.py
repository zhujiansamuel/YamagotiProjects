import math
from typing import Dict, Tuple, List


def z_from_quantile(q: float) -> float:
    """
    根据分位数返回标准正态分布的 Z 值

    Args:
        q: 分位数 (0-1)

    Returns:
        对应的 Z 值
    """
    table = {
        0.90: 1.281,
        0.95: 1.645,
        0.975: 1.960,
        0.99: 2.326
    }
    return table.get(round(q, 3), 1.645)


def drift_vol(bids: List[float], steps_tau: int) -> Tuple[float, float, float, float]:
    """
    计算价格序列的漂移和波动

    Args:
        bids: 价格序列
        steps_tau: τ 对应的步数

    Returns:
        (mu_step, sigma_step, mu_tau, sigma_tau)
    """
    if len(bids) < 2:
        return 0.0, 0.0, 0.0, 0.0

    # 计算价格差分
    db = [bids[i] - bids[i-1] for i in range(1, len(bids))]

    # 单步漂移（均值）
    mu_step = sum(db) / len(db)

    # 样本标准差（ddof=1）
    mean_db = mu_step
    var = sum((x - mean_db)**2 for x in db) / max(1, (len(db) - 1))
    sigma_step = math.sqrt(var)

    # τ 步漂移和波动
    mu_tau = mu_step * steps_tau
    sigma_tau = sigma_step * math.sqrt(steps_tau)

    return mu_step, sigma_step, mu_tau, sigma_tau


def shadow_sell_price(
    b_last: float,
    mu_tau: float,
    sigma_tau: float,
    alpha: float,
    beta: float,
    d_liq: float,
    q: float
) -> Dict:
    """
    计算影子卖价

    Args:
        b_last: 最新收购价
        mu_tau: τ 步漂移
        sigma_tau: τ 步波动
        alpha: 固定价差
        beta: 比例系数
        d_liq: 清货折价
        q: 分位数

    Returns:
        包含 s_shadow, e_b_tau, z, ci 的字典
    """
    z = z_from_quantile(q)
    e_b_tau = b_last + mu_tau
    s_shadow = alpha + beta * e_b_tau - z * beta * sigma_tau - d_liq

    # 标准误差估计（用于置信区间）
    se = max(1.0, 0.1 * beta * z * sigma_tau)

    return {
        's_shadow': s_shadow,
        'e_b_tau': e_b_tau,
        'z': z,
        'ci': (s_shadow - 1.96 * se, s_shadow + 1.96 * se)
    }


def b_max(
    s_shadow: float,
    cost: float,
    mmin: float,
    fxbuf: float,
    i_count: int,
    i_star: float,
    lambda_I: float
) -> Tuple[float, float]:
    """
    计算安全买价上限

    Args:
        s_shadow: 影子卖价
        cost: 单位成本
        mmin: 最小利润
        fxbuf: FX 缓冲
        i_count: 当前库存数量
        i_star: 目标库存
        lambda_I: 库存惩罚系数

    Returns:
        (b_max, inv_penalty)
    """
    inv_penalty = lambda_I * (i_count - i_star)
    bmax = s_shadow - cost - mmin - fxbuf - inv_penalty
    return bmax, inv_penalty


def hour_factor(hour: int) -> float:
    """
    计算时段热度因子（0-23小时）

    Args:
        hour: 小时 (0-23)

    Returns:
        热度因子 (0.4-1.0)
    """
    return max(0.4, 0.6 + 0.4 * math.sin(math.pi * (hour - 6) / 12))


def lambda_of_B(
    bid: float,
    hour: int,
    b_ref: float,
    lambda_ref: float,
    b_elastic: float
) -> float:
    """
    供给曲线：根据报价计算到货率

    Args:
        bid: 报价
        hour: 当前小时
        b_ref: 参考价格
        lambda_ref: 参考到货率
        b_elastic: 价格弹性

    Returns:
        到货率（件/小时）
    """
    step = 100.0
    hf = hour_factor(hour)
    return max(0.0, lambda_ref * math.exp(b_elastic * (bid - b_ref) / step) * hf)


def invert_Bfill(
    q_star: float,
    hour: int,
    b_ref: float,
    lambda_ref: float,
    b_elastic: float
) -> float:
    """
    反解填充买价：根据目标进货量反推需要的报价

    Args:
        q_star: 目标进货量（件/小时）
        hour: 当前小时
        b_ref: 参考价格
        lambda_ref: 参考到货率
        b_elastic: 价格弹性

    Returns:
        填充买价
    """
    hf = hour_factor(hour)
    if q_star <= 0 or lambda_ref <= 0 or hf <= 0 or b_elastic == 0:
        return b_ref

    step = 100.0
    return b_ref + (step / b_elastic) * math.log(q_star / (lambda_ref * hf))


def wac_and_risk(
    costs: List[float],
    s_shadow: float,
    cost_unit: float,
    mmin: float
) -> Tuple[float, float, float]:
    """
    计算加权平均成本和风险指标

    Args:
        costs: 库存成本列表
        s_shadow: 影子卖价
        cost_unit: 单位成本
        mmin: 最小利润

    Returns:
        (wac, unit_shadow_pnl, mvar_total)
    """
    if not costs:
        return 0.0, s_shadow - cost_unit, 0.0

    wac = sum(costs) / len(costs)
    unit_shadow_pnl = s_shadow - wac - cost_unit
    mvar_unit = max(0.0, wac + cost_unit + mmin - s_shadow)
    mvar_total = mvar_unit * len(costs)

    return wac, unit_shadow_pnl, mvar_total


def fx_buffer_reco(
    b_last: float,
    fx_sigma_daily: float,
    multiplier: float = 2.0
) -> float:
    """
    计算 FX 缓冲建议

    Args:
        b_last: 最新收购价
        fx_sigma_daily: 日波动率
        multiplier: 乘数

    Returns:
        建议的 FX 缓冲金额
    """
    return b_last * fx_sigma_daily * multiplier
