from django.utils import timezone
from django.conf import settings
from celery import shared_task
from datetime import datetime, timedelta
from .adapters import fetch_price_series, fetch_inventory_costs, list_skus
from .models import (
    DecisionRun, SupplyCurveParam, ShadowParam, PurchaseLog,
    SellProxy, ClearanceEvent, FxDaily, CoverageReport
)
from .features import (
    drift_vol, shadow_sell_price, b_max, lambda_of_B, invert_Bfill,
    wac_and_risk, fx_buffer_reco
)


def _get_params(sku: str) -> dict:
    """合并默认参数 + 训练参数（若有）"""
    p = dict(settings.BUY_RISK_DEFAULTS)

    # 获取影子卖价参数
    sp = ShadowParam.objects.filter(sku=sku).order_by("-trained_at").first()
    if sp:
        p.update(dict(
            alpha=sp.alpha,
            beta=sp.beta,
            d_liq=sp.d_liq,
            q=sp.q,
            tau_hours=sp.tau_hours,
            fx_sigma_daily=sp.fx_sigma_daily
        ))

    # 获取供给曲线参数
    sup = SupplyCurveParam.objects.filter(sku=sku).order_by("-trained_at").first()
    if sup:
        p.update(dict(
            b_ref=sup.b_ref,
            lambda_ref=sup.lambda_ref,
            b_elastic=sup.b_elastic
        ))

    return p


@shared_task(name="buyrisk.tasks.compute_all_skus")
def compute_all_skus():
    """计算所有 SKU 的决策（每 15 分钟运行）"""
    results = []
    for sku in list_skus():
        try:
            result = compute_decision_for_sku(sku)
            results.append(result)
        except Exception as e:
            results.append({"sku": sku, "error": str(e)})
    return results


@shared_task(name="buyrisk.tasks.compute_decision_for_sku")
def compute_decision_for_sku(sku: str):
    """
    核心决策：每 15 分钟跑一次

    Args:
        sku: SKU 标识符

    Returns:
        决策结果字典
    """
    # 获取价格序列
    series = fetch_price_series(sku)
    if len(series) < 4:  # 样本太少
        return {"sku": sku, "error": "insufficient price series"}

    ts_list, bids = zip(*series)
    b_last = float(bids[-1])

    # 获取参数
    p = _get_params(sku)
    steps_tau = max(1, round(
        p["tau_hours"] * 60 / settings.BUY_RISK_PRICE_STEP_MINUTES
    ))

    # 计算漂移和波动
    mu_step, sigma_step, mu_tau, sigma_tau = drift_vol(list(bids), steps_tau)

    # 计算影子卖价
    ss = shadow_sell_price(
        b_last, mu_tau, sigma_tau,
        p["alpha"], p["beta"], p["d_liq"], p["q"]
    )

    # 获取库存
    costs = fetch_inventory_costs(sku)

    # 计算安全买价上限
    bmax, inv_penalty = b_max(
        ss["s_shadow"], p["cost_per_unit"], p["min_margin"], p["fx_buffer"],
        len(costs), p["i_star"], p["lambda_I"]
    )

    # 填充买价（按当前小时的时段热度）
    now_hour = timezone.now().hour
    bfill = invert_Bfill(
        p["q_star"], now_hour,
        p["b_ref"], p["lambda_ref"], p["b_elastic"]
    )

    # 最终买价
    bfinal = min(bmax, bfill)
    lam_final = lambda_of_B(
        bfinal, now_hour,
        p["b_ref"], p["lambda_ref"], p["b_elastic"]
    )

    # 库存风险
    wac, unit_shadow_pnl, mvar_total = wac_and_risk(
        costs, ss["s_shadow"], p["cost_per_unit"], p["min_margin"]
    )
    daily_out = lam_final * 24.0
    doh = (len(costs) / daily_out) if daily_out > 0 else float("inf")
    fx_reco = fx_buffer_reco(b_last, p["fx_sigma_daily"], 2.0)

    # 闸门状态
    gate_gap = (bfill - bmax) / bfill if bfill > 0 else 0.0
    gate = "OK" if bfill <= bmax else ("SLOW" if gate_gap <= 0.04 else "STOP")

    # 瀑布拆解
    decomposition = {
        "E[B_tau]": ss["e_b_tau"],
        "-z_sigma": -ss["z"] * p["beta"] * sigma_tau,
        "-d_liq": -p["d_liq"],
        "Sshadow": ss["s_shadow"] - ss["e_b_tau"] + ss["z"] * p["beta"] * sigma_tau + p["d_liq"],
        "-cost": -p["cost_per_unit"],
        "-min_margin": -p["min_margin"],
        "-fx_buffer": -p["fx_buffer"],
        "inv_penalty": -inv_penalty,
        "Bmax": bmax - (ss["s_shadow"] - p["cost_per_unit"] - p["min_margin"] - p["fx_buffer"] - inv_penalty)
    }

    # 保存决策结果
    out = DecisionRun.objects.create(
        ts_calc=timezone.now(),
        sku=sku,
        s_shadow=ss["s_shadow"],
        e_b_tau=ss["e_b_tau"],
        mu_step=mu_step,
        sigma_step=sigma_step,
        mu_tau=mu_tau,
        sigma_tau=sigma_tau,
        b_max=bmax,
        b_fill=bfill,
        b_final=bfinal,
        gate=gate,
        gate_gap_ratio=gate_gap,
        lam_final_per_hour=lam_final,
        wac=wac,
        unit_shadow_pnl=unit_shadow_pnl,
        mvar_total=mvar_total,
        doh_days=doh,
        daily_outflow=daily_out,
        fx_buffer_reco=fx_reco,
        inv_penalty=inv_penalty,
        decomposition_json=decomposition,
        params_json=p
    )

    return {
        "id": out.id,
        "sku": sku,
        "gate": gate,
        "b_final": bfinal
    }


@shared_task(name="buyrisk.tasks.train_models_all_skus")
def train_models_all_skus():
    """
    每天跑一次：λ(B)、α/β、δliq、FX

    Returns:
        训练结果字典
    """
    from django.db.models.functions import TruncHour
    from django.db.models import Avg, Sum, Count, Q
    import math

    results = {}

    for sku in set(list_skus()):
        # 1) 供给曲线：log(cnt+ε) ~ a + b*(price - b_ref)/100
        qs = (PurchaseLog.objects
              .filter(sku=sku)
              .annotate(ts_hour=TruncHour("ts"))
              .values("ts_hour")
              .annotate(cnt=Sum("acquired"), price=Avg("offer_price"))
              .order_by("ts_hour"))
        data = list(qs)

        if len(data) < 12:
            results.setdefault(sku, {})["supply"] = "default"
        else:
            eps = 1e-3
            b_ref_default = settings.BUY_RISK_DEFAULTS["b_ref"]
            xs = [((d["price"] or 0.0) - b_ref_default) / 100.0 for d in data]
            ys = [math.log((d["cnt"] or 0.0) + eps) for d in data]

            # 最小二乘（一元线性）
            n = len(xs)
            xbar = sum(xs) / n
            ybar = sum(ys) / n
            num = sum((xs[i] - xbar) * (ys[i] - ybar) for i in range(n))
            den = sum((xs[i] - xbar)**2 for i in range(n)) or 1.0
            b_elastic = num / den
            intercept = ybar - b_elastic * xbar
            lambda_ref = math.exp(intercept) / 0.7  # 折回日内热度均值

            SupplyCurveParam.objects.update_or_create(
                sku=sku,
                defaults=dict(
                    b_ref=b_ref_default,
                    lambda_ref=lambda_ref,
                    b_elastic=b_elastic
                )
            )
            results.setdefault(sku, {})["supply"] = "trained"

        # 2) α/β：代理卖价回归
        sp_qs = SellProxy.objects.filter(sku=sku).order_by("ts").values("ts", "sell_proxy")
        price_series = fetch_price_series(sku)

        if sp_qs and len(price_series) >= 4:
            # 时间近邻匹配
            from bisect import bisect_left
            ts_list, bids = zip(*price_series)
            xs, ys = [], []

            for row in sp_qs:
                ts = int(row["ts"].timestamp() * 1000)
                i = bisect_left(ts_list, ts)
                # 容忍 ±15m
                candidates = []
                if i > 0:
                    candidates.append((abs(ts_list[i-1] - ts), bids[i-1]))
                if i < len(ts_list):
                    candidates.append((abs(ts_list[i] - ts), bids[i]))
                if not candidates:
                    continue
                d, b = min(candidates, key=lambda x: x[0])
                if d <= 15 * 60 * 1000:
                    xs.append(float(b))
                    ys.append(float(row["sell_proxy"]))

            if len(xs) >= 12:
                # 一元线性回归
                n = len(xs)
                xbar = sum(xs) / n
                ybar = sum(ys) / n
                num = sum((xs[i] - xbar) * (ys[i] - ybar) for i in range(n))
                den = sum((xs[i] - xbar)**2 for i in range(n)) or 1.0
                beta = num / den
                alpha = ybar - beta * xbar

                ShadowParam.objects.update_or_create(
                    sku=sku,
                    defaults=dict(alpha=alpha, beta=beta)
                )
                results.setdefault(sku, {})["alpha_beta"] = "trained"
            else:
                results.setdefault(sku, {})["alpha_beta"] = "default"
        else:
            results.setdefault(sku, {})["alpha_beta"] = "default"

        # 3) δ_liq
        ce = ClearanceEvent.objects.filter(sku=sku).values_list("extra_discount", flat=True)
        if ce:
            import statistics
            dliq = statistics.median([float(x) for x in ce])
            ShadowParam.objects.update_or_create(sku=sku, defaults=dict(d_liq=dliq))
            results.setdefault(sku, {})["d_liq"] = "trained"
        else:
            results.setdefault(sku, {})["d_liq"] = "default"

        # 4) FX σ
        fx = FxDaily.objects.all().order_by("ts").values_list("fx", flat=True)
        if fx and len(fx) >= 6:
            import math
            vals = [float(x) for x in fx]
            returns = [math.log(vals[i] / vals[i-1]) for i in range(1, len(vals)) if vals[i-1] > 0]
            if len(returns) >= 2:
                import statistics
                sigma = statistics.stdev(returns)
                ShadowParam.objects.update_or_create(sku=sku, defaults=dict(fx_sigma_daily=sigma))
                results.setdefault(sku, {})["fx_sigma"] = "trained"
            else:
                results.setdefault(sku, {})["fx_sigma"] = "default"
        else:
            results.setdefault(sku, {})["fx_sigma"] = "default"

    return results


@shared_task(name="buyrisk.tasks.backtest_coverage_all_skus")
def backtest_coverage_all_skus(window_days: int = 30):
    """
    用 SellProxy 回测 Sshadow 覆盖率（VaR 覆盖）

    Args:
        window_days: 回测窗口天数

    Returns:
        回测报告字典
    """
    report = {}

    for sku in list_skus():
        sp = list(SellProxy.objects.filter(sku=sku).order_by("ts").values("ts", "sell_proxy"))
        if not sp:
            report[sku] = {"msg": "no proxy"}
            continue

        p = _get_params(sku)
        hits, total = 0, 0

        for row in sp:
            cutoff = row["ts"] - timedelta(days=window_days)
            # 取窗口内的价格序列
            series = [
                (ts, b) for ts, b in fetch_price_series(sku)
                if datetime.utcfromtimestamp(ts / 1000) >= cutoff
            ]
            if len(series) < 4:
                continue

            _, bids = zip(*series)
            steps_tau = max(1, round(
                p["tau_hours"] * 60 / settings.BUY_RISK_PRICE_STEP_MINUTES
            ))
            mu_step, sigma_step, mu_tau, sigma_tau = drift_vol(list(bids), steps_tau)
            b_last = float(bids[-1])

            ss = shadow_sell_price(
                b_last, mu_tau, sigma_tau,
                p["alpha"], p["beta"], p["d_liq"], p["q"]
            )

            total += 1
            hits += 1 if float(row["sell_proxy"]) >= ss["s_shadow"] else 0

        cov = (hits / total) if total > 0 else None
        CoverageReport.objects.create(
            sku=sku,
            window_days=window_days,
            q_target=p["q"],
            coverage_realized=cov,
            n=total
        )
        report[sku] = {"coverage": cov, "n": total, "q": p["q"]}

    return report
