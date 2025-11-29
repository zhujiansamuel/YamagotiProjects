from django.utils import timezone
from django.conf import settings
from celery import shared_task
from datetime import datetime, timedelta
from .adapters import fetch_price_series, fetch_inventory_costs, list_skus
from .models import (
    DecisionRun, SupplyCurveParam, ShadowParam, PurchaseLog,
    SellProxy, ClearanceEvent, FxDaily, CoverageReport
)
from .features_gpu import (
    drift_vol, shadow_sell_price_with_ci, b_max, lambda_of_B, invert_Bfill,
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

    # 转为 numpy 数组用于 GPU 加速
    import numpy as np
    bids_np = np.asarray(bids, dtype=float)

    # 计算漂移和波动（GPU 加速）
    mu_step, sigma_step, mu_tau, sigma_tau = drift_vol(bids_np, steps_tau)

    # 计算影子卖价及置信区间（GPU 加速 Bootstrap）
    ci_bootstrap_n = getattr(settings, "BUY_RISK_SHADOW_CI_BOOTSTRAP_N", 0)
    ss = shadow_sell_price_with_ci(
        b_last, mu_tau, sigma_tau,
        p["alpha"], p["beta"], p["d_liq"], p["q"],
        bids_np=bids_np, steps_tau=steps_tau,
        ci_bootstrap_n=ci_bootstrap_n
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

    # 保存决策结果（将 CI 和后端信息添加到 params_json）
    params_with_ci = dict(p)
    params_with_ci["s_shadow_ci"] = ss["ci"]
    params_with_ci["accel_backend"] = ss.get("backend", "numpy")

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
        params_json=params_with_ci
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
            import numpy as np
            bids_np = np.asarray(bids, dtype=float)
            mu_step, sigma_step, mu_tau, sigma_tau = drift_vol(bids_np, steps_tau)
            b_last = float(bids[-1])

            # 回测时不需要 CI，所以 ci_bootstrap_n=0
            ss = shadow_sell_price_with_ci(
                b_last, mu_tau, sigma_tau,
                p["alpha"], p["beta"], p["d_liq"], p["q"],
                bids_np=None, steps_tau=None, ci_bootstrap_n=0
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


# ============================================================
# 30分钟聚合任务（门店层 → 市场层）
# ============================================================

@shared_task(name="buyrisk.tasks.aggregate_shop_30m")
def aggregate_shop_30m():
    """
    门店层：对 PurchasingShopTimeAnalysis 在近 N 天内进行 30m 聚合。

    逻辑：
      group by (shop, iphone, bin_start)：
        avg_new = 该桶内均值（仅 New_Product_Price，忽略空值）
        rec_cnt/min_ts/max_ts
    upsert 到 ShopIphoneAgg30m

    Returns:
        聚合结果统计
    """
    from django.apps import apps
    from collections import defaultdict
    import numpy as np
    from .models import ShopIphoneAgg30m
    from .aggregations import floor_to_30m

    # 可调整的聚合窗口：默认回补近 3 天（保证迟到数据也能被重算）
    AGG_BACKFILL_DAYS = getattr(settings, "BUY_RISK_AGG_BACKFILL_DAYS", 3)

    # 获取 PurchasingShopTimeAnalysis 模型
    PurchasingShopTimeAnalysis = apps.get_model(
        "AppleStockChecker", "PurchasingShopTimeAnalysis"
    )

    since = timezone.now() - timedelta(days=AGG_BACKFILL_DAYS)
    qs = (PurchasingShopTimeAnalysis.objects
          .filter(Timestamp_Time__gte=since)
          .select_related("shop", "iphone")
          .only("shop_id", "iphone_id", "Timestamp_Time", "New_Product_Price"))

    # 先在 Python 侧做桶归并，避免跨数据库差异
    buckets = defaultdict(lambda: {
        "vals": [],
        "min_ts": None,
        "max_ts": None
    })

    for rec in qs.iterator(chunk_size=5000):
        v = rec.New_Product_Price
        if v is None:  # 只看 New_Product_Price
            continue

        bin_start = floor_to_30m(rec.Timestamp_Time)
        key = (rec.shop_id, rec.iphone_id, bin_start)
        b = buckets[key]

        b["vals"].append(float(v))

        if b["min_ts"] is None or rec.Timestamp_Time < b["min_ts"]:
            b["min_ts"] = rec.Timestamp_Time
        if b["max_ts"] is None or rec.Timestamp_Time > b["max_ts"]:
            b["max_ts"] = rec.Timestamp_Time

    # 写入/更新
    upserted = 0
    for (shop_id, iphone_id, bin_start), v in buckets.items():
        avg_new = float(np.mean(v["vals"])) if v["vals"] else None
        rec_cnt = len(v["vals"])

        ShopIphoneAgg30m.objects.update_or_create(
            shop_id=shop_id,
            iphone_id=iphone_id,
            bin_start=bin_start,
            defaults=dict(
                avg_new=avg_new,
                rec_cnt=rec_cnt,
                min_src_ts=v["min_ts"],
                max_src_ts=v["max_ts"]
            )
        )
        upserted += 1

    return {"shop_bins": len(buckets), "upserted": upserted}


@shared_task(name="buyrisk.tasks.aggregate_market_30m")
def aggregate_market_30m():
    """
    市场层：把门店层聚合结果在近 N 天内进一步跨门店聚合为"市场指数"。

    逻辑：
      对每个 (iphone, bin_start)：
        med_new / mean_new / tmean_new (10% 截断均值)
        bid_pref = med_new

    Returns:
        聚合结果统计
    """
    from collections import defaultdict
    import numpy as np
    from .models import ShopIphoneAgg30m, MarketIphoneAgg30m

    AGG_BACKFILL_DAYS = getattr(settings, "BUY_RISK_AGG_BACKFILL_DAYS", 3)

    since = timezone.now() - timedelta(days=AGG_BACKFILL_DAYS)
    qs = (ShopIphoneAgg30m.objects
          .filter(bin_start__gte=since)
          .only("iphone_id", "bin_start", "avg_new", "shop_id"))

    # 收集 -> 聚合
    groups = defaultdict(lambda: {
        "vals": [],
        "shop_ids": set()
    })

    for r in qs.iterator(chunk_size=5000):
        key = (r.iphone_id, r.bin_start)
        g = groups[key]

        if r.avg_new is not None:
            g["vals"].append(float(r.avg_new))
            g["shop_ids"].add(r.shop_id)

    def _tmean(arr, trim=0.1):
        """截断均值：去掉两端各 trim 比例的数据"""
        if not arr:
            return None
        a = sorted(arr)
        k = int(len(a) * trim)
        a = a[k: len(a) - k] if len(a) > 2 * k else a
        return float(np.mean(a)) if a else None

    upserts = 0
    for (iphone_id, bin_start), g in groups.items():
        vals = g["vals"]
        if not vals:
            continue

        # 中位数
        med_new = float(np.median(vals))
        # 均值
        mean_new = float(np.mean(vals))
        # 截断均值
        tmean_new = _tmean(vals)

        # 首选报价 = 中位数
        bid_pref = med_new

        # sku 直接用 iphone_id 的字符串
        sku = str(iphone_id)

        # 门店数
        shops_included = len(g["shop_ids"])

        MarketIphoneAgg30m.objects.update_or_create(
            iphone_id=iphone_id,
            bin_start=bin_start,
            defaults=dict(
                sku=sku,
                med_new=med_new,
                mean_new=mean_new,
                tmean_new=tmean_new,
                bid_pref=bid_pref,
                shops_included=shops_included
            )
        )
        upserts += 1

    return {"market_bins": upserts}


# ============================================================
# AutoML 训练任务（GPU 队列）
# ============================================================

@shared_task(name="buyrisk.tasks.run_automl")
def run_automl(job_id: str, sku_list=None, hours_back: int = 60*24*14, lr=0.05, epochs=200):
    """
    在 GPU 上训练供给曲线（λ(B)）参数：b_elastic / lambda_ref

    路由到 automl_gpu 队列，由 GPU ECS（automl_gpu worker）专门消费

    Args:
        job_id: 任务 ID（用于审计/日志）
        sku_list: 指定一批 SKU（默认自动发现近 14 天有数据的 SKU）
        hours_back: 训练数据回溯小时数（默认 14 天）
        lr: 学习率
        epochs: 训练轮数

    Returns:
        训练结果字典 {"job_id": ..., "results": {sku: {...}, ...}}
    """
    from django.db.models.functions import TruncHour
    from django.db.models import Avg, Sum
    import math
    import numpy as np

    # 自动发现 SKU
    if not sku_list:
        sku_list = list(set(list_skus()[:20]))  # 限制数量避免过长

    now = timezone.now()
    since = now - timedelta(hours=hours_back)

    results = {}

    # 尝试加载 PyTorch（GPU 优先）
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch = None
        device = "cpu"

    for sku in sku_list:
        # 1) 取样本（按小时聚合：cnt/price）
        qs = (PurchaseLog.objects
              .filter(sku=sku, ts__gte=since)
              .annotate(ts_hour=TruncHour("ts"))
              .values("ts_hour")
              .annotate(cnt=Sum("acquired"), price=Avg("offer_price"))
              .order_by("ts_hour"))
        rows = list(qs)

        if len(rows) < 24:
            results[sku] = {"status": "insufficient", "samples": len(rows)}
            continue

        # 2) 构造特征：x = (price - b_ref)/100；y = cnt；hour one-hot
        b_ref = 3000.0
        xs = []
        ys = []
        hours = []

        for r in rows:
            if r["price"] is None:
                continue
            xs.append((float(r["price"]) - b_ref) / 100.0)
            ys.append(float(r["cnt"] or 0.0))
            hours.append(r["ts_hour"].hour)

        X = np.asarray(xs, dtype=np.float32)
        Y = np.asarray(ys, dtype=np.float32)
        H = np.asarray(hours, dtype=np.int64)

        if X.size < 24:
            results[sku] = {"status": "insufficient", "samples": X.size}
            continue

        # 3) GPU 训练（如果可用）
        if torch and device == "cuda":
            # Poisson 回归： log λ = a + b·X + hour_embed[H]
            torch.manual_seed(42)
            x = torch.tensor(X, device=device).unsqueeze(1)  # [N,1]
            y = torch.tensor(Y, device=device)               # [N]
            h = torch.tensor(H, device=device)               # [N]

            a = torch.nn.Parameter(torch.zeros(1, device=device))
            b = torch.nn.Parameter(torch.zeros(1, device=device))
            emb = torch.nn.Embedding(24, 1, device=device)  # 小时效应
            opt = torch.optim.Adam([a, b, emb.weight], lr=lr)

            for _ in range(epochs):
                lam_log = a + b * x + emb(h)[:, 0]   # [N]
                lam = torch.exp(lam_log).clamp_max(1e6)
                # Poisson NLL（不含常数项）
                loss = (lam - y * lam_log).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # 4) 读参：b 即为价格弹性；lambda_ref ≈ exp(a + E[hour_embed])
            b_elastic = float(b.detach().cpu().item())
            hour_effect = float(emb.weight.detach().mean().cpu().item())
            lambda_ref = math.exp(float(a.detach().cpu().item()) + hour_effect)

            training_method = "gpu_torch"
        else:
            # --- 无 GPU 时的最小二乘近似 ---
            # log(cnt+eps) ~ a + b·X + hour dummies
            eps = 1e-3
            y_log = np.log(Y + eps)

            # 小时 dummy
            Hdm = np.eye(24, dtype=np.float32)[H]  # [N,24]
            Z = np.concatenate([
                np.ones((X.shape[0], 1), np.float32),
                X.reshape(-1, 1),
                Hdm
            ], axis=1)

            coef, *_ = np.linalg.lstsq(Z, y_log, rcond=None)
            a_hat = float(coef[0])
            b_elastic = float(coef[1])
            hour_effect = float(coef[2:].mean()) if coef.shape[0] > 2 else 0.0
            lambda_ref = math.exp(a_hat + hour_effect)

            training_method = "cpu_numpy"

        # 5) 保存到数据库
        SupplyCurveParam.objects.update_or_create(
            sku=sku,
            defaults=dict(
                b_ref=b_ref,
                lambda_ref=lambda_ref,
                b_elastic=b_elastic
            )
        )

        results[sku] = {
            "status": "ok",
            "b_elastic": b_elastic,
            "lambda_ref": lambda_ref,
            "method": training_method,
            "samples": len(xs)
        }

    return {"job_id": job_id, "results": results}
