# -*- coding: utf-8 -*-
"""
AutoML Celery Tasks - 三阶段因果分析 Pipeline (GPU Accelerated)
机器学习和深度学习分析任务 - 支持 GPU 加速与自动降级
"""
from __future__ import annotations
from celery import shared_task
from django.utils import timezone
from django.db import transaction
import pandas as pd
import numpy as np
import logging

# 导入 GPU 工具模块
from AppleStockChecker.utils.gpu_utils import (
    check_gpu_availability,
    get_array_module,
    to_cpu,
    to_gpu,
    ensure_cpu_for_pandas,
    get_gpu_memory_info,
)

logger = logging.getLogger(__name__)

# 在任务模块加载时检查 GPU
logger.info("Checking GPU availability for AutoML tasks...")
GPU_AVAILABLE = check_gpu_availability()
if GPU_AVAILABLE:
    logger.info("✓ GPU acceleration enabled for AutoML pipeline")
    mem_info = get_gpu_memory_info()
    logger.info(f"  GPU Memory: {mem_info['free_gb']:.2f} GB free / {mem_info['total_gb']:.2f} GB total")
else:
    logger.info("✗ GPU not available, using CPU for AutoML pipeline")


# ============================================================================
# 阶段 0: Job 调度与创建
# ============================================================================

@shared_task(name="automl.schedule_jobs", queue="default")
def schedule_automl_jobs():
    """
    周期性调度任务：创建 AutoML 因果分析 Job
    示例：分析过去7天的数据
    """
    from AppleStockChecker.models import AutomlCausalJob, Iphone, PurchasingShopTimeAnalysis
    from celery import group

    logger.info("Starting AutoML job scheduling...")

    now = timezone.now()
    window_end = now
    window_start = now - timezone.timedelta(days=7)

    # 查找近期有PSTA记录的活跃机种
    active_iphones = Iphone.objects.filter(
        purchasing_time_analysis__Timestamp_Time__gte=window_start
    ).distinct()

    logger.info(f"Found {active_iphones.count()} active iphones")

    jobs_to_process = []
    for iphone in active_iphones:
        job, created = AutomlCausalJob.objects.get_or_create(
            iphone=iphone,
            window_start=window_start,
            window_end=window_end,
            bucket_freq="10min",
            defaults={"priority": 100},
        )

        # 只处理需要预处理的 job
        if job.preprocessing_status in [
            AutomlCausalJob.StageStatus.PENDING,
            AutomlCausalJob.StageStatus.FAILED,
        ]:
            jobs_to_process.append(job.id)
            if created:
                logger.info(f"Created new job {job.id} for {iphone.part_number}")
            else:
                logger.info(f"Retrying job {job.id} for {iphone.part_number}")

    # 批量触发预处理任务
    if jobs_to_process:
        g = group(run_preprocessing_for_job.s(job_id) for job_id in jobs_to_process)
        g.apply_async(queue="automl_preprocessing")
        logger.info(f"Scheduled {len(jobs_to_process)} preprocessing jobs")
    else:
        logger.info("No jobs to process")

    return {"scheduled": len(jobs_to_process)}


# ============================================================================
# 阶段 1: 预处理 (Preprocessing-Rapid)
# ============================================================================

@shared_task(
    bind=True,
    name="automl.preprocessing_rapid",
    queue="automl_preprocessing",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def run_preprocessing_for_job(self, job_id: int):
    """
    Preprocessing-Rapid Task
    从 PurchasingShopTimeAnalysis 读取数据 → 生成预处理序列
    """
    from AppleStockChecker.models import (
        AutomlCausalJob,
        AutomlPreprocessedSeries,
        PurchasingShopTimeAnalysis,
    )

    logger.info(f"[Job {job_id}] Starting preprocessing...")

    job = AutomlCausalJob.objects.select_for_update().get(pk=job_id)

    # 如果已经成功,跳过
    if job.preprocessing_status == AutomlCausalJob.StageStatus.SUCCESS:
        logger.info(f"[Job {job_id}] Already preprocessed, skipping")
        return {"status": "already_done", "job_id": job_id}

    # 更新状态为运行中
    job.preprocessing_status = AutomlCausalJob.StageStatus.RUNNING
    job.preprocessing_started_at = timezone.now()
    job.last_error = None
    job.save(update_fields=["preprocessing_status", "preprocessing_started_at", "last_error"])

    try:
        # 1) 从 PSTA 读取原始对齐记录
        qs = PurchasingShopTimeAnalysis.objects.filter(
            iphone=job.iphone,
            Timestamp_Time__gte=job.window_start,
            Timestamp_Time__lt=job.window_end,
        ).select_related("shop")

        if not qs.exists():
            logger.warning(f"[Job {job_id}] No PSTA data found, skipping")
            job.preprocessing_status = AutomlCausalJob.StageStatus.SKIPPED
            job.preprocessing_finished_at = timezone.now()
            job.save(update_fields=["preprocessing_status", "preprocessing_finished_at"])
            return {"status": "skipped", "reason": "no_data"}

        logger.info(f"[Job {job_id}] Found {qs.count()} PSTA records")

        # 2) 转换为 DataFrame
        df = pd.DataFrame.from_records(
            qs.values(
                "shop_id",
                "iphone_id",
                "Timestamp_Time",
                "New_Product_Price",
                "Price_A",
                "Price_B",
            )
        )
        df.rename(columns={"Timestamp_Time": "timestamp"}, inplace=True)

        # 3) 选定价格：优先 A品 > B品 > 新品价
        df["price"] = df["Price_A"].fillna(df["Price_B"]).fillna(df["New_Product_Price"])
        df["price_source"] = "A"
        df.loc[df["Price_A"].isna() & df["Price_B"].notna(), "price_source"] = "B"
        df.loc[df["Price_A"].isna() & df["Price_B"].isna(), "price_source"] = "NEW"

        # 4) 按时间桶聚合
        df["bucket_ts"] = df["timestamp"].dt.floor(job.bucket_freq)

        # 5) 去除异常价格
        df = df[(df["price"] >= 10000) & (df["price"] <= 350000)]

        # 6) 对每个 (shop, bucket_ts) 聚合价格
        df_agg = (
            df.groupby(["shop_id", "iphone_id", "bucket_ts", "price_source"], as_index=False)
              .agg(price=("price", "mean"))
        )

        # 7) 计算 log_price / dlog / z-score (GPU 加速)
        df_agg = df_agg.sort_values(["shop_id", "bucket_ts"])

        try:
            # 尝试使用 GPU 加速 numpy 运算
            if GPU_AVAILABLE:
                logger.info(f"[Job {job_id}] Using GPU for preprocessing calculations")

                # 转换到 GPU
                xp = get_array_module()
                prices_gpu = to_gpu(df_agg["price"].values)

                # GPU 加速的 log 运算
                log_prices_gpu = xp.log(prices_gpu)
                df_agg["log_price"] = ensure_cpu_for_pandas(log_prices_gpu)
            else:
                # CPU 版本
                df_agg["log_price"] = np.log(df_agg["price"])

        except Exception as e:
            logger.warning(f"[Job {job_id}] GPU calculation failed: {e}, falling back to CPU")
            df_agg["log_price"] = np.log(df_agg["price"])

        # diff 运算（pandas groupby，保持在 CPU）
        df_agg["dlog_price"] = df_agg.groupby("shop_id", group_keys=False)["log_price"].diff()

        # 计算每个店铺的均值和标准差
        stats = (
            df_agg
            .groupby("shop_id")["dlog_price"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "dlog_mean", "std": "dlog_std"})
        )
        df_agg = df_agg.join(stats, on="shop_id")

        # 标准化 (GPU 加速)
        try:
            if GPU_AVAILABLE:
                xp = get_array_module()

                # 转换到 GPU
                dlog_gpu = to_gpu(df_agg["dlog_price"].fillna(0).values)
                mean_gpu = to_gpu(df_agg["dlog_mean"].fillna(0).values)
                std_gpu = to_gpu(df_agg["dlog_std"].fillna(1).values)

                # GPU 加速的标准化
                z_dlog_gpu = (dlog_gpu - mean_gpu) / xp.where(std_gpu > 0, std_gpu, 1.0)
                df_agg["z_dlog_price"] = ensure_cpu_for_pandas(z_dlog_gpu)
                df_agg.loc[df_agg["dlog_std"] == 0, "z_dlog_price"] = 0.0
            else:
                # CPU 版本
                df_agg["z_dlog_price"] = (df_agg["dlog_price"] - df_agg["dlog_mean"]) / df_agg["dlog_std"]
                df_agg.loc[df_agg["dlog_std"] == 0, "z_dlog_price"] = 0.0

        except Exception as e:
            logger.warning(f"[Job {job_id}] GPU normalization failed: {e}, falling back to CPU")
            df_agg["z_dlog_price"] = (df_agg["dlog_price"] - df_agg["dlog_mean"]) / df_agg["dlog_std"]
            df_agg.loc[df_agg["dlog_std"] == 0, "z_dlog_price"] = 0.0

        # 8) 写入 AutomlPreprocessedSeries（幂等：先删旧、再写新）
        with transaction.atomic():
            AutomlPreprocessedSeries.objects.filter(job=job).delete()

            objs = [
                AutomlPreprocessedSeries(
                    job=job,
                    shop_id=row["shop_id"],
                    iphone_id=row["iphone_id"],
                    bucket_ts=row["bucket_ts"],
                    raw_price=row["price"],
                    log_price=row["log_price"],
                    dlog_price=row["dlog_price"] if pd.notna(row["dlog_price"]) else None,
                    z_dlog_price=row["z_dlog_price"] if pd.notna(row["z_dlog_price"]) else None,
                    price_source=row["price_source"],
                )
                for _, row in df_agg.iterrows()
            ]

            created_count = 0
            if objs:
                AutomlPreprocessedSeries.objects.bulk_create(objs, batch_size=1000)
                created_count = len(objs)

            job.preprocessing_status = AutomlCausalJob.StageStatus.SUCCESS
            job.preprocessing_finished_at = timezone.now()
            job.save(update_fields=["preprocessing_status", "preprocessing_finished_at"])

            logger.info(f"[Job {job_id}] Preprocessing complete, created {created_count} series")

        # 9) 触发 VAR 阶段任务
        run_var_for_job.apply_async(args=[job.id], queue="automl_cause_effect")

        return {
            "status": "success",
            "job_id": job_id,
            "series_count": created_count,
        }

    except Exception as exc:
        logger.error(f"[Job {job_id}] Preprocessing failed: {exc}", exc_info=True)
        job.preprocessing_status = AutomlCausalJob.StageStatus.FAILED
        job.last_error = str(exc)[:2000]
        job.retry_count += 1
        job.preprocessing_finished_at = timezone.now()
        job.save(update_fields=[
            "preprocessing_status", "last_error",
            "retry_count", "preprocessing_finished_at"
        ])
        raise


# ============================================================================
# 阶段 2: VAR 模型 (Cause-and-Effect-Testing)
# ============================================================================

@shared_task(
    bind=True,
    name="automl.cause_and_effect_testing",
    queue="automl_cause_effect",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def run_var_for_job(self, job_id: int):
    """
    Cause-and-Effect-Testing Task (VAR Model)
    从预处理序列 → 拟合 VAR 模型
    """
    from AppleStockChecker.models import (
        AutomlCausalJob,
        AutomlPreprocessedSeries,
        AutomlVarModel,
    )

    try:
        from statsmodels.tsa.api import VAR
    except ImportError:
        logger.error("statsmodels not installed, cannot run VAR model")
        raise ImportError("Please install statsmodels: pip install statsmodels")

    logger.info(f"[Job {job_id}] Starting VAR modeling...")

    # 记录 GPU 状态
    if GPU_AVAILABLE:
        mem_info = get_gpu_memory_info()
        logger.info(
            f"[Job {job_id}] GPU Memory: {mem_info['free_gb']:.2f} GB free, "
            f"Utilization: {mem_info['utilization']*100:.1f}%"
        )

    job = AutomlCausalJob.objects.select_for_update().get(pk=job_id)

    # 只有在预处理成功后才跑 VAR
    if job.preprocessing_status != AutomlCausalJob.StageStatus.SUCCESS:
        logger.warning(f"[Job {job_id}] Preprocessing not complete, skipping VAR")
        return {"status": "skipped", "reason": "preprocessing_incomplete"}

    if job.cause_effect_status == AutomlCausalJob.StageStatus.SUCCESS:
        logger.info(f"[Job {job_id}] VAR already complete, skipping")
        return {"status": "already_done", "job_id": job_id}

    job.cause_effect_status = AutomlCausalJob.StageStatus.RUNNING
    job.cause_effect_started_at = timezone.now()
    job.last_error = None
    job.save(update_fields=["cause_effect_status", "cause_effect_started_at", "last_error"])

    try:
        # 1) 读取预处理序列
        qs = AutomlPreprocessedSeries.objects.filter(job=job)
        if not qs.exists():
            logger.warning(f"[Job {job_id}] No preprocessed series found, skipping")
            job.cause_effect_status = AutomlCausalJob.StageStatus.SKIPPED
            job.cause_effect_finished_at = timezone.now()
            job.save(update_fields=["cause_effect_status", "cause_effect_finished_at"])
            return {"status": "skipped", "reason": "no_preprocessed_data"}

        df = pd.DataFrame.from_records(
            qs.values("shop_id", "bucket_ts", "z_dlog_price")
        )

        # 2) 转换为 panel (时间 × 店铺)
        panel = df.pivot_table(
            index="bucket_ts",
            columns="shop_id",
            values="z_dlog_price",
        ).sort_index()

        # 简单处理缺失
        panel = panel.dropna(how="any")

        logger.info(f"[Job {job_id}] Panel shape: {panel.shape} (T={panel.shape[0]}, S={panel.shape[1]})")

        if panel.shape[0] < 20 or panel.shape[1] < 2:
            logger.warning(f"[Job {job_id}] Insufficient data for VAR, skipping")
            job.cause_effect_status = AutomlCausalJob.StageStatus.SKIPPED
            job.cause_effect_finished_at = timezone.now()
            job.save(update_fields=["cause_effect_status", "cause_effect_finished_at"])
            return {"status": "skipped", "reason": "insufficient_data"}

        # 3) 拟合 VAR
        model = VAR(panel)
        var_res = model.fit(maxlags=12, ic="aic")

        logger.info(f"[Job {job_id}] VAR fitted: lag_order={var_res.k_ar}, AIC={var_res.aic:.2f}")

        # 4) 保存 VAR 模型
        with transaction.atomic():
            AutomlVarModel.objects.filter(job=job).delete()
            AutomlVarModel.objects.create(
                job=job,
                shop_ids=list(map(int, panel.columns)),  # 确保是 int
                lag_order=var_res.k_ar,
                coefs={
                    "shape": list(var_res.coefs.shape),
                    "data": var_res.coefs.tolist(),
                },
                aic=float(var_res.aic),
                bic=float(var_res.bic),
                sample_size=panel.shape[0],
            )

            job.cause_effect_status = AutomlCausalJob.StageStatus.SUCCESS
            job.cause_effect_finished_at = timezone.now()
            job.save(update_fields=["cause_effect_status", "cause_effect_finished_at"])

        # 5) 触发 Impact 阶段任务
        run_impact_for_job.apply_async(args=[job.id], queue="automl_impact")

        return {
            "status": "success",
            "job_id": job_id,
            "lag_order": var_res.k_ar,
            "aic": float(var_res.aic),
        }

    except Exception as exc:
        logger.error(f"[Job {job_id}] VAR modeling failed: {exc}", exc_info=True)
        job.cause_effect_status = AutomlCausalJob.StageStatus.FAILED
        job.last_error = str(exc)[:2000]
        job.retry_count += 1
        job.cause_effect_finished_at = timezone.now()
        job.save(update_fields=[
            "cause_effect_status", "last_error",
            "retry_count", "cause_effect_finished_at"
        ])
        raise


# ============================================================================
# 阶段 3: 影响量化 (Quantification-of-Impact)
# ============================================================================

@shared_task(
    bind=True,
    name="automl.quantification_of_impact",
    queue="automl_impact",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def run_impact_for_job(self, job_id: int):
    """
    Quantification-of-Impact Task (Granger Causality)
    从 VAR 模型 + 预处理序列 → Granger 因果检验 → 因果边
    """
    from AppleStockChecker.models import (
        AutomlCausalJob,
        AutomlPreprocessedSeries,
        AutomlVarModel,
        AutomlGrangerResult,
        AutomlCausalEdge,
    )

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        logger.error("statsmodels not installed, cannot run Granger test")
        raise ImportError("Please install statsmodels: pip install statsmodels")

    logger.info(f"[Job {job_id}] Starting Impact quantification (Granger)...")

    # 记录 GPU 状态
    if GPU_AVAILABLE:
        mem_info = get_gpu_memory_info()
        logger.info(
            f"[Job {job_id}] GPU Memory: {mem_info['free_gb']:.2f} GB free, "
            f"Utilization: {mem_info['utilization']*100:.1f}%"
        )

    job = AutomlCausalJob.objects.select_for_update().get(pk=job_id)

    if job.cause_effect_status != AutomlCausalJob.StageStatus.SUCCESS:
        logger.warning(f"[Job {job_id}] VAR not complete, skipping Impact")
        return {"status": "skipped", "reason": "var_incomplete"}

    if job.impact_status == AutomlCausalJob.StageStatus.SUCCESS:
        logger.info(f"[Job {job_id}] Impact already complete, skipping")
        return {"status": "already_done", "job_id": job_id}

    job.impact_status = AutomlCausalJob.StageStatus.RUNNING
    job.impact_started_at = timezone.now()
    job.last_error = None
    job.save(update_fields=["impact_status", "impact_started_at", "last_error"])

    try:
        var_model = job.var_model
        shop_ids = var_model.shop_ids

        # 1) 读取 panel
        qs = AutomlPreprocessedSeries.objects.filter(job=job)
        df = pd.DataFrame.from_records(
            qs.values("shop_id", "bucket_ts", "z_dlog_price")
        )
        panel = df.pivot_table(
            index="bucket_ts",
            columns="shop_id",
            values="z_dlog_price",
        ).sort_index()
        panel = panel[shop_ids].dropna(how="any")

        maxlag = min(5, var_model.lag_order)

        logger.info(f"[Job {job_id}] Running Granger tests for {len(shop_ids)} shops (maxlag={maxlag})")

        # 预先计算 VAR 系数（GPU 加速）
        coefs_array = np.array(var_model.coefs["data"])
        if GPU_AVAILABLE:
            try:
                logger.info(f"[Job {job_id}] Loading VAR coefficients to GPU for weight calculation")
                xp = get_array_module()
                coefs_gpu = to_gpu(coefs_array)
            except Exception as e:
                logger.warning(f"[Job {job_id}] Failed to load coefficients to GPU: {e}, using CPU")
                coefs_gpu = coefs_array
                xp = np
        else:
            coefs_gpu = coefs_array
            xp = np

        granger_rows = []
        edges_rows = []

        # 2) 对所有店铺对进行 Granger 检验
        for cause in shop_ids:
            for effect in shop_ids:
                if cause == effect:
                    continue

                data = panel[[effect, cause]].dropna()
                if len(data) < maxlag + 5:
                    continue

                try:
                    # Granger 检验（statsmodels，保持在 CPU）
                    res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                    pvalues_by_lag = {}
                    min_p = 1.0
                    best_lag = None

                    for lag, out in res.items():
                        stat, pvalue, _, _ = out[0]["ssr_ftest"]
                        pvalues_by_lag[str(lag)] = float(pvalue)
                        if pvalue < min_p:
                            min_p = pvalue
                            best_lag = lag

                    is_sig = min_p < 0.05

                    granger_rows.append(
                        AutomlGrangerResult(
                            job=job,
                            cause_shop_id=cause,
                            effect_shop_id=effect,
                            maxlag=maxlag,
                            pvalues_by_lag=pvalues_by_lag,
                            min_pvalue=min_p,
                            best_lag=best_lag,
                            is_significant=is_sig,
                        )
                    )

                    if is_sig:
                        # 计算权重：VAR 系数的绝对值总和（GPU 加速）
                        idx_cause = shop_ids.index(cause)
                        idx_effect = shop_ids.index(effect)

                        try:
                            if GPU_AVAILABLE and hasattr(coefs_gpu, 'get'):
                                # GPU 版本
                                weight_gpu = xp.abs(coefs_gpu[:, idx_effect, idx_cause]).sum()
                                weight = float(ensure_cpu_for_pandas(weight_gpu))
                            else:
                                # CPU 版本
                                weight = float(xp.abs(coefs_gpu[:, idx_effect, idx_cause]).sum())
                        except Exception as e:
                            logger.warning(f"[Job {job_id}] GPU weight calculation failed: {e}, using CPU")
                            weight = float(np.abs(coefs_array[:, idx_effect, idx_cause]).sum())

                        confidence = max(0.0, min(1.0, 1.0 - min_p))

                        edges_rows.append(
                            AutomlCausalEdge(
                                job=job,
                                cause_shop_id=cause,
                                effect_shop_id=effect,
                                main_lag=best_lag or 1,
                                weight=weight,
                                min_pvalue=min_p,
                                confidence=confidence,
                                enabled=True,
                            )
                        )
                except Exception as e:
                    logger.warning(f"[Job {job_id}] Granger test failed for {cause}->{effect}: {e}")
                    continue

        # 3) 批量写入结果
        with transaction.atomic():
            AutomlGrangerResult.objects.filter(job=job).delete()
            AutomlCausalEdge.objects.filter(job=job).delete()

            if granger_rows:
                AutomlGrangerResult.objects.bulk_create(granger_rows, batch_size=500)
            if edges_rows:
                AutomlCausalEdge.objects.bulk_create(edges_rows, batch_size=500)

            job.impact_status = AutomlCausalJob.StageStatus.SUCCESS
            job.impact_finished_at = timezone.now()
            job.save(update_fields=["impact_status", "impact_finished_at"])

            logger.info(
                f"[Job {job_id}] Impact complete: "
                f"{len(granger_rows)} tests, {len(edges_rows)} significant edges"
            )

        return {
            "status": "success",
            "job_id": job_id,
            "granger_tests": len(granger_rows),
            "significant_edges": len(edges_rows),
        }

    except Exception as exc:
        logger.error(f"[Job {job_id}] Impact quantification failed: {exc}", exc_info=True)
        job.impact_status = AutomlCausalJob.StageStatus.FAILED
        job.last_error = str(exc)[:2000]
        job.retry_count += 1
        job.impact_finished_at = timezone.now()
        job.save(update_fields=[
            "impact_status", "last_error",
            "retry_count", "impact_finished_at"
        ])
        raise


# ============================================================================
# 向后兼容的简单任务（保持原API不变）
# ============================================================================

@shared_task(name="automl.preprocessing_rapid_simple", queue="automl_preprocessing")
def preprocessing_rapid():
    """简单版本：打印任务名称"""
    logger.info("Preprocessing-Rapid (simple) task started")
    print("Preprocessing-Rapid")
    logger.info("Preprocessing-Rapid (simple) task completed")
    return {"status": "success", "task": "Preprocessing-Rapid"}


@shared_task(name="automl.cause_and_effect_testing_simple", queue="automl_cause_effect")
def cause_and_effect_testing():
    """简单版本：打印任务名称"""
    logger.info("Cause-and-Effect-Testing (simple) task started")
    print("Cause-and-Effect-Testing")
    logger.info("Cause-and-Effect-Testing (simple) task completed")
    return {"status": "success", "task": "Cause-and-Effect-Testing"}


@shared_task(name="automl.quantification_of_impact_simple", queue="automl_impact")
def quantification_of_impact():
    """简单版本：打印任务名称"""
    logger.info("Quantification-of-Impact (simple) task started")
    print("Quantification-of-Impact")
    logger.info("Quantification-of-Impact (simple) task completed")
    return {"status": "success", "task": "Quantification-of-Impact"}
