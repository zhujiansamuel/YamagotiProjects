from __future__ import annotations
import logging
from datetime import timedelta
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from AppleStockChecker.collectors import collect_items_for_psta
from AppleStockChecker.utils.timebox import nearest_past_minute_iso

from AppleStockChecker.ws_notify import (
    notify_progress_all,
    notify_batch_items_all,
    notify_batch_done_all,
)
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from celery import shared_task, chord
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import transaction, IntegrityError
from decimal import Decimal, ROUND_HALF_UP




logger = logging.getLogger(__name__)


# ---------- 工具：ISO ↔ aware datetime ----------
def _to_aware(dt_or_iso):
    """接受 datetime 或 ISO 字符串，返回 aware datetime（项目时区）。"""
    if isinstance(dt_or_iso, str):
        dt = parse_datetime(dt_or_iso)
        if dt is None:
            raise ValueError(f"Invalid datetime: {dt_or_iso!r}")
    else:
        dt = dt_or_iso
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


def _tz_offset_str(dt: timezone.datetime) -> str:
    """把 datetime 的 tzinfo 转成 '+09:00' 这种偏移字符串。"""
    offset = dt.utcoffset() or timedelta(0)
    sign = "+" if offset.total_seconds() >= 0 else "-"
    total = abs(int(offset.total_seconds()))
    hh, mm = divmod(total // 60, 60)
    return f"{sign}{hh:02d}:{mm:02d}"


# === 可调参数（根据你们前后端链路容量调节） ===
MAX_BUCKET_ERROR_SAMPLES = 50       # 单桶保留的 error 明细条数上限
MAX_BUCKET_CHART_POINTS = 3000      # 单桶打包给回调聚合用的 chart point 上限
MAX_PUSH_POINTS = 20000             # 本次广播给前端的 point 总上限（超过则裁剪到最近 N 条）
#-----------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------------
#---------------------------------------------------------------
#-----------------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------
# -----------------------------------------------
# 子任务：处理“分钟桶”并返回桶级摘要 + 图表增量
# -----------------------------------------------
@shared_task(name="AppleStockChecker.tasks.psta_process_minute_bucket")
def psta_process_minute_bucket(
        *,
        ts_iso: str,
        rows: List[Dict[str, Any]],
        job_id: str,
        do_agg: bool = True,
        agg_start_iso: Optional[str] = None,
        agg_minutes: int = 1
) -> Dict[str, Any]:
    """


    """
    # -----------------------------------------------------
    # --------------------------------------------------------
    # -----------------------------------------------------------
    # -------------------分钟对齐数据写入-----------------------------
    # -----------------------------------------------------------
    # --------------------------------------------------------
    # -----------------------------------------------------
    from django.utils import timezone
    from AppleStockChecker.models import (
        PurchasingShopTimeAnalysis, SecondHandShop, Iphone,
        OverallBar, Cohort, CohortMember, CohortBar,
        FeatureSnapshot, ShopWeightProfile, ShopWeightItem,
    )

    # === 工具函数 ===
    def _to_aware(s: str):
        # 你已有的实现：ISO -> aware datetime
        from django.utils.dateparse import parse_datetime
        from django.utils.timezone import make_aware, is_naive
        dt = parse_datetime(s)
        if dt is None:
            raise ValueError(f"bad datetime iso: {s}")
        return make_aware(dt) if is_naive(dt) else dt

    def _tz_offset_str(dt):
        # 你已有的实现（简单回显 +09:00 等）
        return dt.strftime("%z")[:-2] + ":" + dt.strftime("%z")[-2:]

    def _d4(x):
        if x is None:
            return None
        return (Decimal(str(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _quantile(sorted_vals, p: float):
        """最近邻分位数（sorted_vals 必须升序）。"""
        if not sorted_vals:
            return None
        n = len(sorted_vals)
        if n == 1:
            return float(sorted_vals[0])
        k = int(round((n - 1) * p))
        k = 0 if k < 0 else (n - 1 if k > n - 1 else k)
        return float(sorted_vals[k])

    def _pop_std(vals):
        """总体标准差；N<=1 返回 0."""
        n = len(vals)
        if n <= 1:
            return 0.0
        mu = sum(vals) / n
        s2 = sum((v - mu) ** 2 for v in vals) / n
        return (s2 ** 0.5)

    def _pop_std(vals):
        """总体标准差；N<=1 返回 0."""
        n = len(vals)
        if n <= 1:
            return 0.0
        mu = sum(vals) / n
        s2 = sum((v - mu) ** 2 for v in vals) / n
        return (s2 ** 0.5)

    def _filter_outliers_by_mean_band(vals, lower_factor=0.5, upper_factor=1.5):
        """
        按“相对平均值”过滤异常值：
        - 先算原始均值 m；
        - 保留 [m*lower_factor, m*upper_factor] 区间内的值；
        - 如果全部被过滤掉，则回退到原始列表。
        返回 (filtered_vals, m, low, high)。
        """
        if not vals:
            return [], None, None, None
        m = sum(vals) / len(vals)
        if m <= 0:
            # 极端情况（不太会发生），直接不滤
            return list(vals), m, None, None
        low = m * lower_factor
        high = m * upper_factor
        filtered = [v for v in vals if low <= v <= high]
        if not filtered:
            # 全被判成异常，就用原始值，避免整组丢失
            return list(vals), m, low, high
        return filtered, m, low, high

    ok = 0
    failed = 0
    errors: List[Dict[str, Any]] = []
    err_counter = Counter()

    ts_dt = _to_aware(ts_iso)
    bucket_start = _to_aware(agg_start_iso) if (agg_minutes and agg_start_iso) else ts_dt
    bucket_end = bucket_start + timezone.timedelta(minutes=agg_minutes or 1)
    # 仅用于debug
    agg_ctx = {
        "do_agg": bool(do_agg),
        "bucket_start": bucket_start.isoformat(),
        "bucket_end": bucket_end.isoformat(),
        "agg_minutes": int(agg_minutes or 1),
    }
    ts_tz = _tz_offset_str(ts_dt)
    orig_tz = "+09:00"

    chart_points: List[Dict[str, Any]] = []

    for r in rows:
        try:
            # 轻量校验
            shop_id = r.get("shop_id")
            iphone_id = r.get("iphone_id")
            if not shop_id or not iphone_id:
                raise ValueError("missing shop_id/iphone_id")

            # 外键存在性（抛更友好的 DoesNotExist）
            SecondHandShop.objects.only("id").get(pk=shop_id)
            Iphone.objects.only("id").get(pk=iphone_id)

            rec_dt = _to_aware(r.get("recorded_at"))
            new_price = r.get("price_new") or r.get("New_Product_Price")
            if new_price is None:
                raise ValueError("missing New_Product_Price")

            align_diff = int((rec_dt - ts_dt).total_seconds())

            with transaction.atomic():
                inst = PurchasingShopTimeAnalysis.objects.select_for_update().filter(
                    shop_id=shop_id, iphone_id=iphone_id, Timestamp_Time=ts_dt
                ).first()

                if inst:
                    inst.Job_ID = job_id
                    inst.Original_Record_Time_Zone = orig_tz
                    inst.Timestamp_Time_Zone = ts_tz
                    inst.Record_Time = rec_dt
                    inst.Alignment_Time_Difference = align_diff
                    inst.New_Product_Price = int(new_price)
                    inst.Update_Count = (inst.Update_Count or 0) + 1
                    inst.save()
                else:
                    inst = PurchasingShopTimeAnalysis.objects.create(
                        Batch_ID=None,
                        Job_ID=job_id,
                        Original_Record_Time_Zone=orig_tz,
                        Timestamp_Time_Zone=ts_tz,
                        Record_Time=rec_dt,
                        Timestamp_Time=ts_dt,
                        Alignment_Time_Difference=align_diff,
                        Update_Count=0,
                        shop_id=shop_id,
                        iphone_id=iphone_id,
                        New_Product_Price=int(new_price),
                    )

            ok += 1

            # 收集图表增量（前端去重）
            if len(chart_points) < MAX_BUCKET_CHART_POINTS:
                chart_points.append({
                    "id": inst.pk,
                    "t": ts_iso,
                    "iphone_id": iphone_id,
                    "shop_id": shop_id,
                    "price": int(new_price),
                    "recorded_at": rec_dt.isoformat(),
                })

        except (ObjectDoesNotExist, ValidationError, IntegrityError, TypeError, ValueError) as e:
            failed += 1
            err_counter[e.__class__.__name__] += 1
            if len(errors) < MAX_BUCKET_ERROR_SAMPLES:
                errors.append({
                    "exc": e.__class__.__name__,
                    "msg": str(e),
                    "item": {
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                        "New_Product_Price": r.get("price_new") or r.get("New_Product_Price"),
                    }
                })
        except Exception as e:
            failed += 1
            err_counter[e.__class__.__name__] += 1
            if len(errors) < MAX_BUCKET_ERROR_SAMPLES:
                errors.append({
                    "exc": e.__class__.__name__,
                    "msg": str(e),
                    "item": {
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                    }
                })

    if failed:
        try:
            notify_progress_all(data={
                "type": "bucket_errors",
                "ts_iso": ts_iso,
                "total": ok + failed,
                "ok": ok,
                "failed": failed,
                "error_hist": dict(err_counter),
                "sample": errors[:5],
            })
        except Exception:
            pass

    # -----------------------------------------------------
    # --------------------------------------------------------
    # -----------------------------------------------------------
    # ---统计数据写入（OverallBar, Cohort, CohortMember, CohortBar）---
    # -----------------------------------------------------------
    # --------------------------------------------------------
    # -----------------------------------------------------
    if do_agg:

        from statistics import median
        from collections import defaultdict

        WATERMARK_MINUTES = 5
        now = timezone.now()
        is_final_bar = ts_dt <= (now - timezone.timedelta(minutes=WATERMARK_MINUTES))

        # --- 聚合窗口（1分钟：用 ts_dt；>1分钟：用 [bucket_start, bucket_end) 窗口） ---
        bucket_start = _to_aware(agg_start_iso) if (agg_minutes and agg_start_iso) else ts_dt
        bucket_end = bucket_start + timezone.timedelta(minutes=agg_minutes or 1)
        use_window = (agg_minutes or 1) > 1
        agg_ctx = {
            "do_agg": True,
            "agg_minutes": int(agg_minutes or 1),
            "bucket_start": bucket_start.isoformat(),
            "bucket_end": bucket_end.isoformat(),
        }

        # ================= 1) OverallBar（全部店 × 各 iPhone） =================
        # 自动探测 OverallBar 是否含 iphone 外键；若没有，跳过以免 unique(bucket) 冲突
        ob_has_iphone = any(getattr(f, "name", "") == "iphone" for f in OverallBar._meta.get_fields())
        overallbar_debug = {"agg": agg_ctx, "iphones": [], "skipped": False}

        try:
            if not ob_has_iphone:
                overallbar_debug["skipped"] = True
            else:
                bucket_iphone_ids = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})
                for ipid in bucket_iphone_ids:
                    if use_window:
                        # 窗口内：每店最后一条（PG 的 distinct on 语义：order_by 先分组键，再时间倒序）
                        qs_latest = (PurchasingShopTimeAnalysis.objects
                                     .filter(iphone_id=ipid,
                                             Timestamp_Time__gte=bucket_start,
                                             Timestamp_Time__lt=bucket_end)
                                     .order_by("shop_id", "-Timestamp_Time")
                                     .distinct("shop_id"))
                        prices = [float(p) for p in qs_latest.values_list("New_Product_Price", flat=True) if
                                  p is not None]
                        shop_cnt = qs_latest.values("shop_id").count()
                        ob_bucket = bucket_start
                    else:
                        if use_window:
                            qs_latest = (
                                PurchasingShopTimeAnalysis.objects
                                .filter(iphone_id=ipid, Timestamp_Time__gte=bucket_start, Timestamp_Time__lt=bucket_end)
                                .order_by("shop_id", "-Timestamp_Time")
                                .distinct("shop_id")
                                .values("shop_id", "New_Product_Price", "Timestamp_Time")
                            )
                            prices = [float(r["New_Product_Price"]) for r in qs_latest if
                                      r["New_Product_Price"] is not None]
                            shop_cnt = qs_latest.values("shop_id").count()
                            ob_bucket = bucket_start
                        else:
                            qs_latest = (
                                PurchasingShopTimeAnalysis.objects
                                .filter(iphone_id=ipid, Timestamp_Time=ts_dt)
                                .values("shop_id", "New_Product_Price")
                            )
                            prices = [float(r["New_Product_Price"]) for r in qs_latest if
                                      r["New_Product_Price"] is not None]
                            shop_cnt = qs_latest.values("shop_id").distinct().count()
                            ob_bucket = ts_dt

                    if not prices:
                        continue

                    # 先按“平均值 ±50%”过滤异常值，再计算统计量
                    vals_raw = [float(p) for p in prices]
                    vals_filtered, m0, low_band, high_band = _filter_outliers_by_mean_band(vals_raw)

                    if not vals_filtered:
                        # 极端情况：全被过滤，直接跳过这一组（也可以选择回退 vals_raw）
                        continue

                    vals = sorted(vals_filtered)
                    m_mean = sum(vals) / len(vals)
                    m_median = float(median(vals))
                    m_std = _pop_std(vals)
                    p10 = _quantile(vals, 0.10)
                    p90 = _quantile(vals, 0.90)
                    dispersion = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0

                    OverallBar.objects.update_or_create(
                        bucket=ob_bucket, iphone_id=ipid,
                        defaults=dict(
                            mean=_d4(m_mean),
                            median=_d4(m_median),
                            std=_d4(m_std) if m_std is not None else None,
                            shop_count=shop_cnt,
                            dispersion=_d4(dispersion),
                            is_final=is_final_bar,
                        )
                    )

                    if len(overallbar_debug["iphones"]) < 5:
                        overallbar_debug["iphones"].append({
                            "iphone_id": ipid,
                            "shop_count": shop_cnt,
                            "mean": round(m_mean, 4),
                            "median": round(m_median, 4),
                            "std": (round(m_std, 4) if m_std is not None else None),
                            "dispersion": round(dispersion, 4),
                            "bucket": ob_bucket.isoformat(),
                            "is_final": is_final_bar,
                        })

                try:
                    notify_progress_all(data={
                        "type": "overallbar_update",
                        "bucket": ts_iso,
                        "detail": overallbar_debug,
                    })
                except Exception:
                    pass
        except Exception as e:
            try:
                notify_progress_all(data={
                    "type": "overallbar_error",
                    "bucket": ts_iso,
                    "error": repr(e),
                    "agg": agg_ctx,
                })
            except Exception:
                pass

        # ================ 2) CohortBar（全部店 × 组合 iPhone） ================
        cohort_debug = {"agg": agg_ctx, "bucket": ts_iso, "cohorts": []}
        try:
            if ob_has_iphone:
                cohorts = list(Cohort.objects.all())
                for coh in cohorts:
                    members = list(CohortMember.objects.filter(cohort=coh).values("iphone_id", "weight"))
                    if not members:
                        continue
                    member_ids = [m["iphone_id"] for m in members]
                    weight_map = {m["iphone_id"]: float(m.get("weight") or 1.0) for m in members}

                    # 读取上一步写好的 OverallBar（同一个 ob_bucket）
                    ob_bucket = bucket_start if use_window else ts_dt
                    ob_rows = list(
                        OverallBar.objects.filter(bucket=ob_bucket, iphone_id__in=member_ids)
                        .values("iphone_id", "mean", "shop_count")
                    )
                    vals = [float(r["mean"]) for r in ob_rows if r.get("mean") is not None]
                    if not vals:
                        continue

                    denom = 0.0
                    num = 0.0
                    for r in ob_rows:
                        v = r.get("mean")
                        if v is None:
                            continue
                        w = weight_map.get(r["iphone_id"], 1.0) * float(r.get("shop_count") or 0.0)
                        denom += w
                        num += w * float(v)
                    c_mean = (num / denom) if denom > 0 else (sum(vals) / len(vals))

                    vals_sorted = sorted(vals)
                    c_median = float(median(vals_sorted))
                    c_std = _pop_std(vals_sorted)
                    p10 = _quantile(vals_sorted, 0.10)
                    p90 = _quantile(vals_sorted, 0.90)
                    c_disp = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0
                    n_models = len(vals_sorted)
                    shop_count_agg = sum(int(r.get("shop_count") or 0) for r in ob_rows)

                    CohortBar.objects.update_or_create(
                        bucket=ob_bucket, cohort=coh,
                        defaults=dict(
                            mean=_d4(c_mean),
                            median=_d4(c_median),
                            std=_d4(c_std) if c_std is not None else None,
                            n_models=n_models,
                            shop_count_agg=shop_count_agg,
                            dispersion=_d4(c_disp),
                            is_final=is_final_bar,
                        )
                    )

                    if len(cohort_debug["cohorts"]) < 5:
                        cohort_debug["cohorts"].append({
                            "cohort": {"id": coh.id, "slug": getattr(coh, "slug", str(coh))},
                            "n_models": n_models,
                            "shop_count_agg": shop_count_agg,
                            "mean": round(c_mean, 4),
                            "median": round(c_median, 4),
                            "std": (round(c_std, 4) if c_std is not None else None),
                            "dispersion": round(c_disp, 4),
                            "bucket": ob_bucket.isoformat(),
                            "is_final": is_final_bar,
                        })

                try:
                    notify_progress_all(data={
                        "type": "cohortbar_update",
                        "bucket": ts_iso,
                        "detail": cohort_debug,
                    })
                except Exception:
                    pass
            else:
                try:
                    notify_progress_all(data={
                        "type": "cohortbar_skipped",
                        "bucket": ts_iso,
                        "reason": "OverallBar lacks iphone dimension; skip CohortBar to avoid collisions.",
                        "agg": agg_ctx,
                    })
                except Exception:
                    pass
        except Exception as e:
            try:
                notify_progress_all(data={
                    "type": "cohortbar_error",
                    "bucket": ts_iso,
                    "error": repr(e),
                    "agg": agg_ctx,
                })
            except Exception:
                pass

        # =========== 3) 四类组合统计：写入 FeatureSnapshot（窗口去重 + 时效权） ===========
        try:
            from django.conf import settings
            # —— 预取本桶出现过的 shop/iphone —— #
            shops_seen = sorted({int(r.get("shop_id")) for r in rows if r.get("shop_id")})
            iphones_seen = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})

            if use_window:
                base_qs = (
                    PurchasingShopTimeAnalysis.objects
                    .filter(Timestamp_Time__gte=bucket_start, Timestamp_Time__lt=bucket_end,
                            shop_id__in=shops_seen, iphone_id__in=iphones_seen)
                    .order_by("shop_id", "iphone_id", "-Timestamp_Time")
                    .distinct("shop_id", "iphone_id")
                    .values("shop_id", "iphone_id", "New_Product_Price", "Timestamp_Time")
                )
                feature_bucket = bucket_start
            else:
                base_qs = (
                    PurchasingShopTimeAnalysis.objects
                    .filter(Timestamp_Time=ts_dt,
                            shop_id__in=shops_seen, iphone_id__in=iphones_seen)
                    .values("shop_id", "iphone_id", "New_Product_Price", "Timestamp_Time")
                )
                feature_bucket = ts_dt

            # (shop, iphone) -> (last_price, last_ts)
            data_by_si: Dict[tuple, tuple] = {}
            for rec in base_qs:
                p = rec.get("New_Product_Price");
                t = rec.get("Timestamp_Time")
                if p is None:
                    continue
                s = int(rec["shop_id"]);
                i = int(rec["iphone_id"])
                data_by_si[(s, i)] = (float(p), t)

            # —— 统计与落库工具 —— #
            def _stats(values):
                """"返回 (mean, median, std, dispersion, count)，自动按平均值过滤异常值。"""
                if not values:
                    return None
                vals_raw = [float(v) for v in values]
                vals_filtered, m0, low_band, high_band = _filter_outliers_by_mean_band(vals_raw)
                if not vals_filtered:
                    return None
                vals = sorted(vals_filtered)
                n = len(vals)
                mean_v = sum(vals) / n
                med_v = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
                std_v = _pop_std(vals)
                p10 = _quantile(vals, 0.10)
                p90 = _quantile(vals, 0.90)
                disp_v = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0
                return mean_v, med_v, std_v, disp_v, n

            def upsert_feature(scope: str, name: str, value: float, *, is_final: bool, version: str = 'v1'):
                FeatureSnapshot.objects.update_or_create(
                    bucket=feature_bucket, scope=scope, name=name, version=version,
                    defaults=dict(value=float(_d4(value)), is_final=is_final)
                )

            # —— 时效权重（AGE_CAP + 半衰期/线性） —— #
            AGE_CAP_MIN = float(getattr(settings, "PSTA_AGE_CAP_MIN", 12.0))  # 超过则不计
            RECENCY_HALF_LIFE_MIN = float(getattr(settings, "PSTA_RECENCY_HALF_LIFE_MIN", 6.0))  # 指数半衰期
            RECENCY_DECAY = str(getattr(settings, "PSTA_RECENCY_DECAY", "exp")).lower()  # 'exp'|'linear'

            import math
            def recency_weight(last_ts, ref_end):
                if last_ts is None:
                    return 0.0, None
                age_min = (ref_end - last_ts).total_seconds() / 60.0
                if age_min < 0:
                    age_min = 0.0
                if age_min > AGE_CAP_MIN:
                    return 0.0, age_min
                if RECENCY_DECAY == "linear":
                    w = max(0.0, 1.0 - (age_min / max(AGE_CAP_MIN, 1e-6)))
                else:
                    lam = math.log(2.0) / max(RECENCY_HALF_LIFE_MIN, 1e-6)
                    w = math.exp(-lam * age_min)
                return float(w), age_min

            combo_debug = {
                "agg": agg_ctx,
                "bucket": ts_iso,
                "case1_shop_iphone": 0,
                "case2_shopcohort_iphone": 0,
                "case3_shop_cohortiphone": 0,
                "case4_shopcohort_cohortiphone": 0,
                "skipped": [],
                "samples": [],
            }

            # === CASE 1: 各店 × 各 iPhone（单值；用于原始曲线） ===
            for (sid, iid), (v, t) in data_by_si.items():
                s = _stats([v])
                if not s:
                    continue
                m, med, st, disp, n = s
                scope = f"shop:{sid}|iphone:{iid}"
                upsert_feature(scope, "mean", m, is_final=is_final_bar)
                upsert_feature(scope, "median", med, is_final=is_final_bar)
                upsert_feature(scope, "std", st, is_final=is_final_bar)
                upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                combo_debug["case1_shop_iphone"] += 1
                if len(combo_debug["samples"]) < 5:
                    combo_debug["samples"].append({"case": 1, "scope": scope, "mean": round(m, 4)})

            # 预取店铺组合（ShopWeightProfile）
            profiles = list(ShopWeightProfile.objects.all())
            prof_items = {
                prof.id: {it['shop_id']: float(it.get('weight') or 1.0)
                          for it in ShopWeightItem.objects.filter(profile=prof).values('shop_id', 'weight')}
                for prof in profiles
            }
            has_shop_profile = any(bool(prof_items.get(p.id)) for p in profiles)

            # 预取机型组合（Cohort）
            cohorts = list(Cohort.objects.all())
            cmembers = {
                coh.id: {m['iphone_id']: float(m.get('weight') or 1.0)
                         for m in CohortMember.objects.filter(cohort=coh).values('iphone_id', 'weight')}
                for coh in cohorts
            }

            # === CASE 2: 组合店 × 各 iPhone（店权 × 时效权） ===
            if has_shop_profile:
                for prof in profiles:
                    sw = prof_items.get(prof.id, {})
                    if not sw:
                        continue
                    shops_in = set(sw.keys()) & set(shops_seen)
                    if not shops_in:
                        continue
                    for iid in iphones_seen:
                        vals, ages = [], []
                        wnum = wden = w2 = 0.0
                        for sid in shops_in:
                            pair = data_by_si.get((int(sid), int(iid)))
                            if not pair:
                                continue
                            v, t = pair
                            w_rec, age = recency_weight(t, bucket_end)
                            if w_rec <= 0.0:
                                continue
                            w_shop = float(sw.get(sid, 1.0))
                            w = w_shop * w_rec
                            vals.append(v)
                            if age is not None:
                                ages.append(age)
                            wnum += w * v;
                            wden += w;
                            w2 += w * w
                        if not vals:
                            continue
                        m_unw, med, st, disp, n = _stats(vals)
                        mean_w = (wnum / wden) if wden > 0 else m_unw
                        scope = f"shopcohort:{prof.slug}|iphone:{iid}"
                        upsert_feature(scope, "mean", mean_w, is_final=is_final_bar)
                        upsert_feature(scope, "median", med, is_final=is_final_bar)
                        upsert_feature(scope, "std", st, is_final=is_final_bar)
                        upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                        upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                        combo_debug["case2_shopcohort_iphone"] += 1
                        if len(combo_debug["samples"]) < 5:
                            combo_debug["samples"].append({
                                "case": 2, "scope": scope, "n": n,
                                "mean_w": round(mean_w, 4),
                                "age_p50": (round(_quantile(sorted(ages), 0.5), 2) if ages else None)
                            })
            else:
                combo_debug["skipped"].append("case2: no ShopWeightProfile defined")

            # === CASE 3: 各店 × 组合 iPhone（机型权 × 时效权） ===
            for sid in shops_seen:
                for coh in cohorts:
                    iw = cmembers.get(coh.id, {})
                    if not iw:
                        continue
                    vals, ages = [], []
                    wnum = wden = w2 = 0.0
                    for iid, w_phone in iw.items():
                        pair = data_by_si.get((int(sid), int(iid)))
                        if not pair:
                            continue
                        v, t = pair
                        w_rec, age = recency_weight(t, bucket_end)
                        if w_rec <= 0.0:
                            continue
                        w = float(w_phone) * w_rec
                        vals.append(v)
                        if age is not None:
                            ages.append(age)
                        wnum += w * v;
                        wden += w;
                        w2 += w * w
                    if not vals:
                        continue
                    m_unw, med, st, disp, n = _stats(vals)
                    mean_w = (wnum / wden) if wden > 0 else m_unw
                    scope = f"shop:{sid}|cohort:{coh.slug}"
                    upsert_feature(scope, "mean", mean_w, is_final=is_final_bar)
                    upsert_feature(scope, "median", med, is_final=is_final_bar)
                    upsert_feature(scope, "std", st, is_final=is_final_bar)
                    upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                    upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                    combo_debug["case3_shop_cohortiphone"] += 1
                    if len(combo_debug["samples"]) < 5:
                        combo_debug["samples"].append({
                            "case": 3, "scope": scope, "n": n,
                            "mean_w": round(mean_w, 4),
                            "age_p50": (round(_quantile(sorted(ages), 0.5), 2) if ages else None)
                        })

            # === CASE 4: 组合店 × 组合 iPhone（店权 × 机型权 × 时效权） ===
            if has_shop_profile:
                for prof in profiles:
                    sw = prof_items.get(prof.id, {})
                    if not sw:
                        continue
                    shops_in = set(sw.keys()) & set(shops_seen)
                    if not shops_in:
                        continue
                    for coh in cohorts:
                        iw = cmembers.get(coh.id, {})
                        if not iw:
                            continue
                        vals, ages = [], []
                        wnum = wden = w2 = 0.0
                        for sid, w_shop in sw.items():
                            if int(sid) not in shops_in:
                                continue
                            for iid, w_phone in iw.items():
                                pair = data_by_si.get((int(sid), int(iid)))
                                if not pair:
                                    continue
                                v, t = pair
                                w_rec, age = recency_weight(t, bucket_end)
                                if w_rec <= 0.0:
                                    continue
                                w = float(w_shop) * float(w_phone) * w_rec
                                vals.append(v)
                                if age is not None:
                                    ages.append(age)
                                wnum += w * v;
                                wden += w;
                                w2 += w * w
                        if not vals:
                            continue
                        m_unw, med, st, disp, n = _stats(vals)
                        mean_w = (wnum / wden) if wden > 0 else m_unw
                        scope = f"shopcohort:{prof.slug}|cohort:{coh.slug}"
                        upsert_feature(scope, "mean", mean_w, is_final=is_final_bar)
                        upsert_feature(scope, "median", med, is_final=is_final_bar)
                        upsert_feature(scope, "std", st, is_final=is_final_bar)
                        upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                        upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                        combo_debug["case4_shopcohort_cohortiphone"] += 1
                        if len(combo_debug["samples"]) < 5:
                            combo_debug["samples"].append({
                                "case": 4, "scope": scope, "n": n,
                                "mean_w": round(mean_w, 4),
                                "age_p50": (round(_quantile(sorted(ages), 0.5), 2) if ages else None)
                            })
            else:
                combo_debug["skipped"].append("case4: no ShopWeightProfile defined")

            try:
                notify_progress_all(data={
                    "type": "feature_snapshot_update",
                    "bucket": ts_iso,
                    "summary": {
                        "case1_shop_iphone": combo_debug["case1_shop_iphone"],
                        "case2_shopcohort_iphone": combo_debug["case2_shopcohort_iphone"],
                        "case3_shop_cohortiphone": combo_debug["case3_shop_cohortiphone"],
                        "case4_shopcohort_cohortiphone": combo_debug["case4_shopcohort_cohortiphone"],
                        "skipped": combo_debug["skipped"],
                        "agg": agg_ctx,
                    },
                    "samples": combo_debug["samples"],
                })
            except Exception:
                pass

        except Exception as e:
            try:
                notify_progress_all(data={
                    "type": "feature_snapshot_error",
                    "bucket": ts_iso,
                    "error": repr(e),
                    "agg": agg_ctx,
                })
            except Exception:
                pass

        # ================= 4) 时间序列派生指标：SMA / WMA / EMA（按 FeatureSpec） =================
        from django.apps import apps
        FeatureSpec = apps.get_model('AppleStockChecker', 'FeatureSpec')

        # 本次派生指标的“锚点时间”：与 FeatureSnapshot(mean, …) 一致
        anchor_bucket = bucket_start if use_window else ts_dt
        # Overall/Cohort 的基值写入了 ob_bucket；派生时也需要读取该时间的“当前基值”
        ob_bucket = bucket_start if use_window else ts_dt

        timefeat_debug = {"bucket": ts_iso, "computed": 0, "skipped": [], "samples": []}

        try:
            # 取出所有激活的时序类指标
            specs = list(
                FeatureSpec.objects
                .filter(active=True, family__in=["sma", "ema", "wma"])
                .values("slug", "family", "base_name", "params", "version")
            )

            # —— 收集“当前锚点”的基值 x_t：scope -> x_t —— #
            base_now: Dict[str, float] = {}

            # 4.a 四类组合（FeatureSnapshot.mean@anchor_bucket）
            for row in FeatureSnapshot.objects.filter(
                bucket=anchor_bucket, name="mean"
            ).values("scope", "value"):
                if row["value"] is not None:
                    base_now[row["scope"]] = float(row["value"])

            # 4.b OverallBar.mean -> overall:iphone:<id>（@ob_bucket）
            if ob_has_iphone:
                for row in OverallBar.objects.filter(bucket=ob_bucket).values("iphone_id", "mean"):
                    if row["mean"] is not None:
                        base_now[f"overall:iphone:{row['iphone_id']}"] = float(row["mean"])

            # 4.c CohortBar.mean -> cohort:<slug>（@ob_bucket）
            for row in (CohortBar.objects.filter(bucket=ob_bucket)
                        .select_related("cohort").values("cohort__slug", "mean")):
                if row["mean"] is not None and row["cohort__slug"]:
                    base_now[f"cohort:{row['cohort__slug']}"] = float(row["mean"])

            # —— 工具：回写派生值 —— #
            def upsert_feat(scope: str, name: str, version: str, value: float):
                FeatureSnapshot.objects.update_or_create(
                    bucket=anchor_bucket, scope=scope, name=name, version=version,
                    defaults=dict(value=float(_d4(value)), is_final=is_final_bar)
                )

            # —— 工具：取历史“基值”序列（不包含当前 x_t），按时间从新到旧取 limit 条 —— #
            def fetch_prev_base(scope: str, base_name: str, base_version: str, limit: int, anchor_dt):
                # overall:iphone:<id> -> 读 OverallBar.mean
                if scope.startswith("overall:iphone:"):
                    ipid = int(scope.rsplit(":", 1)[-1])
                    rows = (OverallBar.objects
                            .filter(iphone_id=ipid, bucket__lt=anchor_dt)
                            .order_by("-bucket").values_list("mean", flat=True)[:limit])
                    return [float(v) for v in rows if v is not None]
                # cohort:<slug> -> 读 CohortBar.mean
                if scope.startswith("cohort:"):
                    slug = scope.split(":", 1)[1]
                    rows = (CohortBar.objects
                            .filter(cohort__slug=slug, bucket__lt=anchor_dt)
                            .order_by("-bucket").values_list("mean", flat=True)[:limit])
                    return [float(v) for v in rows if v is not None]
                # 其它 scope -> 读 FeatureSnapshot(base_name)
                rows = (FeatureSnapshot.objects
                        .filter(scope=scope, name=base_name, version=base_version, bucket__lt=anchor_dt)
                        .order_by("-bucket").values_list("value", flat=True)[:limit])
                return [float(v) for v in rows if v is not None]

            # —— 数学器 —— #
            def _ema_from_series(series_old_to_new: list, alpha: float) -> float:
                ema = float(series_old_to_new[0])
                for v in series_old_to_new[1:]:
                    ema = alpha * float(v) + (1.0 - alpha) * ema
                return ema

            def _sma(series_old_to_new: list, W: int) -> Optional[float]:
                if not series_old_to_new:
                    return None
                s = series_old_to_new[-W:] if W < len(series_old_to_new) else series_old_to_new
                return sum(s) / float(len(s))

            def _wma_linear(series_old_to_new: list, W: int) -> Optional[float]:
                if not series_old_to_new:
                    return None
                s = series_old_to_new[-W:] if W < len(series_old_to_new) else series_old_to_new
                n = len(s)
                weights = list(range(1, n + 1))  # 越新权重越大
                denom = float(sum(weights))
                return sum(v * w for v, w in zip(s, weights)) / denom if denom > 0 else None

            def _alpha_from_params(p: dict) -> float:
                if p is None: p = {}
                if p.get("alpha") is not None:
                    a = float(p["alpha"])
                    return max(0.0, min(1.0, a))
                if p.get("window") is not None:
                    W = max(1, int(p["window"]))
                    return 2.0 / (W + 1.0)
                if p.get("half_life") is not None:
                    hl = float(p["half_life"])
                    return 1.0 - 0.5 ** (1.0 / max(hl, 1e-9))
                return 2.0 / (15.0 + 1.0)  # 默认

            # —— 逐 spec × scope 计算 —— #
            for sp in specs:
                family = (sp["family"] or "").lower()
                spec_slug = sp["slug"]
                base_name = sp.get("base_name") or "mean"
                params = sp.get("params") or {}
                base_version = params.get("base_version", sp.get("version") or "v1")

                # 统一窗口/最小样本数
                W = int(params.get("window", 15))
                min_count = int(params.get("min_count", params.get("min_periods", 1)))
                weights_mode = str(params.get("weights", "linear")).lower()

                for scope, x_t in base_now.items():
                    # 拉取历史基值（新->旧），再转为旧->新，并在末尾追加当前 x_t
                    prev_vals = fetch_prev_base(scope, base_name, base_version, W - 1, anchor_bucket)
                    series_old_to_new = list(reversed(prev_vals)) + [float(x_t)]

                    # 样本数校验
                    if len(series_old_to_new) < max(1, min_count):
                        timefeat_debug["skipped"].append(f"{family}:{spec_slug}@{scope}:insufficient({len(series_old_to_new)}<{min_count})")
                        continue

                    try:
                        if family == "ema":
                            alpha = _alpha_from_params(params)
                            val = _ema_from_series(series_old_to_new, alpha)
                            upsert_feat(scope, "ema", spec_slug, val)
                            timefeat_debug["computed"] += 1

                        elif family in ("wma", "wma_linear"):
                            if weights_mode == "linear":
                                val = _wma_linear(series_old_to_new, W)
                            else:
                                # 其它权重模式暂未实现 -> 后备为 SMA
                                val = _sma(series_old_to_new, W)
                            if val is None:
                                timefeat_debug["skipped"].append(f"wma:{spec_slug}@{scope}:no_series")
                                continue
                            upsert_feat(scope, "wma", spec_slug, val)
                            timefeat_debug["computed"] += 1

                        elif family == "sma":
                            val = _sma(series_old_to_new, W)
                            if val is None:
                                timefeat_debug["skipped"].append(f"sma:{spec_slug}@{scope}:no_series")
                                continue
                            upsert_feat(scope, "sma", spec_slug, val)
                            timefeat_debug["computed"] += 1

                        if len(timefeat_debug["samples"]) < 6:
                            timefeat_debug["samples"].append({
                                "scope": scope, "family": family, "spec": spec_slug,
                                "W": W, "x": round(float(x_t), 2), "y": round(float(val), 2)
                            })
                    except Exception as _e:
                        timefeat_debug["skipped"].append(f"{family}:{spec_slug}@{scope}:{repr(_e)}")

            try:
                notify_progress_all(data={
                    "type": "feature_time_series_update",
                    "bucket": ts_iso,
                    "summary": {k: v for k, v in timefeat_debug.items() if k != "samples"},
                    "samples": timefeat_debug["samples"],
                })
            except Exception:
                pass
        except Exception as e:
            try:
                notify_progress_all(data={
                    "type": "feature_time_series_error",
                    "bucket": ts_iso,
                    "error": repr(e),
                })
            except Exception:
                pass

        # ================= 5) 布林带（Bollinger Bands，支持 center_mode="sma" / "ema" / "sma60" 等） =================
        boll_debug = {"bucket": ts_iso, "computed": 0, "skipped": [], "samples": []}
        try:
            specs_boll = list(
                FeatureSpec.objects
                .filter(active=True, family__in=["boll", "bollinger"])
                .values("slug", "base_name", "params", "version")
            )

            if not specs_boll:
                boll_debug["skipped"].append("no_active_bollinger_spec")
            else:
                # 复用 base_now（见 Step 4）
                # 若上面 Step 4 被跳过，你可以在此重新构造 base_now（略）

                def _parse_center_mode(params: dict, default_W: int):
                    cm = str(params.get("center_mode", "sma")).lower()
                    # 支持 "sma" / "ema" / "sma60" / "ema30" 这种写法
                    import re
                    m = re.match(r"^(sma|ema)(\d+)?$", cm)
                    if m:
                        mode = m.group(1)
                        w = int(m.group(2)) if m.group(2) else default_W
                        return mode, w
                    # 其它写法退化到 sma
                    return "sma", default_W

                for sp in specs_boll:
                    spec_slug = sp["slug"]
                    base_name = sp.get("base_name") or "mean"
                    params = sp.get("params") or {}
                    base_version = params.get("base_version", sp.get("version") or "v1")

                    W = max(1, int(params.get("window", 20)))
                    k = float(params.get("k", 2.0))
                    min_periods = int(params.get("min_periods", W))
                    center_mode, center_W = _parse_center_mode(params, W)

                    for scope, x_t in base_now.items():
                        # 最近 W-1 条历史 + 当前 x_t
                        prev_vals = fetch_prev_base(scope, base_name, base_version, W - 1, anchor_bucket)
                        series_old_to_new = list(reversed(prev_vals)) + [float(x_t)]
                        if len(series_old_to_new) < max(1, min_periods):
                            boll_debug["skipped"].append(
                                f"{spec_slug}@{scope}:insufficient({len(series_old_to_new)}<{min_periods})")
                            continue

                        # 中轨
                        if center_mode == "ema":
                            alpha = 2.0 / (center_W + 1.0)
                            mid = _ema_from_series(series_old_to_new, alpha)
                        else:
                            mid = _sma(series_old_to_new, center_W)

                        # 标准差用总体（与你的 _pop_std 保持一致；series_old_to_new 已是旧->新）
                        std = _pop_std(series_old_to_new) or 0.0
                        up = mid + k * std
                        low = mid - k * std
                        width = up - low

                        FeatureSnapshot.objects.update_or_create(
                            bucket=anchor_bucket, scope=scope, name="boll_mid", version=spec_slug,
                            defaults=dict(value=float(_d4(mid)), is_final=is_final_bar)
                        )
                        FeatureSnapshot.objects.update_or_create(
                            bucket=anchor_bucket, scope=scope, name="boll_up", version=spec_slug,
                            defaults=dict(value=float(_d4(up)), is_final=is_final_bar)
                        )
                        FeatureSnapshot.objects.update_or_create(
                            bucket=anchor_bucket, scope=scope, name="boll_low", version=spec_slug,
                            defaults=dict(value=float(_d4(low)), is_final=is_final_bar)
                        )
                        FeatureSnapshot.objects.update_or_create(
                            bucket=anchor_bucket, scope=scope, name="boll_width", version=spec_slug,
                            defaults=dict(value=float(_d4(width)), is_final=is_final_bar)
                        )

                        boll_debug["computed"] += 1
                        if len(boll_debug["samples"]) < 6:
                            boll_debug["samples"].append({
                                "scope": scope, "spec": spec_slug, "W": W, "center": f"{center_mode}{center_W}",
                                "mid": round(mid, 2), "up": round(up, 2), "low": round(low, 2)
                            })

            try:
                notify_progress_all(data={
                    "type": "feature_boll_update",
                    "bucket": ts_iso,
                    "summary": {k: v for k, v in boll_debug.items() if k != "samples"},
                    "samples": boll_debug["samples"],
                })
            except Exception:
                pass
        except Exception as e:
            try:
                notify_progress_all(data={
                    "type": "feature_boll_error",
                    "bucket": ts_iso,
                    "error": repr(e),
                })
            except Exception:
                pass



    return {
        "ts_iso": ts_iso,
        "ok": ok,
        "failed": failed,
        "total": ok + failed,
        "error_hist": dict(err_counter),
        "errors": errors[:MAX_BUCKET_ERROR_SAMPLES],
        "chart_points": chart_points,  # 传给回调聚合后做图表更新
    }

#-----------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------------
#---------------------------------------------------------------
#-----------------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------
# -----------------------------------------------
# 回调：聚合所有分钟桶，广播最终“done + 图表增量”
# -----------------------------------------------
@shared_task(name="AppleStockChecker.tasks.psta_finalize_buckets")
def psta_finalize_buckets(
    results: List[Dict[str, Any]],
    job_id: str,
    ts_iso: str,
    agg_ctx: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    - 作用：在一批“按分钟对齐”的子任务完成后，汇总每个分钟桶的结果，生成给前端的图表增量（真实点 + 必要的“影子点”），并通过通知通道广播最终 done 消息。

    - 输入参数
      - results：每个分钟桶的汇总结果列表。每个元素可能包含：
        - ok、failed：该桶处理成功/失败数
        - error_hist：该桶的错误直方图
        - chart_points：写库成功后用于画图的点列表，每个点含 {id,t,price,recorded_at,iphone_id,shop_id}
      - job_id：本次任务标识
      - ts_iso：目标分钟时间戳（ISO 字符串）

    - 汇总统计
      - 计算总桶数、总 ok、总 failed。
      - 聚合所有桶的 error_hist，得到全局错误直方图 agg_err。

    - 聚合真实点
      - 以 (iphone_id, shop_id) 作为序列键，收集该序列的真实点到 series_map。
      - 统计 total_points（仅统计真实点）。

    - 计算每条序列的“已知最后点”与“是否在 ts_iso 有真实点”
      - last_known：每个序列在 t <= ts_iso 的最后一个真实点。
      - has_real_at_ts：指示该序列在 ts_iso 是否已有真实点。
      - 时间比较基于 ISO 字符串（假设格式一致）。

    - 全局裁剪（仅裁真实点，影子点不受限）
      - 若 total_points 超过 MAX_PUSH_POINTS：
        - 将所有真实点拍平成时间序列升序，保留最近 MAX_PUSH_POINTS 条；
        - 重建 series_map；
        - 标记 clipped = True。

    - 生成图表增量 series_delta
      - 遍历所有出现过的序列键（保证即便被裁剪为空的序列，也可补影子点）。
      - 每个序列内按时间排序；
      - 若该序列在 ts_iso 没有真实点但存在 last_known，则追加一个“影子点”：
        - 结构：{id: None, t: ts_iso, price: 最近真实点价格, recorded_at: 同源记录时间, shadow: True, src_t: 影子来源的时间}
      - 记录本次补入的影子点数量 shadow_points_added。

    - 组织返回与广播
      - summary：包含本次时间戳、job_id、总桶数、ok/failed、全局错误直方图，以及前 100 条桶级摘要。
      - chart_delta：包含 job_id、timestamp、series_delta 以及 meta：
        - total_points：min(真实点总数, MAX_PUSH_POINTS)
        - shadow_points：本次补的影子点数量
        - clipped：是否发生裁剪
      - payload：status=done，step=finalize，progress=100，携带上述 summary 与 chart_delta。
      - 调用 notify_progress_all(data=payload) 广播（异常忽略）。

    - 返回值
      - 返回 payload，便于链路调试或上层任务使用。
    """
    from collections import defaultdict, Counter

    # --- 标准化 results（有时不是 list） ---
    if isinstance(results, dict):
        results = [results]
    elif results is None:
        results = []
    # === 汇总计数 ===
    total_buckets = len(results or [])
    total_ok = sum(int(r.get("ok", 0)) for r in results or [])
    total_failed = sum(int(r.get("failed", 0)) for r in results or [])

    # === 错误直方图 ===
    agg_err = Counter()
    for r in results or []:
        for k, v in (r.get("error_hist") or {}).items():
            agg_err[k] += v

    # === 聚合真实点 ===
    # key: (iphone_id, shop_id) -> List[point]
    series_map = defaultdict(list)
    total_points = 0
    for r in results or []:
        for p in (r.get("chart_points") or []):
            key = (p.get("iphone_id"), p.get("shop_id"))
            series_map[key].append({
                "id": p.get("id"),
                "t": p.get("t"),
                "price": p.get("price"),
                "recorded_at": p.get("recorded_at"),
            })
            total_points += 1

    # === 计算每个序列在 ts_iso 之前（含）的最后一个真实点（last-known），以及是否在 ts_iso 有真实点 ===
    # 为避免歧义，这里以 ISO 字符串的时间比较为准（你们上下文里 t 和 ts_iso 的格式一致）。
    # 若担心跨时区 ISO 文本比较的稳定性，可改为 _to_aware 做 datetime 比较。
    last_known = {}        # key -> dict(point)
    has_real_at_ts = {}    # key -> bool
    for key, pts in series_map.items():
        # 找 <= ts_iso 的最大 t
        latest = None
        latest_t = None
        at_ts = False
        for item in pts:
            t_iso = item["t"]
            if t_iso == ts_iso:
                at_ts = True
            # 选择 <= ts_iso 中最大的 t
            if t_iso <= ts_iso and (latest_t is None or t_iso > latest_t):
                latest = item
                latest_t = t_iso
        if latest:
            last_known[key] = latest
        has_real_at_ts[key] = at_ts

    # === 全局截断（仅对真实点生效；影子点不受 MAX_PUSH_POINTS 限制） ===
    clipped = False
    if total_points > MAX_PUSH_POINTS:
        clipped = True
        flat = []
        for (iphone_id, shop_id), pts in series_map.items():
            for item in pts:
                flat.append((item["t"], iphone_id, shop_id, item))
        flat.sort(key=lambda x: x[0])      # 升序
        flat = flat[-MAX_PUSH_POINTS:]     # 保留最近 N 条

        series_map = defaultdict(list)
        for _, iphone_id, shop_id, item in flat:
            series_map[(iphone_id, shop_id)].append(item)

    # === 生成最终增量：真实点 +（必要时）影子点 ===
    series_delta = []
    shadow_points_added = 0
    # 注意：用所有出现过的 key（包括被截断后的空系列，保证影子点也能出现）
    all_keys = set(last_known.keys()) | set(series_map.keys())

    for (iphone_id, shop_id) in all_keys:
        pts = series_map.get((iphone_id, shop_id), [])
        # 保证时间有序
        pts.sort(key=lambda x: x["t"])

        # 若该序列在 ts_iso 没有真实点，但有 last-known，则补影子点
        if not has_real_at_ts.get((iphone_id, shop_id), False) and (iphone_id, shop_id) in last_known:
            src = last_known[(iphone_id, shop_id)]
            # 避免与真实点重复（理论上 has_real_at_ts 已排除）
            if not any(p["t"] == ts_iso for p in pts):
                shadow_points_added += 1
                pts.append({
                    "id": None,                 # 影子点不落库，无 id
                    "t": ts_iso,                # 影子点放在标的时间戳
                    "price": src["price"],      # 以最近真实点的价格填充
                    "recorded_at": src.get("recorded_at"),
                    "shadow": True,             # ✅ 标识影子点
                    "src_t": src["t"],          # 影子来源时间（便于前端 tooltip/样式）
                })

        series_delta.append({
            "iphone_id": iphone_id,
            "shop_id": shop_id,
            "points": pts,  # [{id,t,price,recorded_at,shadow?,src_t?}, ...]
        })

    # === 构建汇总与广播 payload ===
    summary = {
        "timestamp": ts_iso,
        "job_id": job_id,
        "total_buckets": total_buckets,
        "ok": total_ok,
        "failed": total_failed,
        "error_hist": dict(agg_err),
        "by_bucket": [
            {k: r.get(k) for k in ("ts_iso", "ok", "failed", "total", "error_hist")}
            for r in (results or [])
        ][:100],
    }

    payload = {
        "status": "done",
        "step": "finalize",
        "progress": 100,
        "summary": summary,
        "chart_delta": {
            "job_id": job_id,
            "timestamp": ts_iso,
            "series_delta": series_delta,
            "meta": {
                "total_points": min(total_points, MAX_PUSH_POINTS),  # 仅真实点计数
                "shadow_points": shadow_points_added,                # 本次补的影子点数
                "clipped": clipped,
            }
        }
    }

    try:
        notify_progress_all(data=payload)
    except Exception:
        pass

    return payload
#-----------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------------
#---------------------------------------------------------------
#-----------------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------
# -----------------------------------------------
# 父任务：chord 并行 + 回调（保持你已有写法）
# -----------------------------------------------



def _to_aware(s: str) -> timezone.datetime:
    from django.utils.dateparse import parse_datetime
    from django.utils.timezone import make_aware, is_naive
    dt = parse_datetime(s)
    if dt is None:
        raise ValueError(f"bad datetime iso: {s}")
    return make_aware(dt) if is_naive(dt) else dt

def _floor_to_step(dt: timezone.datetime, step_min: int) -> timezone.datetime:
    return dt - timezone.timedelta(minutes=dt.minute % step_min, seconds=dt.second, microseconds=dt.microsecond)

def _rolling_start(dt: timezone.datetime, step_min: int) -> timezone.datetime:
    return dt.replace(second=0, microsecond=0) - timezone.timedelta(minutes=max(step_min - 1, 0))



@shared_task(bind=True, name="AppleStockChecker.tasks.batch_generate_psta_same_ts")
def batch_generate_psta_same_ts(
    self,
    *,
    job_id: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    timestamp_iso: Optional[str] = None,
    chunk_size: int = 200,
    query_window_minutes: int = 15,
    shop_ids: Optional[List[int]] = None,
    iphone_ids: Optional[List[int]] = None,
    max_items: Optional[int] = None,
    # ✅ 新：聚合控制
    agg_minutes: int = 15,                 # 聚合步长
    agg_mode: str = "boundary",            # 'boundary'|'rolling'|'off'
    force_agg: bool = False,               # 强制本轮聚合
) -> Dict[str, Any]:

    task_job_id = job_id or self.request.id
    ts_iso = timestamp_iso or nearest_past_minute_iso()
    MODE = (agg_mode or "boundary").lower()

    pack = (collect_items_for_psta(
        window_minutes=query_window_minutes,
        timestamp_iso=ts_iso,
        shop_ids=shop_ids,
        iphone_ids=iphone_ids,
        max_items=max_items,
    ) or [{}])[0]

    rows = pack.get("rows") or []
    bucket_minute_key: Dict[str, Dict[str, List[int]]] = pack.get("bucket_minute_key") or {}

    # 广播聚合上下文（便于在前端/日志中确认）
    dt0 = _to_aware(ts_iso)
    step0 = _floor_to_step(dt0, int(agg_minutes))
    ctx = {
        "agg_minutes": int(agg_minutes),
        "agg_mode": MODE,
        "force_agg": bool(force_agg),
        "bucket_start": step0.isoformat(),
        "bucket_end": (step0 + timezone.timedelta(minutes=int(agg_minutes))).isoformat(),
    }
    try:
        notify_progress_all(data={"type":"agg_ctx","timestamp":ts_iso,"job_id":task_job_id,"ctx":ctx})
    except Exception:
        pass

    # 构建子任务（注意：即使该分钟无行，但需要聚合，也要下发“空行聚合”任务）
    subtasks: List = []
    for minute_iso, key_map in bucket_minute_key.items():
        minute_rows: List[Dict[str, Any]] = []
        for _, idx_list in (key_map or {}).items():
            for i in idx_list:
                if 0 <= i < len(rows):
                    r = rows[i]
                    minute_rows.append({
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                        "price_new": r.get("price_new", r.get("New_Product_Price")),
                    })

        mdt = _to_aware(minute_iso)
        boundary = _floor_to_step(mdt, int(agg_minutes))
        is_boundary = (mdt == boundary)

        if MODE == "off":
            do_agg_local = False
            agg_start_iso = None
        elif MODE == "rolling":
            do_agg_local = True
            agg_start_iso = _rolling_start(mdt, int(agg_minutes)).isoformat()
        else:  # boundary
            do_agg_local = bool(force_agg) or is_boundary
            agg_start_iso = boundary.isoformat()

        # 若 minute_rows 空，但 do_agg_local=True（例如边界分钟），仍然下发“仅聚合”的子任务
        if minute_rows or do_agg_local:
            subtasks.append(
                psta_process_minute_bucket.s(
                    ts_iso=minute_iso,
                    rows=minute_rows,
                    job_id=task_job_id,
                    do_agg=do_agg_local,
                    agg_start_iso=agg_start_iso,
                    agg_minutes=int(agg_minutes),
                )
            )

    try:
        notify_progress_all(data={"status":"running","step":"dispatch_buckets","buckets":len(subtasks),"timestamp":ts_iso,"agg":ctx})
    except Exception:
        pass

    if not subtasks:
        empty = {"timestamp": ts_iso, "total_buckets": 0, "ok": 0, "failed": 0, "by_bucket": []}
        try:
            notify_progress_all(data={
                "status":"done","progress":100,"summary":empty,
                "chart_delta":{"job_id":task_job_id,"timestamp":ts_iso,"series_delta":[],"meta":{"total_points":0,"shadow_points":0,"clipped":False}}
            })
        except Exception:
            pass
        return empty

    callback = psta_finalize_buckets.s(job_id=task_job_id, ts_iso=ts_iso, agg_ctx=ctx)   # 可把 ctx 传给回调（可选）
    chord_result = chord(subtasks)(callback)
    return {"timestamp": ts_iso, "total_buckets": len(subtasks), "job_id": task_job_id, "chord_id": chord_result.id}
#-----------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------------
#---------------------------------------------------------------
#-----------------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------

