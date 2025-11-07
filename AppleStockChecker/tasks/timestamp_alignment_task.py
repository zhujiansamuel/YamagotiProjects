from __future__ import annotations
from ..serializers import PurchasingShopTimeAnalysisSerializer
from collections import Counter
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import IntegrityError
from django.utils.dateparse import parse_datetime
from AppleStockChecker.ws_notify import notify_progress_all
from AppleStockChecker.models import SecondHandShop, Iphone
from ..ws_notify import notify_progress, notify_batch_items
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta
from celery import shared_task, group, chord
from celery import shared_task, group
from django.db import transaction, IntegrityError
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from typing import Optional, List, Dict, Any
from AppleStockChecker.collectors import collect_items_for_psta
from AppleStockChecker.utils.timebox import nearest_past_minute_iso
from AppleStockChecker.models import PurchasingShopTimeAnalysis, SecondHandShop, Iphone
from AppleStockChecker.services.time_analysis_services import upsert_purchasing_time_analysis
from AppleStockChecker.serializers import PSTACompactSerializer
from AppleStockChecker.ws_notify import (
    notify_progress_all,
    notify_batch_items_all,
    notify_batch_done_all,
)
# AppleStockChecker/tasks.py
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from celery import shared_task, chord
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import transaction, IntegrityError

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
    job_id: str
) -> Dict[str, Any]:
    """
    为某分钟桶生成/更新 PSTA。
    返回：
      - 错误详情与直方图（便于定位 why ok=0 failed>0）
      - chart_points：给 finalize 汇总后推送到前端的图表增量（去重由前端做）


    总体作用
    - 这是一个 Celery 子任务（名为 AppleStockChecker.tasks.psta_process_minute_bucket），处理“某一个分钟桶”的数据写入与统计。
    - 输入的 rows 是该分钟内按店铺与机型聚合好的轻量行；任务会对每条进行校验、入库（存在则更新、否则创建），并收集可用于前端绘图的“图表点”。
    - 返回该分钟桶的处理摘要（成功/失败计数、错误直方图、错误样例）以及图表增量列表 chart_points。后续由 psta_finalize_buckets 回调聚合，并通过 notify_progress_all 推送前端。

    关键输入
    - ts_iso：该“分钟桶”的时间（ISO 字符串）。
    - rows：行列表，每行至少应包含 shop_id、iphone_id、recorded_at，以及 price_new 或 New_Product_Price。
    - job_id：本次任务流水号，便于审计与幂等追踪。

    处理流程
    1) 初始化上下文
    - 把 ts_iso 转为有时区的 datetime（ts_dt），同时计算该时间的时区偏移字符串 ts_tz（如 +09:00）。
    - 固定源记录时区 orig_tz = +09:00。
    - 预备计数器 ok/failed、错误样本 errors 和错误类型直方图 err_counter，以及前端图表用的 chart_points。

    2) 逐行校验与入库（事务+行锁）
    - 轻量校验：必须有 shop_id、iphone_id。
    - 外键存在性检查：SecondHandShop 与 Iphone 若不存在会抛 DoesNotExist（便于排错）。
    - 解析 recorded_at 为 aware datetime（rec_dt）。
    - 价格 new_price 从 price_new 或 New_Product_Price 取值，缺失则报错。
    - 计算 align_diff = (rec_dt - ts_dt) 的秒差，便于对齐误差诊断。
    - 在 transaction.atomic 中，对 (shop_id, iphone_id, Timestamp_Time=ts_dt) 做 select_for_update()：
      - 若已存在：更新 Job_ID、时区、Record_Time、Alignment_Time_Difference、New_Product_Price，并将 Update_Count +1。
      - 若不存在：创建一条新记录（Update_Count=0）。
    - 成功一条 ok += 1。

    3) 收集图表增量
    - 每次成功写库后，若 chart_points 未超过 MAX_BUCKET_CHART_POINTS，则追加一个点：
      - {id: 数据库主键, t: ts_iso, iphone_id, shop_id, price: new_price, recorded_at: rec_dt.isoformat()}
    - 这些点只作为“真实点”传给回调聚合使用；回调可能再补“影子点”保证时间轴连续。

    4) 错误处理与桶级报警
    - 针对常见异常（DoesNotExist、ValidationError、IntegrityError、TypeError、ValueError）：
      - failed += 1，err_counter 记录异常类名，errors 收集有限条样例（含简要 item 字段）。
    - 兜底 Exception 同样计入 failed/err_counter，并收集精简样例。
    - 若本桶有失败，发送一个 bucket_errors 调试通知（携带 ts_iso、总数、ok/failed、错误直方图、前 5 条样例），通知异常会被忽略不阻断主流程。

    返回结果
    - 返回一个字典：
      - ts_iso、ok、failed、total
      - error_hist：异常类名到次数的直方图
      - errors：失败样例（不超过 MAX_BUCKET_ERROR_SAMPLES）
      - chart_points：成功写库的图表点（用于后续 finalize 聚合与前端增量刷新）

    设计意图与特点
    - 幂等与并发安全：按 (shop, iphone, minute) 做行级锁 + 唯一约束，避免重复与竞争。
    - 轻重分离：子任务只做 upsert 与点收集；复杂的汇总、裁剪、影子点补齐等留给 psta_finalize_buckets。
    - 诊断友好：详细的错误直方图与样例，让“ok=0、failed>0”的问题能被快速定位。
    - 时区与对齐：所有时间使用 aware datetime 计算；Alignment_Time_Difference 使“记录发生时间”与“桶时间”对齐偏差一目了然。



    全部店 × 各 iPhone → 写 OverallBar(bucket, iphone)（若你的 OverallBar 尚未加 iphone 外键，会自动跳过并输出 debug）。

    全部店 × 组合 iPhone → 写 CohortBar(bucket, cohort)。

    各店/店铺组合 × 各 iPhone → 写入 FeatureSnapshot(scope=..., name=...)。

    各店/店铺组合 × 组合 iPhone → 写入 FeatureSnapshot(scope=..., name=...)。

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
        return (Decimal(str(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

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


    ok = 0
    failed = 0
    errors: List[Dict[str, Any]] = []
    err_counter = Counter()

    ts_dt = _to_aware(ts_iso)
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
    if ok > 0:
        from statistics import median

        WATERMARK_MINUTES = 5
        now = timezone.now()
        is_final_bar = ts_dt <= (now - timezone.timedelta(minutes=WATERMARK_MINUTES))

        # 1) OverallBar（全部店 × 各 iPhone）
        # 自动探测 OverallBar 是否含 iphone 外键；若没有，跳过以免 unique(bucket) 冲突
        ob_has_iphone = any(getattr(f, "name", "") == "iphone" for f in OverallBar._meta.get_fields())
        overallbar_debug = {"bucket": ts_iso, "iphones": [], "skipped": False}

        try:
            if not ob_has_iphone:
                overallbar_debug["skipped"] = True
            else:
                bucket_iphone_ids = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})
                for ipid in bucket_iphone_ids:
                    qs_prices = PurchasingShopTimeAnalysis.objects.filter(
                        Timestamp_Time=ts_dt, iphone_id=ipid
                    ).values_list("New_Product_Price", flat=True)

                    prices = [float(p) for p in qs_prices if p is not None]
                    if not prices:
                        continue

                    shop_cnt = PurchasingShopTimeAnalysis.objects.filter(
                        Timestamp_Time=ts_dt, iphone_id=ipid
                    ).values("shop_id").distinct().count()

                    vals = sorted(prices)
                    m_mean = sum(vals) / len(vals)
                    m_median = float(median(vals))
                    m_std = _pop_std(vals)
                    p10 = _quantile(vals, 0.10);
                    p90 = _quantile(vals, 0.90)
                    dispersion = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0

                    OverallBar.objects.update_or_create(
                        bucket=ts_dt, iphone_id=ipid,
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
                })
            except Exception:
                pass

        # 2) CohortBar（全部店 × 组合 iPhone）
        cohort_debug = {"bucket": ts_iso, "cohorts": []}
        try:
            if ob_has_iphone:
                cohorts = list(Cohort.objects.all())
                for coh in cohorts:
                    members = list(CohortMember.objects.filter(cohort=coh).values("iphone_id", "weight"))
                    if not members:
                        continue
                    member_ids = [m["iphone_id"] for m in members]
                    weight_map = {m["iphone_id"]: float(m.get("weight") or 1.0) for m in members}

                    ob_rows = list(
                        OverallBar.objects.filter(bucket=ts_dt, iphone_id__in=member_ids)
                        .values("iphone_id", "mean", "shop_count")
                    )
                    vals = [float(r["mean"]) for r in ob_rows if r.get("mean") is not None]
                    if not vals:
                        continue

                    # 机型加权（成员权重 × 覆盖），无可用权重时退化为等权
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
                    p10 = _quantile(vals_sorted, 0.10);
                    p90 = _quantile(vals_sorted, 0.90)
                    c_disp = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0
                    n_models = len(vals_sorted)
                    shop_count_agg = sum(int(r.get("shop_count") or 0) for r in ob_rows)

                    CohortBar.objects.update_or_create(
                        bucket=ts_dt, cohort=coh,
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
                        "reason": "OverallBar lacks iphone dimension; skip CohortBar to avoid collisions."
                    })
                except Exception:
                    pass
        except Exception as e:
            try:
                notify_progress_all(data={
                    "type": "cohortbar_error",
                    "bucket": ts_iso,
                    "error": repr(e),
                })
            except Exception:
                pass

        # 3) 四类组合统计：写入 FeatureSnapshot
        try:
            # —— 准备桶内 (shop, iphone) 样本 —— #
            shops_seen = sorted({int(r.get("shop_id")) for r in rows if r.get("shop_id")})
            iphones_seen = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})

            qs_all = (PurchasingShopTimeAnalysis.objects
                      .filter(Timestamp_Time=ts_dt,
                              shop_id__in=shops_seen, iphone_id__in=iphones_seen)
                      .values('shop_id', 'iphone_id', 'New_Product_Price'))

            price_by_si = defaultdict(list)  # (shop, iphone) -> [price,...]
            for rec in qs_all:
                p = rec.get('New_Product_Price')
                if p is None:
                    continue
                s = int(rec['shop_id']);
                i = int(rec['iphone_id']);
                v = float(p)
                price_by_si[(s, i)].append(v)

            def _stats(values):
                """返回 (mean, median, std, dispersion, count)。"""
                if not values:
                    return None
                vals = sorted(values)
                n = len(vals)
                mean_v = sum(vals) / n
                # 中位数（偶数取两中间平均）
                med_v = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
                std_v = _pop_std(vals)
                p10 = _quantile(vals, 0.10);
                p90 = _quantile(vals, 0.90)
                disp_v = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0
                return mean_v, med_v, std_v, disp_v, n

            def upsert_feature(scope: str, name: str, value: float, *, is_final: bool, version: str = 'v1'):
                FeatureSnapshot.objects.update_or_create(
                    bucket=ts_dt, scope=scope, name=name, version=version,
                    defaults=dict(value=float(value), is_final=is_final)
                )

            combo_debug = {
                "bucket": ts_iso,
                "case1_shop_iphone": 0,
                "case2_shopcohort_iphone": 0,
                "case3_shop_cohortiphone": 0,
                "case4_shopcohort_cohortiphone": 0,
                "skipped": [],
                "samples": [],
            }

            # === CASE 1: 各店 × 各 iPhone ===
            for (sid, iid), vals in price_by_si.items():
                s = _stats(vals)
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
                    combo_debug["samples"].append({"case": 1, "scope": scope, "n": n, "mean": round(m, 4)})

            # 预取店铺组合（ShopWeightProfile）
            profiles = list(ShopWeightProfile.objects.all())
            prof_items = {
                prof.id: {it['shop_id']: float(it.get('weight') or 1.0)
                          for it in ShopWeightItem.objects.filter(profile=prof).values('shop_id', 'weight')}
                for prof in profiles
            }
            has_shop_profile = any(bool(prof_items.get(p.id)) for p in profiles)

            # === CASE 2: 组合店 × 各 iPhone ===
            if has_shop_profile:
                for prof in profiles:
                    sw = prof_items.get(prof.id, {})
                    if not sw:
                        continue
                    shops_in = set(sw.keys()) & set(shops_seen)
                    if not shops_in:
                        continue
                    for iid in iphones_seen:
                        vals = []
                        wnum = wden = 0.0
                        for sid in shops_in:
                            vlist = price_by_si.get((int(sid), int(iid)))
                            if not vlist:
                                continue
                            v = float(vlist[-1])
                            w = float(sw.get(sid, 1.0))
                            vals.append(v)
                            wnum += w * v
                            wden += w
                        if not vals:
                            continue
                        m, med, st, disp, n = _stats(vals)
                        mean_w = (wnum / wden) if wden > 0 else m
                        scope = f"shopcohort:{prof.slug}|iphone:{iid}"
                        upsert_feature(scope, "mean", mean_w, is_final=is_final_bar)
                        upsert_feature(scope, "median", med, is_final=is_final_bar)
                        upsert_feature(scope, "std", st, is_final=is_final_bar)
                        upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                        upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                        combo_debug["case2_shopcohort_iphone"] += 1
                        if len(combo_debug["samples"]) < 5:
                            combo_debug["samples"].append({"case": 2, "scope": scope, "n": n, "mean": round(mean_w, 4)})
            else:
                combo_debug["skipped"].append("case2: no ShopWeightProfile defined")

            # 预取机型组合（Cohort）
            cohorts = list(Cohort.objects.all())
            cmembers = {
                coh.id: {m['iphone_id']: float(m.get('weight') or 1.0)
                         for m in CohortMember.objects.filter(cohort=coh).values('iphone_id', 'weight')}
                for coh in cohorts
            }

            # === CASE 3: 各店 × 组合 iPhone ===
            for sid in shops_seen:
                for coh in cohorts:
                    iw = cmembers.get(coh.id, {})
                    if not iw:
                        continue
                    vals = []
                    wnum = wden = 0.0
                    for iid, w in iw.items():
                        vlist = price_by_si.get((int(sid), int(iid)))
                        if not vlist:
                            continue
                        v = float(vlist[-1])
                        vals.append(v)
                        wnum += float(w) * v
                        wden += float(w)
                    if not vals:
                        continue
                    m, med, st, disp, n = _stats(vals)
                    mean_w = (wnum / wden) if wden > 0 else m
                    scope = f"shop:{sid}|cohort:{coh.slug}"
                    upsert_feature(scope, "mean", mean_w, is_final=is_final_bar)
                    upsert_feature(scope, "median", med, is_final=is_final_bar)
                    upsert_feature(scope, "std", st, is_final=is_final_bar)
                    upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                    upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                    combo_debug["case3_shop_cohortiphone"] += 1
                    if len(combo_debug["samples"]) < 5:
                        combo_debug["samples"].append({"case": 3, "scope": scope, "n": n, "mean": round(mean_w, 4)})

            # === CASE 4: 组合店 × 组合 iPhone ===
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
                        vals = []
                        wnum = wden = 0.0
                        for sid, w_shop in sw.items():
                            if int(sid) not in shops_in:
                                continue
                            for iid, w_phone in iw.items():
                                vlist = price_by_si.get((int(sid), int(iid)))
                                if not vlist:
                                    continue
                                v = float(vlist[-1])
                                vals.append(v)
                                w = float(w_shop) * float(w_phone)
                                wnum += w * v
                                wden += w
                        if not vals:
                            continue
                        m, med, st, disp, n = _stats(vals)
                        mean_w = (wnum / wden) if wden > 0 else m
                        scope = f"shopcohort:{prof.slug}|cohort:{coh.slug}"
                        upsert_feature(scope, "mean", mean_w, is_final=is_final_bar)
                        upsert_feature(scope, "median", med, is_final=is_final_bar)
                        upsert_feature(scope, "std", st, is_final=is_final_bar)
                        upsert_feature(scope, "dispersion", disp, is_final=is_final_bar)
                        upsert_feature(scope, "count", float(n), is_final=is_final_bar)
                        combo_debug["case4_shopcohort_cohortiphone"] += 1
                        if len(combo_debug["samples"]) < 5:
                            combo_debug["samples"].append({"case": 4, "scope": scope, "n": n, "mean": round(mean_w, 4)})
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
    ts_iso: str
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
@shared_task(bind=True, name="AppleStockChecker.tasks.batch_generate_psta_same_ts")
def batch_generate_psta_same_ts(
    self,
    *,
    job_id: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,   # 兼容
    timestamp_iso: Optional[str] = None,
    chunk_size: int = 200,                          # 兼容
    query_window_minutes: int = 15,
    shop_ids: Optional[List[int]] = None,
    iphone_ids: Optional[List[int]] = None,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    """
    总体作用:
    - 这是一个 Celery 编排任务（父任务），按同一“分钟时间戳”对齐回收价数据，切分为“按分钟的子任务”并行处理，最后用回调汇总结果并通过广播通知前端。

    主要步骤:
    1) 任务标识与目标时间
       - job_id 取入参或当前任务 id。
       - ts_iso 取入参 timestamp_iso，若无则用 nearest_past_minute_iso()（最近的整分钟）。

    2) 收集待处理数据
       - 调 collect_items_for_psta(...)，得到打包结果 pack。
       - 其中 rows 是轻量数据列表；bucket_minute_key 是“分钟 → key 映射 → 行索引”的桶结构。

    3) 构建子任务列表（按分钟桶）
       - 遍历 bucket_minute_key 中的每个分钟 minute_iso。
       - 根据索引收集该分钟对应的行，抽取必要字段：shop_id、iphone_id、recorded_at、price_new。
       - 为该分钟创建一个子任务 psta_process_minute_bucket.s(...)。
       - 每个分钟桶对应一个子任务，形成 subtasks 列表。

    4) 广播开始进度（可选）
       - notify_progress_all 发送 status=running，包含桶数与基准时间。

    5) 无任务时的快速返回
       - 若 subtasks 为空：广播一个 done（空摘要和空图表增量），并直接返回该空摘要。

    6) 并行执行与回调汇总
       - 使用 chord(subtasks)(callback) 并行执行所有分钟子任务。
       - callback 为 psta_finalize_buckets.s(task_job_id, ts_iso)：
         - 汇总每桶的 ok/failed 与错误直方图；
         - 聚合图表点，并在 ts_iso 补“影子点”（某序列该分钟没有真实点时，用最近真实价格补一条仅用于前端展示的点）；
         - 通过 notify_progress_all 广播最终 done + chart_delta。

    7) 返回编排信息
       - 立即返回 {timestamp, total_buckets, job_id, chord_id}，前端可用 chord_id 跟踪，最终结果依赖广播消息。

    要点:
    - 使用 Celery chord 实现“并行子任务 + 汇总回调”的扇出/汇合，不阻塞等待结果（不调用 .get()）。
    - 按分钟分桶，有利于并行与数据库行级锁的分散。
    - 广播环节 try/except 防御，避免通知失败影响主流程。
    """
    task_job_id = job_id or self.request.id
    ts_iso = timestamp_iso or nearest_past_minute_iso()

    pack = collect_items_for_psta(
        window_minutes=query_window_minutes,
        timestamp_iso=ts_iso,
        shop_ids=shop_ids,
        iphone_ids=iphone_ids,
        max_items=max_items,
    )[0]
    rows = pack["rows"]
    bucket_minute_key = pack.get("bucket_minute_key") or {}

    # 按“分钟桶”构建子任务列表
    subtasks: List = []
    for minute_iso, key_map in bucket_minute_key.items():
        minute_rows: List[Dict[str, Any]] = []
        for _, idx_list in key_map.items():
            for i in idx_list:
                if 0 <= i < len(rows):
                    r = rows[i]
                    minute_rows.append({
                        "shop_id": r.get("shop_id"),
                        "iphone_id": r.get("iphone_id"),
                        "recorded_at": r.get("recorded_at"),
                        "price_new": r.get("price_new", r.get("New_Product_Price")),
                    })
        if minute_rows:
            # 一分钟桶一个子任务
            subtasks.append(
                psta_process_minute_bucket.s(ts_iso=minute_iso, rows=minute_rows, job_id=task_job_id)
            )

    total_buckets = len(subtasks)

    # 开始广播（可选）
    try:
        notify_progress_all(data={
            "status": "running",
            "step": "dispatch_buckets",
            "progress": 0,
            "buckets": total_buckets,
            "timestamp": ts_iso
        })
    except Exception:
        pass

    if not subtasks:
        # 无桶可处理，直接返回空摘要
        empty = {"timestamp": ts_iso, "total_buckets": 0, "ok": 0, "failed": 0, "by_bucket": []}
        try:
            notify_progress_all(data={
                "status": "done",
                "progress": 100,
                "summary": empty,
                "chart_delta": {"job_id": task_job_id, "timestamp": ts_iso, "series_delta": [], "meta": {"total_points": 0, "clipped": False}}
            })
        except Exception:
            pass
        return empty

    # ★ 使用 chord 并行执行 + 汇总回调，不要 .get()
    callback = psta_finalize_buckets.s(task_job_id, ts_iso)
    chord_result = chord(subtasks)(callback)

    # 返回“编排任务”信息；前端通过广播拿最终 summary+chart_delta
    return {
        "timestamp": ts_iso,
        "total_buckets": total_buckets,
        "job_id": task_job_id,
        "chord_id": chord_result.id,
    }


#-----------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------------
#---------------------------------------------------------------
#-----------------------------------------------------------
#--------------------------------------------------------
#-----------------------------------------------------

