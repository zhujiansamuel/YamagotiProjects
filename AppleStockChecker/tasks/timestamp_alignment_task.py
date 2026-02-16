from __future__ import annotations
import logging
from datetime import timedelta
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from AppleStockChecker.utils.timestamp_alignment_task import (
    collect_items_for_psta,
    nearest_past_minute_iso,
    notify_progress_all,
    notify_batch_items_all,
    notify_batch_done_all,
    FeatureWriter,
    FeatureRecord,
)
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from celery import shared_task, chord, chain
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import transaction, IntegrityError
from decimal import Decimal, ROUND_HALF_UP
import os
import logging
from typing import Any, Callable, Dict, Iterable, Tuple, Optional, Union
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# 环境开关：参数严格度 & 版本阈值
PARAM_STRICT = os.getenv("PSTA_PARAM_STRICT", "warn").strip().lower()  # ignore|warn|error
MIN_ACCEPTED_TASK_VER = int(os.getenv("PSTA_MIN_ACCEPTED_VER", "0"))  # 小于此版本直接报错（可选）

# ---- 小工具：类型转换 ----
_TRUE = {"1", "true", "t", "y", "yes", "on"}
_FALSE = {"0", "false", "f", "n", "no", "off"}


def to_bool(x: Any) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in _TRUE: return True
        if s in _FALSE: return False
    raise ValueError(f"cannot coerce to bool: {x!r}")


def to_int(x: Any) -> int:
    if x is None: return None  # 允许上层决定是否必填
    return int(x)


def ensure_list(x: Any) -> list:
    if x is None: return []
    return list(x) if not isinstance(x, list) else x


def _isinstance_soft(val: Any, typ: Union[type, Tuple[type, ...]]) -> bool:
    # 允许传入 (int, str) 这样的 tuple
    try:
        return isinstance(val, typ)
    except TypeError:
        # 不做深度校验
        return True


# ---- 守卫核心 ----
def guard_params(
        task_name: str,
        incoming: Dict[str, Any],
        *,
        required: Dict[str, Union[type, Tuple[type, ...]]],
        optional: Dict[str, Union[type, Tuple[type, ...]]] = None,
        defaults: Dict[str, Any] = None,
        aliases: Dict[str, str] = None,  # 形参更名：old -> new（仅顶层）
        coerce: Dict[str, Callable[[Any], Any]] = None,
        task_ver_field: str = "task_ver",
        expected_ver: Optional[int] = None,  # 推荐：与 producer 同步填写
        notify: Optional[Callable[[dict], Any]] = None,  # 例如 notify_progress_all
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    返回 (normalized_kwargs, meta)
    - 对未知参数按策略 ignore/warn/error 处理
    - 进行别名迁移、默认值填充、类型转换与校验
    - 检查 task_ver（若提供）
    """
    optional = optional or {}
    defaults = defaults or {}
    aliases = aliases or {}
    coerce = coerce or {}

    kw = dict(incoming)  # 浅拷贝

    # 1) 顶层别名迁移（old->new）
    used_aliases = {}
    for old, new in aliases.items():
        if old in kw and new not in kw:
            kw[new] = kw.pop(old)
            used_aliases[old] = new

    # 2) 未知参数处理策略
    declared = set(required.keys()) | set(optional.keys()) | {task_ver_field}
    unknown_keys = sorted(k for k in kw.keys() if k not in declared)
    if unknown_keys:
        msg = f"[{task_name}] unknown params: {unknown_keys} (strict={PARAM_STRICT})"
        if PARAM_STRICT == "error":
            raise TypeError(msg)
        elif PARAM_STRICT == "warn":
            logger.warning(msg)
            if notify:
                try:
                    notify({"type": "param_compat_warning", "task": task_name, "unknown": unknown_keys})
                except Exception:
                    pass
        # ignore: 什么也不做

    # 3) 默认值
    for k, v in defaults.items():
        if kw.get(k) is None:
            kw[k] = v

    # 4) 类型转换（coerce）
    for k, fn in coerce.items():
        if k in kw and kw[k] is not None:
            try:
                kw[k] = fn(kw[k])
            except Exception as e:
                raise ValueError(f"[{task_name}] bad param '{k}': {e}")

    # 5) 必填与可选的类型校验
    for k, typ in required.items():
        if kw.get(k) is None:
            raise ValueError(f"[{task_name}] missing required param: '{k}'")
        if not _isinstance_soft(kw[k], typ):
            raise TypeError(f"[{task_name}] param '{k}' type error: got {type(kw[k]).__name__}, expect {typ}")

    for k, typ in optional.items():
        if k in kw and kw[k] is not None and not _isinstance_soft(kw[k], typ):
            raise TypeError(f"[{task_name}] param '{k}' type error: got {type(kw[k]).__name__}, expect {typ}")

    # 6) 任务版本握手（可选但推荐）
    tv = kw.get(task_ver_field)
    ver_meta = {"task_ver": tv, "expected_ver": expected_ver, "min_accepted": MIN_ACCEPTED_TASK_VER}
    try:
        if tv is not None:
            tv = int(tv)
            kw[task_ver_field] = tv
            if tv < MIN_ACCEPTED_TASK_VER:
                raise ValueError(f"[{task_name}] task_ver {tv} < min accepted {MIN_ACCEPTED_TASK_VER}")
            if expected_ver is not None and tv != expected_ver and PARAM_STRICT != "ignore":
                msg = f"[{task_name}] task_ver mismatch: got {tv}, expect {expected_ver}"
                logger.warning(msg)
                if notify:
                    try:
                        notify({"type": "task_version_mismatch", "task": task_name, "got": tv, "expect": expected_ver})
                    except Exception:
                        pass
        else:
            if PARAM_STRICT == "error":
                raise ValueError(f"[{task_name}] missing '{task_ver_field}'")
            elif PARAM_STRICT == "warn":
                logger.warning(f"[{task_name}] missing '{task_ver_field}'")
    except Exception as e:
        # 版本错误直接抛出
        raise

    # 只回传声明字段（避免把未知字段继续往下传）
    filtered = {k: kw[k] for k in declared if k in kw}
    meta = {"unknown": unknown_keys, "aliases_used": used_aliases, "version": ver_meta}
    return filtered, meta


#
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from django.db import transaction, IntegrityError


# === 时间 / 数值工具 ===

def _to_aware(s: str):
    """ISO 字符串 -> timezone aware datetime"""
    from django.utils.dateparse import parse_datetime
    from django.utils.timezone import make_aware, is_naive

    dt = parse_datetime(s)
    if dt is None:
        raise ValueError(f"bad datetime iso: {s}")
    return make_aware(dt) if is_naive(dt) else dt


def _tz_offset_str(dt):
    """格式成 +09:00 这种字符串"""
    z = dt.strftime("%z")  # e.g. +0900
    return z[:-2] + ":" + z[-2:]


def _d4(x):
    """保留 2 位小数并四舍五入（你原来的实现）"""
    if x is None:
        return None
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


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


# ====== 统一的 FeatureSnapshot 安全 upsert（全局工具函数） ======
def safe_upsert_feature_snapshot(
    *,
    bucket,
    scope,
    name,
    version,
    value,
    is_final,
    max_retries: int = 2,
):
    """
    FeatureSnapshot 的 LWW（last-write-wins）安全 upsert。

    - value 会先走 _d4 再转 float
    - 如果并发插入撞唯一键，会自动重试并覆盖
    """
    from AppleStockChecker.models import FeatureSnapshot  # 局部 import 避免循环依赖

    value = float(_d4(value))
    for attempt in range(max_retries + 1):
        try:
            with transaction.atomic():
                # 先锁已有行，存在则直接覆盖（LWW）
                qs = (
                    FeatureSnapshot.objects
                    .select_for_update()
                    .filter(bucket=bucket, scope=scope, name=name, version=version)
                )
                obj = qs.first()
                if obj:
                    obj.value = value
                    obj.is_final = bool(is_final)  # ← 覆盖，而非 OR
                    obj.save(update_fields=["value", "is_final"])
                    return obj
                # 不存在则创建
                return FeatureSnapshot.objects.create(
                    bucket=bucket,
                    scope=scope,
                    name=name,
                    version=version,
                    value=value,
                    is_final=bool(is_final),
                )
        except IntegrityError:
            if attempt >= max_retries:
                raise
            # 并发插入撞唯一键，重试时会读到那行再覆盖
            continue





# 已有的:
# _to_aware, _tz_offset_str, _d4, _quantile, _pop_std, _filter_outliers_by_mean_band, safe_upsert_feature_snapshot
# 这里新增几类：SMA / EMA / WMA / fetch_prev_base


def _ema_from_series(series_old_to_new: List[float], alpha: float) -> float:
    """旧->新序列 + alpha，返回 EMA 值。"""
    if not series_old_to_new:
        return 0.0
    ema = float(series_old_to_new[0])
    for v in series_old_to_new[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return ema


def _sma(series_old_to_new: List[float], window: int) -> Optional[float]:
    """简单移动平均（旧->新序列），返回 None 表示无数据。"""
    if not series_old_to_new:
        return None
    w = max(1, int(window))
    s = series_old_to_new[-w:] if w < len(series_old_to_new) else series_old_to_new
    return sum(s) / float(len(s))


def _wma_linear(series_old_to_new: List[float], window: int) -> Optional[float]:
    """线性权重移动平均，越新的权重越大。"""
    if not series_old_to_new:
        return None
    w = max(1, int(window))
    s = series_old_to_new[-w:] if w < len(series_old_to_new) else series_old_to_new
    n = len(s)
    weights = list(range(1, n + 1))
    denom = float(sum(weights))
    return sum(v * w for v, w in zip(s, weights)) / denom if denom > 0 else None


def _fetch_prev_base(scope: str, base_name: str, base_version: str, limit: int, anchor_dt):
    """
    读取历史“基值”序列（不包含当前 x_t），按时间从新到旧取 limit 条：
    - overall:iphone:<id>  -> OverallBar.mean
    - cohort:<slug>        -> CohortBar.mean
    - 其他 scope           -> FeatureSnapshot(base_name)
    """
    from AppleStockChecker.models import OverallBar, CohortBar, FeatureSnapshot

    if scope.startswith("overall:iphone:"):
        ipid = int(scope.rsplit(":", 1)[-1])
        rows = (
            OverallBar.objects
            .filter(iphone_id=ipid, bucket__lt=anchor_dt)
            .order_by("-bucket")
            .values_list("mean", flat=True)[:limit]
        )
        return [float(v) for v in rows if v is not None]

    if scope.startswith("cohort:"):
        slug = scope.split(":", 1)[1]
        rows = (
            CohortBar.objects
            .filter(cohort__slug=slug, bucket__lt=anchor_dt)
            .order_by("-bucket")
            .values_list("mean", flat=True)[:limit]
        )
        return [float(v) for v in rows if v is not None]

    rows = (
        FeatureSnapshot.objects
        .filter(scope=scope, name=base_name, version=base_version, bucket__lt=anchor_dt)
        .order_by("-bucket")
        .values_list("value", flat=True)[:limit]
    )
    return [float(v) for v in rows if v is not None]

def _agg_overallbar(
    *,
    ts_iso: str,
    ts_dt,
    rows: List[Dict[str, Any]],
    use_window: bool,
    bucket_start,
    bucket_end,
    is_final_bar: bool,
    agg_ctx: Dict[str, Any],
    ob_has_iphone: bool,
):
    """
    1) OverallBar（全部店 × 各 iPhone）
    """
    from statistics import median
    from AppleStockChecker.models import PurchasingShopTimeAnalysis, OverallBar

    overallbar_debug = {"agg": agg_ctx, "iphones": [], "skipped": False}

    try:
        if not ob_has_iphone:
            overallbar_debug["skipped"] = True
        else:
            bucket_iphone_ids = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})
            if not bucket_iphone_ids:
                overallbar_debug["skipped"] = True

            for ipid in bucket_iphone_ids:
                if use_window:
                    # 窗口内：每店最后一条
                    qs_latest = (
                        PurchasingShopTimeAnalysis.objects
                        .filter(
                            iphone_id=ipid,
                            Timestamp_Time__gte=bucket_start,
                            Timestamp_Time__lt=bucket_end,
                            New_Product_Price__isnull=False,  # 只过滤空值，不用固定阈值
                        )
                        .order_by("shop_id", "-Timestamp_Time")
                        .distinct("shop_id")
                    )
                    prices_raw = [
                        float(p)
                        for p in qs_latest.values_list("New_Product_Price", flat=True)
                        if p is not None
                    ]
                    shop_cnt = qs_latest.values("shop_id").count()
                    ob_bucket = bucket_start
                    reference_time = bucket_end  # 使用桶结束时间作为参考时间
                else:
                    qs_latest = (
                        PurchasingShopTimeAnalysis.objects
                        .filter(
                            iphone_id=ipid,
                            Timestamp_Time=ts_dt,
                            New_Product_Price__isnull=False,  # 只过滤空值，不用固定阈值
                        )
                        .values("shop_id", "New_Product_Price")
                    )
                    prices_raw = [
                        float(r["New_Product_Price"])
                        for r in qs_latest
                        if r["New_Product_Price"] is not None
                    ]
                    shop_cnt = qs_latest.values("shop_id").distinct().count()
                    ob_bucket = ts_dt
                    reference_time = ts_dt

                if not prices_raw:
                    continue

                # 第一步：使用动态价格区间过滤明显错误的数据
                price_min, price_max = get_dynamic_price_range(ipid, reference_time)
                prices = [p for p in prices_raw if price_min <= p <= price_max]

                if not prices:
                    # 如果动态区间过滤后没有数据，记录警告并使用原始数据
                    logger.warning(
                        f"动态价格过滤后无数据: iphone_id={ipid}, "
                        f"原始样本数={len(prices_raw)}, "
                        f"区间=[{price_min:.0f}, {price_max:.0f}]"
                    )
                    prices = prices_raw

                # 第二步：使用统计方法过滤异常值（保留原有逻辑）
                vals_raw = [float(p) for p in prices]
                vals_filtered, _, _, _ = _filter_outliers_by_mean_band(vals_raw)
                if not vals_filtered:
                    continue

                vals = sorted(vals_filtered)
                m_mean = sum(vals) / len(vals)
                m_median = float(median(vals))
                m_std = _pop_std(vals)
                p10 = _quantile(vals, 0.10)
                p90 = _quantile(vals, 0.90)
                dispersion = (p90 - p10) if (p10 is not None and p90 is not None) else 0.0

                OverallBar.objects.update_or_create(
                    bucket=ob_bucket,
                    iphone_id=ipid,
                    defaults=dict(
                        mean=_d4(m_mean),
                        median=_d4(m_median),
                        std=_d4(m_std) if m_std is not None else None,
                        shop_count=shop_cnt,
                        dispersion=_d4(dispersion),
                        is_final=is_final_bar,
                    ),
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


def _agg_cohortbar(
    *,
    ts_iso: str,
    ob_bucket,
    is_final_bar: bool,
    agg_ctx: Dict[str, Any],
    ob_has_iphone: bool,
):
    """
    2) CohortBar（全部店 × 组合 iPhone）
    """
    from statistics import median
    from AppleStockChecker.models import Cohort, CohortMember, OverallBar, CohortBar

    cohort_debug = {"agg": agg_ctx, "bucket": ts_iso, "cohorts": []}

    try:
        if ob_has_iphone:
            cohorts = list(Cohort.objects.all())
            for coh in cohorts:
                members = list(
                    CohortMember.objects
                    .filter(cohort=coh)
                    .values("iphone_id", "weight")
                )
                if not members:
                    continue

                member_ids = [m["iphone_id"] for m in members]
                weight_map = {
                    m["iphone_id"]: float(m.get("weight") or 1.0)
                    for m in members
                }

                ob_rows = list(
                    OverallBar.objects
                    .filter(bucket=ob_bucket, iphone_id__in=member_ids)
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
                    bucket=ob_bucket,
                    cohort=coh,
                    defaults=dict(
                        mean=_d4(c_mean),
                        median=_d4(c_median),
                        std=_d4(c_std) if c_std is not None else None,
                        n_models=n_models,
                        shop_count_agg=shop_count_agg,
                        dispersion=_d4(c_disp),
                        is_final=is_final_bar,
                    ),
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


def _agg_feature_combos(
    *,
    ts_iso: str,
    ts_dt,
    rows: List[Dict[str, Any]],
    bucket_start,
    bucket_end,
    use_window: bool,
    anchor_bucket,
    agg_ctx: Dict[str, Any],
    is_final_bar: bool,
    writer,
):
    """
    3) 四类组合统计：写入 FeatureSnapshot（窗口去重 + 时效权）
    - Case1: shop × iphone
    - Case2: shopcohort × iphone
    - Case3: shop × cohort
    - Case4: shopcohort × cohort
    """
    from django.conf import settings
    from AppleStockChecker.models import (
        PurchasingShopTimeAnalysis,
        Cohort, CohortMember,
        ShopWeightProfile, ShopWeightItem,
    )

    try:
        # —— 预取本桶出现过的 shop/iphone —— #
        # Bug修复: 窗口模式时应该从窗口内的 PSTA 数据提取 shop/iphone，而不是从 rows（单分钟数据）
        # 原因: 边界分钟的 rows 可能为空，但窗口内仍有历史数据需要聚合
        if use_window:
            # 窗口模式: 从数据库查询窗口内所有的 shop_id 和 iphone_id
            shops_seen = sorted(set(
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    New_Product_Price__isnull=False,
                )
                .values_list('shop_id', flat=True)
                .distinct()
            ))
            iphones_seen = sorted(set(
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    New_Product_Price__isnull=False,
                )
                .values_list('iphone_id', flat=True)
                .distinct()
            ))
        else:
            # 单分钟模式: 从 rows 提取（原有逻辑）
            shops_seen = sorted({int(r.get("shop_id")) for r in rows if r.get("shop_id")})
            iphones_seen = sorted({int(r.get("iphone_id")) for r in rows if r.get("iphone_id")})

        # 日志输出：数据提取情况
        logger.info(
            f"  📊 [数据源] shops: {len(shops_seen)}个, iphones: {len(iphones_seen)}个 | "
            f"来源: {'窗口PSTA数据' if use_window else 'rows参数'}"
        )

        if use_window:
            base_qs = (
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time__gte=bucket_start,
                    Timestamp_Time__lt=bucket_end,
                    shop_id__in=shops_seen,
                    iphone_id__in=iphones_seen,
                    New_Product_Price__isnull=False,  # 只过滤空值，不用固定阈值
                )
                .order_by("shop_id", "iphone_id", "-Timestamp_Time")
                .distinct("shop_id", "iphone_id")
                .values("shop_id", "iphone_id", "New_Product_Price", "Timestamp_Time")
            )
            reference_time = bucket_end
        else:
            base_qs = (
                PurchasingShopTimeAnalysis.objects
                .filter(
                    Timestamp_Time=ts_dt,
                    shop_id__in=shops_seen,
                    iphone_id__in=iphones_seen,
                    New_Product_Price__isnull=False,  # 只过滤空值，不用固定阈值
                )
                .values("shop_id", "iphone_id", "New_Product_Price", "Timestamp_Time")
            )
            reference_time = ts_dt

        # 为每个 iphone_id 计算动态价格区间
        price_ranges = {}
        for iphone_id in iphones_seen:
            price_ranges[iphone_id] = get_dynamic_price_range(iphone_id, reference_time)

        # (shop, iphone) -> (last_price, last_ts)
        data_by_si: Dict[tuple, tuple] = {}
        filtered_count = 0
        total_count = 0
        for rec in base_qs:
            total_count += 1
            p = rec.get("New_Product_Price")
            t = rec.get("Timestamp_Time")
            if p is None:
                continue
            s = int(rec["shop_id"])
            i = int(rec["iphone_id"])

            # 使用动态价格区间过滤
            price_min, price_max = price_ranges.get(i, (PRICE_FALLBACK_MIN, PRICE_FALLBACK_MAX))
            if not (price_min <= float(p) <= price_max):
                filtered_count += 1
                continue

            data_by_si[(s, i)] = (float(p), t)

        if filtered_count > 0:
            logger.info(
                f"特征计算: 动态价格过滤，总记录={total_count}, 过滤={filtered_count}, "
                f"保留={total_count - filtered_count}"
            )

        # —— 统计工具 —— #
        def _stats(values):
            """返回 (mean, median, std, dispersion, count)，自动按平均值过滤异常值。"""
            if not values:
                return None
            vals_raw = [float(v) for v in values]
            vals_filtered, _, _, _ = _filter_outliers_by_mean_band(vals_raw)
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

            writer.write(scope, "mean", m)
            writer.write(scope, "median", med)
            writer.write(scope, "std", st)
            writer.write(scope, "dispersion", disp)
            writer.write(scope, "count", float(n))

            combo_debug["case1_shop_iphone"] += 1
            if len(combo_debug["samples"]) < 5:
                combo_debug["samples"].append({"case": 1, "scope": scope, "mean": round(m, 4)})

        # 预取店铺组合（ShopWeightProfile）
        profiles = list(ShopWeightProfile.objects.all())
        prof_items = {
            prof.id: {
                it["shop_id"]: float(it.get("weight") or 1.0)
                for it in ShopWeightItem.objects.filter(profile=prof).values("shop_id", "weight")
            }
            for prof in profiles
        }
        has_shop_profile = any(bool(prof_items.get(p.id)) for p in profiles)

        # 预取机型组合（Cohort）
        cohorts = list(Cohort.objects.all())
        cmembers = {
            coh.id: {
                m["iphone_id"]: float(m.get("weight") or 1.0)
                for m in CohortMember.objects.filter(cohort=coh).values("iphone_id", "weight")
            }
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
                    wnum = wden = 0.0
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
                        wnum += w * v
                        wden += w

                    if not vals:
                        continue

                    m_unw, med, st, disp, n = _stats(vals)
                    mean_w = (wnum / wden) if wden > 0 else m_unw
                    scope = f"shopcohort:{prof.slug}|iphone:{iid}"

                    writer.write(scope, "mean", mean_w)
                    writer.write(scope, "median", med)
                    writer.write(scope, "std", st)
                    writer.write(scope, "dispersion", disp)
                    writer.write(scope, "count", float(n))

                    combo_debug["case2_shopcohort_iphone"] += 1
                    if len(combo_debug["samples"]) < 5:
                        combo_debug["samples"].append({
                            "case": 2,
                            "scope": scope,
                            "n": n,
                            "mean_w": round(mean_w, 4),
                            "age_p50": (round(_quantile(sorted(ages), 0.5), 2) if ages else None),
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
                wnum = wden = 0.0
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
                    wnum += w * v
                    wden += w

                if not vals:
                    continue

                m_unw, med, st, disp, n = _stats(vals)
                mean_w = (wnum / wden) if wden > 0 else m_unw
                scope = f"shop:{sid}|cohort:{coh.slug}"

                writer.write(scope, "mean", mean_w)
                writer.write(scope, "median", med)
                writer.write(scope, "std", st)
                writer.write(scope, "dispersion", disp)
                writer.write(scope, "count", float(n))

                combo_debug["case3_shop_cohortiphone"] += 1
                if len(combo_debug["samples"]) < 5:
                    combo_debug["samples"].append({
                        "case": 3,
                        "scope": scope,
                        "n": n,
                        "mean_w": round(mean_w, 4),
                        "age_p50": (round(_quantile(sorted(ages), 0.5), 2) if ages else None),
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
                    wnum = wden = 0.0
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
                            wnum += w * v
                            wden += w

                    if not vals:
                        continue

                    m_unw, med, st, disp, n = _stats(vals)
                    mean_w = (wnum / wden) if wden > 0 else m_unw
                    scope = f"shopcohort:{prof.slug}|cohort:{coh.slug}"

                    writer.write(scope, "mean", mean_w)
                    writer.write(scope, "median", med)
                    writer.write(scope, "std", st)
                    writer.write(scope, "dispersion", disp)
                    writer.write(scope, "count", float(n))

                    combo_debug["case4_shopcohort_cohortiphone"] += 1
                    if len(combo_debug["samples"]) < 5:
                        combo_debug["samples"].append({
                            "case": 4,
                            "scope": scope,
                            "n": n,
                            "mean_w": round(mean_w, 4),
                            "age_p50": (round(_quantile(sorted(ages), 0.5), 2) if ages else None),
                        })
        else:
            combo_debug["skipped"].append("case4: no ShopWeightProfile defined")

        # 汇总日志输出
        total_features = (
            combo_debug["case1_shop_iphone"] +
            combo_debug["case2_shopcohort_iphone"] +
            combo_debug["case3_shop_cohortiphone"] +
            combo_debug["case4_shopcohort_cohortiphone"]
        )
        logger.info(
            f"  ✍️  [特征写入] "
            f"Case1(shop×iphone): {combo_debug['case1_shop_iphone']}, "
            f"Case2(shopcohort×iphone): {combo_debug['case2_shopcohort_iphone']}, "
            f"Case3(shop×cohort): {combo_debug['case3_shop_cohortiphone']}, "
            f"Case4(shopcohort×cohort): {combo_debug['case4_shopcohort_cohortiphone']} | "
            f"总计: {total_features} 个组合"
        )

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

def _agg_time_series_features(
    *,
    ts_iso: str,
    anchor_bucket,
    ob_bucket,
    is_final_bar: bool,
    ob_has_iphone: bool,
    writer,
) -> Dict[str, float]:
    """
    4) 时间序列派生指标：SMA / WMA / EMA（按 FeatureSpec）

    返回:
        base_now: scope -> 当前 x_t 基值（给 Bollinger 复用）
    """
    from django.apps import apps
    from typing import Dict
    from AppleStockChecker.models import FeatureSnapshot, OverallBar, CohortBar

    FeatureSpec = apps.get_model("AppleStockChecker", "FeatureSpec")

    timefeat_debug = {"bucket": ts_iso, "computed": 0, "skipped": [], "samples": []}
    base_now: Dict[str, float] = {}

    try:
        # 取出所有激活的时序类指标
        specs = list(
            FeatureSpec.objects
            .filter(active=True, family__in=["sma", "ema", "wma"])
            .values("slug", "family", "base_name", "params", "version")
        )

        # —— 收集"当前锚点"的基值 x_t：scope -> x_t —— #
        # 4.a 四类组合（FeatureSnapshot.mean@anchor_bucket）
        for row in (
            FeatureSnapshot.objects
            .filter(bucket=anchor_bucket, name="mean")
            .values("scope", "value")
        ):
            if row["value"] is not None:
                base_now[row["scope"]] = float(row["value"])

        # ===== 已禁用：从 OverallBar/CohortBar 收集基值 =====
        # 原因：已停用 OverallBar/CohortBar 计算
        # 如果需要 scope="overall:iphone:*" 或 "cohort:*" 的时间序列指标，
        # 请先恢复 _run_aggregation 中的 OverallBar/CohortBar 计算

        # # 4.b OverallBar.mean -> overall:iphone:<id>（@ob_bucket）
        # if ob_has_iphone:
        #     for row in (
        #         OverallBar.objects
        #         .filter(bucket=ob_bucket)
        #         .values("iphone_id", "mean")
        #     ):
        #         if row["mean"] is not None:
        #             base_now[f"overall:iphone:{row['iphone_id']}"] = float(row["mean"])

        # # 4.c CohortBar.mean -> cohort:<slug>（@ob_bucket）
        # for row in (
        #     CohortBar.objects
        #     .filter(bucket=ob_bucket)
        #     .select_related("cohort")
        #     .values("cohort__slug", "mean")
        # ):
        #     if row["mean"] is not None and row["cohort__slug"]:
        #         base_now[f"cohort:{row['cohort__slug']}"] = float(row["mean"])

        # —— 工具：回写派生值（只给 SMA 用，EMA/WMA 走 writer） —— #
        def upsert_feat(scope: str, name: str, version: str, value: float):
            safe_upsert_feature_snapshot(
                bucket=anchor_bucket,
                scope=scope,
                name=name,
                version=version,
                value=value,
                is_final=is_final_bar,
            )

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

            def _alpha_from_params(p: dict) -> float:
                if p is None:
                    p = {}
                if p.get("alpha") is not None:
                    a = float(p["alpha"])
                    return max(0.0, min(1.0, a))
                if p.get("window") is not None:
                    W_ = max(1, int(p["window"]))
                    return 2.0 / (W_ + 1.0)
                if p.get("half_life") is not None:
                    hl = float(p["half_life"])
                    return 1.0 - 0.5 ** (1.0 / max(hl, 1e-9))
                return 2.0 / (15.0 + 1.0)  # 默认

            for scope, x_t in base_now.items():
                # 拉取历史基值（新->旧），再转为旧->新，并在末尾追加当前 x_t
                prev_vals = _fetch_prev_base(scope, base_name, base_version, W - 1, anchor_bucket)
                series_old_to_new = list(reversed(prev_vals)) + [float(x_t)]

                # 样本数校验
                if len(series_old_to_new) < max(1, min_count):
                    timefeat_debug["skipped"].append(
                        f"{family}:{spec_slug}@{scope}:insufficient({len(series_old_to_new)}<{min_count})"
                    )
                    continue

                try:
                    if family == "ema":
                        alpha = _alpha_from_params(params)
                        val = _ema_from_series(series_old_to_new, alpha)
                        writer.write(scope, "ema", val, version=spec_slug)
                        timefeat_debug["computed"] += 1

                    elif family in ("wma", "wma_linear"):
                        if weights_mode == "linear":
                            val = _wma_linear(series_old_to_new, W)
                        else:
                            # 其它权重模式暂未实现 -> 后备为 SMA
                            val = _sma(series_old_to_new, W)
                        if val is None:
                            timefeat_debug["skipped"].append(
                                f"wma:{spec_slug}@{scope}:no_series"
                            )
                            continue

                        writer.write(scope, "wma", val, version=spec_slug)
                        timefeat_debug["computed"] += 1

                    elif family == "sma":
                        val = _sma(series_old_to_new, W)
                        if val is None:
                            timefeat_debug["skipped"].append(
                                f"sma:{spec_slug}@{scope}:no_series"
                            )
                            continue

                        upsert_feat(scope, "sma", spec_slug, val)
                        timefeat_debug["computed"] += 1

                    if len(timefeat_debug["samples"]) < 6:
                        timefeat_debug["samples"].append({
                            "scope": scope,
                            "family": family,
                            "spec": spec_slug,
                            "W": W,
                            "x": round(float(x_t), 2),
                            "y": round(float(val), 2),
                        })
                except Exception as _e:
                    timefeat_debug["skipped"].append(
                        f"{family}:{spec_slug}@{scope}:{repr(_e)}"
                    )

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

    return base_now


def _agg_bollinger_bands(
    *,
    ts_iso: str,
    anchor_bucket,
    is_final_bar: bool,
    base_now: Dict[str, float],
    writer,
):
    """
    5) Bollinger Bands（按 FeatureSpec，支持 center_mode="sma"/"ema"/"sma60"...）
    """
    from django.apps import apps

    FeatureSpec = apps.get_model("AppleStockChecker", "FeatureSpec")

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
            def _parse_center_mode(params: dict, default_W: int):
                cm = str(params.get("center_mode", "sma")).lower()
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
                    prev_vals = _fetch_prev_base(scope, base_name, base_version, W - 1, anchor_bucket)
                    series_old_to_new = list(reversed(prev_vals)) + [float(x_t)]
                    if len(series_old_to_new) < max(1, min_periods):
                        boll_debug["skipped"].append(
                            f"{spec_slug}@{scope}:insufficient({len(series_old_to_new)}<{min_periods})"
                        )
                        continue

                    # 中轨
                    if center_mode == "ema":
                        alpha = 2.0 / (center_W + 1.0)
                        mid = _ema_from_series(series_old_to_new, alpha)
                    else:
                        mid = _sma(series_old_to_new, center_W)

                    std = _pop_std(series_old_to_new) or 0.0
                    up = mid + k * std
                    low = mid - k * std
                    width = up - low

                    rows = [
                        FeatureRecord(
                            bucket=anchor_bucket,
                            scope=scope,
                            name="boll_mid",
                            version=spec_slug,
                            value=mid,
                            is_final=is_final_bar,
                        ),
                        FeatureRecord(
                            bucket=anchor_bucket,
                            scope=scope,
                            name="boll_up",
                            version=spec_slug,
                            value=up,
                            is_final=is_final_bar,
                        ),
                        FeatureRecord(
                            bucket=anchor_bucket,
                            scope=scope,
                            name="boll_low",
                            version=spec_slug,
                            value=low,
                            is_final=is_final_bar,
                        ),
                        FeatureRecord(
                            bucket=anchor_bucket,
                            scope=scope,
                            name="boll_width",
                            version=spec_slug,
                            value=width,
                            is_final=is_final_bar,
                        ),
                    ]
                    writer.write_many(rows)

                    boll_debug["computed"] += 1
                    if len(boll_debug["samples"]) < 6:
                        boll_debug["samples"].append({
                            "scope": scope,
                            "spec": spec_slug,
                            "W": W,
                            "center": f"{center_mode}{center_W}",
                            "mid": round(mid, 2),
                            "up": round(up, 2),
                            "low": round(low, 2),
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


def _agg_market_log_premium(
    *,
    ts_iso: str,
    anchor_bucket,
    is_final_bar: bool,
    writer,
):
    """
    6) 市场 log 溢价（Market Log Premium）
    公式: b_{k,t}^{wma_{tm}} = log(Q̄^{wma_{tm}}_{k,t} / P^{official}_k)

    其中：
    - Q̄^{wma_{tm}}_{k,t}: FeatureSnapshot 中 scope="shopcohort:full_store|iphone:{k}", name="wma", version="wma{tm}m" 的值
    - P^{official}_k: iPhone ID=k 的官方发布价格（从 settings.IPHONE_OFFICIAL_PRICES 获取）
    - 结果存储为：name="logb", version=对应的 wma 版本（如 "wma120m"）
    """
    import re
    import math
    from django.apps import apps
    from django.conf import settings

    FeatureSnapshot = apps.get_model("AppleStockChecker", "FeatureSnapshot")

    logb_debug = {"bucket": ts_iso, "computed": 0, "skipped": [], "samples": []}

    try:
        # 获取官方价格配置
        official_prices = getattr(settings, "IPHONE_OFFICIAL_PRICES", {})
        if not official_prices:
            logb_debug["skipped"].append("no_official_prices_in_settings")
            try:
                notify_progress_all(data={
                    "type": "feature_logb_warning",
                    "bucket": ts_iso,
                    "message": "IPHONE_OFFICIAL_PRICES not configured in settings",
                })
            except Exception:
                pass
            return

        # 查询当前 bucket 中所有 name="wma" 的 FeatureSnapshot 记录
        # 只关注 scope 包含 "shopcohort:full_store" 的记录
        wma_records = list(
            FeatureSnapshot.objects
            .filter(
                bucket=anchor_bucket,
                name="wma",
                scope__startswith="shopcohort:full_store|iphone:"
            )
            .values("scope", "version", "value")
        )

        if not wma_records:
            logb_debug["skipped"].append("no_wma_records_for_full_store")
        else:
            # 正则匹配 version 格式：wma[0-9]+m
            wma_version_pattern = re.compile(r"^wma\d+m$")

            for rec in wma_records:
                scope = rec["scope"]
                version = rec["version"]
                wma_value = rec["value"]

                # 检查 version 格式
                if not wma_version_pattern.match(version):
                    logb_debug["skipped"].append(f"{scope}@{version}:invalid_version_format")
                    continue

                # 从 scope 中提取 iPhone ID
                # scope 格式: "shopcohort:full_store|iphone:{iphone_id}"
                try:
                    iphone_id = int(scope.split("|iphone:")[-1])
                except (ValueError, IndexError):
                    logb_debug["skipped"].append(f"{scope}@{version}:invalid_scope_format")
                    continue

                # 获取官方价格
                official_price = official_prices.get(iphone_id)
                if official_price is None:
                    logb_debug["skipped"].append(f"{scope}@{version}:no_official_price_for_iphone_{iphone_id}")
                    continue

                if official_price <= 0:
                    logb_debug["skipped"].append(f"{scope}@{version}:invalid_official_price_{official_price}")
                    continue

                # 检查 wma_value 有效性
                if wma_value is None or wma_value <= 0:
                    logb_debug["skipped"].append(f"{scope}@{version}:invalid_wma_value_{wma_value}")
                    continue

                # 计算 log 溢价：log(wma_value / official_price)
                try:
                    logb_value = math.log(wma_value / official_price)
                except (ValueError, ZeroDivisionError) as e:
                    logb_debug["skipped"].append(f"{scope}@{version}:math_error_{repr(e)}")
                    continue

                # 写入新的 FeatureSnapshot
                logb_record = FeatureRecord(
                    bucket=anchor_bucket,
                    scope=scope,
                    name="logb",
                    version=version,  # 使用与 wma 相同的 version
                    value=logb_value,
                    is_final=is_final_bar,
                )
                writer.write_many([logb_record])

                logb_debug["computed"] += 1
                if len(logb_debug["samples"]) < 6:
                    logb_debug["samples"].append({
                        "scope": scope,
                        "version": version,
                        "iphone_id": iphone_id,
                        "wma": round(wma_value, 2),
                        "official": official_price,
                        "logb": round(logb_value, 4),
                    })

        # 发送进度通知
        try:
            notify_progress_all(data={
                "type": "feature_logb_update",
                "bucket": ts_iso,
                "summary": {k: v for k, v in logb_debug.items() if k != "samples"},
                "samples": logb_debug["samples"],
            })
        except Exception:
            pass

    except Exception as e:
        try:
            notify_progress_all(data={
                "type": "feature_logb_error",
                "bucket": ts_iso,
                "error": repr(e),
            })
        except Exception:
            pass


# === 可调参数（根据你们前后端链路容量调节） ===
MAX_BUCKET_ERROR_SAMPLES = 50  # 单桶保留的 error 明细条数上限
MAX_BUCKET_CHART_POINTS = 3000  # 单桶打包给回调聚合用的 chart point 上限
MAX_PUSH_POINTS = 60000  # 本次广播给前端的 point 总上限（超过则裁剪到最近 N 条）

# 废弃：固定的价格阈值（保留用于后备）
PRICE_MIN = 100000
PRICE_MAX = 350000

# 动态价格区间配置
PRICE_LOOKBACK_MINUTES = 30  # 向前查询多少分钟的数据作为参考
PRICE_TOLERANCE_RATIO = 0.10  # 容差比例：±50%
PRICE_MIN_SAMPLES = 3  # 计算参考价格所需的最少样本数
PRICE_FALLBACK_MIN = 100000  # 数据不足时的后备最小值
PRICE_FALLBACK_MAX = 350000  # 数据不足时的后备最大值


def get_dynamic_price_range(
    iphone_id: int,
    reference_time,
    lookback_minutes: int = PRICE_LOOKBACK_MINUTES,
    tolerance_ratio: float = PRICE_TOLERANCE_RATIO,
    min_samples: int = PRICE_MIN_SAMPLES,
) -> tuple[float, float]:
    """
    根据指定 iPhone 型号在参考时间点前 N 分钟内的历史价格，
    动态计算该型号的合理价格区间。

    逻辑：
    1. 查询该 iphone_id 在 [reference_time - lookback_minutes, reference_time) 时间窗口内
       所有不同店铺的最新价格记录
    2. 计算这些价格的平均值作为参考价格
    3. 基于参考价格和容差比例，计算价格区间：
       - price_min = reference_price * (1 - tolerance_ratio)
       - price_max = reference_price * (1 + tolerance_ratio)
    4. 如果样本数不足，返回后备的固定区间

    参数：
        iphone_id: iPhone 型号 ID
        reference_time: 参考时间点（datetime 对象）
        lookback_minutes: 向前查询的时间窗口（分钟）
        tolerance_ratio: 价格容差比例（0.5 表示 ±50%）
        min_samples: 计算参考价格所需的最少样本数

    返回：
        (price_min, price_max): 动态计算的价格区间
    """
    from django.db.models import Avg, Count
    from AppleStockChecker.models import PurchasingShopTimeAnalysis

    # 计算查询时间窗口
    start_time = reference_time - timedelta(minutes=lookback_minutes)

    # 查询该时间窗口内不同店铺的最新价格（每个店铺取最新一条）
    # 使用子查询获取每个店铺的最新记录
    subquery = (
        PurchasingShopTimeAnalysis.objects
        .filter(
            iphone_id=iphone_id,
            Timestamp_Time__gte=start_time,
            Timestamp_Time__lt=reference_time,
            New_Product_Price__isnull=False,
        )
        .order_by("shop_id", "-Timestamp_Time")
        .distinct("shop_id")
        .values_list("New_Product_Price", flat=True)
    )

    prices = list(subquery)

    # 如果样本数不足，返回后备区间
    if len(prices) < min_samples:
        # logger.warning(
        #     f"动态价格区间: iphone_id={iphone_id}, 样本数不足({len(prices)}/{min_samples}), "
        #     f"使用后备区间 [{PRICE_FALLBACK_MIN}, {PRICE_FALLBACK_MAX}]"
        # )
        return PRICE_FALLBACK_MIN, PRICE_FALLBACK_MAX

    # 计算平均价格作为参考价格
    reference_price = sum(prices) / len(prices)

    # 计算动态区间
    price_min = reference_price * (1 - tolerance_ratio)
    price_max = reference_price * (1 + tolerance_ratio)

    # 确保区间不会过小（至少保留一定的范围）
    min_range = reference_price * 0.2  # 至少 ±10%
    if price_max - price_min < min_range:
        price_min = reference_price * 0.9
        price_max = reference_price * 1.1

    # logger.info(
    #     f"动态价格区间: iphone_id={iphone_id}, 样本数={len(prices)}, "
    #     f"参考价格={reference_price:.0f}, 区间=[{price_min:.0f}, {price_max:.0f}]"
    # )

    return price_min, price_max


def is_price_valid(
    price: float,
    iphone_id: int,
    reference_time,
    use_dynamic_range: bool = True,
) -> bool:
    """
    检查价格是否在合理区间内。

    参数：
        price: 待检查的价格
        iphone_id: iPhone 型号 ID
        reference_time: 参考时间点
        use_dynamic_range: 是否使用动态价格区间（默认 True）

    返回：
        bool: 价格是否有效
    """
    if use_dynamic_range:
        price_min, price_max = get_dynamic_price_range(iphone_id, reference_time)
    else:
        # 使用固定区间
        price_min, price_max = PRICE_MIN, PRICE_MAX

    return price_min <= price <= price_max


# -----------------------------------------------------
# --------------------------------------------------------
# -----------------------------------------------------------
# ---------------------------------------------------------------
# -----------------------------------------------------------
# --------------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------
# 子任务：处理"分钟桶"并返回桶级摘要 + 图表增量
# -----------------------------------------------
TASK_VER_PSTA = 3


def _process_minute_rows(*, ts_iso: str, ts_dt, rows, job_id: str):
    """
    模块一：分钟对齐数据写入 PurchasingShopTimeAnalysis，并收集 chart_points。

    返回:
        ok, failed, err_counter, errors, chart_points
    """
    from django.utils import timezone  # 如果不需要可以去掉
    from django.core.exceptions import ObjectDoesNotExist, ValidationError
    from AppleStockChecker.models import (
        PurchasingShopTimeAnalysis, SecondHandShop, Iphone,
    )

    ok = 0
    failed = 0
    errors = []
    err_counter = Counter()

    ts_tz = _tz_offset_str(ts_dt)
    orig_tz = "+09:00"
    chart_points = []

    # 预计算所有 iphone_id 的动态价格区间（优化性能，避免重复查询）
    unique_iphone_ids = {r.get("iphone_id") for r in rows if r.get("iphone_id")}
    price_ranges_cache = {}
    for iphone_id in unique_iphone_ids:
        try:
            price_ranges_cache[iphone_id] = get_dynamic_price_range(iphone_id, ts_dt)
        except Exception as e:
            logger.warning(f"计算动态价格区间失败: iphone_id={iphone_id}, error={e}, 使用后备区间")
            price_ranges_cache[iphone_id] = (PRICE_FALLBACK_MIN, PRICE_FALLBACK_MAX)

    for r in rows:
        try:
            # 轻量校验
            shop_id = r.get("shop_id")
            iphone_id = r.get("iphone_id")
            if not shop_id or not iphone_id:
                raise ValueError("missing shop_id/iphone_id")

            # 外键存在性
            SecondHandShop.objects.only("id").get(pk=shop_id)
            Iphone.objects.only("id").get(pk=iphone_id)

            rec_dt = _to_aware(r.get("recorded_at"))
            new_price = r.get("price_new") or r.get("New_Product_Price")
            if new_price is None:
                raise ValueError("missing New_Product_Price")

            try:
                price = int(new_price)
            except (TypeError, ValueError):
                raise ValueError(f"bad New_Product_Price: {new_price!r}")

            # 使用动态价格区间过滤（替换固定阈值）
            price_min, price_max = price_ranges_cache.get(iphone_id, (PRICE_FALLBACK_MIN, PRICE_FALLBACK_MAX))
            if not (price_min <= price <= price_max):
                logger.debug(
                    f"价格超出动态区间: shop_id={shop_id}, iphone_id={iphone_id}, "
                    f"price={price}, 区间=[{price_min:.0f}, {price_max:.0f}]"
                )
                continue

            align_diff = int((rec_dt - ts_dt).total_seconds())

            from django.db import transaction
            from django.db.utils import IntegrityError as DjangoIntegrityError

            with transaction.atomic():
                inst = (
                    PurchasingShopTimeAnalysis.objects
                    .select_for_update()
                    .filter(
                        shop_id=shop_id,
                        iphone_id=iphone_id,
                        Timestamp_Time=ts_dt,
                    )
                    .first()
                )

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
                    },
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
                    },
                })

    return ok, failed, err_counter, errors, chart_points


def _run_aggregation(
    *,
    ts_iso: str,
    ts_dt,
    rows: List[Dict[str, Any]],
    agg_start_iso: Optional[str],
    agg_minutes: int,
):
    """
    聚合调度器：按顺序调用 6 个子步骤：
    1) OverallBar
    2) CohortBar
    3) FeatureSnapshot 四类组合
    4) 时间序列
    5) Bollinger Bands
    6) Market Log Premium（市场 log 溢价）
    """
    logger.info(
        f"🔄 [FeatureSnapshot 聚合] 进入聚合流程 | "

    )
    from django.utils import timezone
    from AppleStockChecker.models import OverallBar
    # FeatureWriter / FeatureRecord 已在模块顶部从 utils.timestamp_alignment_task 导入

    WATERMARK_MINUTES = 5
    now = timezone.now()
    is_final_bar = ts_dt <= (now - timezone.timedelta(minutes=WATERMARK_MINUTES))

    # --- 聚合窗口（1分钟：用 ts_dt；>1分钟：用 [bucket_start, bucket_end) 窗口） ---
    bucket_start = _to_aware(agg_start_iso) if (agg_minutes and agg_start_iso) else ts_dt
    bucket_end = bucket_start + timezone.timedelta(minutes=agg_minutes or 1)
    use_window = (agg_minutes or 1) > 1

    # ========== FeatureSnapshot 聚合开始 ==========
    logger.info(
        f"🔄 [FeatureSnapshot 聚合] 开始计算 | "
        f"时间点: {ts_iso} | "
        f"窗口: {bucket_start.isoformat()} → {bucket_end.isoformat()} | "
        f"聚合步长: {agg_minutes}分钟 | "
        f"模式: {'窗口' if use_window else '单分钟'}"
    )
    # =============================================

    # === 统一锚点：所有 FeatureSnapshot / 派生指标的 bucket，都用 anchor_bucket ===
    anchor_bucket = bucket_start if use_window else ts_dt
    ob_bucket = anchor_bucket  # OverallBar / CohortBar 也用同一锚点

    agg_ctx = {
        "do_agg": True,
        "agg_minutes": int(agg_minutes or 1),
        "bucket_start": bucket_start.isoformat(),
        "bucket_end": bucket_end.isoformat(),
    }

    # 写 feature 的统一 Writer（LWW）
    writer = FeatureWriter(
        bucket=anchor_bucket,
        default_version="v1",
        is_final=is_final_bar,
        escalate_is_final=False,  # LWW（后写覆盖前写）
    )

    # 自动探测 OverallBar 是否含 iphone 外键；若没有，跳过以免 unique(bucket) 冲突
    ob_has_iphone = any(
        getattr(f, "name", "") == "iphone"
        for f in OverallBar._meta.get_fields()
    )

    # ===== 已禁用：OverallBar 和 CohortBar 计算 =====
    # 原因：主要使用 FeatureSnapshot 四类组合，无需全店聚合统计
    # 如需恢复，取消下面的注释

    # # 1) OverallBar
    # _agg_overallbar(
    #     ts_iso=ts_iso,
    #     ts_dt=ts_dt,
    #     rows=rows,
    #     use_window=use_window,
    #     bucket_start=bucket_start,
    #     bucket_end=bucket_end,
    #     is_final_bar=is_final_bar,
    #     agg_ctx=agg_ctx,
    #     ob_has_iphone=ob_has_iphone,
    # )

    # # 2) CohortBar
    # _agg_cohortbar(
    #     ts_iso=ts_iso,
    #     ob_bucket=ob_bucket,
    #     is_final_bar=is_final_bar,
    #     agg_ctx=agg_ctx,
    #     ob_has_iphone=ob_has_iphone,
    # )

    # 3) FeatureSnapshot 四类组合
    _agg_feature_combos(
        ts_iso=ts_iso,
        ts_dt=ts_dt,
        rows=rows,
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        use_window=use_window,
        anchor_bucket=anchor_bucket,
        agg_ctx=agg_ctx,
        is_final_bar=is_final_bar,
        writer=writer,
    )

    # 查询刚生成的 FeatureSnapshot 数据量
    from AppleStockChecker.models import FeatureSnapshot
    feature_count = FeatureSnapshot.objects.filter(bucket=anchor_bucket).count()
    logger.info(
        f"✅ [FeatureSnapshot 聚合] 完成 | "
        f"时间点: {ts_iso} | "
        f"bucket: {anchor_bucket.isoformat()} | "
        f"生成记录数: {feature_count} 条"
    )

    # 4) 时间序列（返回 base_now 给 Bollinger 用）
    base_now = _agg_time_series_features(
        ts_iso=ts_iso,
        anchor_bucket=anchor_bucket,
        ob_bucket=ob_bucket,
        is_final_bar=is_final_bar,
        ob_has_iphone=ob_has_iphone,
        writer=writer,
    )

    # 5) Bollinger Bands
    _agg_bollinger_bands(
        ts_iso=ts_iso,
        anchor_bucket=anchor_bucket,
        is_final_bar=is_final_bar,
        base_now=base_now,
        writer=writer,
    )

    # 6) Market Log Premium
    _agg_market_log_premium(
        ts_iso=ts_iso,
        anchor_bucket=anchor_bucket,
        is_final_bar=is_final_bar,
        writer=writer,
    )



@shared_task(name="AppleStockChecker.tasks.psta_process_minute_bucket")
def psta_process_minute_bucket(
    *,
    ts_iso: str,
    rows: list[dict],
    job_id: str,
    agg_minutes: int = 1,  # 保留用于日志/调试
    task_ver: int | None = None,
    **_compat,  # 向后兼容：接受已废弃参数 do_agg, agg_start_iso
) -> dict:
    """
    子任务：参数守卫 + 写入分钟数据（v3起聚合移至 finalize）

    v3 变更：
    - 移除 do_agg 参数（聚合逻辑移至 psta_finalize_buckets）
    - 移除 agg_start_iso 参数
    - 保留 agg_minutes 用于日志/调试
    """
    # 检查废弃参数并记录警告
    if "do_agg" in _compat:
        logger.warning(f"[psta_process_minute_bucket] do_agg 参数已废弃(v3)，将被忽略")
    if "agg_start_iso" in _compat:
        logger.warning(f"[psta_process_minute_bucket] agg_start_iso 参数已废弃(v3)，将被忽略")

    # ---------- 参数守卫 ----------
    incoming = dict(
        ts_iso=ts_iso,
        rows=rows,
        job_id=job_id,
        agg_minutes=agg_minutes,
        task_ver=task_ver,
    )

    normalized, meta = guard_params(
        "psta_process_minute_bucket",
        incoming,
        required={"ts_iso": str, "rows": list, "job_id": str},
        optional={
            "agg_minutes": (int, str),
            "task_ver": (int, str, type(None)),
        },
        defaults={"agg_minutes": 1},
        coerce={"agg_minutes": to_int},
        task_ver_field="task_ver",
        expected_ver=TASK_VER_PSTA,
        notify=notify_progress_all,
    )

    # 用归一化后的值覆盖本地变量
    ts_iso = normalized["ts_iso"]
    rows = normalized["rows"] or []
    job_id = normalized["job_id"]
    agg_minutes = normalized.get("agg_minutes", 1)

    # ---------- 写入分钟对齐数据 ----------
    ts_dt = _to_aware(ts_iso)

    ok, failed, err_counter, errors, chart_points = _process_minute_rows(
        ts_iso=ts_iso,
        ts_dt=ts_dt,
        rows=rows,
        job_id=job_id,
    )

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

    # v3: 聚合逻辑已移至 psta_finalize_buckets，此处不再执行

    # ---------- 返回任务结果 ----------
    return {
        "ts_iso": ts_iso,
        "ok": ok,
        "failed": failed,
        "total": ok + failed,
        "error_hist": dict(err_counter),
        "errors": errors[:MAX_BUCKET_ERROR_SAMPLES],
        "chart_points": chart_points,
    }


# 辅助任务：用于 chain 模式下累积结果
# -----------------------------------------------


@shared_task(name="AppleStockChecker.tasks.psta_collect_result")
def psta_collect_result(prev_result, current_result=None):
    """
    用于 chain 模式下累积所有子任务的结果。

    Args:
        prev_result: 上一个任务传递的累积结果列表，或单个结果字典
        current_result: 当前任务的结果（用于首次调用）

    Returns:
        累积的结果列表
    """
    # 初始化累积列表
    if prev_result is None:
        accumulated = []
    elif isinstance(prev_result, list):
        accumulated = prev_result
    else:
        # 第一个结果是单个字典
        accumulated = [prev_result]

    # 添加当前结果
    if current_result is not None:
        accumulated.append(current_result)

    return accumulated


# -----------------------------------------------
# 独立聚合任务（v3新增）
# -----------------------------------------------


@shared_task(
    name="AppleStockChecker.tasks.psta_aggregate_features",
    queue="psta_aggregation",
    soft_time_limit=240,
    time_limit=360,
)
def psta_aggregate_features(
    *,
    ts_iso: str,
    job_id: str,
    agg_ctx: dict,
    task_ver: int | None = None,
    **_compat,
) -> dict:
    """
    独立聚合任务（v3新增）

    在所有分钟桶数据写入完成后执行统计聚合。
    由 psta_finalize_buckets 在边界时间点同步调用。

    参数:
        ts_iso: 时间戳 ISO 格式
        job_id: 任务 ID
        agg_ctx: 聚合上下文，包含:
            - agg_minutes: 聚合步长
            - agg_mode: 聚合模式 (off 禁用，其他 = 边界模式)
            - is_boundary: 是否为边界时间点
            - bucket_start: 聚合窗口起始
            - bucket_end: 聚合窗口结束
        task_ver: 任务版本号

    返回:
        聚合结果或错误信息
    """
    agg_result = {
        "ts_iso": ts_iso,
        "job_id": job_id,
        "aggregation_success": False,
        "error": None,
    }

    try:
        # 参数校验
        if not agg_ctx:
            raise ValueError("agg_ctx is required")

        agg_mode = (agg_ctx.get("agg_mode") or "boundary").lower()
        is_boundary = agg_ctx.get("is_boundary", False)
        agg_minutes = int(agg_ctx.get("agg_minutes", 15))
        agg_start_iso = agg_ctx.get("bucket_start")

        # 如果 agg_mode 为 off，不执行聚合
        if agg_mode == "off":
            logger.info(f"[psta_aggregate_features] agg_mode=off，跳过聚合 | ts_iso={ts_iso}")
            agg_result["aggregation_success"] = True
            agg_result["skipped"] = True
            agg_result["reason"] = "agg_mode=off"
            return agg_result

        # 如果不是边界时间点，不执行聚合
        if not is_boundary:
            logger.info(f"[psta_aggregate_features] 非边界时间点，跳过聚合 | ts_iso={ts_iso}")
            agg_result["aggregation_success"] = True
            agg_result["skipped"] = True
            agg_result["reason"] = "not_boundary"
            return agg_result

        logger.info(
            f"[psta_aggregate_features] 开始聚合 | "
            f"ts_iso={ts_iso} | agg_minutes={agg_minutes} | "
            f"bucket_start={agg_start_iso}"
        )

        # 执行聚合
        ts_dt = _to_aware(ts_iso)
        _run_aggregation(
            ts_iso=ts_iso,
            ts_dt=ts_dt,
            rows=[],  # v3: 聚合从数据库读取，不再依赖 rows 参数
            agg_start_iso=agg_start_iso,
            agg_minutes=agg_minutes,
        )

        agg_result["aggregation_success"] = True
        logger.info(f"[psta_aggregate_features] 聚合完成 | ts_iso={ts_iso}")

        # 广播聚合完成通知
        try:
            notify_progress_all(data={
                "type": "aggregation_complete",
                "ts_iso": ts_iso,
                "job_id": job_id,
                "agg_ctx": agg_ctx,
            })
        except Exception:
            pass

    except Exception as e:
        error_msg = f"聚合失败: {repr(e)}"
        logger.error(f"[psta_aggregate_features] {error_msg} | ts_iso={ts_iso}", exc_info=True)
        agg_result["error"] = error_msg

        # 广播聚合失败通知
        try:
            notify_progress_all(data={
                "type": "aggregation_failed",
                "ts_iso": ts_iso,
                "job_id": job_id,
                "error": error_msg,
            })
        except Exception:
            pass

    return agg_result


# -----------------------------------------------
# 回调：聚合所有分钟桶，广播最终"done + 图表增量"（v3: 触发独立聚合任务）
# -----------------------------------------------


@shared_task(
    name="AppleStockChecker.tasks.psta_finalize_buckets",
    queue="psta_finalize",
    soft_time_limit=240,
    time_limit=360,
)
def psta_finalize_buckets(
        results: List[Dict[str, Any]],
        job_id: str,
        ts_iso: str,
        agg_ctx: Optional[dict] = None,
        task_ver: Optional[int] = None,
        **_compat
) -> Dict[str, Any]:
    """
    回调任务：汇总所有分钟桶结果，广播通知，并触发聚合任务。

    v3 变更：
    - 添加独立队列 psta_finalize
    - 在边界时间点同步调用 psta_aggregate_features 执行聚合
    - 聚合失败不影响数据写入结果的返回
    """
    from collections import defaultdict, Counter
    incoming = dict(results=results, job_id=job_id, ts_iso=ts_iso,
                    agg_ctx=agg_ctx, task_ver=task_ver, **_compat)
    normalized, meta = guard_params(
        "psta_finalize_buckets",
        incoming,
        required={"results": list, "job_id": str, "ts_iso": str},
        optional={"agg_ctx": (dict, type(None)), "task_ver": (int, str, type(None))},
        defaults={"agg_ctx": None},
        task_ver_field="task_ver",
        expected_ver=TASK_VER_PSTA,
        notify=notify_progress_all,
    )
    results = normalized["results"]
    job_id = normalized["job_id"]
    ts_iso = normalized["ts_iso"]
    agg_ctx = normalized.get("agg_ctx")

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
    last_known = {}  # key -> dict(point)
    has_real_at_ts = {}  # key -> bool
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
        flat.sort(key=lambda x: x[0])  # 升序
        flat = flat[-MAX_PUSH_POINTS:]  # 保留最近 N 条

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
                    "id": None,  # 影子点不落库，无 id
                    "t": ts_iso,  # 影子点放在标的时间戳
                    "price": src["price"],  # 以最近真实点的价格填充
                    "recorded_at": src.get("recorded_at"),
                    "shadow": True,  # ✅ 标识影子点
                    "src_t": src["t"],  # 影子来源时间（便于前端 tooltip/样式）
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
                "shadow_points": shadow_points_added,  # 本次补的影子点数
                "clipped": clipped,
            }
        }
    }

    try:
        notify_progress_all(data=payload)
    except Exception:
        pass

    # ========== v3: 触发聚合任务 ==========
    agg_result = None
    if agg_ctx:
        agg_mode = (agg_ctx.get("agg_mode") or "boundary").lower()
        is_boundary = agg_ctx.get("is_boundary", False)

        # 仅在非 off 模式且为边界时间点时触发聚合
        if agg_mode != "off" and is_boundary:
            logger.info(
                f"[psta_finalize_buckets] 触发聚合任务 | "
                f"ts_iso={ts_iso} | is_boundary={is_boundary}"
            )
            try:
                # 同步调用聚合任务并等待结果
                agg_result = psta_aggregate_features.apply(
                    kwargs={
                        "ts_iso": ts_iso,
                        "job_id": job_id,
                        "agg_ctx": agg_ctx,
                        "task_ver": TASK_VER_PSTA,
                    }
                ).get(timeout=300)  # 5分钟超时

                if agg_result and not agg_result.get("aggregation_success"):
                    logger.error(
                        f"[psta_finalize_buckets] 聚合任务返回失败 | "
                        f"ts_iso={ts_iso} | error={agg_result.get('error')}"
                    )
            except Exception as e:
                logger.error(
                    f"[psta_finalize_buckets] 聚合任务执行异常 | "
                    f"ts_iso={ts_iso} | error={repr(e)}",
                    exc_info=True
                )
                agg_result = {
                    "aggregation_success": False,
                    "error": repr(e),
                }
        else:
            logger.info(
                f"[psta_finalize_buckets] 跳过聚合 | "
                f"agg_mode={agg_mode} | is_boundary={is_boundary}"
            )

    # 将聚合结果附加到 payload
    payload["aggregation"] = agg_result

    return payload


# -----------------------------------------------------
# --------------------------------------------------------
# -----------------------------------------------------------
# ---------------------------------------------------------------
# -----------------------------------------------------------
# --------------------------------------------------------
# -----------------------------------------------------
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
        query_window_minutes: int = 15,
        shop_ids: Optional[List[int]] = None,
        iphone_ids: Optional[List[int]] = None,
        max_items: Optional[int] = None,
        # 聚合控制
        agg_minutes: int = 15,  # 聚合步长
        agg_mode: str = "boundary",  # 'off' 禁用聚合，其他值统一为边界模式
        sequential: bool = False,  # 是否顺序执行子任务
        **_compat,  # 向后兼容：接受已废弃参数 chunk_size, force_agg
) -> Dict[str, Any]:
    """
    父任务：分发分钟桶子任务，汇总结果。

    v3 变更：
    - 移除 force_agg 参数（已废弃）
    - agg_mode 简化：off 禁用聚合，其他值统一为边界模式
    - 聚合逻辑移至 psta_finalize_buckets
    - 子任务不再传递 do_agg, agg_start_iso 参数
    """
    # 检查废弃参数
    if "force_agg" in _compat:
        logger.warning("[batch_generate_psta_same_ts] force_agg 参数已废弃(v3)，将被忽略")
    if "chunk_size" in _compat:
        logger.warning("[batch_generate_psta_same_ts] chunk_size 参数已废弃，将被忽略")

    task_job_id = job_id or self.request.id
    ts_iso = timestamp_iso or nearest_past_minute_iso()

    # v3: agg_mode 简化 - off 禁用，其他统一为边界模式
    MODE = (agg_mode or "boundary").lower()
    if MODE not in ("off", "boundary"):
        logger.info(f"[batch_generate_psta_same_ts] agg_mode='{agg_mode}' 将作为边界模式处理")
        MODE = "boundary"

    pack = (collect_items_for_psta(
        window_minutes=query_window_minutes,
        timestamp_iso=ts_iso,
        shop_ids=shop_ids,
        iphone_ids=iphone_ids,
        max_items=max_items,
    ) or [{}])[0]

    rows = pack.get("rows") or []
    bucket_minute_key: Dict[str, Dict[str, List[int]]] = pack.get("bucket_minute_key") or {}

    # 计算边界时间和 is_boundary 标志
    dt0 = _to_aware(ts_iso)
    step0 = _floor_to_step(dt0, int(agg_minutes))
    is_boundary = (dt0 == step0)

    # 构建聚合上下文（传递给 finalize）
    ctx = {
        "agg_minutes": int(agg_minutes),
        "agg_mode": MODE,
        "is_boundary": is_boundary,  # v3: 新增边界标志
        "bucket_start": step0.isoformat(),
        "bucket_end": (step0 + timezone.timedelta(minutes=int(agg_minutes))).isoformat(),
    }

    try:
        notify_progress_all(data={"type": "agg_ctx", "timestamp": ts_iso, "job_id": task_job_id, "ctx": ctx})
    except Exception:
        pass

    # 构建子任务（v3: 子任务仅负责写入数据，不做聚合）
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

        # v3: 只要有数据就下发子任务（聚合由 finalize 统一处理）
        if minute_rows:
            subtasks.append(
                psta_process_minute_bucket.s(
                    ts_iso=minute_iso,
                    rows=minute_rows,
                    job_id=task_job_id,
                    agg_minutes=int(agg_minutes),  # 保留用于日志
                    task_ver=TASK_VER_PSTA,
                )
            )

    try:
        notify_progress_all(
            data={"status": "running", "step": "dispatch_buckets", "buckets": len(subtasks), "timestamp": ts_iso,
                  "agg": ctx})
    except Exception:
        pass

    if not subtasks:
        empty = {"timestamp": ts_iso, "total_buckets": 0, "ok": 0, "failed": 0, "by_bucket": []}
        try:
            notify_progress_all(data={
                "status": "done", "progress": 100, "summary": empty,
                "chart_delta": {"job_id": task_job_id, "timestamp": ts_iso, "series_delta": [],
                                "meta": {"total_points": 0, "shadow_points": 0, "clipped": False}}
            })
        except Exception:
            pass
        return empty

    # 根据 sequential 参数选择执行方式
    if sequential:
        # 顺序执行模式：逐个执行子任务并收集结果
        results = []
        for i, subtask in enumerate(subtasks):
            try:
                # 同步调用子任务（阻塞等待完成）
                result = subtask.apply().get()
                results.append(result)

                # 可选：报告进度
                try:
                    notify_progress_all(
                        data={
                            "status": "running",
                            "step": f"processing_bucket_{i+1}",
                            "progress": int((i + 1) * 100 / len(subtasks)),
                            "current": i + 1,
                            "total": len(subtasks),
                            "timestamp": ts_iso,
                        }
                    )
                except Exception:
                    pass
            except Exception as e:
                # 记录错误但继续执行
                results.append({
                    "ok": 0,
                    "failed": 1,
                    "error": str(e),
                    "error_hist": {"sequential_execution_error": 1},
                })

        # 直接调用 finalize 处理结果
        final_result = psta_finalize_buckets(
            results=results,
            job_id=task_job_id,
            ts_iso=ts_iso,
            agg_ctx=ctx,
            task_ver=TASK_VER_PSTA,
        )
        return {
            "timestamp": ts_iso,
            "total_buckets": len(subtasks),
            "job_id": task_job_id,
            "sequential": True,
            "result": final_result,
        }
    else:
        # 并发执行模式（默认）：使用 chord
        callback = psta_finalize_buckets.s(job_id=task_job_id, ts_iso=ts_iso, agg_ctx=ctx,
                                           task_ver=TASK_VER_PSTA)  # 可把 ctx 传给回调（可选）
        chord_result = chord(subtasks)(callback)
        return {"timestamp": ts_iso, "total_buckets": len(subtasks), "job_id": task_job_id, "chord_id": chord_result.id}

# -----------------------------------------------------
# --------------------------------------------------------
# -----------------------------------------------------------
# ---------------------------------------------------------------
# -----------------------------------------------------------
# --------------------------------------------------------
# -----------------------------------------------------
