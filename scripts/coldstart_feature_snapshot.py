#!/usr/bin/env python
"""
FeatureSnapshot 冷启动初始化脚本

用途：
  当 FeatureSnapshot 表为空时，使用原始数据直接填充统计指标，
  为后续的时间序列计算提供基础数据。

策略：
  从给定时间戳开始，往前取 40 个边界时间点（15分钟间隔），
  对每个时间点，将窗口内的原始数据值直接作为统计指标填充。

使用方法：
  # 从最近的边界时间开始，往前填充 40 个时间点
  python scripts/coldstart_feature_snapshot.py --auto

  # 从指定时间开始，往前填充 40 个时间点
  python scripts/coldstart_feature_snapshot.py \
    --start "2025-10-04T01:00:00+00:00" \
    --count 40

  # 自定义填充数量
  python scripts/coldstart_feature_snapshot.py \
    --start "2025-10-04T01:00:00+00:00" \
    --count 100

  # 只检查状态，不执行
  python scripts/coldstart_feature_snapshot.py --check-only

注意事项：
  1. 生成的数据是"简化"的统计指标（原始值直接填充）
  2. 适合冷启动场景，为时间序列提供初始数据
  3. 如果窗口内有多条数据，取最后一条（最新）
  4. 生成的数据标记为 is_final=True
"""

import os
import sys
import django
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")
django.setup()

from datetime import datetime, timedelta, timezone as tz
from typing import List, Dict, Any
from AppleStockChecker.models import (
    FeatureSnapshot,
    PurchasingShopTimeAnalysis,
    Cohort,
    CohortMember,
    ShopWeightProfile,
    ShopWeightItem,
)
from decimal import Decimal

UTC = tz.utc


def find_latest_boundary(agg_minutes=15):
    """查找最近的边界时间"""
    now = datetime.now(UTC)

    # 向下取整到边界
    minute = (now.minute // agg_minutes) * agg_minutes
    boundary = now.replace(minute=minute, second=0, microsecond=0)

    # 确保是过去的时间（至少5分钟前）
    while (now - boundary).total_seconds() < 300:
        boundary -= timedelta(minutes=agg_minutes)

    return boundary


def generate_coldstart_timestamps(start_dt, count, interval_minutes=15):
    """生成冷启动时间戳序列（往前推）"""
    timestamps = []
    current = start_dt

    for _ in range(count):
        timestamps.append(current)
        current -= timedelta(minutes=interval_minutes)

    return timestamps


def coldstart_single_bucket(bucket_dt, agg_minutes=15):
    """
    为单个 bucket 填充冷启动数据

    策略：
    1. 查询窗口内的原始数据（去重：每个 shop×iphone 取最后一条）
    2. 将原始值直接填充为统计指标（mean=value, median=value, count=1）
    3. 生成 4 种组合的 FeatureSnapshot
    """
    bucket_start = bucket_dt
    bucket_end = bucket_dt + timedelta(minutes=agg_minutes)

    print(f"\n处理 bucket: {bucket_dt.isoformat()}")
    print(f"  窗口: {bucket_start.isoformat()} → {bucket_end.isoformat()}")

    # 检查是否已有数据
    existing_count = FeatureSnapshot.objects.filter(bucket=bucket_dt).count()
    if existing_count > 0:
        print(f"  ⏭️  已有 {existing_count} 条记录，跳过")
        return {"bucket": bucket_dt.isoformat(), "skipped": True, "existing": existing_count}

    # 查询窗口内的原始数据（去重：每个 shop×iphone 取最后一条）
    base_qs = (
        PurchasingShopTimeAnalysis.objects
        .filter(
            Timestamp_Time__gte=bucket_start,
            Timestamp_Time__lt=bucket_end,
            New_Product_Price__isnull=False,
        )
        .order_by("shop_id", "iphone_id", "-Timestamp_Time")
        .distinct("shop_id", "iphone_id")
        .values("shop_id", "iphone_id", "New_Product_Price")
    )

    # (shop, iphone) -> price
    data_by_si: Dict[tuple, float] = {}
    for rec in base_qs:
        sid = int(rec["shop_id"])
        iid = int(rec["iphone_id"])
        price = float(rec["New_Product_Price"])
        data_by_si[(sid, iid)] = price

    if not data_by_si:
        print(f"  ⚠️  窗口内没有数据，跳过")
        return {"bucket": bucket_dt.isoformat(), "skipped": True, "reason": "no_data"}

    print(f"  📊 找到 {len(data_by_si)} 个 shop×iphone 组合")

    # 统计计数器
    stats = {
        "case1_shop_iphone": 0,
        "case2_shopcohort_iphone": 0,
        "case3_shop_cohort": 0,
        "case4_shopcohort_cohort": 0,
    }

    created_features = []

    # ===== Case 1: shop × iphone =====
    for (sid, iid), price in data_by_si.items():
        scope = f"shop:{sid}|iphone:{iid}"
        features = create_simple_features(scope, price, bucket_dt)
        created_features.extend(features)
        stats["case1_shop_iphone"] += 1

    # 预取组合配置
    profiles = list(ShopWeightProfile.objects.all())
    prof_items = {
        prof.id: {
            it["shop_id"]: float(it.get("weight") or 1.0)
            for it in ShopWeightItem.objects.filter(profile=prof).values("shop_id", "weight")
        }
        for prof in profiles
    }
    has_shop_profile = any(bool(prof_items.get(p.id)) for p in profiles)

    cohorts = list(Cohort.objects.all())
    cmembers = {
        coh.id: {
            m["iphone_id"]: float(m.get("weight") or 1.0)
            for m in CohortMember.objects.filter(cohort=coh).values("iphone_id", "weight")
        }
        for coh in cohorts
    }

    shops_seen = sorted(set(sid for sid, _ in data_by_si.keys()))
    iphones_seen = sorted(set(iid for _, iid in data_by_si.keys()))

    # ===== Case 2: shopcohort × iphone =====
    if has_shop_profile:
        for prof in profiles:
            sw = prof_items.get(prof.id, {})
            if not sw:
                continue

            shops_in = set(sw.keys()) & set(shops_seen)
            if not shops_in:
                continue

            for iid in iphones_seen:
                # 加权平均价格
                weighted_sum = 0.0
                weight_total = 0.0
                for sid in shops_in:
                    price = data_by_si.get((int(sid), int(iid)))
                    if price is None:
                        continue
                    w = float(sw.get(sid, 1.0))
                    weighted_sum += w * price
                    weight_total += w

                if weight_total > 0:
                    avg_price = weighted_sum / weight_total
                    scope = f"shopcohort:{prof.slug}|iphone:{iid}"
                    features = create_simple_features(scope, avg_price, bucket_dt)
                    created_features.extend(features)
                    stats["case2_shopcohort_iphone"] += 1

    # ===== Case 3: shop × cohort =====
    for coh in cohorts:
        cm = cmembers.get(coh.id, {})
        if not cm:
            continue

        iphones_in = set(cm.keys()) & set(iphones_seen)
        if not iphones_in:
            continue

        for sid in shops_seen:
            # 加权平均价格
            weighted_sum = 0.0
            weight_total = 0.0
            for iid in iphones_in:
                price = data_by_si.get((int(sid), int(iid)))
                if price is None:
                    continue
                w = float(cm.get(iid, 1.0))
                weighted_sum += w * price
                weight_total += w

            if weight_total > 0:
                avg_price = weighted_sum / weight_total
                scope = f"shop:{sid}|cohort:{coh.slug}"
                features = create_simple_features(scope, avg_price, bucket_dt)
                created_features.extend(features)
                stats["case3_shop_cohort"] += 1

    # ===== Case 4: shopcohort × cohort =====
    if has_shop_profile:
        for prof in profiles:
            sw = prof_items.get(prof.id, {})
            if not sw:
                continue

            shops_in = set(sw.keys()) & set(shops_seen)
            if not shops_in:
                continue

            for coh in cohorts:
                cm = cmembers.get(coh.id, {})
                if not cm:
                    continue

                iphones_in = set(cm.keys()) & set(iphones_seen)
                if not iphones_in:
                    continue

                # 双重加权平均
                weighted_sum = 0.0
                weight_total = 0.0
                for sid in shops_in:
                    for iid in iphones_in:
                        price = data_by_si.get((int(sid), int(iid)))
                        if price is None:
                            continue
                        w_shop = float(sw.get(sid, 1.0))
                        w_iphone = float(cm.get(iid, 1.0))
                        w = w_shop * w_iphone
                        weighted_sum += w * price
                        weight_total += w

                if weight_total > 0:
                    avg_price = weighted_sum / weight_total
                    scope = f"shopcohort:{prof.slug}|cohort:{coh.slug}"
                    features = create_simple_features(scope, avg_price, bucket_dt)
                    created_features.extend(features)
                    stats["case4_shopcohort_cohort"] += 1

    # 批量创建
    if created_features:
        FeatureSnapshot.objects.bulk_create(created_features, ignore_conflicts=True)

    total = len(created_features)
    print(f"  ✅ 生成 {total} 条 FeatureSnapshot 记录")
    print(f"     Case1: {stats['case1_shop_iphone']}, "
          f"Case2: {stats['case2_shopcohort_iphone']}, "
          f"Case3: {stats['case3_shop_cohort']}, "
          f"Case4: {stats['case4_shopcohort_cohort']}")

    return {
        "bucket": bucket_dt.isoformat(),
        "success": True,
        "total_features": total,
        "stats": stats,
    }


def create_simple_features(scope: str, value: float, bucket_dt: datetime) -> List[FeatureSnapshot]:
    """
    创建简化的统计特征

    策略：将单个原始值填充为统计指标
    - mean = value
    - median = value
    - std = 0
    - min = value
    - max = value
    - count = 1
    - dispersion = 0
    等等
    """
    features = []

    # 四舍五入到4位小数
    def _d4(v):
        return float(Decimal(str(v)).quantize(Decimal('0.0001')))

    v = _d4(value)

    # 基础统计指标
    feature_specs = [
        ("mean", v),
        ("median", v),
        ("std", 0.0),
        ("min", v),
        ("max", v),
        ("count", 1.0),
        ("dispersion", 0.0),
        ("range", 0.0),
        ("q25", v),
        ("q75", v),
        ("iqr", 0.0),
        ("cv", 0.0),  # coefficient of variation
        ("skew", 0.0),
        ("kurtosis", 0.0),
        # 可以根据需要添加更多指标
    ]

    for name, val in feature_specs:
        features.append(FeatureSnapshot(
            bucket=bucket_dt,
            scope=scope,
            name=name,
            value=val,
            version="v1",
            is_final=True,  # 冷启动数据标记为 final
        ))

    return features


def coldstart_initialization(start_dt, count, dry_run=False):
    """执行冷启动初始化"""
    timestamps = generate_coldstart_timestamps(start_dt, count)

    print("=" * 70)
    print(f"{'[DRY RUN] ' if dry_run else ''}FeatureSnapshot 冷启动初始化")
    print("=" * 70)
    print(f"起始时间: {start_dt.isoformat()}")
    print(f"时间点数量: {count}")
    print(f"方向: 往前推（最新 → 最旧）")
    print(f"间隔: 15 分钟")
    print(f"时间范围: {timestamps[-1].isoformat()} → {timestamps[0].isoformat()}")
    print("=" * 70)

    if dry_run:
        print("\n[DRY RUN] 将要处理的时间点:")
        for i, ts in enumerate(timestamps[:10], 1):
            print(f"  {i}. {ts.isoformat()}")
        if len(timestamps) > 10:
            print(f"  ... (共 {len(timestamps)} 个)")
        return

    results = []

    for i, ts_dt in enumerate(timestamps, 1):
        print(f"\n[{i}/{len(timestamps)}]", end=" ")
        result = coldstart_single_bucket(ts_dt)
        results.append(result)

    # 汇总
    print("\n" + "=" * 70)
    print("冷启动完成")
    print("=" * 70)

    success_count = sum(1 for r in results if r.get('success'))
    skipped_count = sum(1 for r in results if r.get('skipped'))

    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")

    total_features = sum(r.get('total_features', 0) for r in results if r.get('success'))
    print(f"生成的 FeatureSnapshot 记录总数: {total_features}")

    # 显示统计汇总
    case1 = sum(r.get('stats', {}).get('case1_shop_iphone', 0) for r in results if r.get('success'))
    case2 = sum(r.get('stats', {}).get('case2_shopcohort_iphone', 0) for r in results if r.get('success'))
    case3 = sum(r.get('stats', {}).get('case3_shop_cohort', 0) for r in results if r.get('success'))
    case4 = sum(r.get('stats', {}).get('case4_shopcohort_cohort', 0) for r in results if r.get('success'))

    print(f"\n各 Case 统计:")
    print(f"  Case1 (shop×iphone): {case1}")
    print(f"  Case2 (shopcohort×iphone): {case2}")
    print(f"  Case3 (shop×cohort): {case3}")
    print(f"  Case4 (shopcohort×cohort): {case4}")

    return results


def check_status():
    """检查当前状态"""
    total = FeatureSnapshot.objects.count()

    print("=" * 70)
    print("FeatureSnapshot 表状态")
    print("=" * 70)
    print(f"总记录数: {total}")

    if total == 0:
        print("⚠️  表为空，建议执行冷启动初始化")
        return

    stats = FeatureSnapshot.objects.aggregate(
        min_bucket=Min('bucket'),
        max_bucket=Max('bucket'),
    )

    from django.db.models import Min, Max

    print(f"最早 bucket: {stats['min_bucket']}")
    print(f"最晚 bucket: {stats['max_bucket']}")

    # 按 scope 统计
    from django.db.models import Count
    scope_counts = (
        FeatureSnapshot.objects
        .values('scope')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )

    print(f"\n前 5 个 scope:")
    for item in scope_counts:
        print(f"  {item['scope']}: {item['count']} 条")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FeatureSnapshot 冷启动初始化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start",
        help="起始时间 (ISO 8601 格式)，往前推",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="填充的时间点数量（默认40）",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动从最近的边界时间开始",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查状态，不执行",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演习模式：不实际创建数据",
    )

    args = parser.parse_args()

    if args.check_only:
        check_status()
        sys.exit(0)

    # 确定起始时间
    if args.auto:
        start_dt = find_latest_boundary()
        print(f"自动选择起始时间: {start_dt.isoformat()}\n")
    elif args.start:
        start_dt = datetime.fromisoformat(args.start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
    else:
        parser.print_help()
        print("\n提示: 使用 --auto 自动选择起始时间")
        sys.exit(1)

    # 执行初始化
    coldstart_initialization(start_dt, args.count, dry_run=args.dry_run)
