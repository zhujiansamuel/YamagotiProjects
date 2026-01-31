#!/usr/bin/env python
"""
诊断边界判断问题

用于检查为什么 force_agg=False 时边界聚合没有触发

使用方法：
  python scripts/diagnose_boundary_issue.py --timestamp "2025-10-03T23:00:00+00:00"
  python scripts/diagnose_boundary_issue.py --auto  # 自动检测最近的边界时间
"""

import os
import sys
import django
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")
django.setup()

from datetime import datetime, timedelta, timezone
from AppleStockChecker.tasks.timestamp_alignment_task import (
    _to_aware,
    _floor_to_step,
    batch_generate_psta_same_ts,
)
from AppleStockChecker.models import FeatureSnapshot, PurchasingShopTimeAnalysis
from uuid import uuid4

UTC = timezone.utc


def check_boundary_logic(timestamp_iso, agg_minutes=15):
    """检查边界判断逻辑"""
    print("=" * 70)
    print(f"边界判断诊断：{timestamp_iso}")
    print("=" * 70)

    # 解析时间
    ts_dt = datetime.fromisoformat(timestamp_iso)
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=UTC)

    print(f"\n输入时间：{ts_dt}")
    print(f"  年: {ts_dt.year}")
    print(f"  月: {ts_dt.month}")
    print(f"  日: {ts_dt.day}")
    print(f"  时: {ts_dt.hour}")
    print(f"  分: {ts_dt.minute}")
    print(f"  秒: {ts_dt.second}")

    # 计算边界
    boundary = _floor_to_step(ts_dt, agg_minutes)
    print(f"\n边界时间：{boundary}")
    print(f"  年: {boundary.year}")
    print(f"  月: {boundary.month}")
    print(f"  日: {boundary.day}")
    print(f"  时: {boundary.hour}")
    print(f"  分: {boundary.minute}")
    print(f"  秒: {boundary.second}")

    # 判断是否为边界
    is_boundary = (ts_dt == boundary)
    print(f"\n是否为边界：{'✅ 是' if is_boundary else '❌ 否'}")

    if not is_boundary:
        print(f"  差异：{(ts_dt - boundary).total_seconds()} 秒")
        print(f"  原因：输入时间不在 {agg_minutes} 分钟边界上")
        print(f"  最近的边界时间：{boundary}")
        next_boundary = boundary + timedelta(minutes=agg_minutes)
        print(f"  下一个边界时间：{next_boundary}")

    # 检查分钟是否能被 agg_minutes 整除
    print(f"\n分钟检查：")
    print(f"  minute % {agg_minutes} = {ts_dt.minute % agg_minutes}")
    print(f"  是否整除：{'✅ 是' if ts_dt.minute % agg_minutes == 0 else '❌ 否'}")
    print(f"  秒数是否为0：{'✅ 是' if ts_dt.second == 0 else '❌ 否 (秒=' + str(ts_dt.second) + ')'}")
    print(f"  微秒是否为0：{'✅ 是' if ts_dt.microsecond == 0 else '❌ 否 (微秒=' + str(ts_dt.microsecond) + ')'}")

    # 计算 do_agg_local
    print(f"\ndo_agg 计算逻辑：")
    print(f"  force_agg=True:  do_agg = True or {is_boundary} = True  ✅ 会执行聚合")
    print(f"  force_agg=False: do_agg = False or {is_boundary} = {is_boundary}  {'✅ 会执行聚合' if is_boundary else '❌ 不会执行聚合'}")

    return is_boundary


def check_feature_snapshot_data(timestamp_iso):
    """检查该时间点是否有 FeatureSnapshot 数据"""
    print("\n" + "=" * 70)
    print("FeatureSnapshot 数据检查")
    print("=" * 70)

    ts_dt = datetime.fromisoformat(timestamp_iso)
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=UTC)

    # 查询该时间点的 FeatureSnapshot
    snapshots = FeatureSnapshot.objects.filter(bucket=ts_dt)
    count = snapshots.count()

    print(f"\n时间点：{ts_dt}")
    print(f"FeatureSnapshot 记录数：{count}")

    if count == 0:
        print("  ❌ 没有数据")
        # 检查附近是否有数据
        before_count = FeatureSnapshot.objects.filter(
            bucket__gte=ts_dt - timedelta(minutes=30),
            bucket__lt=ts_dt,
        ).count()
        after_count = FeatureSnapshot.objects.filter(
            bucket__gt=ts_dt,
            bucket__lte=ts_dt + timedelta(minutes=30),
        ).count()
        print(f"  前30分钟内：{before_count} 条")
        print(f"  后30分钟内：{after_count} 条")
    else:
        print("  ✅ 有数据")
        # 显示前几条
        print(f"\n前5条记录：")
        for snap in snapshots[:5]:
            print(f"  {snap.scope} | {snap.name} = {snap.value}")

        # 按 scope 统计
        scope_counts = {}
        for snap in snapshots:
            scope_counts[snap.scope] = scope_counts.get(snap.scope, 0) + 1

        print(f"\n按 scope 统计：")
        for scope, cnt in sorted(scope_counts.items()):
            print(f"  {scope}: {cnt} 条")


def check_psta_data(timestamp_iso, agg_minutes=15):
    """检查 PurchasingShopTimeAnalysis 数据"""
    print("\n" + "=" * 70)
    print("PurchasingShopTimeAnalysis 数据检查")
    print("=" * 70)

    ts_dt = datetime.fromisoformat(timestamp_iso)
    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.replace(tzinfo=UTC)

    boundary = _floor_to_step(ts_dt, agg_minutes)
    window_start = boundary
    window_end = boundary + timedelta(minutes=agg_minutes)

    print(f"\n聚合窗口：")
    print(f"  起始：{window_start}")
    print(f"  结束：{window_end}")

    # 查询该窗口内的数据
    psta_data = PurchasingShopTimeAnalysis.objects.filter(
        bucket__gte=window_start,
        bucket__lt=window_end,
    )
    count = psta_data.count()

    print(f"\nPurchasingShopTimeAnalysis 记录数：{count}")

    if count == 0:
        print("  ❌ 窗口内没有数据")
        print("  原因：可能是 psta_process_minute_bucket 没有写入分钟数据")
    else:
        print("  ✅ 窗口内有数据")
        # 按 bucket 统计
        bucket_counts = {}
        for psta in psta_data:
            bucket_iso = psta.bucket.isoformat()
            bucket_counts[bucket_iso] = bucket_counts.get(bucket_iso, 0) + 1

        print(f"\n按分钟统计：")
        for bucket_iso, cnt in sorted(bucket_counts.items()):
            print(f"  {bucket_iso}: {cnt} 条")


def test_with_force_agg(timestamp_iso, force_agg_value):
    """测试不同 force_agg 值的效果"""
    print("\n" + "=" * 70)
    print(f"测试 force_agg={force_agg_value}")
    print("=" * 70)

    job_id = uuid4().hex
    print(f"\n任务 ID：{job_id}")
    print(f"参数：")
    print(f"  timestamp_iso: {timestamp_iso}")
    print(f"  agg_minutes: 15")
    print(f"  agg_mode: boundary")
    print(f"  force_agg: {force_agg_value}")
    print(f"  sequential: True")

    try:
        result = batch_generate_psta_same_ts(
            job_id=job_id,
            timestamp_iso=timestamp_iso,
            agg_minutes=15,
            agg_mode="boundary",
            force_agg=force_agg_value,
            sequential=True,
        )

        print(f"\n✅ 执行成功")
        print(f"结果：")
        print(f"  total_buckets: {result.get('total_buckets', 0)}")

        # 检查生成的 FeatureSnapshot 数据
        ts_dt = datetime.fromisoformat(timestamp_iso)
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=UTC)

        new_count = FeatureSnapshot.objects.filter(bucket=ts_dt).count()
        print(f"  FeatureSnapshot 记录数: {new_count}")

        if new_count == 0:
            print(f"  ⚠️  警告：没有生成 FeatureSnapshot 数据")
        else:
            print(f"  ✅ 成功生成数据")

        return result

    except Exception as e:
        print(f"\n❌ 执行失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def find_recent_boundary(agg_minutes=15):
    """查找最近的边界时间"""
    now = datetime.now(UTC)

    # 向下取整到边界
    boundary = _floor_to_step(now, agg_minutes)

    # 往前找一个已经过去的边界（至少5分钟前，保证数据稳定）
    target = boundary
    while (now - target).total_seconds() < 300:  # 5分钟
        target -= timedelta(minutes=agg_minutes)

    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="诊断边界判断问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--timestamp",
        help="要检查的时间点 (ISO 8601 格式)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动选择最近的边界时间进行检查",
    )
    parser.add_argument(
        "--agg-minutes",
        type=int,
        default=15,
        help="聚合步长（分钟，默认15）",
    )
    parser.add_argument(
        "--test-force-agg",
        action="store_true",
        help="测试 force_agg=True 和 force_agg=False 的效果",
    )
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="只检查数据，不执行任务",
    )

    args = parser.parse_args()

    if args.auto:
        timestamp_iso = find_recent_boundary(args.agg_minutes).isoformat()
        print(f"自动选择的边界时间：{timestamp_iso}\n")
    elif args.timestamp:
        timestamp_iso = args.timestamp
    else:
        parser.print_help()
        sys.exit(1)

    # 1. 检查边界逻辑
    is_boundary = check_boundary_logic(timestamp_iso, args.agg_minutes)

    # 2. 检查现有数据
    check_feature_snapshot_data(timestamp_iso)
    check_psta_data(timestamp_iso, args.agg_minutes)

    # 3. 测试 force_agg
    if args.test_force_agg and not args.check_data_only:
        print("\n" + "=" * 70)
        print("开始测试")
        print("=" * 70)

        # 先测试 force_agg=False
        print("\n\n### 测试 1: force_agg=False（正常模式）###")
        test_with_force_agg(timestamp_iso, False)

        # 如果不是边界，再测试 force_agg=True
        if not is_boundary:
            print("\n\n### 测试 2: force_agg=True（强制模式）###")
            test_with_force_agg(timestamp_iso, True)

    # 4. 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)

    if is_boundary:
        print("\n✅ 该时间点是边界时间")
        print("   force_agg=False 时应该会执行聚合")
        print("   如果仍然没有数据，可能的原因：")
        print("   1. PurchasingShopTimeAnalysis 窗口内没有数据")
        print("   2. _run_aggregation 函数内部有其他条件跳过了聚合")
        print("   3. 聚合计算逻辑有问题")
    else:
        print("\n❌ 该时间点不是边界时间")
        print("   force_agg=False 时不会执行聚合")
        print(f"   建议使用边界时间进行测试")

    print("\n建议下一步：")
    if is_boundary:
        print("  1. 使用 --test-force-agg 参数测试该边界时间")
        print("  2. 检查 _run_aggregation 函数的日志输出")
        print("  3. 确认 PurchasingShopTimeAnalysis 窗口内有足够的数据")
    else:
        print("  1. 使用 --auto 参数自动选择边界时间")
        print("  2. 或手动指定一个边界时间（分钟能被15整除，秒和微秒为0）")
        print(f"  3. 例如：{find_recent_boundary(args.agg_minutes).isoformat()}")
