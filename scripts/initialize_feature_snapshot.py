#!/usr/bin/env python
"""
初始化 FeatureSnapshot 数据

用途：
  当 FeatureSnapshot 表为空时，处理边界时间生成初始数据。
  使用边界模式（boundary mode），只在边界分钟（0, 15, 30, 45）执行聚合。

使用方法：
  # 初始化指定时间范围（15分钟间隔）
  python scripts/initialize_feature_snapshot.py \
    --start "2025-10-03T23:00:00+00:00" \
    --end "2025-10-04T01:00:00+00:00"

  # 只处理单个时间点
  python scripts/initialize_feature_snapshot.py \
    --timestamp "2025-10-03T23:00:00+00:00"

  # 检查当前状态不执行初始化
  python scripts/initialize_feature_snapshot.py --check-only

  # 自动查找最早可用数据并初始化
  python scripts/initialize_feature_snapshot.py --auto

注意事项：
  1. 使用 agg_mode=boundary，只在边界分钟聚合（不会在所有15个分钟桶都聚合）
  2. 使用 sequential=True 顺序执行避免数据库压力
  3. 每个时间点处理完会验证 FeatureSnapshot 数据
  4. 建议从最早的有效数据开始初始化
"""

import os
import sys
import django
from pathlib import Path

# Django setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")
django.setup()

from datetime import datetime, timedelta, timezone
from uuid import uuid4
from django.db.models import Min, Max, Count
from AppleStockChecker.models import (
    FeatureSnapshot,
    PurchasingShopTimeAnalysis,
    PurchasingShopPriceRecord,
)
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts


UTC = timezone.utc


def check_feature_snapshot_status():
    """检查 FeatureSnapshot 当前状态"""
    total_count = FeatureSnapshot.objects.count()

    print("=" * 60)
    print("FeatureSnapshot 表状态检查")
    print("=" * 60)
    print(f"总记录数: {total_count}")

    if total_count == 0:
        print("⚠️  表为空，需要初始化")
        return False

    # 统计信息
    stats = FeatureSnapshot.objects.aggregate(
        min_bucket=Min('bucket'),
        max_bucket=Max('bucket'),
        total=Count('id'),
    )

    print(f"最早时间: {stats['min_bucket']}")
    print(f"最晚时间: {stats['max_bucket']}")

    # 按 scope 统计
    scope_counts = (
        FeatureSnapshot.objects
        .values('scope')
        .annotate(count=Count('id'))
        .order_by('scope')
    )

    print("\n各 scope 记录数:")
    for item in scope_counts:
        print(f"  {item['scope']}: {item['count']}")

    return True


def find_earliest_available_data():
    """查找最早的可用原始数据时间"""
    print("\n查找最早的可用原始数据...")

    # 从 PurchasingShopPriceRecord 查找
    earliest_record = (
        PurchasingShopPriceRecord.objects
        .order_by('recorded_at')
        .values('recorded_at')
        .first()
    )

    if not earliest_record:
        print("❌ 没有找到任何原始数据记录")
        return None

    earliest_dt = earliest_record['recorded_at']
    print(f"✅ 找到最早记录: {earliest_dt}")

    # 向上取整到 15 分钟边界
    minute = earliest_dt.minute
    rounded_minute = ((minute // 15) + 1) * 15

    if rounded_minute >= 60:
        rounded_dt = earliest_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        rounded_dt = earliest_dt.replace(minute=rounded_minute, second=0, microsecond=0)

    print(f"建议起始时间（15分钟边界）: {rounded_dt}")

    return rounded_dt


def generate_timestamps(start_dt, end_dt, interval_minutes=15):
    """生成时间戳序列（15分钟间隔）"""
    timestamps = []
    current = start_dt

    while current <= end_dt:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)

    return timestamps


def initialize_single_timestamp(timestamp_dt, dry_run=False):
    """初始化单个时间点的 FeatureSnapshot 数据"""
    ts_iso = timestamp_dt.isoformat()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}处理时间点: {ts_iso}")

    if dry_run:
        # 只检查是否已有数据
        existing_count = FeatureSnapshot.objects.filter(bucket=timestamp_dt).count()
        print(f"  现有 FeatureSnapshot 记录数: {existing_count}")
        return {"timestamp": ts_iso, "existing": existing_count, "dry_run": True}

    # 检查是否已经有数据
    existing_count = FeatureSnapshot.objects.filter(bucket=timestamp_dt).count()
    if existing_count > 0:
        print(f"  ⏭️  已有 {existing_count} 条记录，跳过")
        return {"timestamp": ts_iso, "existing": existing_count, "skipped": True}

    # 执行任务
    job_id = uuid4().hex

    print(f"  任务 ID: {job_id}")
    print(f"  参数: agg_minutes=15, agg_mode=boundary, sequential=True")

    try:
        result = batch_generate_psta_same_ts(
            job_id=job_id,
            timestamp_iso=ts_iso,
            agg_minutes=15,
            agg_mode="boundary",
            force_agg=False,  # 已废弃：边界模式自动只在边界分钟聚合
            sequential=True,  # 顺序执行
        )

        # 检查生成的数据
        new_count = FeatureSnapshot.objects.filter(bucket=timestamp_dt).count()

        print(f"  ✅ 处理完成")
        print(f"  total_buckets: {result.get('total_buckets', 0)}")
        print(f"  FeatureSnapshot 记录数: {new_count}")

        if new_count == 0:
            print(f"  ⚠️  警告：没有生成任何 FeatureSnapshot 数据")

        return {
            "timestamp": ts_iso,
            "success": True,
            "total_buckets": result.get('total_buckets', 0),
            "feature_snapshot_count": new_count,
            "result": result,
        }

    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            "timestamp": ts_iso,
            "success": False,
            "error": str(e),
        }


def initialize_range(start_dt, end_dt, dry_run=False):
    """初始化时间范围内的 FeatureSnapshot 数据"""
    timestamps = generate_timestamps(start_dt, end_dt)

    print("=" * 60)
    print(f"{'[DRY RUN] ' if dry_run else ''}初始化 FeatureSnapshot 数据")
    print("=" * 60)
    print(f"起始时间: {start_dt}")
    print(f"结束时间: {end_dt}")
    print(f"时间点数量: {len(timestamps)}")
    print(f"参数: force_agg=True, sequential=True")
    print("=" * 60)

    results = []

    for i, ts_dt in enumerate(timestamps, 1):
        print(f"\n[{i}/{len(timestamps)}]", end=" ")
        result = initialize_single_timestamp(ts_dt, dry_run=dry_run)
        results.append(result)

    # 汇总结果
    print("\n" + "=" * 60)
    print("初始化完成")
    print("=" * 60)

    success_count = sum(1 for r in results if r.get('success'))
    skipped_count = sum(1 for r in results if r.get('skipped'))
    failed_count = sum(1 for r in results if not r.get('success') and not r.get('skipped') and not r.get('dry_run'))

    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")

    if not dry_run:
        # 统计生成的 FeatureSnapshot 数据
        total_generated = sum(r.get('feature_snapshot_count', 0) for r in results if r.get('success'))
        print(f"生成的 FeatureSnapshot 记录总数: {total_generated}")

        # 显示失败的时间点
        if failed_count > 0:
            print("\n失败的时间点:")
            for r in results:
                if not r.get('success') and not r.get('skipped'):
                    print(f"  {r['timestamp']}: {r.get('error', 'Unknown error')}")

    return results


def auto_initialize(hours_back=24, dry_run=False):
    """自动初始化：从最早数据开始"""
    earliest_dt = find_earliest_available_data()

    if not earliest_dt:
        print("❌ 无法找到可用数据，终止初始化")
        return

    # 计算结束时间（最早数据 + hours_back 小时）
    end_dt = earliest_dt + timedelta(hours=hours_back)

    # 确保结束时间不超过当前时间
    now = datetime.now(UTC)
    if end_dt > now:
        end_dt = now.replace(second=0, microsecond=0)
        # 向下取整到 15 分钟边界
        minute = end_dt.minute
        rounded_minute = (minute // 15) * 15
        end_dt = end_dt.replace(minute=rounded_minute)

    print(f"\n自动初始化参数:")
    print(f"  起始: {earliest_dt}")
    print(f"  结束: {end_dt}")
    print(f"  范围: {(end_dt - earliest_dt).total_seconds() / 3600:.1f} 小时")

    if not dry_run:
        confirm = input("\n确认开始初始化? (yes/no): ")
        if confirm.lower() != 'yes':
            print("已取消")
            return

    return initialize_range(earliest_dt, end_dt, dry_run=dry_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="初始化 FeatureSnapshot 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--start",
        help="起始时间 (ISO 8601 格式，例如: 2025-10-03T23:00:00+00:00)",
    )
    parser.add_argument(
        "--end",
        help="结束时间 (ISO 8601 格式)",
    )
    parser.add_argument(
        "--timestamp",
        help="单个时间点 (ISO 8601 格式)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动从最早可用数据开始初始化",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="自动模式下的时间范围（小时数，默认24）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查当前状态，不执行初始化",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="演习模式：不实际执行任务，只显示将要处理的时间点",
    )

    args = parser.parse_args()

    # 检查状态
    if args.check_only:
        check_feature_snapshot_status()
        sys.exit(0)

    # 显示当前状态
    has_data = check_feature_snapshot_status()

    # 自动初始化
    if args.auto:
        print("\n" + "=" * 60)
        print("自动初始化模式")
        print("=" * 60)
        auto_initialize(hours_back=args.hours, dry_run=args.dry_run)

    # 单个时间点
    elif args.timestamp:
        ts_dt = datetime.fromisoformat(args.timestamp)
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=UTC)

        initialize_single_timestamp(ts_dt, dry_run=args.dry_run)

    # 时间范围
    elif args.start and args.end:
        start_dt = datetime.fromisoformat(args.start)
        end_dt = datetime.fromisoformat(args.end)

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)

        initialize_range(start_dt, end_dt, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("提示：使用 --auto 参数自动初始化，或使用 --help 查看所有选项")
        print("=" * 60)
