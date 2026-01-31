#!/usr/bin/env python
"""
验证 FeatureSnapshot 数据生成的脚本

用途：
1. 检查指定时间范围内的 FeatureSnapshot 数据
2. 验证数据是否正确生成
3. 诊断可能的问题

使用方法：
    python scripts/verify_feature_snapshot.py --start "2025-10-03T23:00:00+00:00" --end "2025-10-04T01:00:00+00:00"
"""

import os
import sys
import django
from pathlib import Path

# 设置 Django 环境
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AppleStockChecker.settings')
django.setup()

from django.utils import timezone
from datetime import datetime, timedelta
from AppleStockChecker.models import (
    FeatureSnapshot,
    PurchasingShopTimeAnalysis,
    PurchasingShopPriceRecord,
)
from collections import Counter
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='验证 FeatureSnapshot 数据生成')
    parser.add_argument('--start', type=str, required=True,
                        help='开始时间 (ISO格式, 如: 2025-10-03T23:00:00+00:00)')
    parser.add_argument('--end', type=str, required=True,
                        help='结束时间 (ISO格式, 如: 2025-10-04T01:00:00+00:00)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    return parser.parse_args()


def check_feature_snapshots(start_dt, end_dt, verbose=False):
    """检查 FeatureSnapshot 数据"""
    print("\n" + "="*80)
    print("📊 FeatureSnapshot 数据检查")
    print("="*80)

    snapshots = FeatureSnapshot.objects.filter(
        bucket__gte=start_dt,
        bucket__lte=end_dt,
    ).order_by('bucket', 'scope', 'name')

    total_count = snapshots.count()
    print(f"\n✅ 找到 {total_count} 条 FeatureSnapshot 记录")

    if total_count == 0:
        print("\n❌ 警告：没有找到任何 FeatureSnapshot 数据！")
        print("   可能的原因：")
        print("   1. 聚合任务未在边界时刻触发")
        print("   2. 原始数据不足（PurchasingShopPriceRecord）")
        print("   3. collect_items_for_psta 的 bug（bucket_by_minute 初始化）")
        return False

    # 按时间桶统计
    by_bucket = Counter()
    for snap in snapshots:
        by_bucket[snap.bucket] += 1

    print(f"\n📅 时间桶分布（共 {len(by_bucket)} 个时间桶）：")
    for bucket, count in sorted(by_bucket.items()):
        print(f"   {bucket.isoformat()}: {count} 条记录")

    # 按 scope 统计
    by_scope = Counter()
    for snap in snapshots:
        scope_type = snap.scope.split('|')[0].split(':')[0]  # 提取 scope 类型
        by_scope[scope_type] += 1

    print(f"\n🔍 Scope 类型分布：")
    for scope_type, count in by_scope.most_common():
        print(f"   {scope_type}: {count} 条记录")

    # 按 name 统计
    by_name = Counter()
    for snap in snapshots:
        by_name[snap.name] += 1

    print(f"\n📈 指标名称分布：")
    for name, count in by_name.most_common():
        print(f"   {name}: {count} 条记录")

    # 详细信息
    if verbose:
        print(f"\n🔬 详细记录（前20条）：")
        for snap in snapshots[:20]:
            print(f"   {snap.bucket.isoformat()} | {snap.scope:50s} | {snap.name:15s} | {snap.value:.4f}")

    return True


def check_raw_data(start_dt, end_dt):
    """检查原始数据"""
    print("\n" + "="*80)
    print("📦 原始数据检查（PurchasingShopPriceRecord）")
    print("="*80)

    records = PurchasingShopPriceRecord.objects.filter(
        recorded_at__gte=start_dt,
        recorded_at__lte=end_dt,
    )

    total_count = records.count()
    print(f"\n✅ 找到 {total_count} 条原始价格记录")

    if total_count == 0:
        print("\n❌ 警告：没有找到任何原始数据！")
        print("   这解释了为什么没有 FeatureSnapshot 数据")
        return False

    # 按店铺统计
    by_shop = Counter()
    for rec in records:
        by_shop[rec.shop_id] += 1

    print(f"\n🏪 店铺分布（共 {len(by_shop)} 个店铺）：")
    for shop_id, count in by_shop.most_common(10):
        print(f"   店铺 {shop_id}: {count} 条记录")

    # 按 iPhone 统计
    by_iphone = Counter()
    for rec in records:
        by_iphone[rec.iphone_id] += 1

    print(f"\n📱 iPhone 分布（共 {len(by_iphone)} 个型号）：")
    for iphone_id, count in by_iphone.most_common(10):
        print(f"   iPhone {iphone_id}: {count} 条记录")

    return True


def check_aligned_data(start_dt, end_dt):
    """检查对齐后的数据"""
    print("\n" + "="*80)
    print("🎯 对齐数据检查（PurchasingShopTimeAnalysis）")
    print("="*80)

    analyses = PurchasingShopTimeAnalysis.objects.filter(
        Timestamp_Time__gte=start_dt,
        Timestamp_Time__lte=end_dt,
    )

    total_count = analyses.count()
    print(f"\n✅ 找到 {total_count} 条对齐记录")

    if total_count == 0:
        print("\n⚠️  警告：没有找到对齐数据")
        print("   这表示 psta_process_minute_bucket 任务可能没有写入数据")
        return False

    # 按时间统计
    by_time = Counter()
    for ana in analyses:
        by_time[ana.Timestamp_Time] += 1

    print(f"\n📅 时间分布（共 {len(by_time)} 个时间点）：")
    for ts, count in sorted(by_time.items())[:20]:
        print(f"   {ts.isoformat()}: {count} 条记录")

    return True


def suggest_fixes():
    """建议修复方案"""
    print("\n" + "="*80)
    print("💡 建议修复方案")
    print("="*80)
    print("""
1. 确保传入的 timestamp_iso 是 15 分钟的边界时刻
   例如: 23:00, 23:15, 23:30, 23:45, 00:00 等

2. 检查 collectors.py 中的 bucket_by_minute 初始化
   应该是: bucket_by_minute = {tick: [] for tick in ticks_iso}
   而不是: bucket_by_minute = {}

3. 确保有足够的原始数据（PurchasingShopPriceRecord）

4. 使用 force_agg=True 强制触发聚合（测试用）

5. 查看 Celery worker 日志，检查是否有错误
   docker compose logs worker -f

6. 测试单个时间点的处理：
   python scripts/test_single_bucket.py --timestamp "2025-10-03T23:00:00+00:00"
""")


def main():
    args = parse_args()

    # 解析时间
    try:
        start_dt = datetime.fromisoformat(args.start)
        end_dt = datetime.fromisoformat(args.end)

        # 确保时区aware
        if timezone.is_naive(start_dt):
            start_dt = timezone.make_aware(start_dt)
        if timezone.is_naive(end_dt):
            end_dt = timezone.make_aware(end_dt)

    except ValueError as e:
        print(f"❌ 时间格式错误: {e}")
        print("   请使用 ISO 格式，如: 2025-10-03T23:00:00+00:00")
        sys.exit(1)

    print(f"\n🔍 检查时间范围: {start_dt.isoformat()} 到 {end_dt.isoformat()}")
    print(f"   时长: {end_dt - start_dt}")

    # 执行检查
    has_raw = check_raw_data(start_dt, end_dt)
    has_aligned = check_aligned_data(start_dt, end_dt)
    has_features = check_feature_snapshots(start_dt, end_dt, verbose=args.verbose)

    # 总结
    print("\n" + "="*80)
    print("📋 检查总结")
    print("="*80)
    print(f"   原始数据 (PurchasingShopPriceRecord): {'✅ 有' if has_raw else '❌ 无'}")
    print(f"   对齐数据 (PurchasingShopTimeAnalysis): {'✅ 有' if has_aligned else '❌ 无'}")
    print(f"   特征快照 (FeatureSnapshot): {'✅ 有' if has_features else '❌ 无'}")

    if not has_features:
        suggest_fixes()
        sys.exit(1)
    else:
        print("\n✅ 所有检查通过！FeatureSnapshot 数据正常生成")
        sys.exit(0)


if __name__ == '__main__':
    main()
