#!/usr/bin/env python
"""
测试 FeatureSnapshot 聚合日志输出

用途：
  手动触发一次聚合，检查日志是否正确输出

使用方法：
  python scripts/test_logging.py --timestamp "2025-10-04T01:00:00+00:00"
  python scripts/test_logging.py --auto  # 使用最近的边界时间
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
from uuid import uuid4
from AppleStockChecker.tasks.timestamp_alignment_task import batch_generate_psta_same_ts
import logging

# 确保日志配置生效
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(message)s',
)

UTC = tz.utc


def find_recent_boundary(agg_minutes=15):
    """查找最近的边界时间"""
    now = datetime.now(UTC)
    minute = (now.minute // agg_minutes) * agg_minutes
    boundary = now.replace(minute=minute, second=0, microsecond=0)

    # 确保是过去的时间（至少5分钟前）
    while (now - boundary).total_seconds() < 300:
        boundary -= timedelta(minutes=agg_minutes)

    return boundary


def test_logging(timestamp_iso):
    """测试日志输出"""
    print("=" * 70)
    print("测试 FeatureSnapshot 聚合日志")
    print("=" * 70)
    print(f"时间点: {timestamp_iso}")
    print(f"预期日志输出:")
    print(f"  1. 🔄 [FeatureSnapshot 聚合] 开始计算")
    print(f"  2. 📊 [数据源] shops/iphones 统计")
    print(f"  3. ✍️ [特征写入] 各 Case 统计")
    print(f"  4. ✅ [FeatureSnapshot 聚合] 完成")
    print("=" * 70)
    print()

    job_id = uuid4().hex

    print(f"触发任务...")
    print(f"  Job ID: {job_id}")
    print(f"  Timestamp: {timestamp_iso}")
    print()

    try:
        result = batch_generate_psta_same_ts(
            job_id=job_id,
            timestamp_iso=timestamp_iso,
            agg_minutes=15,
            agg_mode="boundary",
            force_agg=False,
            sequential=True,  # 顺序执行便于查看日志
        )

        print()
        print("=" * 70)
        print("任务执行完成")
        print("=" * 70)
        print(f"结果:")
        print(f"  total_buckets: {result.get('total_buckets', 0)}")
        print(f"  sequential: {result.get('sequential', False)}")

        if result.get('sequential'):
            print(f"  处理结果: {len(result.get('result', {}).get('results', []))} 个子任务")

        # 检查是否看到了日志
        print()
        print("注意：")
        print("  如果上面没有看到 🔄/📊/✍️/✅ 开头的日志，")
        print("  说明日志配置可能还有问题。")
        print()
        print("  检查:")
        print("  1. Django settings.py 中的 LOGGING 配置")
        print("  2. Celery worker 是否需要重启")
        print("  3. 是否在 Docker 环境中运行")

    except Exception as e:
        print()
        print(f"❌ 任务执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="测试 FeatureSnapshot 聚合日志",
    )

    parser.add_argument(
        "--timestamp",
        help="测试时间点 (ISO 8601 格式)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="使用最近的边界时间",
    )

    args = parser.parse_args()

    if args.auto:
        timestamp_iso = find_recent_boundary().isoformat()
        print(f"使用最近的边界时间: {timestamp_iso}\n")
    elif args.timestamp:
        timestamp_iso = args.timestamp
    else:
        parser.print_help()
        sys.exit(1)

    test_logging(timestamp_iso)
