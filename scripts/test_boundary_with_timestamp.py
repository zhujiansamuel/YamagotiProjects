#!/usr/bin/env python
"""
测试边界判断逻辑的实际行为

用法：
  python scripts/test_boundary_with_timestamp.py "2025-09-20T05:00:00+00:00"
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

# 从 timestamp_alignment_task 导入相关函数
from AppleStockChecker.tasks.timestamp_alignment_task import _to_aware, _floor_to_step

def test_timestamp(timestamp_iso, agg_minutes=15):
    """测试时间戳的边界判断"""
    print("=" * 70)
    print(f"测试时间戳: {timestamp_iso}")
    print("=" * 70)

    # 解析时间
    mdt = _to_aware(timestamp_iso)

    print(f"\n解析后的时间:")
    print(f"  {mdt}")
    print(f"  年: {mdt.year}")
    print(f"  月: {mdt.month}")
    print(f"  日: {mdt.day}")
    print(f"  时: {mdt.hour}")
    print(f"  分: {mdt.minute}")
    print(f"  秒: {mdt.second}")
    print(f"  微秒: {mdt.microsecond}")
    print(f"  时区: {mdt.tzinfo}")

    # 计算边界
    boundary = _floor_to_step(mdt, int(agg_minutes))

    print(f"\n边界时间:")
    print(f"  {boundary}")
    print(f"  年: {boundary.year}")
    print(f"  月: {boundary.month}")
    print(f"  日: {boundary.day}")
    print(f"  时: {boundary.hour}")
    print(f"  分: {boundary.minute}")
    print(f"  秒: {boundary.second}")
    print(f"  微秒: {boundary.microsecond}")

    # 判断是否为边界
    is_boundary = (mdt == boundary)

    print(f"\n边界判断:")
    print(f"  mdt == boundary: {is_boundary}")
    print(f"  mdt: {repr(mdt)}")
    print(f"  boundary: {repr(boundary)}")

    if not is_boundary:
        print(f"\n差异分析:")
        print(f"  时间差: {(mdt - boundary).total_seconds()} 秒")
        print(f"  分钟差: {mdt.minute - boundary.minute}")
        print(f"  秒差: {mdt.second - boundary.second}")
        print(f"  微秒差: {mdt.microsecond - boundary.microsecond}")

    # 模拟边界模式的 do_agg 计算
    print(f"\ndo_agg 计算 (boundary 模式):")
    print(f"  force_agg=False: do_agg = False or {is_boundary} = {is_boundary}")
    print(f"  force_agg=True:  do_agg = True or {is_boundary} = True")

    # 分钟检查
    print(f"\n分钟检查:")
    print(f"  minute % {agg_minutes} = {mdt.minute % agg_minutes}")
    if mdt.minute % agg_minutes == 0:
        print(f"  ✅ 分钟能被 {agg_minutes} 整除")
    else:
        print(f"  ❌ 分钟不能被 {agg_minutes} 整除")

    if mdt.second == 0:
        print(f"  ✅ 秒数为 0")
    else:
        print(f"  ❌ 秒数不为 0 (second={mdt.second})")

    if mdt.microsecond == 0:
        print(f"  ✅ 微秒为 0")
    else:
        print(f"  ❌ 微秒不为 0 (microsecond={mdt.microsecond})")

    # 测试 _floor_to_step 函数的实际计算
    print(f"\n_floor_to_step 计算过程:")
    print(f"  dt.minute % {agg_minutes} = {mdt.minute % agg_minutes}")
    print(f"  dt.second = {mdt.second}")
    print(f"  dt.microsecond = {mdt.microsecond}")

    delta = timedelta(
        minutes=mdt.minute % agg_minutes,
        seconds=mdt.second,
        microseconds=mdt.microsecond
    )
    print(f"  减去的 timedelta: {delta}")
    print(f"  mdt - delta = {mdt - delta}")

    # 结论
    print(f"\n" + "=" * 70)
    print("结论:")
    print("=" * 70)

    if is_boundary:
        print(f"✅ 这是一个边界时间")
        print(f"   force_agg=False 时会执行聚合")
    else:
        print(f"❌ 这不是一个边界时间")
        print(f"   force_agg=False 时不会执行聚合")
        print(f"   force_agg=True 时会强制执行聚合")

        # 给出最近的边界时间
        print(f"\n最近的边界时间:")
        print(f"  当前边界: {boundary.isoformat()}")
        next_boundary = boundary + timedelta(minutes=agg_minutes)
        print(f"  下一边界: {next_boundary.isoformat()}")

    return is_boundary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/test_boundary_with_timestamp.py <timestamp_iso>")
        print("示例: python scripts/test_boundary_with_timestamp.py '2025-09-20T05:00:00+00:00'")
        sys.exit(1)

    timestamp_iso = sys.argv[1]
    agg_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    test_timestamp(timestamp_iso, agg_minutes)
