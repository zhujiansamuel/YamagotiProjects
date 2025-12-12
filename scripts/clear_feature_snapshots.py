#!/usr/bin/env python3
"""
临时脚本：清除 FeatureSnapshot 表的所有数据

用途：
  - 重构统计指标后清除历史数据
  - 测试环境数据重置
  - 修复数据错误后的全量重算

安全特性：
  - 支持 dry-run 模式（默认）
  - 显示详细统计信息
  - 二次确认机制
  - 分批删除避免长事务
  - 删除前建议备份

用法：
  # 1. Dry-run（仅查看统计，不删除）
  python scripts/clear_feature_snapshots.py
  # 或使用 shell 包装器：
  ./scripts/clear_feature_snapshots.sh

  # 2. 实际删除（需要二次确认）
  python scripts/clear_feature_snapshots.py --execute

  # 3. 静默删除（跳过确认，危险！）
  python scripts/clear_feature_snapshots.py --execute --force

  # 4. 指定批量大小
  python scripts/clear_feature_snapshots.py --execute --batch-size 5000

  # Docker 环境下：
  docker compose exec web python scripts/clear_feature_snapshots.py

警告：
  ⚠️  删除操作不可逆！建议先执行数据库备份
"""

import os
import sys
import argparse
from datetime import datetime

# Django 环境初始化
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'YamagotiProjects.settings')

import django
django.setup()

from django.db import connection, transaction
from django.utils import timezone
from AppleStockChecker.models import FeatureSnapshot


def get_statistics():
    """获取 FeatureSnapshot 的统计信息"""
    total_count = FeatureSnapshot.objects.count()

    if total_count == 0:
        return {
            'total_count': 0,
            'by_name': [],
            'by_scope_prefix': [],
            'by_is_final': {},
            'date_range': None,
            'table_size': None
        }

    # 按 name 统计
    by_name = list(
        FeatureSnapshot.objects
        .values('name')
        .annotate(count=django.db.models.Count('id'))
        .order_by('-count')
    )

    # 按 scope 前缀统计（取冒号前的部分）
    scopes = list(
        FeatureSnapshot.objects
        .values_list('scope', flat=True)
        .distinct()[:1000]
    )
    scope_prefixes = {}
    for scope in scopes:
        prefix = scope.split(':')[0] if ':' in scope else scope
        scope_prefixes[prefix] = scope_prefixes.get(prefix, 0) + 1
    by_scope_prefix = sorted(
        [{'prefix': k, 'count': v} for k, v in scope_prefixes.items()],
        key=lambda x: x['count'],
        reverse=True
    )

    # 按 is_final 统计
    by_is_final = {
        'final': FeatureSnapshot.objects.filter(is_final=True).count(),
        'not_final': FeatureSnapshot.objects.filter(is_final=False).count()
    }

    # 时间范围
    date_range = FeatureSnapshot.objects.aggregate(
        min_bucket=django.db.models.Min('bucket'),
        max_bucket=django.db.models.Max('bucket')
    )

    # 表大小（PostgreSQL）
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT pg_size_pretty(pg_total_relation_size('AppleStockChecker_featuresnapshot'))
            """)
            table_size = cursor.fetchone()[0]
    except Exception:
        table_size = 'N/A'

    return {
        'total_count': total_count,
        'by_name': by_name,
        'by_scope_prefix': by_scope_prefix,
        'by_is_final': by_is_final,
        'date_range': date_range,
        'table_size': table_size
    }


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*70)
    print(" FeatureSnapshot 表统计信息")
    print("="*70)

    print(f"\n📊 总记录数: {stats['total_count']:,}")

    if stats['total_count'] == 0:
        print("   ✅ 表为空，无需清理")
        return

    print(f"   表大小: {stats['table_size']}")

    if stats['date_range']['min_bucket'] and stats['date_range']['max_bucket']:
        print(f"   时间范围: {stats['date_range']['min_bucket']} ~ {stats['date_range']['max_bucket']}")

    print(f"\n📈 数据最终化状态:")
    print(f"   is_final=True:  {stats['by_is_final']['final']:,}")
    print(f"   is_final=False: {stats['by_is_final']['not_final']:,}")

    print(f"\n🏷️  特征名 (name):")
    for item in stats['by_name']:
        print(f"   {item['name']:20} {item['count']:>10,} 条")

    print(f"\n🎯 作用域前缀 (scope):")
    for item in stats['by_scope_prefix']:
        print(f"   {item['prefix']:20} ~{item['count']:>9} 个")

    print("\n" + "="*70)


def confirm_deletion(total_count):
    """二次确认删除操作"""
    print(f"\n⚠️  警告：即将删除 {total_count:,} 条 FeatureSnapshot 记录！")
    print("   此操作不可逆，建议先执行数据库备份：./scripts/pg_dump.sh")
    print()

    response = input("   确认删除？请输入 'DELETE' 继续，或按 Enter 取消: ").strip()
    return response == 'DELETE'


def delete_all_snapshots(batch_size=10000, force=False):
    """分批删除所有 FeatureSnapshot 记录"""
    stats = get_statistics()
    total_count = stats['total_count']

    if total_count == 0:
        print("✅ FeatureSnapshot 表已为空，无需清理")
        return 0

    print_statistics(stats)

    if not force:
        if not confirm_deletion(total_count):
            print("\n❌ 用户取消操作")
            return 0

    print(f"\n🔄 开始删除 {total_count:,} 条记录（批量大小: {batch_size:,}）...")

    deleted_total = 0
    batch_num = 0
    start_time = datetime.now()

    while True:
        with transaction.atomic():
            # 获取一批 ID
            ids = list(
                FeatureSnapshot.objects
                .values_list('id', flat=True)[:batch_size]
            )

            if not ids:
                break

            # 删除这批记录
            deleted_count = FeatureSnapshot.objects.filter(id__in=ids).delete()[0]
            deleted_total += deleted_count
            batch_num += 1

            progress = (deleted_total / total_count) * 100
            print(f"   Batch #{batch_num}: 已删除 {deleted_total:,}/{total_count:,} ({progress:.1f}%)")

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n✅ 删除完成！")
    print(f"   总删除记录: {deleted_total:,}")
    print(f"   总批次数: {batch_num}")
    print(f"   耗时: {elapsed:.2f} 秒")
    print(f"   速度: {deleted_total/elapsed:.0f} 条/秒")

    # 验证表已清空
    remaining = FeatureSnapshot.objects.count()
    if remaining > 0:
        print(f"\n⚠️  警告：仍有 {remaining:,} 条记录未删除！")
    else:
        print(f"\n✅ 验证通过：FeatureSnapshot 表已完全清空")

    return deleted_total


def main():
    parser = argparse.ArgumentParser(
        description='清除 FeatureSnapshot 表的所有数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='实际执行删除操作（默认为 dry-run）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='跳过二次确认（危险！）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='每批删除的记录数（默认: 10000）'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" FeatureSnapshot 清理脚本")
    print("="*70)
    print(f" 时间: {timezone.now()}")
    print(f" 模式: {'🔴 执行模式 (EXECUTE)' if args.execute else '🟢 试运行模式 (DRY-RUN)'}")

    if not args.execute:
        # Dry-run 模式：仅显示统计
        stats = get_statistics()
        print_statistics(stats)

        print("\n💡 提示：这是 dry-run 模式，数据未被删除")
        print("   如需实际删除，请使用: --execute")
        print("   删除前建议备份: ./scripts/pg_dump.sh")
        return 0

    # 执行模式
    try:
        deleted_count = delete_all_snapshots(
            batch_size=args.batch_size,
            force=args.force
        )
        return 0 if deleted_count >= 0 else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
        return 130
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
