"""
Django management command: 同步外部商品映射

用法:
    python manage.py sync_external_goods
    python manage.py sync_external_goods --api-url http://localhost:8080
    python manage.py sync_external_goods --clear  # 清空现有映射后重新同步
"""
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from AppleStockChecker.services import ExternalGoodsSyncService


class Command(BaseCommand):
    help = '从外部项目同步商品列表并生成与本项目Iphone实例的映射'

    def add_arguments(self, parser):
        parser.add_argument(
            '--api-url',
            type=str,
            help='外部API的URL (可选,默认使用配置文件中的值)',
        )

        parser.add_argument(
            '--api-token',
            type=str,
            help='外部API的访问令牌 (可选,默认使用配置文件中的值)',
        )

        parser.add_argument(
            '--db-path',
            type=str,
            default='auto_price.sqlite3',
            help='SQLite数据库文件路径 (默认: auto_price.sqlite3)',
        )

        parser.add_argument(
            '--clear',
            action='store_true',
            help='清空现有映射后重新同步 (谨慎使用!)',
        )

        parser.add_argument(
            '--show-stats',
            action='store_true',
            help='仅显示当前映射统计信息,不执行同步',
        )

        parser.add_argument(
            '--show-unmatched',
            action='store_true',
            help='显示所有未匹配的商品',
        )

        parser.add_argument(
            '--category-filter',
            type=str,
            help='商品类别过滤 (如 "iPhone")，只同步指定类别的商品',
        )

    def handle(self, *args, **options):
        """执行命令"""
        try:
            # 创建同步服务
            sync_service = ExternalGoodsSyncService(
                api_url=options.get('api_url'),
                api_token=options.get('api_token'),
                db_path=options.get('db_path'),
                category_filter=options.get('category_filter'),
            )

            # 仅显示统计信息
            if options['show_stats']:
                self._show_statistics(sync_service)
                return

            # 显示未匹配商品
            if options['show_unmatched']:
                self._show_unmatched(sync_service)
                return

            # 清空现有映射
            if options['clear']:
                self.stdout.write(self.style.WARNING('正在清空现有映射...'))
                sync_service.db_manager.clear_all_mappings()
                self.stdout.write(self.style.SUCCESS('✓ 已清空所有映射'))

            # 执行同步
            self.stdout.write(self.style.MIGRATE_HEADING('开始同步外部商品映射...'))
            self.stdout.write(f'时间: {timezone.now().strftime("%Y-%m-%d %H:%M:%S")}')
            self.stdout.write('')

            stats = sync_service.sync_goods_mappings()

            # 显示结果
            self.stdout.write('')
            self.stdout.write(self.style.SUCCESS('=' * 60))
            self.stdout.write(self.style.SUCCESS('同步完成!'))
            self.stdout.write(self.style.SUCCESS('=' * 60))
            self.stdout.write(f'总商品数:    {stats["total_items"]}')
            self.stdout.write(
                self.style.SUCCESS(f'✓ 已匹配:    {stats["matched_items"]}')
            )
            self.stdout.write(
                self.style.WARNING(f'⚠ 未匹配:    {stats["unmatched_items"]}')
            )
            if stats['error_items'] > 0:
                self.stdout.write(
                    self.style.ERROR(f'✗ 错误:      {stats["error_items"]}')
                )
            self.stdout.write(self.style.SUCCESS('=' * 60))

            # 如果有未匹配的商品,提示查看详情
            if stats['unmatched_items'] > 0:
                self.stdout.write('')
                self.stdout.write(
                    self.style.WARNING(
                        '提示: 运行以下命令查看未匹配商品详情:'
                    )
                )
                self.stdout.write(
                    f'  python manage.py sync_external_goods --show-unmatched'
                )

        except Exception as e:
            raise CommandError(f'同步失败: {str(e)}')

    def _show_statistics(self, sync_service):
        """显示统计信息"""
        stats = sync_service.get_mapping_statistics()

        self.stdout.write(self.style.MIGRATE_HEADING('映射统计信息'))
        self.stdout.write('=' * 60)
        self.stdout.write(f'总映射数:    {stats.get("total", 0)}')
        self.stdout.write(
            self.style.SUCCESS(f'已匹配:      {stats.get("matched", 0)}')
        )
        self.stdout.write(
            self.style.WARNING(f'未匹配:      {stats.get("unmatched", 0)}')
        )
        self.stdout.write(f'待处理:      {stats.get("pending", 0)}')
        if stats.get('error', 0) > 0:
            self.stdout.write(
                self.style.ERROR(f'错误:        {stats.get("error", 0)}')
            )

        if stats.get('last_sync_at'):
            self.stdout.write(f'最后同步:    {stats["last_sync_at"]}')
        self.stdout.write('=' * 60)

    def _show_unmatched(self, sync_service):
        """显示未匹配的商品"""
        unmatched = sync_service.get_all_mappings(status='unmatched')

        if not unmatched:
            self.stdout.write(self.style.SUCCESS('没有未匹配的商品'))
            return

        self.stdout.write(self.style.WARNING(f'未匹配商品列表 (共 {len(unmatched)} 个)'))
        self.stdout.write('=' * 80)
        self.stdout.write(
            f'{"ID":<8} {"标题":<30} {"颜色":<20} {"机型":<15}'
        )
        self.stdout.write('-' * 80)

        for mapping in unmatched:
            self.stdout.write(
                f'{mapping["external_goods_id"]:<8} '
                f'{mapping["external_title"][:28]:<30} '
                f'{mapping["external_spec_name"][:18]:<20} '
                f'{mapping["external_category_three_name"][:13]:<15}'
            )

        self.stdout.write('=' * 80)
        self.stdout.write('')
        self.stdout.write(self.style.WARNING('可能的原因:'))
        self.stdout.write('  1. 颜色名称不匹配 (需要在映射规则中添加颜色映射)')
        self.stdout.write('  2. 机型名称不匹配 (需要确认category_three_name与model_name对应)')
        self.stdout.write('  3. 容量解析失败 (需要检查title格式)')
        self.stdout.write('  4. 本项目数据库中不存在对应的Iphone记录')
