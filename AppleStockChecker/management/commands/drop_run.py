"""
management command: drop_run
参考: docs/REFACTOR_PLAN_V1.md §7.2, §20 Phase 1
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "删除 ClickHouse 中指定 run_id 的全部数据 (按分区删除，毫秒级)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--run-id", required=True,
            help="要删除的 run_id",
        )
        parser.add_argument(
            "--confirm", action="store_true",
            help="确认删除 (必须指定，防止误操作)",
        )
        parser.add_argument(
            "--tables",
            help="限定表, 逗号分隔: price_aligned,features_wide (默认全部)",
        )

    def handle(self, *args, **options):
        from AppleStockChecker.services.clickhouse_service import ClickHouseService

        run_id = options["run_id"]

        if not options["confirm"]:
            raise CommandError("请加 --confirm 参数确认删除操作")

        if run_id == "live":
            self.stderr.write(self.style.WARNING(
                "WARNING: 你正在删除 live 数据！"
            ))

        tables = None
        if options["tables"]:
            tables = [t.strip() for t in options["tables"].split(",")]

        ch = ClickHouseService()

        self.stdout.write(f"Dropping run_id={run_id} ...")
        result = ch.drop_run(run_id, tables=tables)

        for table, count in result.items():
            self.stdout.write(self.style.SUCCESS(
                f"  {table}: {count} partition(s) dropped"
            ))

        self.stdout.write(self.style.SUCCESS("Done."))
