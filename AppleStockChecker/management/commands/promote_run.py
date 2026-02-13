"""
management command: promote_run
参考: docs/REFACTOR_PLAN_V1.md §7.4, §12, §20 Phase 2 Step 2.5
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "将 ClickHouse 中的一个 run_id 提升为另一个 (通常 backfill → live)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--from", dest="from_run", required=True,
            help="源 run_id, e.g. backfill_v3",
        )
        parser.add_argument(
            "--to", dest="to_run", required=True,
            help="目标 run_id, 通常是 'live'",
        )
        parser.add_argument(
            "--keep-backup", action="store_true",
            help="promote 前把当前目标 run 备份为 backup_YYYYMMDD",
        )
        parser.add_argument(
            "--confirm", action="store_true",
            help="确认操作 (必须指定, 防误操作)",
        )

    def handle(self, *args, **options):
        from AppleStockChecker.services.clickhouse_service import ClickHouseService

        from_run = options["from_run"]
        to_run = options["to_run"]

        if not options["confirm"]:
            raise CommandError("请加 --confirm 参数确认操作")

        ch = ClickHouseService()

        self.stdout.write(self.style.NOTICE(
            f"Promoting run: {from_run} → {to_run}  "
            f"keep_backup={options['keep_backup']}"
        ))

        result = ch.promote_run(
            from_run=from_run,
            to_run=to_run,
            keep_backup=options["keep_backup"],
        )

        if "backup" in result:
            self.stdout.write(self.style.SUCCESS(
                f"  Backup created: {result['backup']}"
            ))

        if "dropped" in result:
            for table, count in result["dropped"].items():
                self.stdout.write(f"  Dropped {table}: {count} partition(s)")

        if "promoted" in result:
            for table, count in result["promoted"].items():
                self.stdout.write(self.style.SUCCESS(
                    f"  Promoted {table}: {count} rows"
                ))

        self.stdout.write(self.style.SUCCESS("Done."))
