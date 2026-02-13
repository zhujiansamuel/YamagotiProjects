"""
management command: run_pipeline
参考: docs/REFACTOR_PLAN_V1.md §7.1, §20 Phase 1
"""
from __future__ import annotations

import json
from datetime import date

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "执行 GPU + ClickHouse pipeline (时间对齐 → 聚合 → 特征 → Cohort)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--run-id", required=True,
            help="写入 CH 的 run_id 标识, e.g. backfill_v1",
        )
        parser.add_argument(
            "--from", dest="date_from", required=True,
            help="起始日期 YYYY-MM-DD",
        )
        parser.add_argument(
            "--to", dest="date_to", required=True,
            help="结束日期 YYYY-MM-DD",
        )
        parser.add_argument(
            "--device",
            default=getattr(settings, "PIPELINE_DEVICE", "cpu"),
            help="PyTorch 设备, e.g. cuda:0 | cpu (默认 settings.PIPELINE_DEVICE)",
        )
        parser.add_argument(
            "--batch-days", type=int,
            default=int(getattr(settings, "PIPELINE_BATCH_DAYS", 30)),
            help="每次从 PG 读取的天数 (默认 %(default)s)",
        )
        parser.add_argument(
            "--steps",
            help="逗号分隔的步骤: align,aggregate,features,cohorts (默认全部)",
        )
        parser.add_argument(
            "--iphone-ids",
            help="限定 iPhone ID, 逗号分隔",
        )
        parser.add_argument(
            "--shop-ids",
            help="限定 Shop ID, 逗号分隔",
        )

    def handle(self, *args, **options):
        from AppleStockChecker.engine.pipeline import run

        try:
            date_from = date.fromisoformat(options["date_from"])
            date_to = date.fromisoformat(options["date_to"])
        except ValueError as e:
            raise CommandError(f"日期格式错误: {e}")

        if date_from >= date_to:
            raise CommandError("--from 必须早于 --to")

        steps = None
        if options["steps"]:
            steps = [s.strip() for s in options["steps"].split(",")]

        iphone_ids = None
        if options["iphone_ids"]:
            iphone_ids = [int(x) for x in options["iphone_ids"].split(",")]

        shop_ids = None
        if options["shop_ids"]:
            shop_ids = [int(x) for x in options["shop_ids"].split(",")]

        self.stdout.write(self.style.NOTICE(
            f"Pipeline START  run_id={options['run_id']}  "
            f"range={date_from}→{date_to}  device={options['device']}"
        ))

        stats = run(
            run_id=options["run_id"],
            date_from=date_from,
            date_to=date_to,
            device=options["device"],
            steps=steps,
            batch_days=options["batch_days"],
            iphone_ids=iphone_ids,
            shop_ids=shop_ids,
        )

        self.stdout.write(self.style.SUCCESS(
            f"Pipeline DONE  total_time={stats.get('total_seconds', '?')}s"
        ))
        self.stdout.write(json.dumps(stats, indent=2, ensure_ascii=False))
