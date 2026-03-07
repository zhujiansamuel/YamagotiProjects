"""
管理命令: 训练 LightGBM 价格预测模型。

用法:
  python manage.py train_price_model --iphone-ids 1,2,3
  python manage.py train_price_model --all
  python manage.py train_price_model --iphone-ids 1 --days 7 --run-id backfill_v3
"""
from __future__ import annotations

import logging
from datetime import datetime

from django.core.management.base import BaseCommand

from AppleStockChecker.engine.prediction import (
    train_models_for_iphone,
    save_artifacts,
)
from AppleStockChecker.models import Iphone
from AppleStockChecker.services.clickhouse_service import ClickHouseService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Train LightGBM price prediction models for specified iPhones"

    def add_arguments(self, parser):
        parser.add_argument(
            "--iphone-ids",
            type=str,
            default="",
            help="Comma-separated iPhone IDs to train, or empty for --all",
        )
        parser.add_argument("--all", action="store_true", help="Train for all iPhones")
        parser.add_argument("--days", type=int, default=14, help="Training window in days")
        parser.add_argument("--run-id", type=str, default="live", help="ClickHouse run_id")
        parser.add_argument("--version", type=str, default="v1", help="Model version tag")

    def handle(self, *args, **options):
        if options["all"]:
            iphone_ids = list(Iphone.objects.values_list("id", flat=True))
        elif options["iphone_ids"]:
            iphone_ids = [int(x.strip()) for x in options["iphone_ids"].split(",")]
        else:
            self.stderr.write("Specify --iphone-ids or --all")
            return

        ch = ClickHouseService()
        run_id = options["run_id"]
        days = options["days"]
        version = options["version"]

        self.stdout.write(
            f"Training models: {len(iphone_ids)} iPhones, {days} days, run_id={run_id}"
        )

        ok, fail = 0, 0
        for iid in iphone_ids:
            try:
                result = train_models_for_iphone(
                    ch, iid, run_id=run_id, train_days=days,
                )
                artifacts = save_artifacts(iid, result, version=version)
                self.stdout.write(
                    f"  iphone={iid}: OK ({len(artifacts)} artifacts saved)"
                )
                for h, m in result["metrics"].items():
                    self.stdout.write(
                        f"    h={h}min  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}"
                    )
                ok += 1
            except Exception as e:
                logger.exception("Failed training iphone=%d", iid)
                self.stderr.write(f"  iphone={iid}: FAILED - {e}")
                fail += 1

        self.stdout.write(f"\nDone: {ok} succeeded, {fail} failed")
