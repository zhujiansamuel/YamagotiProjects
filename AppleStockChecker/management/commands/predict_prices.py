"""
管理命令: 对指定机型执行价格预测推理。

用法:
  python manage.py predict_prices --iphone-ids 1,2,3
  python manage.py predict_prices --all
  python manage.py predict_prices --iphone-ids 1 --run-id live
"""
from __future__ import annotations

import logging

from django.core.management.base import BaseCommand

from AppleStockChecker.engine.prediction import (
    predict_for_iphone,
    save_forecasts,
)
from AppleStockChecker.models import Iphone
from AppleStockChecker.services.clickhouse_service import ClickHouseService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run LightGBM price prediction inference for specified iPhones"

    def add_arguments(self, parser):
        parser.add_argument(
            "--iphone-ids",
            type=str,
            default="",
            help="Comma-separated iPhone IDs, or empty for --all",
        )
        parser.add_argument("--all", action="store_true", help="Predict for all iPhones")
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
        version = options["version"]

        self.stdout.write(f"Predicting: {len(iphone_ids)} iPhones, run_id={run_id}")

        ok, fail = 0, 0
        total_forecasts = 0
        for iid in iphone_ids:
            try:
                preds = predict_for_iphone(
                    ch, iid, run_id=run_id, version=version,
                )
                n = save_forecasts(preds)
                total_forecasts += len(preds)
                self.stdout.write(
                    f"  iphone={iid}: {len(preds)} predictions ({n} new)"
                )
                for p in preds:
                    self.stdout.write(
                        f"    h={p['horizon_min']}min  yhat={p['yhat']:.2f}"
                    )
                ok += 1
            except Exception as e:
                logger.exception("Failed prediction iphone=%d", iid)
                self.stderr.write(f"  iphone={iid}: FAILED - {e}")
                fail += 1

        self.stdout.write(
            f"\nDone: {ok} succeeded, {fail} failed, {total_forecasts} total forecasts"
        )
