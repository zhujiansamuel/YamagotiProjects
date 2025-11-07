# AppleStockChecker/management/commands/preview_external_csv.py
from __future__ import annotations
import json
from django.core.management.base import BaseCommand
from django.conf import settings
from AppleStockChecker.services.external_ingest_service import ingest_external_sources

class Command(BaseCommand):
    """
    目前已经不用了
    """
    help = "预览从外部平台拉取 CSV 并清洗后的前 10 行（不入库）"

    def add_arguments(self, parser):
        parser.add_argument("--names", nargs="*", default=[], help="指定 settings.EXTERNAL_TRADEIN_SOURCES 里的 name 子集")

    def handle(self, *args, **opts):
        names = opts["names"]
        preset = getattr(settings, "EXTERNAL_TRADEIN_SOURCES", [])
        if names:
            by = {it["name"]: it for it in preset}
            sources = [by[n] for n in names if n in by]
        else:
            sources = preset

        res = ingest_external_sources(sources, dry_run=True)
        self.stdout.write(json.dumps(res, ensure_ascii=False, indent=2))
