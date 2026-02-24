"""
management command: write_psta_alignment

回补 PurchasingShopPriceRecord → PurchasingShopTimeAnalysis 时间对齐写入。
仅供开发阶段使用，直接调用 timestamp_alignment_task 内部函数。

时间戳生成规则：
    T₁ = floor(start) + 14min  → 覆盖 [start,    start+14min]
    T₂ = T₁ + 15min            → 覆盖 [start+15, start+29min]
    ...
    停止条件：T_n - 14min > floor(end)

用法示例：
    python manage.py write_psta_alignment \\
        --from 2024-01-15T09:00:00+09:00 \\
        --to   2024-01-15T11:00:00+09:00

    # 只查看不写入
    python manage.py write_psta_alignment \\
        --from 2024-01-15T09:00:00+09:00 \\
        --to   2024-01-15T11:00:00+09:00 \\
        --dry-run

    # 过滤范围
    python manage.py write_psta_alignment \\
        --from 2024-01-15T09:00:00+09:00 \\
        --to   2024-01-15T11:00:00+09:00 \\
        --shop-ids 1,2 --iphone-ids 3,4
"""
from __future__ import annotations

import uuid
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError


def _floor_minute(dt):
    """截断到整分钟（aware datetime）。"""
    return dt.replace(second=0, microsecond=0)


def _generate_timestamps(start_dt, end_dt) -> list:
    """
    生成覆盖 [start_dt, end_dt] 的时间戳序列，步长 15 分钟。

    每个 T 作为 collect_items_for_psta 的参考时间，其窗口为 [T-14min, T]。
    T₁ = floor(start) + 14min，后续每步 +15min，
    直到 T_n - 14min > floor(end) 为止。
    """
    t = _floor_minute(start_dt) + timedelta(minutes=14)
    end_floor = _floor_minute(end_dt)
    timestamps = []
    while t - timedelta(minutes=14) <= end_floor:
        timestamps.append(t)
        t += timedelta(minutes=15)
    return timestamps


class Command(BaseCommand):
    help = "回补 PurchasingShopPriceRecord → PurchasingShopTimeAnalysis（仅开发用）"

    def add_arguments(self, parser):
        parser.add_argument(
            "--from",
            dest="dt_from",
            required=True,
            help="回补区间起点，ISO 格式，例如 2024-01-15T09:00:00+09:00",
        )
        parser.add_argument(
            "--to",
            dest="dt_to",
            required=True,
            help="回补区间终点，ISO 格式，例如 2024-01-15T11:00:00+09:00",
        )
        parser.add_argument(
            "--job-id",
            default=None,
            help="Job ID 标识（默认自动生成 uuid4）",
        )
        parser.add_argument(
            "--shop-ids",
            default=None,
            help="限定 Shop ID，逗号分隔，例如 1,2,3",
        )
        parser.add_argument(
            "--iphone-ids",
            default=None,
            help="限定 iPhone ID，逗号分隔，例如 1,2,3",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="只打印会写入的记录，不实际写入数据库",
        )

    def handle(self, *args, **options):
        # 延迟导入：直接引用私有函数，仅开发用
        from AppleStockChecker.tasks.timestamp_alignment_task import (
            _process_minute_rows,
            _to_aware,
        )
        from AppleStockChecker.utils.timestamp_alignment_task import collect_items_for_psta

        dry_run: bool = options["dry_run"]
        job_id: str = options["job_id"] or str(uuid.uuid4())

        # ── 解析起止时间 ───────────────────────────────────────────────────
        try:
            dt_from = _to_aware(options["dt_from"])
            dt_to = _to_aware(options["dt_to"])
        except ValueError as e:
            raise CommandError(f"时间格式错误: {e}")

        if dt_from >= dt_to:
            raise CommandError("--from 必须早于 --to")

        shop_ids = None
        if options["shop_ids"]:
            try:
                shop_ids = [int(x) for x in options["shop_ids"].split(",")]
            except ValueError as e:
                raise CommandError(f"--shop-ids 格式错误: {e}")

        iphone_ids = None
        if options["iphone_ids"]:
            try:
                iphone_ids = [int(x) for x in options["iphone_ids"].split(",")]
            except ValueError as e:
                raise CommandError(f"--iphone-ids 格式错误: {e}")

        # ── 生成时间戳序列 ─────────────────────────────────────────────────
        timestamps = _generate_timestamps(dt_from, dt_to)

        if dry_run:
            self.stdout.write(self.style.WARNING("*** DRY-RUN 模式：不写入数据库 ***\n"))

        self.stdout.write(self.style.NOTICE(
            f"job_id={job_id}\n"
            f"区间: {options['dt_from']} → {options['dt_to']}\n"
            f"共生成 {len(timestamps)} 个时间戳（每 15 分钟一个）\n"
        ))

        total_ok = 0
        total_failed = 0

        # ── 逐时间戳处理 ──────────────────────────────────────────────────
        for batch_idx, ts_dt in enumerate(timestamps, 1):
            ts_iso = ts_dt.isoformat(timespec="seconds")

            self.stdout.write(f"[{batch_idx}/{len(timestamps)}] 时间戳 {ts_iso}")

            results = collect_items_for_psta(
                timestamp_iso=ts_iso,
                shop_ids=shop_ids,
                iphone_ids=iphone_ids,
            )

            if not results:
                self.stdout.write(self.style.WARNING("  collect_items_for_psta 返回空，跳过"))
                continue

            data = results[0]
            ticks: list[str] = data["ticks"]
            rows: list[dict] = data["rows"]
            bucket_by_minute: dict[str, list[int]] = data["bucket_by_minute"]

            self.stdout.write(
                f"  窗口: {data['window']['start']} → {data['window']['end']}  "
                f"原始记录: {len(rows)} 条"
            )

            # ── 逐刻度对齐写入 ─────────────────────────────────────────
            for tick_iso in ticks:
                tick_indices = bucket_by_minute.get(tick_iso, [])
                if not tick_indices:
                    continue

                tick_rows = [rows[i] for i in tick_indices]

                if dry_run:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  [DRY-RUN] [{tick_iso}] 将写入 {len(tick_rows)} 条"
                        )
                    )
                    for r in tick_rows:
                        self.stdout.write(
                            f"    shop_id={r.get('shop_id')}  "
                            f"iphone_id={r.get('iphone_id')}  "
                            f"recorded_at={r.get('recorded_at')}  "
                            f"price_new={r.get('price_new')}"
                        )
                    continue

                ok, failed, _err_counter, errors, _ = _process_minute_rows(
                    ts_iso=tick_iso,
                    ts_dt=_to_aware(tick_iso),
                    rows=tick_rows,
                    job_id=job_id,
                )
                total_ok += ok
                total_failed += failed

                if not failed:
                    self.stdout.write(
                        self.style.SUCCESS(f"  [{tick_iso}] OK  ok={ok}")
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  [{tick_iso}] PARTIAL  ok={ok}  failed={failed}"
                        )
                    )
                    for e in errors[:3]:
                        self.stdout.write(f"    ERR {e['exc']}: {e['msg']}")

        # ── 汇总 ──────────────────────────────────────────────────────────
        if dry_run:
            self.stdout.write(self.style.WARNING("\nDRY-RUN 完成，未写入任何数据"))
            return

        summary_style = self.style.SUCCESS if total_failed == 0 else self.style.WARNING
        self.stdout.write(summary_style(
            f"\n完成  total_ok={total_ok}  total_failed={total_failed}"
        ))
