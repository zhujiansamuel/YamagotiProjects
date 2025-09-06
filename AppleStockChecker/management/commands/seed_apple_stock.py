# -*- coding: utf-8 -*-
"""
用法示例：
    python manage.py seed_apple_stock \
        --stores 8 --variants 24 --records-per-store 80 --days 7 \
        --clear --seed 42

不带参数时使用合理默认值。
"""
from __future__ import annotations

import random
import string
from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from AppleStockChecker.models import Iphone, OfficialStore, InventoryRecord


MODEL_CATALOG = [
    # (model_name, colors, release_date)
    ("iPhone 15", ["黑色", "蓝色", "粉色", "绿色", "黄色"], date(2023, 9, 22)),
    ("iPhone 15 Plus", ["黑色", "蓝色", "粉色", "绿色", "黄色"], date(2023, 9, 22)),
    ("iPhone 15 Pro", ["黑钛", "白钛", "蓝钛", "天然钛"], date(2023, 9, 22)),
    ("iPhone 15 Pro Max", ["黑钛", "白钛", "蓝钛", "天然钛"], date(2023, 9, 22)),
    ("iPhone 16", ["黑色", "白色", "粉色", "蓝色", "绿色"], date(2024, 9, 20)),
    ("iPhone 16 Plus", ["黑色", "白色", "粉色", "蓝色", "绿色"], date(2024, 9, 20)),
    ("iPhone 16 Pro", ["黑钛", "白钛", "沙岩色", "天然钛"], date(2024, 9, 20)),
    ("iPhone 16 Pro Max", ["黑钛", "白钛", "沙岩色", "天然钛"], date(2024, 9, 20)),
]
CAPACITIES = [128, 256, 512, 1024]  # GB

JP_STORES = [
    "Apple 銀座", "Apple 表参道", "Apple 新宿", "Apple 丸の内",
    "Apple 心斎橋", "Apple 名古屋栄", "Apple 京都", "Apple 川崎",
    "Apple 福岡", "Apple 札幌", "Apple 渋谷", "Apple 二子玉川",
]


def pn_generator():
    """
    简单的日本区 PN 生成器（示例）：
    由 4 个前缀字母 + 3 位序号 + 'J/A' 组成，例如：MTQX005J/A
    """
    letters = "MTNPQRSUVWXYZ"
    counter = 5  # 起始序号，避免太短
    while True:
        prefix = "".join(random.choice(letters) for _ in range(4))
        yield f"{prefix}{counter:03d}J/A"
        counter += 1


def tb_label(gb: int) -> str:
    return f"{gb // 1024}TB" if gb % 1024 == 0 else f"{gb}GB"


class Command(BaseCommand):
    help = "生成 AppleStockChecker 的模拟数据（门店 / iPhone 变体 / 库存记录）"

    def add_arguments(self, parser):
        parser.add_argument("--stores", type=int, default=6, help="创建的门店数量")
        parser.add_argument("--variants", type=int, default=16, help="创建的 iPhone 变体数量（型号×容量×颜色 的去重组合）")
        parser.add_argument("--records-per-store", type=int, default=60, help="每个门店要创建的库存记录数量")
        parser.add_argument("--days", type=int, default=5, help="记录分布在最近 N 天内")
        parser.add_argument("--clear", action="store_true", help="生成前清空现有数据（本应用内的三张表）")
        parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")

    @transaction.atomic
    def handle(self, *args, **opts):
        if opts["seed"] is not None:
            random.seed(opts["seed"])

        if opts["clear"]:
            self.stdout.write(self.style.WARNING("清空旧数据…"))
            InventoryRecord.objects.all().delete()
            OfficialStore.objects.all().delete()
            Iphone.objects.all().delete()

        stores = self._create_stores(opts["stores"])
        iphones = self._create_iphone_variants(opts["variants"])

        self.stdout.write(self.style.SUCCESS(f"✔ 门店 {len(stores)} 家，iPhone 变体 {len(iphones)} 个"))

        self._create_inventory_records(
            stores=stores,
            iphones=iphones,
            per_store=opts["records_per_store"],
            days=opts["days"],
        )

        self.stdout.write(self.style.SUCCESS("完成！可在 /AppleStockChecker/stores、/AppleStockChecker/iphones、/AppleStockChecker/inventory-records 查看。"))

    # ---------- helpers ----------

    def _create_stores(self, n: int) -> list[OfficialStore]:
        names = JP_STORES[:]
        random.shuffle(names)
        if n > len(names):
            # 不够就拼接序号
            extra = [f"Apple 门店 #{i}" for i in range(n - len(names))]
            names.extend(extra)
        names = names[:n]

        stores = []
        for i, name in enumerate(names, 1):
            addr = f"日本 东京都 XX 区 YY 路 {100+i} 号"
            store, _ = OfficialStore.objects.get_or_create(name=name, defaults={"address": addr})
            stores.append(store)
        return stores

    def _create_iphone_variants(self, target: int) -> list[Iphone]:
        # 生成 (model_name, color, capacity, release_date) 的组合，直到满足 target
        combos: list[tuple[str, str, int, date]] = []
        for model_name, colors, rdate in MODEL_CATALOG:
            for color in colors:
                for cap in CAPACITIES:
                    combos.append((model_name, color, cap, rdate))
        random.shuffle(combos)
        combos = combos[:target]

        gen = pn_generator()
        created: list[Iphone] = []
        used = set()
        for model_name, color, cap, rdate in combos:
            # 确保 (model_name, color, cap) 组合唯一
            key = (model_name, color, cap)
            if key in used:
                continue
            used.add(key)

            # 生成唯一 PN
            part_number = next(gen)
            while Iphone.objects.filter(part_number=part_number).exists():
                part_number = next(gen)

            obj, _ = Iphone.objects.get_or_create(
                model_name=model_name,
                color=color,
                capacity_gb=cap,
                defaults={
                    "part_number": part_number,
                    "release_date": rdate,
                },
            )
            created.append(obj)
        return created

    def _create_inventory_records(self, stores, iphones, per_store: int, days: int):
        now = timezone.now()
        total = 0
        for store in stores:
            for _ in range(per_store):
                phone = random.choice(iphones)

                # 记录时间：最近 N 天内随机
                rec_time = now - timedelta(
                    days=random.randint(0, max(0, days)),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )

                # 在库状态：更贴近现实的权重（40% 有货 / 60% 无货）
                has_stock = random.random() < 0.40

                # 预计到达窗口（覆盖多种情况）
                eta_earliest, eta_latest = self._random_arrival_window(now, has_stock)

                # 创建（注意 recorded_at 使用 auto_now_add，需要二次更新）
                rec = InventoryRecord.objects.create(
                    store=store,
                    iphone=phone,
                    has_stock=has_stock,
                    estimated_arrival_earliest=eta_earliest,
                    estimated_arrival_latest=eta_latest,
                )
                # 覆盖 recorded_at（绕过 auto_now_add 的 pre_save）
                InventoryRecord.objects.filter(pk=rec.pk).update(recorded_at=rec_time)
                total += 1

        self.stdout.write(self.style.SUCCESS(f"✔ 已生成库存记录 {total} 条"))

    def _random_arrival_window(self, now, has_stock: bool):
        """
        返回 (earliest, latest)：
         - 有货：大多不设置 ETA（None, None），也可能设置 “稍后到达/自提准备中”
         - 无货：设置未来 4h ~ 10 天的窗口；也随机产生“只有最早”或“只有最晚”的情况
        """
        # 20% 只给 earliest，10% 只给 latest，70% 给完整窗口（无货时）
        only_earliest = random.random() < 0.20
        only_latest = (not only_earliest) and (random.random() < 0.10)

        if has_stock:
            mode = random.random()
            if mode < 0.70:
                # 70%：现货，无 ETA
                return (None, None)
            elif mode < 0.85:
                # 15%：预计稍后可提（1~12 小时内）
                e = now + timedelta(hours=random.randint(1, 12))
                return (e, e)
            else:
                # 15%：小窗口补货（2~24 小时）
                e = now + timedelta(hours=random.randint(2, 24))
                l = e + timedelta(hours=random.randint(1, 6))
                return (e, l)

        # 无货：未来 4 小时 ~ 10 天
        e = now + timedelta(hours=random.randint(4, 24 * 10))
        # 两种窗口长度：短窗口（1~12h）或长窗口（1~3d）
        if random.random() < 0.6:
            l = e + timedelta(hours=random.randint(1, 12))
        else:
            l = e + timedelta(days=random.randint(1, 3), hours=random.randint(0, 8))

        if only_earliest:
            return (e, None)
        if only_latest:
            return (None, l)
        return (e, l)