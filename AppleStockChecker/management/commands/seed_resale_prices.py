# -*- coding: utf-8 -*-
"""
生成“二手店 + 回收价格记录”模拟数据。

用法示例：
  # 基本用法：店铺 6 家，每店 80 条，最近 7 天
  python manage.py seed_resale_prices --shops 6 --records-per-shop 80 --days 7

  # 清空旧数据后重建，并固定随机种子（便于复现）
  python manage.py seed_resale_prices --clear --seed 42

  # 当库内 iPhone 太少时自动补 16 个变体
  python manage.py seed_resale_prices --autofill-iphones 16
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from AppleStockChecker.models import (
    Iphone,
    SecondHandShop,
    PurchasingShopPriceRecord,
)

# —— 一些现实里的二手店品牌名（示例） —— #
SHOP_CATALOG = [
    ("イオシス", "https://iosys.co.jp"),
    ("じゃんぱら", "https://www.janpara.co.jp"),
    ("ソフマップ", "https://www.sofmap.com"),
    ("ブックオフ", "https://www.bookoff.co.jp"),
    ("ゲオモバイル", "https://geo-mobile.jp"),
    ("ハードオフ", "https://www.hardoff.co.jp"),
    ("買取一丁目", "https://kaitori-1.jp"),
    ("Rmobile", "https://example.com"),
]

JP_ADDR_SAMPLES = [
    "東京都新宿区西新宿 1-1-1",
    "東京都千代田区丸の内 2-2-2",
    "東京都渋谷区神南 3-3-3",
    "大阪府大阪市中央区難波 4-4-4",
    "神奈川県川崎市川崎区駅前本町 5-5-5",
    "愛知県名古屋市中区栄 6-6-6",
    "京都府京都市中京区寺町通 7-7-7",
    "北海道札幌市中央区南一条西 8-8-8",
]

# 若库内 iPhone 不足，可用该小目录快速补齐（与前述项目模型一致）
MODEL_CATALOG = [
    ("iPhone 16 Pro", ["黑钛", "白钛", "沙岩色", "天然钛"], date(2024, 9, 20)),
    ("iPhone 16", ["黑色", "白色", "粉色", "蓝色", "绿色"], date(2024, 9, 20)),
    ("iPhone 15 Pro", ["黑钛", "白钛", "蓝钛", "天然钛"], date(2023, 9, 22)),
    ("iPhone 15", ["黑色", "蓝色", "粉色", "绿色", "黄色"], date(2023, 9, 22)),
]
CAPACITIES = [128, 256, 512, 1024]  # GB


def pn_generator():
    """非常简化的日本区 PN 生成器（示例），保证唯一即可。"""
    letters = "MTNPQRSUVWXYZ"
    counter = 100
    while True:
        prefix = "".join(random.choice(letters) for _ in range(4))
        yield f"{prefix}{counter:03d}J/A"
        counter += 1


@dataclass
class Options:
    shops: int
    records_per_shop: int
    days: int
    clear: bool
    seed: int | None
    autofill_iphones: int | None


class Command(BaseCommand):
    help = "生成二手店与回收价格记录的模拟数据"

    def add_arguments(self, parser):
        parser.add_argument("--shops", type=int, default=6, help="创建的二手店数量")
        parser.add_argument("--records-per-shop", type=int, default=80, help="每家店生成的价格记录数")
        parser.add_argument("--days", type=int, default=7, help="记录分布在最近 N 天内")
        parser.add_argument("--clear", action="store_true", help="生成前清空现有二手店与价格记录")
        parser.add_argument("--seed", type=int, default=None, help="随机种子（便于复现）")
        parser.add_argument(
            "--autofill-iphones",
            type=int,
            default=None,
            help="若现有 iPhone 数量不足，则自动补充到该数量（生成简单变体与 PN）",
        )

    @transaction.atomic
    def handle(self, *args, **opts):
        options = Options(
            shops=opts["shops"],
            records_per_shop=opts["records_per_shop"],
            days=opts["days"],
            clear=opts["clear"],
            seed=opts["seed"],
            autofill_iphones=opts["autofill_iphones"],
        )

        if options.seed is not None:
            random.seed(options.seed)

        if options.clear:
            self.stdout.write(self.style.WARNING("清空旧数据（价格记录 + 二手店）…"))
            PurchasingShopPriceRecord.objects.all().delete()
            SecondHandShop.objects.all().delete()

        # 确保 iPhone 变体充足
        self._ensure_iphones(options.autofill_iphones)

        shops = self._create_shops(options.shops)
        iphones = list(Iphone.objects.all())
        if not iphones:
            self.stderr.write(self.style.ERROR("库内没有 iPhone 变体，请先导入或使用 --autofill-iphones"))
            return

        self.stdout.write(self.style.SUCCESS(f"✔ 二手店 {len(shops)} 家，iPhone 变体 {len(iphones)} 个"))

        total = self._create_price_records(
            shops=shops,
            iphones=iphones,
            per_shop=options.records_per_shop,
            days=options.days,
        )
        self.stdout.write(self.style.SUCCESS(f"完成！共生成价格记录 {total} 条。"))
        self.stdout.write(self.style.SUCCESS("接口验证：/api/secondhand-shops/ 与 /api/purchasing-price-records/"))

    # ---------- helpers ----------

    def _ensure_iphones(self, target: int | None):
        if target is None:
            return
        current = Iphone.objects.count()
        if current >= target:
            return

        need = target - current
        self.stdout.write(self.style.WARNING(f"现有 iPhone {current} 个，自动补充 {need} 个…"))

        gen = pn_generator()
        created = 0
        for model_name, colors, rdate in MODEL_CATALOG:
            for color in colors:
                for cap in CAPACITIES:
                    if created >= need:
                        break
                    pn = next(gen)
                    # 避免 PN 冲突
                    while Iphone.objects.filter(part_number=pn).exists():
                        pn = next(gen)
                    Iphone.objects.create(
                        part_number=pn,
                        model_name=model_name,
                        capacity_gb=cap,
                        color=color,
                        release_date=rdate,
                    )
                    created += 1
                if created >= need:
                    break
            if created >= need:
                break

        self.stdout.write(self.style.SUCCESS(f"✔ 已补充 iPhone 变体 {created} 个"))

    def _create_shops(self, n: int) -> List[SecondHandShop]:
        # 先乱序取常见品牌，不够则补“某某二手店 #n”
        base = SHOP_CATALOG[:]
        random.shuffle(base)
        rows = base[: min(n, len(base))]
        while len(rows) < n:
            rows.append((f"某某二手店 #{len(rows)+1}", "https://example.com"))
        shops: List[SecondHandShop] = []
        for i in range(n):
            name, site = rows[i]
            addr = JP_ADDR_SAMPLES[i % len(JP_ADDR_SAMPLES)]
            obj, _ = SecondHandShop.objects.get_or_create(
                name=name,
                address=addr,
                defaults={"website": site},
            )
            shops.append(obj)
        return shops

    # —— 价格生成逻辑 —— #
    def _estimate_new_price(self, iphone: Iphone) -> int:
        """
        粗略估价：基于容量与上市时间衰减，单位：JPY。
        仅为模拟用途，非真实行情。
        """
        cap = iphone.capacity_gb
        # 容量基价（可按需微调）
        cap_base = {128: 85000, 256: 98000, 512: 118000, 1024: 148000}.get(cap, 90000)

        # 上市时间折旧：每月 -3%，最多打五折
        today = date.today()
        months = max(0, (today.year - iphone.release_date.year) * 12 + (today.month - iphone.release_date.month))
        depreciation = 1.0 - min(0.50, months * 0.03)

        # 轻微型号/颜色偏置（模拟不同受欢迎度）
        bias = 1.0 + random.uniform(-0.05, 0.08)

        price = int(cap_base * depreciation * bias)
        # 抖动 ±2k
        price += random.randint(-2000, 2000)
        return max(5000, price)

    def _create_price_records(self, shops: List[SecondHandShop], iphones: List[Iphone], per_shop: int, days: int) -> int:
        now = timezone.now()
        total = 0
        for shop in shops:
            for _ in range(per_shop):
                phone = random.choice(iphones)

                # 记录时间散布在最近 N 天
                rec_time = now - timedelta(
                    days=random.randint(0, max(0, days)),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )

                # 新品价必填：基于估价并加入波动
                new_price = max(3000, int(self._estimate_new_price(phone) * random.uniform(0.95, 1.05)))

                # A/B 可空：设置为空或按比率折价
                # 80% 有 A 品；65% 有 B 品
                a_price = None
                b_price = None
                if random.random() < 0.80:
                    a_price = max(2000, int(new_price * random.uniform(0.88, 0.95)))
                if random.random() < 0.65:
                    b_price = max(1500, int(new_price * random.uniform(0.75, 0.88)))

                rec = PurchasingShopPriceRecord.objects.create(
                    shop=shop,
                    iphone=phone,
                    price_new=new_price,
                    price_grade_a=a_price,
                    price_grade_b=b_price,
                )
                # 覆盖 recorded_at（因为模型里 auto_now_add=True）
                PurchasingShopPriceRecord.objects.filter(pk=rec.pk).update(recorded_at=rec_time)
                total += 1
        return total