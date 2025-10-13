from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import transaction, connection
from django.core.exceptions import FieldDoesNotExist

PRESERVE_SHOP_NAME = "買取商店"

def get_model_by_class_name(cls_name: str):
    for m in apps.get_models():
        if m.__name__ == cls_name:
            return m
    return None

class Command(BaseCommand):
    help = f"Delete ALL PurchasingShopPriceRecord rows EXCEPT those with shop.name == '{PRESERVE_SHOP_NAME}'."

    def add_arguments(self, parser):
        parser.add_argument("--yes-i-really-mean-it", action="store_true", help="Confirmation flag.")
        parser.add_argument("--shop-name", default=PRESERVE_SHOP_NAME, help="Shop name to PRESERVE.")
        parser.add_argument("--fast", action="store_true", help="Use single SQL DELETE (skips signals).")

    def handle(self, *args, **opts):
        if not opts["yes_i_really_mean_it"]:
            self.stdout.write(self.style.ERROR("Refusing to run without --yes-i-really-mean-it"))
            return

        Model = get_model_by_class_name("PurchasingShopPriceRecord")
        if Model is None:
            self.stdout.write(self.style.ERROR("Model PurchasingShopPriceRecord not found in INSTALLED_APPS"))
            return

        shop_name = opts["shop_name"]

        if not opts["fast"]:
            total = Model.objects.count()
            keep = Model.objects.filter(shop__name=shop_name).count()
            with transaction.atomic():
                deleted, _ = Model.objects.exclude(shop__name=shop_name).delete()
            self.stdout.write(self.style.SUCCESS(
                f"✅ ORM 删除完成：总计 {total}，保留 {keep}（shop.name='{shop_name}'），删除 {deleted}。"
            ))
        else:
            # FAST: 原生 SQL，删除所有不属于指定店名的记录
            try:
                fk = Model._meta.get_field("shop")
            except FieldDoesNotExist:
                self.stdout.write(self.style.ERROR("Field 'shop' not found on PurchasingShopPriceRecord"))
                return
            ptab = Model._meta.db_table
            stab = fk.remote_field.model._meta.db_table
            fkcol = fk.column  # 一般是 'shop_id'

            sql = (
                f'DELETE FROM "{ptab}" p '
                f'WHERE NOT EXISTS ('
                f'  SELECT 1 FROM "{stab}" s '
                f'  WHERE s."id" = p."{fkcol}" AND s."name" = %s'
                f');'
            )
            with connection.cursor() as cur, transaction.atomic():
                cur.execute(sql, [shop_name])
                deleted = cur.rowcount  # 有的后端可能不精确
            self.stdout.write(self.style.SUCCESS(
                f"✅ FAST 删除完成：约删 {deleted} 行；已保留 shop.name='{shop_name}' 的记录。"
            ))
