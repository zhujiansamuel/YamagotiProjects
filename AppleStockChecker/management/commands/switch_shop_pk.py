# yourapp/management/commands/switch_shop_pk.py
from django.core.management.base import BaseCommand, CommandError
from django.apps import apps
from django.db import transaction, router, connections
from django.db.models import ForeignKey, OneToOneField
import sys

class Command(BaseCommand):
    help = (
        "Safely make the Shop with pk=SOURCE_PK become pk=TARGET_PK by:\n"
        " 1) reassign FK->TARGET_PK (from any existing TARGET_PK refs) to SOURCE_PK,\n"
        " 2) delete old TARGET_PK,\n"
        " 3) copy SOURCE_PK -> new TARGET_PK,\n"
        " 4) reassign FK that pointed to SOURCE_PK to TARGET_PK,\n        (optionally) delete SOURCE_PK.\n\n"
        "This avoids directly updating primary key values (which can break FK constraints).\n"
        "Default model: yourapp.Shop — override with --app and --model.\n"
        "Dry run by default. Use --force to execute, and --remove-source to delete the original source record."
    )

    def add_arguments(self, parser):
        parser.add_argument('--app', default='AppleStockChecker', help='app label (default: AppleStockChecker)')
        parser.add_argument('--model', default='SecondHandShop', help='model name (default: SecondHandShop)')
        parser.add_argument('--target', type=int, required=True, help='target pk to delete / replace (e.g. 19)')
        parser.add_argument('--source', type=int, required=True, help='source pk to become the new target (e.g. 21)')
        parser.add_argument('--force', action='store_true', help='actually perform changes (otherwise dry-run)')
        parser.add_argument('--remove-source', action='store_true', help='delete original source record after migration (default: false)')

    def handle(self, *args, **options):
        app_label = options['app']
        model_name = options['model']
        target_pk = options['target']   # the pk to be removed and replaced (19)
        source_pk = options['source']   # the pk whose data we want to end up at target (21)
        do_execute = options['force']
        remove_source = options['remove_source']

        Model = apps.get_model(app_label, model_name)
        if Model is None:
            raise CommandError(f"找不到模型 {app_label}.{model_name}")

        self.stdout.write(self.style.NOTICE(f"Model: {app_label}.{model_name}"))
        self.stdout.write(self.style.NOTICE(f"目标 target_pk={target_pk}, 源 source_pk={source_pk}"))
        if not do_execute:
            self.stdout.write(self.style.WARNING("当前为 dry-run 模式。使用 --force 才会真正执行更改。"))

        src_obj = Model.objects.filter(pk=source_pk).first()
        if not src_obj:
            raise CommandError(f"找不到源记录：{app_label}.{model_name} pk={source_pk}")

        # collect all FK/O2O fields that point to this model
        referring = []  # list of tuples (RefModel, field)
        for m in apps.get_models():
            for f in m._meta.get_fields():
                # We consider ForeignKey and OneToOneField fields declared on the model
                if isinstance(f, (ForeignKey, OneToOneField)) and f.remote_field and f.remote_field.model == Model:
                    # skip auto-created reverse relations
                    # f.auto_created True typically means reverse accessor; but for declared FK it's False.
                    if f.auto_created:
                        continue
                    referring.append((m, f))

        # Print plan
        self.stdout.write("\n计划步骤（概览）：")
        self.stdout.write("1) 将所有引用 target_pk 的 FK 更新为 source_pk（使删除 target 时不会有引用）")
        self.stdout.write("2) 删除 target_pk 的记录")
        self.stdout.write("3) 复制 source_pk 的数据创建新的记录 pk=target_pk")
        self.stdout.write("4) 将所有引用 source_pk 的 FK 更新为 target_pk")
        if remove_source:
            self.stdout.write("5) 删除原 source_pk 的记录（如果 --remove-source 指定）")
        self.stdout.write("\n将会影响的引用字段（模型.字段）：")
        for m, f in referring:
            self.stdout.write(f" - {m._meta.app_label}.{m._meta.model_name}.{f.name}")

        if not do_execute:
            self.stdout.write(self.style.SUCCESS("\nDry-run 完成。没有实际修改。"))
            return

        # Execute within a transaction
        with transaction.atomic():
            # 1) Update refs to target -> point to source
            for RefModel, fk in referring:
                kw_from = {f"{fk.name}": target_pk}
                kw_to = {f"{fk.name}": source_pk}
                q = RefModel.objects.filter(**kw_from)
                count = q.count()
                if count:
                    self.stdout.write(f"更新 {RefModel._meta.app_label}.{RefModel._meta.model_name}.{fk.name} : {count} rows, {target_pk} -> {source_pk}")
                    q.update(**kw_to)

            # 2) Delete old target record (if exists)
            target_obj = Model.objects.filter(pk=target_pk).first()
            if target_obj:
                self.stdout.write(f"删除旧的 target {Model._meta.app_label}.{Model._meta.model_name} pk={target_pk}")
                target_obj.delete()
            else:
                self.stdout.write(f"未找到旧的 target (pk={target_pk})，跳过删除。")

            # 3) Clone source object to new pk=target_pk
            # copy all concrete fields except AutoField PK if present
            field_names = []
            defaults = {}
            for f in src_obj._meta.concrete_fields:
                if f.primary_key:
                    continue
                field_names.append(f.name)
                defaults[f.name] = getattr(src_obj, f.name)
            # create new object with pk=target_pk
            self.stdout.write(f"创建新的 {Model._meta.app_label}.{Model._meta.model_name} pk={target_pk}（复制自 pk={source_pk}）")
            # Use bulk insert via Model(**kwargs) so any defaults/signals apply
            create_kwargs = dict(defaults)
            create_kwargs['pk'] = target_pk
            new_obj = Model(**create_kwargs)
            new_obj.save(force_insert=True)

            # 4) Update refs that point to source_pk -> point to target_pk
            for RefModel, fk in referring:
                kw_from = {f"{fk.name}": source_pk}
                kw_to = {f"{fk.name}": target_pk}
                q = RefModel.objects.filter(**kw_from)
                count = q.count()
                if count:
                    self.stdout.write(f"更新 {RefModel._meta.app_label}.{RefModel._meta.model_name}.{fk.name} : {count} rows, {source_pk} -> {target_pk}")
                    q.update(**kw_to)

            # 5) Optionally delete the original source object
            if remove_source:
                self.stdout.write(f"删除源对象 {Model._meta.app_label}.{Model._meta.model_name} pk={source_pk}")
                Model.objects.filter(pk=source_pk).delete()

        self.stdout.write(self.style.SUCCESS("操作完成。"))
