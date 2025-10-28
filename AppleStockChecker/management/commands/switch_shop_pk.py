# AppleStockChecker/management/commands/switch_shop_pk.py
from django.core.management.base import BaseCommand, CommandError
from django.apps import apps
from django.db import transaction
from django.db.models import ForeignKey, OneToOneField
from django.db.models import Q

class Command(BaseCommand):
    help = "安全地把 source_pk 的 Shop 变成 target_pk（带冲突检测与可选的冲突删除）"

    def add_arguments(self, parser):
        parser.add_argument('--app', default='AppleStockChecker', help='app label (default AppleStockChecker)')
        parser.add_argument('--model', default='SecondHandShop', help='model name (default SecondHandShop)')
        parser.add_argument('--target', type=int, required=True, help='target pk to delete/replace (e.g. 19)')
        parser.add_argument('--source', type=int, required=True, help='source pk to become the new target (e.g. 21)')
        parser.add_argument('--force', action='store_true', help='actually perform changes (otherwise dry-run)')
        parser.add_argument('--remove-source', action='store_true', help='delete original source record after migration')
        parser.add_argument('--delete-conflicts', action='store_true', help='when unique conflicts are found, delete conflicting records before creating new target (use with --force)')

    def handle(self, *args, **options):
        app_label = options['app']
        model_name = options['model']
        target_pk = options['target']
        source_pk = options['source']
        do_execute = options['force']
        remove_source = options['remove_source']
        delete_conflicts = options['delete_conflicts']

        Model = apps.get_model(app_label, model_name)
        if Model is None:
            raise CommandError(f"找不到模型 {app_label}.{model_name}")

        self.stdout.write(self.style.NOTICE(f"Model: {app_label}.{model_name}"))
        self.stdout.write(self.style.NOTICE(f"目标 target_pk={target_pk}, 源 source_pk={source_pk}"))
        if not do_execute:
            self.stdout.write(self.style.WARNING("当前为 dry-run 模式：只打印计划，不会实际修改。使用 --force 才会执行。"))

        src_obj = Model.objects.filter(pk=source_pk).first()
        if not src_obj:
            raise CommandError(f"找不到源记录：{app_label}.{model_name} pk={source_pk}")

        # 收集指向该模型的 FK/O2O（仅声明在模型上的字段）
        referring = []
        for m in apps.get_models():
            for f in m._meta.get_fields():
                if isinstance(f, (ForeignKey, OneToOneField)) and f.remote_field and f.remote_field.model == Model:
                    if f.auto_created:
                        continue
                    referring.append((m, f))

        # 解析模型的唯一约束集合（单列 unique 与 unique_together）
        unique_sets = []
        for f in Model._meta.fields:
            if getattr(f, 'unique', False):
                unique_sets.append((f.name,))
        # unique_together may be a tuple of tuples
        ut = getattr(Model._meta, 'unique_together', None)
        if ut:
            # ensure iterable of tuples
            if isinstance(ut[0], (list, tuple)):
                for t in ut:
                    unique_sets.append(tuple(t))
            else:
                unique_sets.append(tuple(ut))

        # 打印计划
        self.stdout.write("\n计划步骤（概览）：")
        self.stdout.write("1) 将所有引用 target_pk 的 FK 更新为 source_pk（避免删除 target 时有外键引用）")
        self.stdout.write("2) 删除 target_pk（若存在）")
        self.stdout.write("3) 检查唯一约束冲突（unique fields / unique_together）")
        self.stdout.write("   - 如果发现冲突，dry-run 会列出来并停止。")
        self.stdout.write("   - 实际执行时可用 --delete-conflicts 一并删除冲突记录（谨慎）")
        self.stdout.write("4) 复制 source_pk 的数据创建新的记录 pk=target_pk")
        self.stdout.write("5) 将所有引用 source_pk 的 FK 更新为 target_pk")
        if remove_source:
            self.stdout.write("6) 删除原 source_pk（若指定 --remove-source）")

        self.stdout.write("\n将会影响的引用字段（模型.字段）：")
        for m, f in referring:
            self.stdout.write(f" - {m._meta.app_label}.{m._meta.model_name}.{f.name}")

        # 检查唯一约束冲突
        conflicts = []
        for uniq in unique_sets:
            q = Q()
            vals = {}
            for field_name in uniq:
                vals[field_name] = getattr(src_obj, field_name)
                q &= Q(**{field_name: getattr(src_obj, field_name)})
            # find other objects (exclude source itself) that match these unique values
            qs = Model.objects.exclude(pk=source_pk).filter(q)
            if qs.exists():
                for obj in qs:
                    conflicts.append((uniq, obj))
        if conflicts:
            self.stdout.write(self.style.ERROR("\n检测到唯一约束冲突（基于 source 的唯一字段）："))
            for uniq, obj in conflicts:
                self.stdout.write(f" - 冲突唯一组 {uniq} 与对象 pk={obj.pk}（值: {', '.join(str(getattr(obj, f)) for f in uniq)}）")
            if not do_execute:
                self.stdout.write(self.style.WARNING("\nDry-run: 因为存在唯一冲突，停止并请确认（或使用 --force --delete-conflicts 在确认后删除冲突）"))
                return
            # 如果在实际执行模式但没有指定删除冲突，直接拒绝以避免无意识的数据删除
            if do_execute and not delete_conflicts:
                raise CommandError("执行中检测到唯一冲突。若确认要删除冲突记录，请重新运行并加上 --delete-conflicts（同时带 --force）")
            # 若允许删除冲突，列出将被删除的对象（并删除）
            if delete_conflicts:
                self.stdout.write(self.style.WARNING("\n将删除以下冲突对象（因为使用了 --delete-conflicts）："))
                for uniq, obj in conflicts:
                    self.stdout.write(f" - 将删除 pk={obj.pk}，值: {', '.join(str(getattr(obj, f)) for f in uniq)}")
                if do_execute:
                    for uniq, obj in conflicts:
                        # 注意：删除可能会 cascade；这是用户授权的操作
                        obj.delete()
                    self.stdout.write(self.style.SUCCESS("冲突对象已删除。"))

        if not do_execute:
            self.stdout.write(self.style.SUCCESS("\nDry-run 完成：没有实际修改。"))
            return

        # 实际执行（事务内）
        with transaction.atomic():
            # 1) 更新引用 target -> source
            for RefModel, fk in referring:
                kw_from = {f"{fk.name}": target_pk}
                kw_to = {f"{fk.name}": source_pk}
                q = RefModel.objects.filter(**kw_from)
                count = q.count()
                if count:
                    self.stdout.write(f"更新 {RefModel._meta.app_label}.{RefModel._meta.model_name}.{fk.name} : {count} rows, {target_pk} -> {source_pk}")
                    q.update(**kw_to)

            # 2) 删除旧的 target（若存在）
            target_obj = Model.objects.filter(pk=target_pk).first()
            if target_obj:
                self.stdout.write(f"删除旧的 target {Model._meta.app_label}.{Model._meta.model_name} pk={target_pk}")
                target_obj.delete()
            else:
                self.stdout.write(f"未找到旧的 target (pk={target_pk})，继续下一步。")

            # 3) 复制 source -> 新 pk=target（此时冲突已处理或已删除）
            # 复制 concrete fields（跳过主键）
            field_values = {}
            for f in src_obj._meta.concrete_fields:
                if f.primary_key:
                    continue
                field_values[f.name] = getattr(src_obj, f.name)
            create_kwargs = dict(field_values)
            create_kwargs['pk'] = target_pk
            self.stdout.write(f"创建新的 {Model._meta.app_label}.{Model._meta.model_name} pk={target_pk}（复制自 pk={source_pk}）")
            new_obj = Model(**create_kwargs)
            new_obj.save(force_insert=True)

            # 4) 更新引用 source -> target
            for RefModel, fk in referring:
                kw_from = {f"{fk.name}": source_pk}
                kw_to = {f"{fk.name}": target_pk}
                q = RefModel.objects.filter(**kw_from)
                count = q.count()
                if count:
                    self.stdout.write(f"更新 {RefModel._meta.app_label}.{RefModel._meta.model_name}.{fk.name} : {count} rows, {source_pk} -> {target_pk}")
                    q.update(**kw_to)

            # 5) 可选删除 source
            if remove_source:
                self.stdout.write(f"删除源对象 {Model._meta.app_label}.{Model._meta.model_name} pk={source_pk}")
                Model.objects.filter(pk=source_pk).delete()

        self.stdout.write(self.style.SUCCESS("操作完成。"))
