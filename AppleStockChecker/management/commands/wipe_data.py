# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from typing import Iterable, List

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
from django.core.management.color import no_style
from django.db import connections, DEFAULT_DB_ALIAS, transaction, IntegrityError
from django.db.models.deletion import ProtectedError

APP_LABEL = "AppleStockChecker"


def get_app_models():
    return list(apps.get_app_config(APP_LABEL).get_models())


class Command(BaseCommand):
    help = (
        "清空数据库记录。\n"
        "  --scope app   仅清空 AppleStockChecker 应用的数据（默认）\n"
        "  --scope all   清空整个数据库（相当于 manage.py flush）⚠️谨慎\n"
        "可选：--dry-run 仅预览; --force 跳过确认; --reset-seq 重置自增; --database 指定库"
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--scope",
            choices=["app", "all"],
            default="app",
            help="清空范围：仅本应用(app) 或 全库(all)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="仅打印将被清空的表/模型，不实际删除",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="无需确认直接执行",
        )
        parser.add_argument(
            "--reset-seq",
            action="store_true",
            help="清空后重置（本应用表的）自增序列/identity（app 模式有效）",
        )
        parser.add_argument(
            "--database",
            default=DEFAULT_DB_ALIAS,
            help="数据库别名，默认 default",
        )

    def handle(self, *args, **opts):
        scope = opts["scope"]
        dry_run = opts["dry_run"]
        force = opts["force"]
        reset_seq = opts["reset_seq"]
        alias = opts["database"]
        conn = connections[alias]
        engine = conn.settings_dict.get("ENGINE", "")

        if scope == "all":
            self._confirm_or_abort(
                force,
                "⚠️ 将清空整个数据库并重置主键！包括管理员账号、权限数据等。\n"
                "等价于：manage.py flush --no-input\n"
                "确定继续？ [y/N] ",
            )
            if dry_run:
                self.stdout.write(self.style.NOTICE("[dry-run] 将执行 flush（全库清空）"))
                return
            call_command("flush", interactive=False, database=alias)
            self.stdout.write(self.style.SUCCESS("✔ 已执行全库 flush"))
            return

        # scope = app
        models = get_app_models()
        if not models:
            raise CommandError(f"未找到应用 {APP_LABEL} 的模型")

        table_names = [m._meta.db_table for m in models]
        self.stdout.write(self.style.NOTICE(f"目标应用：{APP_LABEL}"))
        self.stdout.write(self.style.NOTICE("涉及表：\n  - " + "\n  - ".join(table_names)))

        if dry_run:
            self.stdout.write(self.style.SUCCESS("[dry-run] 仅预览，不会删除任何数据"))
            return

        self._confirm_or_abort(
            force,
            f"⚠️ 将清空 {APP_LABEL} 的所有业务数据（{len(models)} 个表）。是否继续？ [y/N] ",
        )

        # 优先用 PostgreSQL 的 TRUNCATE，最快
        if "postgresql" in engine:
            self._truncate_postgres(conn, table_names)
            self.stdout.write(self.style.SUCCESS("✔ PostgreSQL TRUNCATE 成功"))
        else:
            # 其它数据库：多轮 ORM 级联删除，自动跳过 PROTECT 的父表，直到干净或失败
            self._delete_via_orm(alias, models)
            self.stdout.write(self.style.SUCCESS("✔ ORM 级联删除成功"))

        if reset_seq:
            self._reset_sequences(conn, models)
            self.stdout.write(self.style.SUCCESS("✔ 已重置自增序列"))

    # ---------- helpers ----------

    def _confirm_or_abort(self, force: bool, prompt: str):
        if force:
            return
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            ans = "n"
        if ans not in {"y", "yes"}:
            raise CommandError("已取消。")

    @transaction.atomic
    def _truncate_postgres(self, conn, table_names: List[str]):
        # 注意：TRUNCATE 会要求引用关系可被 CASCADE 处理；我们一次性传入所有表名并使用 CASCADE
        quoted = [conn.ops.quote_name(t) for t in table_names]
        sql = f"TRUNCATE TABLE {', '.join(quoted)} RESTART IDENTITY CASCADE;"
        with conn.cursor() as cur:
            cur.execute(sql)

    def _delete_via_orm(self, alias: str, models: List[type]):
        """
        逐模型 .objects.all().delete()，遇到 PROTECT/完整性错误就先跳过，做多轮直到清空或无进展。
        这样无需手写依赖顺序（如 InventoryRecord → OfficialStore）。
        """
        remaining = set(models)
        round_no = 0
        while remaining:
            round_no += 1
            progressed = False
            for m in list(remaining):
                try:
                    with transaction.atomic(using=alias):
                        deleted, _ = m.objects.using(alias).all().delete()
                    self.stdout.write(self.style.NOTICE(f"第{round_no}轮：{m.__name__} 删除 {deleted} 行"))
                    remaining.remove(m)
                    progressed = True
                except (ProtectedError, IntegrityError) as e:
                    # 先删除依赖它的子表，下一轮再试
                    continue
            if not progressed:
                names = ", ".join(sorted(x.__name__ for x in remaining))
                raise CommandError(f"无法继续删除，可能存在 PROTECT 或循环依赖：{names}")

    def _reset_sequences(self, conn, models: List[type]):
        # 仅对当前应用的表重置序列/identity
        sql_list = conn.ops.sequence_reset_sql(no_style(), models)
        if not sql_list:
            return
        with conn.cursor() as cur:
            for sql in sql_list:
                cur.execute(sql)
