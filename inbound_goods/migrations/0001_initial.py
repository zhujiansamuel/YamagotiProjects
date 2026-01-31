# -*- coding: utf-8 -*-
# Generated migration for inbound_goods app

from django.db import migrations, models
import django.db.models.deletion
import django.core.validators


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='InboundInventory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_object_id', models.PositiveIntegerField(help_text='关联商品的ID', verbose_name='商品ID')),
                ('unique_code', models.CharField(db_index=True, help_text='商品的唯一标识码，如序列号或内部编码', max_length=64, unique=True, validators=[django.core.validators.RegexValidator('^[A-Z0-9\\-_]+$', code='invalid_code', message='商品编码只能包含大写字母、数字、连字符和下划线')], verbose_name='唯一商品编码')),
                ('status', models.CharField(choices=[('CORPORATE_RESERVED_ARRIVAL', '法人预订到货'), ('PERSONAL_RESERVED_ARRIVAL', '个人预订到货'), ('PURCHASE_RESERVED_ARRIVAL', '购买预订到货'), ('IN_STOCK', '在库'), ('PREPARING_SHIPMENT', '出货准备'), ('SHIPPED', '已出货'), ('CANCELLED_RETURNED', '取消退回'), ('STATUS_ABNORMAL', '状态异常')], db_index=True, default='IN_STOCK', help_text='商品当前的在库状态', max_length=32, verbose_name='在库状态')),
                ('special_description', models.TextField(blank=True, default='', help_text='商品的特别说明或备注', verbose_name='特别描述')),
                ('reserved_arrival_time', models.DateTimeField(blank=True, db_index=True, help_text='预订商品的预计到货时间（仅适用于预订类状态）', null=True, verbose_name='预订到货时间')),
                ('abnormal_remark', models.TextField(blank=True, default='', help_text='状态异常时的详细说明（仅适用于状态异常）', verbose_name='状态异常备注')),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='创建时间')),
                ('updated_at', models.DateTimeField(auto_now=True, db_index=True, verbose_name='更新时间')),
                ('product_content_type', models.ForeignKey(help_text='商品类型（如：iPhone等）', limit_choices_to={'model__in': ['iphone']}, on_delete=django.db.models.deletion.PROTECT, to='contenttypes.contenttype', verbose_name='商品种类')),
            ],
            options={
                'verbose_name': '入库商品',
                'verbose_name_plural': '入库商品',
                'ordering': ['-created_at'],
                'indexes': [
                    models.Index(fields=['product_content_type', 'product_object_id'], name='inbound_goo_product_idx'),
                    models.Index(fields=['status', 'created_at'], name='inbound_goo_status_idx'),
                    models.Index(fields=['unique_code'], name='inbound_goo_unique_idx'),
                    models.Index(fields=['reserved_arrival_time'], name='inbound_goo_reserve_idx'),
                ],
            },
        ),
        migrations.CreateModel(
            name='InventoryStatusHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('old_status', models.CharField(blank=True, choices=[('CORPORATE_RESERVED_ARRIVAL', '法人预订到货'), ('PERSONAL_RESERVED_ARRIVAL', '个人预订到货'), ('PURCHASE_RESERVED_ARRIVAL', '购买预订到货'), ('IN_STOCK', '在库'), ('PREPARING_SHIPMENT', '出货准备'), ('SHIPPED', '已出货'), ('CANCELLED_RETURNED', '取消退回'), ('STATUS_ABNORMAL', '状态异常')], help_text='变更前的状态（为空表示初始创建）', max_length=32, null=True, verbose_name='旧状态')),
                ('new_status', models.CharField(choices=[('CORPORATE_RESERVED_ARRIVAL', '法人预订到货'), ('PERSONAL_RESERVED_ARRIVAL', '个人预订到货'), ('PURCHASE_RESERVED_ARRIVAL', '购买预订到货'), ('IN_STOCK', '在库'), ('PREPARING_SHIPMENT', '出货准备'), ('SHIPPED', '已出货'), ('CANCELLED_RETURNED', '取消退回'), ('STATUS_ABNORMAL', '状态异常')], help_text='变更后的状态', max_length=32, verbose_name='新状态')),
                ('changed_at', models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='变更时间')),
                ('change_reason', models.TextField(blank=True, default='', help_text='状态变更的原因或备注', verbose_name='变更原因')),
                ('changed_by', models.CharField(blank=True, default='', help_text='执行状态变更的操作人（系统/用户名）', max_length=128, verbose_name='操作人')),
                ('inventory', models.ForeignKey(db_index=True, on_delete=django.db.models.deletion.CASCADE, related_name='status_history', to='inbound_goods.inboundinventory', verbose_name='关联商品')),
            ],
            options={
                'verbose_name': '状态变更历史',
                'verbose_name_plural': '状态变更历史',
                'ordering': ['-changed_at'],
                'indexes': [
                    models.Index(fields=['inventory', '-changed_at'], name='inbound_goo_invento_idx'),
                    models.Index(fields=['new_status', '-changed_at'], name='inbound_goo_new_sta_idx'),
                    models.Index(fields=['-changed_at'], name='inbound_goo_changed_idx'),
                ],
            },
        ),
        migrations.AddConstraint(
            model_name='inboundinventory',
            constraint=models.CheckConstraint(check=models.Q(('status__in', ['CORPORATE_RESERVED_ARRIVAL', 'PERSONAL_RESERVED_ARRIVAL', 'PURCHASE_RESERVED_ARRIVAL']), _negated=True) | models.Q(('reserved_arrival_time__isnull', False)), name='chk_reserved_arrival_time'),
        ),
        migrations.AddConstraint(
            model_name='inboundinventory',
            constraint=models.CheckConstraint(check=models.Q(('status', 'STATUS_ABNORMAL'), _negated=True) | models.Q(('abnormal_remark', ''), _negated=True), name='chk_abnormal_remark'),
        ),
    ]
