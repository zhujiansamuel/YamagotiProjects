# -*- coding: utf-8 -*-
from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.validators import RegexValidator
from django.utils import timezone


class InboundInventory(models.Model):
    """
    入库商品管理主模型
    用于记录各种类型商品的在库状态和信息
    """

    class InventoryStatus(models.TextChoices):
        """在库状态选择"""
        CORPORATE_RESERVED_ARRIVAL = "CORPORATE_RESERVED_ARRIVAL", "法人预订到货"
        PERSONAL_RESERVED_ARRIVAL = "PERSONAL_RESERVED_ARRIVAL", "个人预订到货"
        PURCHASE_RESERVED_ARRIVAL = "PURCHASE_RESERVED_ARRIVAL", "购买预订到货"
        IN_STOCK = "IN_STOCK", "在库"
        PREPARING_SHIPMENT = "PREPARING_SHIPMENT", "出货准备"
        SHIPPED = "SHIPPED", "已出货"
        CANCELLED_RETURNED = "CANCELLED_RETURNED", "取消退回"
        STATUS_ABNORMAL = "STATUS_ABNORMAL", "状态异常"

    # 商品种类 - 使用 GenericForeignKey 支持多种商品类型
    product_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.PROTECT,
        verbose_name="商品种类",
        help_text="商品类型（如：iPhone等）",
        limit_choices_to={'model__in': ['iphone']},  # 目前只支持 iPhone，将来可扩展
    )
    product_object_id = models.PositiveIntegerField(
        verbose_name="商品ID",
        help_text="关联商品的ID"
    )
    product = GenericForeignKey('product_content_type', 'product_object_id')

    # 唯一商品编码
    unique_code = models.CharField(
        "唯一商品编码",
        max_length=64,
        unique=True,
        db_index=True,
        help_text="商品的唯一标识码，如序列号或内部编码",
        validators=[
            RegexValidator(
                r'^[A-Z0-9\-_]+$',
                message="商品编码只能包含大写字母、数字、连字符和下划线",
                code="invalid_code"
            )
        ],
    )

    # 商品在库状态
    status = models.CharField(
        "在库状态",
        max_length=32,
        choices=InventoryStatus.choices,
        default=InventoryStatus.IN_STOCK,
        db_index=True,
        help_text="商品当前的在库状态"
    )

    # 商品特别描述
    special_description = models.TextField(
        "特别描述",
        blank=True,
        default="",
        help_text="商品的特别说明或备注"
    )

    # 预订到货时间（仅适用于预订类状态）
    reserved_arrival_time = models.DateTimeField(
        "预订到货时间",
        null=True,
        blank=True,
        db_index=True,
        help_text="预订商品的预计到货时间（仅适用于预订类状态）"
    )

    # 状态异常备注（仅适用于异常状态）
    abnormal_remark = models.TextField(
        "状态异常备注",
        blank=True,
        default="",
        help_text="状态异常时的详细说明（仅适用于状态异常）"
    )

    # 时间戳
    created_at = models.DateTimeField(
        "创建时间",
        auto_now_add=True,
        db_index=True
    )
    updated_at = models.DateTimeField(
        "更新时间",
        auto_now=True,
        db_index=True
    )

    class Meta:
        verbose_name = "入库商品"
        verbose_name_plural = "入库商品"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["product_content_type", "product_object_id"]),
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["unique_code"]),
            models.Index(fields=["reserved_arrival_time"]),
        ]
        constraints = [
            # 确保预订类状态必须有预订到货时间
            models.CheckConstraint(
                name="chk_reserved_arrival_time",
                check=(
                    ~models.Q(
                        status__in=[
                            "CORPORATE_RESERVED_ARRIVAL",
                            "PERSONAL_RESERVED_ARRIVAL",
                            "PURCHASE_RESERVED_ARRIVAL"
                        ]
                    ) | models.Q(reserved_arrival_time__isnull=False)
                ),
            ),
            # 确保异常状态必须有异常备注
            models.CheckConstraint(
                name="chk_abnormal_remark",
                check=(
                    ~models.Q(status="STATUS_ABNORMAL") |
                    ~models.Q(abnormal_remark="")
                ),
            ),
        ]

    def __str__(self) -> str:
        product_name = str(self.product) if self.product else "未关联商品"
        return f"{self.unique_code} · {product_name} · {self.get_status_display()}"

    def save(self, *args, **kwargs):
        """保存时自动创建状态变更历史"""
        # 检查是否是更新操作
        is_update = self.pk is not None

        if is_update:
            # 获取旧的状态
            try:
                old_instance = InboundInventory.objects.get(pk=self.pk)
                old_status = old_instance.status
            except InboundInventory.DoesNotExist:
                old_status = None
        else:
            old_status = None

        # 先保存主记录
        super().save(*args, **kwargs)

        # 创建状态变更历史
        if is_update and old_status and old_status != self.status:
            InventoryStatusHistory.objects.create(
                inventory=self,
                old_status=old_status,
                new_status=self.status,
                change_reason=f"状态从 {self.InventoryStatus(old_status).label} 变更为 {self.InventoryStatus(self.status).label}",
            )
        elif not is_update:
            # 新建记录时也创建历史
            InventoryStatusHistory.objects.create(
                inventory=self,
                old_status=None,
                new_status=self.status,
                change_reason=f"初始状态：{self.InventoryStatus(self.status).label}",
            )


class InventoryStatusHistory(models.Model):
    """
    商品状态变更历史记录
    记录每个商品的状态变化轨迹
    """

    # 关联的库存商品
    inventory = models.ForeignKey(
        "InboundInventory",
        on_delete=models.CASCADE,
        related_name="status_history",
        verbose_name="关联商品",
        db_index=True,
    )

    # 旧状态（可为空，表示初始创建）
    old_status = models.CharField(
        "旧状态",
        max_length=32,
        choices=InboundInventory.InventoryStatus.choices,
        null=True,
        blank=True,
        help_text="变更前的状态（为空表示初始创建）"
    )

    # 新状态
    new_status = models.CharField(
        "新状态",
        max_length=32,
        choices=InboundInventory.InventoryStatus.choices,
        help_text="变更后的状态"
    )

    # 变更时间
    changed_at = models.DateTimeField(
        "变更时间",
        auto_now_add=True,
        db_index=True
    )

    # 变更原因/备注
    change_reason = models.TextField(
        "变更原因",
        blank=True,
        default="",
        help_text="状态变更的原因或备注"
    )

    # 操作人（可选，预留字段）
    changed_by = models.CharField(
        "操作人",
        max_length=128,
        blank=True,
        default="",
        help_text="执行状态变更的操作人（系统/用户名）"
    )

    class Meta:
        verbose_name = "状态变更历史"
        verbose_name_plural = "状态变更历史"
        ordering = ["-changed_at"]
        indexes = [
            models.Index(fields=["inventory", "-changed_at"]),
            models.Index(fields=["new_status", "-changed_at"]),
            models.Index(fields=["-changed_at"]),
        ]

    def __str__(self) -> str:
        old_display = (
            self.get_old_status_display() if self.old_status
            else "初始创建"
        )
        new_display = self.get_new_status_display()
        return (
            f"[{self.changed_at:%Y-%m-%d %H:%M}] {self.inventory.unique_code} · "
            f"{old_display} → {new_display}"
        )
