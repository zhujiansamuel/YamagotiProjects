from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.db.models import Q, F

class Iphone(models.Model):
    """
    iPhone 机型变体（型号 x 容量 x 颜色），以 Apple Part Number 作为唯一编码
    """

    # 新增：唯一编码（Apple Part Number，如 "MTUW3J/A"）
    part_number = models.CharField(
        "唯一编码(Part Number)",
        max_length=32,
        unique=True,              # 数据库层面唯一
        help_text="Apple 部件号，如 MTLX3J/A",
    )

    model_name = models.CharField("型号", max_length=64, db_index=True)
    capacity_gb = models.PositiveSmallIntegerField(
        "容量(GB)",
        validators=[MinValueValidator(1), MaxValueValidator(4096)],
        help_text="以 GB 为单位；如需 1TB = 1024GB",
    )
    color = models.CharField("颜色", max_length=32, db_index=True)
    release_date = models.DateField("上市时间", db_index=True)

    class Meta:
        verbose_name = "iPhone 机型"
        verbose_name_plural = "iPhone 机型"
        ordering = ["-release_date", "model_name", "capacity_gb"]
        constraints = [
            # 继续保留业务唯一性（防止误填 Part Number 但规格重复）
            models.UniqueConstraint(
                fields=["model_name", "capacity_gb", "color"],
                name="unique_iphone_variant",
            )
        ]
        indexes = [
            models.Index(fields=["model_name", "capacity_gb"], name="idx_model_cap"),
        ]

    def __str__(self) -> str:
        cap = (
            f"{self.capacity_gb // 1024}TB"
            if self.capacity_gb % 1024 == 0
            else f"{self.capacity_gb}GB"
        )
        return f"{self.part_number} · {self.model_name} {cap} {self.color}"

class OfficialStore(models.Model):
    """
    Apple 官方门店（或授权店）
    """
    name = models.CharField("商店名", max_length=128, db_index=True)
    address = models.CharField("地址", max_length=255)

    class Meta:
        verbose_name = "官方门店"
        verbose_name_plural = "官方门店"
        ordering = ["name"]
        indexes = [
            models.Index(fields=["name"], name="idx_store_name"),
        ]

    def __str__(self) -> str:
        return self.name

class InventoryRecord(models.Model):
    """
    门店-机型 的库存记录（时间序列）
    - store: 关联 OfficialStore
    - iphone: 关联 Iphone
    - has_stock: 当前是否有现货
    - estimated_arrival_earliest / estimated_arrival_latest: 预计到达的最早/最晚时间（可空）
    - recorded_at: 记录时间（自动）
    """
    store = models.ForeignKey(
        "OfficialStore",
        on_delete=models.PROTECT,  # 保留历史记录，避免误删门店导致记录丢失
        related_name="inventory_records",
        verbose_name="店铺",
    )
    iphone = models.ForeignKey(
        "Iphone",
        on_delete=models.PROTECT,  # 同理，避免误删机型导致历史丢失
        related_name="inventory_records",
        verbose_name="iPhone",
    )
    has_stock = models.BooleanField("是否有库存", default=False, db_index=True)

    estimated_arrival_earliest = models.DateTimeField(
        "配送到达最早时间", null=True, blank=True
    )
    estimated_arrival_latest = models.DateTimeField(
        "配送到达最晚时间", null=True, blank=True
    )

    recorded_at = models.DateTimeField("记录时间", auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "库存记录"
        verbose_name_plural = "库存记录"
        ordering = ["-recorded_at"]
        indexes = [
            models.Index(
                fields=["store", "iphone", "-recorded_at"],
                name="idx_store_iphone_time",
            ),
            models.Index(fields=["iphone", "-recorded_at"], name="idx_iphone_time"),
            models.Index(fields=["store", "has_stock"], name="idx_store_stock"),
        ]
        constraints = [
            # 仅当两者均不为 NULL 时，要求 最早 <= 最晚
            models.CheckConstraint(
                name="chk_arrival_window_order",
                check=(
                        Q(estimated_arrival_earliest__isnull=True)
                        | Q(estimated_arrival_latest__isnull=True)
                        | Q(estimated_arrival_earliest__lte=F("estimated_arrival_latest"))
                ),
            ),
        ]

    def __str__(self) -> str:
        return f"[{self.recorded_at:%Y-%m-%d %H:%M}] {self.store} · {self.iphone} · {'有货' if self.has_stock else '无货'}"