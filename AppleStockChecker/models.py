from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.db.models import Q, F
from django.core.validators import RegexValidator

class Iphone(models.Model):
    """
    iPhone 机型变体（型号 x 容量 x 颜色），以 Apple Part Number 作为唯一编码
    """
    jan = models.CharField(
        "JAN",
        max_length=13,
        blank=True,
        null=True,
        db_index=True,
        help_text="13位JAN码，如 4549995536515",
        validators=[RegexValidator(r"^\d{13}$", message="JAN 必须是 13 位数字", code="invalid_jan")],
    )
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
            models.Index(fields=["jan"], name="idx_iphone_jan"),
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


class SecondHandShop(models.Model):
    """
    二手店/回收店
    """
    name = models.CharField("店铺名称", max_length=128, db_index=True)
    website = models.URLField("网址", max_length=255, blank=True)
    address = models.CharField("地址", max_length=255)

    class Meta:
        verbose_name = "二手店"
        verbose_name_plural = "二手店"
        ordering = ["name"]
        constraints = [
            # 同名同址视为同一家，避免重复
            models.UniqueConstraint(
                fields=["name", "address"],
                name="uniq_secondhandshop_name_addr",
            )
        ]
        indexes = [
            models.Index(fields=["name"], name="idx_shs_name"),
            models.Index(fields=["address"], name="idx_shs_addr"),
        ]

    def __str__(self) -> str:
        return self.name


class PurchasingShopPriceRecord(models.Model):
    """
    二手店对某 iPhone 变体的回收报价记录（时间序列）
    - 必填：新品卖取价格
    - 可空：A品卖取价格、B品卖取价格
    价格单位默认按日元计（整数）
    """
    batch_id = models.UUIDField(
        "批次ID",
        null=True, blank=True, db_index=True, editable=False,
        help_text="一次导入/任务的批次标识（uuid4）"
    )


    shop = models.ForeignKey(
        "SecondHandShop",
        on_delete=models.PROTECT,        # 保留历史记录，防止误删门店
        related_name="price_records",
        verbose_name="二手店",
    )
    iphone = models.ForeignKey(
        "Iphone",
        on_delete=models.PROTECT,        # 同理，保留历史
        related_name="purchase_price_records",
        verbose_name="iPhone",
    )

    price_new = models.PositiveIntegerField("新品卖取价格(円)")
    price_grade_a = models.PositiveIntegerField("A品卖取价格(円)", null=True, blank=True)
    price_grade_b = models.PositiveIntegerField("B品卖取价格(円)", null=True, blank=True)

    recorded_at = models.DateTimeField("记录时间", auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "二手店回收价格记录"
        verbose_name_plural = "二手店回收价格记录"
        ordering = ["-recorded_at"]
        indexes = [
            models.Index(fields=["shop", "iphone", "-recorded_at"], name="idx_pspr_shop_phone_time"),
            models.Index(fields=["iphone", "-recorded_at"], name="idx_pspr_phone_time"),
            models.Index(fields=["shop", "-recorded_at"], name="idx_pspr_shop_time"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["shop", "iphone", "recorded_at"],
                name="uniq_shop_iphone_recorded_at",
            )
        ]

    def __str__(self) -> str:
        return f"[{self.recorded_at:%Y-%m-%d %H:%M}] {self.shop} · {self.iphone} · 新品¥{self.price_new:,}"



class PurchasingShopTimeAnalysis(models.Model):
    Batch_ID = models.UUIDField(
        "批次ID",
        null=True, blank=True, db_index=True, editable=False,
        help_text="一次导入/任务的批次标识（uuid4）",
        db_column = "batch_id",
    )
    Job_ID = models.CharField("Job_ID",max_length=255, editable=False,db_column = "job_id")


    Original_Record_Time_Zone = models.CharField("原始记录时区",max_length=10,db_column = "original_record_time_zone")
    Timestamp_Time_Zone = models.CharField("时间戳时区",max_length=10,null=False, blank=False,db_column = "timestamp_time_zone")
    Record_Time = models.DateTimeField("原始记录时间", null=False,blank=False,db_column = "record_time")
    Timestamp_Time = models.DateTimeField("时间戳时间", null=False,blank=False, db_index=True,db_column = "timestamp_time")
    Alignment_Time_Difference = models.IntegerField("对齐时间差（s）", null=False,blank=False,db_column = "alignment_time_difference")
    Update_Count = models.IntegerField("更新次数",default=0,db_column = "update_count")

    shop = models.ForeignKey(
        "SecondHandShop",
        on_delete=models.PROTECT,        # 保留历史记录，防止误删门店
        related_name="purchasing_time_analysis",
        verbose_name="二手店",
        db_index=True,
    )
    iphone = models.ForeignKey(
        "Iphone",
        on_delete=models.PROTECT,  # 同理，保留历史
        related_name="purchasing_time_analysis",
        verbose_name="iPhone",
        db_index=True,
    )
    New_Product_Price = models.PositiveIntegerField("新品卖取价格(円)",db_column = "new_product_price")
    Price_A = models.PositiveIntegerField("A品卖取价格(円)", null=True, blank=True,db_column = "price_a")
    Price_B = models.PositiveIntegerField("B品卖取价格(円)", null=True, blank=True,db_column = "price_b")
    Warehouse_Receipt_Time = models.DateTimeField("落库时间", auto_now_add=True,db_column = "warehouse_receipt_time")

    class Meta:
        verbose_name = "二手店回收价格记录对齐表"
        verbose_name_plural = "二手店回收价格记录对齐表"
        ordering = ["-Timestamp_Time"]
        indexes = [
            models.Index(fields=["shop", "iphone", "-Timestamp_Time"], name="idx_psta_shop_phone_time"),
            models.Index(fields=["iphone", "-Timestamp_Time"], name="idx_psta_phone_time"),
            models.Index(fields=["shop", "-Timestamp_Time"], name="idx_psta_shop_time"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["shop", "iphone", "Timestamp_Time"],
                name="uniq_shop_iphone_Timestamp_Time",
            )
        ]


    def save(self, *args, **kwargs):

        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"[{self.Timestamp_Time:%Y-%m-%d %H:%M}] {self.shop} · {self.iphone} · 新品¥{self.New_Product_Price:,}"

