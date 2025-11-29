from django.db import models


class SupplyCurveParam(models.Model):
    """供给曲线 λ(B) 参数：按 SKU 训练"""
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    b_ref = models.FloatField(verbose_name="参考价格")
    lambda_ref = models.FloatField(verbose_name="参考到货率（件/小时）")
    b_elastic = models.FloatField(verbose_name="价格弹性（每+¥100）")
    trained_at = models.DateTimeField(auto_now=True, verbose_name="训练时间")

    class Meta:
        verbose_name = "供给曲线参数"
        verbose_name_plural = "供给曲线参数"
        ordering = ['-trained_at']
        indexes = [
            models.Index(fields=['sku', '-trained_at']),
        ]

    def __str__(self):
        return f"{self.sku} - λ_ref={self.lambda_ref:.2f}"


class ShadowParam(models.Model):
    """影子卖价参数：α/β、δliq、q、τ、FX 波动（可训练/可人工覆盖）"""
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    alpha = models.FloatField(default=120.0, verbose_name="固定价差 α")
    beta = models.FloatField(default=1.0, verbose_name="比例系数 β")
    d_liq = models.FloatField(default=30.0, verbose_name="清货折价 δ_liq")
    q = models.FloatField(default=0.95, verbose_name="分位数 q")
    tau_hours = models.FloatField(default=2.0, verbose_name="清货时长 τ（小时）")
    fx_sigma_daily = models.FloatField(default=0.008, verbose_name="FX日波动率")
    trained_at = models.DateTimeField(auto_now=True, verbose_name="训练时间")

    class Meta:
        verbose_name = "影子卖价参数"
        verbose_name_plural = "影子卖价参数"
        ordering = ['-trained_at']
        indexes = [
            models.Index(fields=['sku', '-trained_at']),
        ]

    def __str__(self):
        return f"{self.sku} - α={self.alpha:.1f}, β={self.beta:.2f}"


class PurchaseLog(models.Model):
    """回收成交日志（拟合 λ(B) 用）：你写入即可"""
    ts = models.DateTimeField(db_index=True, verbose_name="时间戳")
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    offer_price = models.FloatField(verbose_name="报价")
    acquired = models.BooleanField(default=False, verbose_name="是否成交")
    channel = models.CharField(max_length=64, blank=True, default="", verbose_name="渠道")
    store = models.CharField(max_length=64, blank=True, default="", verbose_name="门店")

    class Meta:
        verbose_name = "回收成交日志"
        verbose_name_plural = "回收成交日志"
        ordering = ['-ts']
        indexes = [
            models.Index(fields=['sku', '-ts']),
            models.Index(fields=['ts', 'acquired']),
        ]

    def __str__(self):
        status = "成交" if self.acquired else "未成交"
        return f"{self.sku} - ¥{self.offer_price} ({status})"


class SellProxy(models.Model):
    """代理卖价（平台最低价/清算价/自家少量成交价）"""
    ts = models.DateTimeField(db_index=True, verbose_name="时间戳")
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    sell_proxy = models.FloatField(verbose_name="代理卖价")
    source = models.CharField(max_length=64, blank=True, default="", verbose_name="来源")

    class Meta:
        verbose_name = "代理卖价"
        verbose_name_plural = "代理卖价"
        ordering = ['-ts']
        indexes = [
            models.Index(fields=['sku', '-ts']),
        ]

    def __str__(self):
        return f"{self.sku} - ¥{self.sell_proxy} ({self.source})"


class ClearanceEvent(models.Model):
    """加速出清事件（在 τ 内卖出时的额外折价样本）"""
    ts = models.DateTimeField(db_index=True, verbose_name="时间戳")
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    extra_discount = models.FloatField(verbose_name="额外折价")
    note = models.CharField(max_length=128, blank=True, default="", verbose_name="备注")

    class Meta:
        verbose_name = "清货事件"
        verbose_name_plural = "清货事件"
        ordering = ['-ts']
        indexes = [
            models.Index(fields=['sku', '-ts']),
        ]

    def __str__(self):
        return f"{self.sku} - 折价¥{self.extra_discount}"


class FxDaily(models.Model):
    """FX 日级序列"""
    ts = models.DateField(db_index=True, unique=True, verbose_name="日期")
    fx = models.FloatField(verbose_name="汇率")

    class Meta:
        verbose_name = "FX汇率"
        verbose_name_plural = "FX汇率"
        ordering = ['-ts']

    def __str__(self):
        return f"{self.ts} - {self.fx}"


class InventoryLot(models.Model):
    """新建的库存表"""
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    cost = models.FloatField(verbose_name="成本")
    received_at = models.DateTimeField(verbose_name="入库时间")
    condition = models.CharField(max_length=32, blank=True, default="", verbose_name="成色")
    storage = models.CharField(max_length=32, blank=True, default="", verbose_name="仓库")
    status = models.CharField(
        max_length=32,
        db_index=True,
        default="in_stock",
        verbose_name="状态",
        choices=[
            ('in_stock', '在库'),
            ('ready', '待售'),
            ('sold', '已售'),
            ('reserved', '预留'),
        ]
    )

    class Meta:
        verbose_name = "库存批次"
        verbose_name_plural = "库存批次"
        ordering = ['-received_at']
        indexes = [
            models.Index(fields=['sku', 'status']),
            models.Index(fields=['status', '-received_at']),
        ]

    def __str__(self):
        return f"{self.sku} - ¥{self.cost} ({self.status})"


class DecisionRun(models.Model):
    """每次 15m 决策的落库结果（看板/审计友好）"""
    ts_calc = models.DateTimeField(db_index=True, verbose_name="计算时间")
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")

    # 影子卖价
    s_shadow = models.FloatField(verbose_name="影子卖价 S_shadow")
    e_b_tau = models.FloatField(verbose_name="期望收购价 E[B_τ]")

    # 漂移/波动
    mu_step = models.FloatField(verbose_name="单步漂移 μ_step")
    sigma_step = models.FloatField(verbose_name="单步波动 σ_step")
    mu_tau = models.FloatField(verbose_name="τ步漂移 μ_τ")
    sigma_tau = models.FloatField(verbose_name="τ步波动 σ_τ")

    # 决策价格
    b_max = models.FloatField(verbose_name="安全买价上限 B_max")
    b_fill = models.FloatField(verbose_name="填充买价 B_fill")
    b_final = models.FloatField(verbose_name="最终买价 B_final")

    # 闸门状态
    gate = models.CharField(max_length=8, verbose_name="闸门状态")
    gate_gap_ratio = models.FloatField(verbose_name="闸门缺口比例")

    # 供给曲线
    lam_final_per_hour = models.FloatField(verbose_name="最终到货率（件/小时）")

    # 库存风险
    wac = models.FloatField(verbose_name="加权平均成本 WAC")
    unit_shadow_pnl = models.FloatField(verbose_name="单件影子利润")
    mvar_total = models.FloatField(verbose_name="总边际风险")
    doh_days = models.FloatField(verbose_name="库存周转天数 DoH")
    daily_outflow = models.FloatField(verbose_name="日均出货量")

    # FX 缓冲
    fx_buffer_reco = models.FloatField(verbose_name="FX缓冲建议")
    inv_penalty = models.FloatField(verbose_name="库存惩罚")

    # JSON 存储
    decomposition_json = models.JSONField(default=dict, verbose_name="价格分解JSON")
    params_json = models.JSONField(default=dict, verbose_name="参数JSON")

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "决策运行记录"
        verbose_name_plural = "决策运行记录"
        ordering = ['-ts_calc']
        indexes = [
            models.Index(fields=['sku', '-ts_calc']),
            models.Index(fields=['-ts_calc', 'gate']),
        ]

    def __str__(self):
        return f"{self.sku} @ {self.ts_calc} - {self.gate} B_final=¥{self.b_final:.0f}"


class CoverageReport(models.Model):
    """影子卖价覆盖率 VaR 回测结果（日更）"""
    sku = models.CharField(max_length=128, db_index=True, verbose_name="SKU")
    window_days = models.IntegerField(default=30, verbose_name="回测窗口（天）")
    q_target = models.FloatField(verbose_name="目标分位数 q")
    coverage_realized = models.FloatField(null=True, verbose_name="实现覆盖率")
    n = models.IntegerField(default=0, verbose_name="样本数")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "覆盖率回测报告"
        verbose_name_plural = "覆盖率回测报告"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['sku', '-created_at']),
        ]

    def __str__(self):
        if self.coverage_realized is not None:
            return f"{self.sku} - 目标{self.q_target:.2%} 实现{self.coverage_realized:.2%} (n={self.n})"
        return f"{self.sku} - 目标{self.q_target:.2%} (n={self.n})"
