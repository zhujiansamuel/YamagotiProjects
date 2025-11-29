# 买方库存风控决策模块 (BuyRisk)

## 概述

本模块实现了"买方 + 库存风控决策"系统，集成到 Django + Celery 项目中。

### 核心功能

1. **数据源适配**：通过可配置的适配器，从现有的 ORM 模型读取收购价序列和库存数据
2. **计算链条**：
   - 收购价 → 漂移/波动 → 影子卖价 S^{shadow}
   - → 安全买价上限 B^{max} → 填充买价 B^{fill}
   - → 最终买价 B^{final}=min(B^{max},B^{fill})
   - → 闸门（OK/减速/停买）
3. **日更训练**：训练/校准供给曲线 λ(B)、α/β、清货折价 δ_{liq}、FX 波动
4. **覆盖率回测**：验证影子卖价的 VaR 覆盖率

## 项目结构

```
buyrisk/
├── __init__.py
├── apps.py              # Django 应用配置
├── models.py            # 数据模型（新增表）
├── adapters.py          # 数据适配器（读取现有表）
├── features.py          # 核心计算逻辑
├── tasks.py             # Celery 任务
├── admin.py             # Django admin 配置
└── migrations/          # 数据库迁移
```

## 安装与配置

### 1. 添加到 INSTALLED_APPS

在 `settings.py` 中：

```python
INSTALLED_APPS = [
    # ... 其他应用
    "buyrisk",  # 买方库存风控决策
]
```

### 2. 配置数据源映射

在 `settings.py` 中配置现有表的映射关系：

```python
# 价格序列表配置（必须）
BUY_RISK_PRICE_MODEL = "AppleStockChecker.PurchasingShopTimeAnalysis"
BUY_RISK_PRICE_TIME_FIELD = "Timestamp_Time"      # 时间字段
BUY_RISK_PRICE_VALUE_FIELD = "Price_A"            # 价格字段
BUY_RISK_PRICE_SKU_FIELD = "iphone__model_name"   # SKU字段（支持关联查询）
BUY_RISK_PRICE_WINDOW_DAYS = 7                    # 数据窗口（天）
BUY_RISK_PRICE_STEP_MINUTES = 15                  # 采样步长（分钟）

# 库存表配置（可选，不配置则使用 buyrisk.InventoryLot）
BUY_RISK_INVENTORY_MODEL = None
# 如果有现有库存表：
# BUY_RISK_INVENTORY_MODEL = "YourApp.Inventory"
# BUY_RISK_INVENTORY_SKU_FIELD = "sku"
# BUY_RISK_INVENTORY_COST_FIELD = "cost"
# BUY_RISK_INVENTORY_STATUS_FIELD = "status"
# BUY_RISK_INVENTORY_STATUS_VALUES = ["in_stock", "ready"]
```

**重要提示**：根据实际的数据模型调整字段映射。如果 SKU 是通过外键关联的，可以使用 Django 的双下划线语法（如 `iphone__model_name`）。

### 3. 配置默认参数

```python
BUY_RISK_DEFAULTS = {
    "tau_hours": 2.0,           # 清货时长 τ（小时）
    "q": 0.95,                  # 左尾分位数
    "alpha": 120.0,             # 卖-买固定价差
    "beta": 1.0,                # 卖-买比例
    "d_liq": 30.0,              # 极速清货折价
    "cost_per_unit": 100.0,     # 单位成本
    "min_margin": 150.0,        # 最小利润
    "fx_buffer": 20.0,          # FX 缓冲
    "i_star": 150.0,            # 目标库存
    "lambda_I": 1.0,            # 库存惩罚系数
    "b_ref": 3000.0,            # 供给曲线参考价
    "lambda_ref": 6.0,          # 参考到货率（件/小时）
    "b_elastic": 0.30,          # 价格弹性
    "q_star": 12.0,             # 目标进货（件/小时）
    "fx_sigma_daily": 0.008     # FX 日波动率
}
```

### 4. 配置 Celery Beat 定时任务

```python
CELERY_BEAT_SCHEDULE = {
    # 每 15 分钟计算决策
    "buyrisk_compute_all_skus_15m": {
        "task": "buyrisk.tasks.compute_all_skus",
        "schedule": 60 * 15,
    },
    # 每日训练模型
    "buyrisk_train_daily": {
        "task": "buyrisk.tasks.train_models_all_skus",
        "schedule": 60 * 60 * 24,
    },
    # 每日回测
    "buyrisk_backtest_daily": {
        "task": "buyrisk.tasks.backtest_coverage_all_skus",
        "schedule": 60 * 60 * 24,
    },
}
```

### 5. 运行迁移

```bash
# 生成迁移文件
python manage.py makemigrations buyrisk

# 执行迁移
python manage.py migrate buyrisk
```

## 数据模型

### 核心模型

1. **DecisionRun**：决策运行记录（每15分钟一次）
2. **SupplyCurveParam**：供给曲线参数（训练得到）
3. **ShadowParam**：影子卖价参数（训练得到）
4. **InventoryLot**：库存批次（如果不使用现有库存表）
5. **PurchaseLog**：回收成交日志（用于训练供给曲线）
6. **SellProxy**：代理卖价（用于训练 α/β）
7. **ClearanceEvent**：清货事件（用于训练 δ_liq）
8. **FxDaily**：FX 汇率日度数据
9. **CoverageReport**：覆盖率回测报告

## 使用方法

### 1. 手动触发决策计算

```python
from buyrisk.tasks import compute_decision_for_sku

# 计算单个 SKU
result = compute_decision_for_sku("iphone-17-pro-256")

# 计算所有 SKU
from buyrisk.tasks import compute_all_skus
results = compute_all_skus()
```

### 2. 训练模型

```python
from buyrisk.tasks import train_models_all_skus

# 训练所有模型参数
results = train_models_all_skus()
```

### 3. 运行回测

```python
from buyrisk.tasks import backtest_coverage_all_skus

# 回测覆盖率
report = backtest_coverage_all_skus(window_days=30)
```

### 4. 在 Django Admin 中查看

访问 `/admin/buyrisk/` 可以查看和管理：
- 决策运行记录
- 模型参数
- 训练数据
- 回测报告

## 工作流程

### 每15分钟自动运行

1. 从价格表获取近7天的收购价序列
2. 计算价格漂移和波动
3. 计算影子卖价 S_shadow
4. 获取当前库存
5. 计算安全买价上限 B_max
6. 根据目标进货量计算填充买价 B_fill
7. 确定最终买价 B_final = min(B_max, B_fill)
8. 判断闸门状态（OK/SLOW/STOP）
9. 保存决策结果到 DecisionRun 表

### 每日训练（建议凌晨运行）

1. **供给曲线训练**：从 PurchaseLog 拟合 λ(B) 参数
2. **α/β 训练**：从 SellProxy 回归卖-买价差关系
3. **δ_liq 训练**：从 ClearanceEvent 统计清货折价
4. **FX 波动训练**：从 FxDaily 计算汇率波动率

### 每日回测

- 使用滚动窗口验证影子卖价的 VaR 覆盖率
- 生成 CoverageReport 报告

## 注意事项

1. **字段映射**：务必根据实际数据模型调整 `settings.py` 中的字段映射
2. **数据准备**：确保有足够的历史数据用于训练（建议至少7天）
3. **性能优化**：对于大量 SKU，建议调整 Celery 并发数
4. **参数调优**：根据业务实际情况调整 `BUY_RISK_DEFAULTS` 中的参数
5. **监控告警**：建议对 DecisionRun 的 gate 状态设置监控

## 常见问题

### Q: 如何自定义 SKU 列表？

A: 在 `settings.py` 中设置：
```python
BUY_RISK_SKUS = ["sku1", "sku2", "sku3"]
```

### Q: 如何调整决策频率？

A: 修改 CELERY_BEAT_SCHEDULE 中的 schedule 值（单位：秒）

### Q: 如何查看决策结果？

A:
1. Django Admin: `/admin/buyrisk/decisionrun/`
2. 代码查询: `DecisionRun.objects.filter(sku="...").latest('ts_calc')`

### Q: 模型参数多久更新一次？

A: 默认每天更新一次，可以在 Celery Beat 中调整

## 开发与调试

### 测试单个 SKU 决策

```python
from buyrisk.tasks import compute_decision_for_sku
result = compute_decision_for_sku("test-sku")
print(result)
```

### 查看最新决策

```python
from buyrisk.models import DecisionRun
latest = DecisionRun.objects.filter(sku="test-sku").latest('ts_calc')
print(f"Gate: {latest.gate}")
print(f"B_final: {latest.b_final}")
print(f"S_shadow: {latest.s_shadow}")
```

## 许可与贡献

本模块是项目的一部分，遵循项目整体的许可协议。
