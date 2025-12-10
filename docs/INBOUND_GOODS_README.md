# 入库商品管理系统 (Inbound Goods Management)

## 概述

入库商品管理系统是一个用于记录和追踪各种类型商品在库状态的 Django 应用。该系统设计灵活，支持多种商品类型，并自动记录商品状态变更历史。

## 功能特性

### 1. 主模型：InboundInventory（入库商品）

记录商品的基本信息和当前状态。

#### 字段说明

- **商品种类** (`product_content_type` / `product_object_id`)
  - 使用 Django GenericForeignKey 实现
  - 目前支持：iPhone（`AppleStockChecker.Iphone`）
  - 可扩展支持更多商品类型

- **唯一商品编码** (`unique_code`)
  - 商品的唯一标识码
  - 格式：仅支持大写字母、数字、连字符（-）和下划线（_）
  - 数据库级别唯一约束
  - 示例：`IPN-12345`, `IPHONE16-001`

- **在库状态** (`status`)
  - `CORPORATE_RESERVED_ARRIVAL`: 法人预订到货
  - `PERSONAL_RESERVED_ARRIVAL`: 个人预订到货
  - `PURCHASE_RESERVED_ARRIVAL`: 购买预订到货
  - `IN_STOCK`: 在库
  - `PREPARING_SHIPMENT`: 出货准备
  - `SHIPPED`: 已出货
  - `CANCELLED_RETURNED`: 取消退回
  - `STATUS_ABNORMAL`: 状态异常

- **特别描述** (`special_description`)
  - 商品的特别说明或备注
  - 可选字段

- **预订到货时间** (`reserved_arrival_time`)
  - 仅适用于预订类状态
  - 约束：预订类状态**必须**填写此字段

- **状态异常备注** (`abnormal_remark`)
  - 仅适用于状态异常
  - 约束：状态异常时**必须**填写此字段

- **时间戳**
  - `created_at`: 创建时间（自动）
  - `updated_at`: 更新时间（自动）

#### 数据库约束

1. **预订到货时间约束**
   - 当状态为预订类状态时，必须填写 `reserved_arrival_time`

2. **异常备注约束**
   - 当状态为 `STATUS_ABNORMAL` 时，必须填写 `abnormal_remark`

### 2. 历史模型：InventoryStatusHistory（状态变更历史）

自动记录商品的所有状态变更。

#### 字段说明

- **关联商品** (`inventory`)
  - 外键关联到 `InboundInventory`
  - 级联删除

- **旧状态** (`old_status`)
  - 变更前的状态
  - 初始创建时为 `null`

- **新状态** (`new_status`)
  - 变更后的状态

- **变更时间** (`changed_at`)
  - 自动记录变更时间

- **变更原因** (`change_reason`)
  - 状态变更的原因或备注
  - 自动生成或手动填写

- **操作人** (`changed_by`)
  - 执行变更的操作人
  - 预留字段，可为空

#### 自动记录机制

- 每次创建新商品时，自动创建初始状态历史
- 每次更新商品状态时，自动创建状态变更历史
- 历史记录**不可编辑、不可删除**（仅在 Admin 中可读）

## 使用指南

### 1. 应用迁移

```bash
# 使用项目提供的迁移脚本
./scripts/migrate.sh

# 或者在 Docker 环境中
docker-compose exec web python manage.py migrate
```

### 2. 在 Django Admin 中使用

#### 添加新商品

1. 进入 Admin 后台：`/admin/inbound_goods/inboundinventory/`
2. 点击"添加入库商品"
3. 填写必填字段：
   - 选择商品种类（目前只有 iPhone）
   - 输入商品ID（关联的 iPhone 记录的 ID）
   - 输入唯一商品编码
   - 选择初始状态
4. 根据状态填写条件字段：
   - 如果是预订类状态，填写预订到货时间
   - 如果是状态异常，填写异常备注
5. 保存

#### 查看状态历史

1. 在商品详情页面，向下滚动查看"状态变更历史"内联表格
2. 或者访问：`/admin/inbound_goods/inventorystatushistory/`

#### 更新商品状态

1. 进入商品编辑页面
2. 修改状态字段
3. 保存 - 系统会自动创建历史记录

### 3. 编程接口示例

```python
from inbound_goods.models import InboundInventory, InventoryStatusHistory
from AppleStockChecker.models import Iphone
from django.contrib.contenttypes.models import ContentType

# 获取 iPhone 内容类型
iphone_ct = ContentType.objects.get_for_model(Iphone)

# 获取一个 iPhone 实例
iphone = Iphone.objects.get(part_number="MTUW3J/A")

# 创建新的入库商品
inventory = InboundInventory.objects.create(
    product_content_type=iphone_ct,
    product_object_id=iphone.id,
    unique_code="IPN-MTUW3J-001",
    status=InboundInventory.InventoryStatus.IN_STOCK,
    special_description="全新未拆封"
)

# 更新状态（会自动创建历史记录）
inventory.status = InboundInventory.InventoryStatus.PREPARING_SHIPMENT
inventory.save()

# 查询商品的所有历史记录
history = inventory.status_history.all()
for record in history:
    print(f"{record.changed_at}: {record.old_status} → {record.new_status}")

# 查询特定状态的商品
in_stock_items = InboundInventory.objects.filter(
    status=InboundInventory.InventoryStatus.IN_STOCK
)

# 查询预订到货的商品（未来7天内）
from django.utils import timezone
from datetime import timedelta

upcoming_arrivals = InboundInventory.objects.filter(
    status__in=[
        InboundInventory.InventoryStatus.CORPORATE_RESERVED_ARRIVAL,
        InboundInventory.InventoryStatus.PERSONAL_RESERVED_ARRIVAL,
        InboundInventory.InventoryStatus.PURCHASE_RESERVED_ARRIVAL,
    ],
    reserved_arrival_time__lte=timezone.now() + timedelta(days=7)
).order_by('reserved_arrival_time')
```

## 数据库索引

系统已为以下字段创建索引，优化查询性能：

### InboundInventory
- `unique_code`（唯一索引）
- `(product_content_type, product_object_id)`（复合索引）
- `(status, created_at)`（复合索引）
- `reserved_arrival_time`

### InventoryStatusHistory
- `(inventory, -changed_at)`（复合索引）
- `(new_status, -changed_at)`（复合索引）
- `-changed_at`

## 扩展商品类型

将来如需添加新的商品类型（如 iPad、MacBook 等）：

1. 在对应的 app 中创建新的模型（如 `iPad`）
2. 更新 `InboundInventory` 模型的 `limit_choices_to` 参数：

```python
limit_choices_to={'model__in': ['iphone', 'ipad', 'macbook']},
```

3. 生成并应用新的迁移

## 最佳实践

1. **唯一编码规范**
   - 建议使用统一的编码格式，如：`{类型}-{Part Number}-{序号}`
   - 示例：`IPN-MTUW3J-001`, `IPD-MK2K3-042`

2. **状态流转规范**
   - 预订到货 → 在库 → 出货准备 → 已出货
   - 任何状态 → 取消退回
   - 任何状态 → 状态异常

3. **历史记录**
   - 不要删除历史记录，保持完整的审计轨迹
   - 在更新状态时填写详细的 `change_reason`

4. **性能优化**
   - 使用 `select_related('product_content_type')` 优化查询
   - 使用 `prefetch_related('status_history')` 预加载历史记录

## 回滚方案

如需回滚此功能：

```bash
# 回滚迁移
python manage.py migrate inbound_goods zero

# 从 settings.py 中移除
# 'inbound_goods.apps.InboundGoodsConfig',
```

## 版本历史

- **v0.1** (2025-12-04): 初始版本
  - 创建 InboundInventory 主模型
  - 创建 InventoryStatusHistory 历史模型
  - 支持 iPhone 商品类型
  - 实现自动状态历史记录
  - 添加数据库约束和索引
