# 外部商品价格同步系统

本文档介绍如何使用外部商品价格同步系统,该系统可以将本项目的 iPhone 价格数据与外部电商项目进行同步。

## 📋 目录

- [系统概述](#系统概述)
- [架构设计](#架构设计)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [API 接口](#api-接口)
- [数据映射规则](#数据映射规则)
- [故障排查](#故障排查)

## 🎯 系统概述

### 功能特性

1. **自动商品映射**: 从外部项目获取商品列表,自动匹配本项目的 iPhone 实例
2. **映射数据存储**: 使用 SQLite 数据库(`auto_price.sqlite3`)存储映射关系
3. **价格同步**: 根据映射关系,将本项目的价格分析结果同步到外部项目
4. **映射统计**: 提供详细的映射统计信息和未匹配商品列表

### 数据流程

```
外部项目 (localhost:8080)
    ↓ [商品列表API]
本项目同步服务
    ↓ [解析&映射]
auto_price.sqlite3
    ↓ [查询映射关系]
价格更新API
    ↓ [更新价格]
外部项目商品
```

## 🏗️ 架构设计

### 核心组件

1. **AutoPriceSQLiteManager** (`services/auto_price_db.py`)
   - 管理 SQLite 数据库
   - 存储和查询映射关系
   - 维护同步历史记录

2. **ExternalGoodsClient** (`services/external_goods_sync.py`)
   - 与外部API通信
   - 获取商品列表
   - 更新商品价格

3. **IphoneMappingService** (`services/external_goods_sync.py`)
   - 解析外部商品信息
   - 映射到本项目 Iphone 实例
   - 计算映射置信度

4. **ExternalGoodsSyncService** (`services/external_goods_sync.py`)
   - 协调各组件工作
   - 执行完整同步流程

### 数据库结构

#### goods_iphone_mapping 表

```sql
CREATE TABLE goods_iphone_mapping (
    id INTEGER PRIMARY KEY,
    external_goods_id INTEGER NOT NULL,          -- 外部商品ID
    external_spec_index INTEGER NOT NULL,        -- 外部规格索引
    iphone_id INTEGER,                           -- 本项目Iphone ID
    external_title TEXT NOT NULL,                -- 外部商品标题
    external_spec_name TEXT NOT NULL,            -- 外部规格名称(颜色)
    external_category_name TEXT,                 -- 外部大类
    external_category_second_name TEXT,          -- 外部系列
    external_category_three_name TEXT,           -- 外部机型
    external_price INTEGER,                      -- 外部当前价格
    model_name TEXT,                             -- 解析的机型名
    capacity_gb INTEGER,                         -- 解析的容量(GB)
    color TEXT,                                  -- 解析的颜色
    confidence_score REAL DEFAULT 0.0,           -- 映射置信度(0-1)
    sync_status TEXT DEFAULT 'pending',          -- 状态: matched/unmatched/pending/error
    error_message TEXT,                          -- 错误信息
    last_sync_at TIMESTAMP,                      -- 最后同步时间
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    UNIQUE(external_goods_id, external_spec_index)
);
```

## ⚙️ 配置说明

### 环境变量配置

在 `.env` 文件中添加以下配置:

```bash
# 外部商品价格同步配置
EXTERNAL_GOODS_API_URL=http://localhost:8080
EXTERNAL_GOODS_API_TOKEN=your-external-api-token-here
```

### 获取 API Token

从外部项目获取访问令牌:

```bash
# 示例: 使用外部项目的登录接口获取 token
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}' \
  | jq -r '.token'
```

## 📖 使用方法

### 方法 1: 使用 Management Command (推荐)

#### 执行完整同步

```bash
python manage.py sync_external_goods
```

#### 清空现有映射后重新同步

```bash
python manage.py sync_external_goods --clear
```

#### 查看映射统计

```bash
python manage.py sync_external_goods --show-stats
```

#### 查看未匹配商品

```bash
python manage.py sync_external_goods --show-unmatched
```

#### 指定外部 API URL

```bash
python manage.py sync_external_goods --api-url http://example.com:8080
```

### 方法 2: 使用 API 接口

#### 1. 获取认证 Token

```bash
curl -X POST http://localhost:8000/AppleStockChecker/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username":"your_username","password":"your_password"}'
```

#### 2. 同步商品映射

```bash
curl -X POST http://localhost:8000/AppleStockChecker/goods-sync/fetch/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 3. 查看映射列表

```bash
# 查看所有映射
curl http://localhost:8000/AppleStockChecker/goods-sync/mappings/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# 只查看已匹配的
curl "http://localhost:8000/AppleStockChecker/goods-sync/mappings/?status=matched" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# 只查看未匹配的
curl "http://localhost:8000/AppleStockChecker/goods-sync/mappings/?status=unmatched" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 4. 查看统计信息

```bash
curl http://localhost:8000/AppleStockChecker/goods-sync/statistics/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 5. 更新外部商品价格

```bash
# 更新单个 iPhone 对应的所有外部商品
curl -X POST http://localhost:8000/AppleStockChecker/goods-sync/update-price/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "iphone_id": 33,
    "new_price": 195000
  }'
```

#### 6. 批量更新价格

```bash
curl -X POST http://localhost:8000/AppleStockChecker/goods-sync/batch-update-prices/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"iphone_id": 33, "new_price": 195000},
      {"iphone_id": 34, "new_price": 198000},
      {"iphone_id": 35, "new_price": 202000}
    ]
  }'
```

### 方法 3: 在 Python 代码中使用

```python
from AppleStockChecker.services import ExternalGoodsSyncService

# 创建同步服务
sync_service = ExternalGoodsSyncService()

# 执行同步
stats = sync_service.sync_goods_mappings()
print(f"同步完成: {stats}")

# 查看统计
statistics = sync_service.get_mapping_statistics()
print(f"映射统计: {statistics}")

# 更新价格
results = sync_service.update_external_price(
    iphone_id=33,
    new_price=195000
)
print(f"价格更新结果: {results}")
```

## 🔌 API 接口

### 1. POST /AppleStockChecker/goods-sync/fetch/

同步外部商品映射

**请求参数**:
- `api_url` (可选): 外部API URL
- `api_token` (可选): 外部API token

**响应示例**:
```json
{
  "success": true,
  "message": "商品映射同步完成",
  "statistics": {
    "total_items": 48,
    "matched_items": 44,
    "unmatched_items": 4,
    "error_items": 0
  }
}
```

### 2. GET /AppleStockChecker/goods-sync/mappings/

获取商品映射列表

**查询参数**:
- `status` (可选): matched/unmatched/pending/error
- `limit` (可选): 返回数量限制

**响应示例**:
```json
{
  "success": true,
  "total": 48,
  "mappings": [
    {
      "id": 1,
      "external_goods_id": 36,
      "external_spec_index": 1,
      "iphone_id": 41,
      "external_title": "iPhone Air 1TB",
      "external_spec_name": "クラウドホワイト",
      "model_name": "iPhone Air",
      "capacity_gb": 1024,
      "color": "クラウドホワイト",
      "confidence_score": 1.0,
      "sync_status": "matched",
      "last_sync_at": "2025-12-09 10:30:00"
    }
  ]
}
```

### 3. GET /AppleStockChecker/goods-sync/statistics/

获取映射统计信息

**响应示例**:
```json
{
  "success": true,
  "statistics": {
    "total": 48,
    "matched": 44,
    "unmatched": 4,
    "pending": 0,
    "error": 0,
    "last_sync_at": "2025-12-09 10:30:00"
  }
}
```

### 4. POST /AppleStockChecker/goods-sync/update-price/

更新外部项目商品价格

**请求体**:
```json
{
  "iphone_id": 33,
  "new_price": 195000
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "已更新 1/1 个商品价格",
  "iphone_info": {
    "id": 33,
    "part_number": "MG284J/A",
    "model_name": "iPhone Air",
    "capacity_gb": 256,
    "color": "クラウドホワイト"
  },
  "results": {
    "total": 1,
    "success": 1,
    "failed": 0,
    "details": [
      {
        "goods_id": 34,
        "spec_index": 1,
        "success": true
      }
    ]
  }
}
```

## 🔍 数据映射规则

### 机型名称映射

外部项目的 `category_three_name` 直接对应本项目的 `model_name`:

| 外部 category_three_name | 本项目 model_name |
|-------------------------|------------------|
| iPhone Air              | iPhone Air       |
| iPhone 17 Pro           | iPhone 17 Pro    |
| iPhone 17 Pro Max       | iPhone 17 Pro Max|

### 容量提取规则

从外部 `title` 字段中提取容量:

| title 示例         | 提取的容量 (GB) |
|-------------------|----------------|
| iPhone Air 256GB  | 256            |
| iPhone Air 512GB  | 512            |
| iPhone Air 1TB    | 1024           |

支持的容量格式:
- `\d+ GB` (如: 256GB)
- `\d+ TB` (如: 1TB, 自动转换为 1024GB)

### 颜色映射

外部项目的 `spec_name` 直接对应本项目的 `color`:

| 外部 spec_name     | 本项目 color      | 置信度 |
|-------------------|------------------|-------|
| クラウドホワイト    | クラウドホワイト   | 1.0   |
| スペースブラック    | スペースブラック   | 1.0   |

**添加自定义颜色映射**:

编辑 `AppleStockChecker/services/external_goods_sync.py`:

```python
class IphoneMappingService:
    COLOR_MAPPINGS = {
        # 外部颜色名 -> 本项目颜色名
        'Space Black': 'スペースブラック',
        'Cloud White': 'クラウドホワイト',
        # 添加更多映射...
    }
```

### 置信度计算

映射置信度分为以下几个等级:

| 匹配条件                  | 置信度 | 说明         |
|-------------------------|-------|-------------|
| 机型+容量+颜色完全匹配     | 1.0   | 精确匹配     |
| 机型+容量匹配,颜色不匹配   | 0.7   | 中等置信度   |
| 机型+颜色匹配,容量不匹配   | 0.5   | 较低置信度   |
| 仅机型匹配                | 0.3   | 低置信度     |
| 无匹配                   | 0.0   | 未匹配       |

## 🔧 故障排查

### 问题 1: 同步失败,提示无法连接外部API

**可能原因**:
- 外部项目未启动
- API URL 配置错误
- 网络连接问题

**解决方法**:
```bash
# 1. 检查外部项目是否运行
curl http://localhost:8080/api/goodsprice/list

# 2. 检查环境变量配置
cat .env | grep EXTERNAL_GOODS

# 3. 测试网络连接
ping localhost
```

### 问题 2: 大量商品未匹配

**可能原因**:
- 颜色名称不一致
- 机型名称不匹配
- 容量格式无法识别
- 本项目数据库中缺少对应的 Iphone 记录

**解决方法**:
```bash
# 1. 查看未匹配商品详情
python manage.py sync_external_goods --show-unmatched

# 2. 检查本项目 Iphone 数据
python manage.py shell
>>> from AppleStockChecker.models import Iphone
>>> Iphone.objects.filter(model_name='iPhone Air').values_list('color', flat=True)

# 3. 添加颜色映射规则 (见上文"数据映射规则"章节)
```

### 问题 3: 价格更新失败

**可能原因**:
- API Token 过期或无效
- 外部API权限不足
- 商品ID或规格索引错误

**解决方法**:
```bash
# 1. 重新获取 API Token
# 2. 检查映射数据
python manage.py sync_external_goods --show-stats

# 3. 手动测试价格更新API
curl -X POST http://localhost:8080/api/goodsprice/update \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"goods_id": 36, "spec_index": 1, "price": 195000}'
```

### 问题 4: SQLite 数据库损坏

**解决方法**:
```bash
# 1. 备份现有数据库
cp auto_price.sqlite3 auto_price.sqlite3.backup

# 2. 重新初始化数据库
rm auto_price.sqlite3
python manage.py sync_external_goods --clear
```

## 📝 注意事项

1. **首次同步**: 第一次运行同步时,建议使用 `--show-stats` 查看结果,确保映射正确
2. **定期同步**: 建议设置定时任务(如 cron)定期同步商品映射
3. **价格更新**: 在更新价格前,务必确认映射关系正确
4. **数据备份**: 定期备份 `auto_price.sqlite3` 数据库
5. **权限控制**: API 接口需要认证,确保 Token 安全

## 🔄 定时同步配置

### 使用 Celery Beat

在 `AppleStockChecker/tasks.py` 中添加:

```python
from celery import shared_task
from AppleStockChecker.services import ExternalGoodsSyncService

@shared_task
def sync_external_goods_task():
    """定时同步外部商品映射"""
    sync_service = ExternalGoodsSyncService()
    stats = sync_service.sync_goods_mappings()
    return stats
```

在 Celery Beat 配置中添加:

```python
# settings.py
CELERY_BEAT_SCHEDULE = {
    'sync-external-goods-every-hour': {
        'task': 'AppleStockChecker.tasks.sync_external_goods_task',
        'schedule': crontab(minute=0),  # 每小时执行
    },
}
```

### 使用 Cron

```bash
# 编辑 crontab
crontab -e

# 添加以下行 (每小时执行一次)
0 * * * * cd /path/to/project && python manage.py sync_external_goods >> /var/log/goods_sync.log 2>&1
```

## 📚 相关文档

- [Django Management Commands](https://docs.djangoproject.com/en/stable/howto/custom-management-commands/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Django REST Framework Authentication](https://www.django-rest-framework.org/api-guide/authentication/)

## 🤝 支持与反馈

如有问题或建议,请联系项目维护者或提交 Issue。
