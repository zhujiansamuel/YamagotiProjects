# API 汇总文档

本文档汇总了 YamagotiProjects 项目中的全部 API，按模块与用途分类列出。**被内部调用的 API** 会标注调用来源与位置。

---

## 一、全局路由（根级）

| 路径 | 用途 | 被内部调用 |
|------|------|------------|
| `GET /admin/` | Django 管理后台 | `YamagotiProjects/settings.py`（菜单配置） |
| `GET /sp/` | SimplePro 扩展 | — |
| `GET /AppleStockChecker/schema/` | OpenAPI Schema（drf-spectacular） | — |
| `GET /AppleStockChecker/docs/` | Swagger UI 文档 | — |
| `GET /AppleStockChecker/redoc/` | ReDoc 文档 | — |
| `GET /healthz/` | 健康检查（返回 "ok"） | 外部/运维探针 |
| `GET/POST /accounts/...` | Django 认证相关（登录/登出等） | 各前端页面 `handleAuth` 重定向 |

---

## 二、AppleStockChecker 模块（前缀：`/AppleStockChecker/`）

### 2.1 基础接口

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/` | GET | API 根路径，列出可用接口 | `AppleStockChecker/views.py`（ApiRoot 响应中引用 `/api/health/`） |
| `/health/` | GET | API 健康检查 | — |
| `/me/` | GET | 当前登录用户信息 | — |

### 2.2 JWT 认证

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/auth/token/` | POST | 获取 JWT Access/Refresh Token | `templates/apple_stock/import_iphone_csv.html`、`import_price_csv.html`（文案提示）；`docs/GOODS_PRICE_SYNC.md`（示例） |
| `/auth/token/refresh/` | POST | 刷新 Access Token | — |
| `/auth/token/verify/` | POST | 校验 Token 有效性 | — |

### 2.3 REST ViewSet（CRUD）

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/iphones/` | GET, POST | iPhone 机型 CRUD | `store_latest.html`（fetchAll）、`delivery_trend.html`、`dashboard.html`、`import_iphone_csv.html`（import-csv）、`EChart.html`、`TemplateChartjs.html` |
| `/iphones/<id>/` | GET, PUT, PATCH, DELETE | 单条 iPhone 操作 | `store_latest.html`、`dashboard.html`（链接） |
| `/iphones/import-csv/` | POST | iPhone CSV 导入 | `import_iphone_csv.html` |
| `/stores/` | GET, POST | 官方门店 CRUD | `store_latest.html`、`dashboard.html`、`delivery_trend.html` |
| `/inventory-records/` | GET, POST | 库存记录 CRUD | `store_latest.html`、`dashboard.html` |
| `/inventory-records/trend/` | GET | 库存送达趋势（按 PN、天数） | `delivery_trend.html` |
| `/inventory-records/<id>/` | GET, PUT, PATCH, DELETE | 单条库存记录操作 | — |
| `/secondhand-shops/` | GET, POST | 二手店铺 CRUD | `price_matrix.html`、`EChart.html`、`TemplateChartjs.html` |
| `/purchasing-price-records/` | GET, POST | 采购价格记录 CRUD | `raw_price_charts.html`、`resale_trend_pn_merged.html` |
| `/purchasing-price-records/import-csv/` | POST | 采购价 CSV 导入 | `import_price_csv.html` |
| `/purchasing-price-records/<id>/` | GET, PUT, PATCH, DELETE | 单条采购价格记录操作 | — |
| `/purchasing-time-analyses/` | GET | 采购时间分析列表 | — |
| `/purchasing-time-analyses/<id>/` | GET | 单条采购时间分析 | — |
| `/purchasing-time-analyses-psta-compact/` | GET | PSTA 紧凑版分析 | `price_matrix.html`、`psta_raw_charts.html`、`EChart.html`、`TemplateChartjs.html` |
| `/overall-bars/` | GET | 整体条形图数据 | — |
| `/overall-bars/<id>/` | GET | 单条整体条形图 | — |
| `/shop-profiles/` | GET | 店铺权重组合列表 | — |
| `/shop-profiles/<id>/` | GET | 单条店铺组合 | — |
| `/ch/psta/` | GET | ClickHouse PSTA 完整数据 | — |
| `/ch/psta-compact/` | GET | ClickHouse PSTA 紧凑版 | — |

### 2.4 看板与数据展示

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/dashboard/` | GET | Apple 库存看板 | 前端页面，数据来自 `stores/`、`inventory-records/`、`iphones/` |
| `/store-latest/` | GET | 按门店分组的各 iPhone 最新库存（含收货延迟） | 同上 |
| `/analysis-dashboard/` | GET | 二手店价格分析看板 | 页面内调用 `api/trends/model-colors/avg-only/`、`api/trends/model-color/std/` |
| `/price-matrix/` | GET | 价格矩阵展示 | 页面内调用 `secondhand-shops/`、`purchasing-time-analyses-psta-compact/`、`post-to-x/` |
| `/raw-price-charts/` | GET | 近 2 天所有机种原始价格图表 | 页面内调用 `options/iphones/`、`purchasing-price-records/` |
| `/psta-raw-charts/` | GET | 近 2 天所有机种 PSTA 时间对齐原始图表 | 页面内调用 `purchasing-time-analyses-psta-compact/`，以及 `features/points/`（logb） |
| `/template-chartjs/` | GET | Chart.js 模板示例 | 页面内调用 `secondhand-shops/`、`iphones/`、`purchasing-time-analyses-psta-compact/`、`options/scopes/` |
| `/resale-trend-pn/` | GET | 二手店回收价历史（按 PN → 各店） | — |
| `/resale-trend-pn-merged/` | GET | 二手店回收价历史（合并版） | 页面内调用 `purchasing-price-records/` |
| `/resale-trend-colors-merged/` | GET | 机型+容量 → 各颜色新品价格趋势 | `YamagotiProjects/settings.py`（菜单配置，已注释） |
| `/delivery-trend/` | GET | 送达天数趋势（按 PN → 门店） | 页面内调用 `iphones/`、`inventory-records/trend/` |
| `/model_capacity_colors_trend`（model_capacity_colors_trend.html） | GET | 机型容量颜色趋势 | 页面内调用 `api/trends/model-colors/`、`api/trends/model-color/std/`、`api/trends/model-colors/avg-only/` |
| `/statistical-data-summary/` | GET | 统计摘要数据 | `YamagotiProjects/settings.py`（菜单配置，已注释） |

### 2.5 趋势 API

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/api/trends/model-colors/` | GET | 机型颜色趋势数据 | `model_capacity_colors_trend.html` |
| `/api/trends/model-color/std/` | GET/POST | 机型颜色标准差 | `analysis_dashboard.html`、`model_capacity_colors_trend.html` |
| `/api/trends/model-colors/avg-only/` | GET/POST | 机型颜色仅平均值 | `analysis_dashboard.html`、`model_capacity_colors_trend.html` |

### 2.6 数据导入

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/import-iphone-csv/` | GET | iPhone CSV 导入页面 | 页面内表单提交到 `iphones/import-csv/` |
| `/import-resale-csv/` | POST | 二手店回收价 CSV 导入 | — |
| `/external-ingest/` | POST | 外部平台拉取、预览与入库 | — |

### 2.7 PSTA 与调度

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/purchasing-time-analyses/dispatch_ts/` | POST | 按时间戳批量调度 PSTA 分析 | `AppleStockChecker/api/api.py`（注释中的 curl 示例）；`docs/CLEAR_FEATURE_SNAPSHOTS.md`；外部脚本/运维调用 |
| `/post-to-x/` | POST | 向 X 平台推送内容 | `price_matrix.html`（分享到 X 按钮） |

### 2.8 AutoML 相关

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/automl/` | GET | AutoML 主页面/总览 | — |
| `/automl/trigger/preprocessing-rapid/` | POST | 触发快速预处理 | — |
| `/automl/trigger/cause-and-effect-testing/` | POST | 触发因果效应测试 | — |
| `/automl/trigger/quantification-of-impact/` | POST | 触发影响量化 | — |
| `/automl/schedule/` | POST | 调度 AutoML 任务 | — |
| `/automl/jobs/create/` | POST | 创建单个 AutoML 任务 | `AutoML.html` |
| `/automl/jobs/batch-create/` | POST | 批量创建 AutoML 任务 | `AutoML.html` |
| `/automl/jobs/status/` | GET | 任务状态列表 | — |
| `/automl/jobs/status/<job_id>/` | GET | 指定任务状态 | — |
| `/automl/jobs/result/<job_id>/` | GET | 指定任务结果 | `AutoML.html` |
| `/automl/jobs/completed/` | GET | 已完成任务列表 | `AutoML.html` |
| `/automl/iphones/` | GET | 机型列表（AutoML 用） | `AutoML.html` |
| `/automl/jobs/sliding-window/` | POST | 滑窗分析 | `AutoML.html` |

### 2.9 外部商品价格同步

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/goods-sync/fetch/` | POST | 拉取外部商品数据 | `docs/GOODS_PRICE_SYNC.md` |
| `/goods-sync/mappings/` | GET | 商品映射列表 | `docs/GOODS_PRICE_SYNC.md` |
| `/goods-sync/statistics/` | GET | 商品映射统计 | `docs/GOODS_PRICE_SYNC.md` |
| `/goods-sync/update-price/` | POST | 更新单条外部价格 | `docs/GOODS_PRICE_SYNC.md` |
| `/goods-sync/batch-update-prices/` | POST | 批量更新外部价格 | `docs/GOODS_PRICE_SYNC.md` |

### 2.10 选项/下拉数据

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/options/scopes/` | GET | 范围选项（店铺组合、cohort 等） | `EChart.html`、`TemplateChartjs.html` |
| `/options/shops/` | GET | 店铺选项 | — |
| `/options/iphones/` | GET | iPhone 机型选项 | `raw_price_charts.html`、`psta_raw_charts.html` |
| `/options/cohorts/` | GET | Cohort（机型分组）选项 | — |
| `/options/shop-profiles/` | GET | 店铺权重组合选项 | — |

### 2.11 其他（路由已注释/备用）

| 路径 | 说明 | 被内部调用 |
|------|------|------------|
| `/features/`、`/features/points/` | FeatureSnapshot ViewSet（当前 urls 中已注释） | `EChart.html`、`TemplateChartjs.html`、`psta_raw_charts.html`（logb 数据） |

---

## 三、Inbound Goods 模块（前缀：`/inbound-goods/`）

| 路径 | 方法 | 用途 | 被内部调用 |
|------|------|------|------------|
| `/` | GET | 在库管理主页面 | `YamagotiProjects/settings.py`（菜单配置，已注释） |
| `/api/products/` | GET | 获取商品列表（用于下拉选择，目前为 iPhone） | `inbound_goods/templates/inbound_goods/inventory_management.html` |
| `/api/inventory/create/` | POST | 创建入库记录 | `inventory_management.html` |
| `/api/inventory/bulk-update/` | POST | 批量更新库存状态 | `inventory_management.html` |
| `/api/inventory/<inventory_id>/` | GET | 获取单条库存详情（含状态变更历史） | `inventory_management.html` |

---

## 四、内部调用关系速查（按调用方）

| 调用方 | 调用的 API |
|--------|------------|
| `templates/apple_stock/dashboard.html` | `stores/`、`inventory-records/`、`iphones/` |
| `templates/apple_stock/store_latest.html` | `iphones/`、`inventory-records/`、`stores/` |
| `templates/apple_stock/delivery_trend.html` | `iphones/`、`inventory-records/trend/` |
| `templates/apple_stock/resale_trend_pn_merged.html` | `purchasing-price-records/` |
| `templates/apple_stock/raw_price_charts.html` | `options/iphones/`、`purchasing-price-records/` |
| `templates/apple_stock/psta_raw_charts.html` | `options/iphones/`、`purchasing-time-analyses-psta-compact/` |
| `templates/apple_stock/price_matrix.html` | `secondhand-shops/`、`purchasing-time-analyses-psta-compact/`、`post-to-x/` |
| `templates/apple_stock/analysis_dashboard.html` | `api/trends/model-colors/avg-only/`、`api/trends/model-color/std/` |
| `templates/apple_stock/model_capacity_colors_trend.html` | `api/trends/model-colors/`、`api/trends/model-color/std/`、`api/trends/model-colors/avg-only/` |
| `templates/apple_stock/import_iphone_csv.html` | `iphones/import-csv/` |
| `templates/apple_stock/import_price_csv.html` | `purchasing-price-records/import-csv/` |
| `templates/apple_stock/AutoML.html` | `automl/iphones/`、`automl/jobs/create/`、`automl/jobs/batch-create/`、`automl/jobs/completed/`、`automl/jobs/result/<id>/`、`automl/jobs/sliding-window/` |
| `templates/apple_stock/EChart.html`、`TemplateChartjs.html` | `secondhand-shops/`、`iphones/`、`purchasing-time-analyses-psta-compact/`、`options/scopes/`、`features/points/` |
| `inbound_goods/.../inventory_management.html` | `api/products/`、`api/inventory/create/`、`api/inventory/bulk-update/`、`api/inventory/<id>/` |
| `inbound_goods/models`（Iphone） | `api/products/` 查询机型来源 |

---

## 五、文档与 Schema

- **Swagger UI**：`/AppleStockChecker/docs/`
- **ReDoc**：`/AppleStockChecker/redoc/`
- **OpenAPI Schema**：`/AppleStockChecker/schema/`

---

*文档生成日期：2025-02-21 | 含内部调用关系*
