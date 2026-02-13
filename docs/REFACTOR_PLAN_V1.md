# GPU + ClickHouse 重构方案 V1

> 状态: 设计基本确定，准备实施
> 最后更新: 2026-02-13
> 范围: 时间对齐 / 聚合 / 特征计算 pipeline 的全面重构
> 不包含: 数据摄入层(WebScraper/cleaners)、AutoML 因果分析

---

## 1. 重构动机

当前系统在"写数据 → 发现问题 → 修改逻辑 → 重新写数据"的循环中效率极低：

- PSTA / OverallBar / FeatureSnapshot 全部存储在 PostgreSQL
- 修改计算逻辑后需要 DELETE 大量行再重新逐行写入
- 98KB 的 `timestamp_alignment_task.py` 用 Python 循环逐行计算，无法利用 GPU
- 16+ Celery 队列增加运维复杂度，调试困难

---

## 2. 已确定的技术选型

| 项目 | 决策 | 备注 |
|------|------|------|
| GPU 框架 | **PyTorch** | 替代现有 CuPy，统一 GPU 计算 |
| 时序存储 | **ClickHouse 单节点 (Docker)** | 替代 PG 存储聚合/特征数据 |
| CH 连接方式 | **clickhouse-driver (Native TCP)** | 非 HTTP，最高吞吐 |
| 触发方式 | **Django management command** | 非 Celery，方便回写调试 |
| 数据摄入层 | **不改动** | WebScraper → shop cleaners → PG 保持原样 |
| AutoML | **不纳入本次重构** | 保留现有 CuPy 实现 |

---

## 3. 数据分层架构

```
保留不动                              推翻重建
┌────────────────────┐              ┌──────────────────────────────┐
│ WebScraper / XLSX  │              │                              │
│ 20 shop cleaners   │              │                              │
│ (Celery workers)   │              │                              │
└────────┬───────────┘              │                              │
         │                          │                              │
┌────────▼───────────┐              │                              │
│ PostgreSQL         │── 批量读 ──► │  PyTorch GPU Engine          │
│ PurchasingShop     │  pd.read_sql │  (management command)        │
│ PriceRecord        │              │                              │
│ (source of truth)  │              │  align → aggregate →         │
│                    │              │  features → cohorts          │
└────────────────────┘              │                              │
                                    └──────────┬───────────────────┘
                                               │ 批量 INSERT
                                    ┌──────────▼───────────────────┐
                                    │  ClickHouse (Docker)         │
                                    │  price_aligned               │
                                    │  features_wide               │
                                    │  (run_id 分区)               │
                                    └──────────────────────────────┘
```

---

## 4. 时间桶设计

### 4.1 数据到达特性

- 抓取间隔: ~15 分钟（非严格，受并发限制）
- 抓取并发: 2 个任务同时执行
- 店铺数: 20+，理想状态 16+ 店铺参与聚合
- 延迟: 20 个店铺 ÷ 2 并发 ≈ 10~15 分钟完成一轮

### 4.2 两层结构

**第一层: `price_aligned`** — 每店每次抓取的精确时间

将 PG 中 PriceRecord 的原始时间截断到分钟，按 (shop, iphone, bucket) 去重取最新值。
这一层保留数据的真实到达时间，不做人为对齐。

**第二层: 15 分钟聚合桶** — 跨店统计的基础

对齐到 15 分钟整点 (00/15/30/45)。
每个桶内，每个 shop 取 lookback 窗口内的最新价格。

### 4.3 聚合参数

```python
bucket_interval_min  = 15    # 聚合桶间隔
lookback_window_min  = 15    # 向前取数据的窗口
min_shop_quorum      = 16    # 期望的最少店铺数（仅记录，不强制跳过）
quorum_policy        = "use_anyway"  # 不管几个店都算，记录 shop_count
```

### 4.4 特征窗口映射

| 用户定义窗口 | 桶数 (÷15min) | 特征列名 |
|-------------|---------------|---------|
| 120 分钟 | 8 个桶 | `ema_120`, `sma_120`, `wma_120`, `boll_*_120` |
| 900 分钟 | 60 个桶 | `ema_900`, `sma_900`, `wma_900`, `boll_*_900` |
| 1800 分钟 | 120 个桶 | `ema_1800`, `sma_1800`, `wma_1800`, `boll_*_1800` |

列名使用分钟数命名（对人类直观）。

---

## 5. ClickHouse 表结构

### 5.1 `price_aligned` — 替代 PurchasingShopTimeAnalysis

```sql
CREATE TABLE price_aligned (
    run_id             String,
    bucket             DateTime('Asia/Tokyo'),
    shop_id            UInt32,
    iphone_id          UInt32,
    price_new          UInt32,
    price_a            Nullable(UInt32),
    price_b            Nullable(UInt32),
    alignment_diff_sec Int32,
    record_time        DateTime('Asia/Tokyo'),
    inserted_at        DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY (run_id, toYYYYMM(bucket))
ORDER BY (run_id, iphone_id, shop_id, bucket)
```

### 5.2 `features_wide` — 替代 OverallBar + FeatureSnapshot + CohortBar

```sql
CREATE TABLE features_wide (
    run_id              String,
    bucket              DateTime('Asia/Tokyo'),
    scope               String,        -- 'iphone:42' | 'cohort:top3'

    -- 基础跨店聚合
    mean                Float64,
    median              Float64,
    std                 Nullable(Float64),
    shop_count          UInt16,
    dispersion          Nullable(Float64),

    -- EMA (3 窗口)
    ema_120             Nullable(Float64),
    ema_900             Nullable(Float64),
    ema_1800            Nullable(Float64),

    -- SMA (3 窗口)
    sma_120             Nullable(Float64),
    sma_900             Nullable(Float64),
    sma_1800            Nullable(Float64),

    -- WMA (3 窗口)
    wma_120             Nullable(Float64),
    wma_900             Nullable(Float64),
    wma_1800            Nullable(Float64),

    -- Bollinger Bands (3 窗口 x 4 值)
    boll_mid_120        Nullable(Float64),
    boll_up_120         Nullable(Float64),
    boll_low_120        Nullable(Float64),
    boll_width_120      Nullable(Float64),

    boll_mid_900        Nullable(Float64),
    boll_up_900         Nullable(Float64),
    boll_low_900        Nullable(Float64),
    boll_width_900      Nullable(Float64),

    boll_mid_1800       Nullable(Float64),
    boll_up_1800        Nullable(Float64),
    boll_low_1800       Nullable(Float64),
    boll_width_1800     Nullable(Float64),

    inserted_at         DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY (run_id, toYYYYMM(bucket))
ORDER BY (run_id, scope, bucket)
```

### 5.3 run_id 工作流

- `live` — 线上正式版本
- `backfill_YYYYMMDD_vN` — 回写试验版本

```bash
# 回写试验
python manage.py run_pipeline --run-id backfill_v3 --from 2025-01-01 --to 2025-12-31

# 不满意 → 毫秒级删除
python manage.py drop_run --run-id backfill_v3 --confirm
# 内部执行: ALTER TABLE price_aligned DROP PARTITION ...
#           ALTER TABLE features_wide DROP PARTITION ...

# 满意 → 提升为 live
python manage.py promote_run --from backfill_v3 --to live
```

---

## 6. PyTorch 计算引擎

### 6.1 全部在 PyTorch 中完成的计算

| 计算 | 输入 | 输出 | 现状 |
|------|------|------|------|
| 时间对齐 | PG PriceRecord | price_aligned | 98KB task 逐行写 |
| 跨店聚合 (mean/median/std) | price_aligned | features_wide | Celery OverallBar 逐桶写 |
| EMA / SMA / WMA | 聚合后的时间序列 | features_wide | Python 循环逐标量算 |
| Bollinger Bands | 聚合后的时间序列 | features_wide | Python 循环逐标量算 |
| Cohort 加权聚合 | 聚合结果 + 权重配置 | features_wide (scope=cohort:*) | Celery CohortBar |
| 异常值过滤 | 跨店价格 | 过滤后再聚合 | _filter_outliers_by_mean_band |

### 6.2 计算流程

```python
def run_pipeline(run_id, date_from, date_to, device='cuda:0'):
    # Step 1: PG 批量读取
    df = pd.read_sql(
        "SELECT shop_id, iphone_id, price_new, price_grade_a, "
        "price_grade_b, recorded_at "
        "FROM PurchasingShopPriceRecord "
        "WHERE recorded_at BETWEEN %s AND %s",
        pg_conn, params=[date_from, date_to]
    )

    # Step 2: 15 分钟桶对齐，每 (shop, iphone, bucket) 取最新
    df['bucket'] = df['recorded_at'].dt.floor('15min')
    aligned = (df.sort_values('recorded_at')
                 .groupby(['bucket', 'shop_id', 'iphone_id'])
                 .last()
                 .reset_index())

    # Step 3: 写 price_aligned → ClickHouse
    ch_insert(aligned, table='price_aligned', run_id=run_id)

    # Step 4: 构建 3D tensor (n_iphones, n_shops, n_buckets)
    price_tensor = build_price_tensor(aligned, device=device)

    # Step 5: 跨店聚合 → (n_iphones, n_buckets)
    agg = aggregate_cross_shop(price_tensor, min_quorum=16)
    # agg.mean, agg.median, agg.std, agg.shop_count

    # Step 6: 特征计算（全部向量化，iPhone 维度并行）
    features = {}
    for window_min, window_buckets in [(120, 8), (900, 60), (1800, 120)]:
        features[f'ema_{window_min}']  = compute_ema_batch(agg.mean, window_buckets)
        features[f'sma_{window_min}']  = compute_sma_batch(agg.mean, window_buckets)
        features[f'wma_{window_min}']  = compute_wma_batch(agg.mean, window_buckets)
        boll = compute_bollinger_batch(agg.mean, window_buckets)
        features[f'boll_mid_{window_min}']   = boll.mid
        features[f'boll_up_{window_min}']    = boll.upper
        features[f'boll_low_{window_min}']   = boll.lower
        features[f'boll_width_{window_min}'] = boll.width

    # Step 7: Cohort 加权聚合
    cohort_features = compute_cohorts(agg, cohort_configs, device=device)

    # Step 8: 全部写入 ClickHouse features_wide
    ch_insert_features(agg, features, run_id=run_id, scope_prefix='iphone')
    ch_insert_features(cohort_features, run_id=run_id, scope_prefix='cohort')
```

### 6.3 PyTorch 向量化要点

```python
# EMA: 时间维度串行，但全部 iPhone 并行
def compute_ema_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """series: (n_iphones, n_buckets)"""
    alpha = 2.0 / (window + 1.0)
    ema = torch.zeros_like(series)
    ema[:, 0] = series[:, 0]
    for t in range(1, series.shape[1]):
        ema[:, t] = alpha * series[:, t] + (1 - alpha) * ema[:, t - 1]
    return ema

# SMA: 全部用 unfold 向量化，无 Python 循环
def compute_sma_batch(series: torch.Tensor, window: int) -> torch.Tensor:
    """series: (n_iphones, n_buckets)"""
    kernel = torch.ones(1, 1, window, device=series.device) / window
    padded = F.pad(series.unsqueeze(1), (window - 1, 0), mode='replicate')
    return F.conv1d(padded, kernel).squeeze(1)

# Bollinger: 基于 SMA + rolling std
def compute_bollinger_batch(series, window, k=2.0):
    mid = compute_sma_batch(series, window)
    # rolling std via unfold
    unfolded = series.unfold(dimension=1, size=window, step=1)
    std = unfolded.std(dim=-1)
    # pad front
    std = F.pad(std, (window - 1, 0), value=float('nan'))
    return BollingerResult(mid=mid, upper=mid + k*std, lower=mid - k*std, width=2*k*std)
```

---

## 7. Management Commands

### 7.1 `run_pipeline`

```bash
python manage.py run_pipeline \
    --run-id backfill_v3 \
    --from 2025-01-01 \
    --to 2025-12-31 \
    --device cuda:0 \
    --batch-days 30

# 只算某些 iPhone
python manage.py run_pipeline \
    --run-id test_iphone42 \
    --iphone-ids 42,43,44 \
    --from 2025-06-01 --to 2025-06-30

# 只算特定步骤
python manage.py run_pipeline \
    --run-id backfill_v3 \
    --steps features,cohorts
```

### 7.2 `drop_run`

```bash
python manage.py drop_run --run-id backfill_v2 --confirm
```

### 7.3 `compare_runs`

```bash
python manage.py compare_runs --base live --target backfill_v3
```

### 7.4 `promote_run`

```bash
python manage.py promote_run --from backfill_v3 --to live
```

---

## 8. Docker 新增服务

在 `docker-compose.yml` 中新增 ClickHouse 单节点：

```yaml
clickhouse:
  image: clickhouse/clickhouse-server:24-alpine
  container_name: ypa_clickhouse
  ports:
    - "9000:9000"    # Native TCP (clickhouse-driver)
    - "8123:8123"    # HTTP (备用/调试)
  volumes:
    - ypa_clickhouse_data:/var/lib/clickhouse
    - ./clickhouse/init.sql:/docker-entrypoint-initdb.d/init.sql
  environment:
    - CLICKHOUSE_DB=yamagoti
    - CLICKHOUSE_USER=default
    - CLICKHOUSE_PASSWORD=
  networks:
    - ypa_network
  ulimits:
    nofile:
      soft: 262144
      hard: 262144
```

---

## 9. Python 依赖变更

```
# 新增
torch                    # GPU 计算引擎
clickhouse-driver        # ClickHouse Native TCP 连接

# 保留
numpy
pandas
psycopg2-binary          # PG 读取原始数据

# 移除（计算层不再使用，AutoML 仍保留）
# cupy 暂时保留给 AutoML
```

---

## 10. 不改动的部分

以下模块保持现状，不纳入本次重构：

- **数据摄入层**: WebScraper tasks, shop cleaners, DataIngestionLog
- **PG 原始数据表**: Iphone, SecondHandShop, PurchasingShopPriceRecord, OfficialStore, InventoryRecord
- **AutoML pipeline**: AutomlCausalJob, automl_tasks.py, gpu_utils.py (CuPy)
- **Django Admin**: SimpleUI/SimplePro 管理界面
- **认证/权限**: JWT + Session 认证体系
- **WebSocket**: consumers.py 实时通知

---

## 11. 前端 API 改造（已确定：方案 A — ClickHouse Service 层）

### 11.1 改造策略

不使用双写方案，直接让 ViewSet 通过 `ClickHouseService` 查询 CH。
ViewSet 从 `ModelViewSet` 改为 `APIView` / `ViewSet`，手动组装 Response。
返回的 JSON 结构对前端保持不变（透明迁移）。

### 11.2 需要改造的 API

| 现有 ViewSet | 原读 PG 表 | 改为读 CH |
|---|---|---|
| `PurchasingShopTimeAnalysisViewSet` | PurchasingShopTimeAnalysis | → `price_aligned` WHERE run_id='live' |
| `PSTACompactViewSet` | PurchasingShopTimeAnalysis | → `price_aligned` WHERE run_id='live' |
| `OverallBarViewSet` / `PointsViewSet` | OverallBar | → `features_wide` WHERE scope LIKE 'iphone:%' |
| `CohortBarViewSet` / `PointsViewSet` | CohortBar | → `features_wide` WHERE scope LIKE 'cohort:%' |
| `FeatureSnapshotViewSet` / `PointsViewSet` | FeatureSnapshot | → `features_wide` |

### 11.3 不改造的 API（继续直接读 PG）

- Trends API (`trends_model_colors`, `TrendsColorStdApiView`, `TrendsAvgOnlyApiView`) — 直接读 PG PriceRecord
- 所有数据摄入相关 API
- iPhone / Shop / InventoryRecord 等基础数据 API
- AutoML 相关 API

### 11.4 ClickHouseService 概要

```python
# AppleStockChecker/services/clickhouse_service.py

class ClickHouseService:
    """封装所有 CH 读写操作"""

    def __init__(self):
        self.client = clickhouse_driver.Client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            database=settings.CLICKHOUSE_DB,
        )

    # ── 读 ──
    def query_price_aligned(self, run_id='live', filters=None, order='-bucket', limit=100): ...
    def query_features(self, run_id='live', scope=None, bucket_gte=None, bucket_lte=None): ...

    # ── 写 ──
    def insert_price_aligned(self, data, run_id): ...
    def insert_features(self, data, run_id): ...

    # ── 管理 ──
    def drop_run(self, run_id): ...
    def promote_run(self, from_run, to_run): ...
    def list_runs(self): ...
```

---

## 12. promote_run 策略（已确定：INSERT SELECT）

### 12.1 流程

```bash
# Step 1: 删掉旧 live 分区
ALTER TABLE price_aligned DROP PARTITION ('live', 202501);
ALTER TABLE price_aligned DROP PARTITION ('live', 202502);
...
ALTER TABLE features_wide DROP PARTITION ('live', 202501);
...

# Step 2: 把源 run 的数据复制为 live
INSERT INTO price_aligned
SELECT 'live' AS run_id, bucket, shop_id, iphone_id, price_new, price_a, price_b,
       alignment_diff_sec, record_time, now() AS inserted_at
FROM price_aligned
WHERE run_id = 'backfill_v3';

INSERT INTO features_wide
SELECT 'live' AS run_id, bucket, scope,
       mean, median, std, shop_count, dispersion,
       ema_120, ema_900, ema_1800,
       sma_120, sma_900, sma_1800,
       wma_120, wma_900, wma_1800,
       boll_mid_120, boll_up_120, boll_low_120, boll_width_120,
       boll_mid_900, boll_up_900, boll_low_900, boll_width_900,
       boll_mid_1800, boll_up_1800, boll_low_1800, boll_width_1800,
       now() AS inserted_at
FROM features_wide
WHERE run_id = 'backfill_v3';

# Step 3: (可选) 删掉源 run
ALTER TABLE price_aligned DROP PARTITION ...  WHERE run_id = 'backfill_v3';
```

### 12.2 不使用 ALTER UPDATE 的原因

ClickHouse 的 UPDATE 是异步 mutation，性能差且会破坏分区结构。
INSERT SELECT 是原子性的写入操作，速度更快，与 ReplacingMergeTree 配合良好。

---

## 13. 新代码模块结构

```
AppleStockChecker/
├── engine/                          # ★ 新增：GPU 计算引擎
│   ├── __init__.py
│   ├── config.py                    # BucketConfig, 窗口参数等
│   ├── reader.py                    # PG 批量读取 PriceRecord → DataFrame
│   ├── align.py                     # 时间对齐 (PriceRecord → price_aligned 格式)
│   ├── aggregate.py                 # 跨店聚合 (mean/median/std/dispersion)
│   ├── features.py                  # EMA/SMA/WMA/Bollinger 向量化计算
│   ├── cohorts.py                   # Cohort 加权聚合
│   └── pipeline.py                  # 串联以上步骤的主流程
│
├── services/
│   ├── clickhouse_service.py        # ★ 新增：CH 连接 + 读写封装
│   ├── auto_price_db.py             # 保留
│   ├── external_ingest_service.py   # 保留
│   └── time_analysis_services.py    # 保留（加注释说明已废弃，保留供参考）
│
├── management/commands/
│   ├── run_pipeline.py              # ★ 新增：主命令
│   ├── drop_run.py                  # ★ 新增：删除 run
│   ├── compare_runs.py              # ★ 新增：对比两个 run
│   ├── promote_run.py               # ★ 新增：提升 run 为 live
│   └── ... (现有命令保留)
│
├── clickhouse/                      # ★ 新增：CH DDL
│   └── init.sql                     # 建表语句 (price_aligned, features_wide)
│
├── tasks/
│   ├── timestamp_alignment_task.py  # 保留 + 头部注释说明不再是主计算路径
│   ├── webscraper_tasks.py          # 保留
│   └── automl_tasks.py              # 保留
│
├── features/
│   └── api.py                       # FeatureWriter — 保留（AutoML 可能仍引用）
```

### 13.1 各模块职责与估算

| 文件 | 职责 | 估算行数 |
|---|---|---|
| `engine/config.py` | `BucketConfig` dataclass, 特征列定义, 设备选择 | ~50 |
| `engine/reader.py` | `read_price_records(date_from, date_to)` → DataFrame | ~40 |
| `engine/align.py` | DataFrame → 15min 桶对齐 → price_aligned 格式 | ~80 |
| `engine/aggregate.py` | 3D tensor → 跨店 mean/median/std + 异常值过滤 | ~120 |
| `engine/features.py` | `compute_ema_batch`, `compute_sma_batch`, `compute_bollinger_batch` 等 | ~150 |
| `engine/cohorts.py` | 读 PG Cohort/CohortMember 配置 → 加权计算 | ~80 |
| `engine/pipeline.py` | `run(run_id, date_from, date_to, device, steps)` 主流程 | ~100 |
| `services/clickhouse_service.py` | ClickHouseService 类, 连接池, 批量 INSERT, DROP PARTITION | ~150 |

---

## 14. 现有文件处置策略

| 文件 | 处置 | 说明 |
|---|---|---|
| `tasks/timestamp_alignment_task.py` | **保留 + 加注释** | 头部标注 `# DEPRECATED: 主计算路径已迁移至 engine/pipeline.py`，AutoML 可能仍引用其工具函数 |
| `services/time_analysis_services.py` | **保留 + 加注释** | 同上，标注已废弃 |
| `features/api.py` (FeatureWriter) | **保留** | AutoML 仍可能使用，后续视情况决定 |
| PG Model (OverallBar, CohortBar, FeatureSnapshot) | **保留 Model 定义** | Django migration 不动，PG 表保留但不再写入新数据 |
| 现有 Serializer (OverallBarSerializer 等) | **保留** | API 改造后不再使用，但不删除，避免 import 链断裂 |

---

## 15. 待讨论 / 后续

- [ ] `features_wide` 特征列是否需要扩展（列固定 vs Map 类型）
- [x] ~~前端 API 改造~~ → 已确定：方案 A（ClickHouseService 层直接查 CH）
- [ ] 实时模式恢复时的 Celery 最小化方案
- [x] ~~promote_run 策略~~ → 已确定：INSERT SELECT（先删旧 live → 复制为 live → 可选删源 run）
- [ ] ClickHouse 备份策略
- [ ] Django settings 中 ClickHouse 连接配置的具体格式
