# Timestamp Alignment Task 流程图

本文档描述 `timestamp_alignment_task.py` 中的任务执行流程。

> **v3 重大变更**（2026-01）：
> - 聚合逻辑从子任务移至 `psta_finalize_buckets`
> - 新增独立聚合任务 `psta_aggregate_features`
> - 新增专用队列 `psta_finalize` 和 `psta_aggregation`
> - 简化 `agg_mode`：`off` 禁用聚合，其他值统一为边界模式
> - 废弃参数：`do_agg`, `agg_start_iso`, `force_agg`, `chunk_size`

## 整体架构图（v3）

```mermaid
flowchart TB
    subgraph Entry["入口层"]
        A[batch_generate_psta_same_ts<br/>父任务入口]
    end

    subgraph DataCollection["数据收集层"]
        B[collect_items_for_psta<br/>收集价格记录数据]
        C[按分钟桶分组数据<br/>bucket_minute_key]
        D[计算 is_boundary<br/>判断是否为边界时间]
    end

    subgraph AggControl["聚合控制层"]
        E{聚合模式<br/>agg_mode}
        E1[boundary<br/>边界模式]
        E2[off<br/>关闭聚合]
    end

    subgraph Execution["执行层"]
        F{执行模式<br/>sequential}
        F1[顺序执行<br/>逐个处理子任务]
        F2[并发执行<br/>Celery chord]
    end

    subgraph Processing["处理层 (psta_finalize 队列)"]
        G[psta_process_minute_bucket<br/>分钟桶处理任务<br/>仅写入数据，不做聚合]
    end

    subgraph Finalization["汇总层 (psta_finalize 队列)"]
        H[psta_finalize_buckets<br/>汇总结果回调]
        H1[汇总计数统计]
        H2[生成影子点]
        H3[WebSocket广播]
        H4{is_boundary?}
    end

    subgraph Aggregation["聚合层 (psta_aggregation 队列)"]
        I[psta_aggregate_features<br/>独立聚合任务]
        I1[_run_aggregation<br/>执行统计聚合]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> E1
    E --> E2
    E1 --> F
    E2 --> F
    F -->|sequential=True| F1
    F -->|sequential=False| F2
    F1 --> G
    F2 --> G
    G --> H
    H --> H1
    H1 --> H2
    H2 --> H3
    H3 --> H4
    H4 -->|是 且 mode≠off| I
    H4 -->|否 或 mode=off| END([结束])
    I --> I1
    I1 --> END
```

## 详细流程图（v3）

```mermaid
flowchart TD
    START([开始]) --> A

    subgraph batch["batch_generate_psta_same_ts"]
        A[接收参数<br/>job_id, timestamp_iso, agg_minutes, agg_mode等]
        A --> B[调用 collect_items_for_psta<br/>查询 query_window_minutes 内的价格记录]
        B --> C[获取 rows 和 bucket_minute_key]
        C --> D[计算 is_boundary<br/>dt0 == floor_to_step]
        D --> E[构建 agg_ctx<br/>包含 is_boundary 标志]
        E --> F[广播 agg_ctx 通知]

        F --> G{遍历每个分钟桶}
        G --> H[提取该分钟的行数据<br/>minute_rows]
        H --> I{有数据?}
        I -->|是| J[创建子任务 signature<br/>psta_process_minute_bucket.s]
        I -->|否| G
        J --> G

        G -->|所有桶处理完| K{subtasks 是否为空?}
        K -->|是| L[广播空结果并返回]
        K -->|否| M{sequential 参数}

        M -->|True| N[顺序执行模式]
        M -->|False| O[并发执行模式]
    end

    subgraph seq["顺序执行流程"]
        N --> N1[逐个执行 subtask.apply]
        N1 --> N2[收集结果到 results 列表]
        N2 --> N3[报告进度通知]
        N3 --> N4[直接调用 psta_finalize_buckets]
    end

    subgraph para["并发执行流程"]
        O --> O1[创建 chord 回调<br/>psta_finalize_buckets.s]
        O1 --> O2[执行 chord subtasks callback]
        O2 --> O3[Celery 并行处理所有子任务]
    end

    subgraph process["psta_process_minute_bucket (v3简化)"]
        P[接收参数<br/>ts_iso, rows, job_id]
        P --> P1[guard_params 参数守卫]
        P1 --> P2[_process_minute_rows<br/>处理并写入分钟数据]
        P2 --> P3{有错误?}
        P3 -->|是| P4[广播 bucket_errors 通知]
        P3 -->|否| P5[返回结果]
        P4 --> P5
    end

    subgraph finalize["psta_finalize_buckets (v3增强)"]
        Q[接收所有子任务结果<br/>results 列表 + agg_ctx]
        Q --> Q1[guard_params 参数守卫]
        Q1 --> Q2[汇总计数<br/>total_ok, total_failed]
        Q2 --> Q3[聚合真实数据点]
        Q3 --> Q4[生成影子点 Shadow Points]
        Q4 --> Q5[WebSocket 广播<br/>status=done]
        Q5 --> Q6{检查 agg_ctx}
        Q6 --> Q7{agg_mode ≠ off<br/>且 is_boundary?}
        Q7 -->|是| Q8[同步调用<br/>psta_aggregate_features]
        Q7 -->|否| Q9[跳过聚合]
        Q8 --> Q10[返回 payload<br/>含聚合结果]
        Q9 --> Q10
    end

    subgraph aggregate["psta_aggregate_features (v3新增)"]
        R[接收 agg_ctx]
        R --> R1{agg_mode = off?}
        R1 -->|是| R2[跳过，返回 skipped]
        R1 -->|否| R3{is_boundary?}
        R3 -->|否| R2
        R3 -->|是| R4[_run_aggregation<br/>执行统计聚合]
        R4 --> R5{聚合成功?}
        R5 -->|是| R6[返回成功]
        R5 -->|否| R7[记录 ERROR log<br/>返回失败原因]
    end

    N4 --> Q
    O3 --> P
    P5 --> Q
    Q8 --> R

    L --> END([结束])
    Q10 --> END
    R2 --> END
    R6 --> END
    R7 --> END
```

## 聚合模式详解（v3简化）

```mermaid
flowchart LR
    subgraph modes["聚合模式 agg_mode (v3)"]
        direction TB
        M1["boundary (默认)<br/>━━━━━━━━━━━━━━━<br/>仅在时间边界触发聚合<br/>例: 15分钟步长时<br/>00:00, 00:15, 00:30, 00:45<br/>这些时刻才会聚合"]

        M2["off<br/>━━━━━━━━━━━━━━━<br/>完全关闭聚合<br/>仅写入分钟级原始数据<br/>不计算统计特征"]

        M3["其他值<br/>━━━━━━━━━━━━━━━<br/>v3起统一作为 boundary 处理<br/>rolling 模式已废弃"]
    end
```

## 数据流图（v3）

```mermaid
flowchart LR
    subgraph Input["输入数据"]
        I1[PurchasingShopPriceRecord<br/>原始价格记录]
        I2[SecondHandShop<br/>店铺信息]
        I3[Iphone<br/>商品信息]
    end

    subgraph Processing["处理过程"]
        P1[collect_items_for_psta<br/>数据收集]
        P2[psta_process_minute_bucket<br/>分钟对齐写入]
        P3[psta_finalize_buckets<br/>汇总结果]
        P4[psta_aggregate_features<br/>统计聚合]
    end

    subgraph Output["输出数据"]
        O1[PurchasingShopTimeAnalysis<br/>分钟级对齐数据]
        O2[FeatureSnapshot<br/>统计特征快照]
        O3[WebSocket Notification<br/>实时推送]
    end

    I1 --> P1
    I2 --> P1
    I3 --> P1
    P1 --> P2
    P2 --> O1
    P2 --> P3
    P3 --> O3
    P3 -->|边界时间| P4
    P4 --> O2
```

## 任务关系图（v3）

```mermaid
flowchart TB
    subgraph Tasks["Celery Tasks"]
        T1["batch_generate_psta_same_ts<br/>━━━━━━━━━━━━━━━━━━━━━<br/>@shared_task(bind=True)<br/>父任务 / 编排器<br/>队列: default"]

        T2["psta_process_minute_bucket<br/>━━━━━━━━━━━━━━━━━━━━━<br/>v3: 仅写入数据<br/>队列: default"]

        T3["psta_collect_result<br/>━━━━━━━━━━━━━━━━━━━━━<br/>结果收集器<br/>chain模式下使用"]

        T4["psta_finalize_buckets<br/>━━━━━━━━━━━━━━━━━━━━━<br/>v3: 汇总 + 触发聚合<br/>队列: psta_finalize"]

        T5["psta_aggregate_features<br/>━━━━━━━━━━━━━━━━━━━━━<br/>v3新增: 独立聚合任务<br/>队列: psta_aggregation"]
    end

    T1 -->|创建子任务| T2
    T1 -->|chord/chain| T4
    T2 -->|结果传递| T3
    T3 -->|累积结果| T4
    T2 -->|chord回调| T4
    T4 -->|边界时间同步调用| T5
```

## 关键函数说明（v3）

| 函数名 | 说明 | v3 变更 |
|--------|------|---------|
| `batch_generate_psta_same_ts` | 父任务入口，负责数据收集、分桶、任务编排 | 计算 is_boundary，不再传递 do_agg/agg_start_iso |
| `psta_process_minute_bucket` | 子任务，写入分钟数据 | 移除聚合逻辑，仅做数据写入 |
| `psta_finalize_buckets` | 回调任务，汇总结果并触发聚合 | 新增聚合任务调用逻辑 |
| `psta_aggregate_features` | **v3新增** 独立聚合任务 | - |
| `psta_collect_result` | chain模式下累积子任务结果 | 无变化 |
| `guard_params` | 参数守卫，类型校验、版本检查 | 无变化 |
| `_process_minute_rows` | 处理并写入分钟级数据 | 无变化 |
| `_run_aggregation` | 执行统计聚合计算 | 由 psta_aggregate_features 调用 |

## 执行模式对比

| 特性 | 顺序执行 (sequential=True) | 并发执行 (sequential=False) |
|------|---------------------------|---------------------------|
| 执行方式 | `subtask.apply().get()` | Celery `chord` |
| 资源占用 | 低，单worker | 高，多worker并行 |
| 执行速度 | 较慢 | 较快 |
| 错误处理 | 逐个捕获，继续执行 | chord失败可能中断 |
| 进度通知 | 每个桶完成后通知 | 仅最终结果通知 |
| 适用场景 | 调试、资源受限环境 | 生产环境、大数据量 |

---

## 默认数字参数汇总（v3）

### 1. 任务版本控制

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `TASK_VER_PSTA` | `3` | 当前任务版本号（v3） |
| `MIN_ACCEPTED_TASK_VER` | `0` | 最低可接受的任务版本 |

### 2. 父任务入口参数 (`batch_generate_psta_same_ts`)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `query_window_minutes` | `15` | 数据查询窗口（分钟） |
| `agg_minutes` | `15` | 聚合步长（分钟） |
| `agg_mode` | `"boundary"` | 聚合模式：`off` 禁用，其他 = 边界模式 |
| `sequential` | `False` | 顺序执行模式（默认并发） |
| ~~`chunk_size`~~ | - | 已废弃 |
| ~~`force_agg`~~ | - | 已废弃 |

### 3. 子任务参数 (`psta_process_minute_bucket`)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `agg_minutes` | `1` | 保留用于日志/调试 |
| ~~`do_agg`~~ | - | 已废弃（v3移除） |
| ~~`agg_start_iso`~~ | - | 已废弃（v3移除） |

### 4. 聚合任务参数 (`psta_aggregate_features`) - v3新增

| 参数名 | 说明 |
|--------|------|
| `ts_iso` | 时间戳 ISO 格式 |
| `job_id` | 任务 ID |
| `agg_ctx` | 聚合上下文字典 |
| `task_ver` | 任务版本号 |

### 5. 数据容量限制

| 常量名 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_BUCKET_ERROR_SAMPLES` | `50` | 单桶保留的错误明细条数上限 |
| `MAX_BUCKET_CHART_POINTS` | `3000` | 单桶打包给回调聚合用的图表点上限 |
| `MAX_PUSH_POINTS` | `60000` | 本次广播给前端的真实点总上限 |

### 6. 价格验证参数

| 常量名 | 默认值 | 说明 |
|--------|--------|------|
| `PRICE_LOOKBACK_MINUTES` | `30` | 动态价格区间：向前查询的时间窗口（分钟） |
| `PRICE_TOLERANCE_RATIO` | `0.10` | 动态价格区间：容差比例（±10%） |
| `PRICE_MIN_SAMPLES` | `3` | 计算参考价格所需的最少样本数 |
| `PRICE_FALLBACK_MIN` | `100000` | 数据不足时的后备最小值 |
| `PRICE_FALLBACK_MAX` | `350000` | 数据不足时的后备最大值 |

### 7. 聚合计算参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `WATERMARK_MINUTES` | `5` | 水位线（分钟）：超过此时间的数据标记为 `is_final=True` |
| `AGE_CAP_MIN` | `12.0` | 时效权重：超过此分钟数的数据不计入加权 |
| `RECENCY_HALF_LIFE_MIN` | `6.0` | 时效权重：指数半衰期（分钟） |

### 8. 任务超时配置（v3新增）

| 任务 | soft_time_limit | time_limit |
|------|-----------------|------------|
| `psta_finalize_buckets` | 240s | 360s |
| `psta_aggregate_features` | 240s | 360s |

---

## Celery 队列配置（v3）

| 队列名称 | 用途 | Worker 并发数 |
|---------|------|--------------|
| `default` | 默认队列、父任务、子任务 | 4 |
| `psta_finalize` | finalize 回调任务 | 2 |
| `psta_aggregation` | 聚合任务 | 4 |
| `webscraper` | 网页爬虫任务 | 2 |
| `automl_preprocessing` | AutoML 预处理 | 2 |
| `automl_cause_effect` | AutoML 因果分析 | 2 |
| `automl_impact` | AutoML 影响量化 | 2 |

**Worker 启动示例：**
```bash
# PSTA Finalize Worker
celery -A YamagotiProjects worker -Q psta_finalize -l info -c 2

# PSTA Aggregation Worker
celery -A YamagotiProjects worker -Q psta_aggregation -l info -c 4
```

---

## 参数配置示意图（v3）

```mermaid
flowchart TB
    subgraph TaskParams["任务参数层级 (v3)"]
        direction TB

        subgraph Parent["batch_generate_psta_same_ts"]
            P1["query_window_minutes = 15"]
            P2["agg_minutes = 15"]
            P3["agg_mode = 'boundary' | 'off'"]
            P4["sequential = False"]
        end

        subgraph Child["psta_process_minute_bucket"]
            C1["agg_minutes (日志用)"]
            C2["task_ver = 3"]
        end

        subgraph Finalize["psta_finalize_buckets"]
            F1["agg_ctx.is_boundary"]
            F2["agg_ctx.agg_mode"]
            F3["agg_ctx.bucket_start/end"]
        end

        subgraph Aggregate["psta_aggregate_features"]
            AG1["soft_time_limit = 240"]
            AG2["time_limit = 360"]
        end

        subgraph Limits["容量限制"]
            L1["MAX_BUCKET_ERROR_SAMPLES = 50"]
            L2["MAX_BUCKET_CHART_POINTS = 3000"]
            L3["MAX_PUSH_POINTS = 60000"]
        end
    end

    Parent --> Child
    Child --> Finalize
    Finalize -->|is_boundary| Aggregate
    Finalize --> Limits
```

## 环境变量配置

| 环境变量 / Settings | 默认值 | 说明 |
|---------------------|--------|------|
| `PSTA_PARAM_STRICT` | `"warn"` | 参数严格度：`ignore` / `warn` / `error` |
| `PSTA_MIN_ACCEPTED_VER` | `0` | 最低可接受的任务版本 |
| `settings.PSTA_AGE_CAP_MIN` | `12.0` | 时效权重年龄上限（分钟） |
| `settings.PSTA_RECENCY_HALF_LIFE_MIN` | `6.0` | 时效衰减半衰期（分钟） |
| `settings.PSTA_RECENCY_DECAY` | `"exp"` | 时效衰减模式 |
| `settings.IPHONE_OFFICIAL_PRICES` | `{}` | iPhone 官方价格字典 |

---

## v3 迁移指南

### 废弃参数处理

调用 `batch_generate_psta_same_ts` 时：
- `force_agg` 参数会被忽略并记录警告
- `chunk_size` 参数会被忽略并记录警告

调用 `psta_process_minute_bucket` 时：
- `do_agg` 参数会被忽略并记录警告
- `agg_start_iso` 参数会被忽略并记录警告

### 新增 Worker 配置

需要在 docker-compose.yml 中添加：
```yaml
celery_worker_psta_finalize:
  command: celery -A YamagotiProjects worker -Q psta_finalize -l info -c 2

celery_worker_psta_aggregation:
  command: celery -A YamagotiProjects worker -Q psta_aggregation -l info -c 4
```

### 聚合模式变化

| 原模式 | v3 行为 |
|--------|---------|
| `boundary` | 保持不变 |
| `rolling` | 自动转为 `boundary` |
| `off` | 保持不变 |
