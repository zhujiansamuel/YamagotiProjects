# Timestamp Alignment Task 流程图

本文档描述 `timestamp_alignment_task.py` 中的任务执行流程。

## 整体架构图

```mermaid
flowchart TB
    subgraph Entry["入口层"]
        A[batch_generate_psta_same_ts<br/>父任务入口]
    end

    subgraph DataCollection["数据收集层"]
        B[collect_items_for_psta<br/>收集价格记录数据]
        C[按分钟桶分组数据<br/>bucket_minute_key]
    end

    subgraph AggControl["聚合控制层"]
        D{聚合模式<br/>agg_mode}
        D1[boundary<br/>边界模式]
        D2[rolling<br/>滚动模式]
        D3[off<br/>关闭聚合]
    end

    subgraph Execution["执行层"]
        E{执行模式<br/>sequential}
        E1[顺序执行<br/>逐个处理子任务]
        E2[并发执行<br/>Celery chord]
    end

    subgraph Processing["处理层"]
        F[psta_process_minute_bucket<br/>分钟桶处理任务]
        F1[guard_params<br/>参数守卫]
        F2[_process_minute_rows<br/>写入分钟数据]
        F3[_run_aggregation<br/>统计聚合]
    end

    subgraph Finalization["聚合层"]
        G[psta_finalize_buckets<br/>最终聚合回调]
        G1[汇总计数统计]
        G2[生成影子点<br/>Shadow Points]
        G3[WebSocket广播<br/>notify_progress_all]
    end

    A --> B
    B --> C
    C --> D
    D --> D1
    D --> D2
    D --> D3
    D1 --> E
    D2 --> E
    D3 --> E
    E -->|sequential=True| E1
    E -->|sequential=False| E2
    E1 --> F
    E2 --> F
    F --> F1
    F1 --> F2
    F2 --> F3
    F3 --> G
    E1 -->|直接调用| G
    E2 -->|chord回调| G
    G --> G1
    G1 --> G2
    G2 --> G3
```

## 详细流程图

```mermaid
flowchart TD
    START([开始]) --> A

    subgraph batch["batch_generate_psta_same_ts"]
        A[接收参数<br/>job_id, timestamp_iso, agg_minutes, agg_mode等]
        A --> B[调用 collect_items_for_psta<br/>查询 query_window_minutes 内的价格记录]
        B --> C[获取 rows 和 bucket_minute_key]
        C --> D[计算聚合上下文 ctx<br/>bucket_start, bucket_end]
        D --> E[广播 agg_ctx 通知]

        E --> F{遍历每个分钟桶}
        F --> G[提取该分钟的行数据<br/>minute_rows]
        G --> H{检查聚合模式}

        H -->|off| I1[do_agg = False<br/>agg_start_iso = None]
        H -->|rolling| I2[do_agg = True<br/>agg_start_iso = rolling_start]
        H -->|boundary| I3{是否为边界分钟?}
        I3 -->|是| I4[do_agg = True]
        I3 -->|否| I5[do_agg = False]

        I1 --> J
        I2 --> J
        I4 --> J
        I5 --> J

        J[创建子任务 signature<br/>psta_process_minute_bucket.s]
        J --> F

        F -->|所有桶处理完| K{subtasks 是否为空?}
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

    subgraph process["psta_process_minute_bucket"]
        P[接收参数<br/>ts_iso, rows, job_id, do_agg等]
        P --> P1[guard_params 参数守卫<br/>类型校验/版本检查]
        P1 --> P2[_to_aware 转换时间戳]
        P2 --> P3[_process_minute_rows<br/>处理并写入分钟数据]
        P3 --> P4{有错误?}
        P4 -->|是| P5[广播 bucket_errors 通知]
        P4 -->|否| P6{do_agg = True?}
        P5 --> P6
        P6 -->|是| P7[_run_aggregation<br/>执行统计聚合]
        P6 -->|否| P8[跳过聚合]
        P7 --> P9[返回结果<br/>ok, failed, chart_points等]
        P8 --> P9
    end

    subgraph finalize["psta_finalize_buckets"]
        Q[接收所有子任务结果<br/>results 列表]
        Q --> Q1[guard_params 参数守卫]
        Q1 --> Q2[汇总计数<br/>total_ok, total_failed]
        Q2 --> Q3[聚合错误直方图]
        Q3 --> Q4[聚合真实数据点<br/>series_map]
        Q4 --> Q5[计算 last_known 点]
        Q5 --> Q6{超过 MAX_PUSH_POINTS?}
        Q6 -->|是| Q7[截断保留最近N条]
        Q6 -->|否| Q8[生成影子点 Shadow Points]
        Q7 --> Q8
        Q8 --> Q9[构建 series_delta]
        Q9 --> Q10[构建最终 payload]
        Q10 --> Q11[WebSocket 广播<br/>status=done]
        Q11 --> Q12[返回 payload]
    end

    N4 --> Q
    O3 --> P
    P9 --> Q

    L --> END([结束])
    Q12 --> END
```

## 聚合模式详解

```mermaid
flowchart LR
    subgraph modes["聚合模式 agg_mode"]
        direction TB
        M1["boundary (默认)<br/>━━━━━━━━━━━━━━━<br/>仅在时间边界触发聚合<br/>例: 15分钟步长时<br/>00:00, 00:15, 00:30, 00:45<br/>这些时刻才会聚合"]

        M2["rolling<br/>━━━━━━━━━━━━━━━<br/>每分钟都执行滚动聚合<br/>聚合窗口: [当前-步长+1, 当前]<br/>例: 15分钟步长<br/>00:07 聚合 [00:00, 00:07]"]

        M3["off<br/>━━━━━━━━━━━━━━━<br/>完全关闭聚合<br/>仅写入分钟级原始数据<br/>不计算统计特征"]
    end
```

## 数据流图

```mermaid
flowchart LR
    subgraph Input["输入数据"]
        I1[PurchasingShopPriceRecord<br/>原始价格记录]
        I2[SecondHandShop<br/>店铺信息]
        I3[Iphone<br/>商品信息]
    end

    subgraph Processing["处理过程"]
        P1[collect_items_for_psta<br/>数据收集]
        P2[psta_process_minute_bucket<br/>分钟对齐]
        P3[_run_aggregation<br/>统计聚合]
    end

    subgraph Output["输出数据"]
        O1[PurchasingShopTimeAnalysis<br/>分钟级对齐数据]
        O2[FeatureSnapshot<br/>统计特征快照]
        O3[OverallBar<br/>全局统计条]
        O4[CohortBar<br/>群组统计条]
        O5[WebSocket Notification<br/>实时推送]
    end

    I1 --> P1
    I2 --> P1
    I3 --> P1
    P1 --> P2
    P2 --> O1
    P2 --> P3
    P3 --> O2
    P3 --> O3
    P3 --> O4
    P2 --> O5
```

## 任务关系图

```mermaid
flowchart TB
    subgraph Tasks["Celery Tasks"]
        T1["batch_generate_psta_same_ts<br/>━━━━━━━━━━━━━━━━━━━━━<br/>@shared_task(bind=True)<br/>父任务 / 编排器"]

        T2["psta_process_minute_bucket<br/>━━━━━━━━━━━━━━━━━━━━━<br/>核心处理任务<br/>分钟桶数据处理"]

        T3["psta_collect_result<br/>━━━━━━━━━━━━━━━━━━━━━<br/>结果收集器<br/>chain模式下使用"]

        T4["psta_finalize_buckets<br/>━━━━━━━━━━━━━━━━━━━━━<br/>@shared_task<br/>最终聚合回调"]
    end

    T1 -->|创建子任务| T2
    T1 -->|chord/chain| T4
    T2 -->|结果传递| T3
    T3 -->|累积结果| T4
    T2 -->|chord回调| T4
```

## 关键函数说明

| 函数名 | 位置 | 说明 |
|--------|------|------|
| `batch_generate_psta_same_ts` | L3518 | 父任务入口，负责数据收集、分桶、任务编排 |
| `psta_process_minute_bucket` | L1988 | 核心处理任务，写入分钟数据并可选执行聚合 |
| `psta_collect_result` | L3287 | chain模式下累积子任务结果 |
| `psta_finalize_buckets` | L3320 | 最终聚合回调，汇总结果并广播通知 |
| `guard_params` | L67 | 参数守卫，类型校验、版本检查、别名迁移 |
| `_process_minute_rows` | L1690 | 处理并写入分钟级数据 |
| `_run_aggregation` | L1838 | 执行统计聚合计算 |
| `collect_items_for_psta` | (collectors) | 从数据库收集待处理的价格记录 |

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

## 默认数字参数汇总

### 1. 任务版本控制

| 参数名 | 默认值 | 说明 | 位置 |
|--------|--------|------|------|
| `TASK_VER_PSTA` | `2` | 当前任务版本号，用于参数握手校验 | L1687 |
| `MIN_ACCEPTED_TASK_VER` | `0` | 最低可接受的任务版本（可通过环境变量 `PSTA_MIN_ACCEPTED_VER` 配置） | L30 |

### 2. 父任务入口参数 (`batch_generate_psta_same_ts`)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `chunk_size` | `200` | 分块大小 |
| `query_window_minutes` | `15` | 数据查询窗口（分钟） |
| `agg_minutes` | `15` | 聚合步长（分钟） |
| `agg_mode` | `"boundary"` | 聚合模式：`boundary` / `rolling` / `off` |
| `force_agg` | `False` | 强制聚合开关（已废弃，仅向后兼容） |
| `sequential` | `False` | 顺序执行模式（默认并发） |

### 3. 子任务参数 (`psta_process_minute_bucket`)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `do_agg` | `True` | 是否执行聚合 |
| `agg_minutes` | `1` | 聚合窗口（分钟） |

### 4. 数据容量限制

| 常量名 | 默认值 | 说明 | 位置 |
|--------|--------|------|------|
| `MAX_BUCKET_ERROR_SAMPLES` | `50` | 单桶保留的错误明细条数上限 | L1552 |
| `MAX_BUCKET_CHART_POINTS` | `3000` | 单桶打包给回调聚合用的图表点上限 | L1553 |
| `MAX_PUSH_POINTS` | `20000` | 本次广播给前端的真实点总上限（超过则截断保留最近N条） | L1554 |

### 5. 价格验证参数

| 常量名 | 默认值 | 说明 | 位置 |
|--------|--------|------|------|
| `PRICE_MIN` | `10000` | 固定价格下限（后备值，已废弃） | L1557 |
| `PRICE_MAX` | `350000` | 固定价格上限（后备值，已废弃） | L1558 |
| `PRICE_LOOKBACK_MINUTES` | `30` | 动态价格区间：向前查询的时间窗口（分钟） | L1561 |
| `PRICE_TOLERANCE_RATIO` | `0.10` | 动态价格区间：容差比例（±10%） | L1562 |
| `PRICE_MIN_SAMPLES` | `3` | 动态价格区间：计算参考价格所需的最少样本数 | L1563 |
| `PRICE_FALLBACK_MIN` | `10000` | 动态价格区间：数据不足时的后备最小值 | L1564 |
| `PRICE_FALLBACK_MAX` | `350000` | 动态价格区间：数据不足时的后备最大值 | L1565 |

### 6. 聚合计算参数

| 参数名 | 默认值 | 说明 | 来源 |
|--------|--------|------|------|
| `WATERMARK_MINUTES` | `5` | 水位线（分钟）：超过此时间的数据标记为 `is_final=True` | L1864 |
| `AGE_CAP_MIN` | `12.0` | 时效权重：超过此分钟数的数据不计入加权（可通过 `settings.PSTA_AGE_CAP_MIN` 配置） | L809 |
| `RECENCY_HALF_LIFE_MIN` | `6.0` | 时效权重：指数半衰期（分钟）（可通过 `settings.PSTA_RECENCY_HALF_LIFE_MIN` 配置） | L810 |
| `RECENCY_DECAY` | `"exp"` | 时效衰减模式：`exp`（指数） / `linear`（线性）（可通过 `settings.PSTA_RECENCY_DECAY` 配置） | L811 |

### 7. 时间序列特征参数 (SMA/EMA/WMA)

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `window` | `15` | 移动平均窗口大小 |
| `min_count` / `min_periods` | `1` | 计算所需的最小样本数 |
| `weights` | `"linear"` | WMA 权重模式 |
| `alpha` (EMA) | `2.0 / (window + 1.0)` | EMA 平滑系数（若未指定，由 window 推导） |

### 8. Bollinger Bands 参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `window` | `20` | 布林带窗口大小 |
| `k` | `2.0` | 标准差倍数（上下轨距离） |
| `min_periods` | `= window` | 计算所需的最小样本数 |
| `center_mode` | `"sma"` | 中轨计算模式：`sma` / `ema` / `sma60` 等 |

### 9. 安全 Upsert 参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `max_retries` | `2` | `safe_upsert_feature_snapshot` 重试次数 |

---

## 参数配置示意图

```mermaid
flowchart TB
    subgraph TaskParams["任务参数层级"]
        direction TB

        subgraph Parent["batch_generate_psta_same_ts"]
            P1["chunk_size = 200"]
            P2["query_window_minutes = 15"]
            P3["agg_minutes = 15"]
            P4["agg_mode = 'boundary'"]
            P5["sequential = False"]
        end

        subgraph Child["psta_process_minute_bucket"]
            C1["do_agg = True"]
            C2["agg_minutes = 1"]
            C3["task_ver = 2"]
        end

        subgraph Limits["容量限制"]
            L1["MAX_BUCKET_ERROR_SAMPLES = 50"]
            L2["MAX_BUCKET_CHART_POINTS = 3000"]
            L3["MAX_PUSH_POINTS = 20000"]
        end

        subgraph Price["价格验证"]
            PR1["PRICE_LOOKBACK_MINUTES = 30"]
            PR2["PRICE_TOLERANCE_RATIO = 0.10"]
            PR3["PRICE_MIN_SAMPLES = 3"]
            PR4["PRICE_FALLBACK_MIN = 10000"]
            PR5["PRICE_FALLBACK_MAX = 350000"]
        end

        subgraph Agg["聚合计算"]
            A1["WATERMARK_MINUTES = 5"]
            A2["AGE_CAP_MIN = 12.0"]
            A3["RECENCY_HALF_LIFE_MIN = 6.0"]
        end

        subgraph Features["特征计算"]
            F1["SMA/EMA/WMA window = 15"]
            F2["Bollinger window = 20"]
            F3["Bollinger k = 2.0"]
        end
    end

    Parent --> Child
    Child --> Limits
    Child --> Price
    Child --> Agg
    Agg --> Features
```

## 环境变量配置

以下参数可通过环境变量或 Django settings 进行配置：

| 环境变量 / Settings | 默认值 | 说明 |
|---------------------|--------|------|
| `PSTA_PARAM_STRICT` | `"warn"` | 参数严格度：`ignore` / `warn` / `error` |
| `PSTA_MIN_ACCEPTED_VER` | `0` | 最低可接受的任务版本 |
| `settings.PSTA_AGE_CAP_MIN` | `12.0` | 时效权重年龄上限（分钟） |
| `settings.PSTA_RECENCY_HALF_LIFE_MIN` | `6.0` | 时效衰减半衰期（分钟） |
| `settings.PSTA_RECENCY_DECAY` | `"exp"` | 时效衰减模式 |
| `settings.IPHONE_OFFICIAL_PRICES` | `{}` | iPhone 官方价格字典（用于 log 溢价计算） |
