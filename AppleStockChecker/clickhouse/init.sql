-- ============================================================================
-- ClickHouse 初始化 DDL
-- 自动在容器首次启动时执行 (docker-entrypoint-initdb.d)
-- 参考: docs/REFACTOR_PLAN_V1.md §5
-- ============================================================================

CREATE DATABASE IF NOT EXISTS yamagoti;

-- --------------------------------------------------------------------------
-- price_aligned: 替代 PG PurchasingShopTimeAnalysis
-- 每 (shop, iphone, 15min bucket) 保留最新一条价格
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS yamagoti.price_aligned (
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
ORDER BY (run_id, iphone_id, shop_id, bucket);

-- --------------------------------------------------------------------------
-- features_wide: 替代 PG OverallBar + FeatureSnapshot + CohortBar
-- scope 列区分维度: 'iphone:42', 'cohort:top3' 等
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS yamagoti.features_wide (
    run_id              String,
    bucket              DateTime('Asia/Tokyo'),
    scope               String,

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
ORDER BY (run_id, scope, bucket);
