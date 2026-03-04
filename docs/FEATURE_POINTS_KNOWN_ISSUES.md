# features/points 端点修复后 — 已知遗留问题

> 创建时间: 2026-03-04
> 关联修改: features/points 端点全面修复（后端适配前端）

---

## BUG 2（中等）：`_per_shop_features_df` 大范围回填时的内存/时间风险

**文件**: `AppleStockChecker/engine/pipeline.py` — `_per_shop_features_df()`

**现象**:
该函数对**每个 shop** 调用一次 `compute_all_features(shop_slice)`，每次产出 44 个
`(n_iphones, n_buckets)` Tensor，然后逐行构建 Python dict。

**数量级估算**:
- 1 天 (96 桶): 19 shops × 40 iphones × 96 buckets ≈ **72K 行** — 可接受
- 30 天回填 (2880 桶): 19 × 40 × 2880 ≈ **2.2M 行** — `itertuples` + dict 构造极慢；
  `compute_all_features` 对每个 shop 做 `(40, 2880)` 的卷积/EMA 共 19 次

**触发条件**: `pipeline.run()` 使用较大的 `date_from ~ date_to` 范围（回填场景）。
日常增量（1 天）不受影响。

**建议修复方向**:
1. 在 `pipeline.run()` 层面按天循环 per-shop 计算，而非一次性全量
2. 或给 `_per_shop_features_df` 增加 `max_rows` 安全阈值 + 日志警告
3. 或将逐行 dict 构造改为 numpy/pandas 向量化组装

---

## BUG 3（低）：单店稀疏数据导致大量 NaN 特征列

**文件**: `AppleStockChecker/engine/pipeline.py` — `_per_shop_features_df()`

**现象**:
单店的 `shop_slice[:, s_idx, :]` 在很多 `(iphone, bucket)` 上是 NaN（某店某时段无该机型数据）。

EMA 的迭代逻辑：
```python
ema[:, 0] = series[:, 0]   # 若 NaN → 整条 EMA 链 NaN
for t in range(1, ...):
    ema[:, t] = alpha * series[:, t] + (1 - alpha) * ema[:, t - 1]
```

一旦序列首值为 NaN，整条 EMA 链全部 NaN。SMA/WMA 的 `F.conv1d` 也因 NaN 传播产出 NaN。

**影响**:
- 不会写入错误数据（`insert_features` 将 NaN 转 NULL，API 层跳过 None）
- 但 CH 中大量行的特征列全是 NULL，只有 `mean` 有值，浪费存储和 INSERT 时间

**建议修复方向**:
1. 写入前过滤"所有特征列均为 NaN"的行
2. 或只对连续有数据的 `(shop, iphone)` 对计算特征

---

## ~~BUG 4（低）：per-profile 的 `std=0.0` 不精确~~ — 已修复 (2026-03-04)

已改为计算加权标准差 `sqrt(sum(w_i * (x_i - mean)^2) / sum(w_i))`，
dispersion 同步更新为 `std / mean`。

---

## 注意事项（非 Bug）

### `_query_ch_wide` 的 `limit: 10000` 可能截断

**文件**: `AppleStockChecker/views.py` — `FeatureSnapshotViewSet._query_ch_wide()`

`limit` 硬编码为 10000。per-shop scope 有 19×40=760 个，如果查询多天或无 scope
过滤，10000 行会截断。当前前端总是带单个 scope 过滤（不会触发），但 API 层面
存在截断风险。

### ALTER TABLE 迁移需手动执行

**文件**: `AppleStockChecker/clickhouse/init.sql`

init.sql 中的 ALTER TABLE 语句只在 ClickHouse 容器**首次启动**时执行
（`docker-entrypoint-initdb.d`）。对已有环境需手动运行 init.sql 尾部的
ALTER TABLE 语句块。

### ~~`compute_cohort_features` 的 `features` 参数未使用~~ — 已修复 (2026-03-04)

已移除死参数，同步更新调用方 `pipeline.py`。
