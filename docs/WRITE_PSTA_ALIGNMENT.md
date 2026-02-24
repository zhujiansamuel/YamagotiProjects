# write_psta_alignment — 时间对齐回补命令

> **仅供开发阶段使用。**
> 直接调用 `timestamp_alignment_task` 内部私有函数，不经过 Celery 调度。

---

## 概述

将 `PurchasingShopPriceRecord` 中的原始记录，按时间对齐写入 `PurchasingShopTimeAnalysis`。

指定一个时间区间 `[--from, --to]`，命令自动每 15 分钟生成一个参考时间戳，每个时间戳覆盖其前 15 分钟窗口内的记录，确保区间内每一分钟都有对应的对齐数据。

---

## 时间戳生成规则

```
T₁ = floor(--from) + 14min   → 覆盖 [--from,    --from+14min]
T₂ = T₁ + 15min              → 覆盖 [--from+15, --from+29min]
T₃ = T₁ + 30min              → 覆盖 [--from+30, --from+44min]
...
停止条件：T_n - 14min > floor(--to)
```

**示例**：`--from 09:00 --to 11:00` 生成 9 个时间戳（09:14 / 09:29 / … / 11:14），完整覆盖 09:00–11:00。

> 最后一个时间戳的窗口可能略微超出 `--to`，但超出部分若无原始记录会直接跳过，不写入脏数据。

---

## 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--from` | ✅ | 回补区间起点，ISO 格式，含时区 |
| `--to` | ✅ | 回补区间终点，ISO 格式，含时区 |
| `--job-id` | | Job ID 标识，默认自动生成 uuid4 |
| `--shop-ids` | | 限定 Shop ID，逗号分隔，例如 `1,2,3` |
| `--iphone-ids` | | 限定 iPhone ID，逗号分隔，例如 `1,2,3` |
| `--dry-run` | | 只打印会写入的记录，不实际写入数据库 |

---

## 用法示例

### 基本回补

```bash
python manage.py write_psta_alignment \
    --from 2024-01-15T09:00:00+09:00 \
    --to   2024-01-15T11:00:00+09:00
```

### 写入前预览（推荐先执行）

```bash
python manage.py write_psta_alignment \
    --from 2024-01-15T09:00:00+09:00 \
    --to   2024-01-15T11:00:00+09:00 \
    --dry-run
```

### 指定 Job ID

```bash
python manage.py write_psta_alignment \
    --from 2024-01-15T09:00:00+09:00 \
    --to   2024-01-15T11:00:00+09:00 \
    --job-id backfill_2024_01_15
```

### 限定店铺与机型

```bash
python manage.py write_psta_alignment \
    --from 2024-01-15T09:00:00+09:00 \
    --to   2024-01-15T11:00:00+09:00 \
    --shop-ids 1,2 \
    --iphone-ids 3,4
```

### 组合使用

```bash
python manage.py write_psta_alignment \
    --from   2024-01-15T09:00:00+09:00 \
    --to     2024-01-15T11:00:00+09:00 \
    --job-id backfill_2024_01_15 \
    --shop-ids 1,2 \
    --iphone-ids 3,4 \
    --dry-run
```

---

## 输出说明

```
job_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
区间: 2024-01-15T09:00:00+09:00 → 2024-01-15T11:00:00+09:00
共生成 9 个时间戳（每 15 分钟一个）

[1/9] 时间戳 2024-01-15T09:14:00+09:00
  窗口: 2024-01-15T09:00:00+09:00 → 2024-01-15T09:14:00+09:00  原始记录: 42 条
  [2024-01-15T09:00:00+09:00] OK  ok=5
  [2024-01-15T09:01:00+09:00] OK  ok=3
  ...
[2/9] 时间戳 2024-01-15T09:29:00+09:00
  ...

完成  total_ok=180  total_failed=0
```

| 符号 | 含义 |
|---|---|
| `OK` | 该刻度所有记录写入成功 |
| `PARTIAL` | 部分失败，后跟具体错误信息 |
| 刻度行缺失 | 该分钟无原始记录，已跳过 |

---

## 内部调用链

```
Command.handle()
  └─ collect_items_for_psta(timestamp_iso, ...)   # utils/timestamp_alignment_task/collectors.py
       └─ PurchasingShopPriceRecord（查询）
  └─ _process_minute_rows(ts_iso, ts_dt, rows, job_id)   # tasks/timestamp_alignment_task.py（私有）
       └─ PurchasingShopTimeAnalysis（upsert）
```

---

## 注意事项

- `PurchasingShopTimeAnalysis` 的唯一约束为 `(shop, iphone, Timestamp_Time)`，重复执行会 **upsert**（更新已有记录的 `Update_Count`），不会产生重复数据。
- `--dry-run` 不保证和实际写入完全等价（例如动态价格区间过滤在 `_process_minute_rows` 内部执行，dry-run 阶段不运行），仅用于确认数据范围和条数。
- 该命令直接 import 私有函数 `_process_minute_rows`，如 `timestamp_alignment_task.py` 重构后需同步检查兼容性。
