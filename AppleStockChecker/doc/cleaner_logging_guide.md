# Shop Cleaner 统一日志规范

> 适用范围：shop2 / shop3 / shop4 / shop9 / shop11 / shop12 / shop14 / shop15 / shop16 / shop17
>
> 最后更新：2026-02-12

---

## 1. 设计目标

所有清洗器的 DEBUG/INFO 级日志使用**统一字段名和结构**，使得：

- ELK 可以用相同的查询条件跨 shop 分析提取质量
- 新增 shop 时有明确的日志字段规范可参照
- 日志中清晰区分 **abs（绝对价）** 和 **delta（差额）** 两种定价类型

### 1.1 已纳入规范的清洗器

| cleaner | shop_name | 定价模式 | 纳入日期 |
|---|---|---|---|
| shop2 | 海峡通信 | delta-only（group→color 匹配） | 2026-02-12 |
| shop3 | 買取一丁目 | delta-only | 2026-02-12 |
| shop4 | モバイルミックス | delta-only（含"全色"ALL 机制） | 2026-02-12 |
| shop9 | ダイワンテレコム | delta-only | 2026-02-12 |
| shop11 | アメモバ | delta-only | 2026-02-11 |
| shop12 | リンクサスモバイル | delta-only | 2026-02-11 |
| shop14 | じゃんぱら | delta-only | 2026-02-11 |
| shop15 | — | abs + delta 混合 | 2026-02-11 |
| shop16 | — | abs + delta 混合 | 2026-02-11 |
| shop17 | — | delta-only（预留 abs 字段） | 2026-02-11 |

---

## 2. 核心概念

### 2.1 定价类型（effective_source）

| effective_source | 含义 | 典型场景 |
|---|---|---|
| `"abs_price"` | 颜色有独立的绝对价格，不依赖 base_price | shop15: `ブルー229,000円` |
| `"matched_label"` | 颜色有相对于 base_price 的差额 | shop15: `ブルー-1000円`；shop4: `全色-1000`；shop2: `青-1000` |
| `"default_zero"` | 未命中任何 spec，直接使用 base_price | 所有颜色均无特殊标注时 |

> **注意**：shop2–shop14 均为 delta-only 模式，`effective_source` 仅为 `"matched_label"` 或 `"default_zero"`。
> `abs_prices` 字段固定为空数组 `[]`，`abs_prices_count` 固定为 `0`。

### 2.2 关键字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `effective_source` | `str` | 定价来源类型（见上表） |
| `matched_label` | `str \| None` | 命中的原始标签文本（如 `"ブルー"`、`"シルバー"`） |
| `spec_value` | `int \| None` | 从原文提取的原始值。abs 时为绝对价，delta 时为差额（signed） |
| `final_price` | `int` | 入库的最终价格 |

**注意**：

- `spec_value` 记录的是**提取阶段的原始值**，而非计算后的差异。
  - abs 类型：`spec_value == final_price`
  - delta 类型：`final_price == base_price + spec_value`
  - default_zero：`spec_value == None`，`final_price == base_price`
- 不再使用 `delta` 作为日志字段（避免 abs 类型时的语义歧义）。

---

## 3. 日志事件类型（event_type）

清洗器在处理每行数据时，按以下顺序产生日志事件：

### 3.1 `extraction_result`（DEBUG）

提取完成后立即记录。所有字段：

```
event_type: "extraction_result"
log_seq: int
shop_name: str
cleaner_name: str
row_index: int
model_text: str
model_norm: str
capacity_gb: int
base_price: int | None
source_text_raw: str                                    # 截断版（≤200字符）
source_text_raw_full: str                               # 完整版
source_text_normalized: str                             # 归一化后的截断版（≤200字符）
extraction_method: "regex" | "llm" | "auto" | "none"
labels_and_deltas: [{"label": str, "delta": int}]      # delta 类型的提取结果
abs_prices: [{"label": str, "amount": int}]             # abs 类型的提取结果（仅 shop15/16；delta-only shop 固定为 []）
labels_extracted_count: int                             # delta 标签数量
abs_prices_count: int                                   # abs 标签数量（delta-only shop 固定为 0）
available_colors: [{"color_norm", "part_number", "color_raw"}]
colors_in_catalog: int
```

### 3.2 `label_matching`（DEBUG）

每个提取到的标签的颜色匹配详情。shop15/16 有 delta 和 abs 两个变体，其余 shop 仅 delta。
消息格式：`"Label matching (delta): {label}"` 或 `"Label matching (abs): {label}"`。

```
event_type: "label_matching"
log_seq: int
shop_name: str
cleaner_name: str
row_index: int
model_text: str
model_norm: str
capacity_gb: int
base_price: int
label: str                    # 原始标签
delta: int                    # delta 变体时的差额值（delta 变体独有）
abs_price: int                # abs 变体时的绝对价（abs 变体独有，仅 shop15/16）
match_type: "delta" | "abs"   # 匹配类型
matched_colors: [str]         # 命中的 color_norm 列表
matched_part_numbers: [str]   # 命中的 part_number 列表
match_count: int              # 命中数量
source_text_raw_full: str     # 完整原文
labels_and_deltas: [{"label": str, "delta": int}]
```

### 3.3 `label_no_match`（WARNING）

标签未命中任何颜色时触发。消息格式：`"Label not matched (delta): {label}"` 或 `"Label not matched (abs): {label}"`。

```
event_type: "label_no_match"
log_seq: int
shop_name: str
cleaner_name: str
row_index: int
model_text: str
model_norm: str
capacity_gb: int
base_price: int
label: str
delta: int                    # delta 变体时（delta 变体独有）
abs_price: int                # abs 变体时（abs 变体独有，仅 shop15/16）
match_type: "delta" | "abs"   # 匹配类型
available_colors: [str]       # 该机型可用的所有 color_norm
source_text_raw_full: str     # 完整原文
labels_and_deltas: [{"label": str, "delta": int}]
```

### 3.4 `output_record`（DEBUG）

每条输出记录（per part_number）的详情。消息格式：`"Output record: {pn}"`。

```
event_type: "output_record"
log_seq: int
shop_name: str
cleaner_name: str
row_index: int
model_text: str
model_norm: str
capacity_gb: int
part_number: str
color_norm: str
color_raw: str
base_price: int
final_price: int
effective_source: "abs_price" | "matched_label" | "default_zero"
matched_label: str | None
spec_value: int | None
recorded_at: str | None
source_text_raw_full: str
labels_and_deltas: [{"label": str, "delta": int}]
```

### 3.5 `row_processing_summary`（DEBUG + INFO）

**DEBUG 级**：行级详细汇总，包含按类型分组的输出记录：

```
event_type: "row_processing_summary"
base_price: int
abs_applied_details: [
    {"pn": str, "color": str, "final_price": int, "matched_label": str, "spec_value": int}
]
delta_applied_details: [
    {"pn": str, "color": str, "final_price": int, "matched_label": str, "spec_value": int}
]
default_applied_pns: [str]
```

> **ELK 显示特性**：`abs_applied_details` 和 `delta_applied_details` 中的每个字段
> 会被 ELK 自动展开为并行数组。例如：
> - `abs_applied_details.pn`: `["MU7R3J/A", "MU7S3J/A"]`
> - `abs_applied_details.spec_value`: `[229000, 229000]`

**INFO 级**：简洁的一行概览，消息格式：

```
Row {idx} | {model_text} | deltas: {N} | abs: {N} | matched: {N} | records: {N} | method: {extraction_method}
```

---

## 4. 内部数据结构

### 4.1 `current_row_records`（循环内部累加器）

在遍历 `color_map` 生成输出时，每条记录追加到 `current_row_records`：

```python
current_row_records.append({
    "part_number": pn,
    "color_norm": col_norm,
    "final_price": final_price,
    "recorded_at": rec_at,
    "effective_source": effective_source,     # "abs_price" | "matched_label" | "default_zero"
    "matched_label": matched_label,           # str | None
    "spec_value": spec_value,                 # int | None
})
```

该列表用于生成 Row summary 中的 `abs_applied_details`、`delta_applied_details`、`default_applied_pns`。

### 4.2 Label 追踪 Map

在颜色匹配阶段，除了记录匹配值外，还需同步记录匹配标签：

```python
# delta 类型
color_delta_label_map: Dict[str, str] = {}    # col_norm → label_raw

# abs 类型（仅 shop15/16）
color_abs_label_map: Dict[str, str] = {}      # col_norm → label_raw
```

**用途**：在输出阶段直接从 map 获取 `matched_label`，避免反查逻辑的复杂性和潜在 bug。

---

## 5. Shop 特性差异速查

### 5.1 abs + delta 混合型（shop15 / shop16）

| 特性 | shop15 | shop16 |
|---|---|---|
| base_price 来源 | price 列开头 | 買取価格 列开头 |
| 颜色信息来源 | price 列后半段 | 買取価格 列后半段 |
| 支持 abs 类型 | 是 | 是 |
| 支持 delta 类型 | 是 | 是 |
| 提取模式调度 | `_extract_price_parts_shop15_dispatch` | `_extract_price_parts_shop16_dispatch` |
| specs 统一格式 | `[(label, kind, value)]` | 分开的 `deltas` + `absps` |
| 匹配函数 | `_label_matches_color` | `_label_matches_color_shop16` |

### 5.2 delta-only 型

| 特性 | shop2 | shop3 | shop4 | shop9 | shop11 | shop12 | shop14 | shop17 |
|---|---|---|---|---|---|---|---|---|
| shop_name | 海峡通信 | 買取一丁目 | モバイルミックス | ダイワンテレコム | アメモバ | リンクサスモバイル | じゃんぱら | — |
| base_price 来源 | data3 | data5 | data 行内价格 | — | — | — | — | 独立列 |
| 颜色信息来源 | data5 | 减价1 | data 行内 block | — | — | — | — | 独立列 `色減額` |
| 提取模式 | regex/llm/auto | regex/llm/auto | regex/llm/auto | regex/llm/auto | regex/llm/auto | regex/llm/auto | regex/llm/auto | regex |
| "全色" ALL 机制 | 否 | 否 | 是 | 否 | 否 | 否 | 否 | 否 |
| group→color 匹配 | 是（`_match_color_group`） | 否 | 否 | 否 | 否 | 否 | 否 | 否 |
| abs_prices 字段 | `[]`（固定） | `[]`（固定） | `[]`（固定） | `[]`（固定） | `[]`（固定） | `[]`（固定） | `[]`（固定） | `[]`（预留） |

### 5.3 模块级常量

所有已纳入规范的清洗器均在模块顶部定义以下常量（不再在函数内定义）：

```python
CLEANER_NAME = "shopN"
SHOP_NAME = "店铺日文名"
```

提取函数（`_llm`/`_dispatch`）不再接受 `shop_name`/`cleaner_name`/`row_context` 参数，
而是直接使用模块级常量。需要行级标识时通过 `row_index` 参数传入。

---

## 6. 新增 / 改造清洗器 Checklist

新增 shopN 清洗器或将已有清洗器纳入统一日志框架时，按以下步骤确保规范一致：

1. **模块级常量**
   - 在模块顶部定义 `CLEANER_NAME = "shopN"` 和 `SHOP_NAME = "店铺日文名"`
   - 提取函数（`_llm`/`_dispatch`）不接受 `shop_name`/`cleaner_name`/`row_context` 参数，直接使用模块常量
   - 需要行级标识时使用 `row_index` 参数

2. **`cleaner_start` / `cleaner_complete`（INFO）**
   - 包含 `log_seq`、`extraction_mode`（配置值如 `"auto"`）
   - 不使用 `start_time`/`end_time`，只使用 `elapsed_seconds`

3. **提取阶段 — `extraction_result`（DEBUG）**
   - 使用 `source_text_raw` / `source_text_raw_full` / `source_text_normalized`（不使用自定义名如 `data5_raw`/`color_discount_raw`）
   - 包含 `labels_and_deltas: [{"label": str, "delta": int}]`
   - 包含 `abs_prices: [{"label": str, "amount": int}]`（delta-only shop 固定为 `[]`）
   - 包含 `labels_extracted_count` 和 `abs_prices_count`

4. **匹配阶段 — `label_matching`（DEBUG）/ `label_no_match`（WARNING）**
   - 使用 `color_delta_label_map`（和 `color_abs_label_map` 如有 abs 类型）
   - 每个标签独立发射日志（不 batch 合并）
   - 包含 `match_type: "delta" | "abs"`
   - 消息格式：`"Label matching (delta): {label}"` / `"Label not matched (delta): {label}"`

5. **输出阶段 — `output_record`（DEBUG）**
   - 每条记录使用 `effective_source` / `matched_label` / `spec_value` 三元组
   - `current_row_records` 结构与第 4.1 节一致

6. **Row summary — `row_processing_summary`（DEBUG + INFO）**
   - DEBUG 日志包含 `abs_applied_details` / `delta_applied_details` / `default_applied_pns`
   - INFO 日志格式：`Row {idx} | {model} | deltas: {N} | abs: {N} | matched: {N} | records: {N} | method: {method}`

7. **字段命名一致性**
   - 不使用 `delta_source`（旧字段名），统一用 `effective_source`
   - 不使用 `adjustment`（旧字段名），统一用 `spec_value`
   - 不使用 `data5_raw`/`color_discount_raw`/`block_text_full` 等自定义名，统一用 `source_text_raw_full`
   - 不在日志中输出计算得出的 `delta`（`final_price - base_price`），而是输出 `spec_value`（原始提取值）
   - `matched_label` 从 label map 获取，不做反查

---

## 7. 字段废弃说明

以下旧字段已从日志输出中移除，如在 ELK 中发现历史数据仍包含这些字段属于正常：

| 旧字段 | 替代 | 原因 |
|---|---|---|
| `delta_source` | `effective_source` | 命名统一 |
| `delta`（output_record / row_summary 中） | `spec_value` | abs 类型时 delta 是反推值，有歧义 |
| `adjustment`（shop2 特有） | `spec_value` | 与其他 shop 的 spec_value 统一 |
| `current_row_records`（Row summary 中） | `abs_applied_details` + `delta_applied_details` + `default_applied_pns` | 按类型分组更清晰 |
| `data5_raw` / `data5_raw_full`（shop2） | `source_text_raw` / `source_text_raw_full` | 跨 shop 统一字段名 |
| `color_discount_raw` / `color_discount_raw_full`（shop3） | `source_text_raw` / `source_text_raw_full` | 跨 shop 统一字段名 |
| `block_text_preview` / `block_text_full`（shop4） | `source_text_raw` / `source_text_raw_full` | 跨 shop 统一字段名 |
| `adjustments`（shop4 extraction_result 中） | `labels_and_deltas` | 结构化统一 |
| `parsed_rules` / `rules_count`（shop2） | `labels_and_deltas` / `labels_extracted_count` | 结构化统一 |
| `start_time` / `end_time`（cleaner_start/complete 中） | `elapsed_seconds` | 精简冗余字段 |
| `shop_name`/`cleaner_name`/`row_context` 函数参数 | 模块级 `CLEANER_NAME`/`SHOP_NAME` 常量 | 消除参数传递冗余 |
