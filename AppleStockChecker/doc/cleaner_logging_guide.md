# Shop Cleaner 统一日志规范

> 适用范围：shop15 / shop16 / shop17 及后续新增的清洗器
>
> 最后更新：2026-02-11

---

## 1. 设计目标

所有清洗器的 DEBUG/INFO 级日志使用**统一字段名和结构**，使得：

- ELK 可以用相同的查询条件跨 shop 分析提取质量
- 新增 shop 时有明确的日志字段规范可参照
- 日志中清晰区分 **abs（绝对价）** 和 **delta（差额）** 两种定价类型

---

## 2. 核心概念

### 2.1 定价类型（effective_source）

| effective_source | 含义 | 典型场景 |
|---|---|---|
| `"abs_price"` | 颜色有独立的绝对价格，不依赖 base_price | shop15: `ブルー229,000円` |
| `"matched_label"` | 颜色有相对于 base_price 的差额 | shop15: `ブルー-1000円`；shop17: `色減額:シルバー-3000` |
| `"default_zero"` | 未命中任何 spec，直接使用 base_price | 所有颜色均无特殊标注时 |

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
abs_prices: [{"label": str, "amount": int}]             # abs 类型的提取结果（仅 shop15/16；shop17 无此字段）
labels_extracted_count: int                             # delta 标签数量
abs_prices_count: int                                   # abs 标签数量（仅 shop15/16；shop17 无此字段）
available_colors: [{"color_norm", "part_number", "color_raw"}]
colors_in_catalog: int
```

### 3.2 `label_matching`（DEBUG）

每个提取到的标签的颜色匹配详情。shop15/16 有 delta 和 abs 两个变体，shop17 仅 delta。
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

| 特性 | shop15 | shop16 | shop17 |
|---|---|---|---|
| base_price 来源 | price 列开头 | 買取価格 列开头 | 独立列 `新未開封品` |
| 颜色信息来源 | price 列后半段 | 買取価格 列后半段 | 独立列 `色減額` |
| 支持 abs 类型 | 是 | 是 | 否（预留字段） |
| 支持 delta 类型 | 是 | 是 | 是 |
| 提取模式调度 | `_extract_price_parts_shop15_dispatch` | `_extract_price_parts_shop16_dispatch` | `_extract_color_deltas_shop17` |
| specs 统一格式 | `[(label, kind, value)]` | 分开的 `deltas` + `absps` | `[(label, delta)]` |
| 匹配函数 | `_label_matches_color` | `_label_matches_color_shop16` | `_label_matches_color_shop17` |

---

## 6. 新增清洗器 Checklist

新增 shopN 清洗器时，按以下步骤确保日志规范一致：

1. **定义常量**
   - `CLEANER_NAME = "shopN"`
   - `SHOP_NAME = "店铺日文名"`

2. **提取阶段**
   - 产出 `extraction_result` 日志（DEBUG）
   - 包含 `extraction_method`、`labels_and_deltas`、`abs_prices`（如有）

3. **匹配阶段**
   - 使用 `color_delta_label_map`（和 `color_abs_label_map` 如有 abs 类型）
   - 产出 `label_matching`（DEBUG）和 `label_no_match`（WARNING）日志

4. **输出阶段**
   - 每条记录使用 `effective_source` / `matched_label` / `spec_value` 三元组
   - `current_row_records` 结构与第 4.1 节一致
   - 产出 `output_record` 日志（DEBUG）

5. **Row summary**
   - DEBUG 日志包含 `abs_applied_details` / `delta_applied_details` / `default_applied_pns`
   - INFO 日志包含简洁的一行概览

6. **字段命名一致性**
   - 不使用 `delta_source`（旧字段名），统一用 `effective_source`
   - 不在日志中输出计算得出的 `delta`（`final_price - base_price`），而是输出 `spec_value`（原始提取值）
   - `matched_label` 从 label map 获取，不做反查

---

## 7. 字段废弃说明

以下旧字段已从日志输出中移除，如在 ELK 中发现历史数据仍包含这些字段属于正常：

| 旧字段 | 替代 | 原因 |
|---|---|---|
| `delta_source` | `effective_source` | 命名统一 |
| `delta`（output_record / row_summary 中） | `spec_value` | abs 类型时 delta 是反推值，有歧义 |
| `current_row_records`（Row summary 中） | `abs_applied_details` + `delta_applied_details` + `default_applied_pns` | 按类型分组更清晰 |
