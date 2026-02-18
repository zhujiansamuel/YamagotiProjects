# Shop9 / Shop13 输出行数分析 — 总结

**背景**：正常情况输出预期为 43 行（catalog 全量 part_number）。shop9 输出 26 行，shop13 输出 224 行。结合清洗器源码与数据，分析原因。

---

## 一、Catalog 基准

- **color_map**：12 个 (model_norm, capacity_gb)，共 43 个 part_number
- **43 行**：当数据覆盖全量 catalog 且每种颜色只输出一次时，输出应为 43 行

---

## 二、Shop9（アキモバ）— 输出 26 行

### 2.1 清洗器逻辑

- 使用 C 类流水线：`setup_color_cleaner` + `resolve_color_prices`
- 每行输入：`機種名` → (model_norm, cap_gb) → 查 color_map → 解析 `買取価格` + `色・詳細等` 的颜色减价 → 输出每个颜色的 part_number
- **只处理在 color_map 中存在的 (model, cap)**，并按颜色展开

### 2.2 数据情况

- **输入行数**：约 9 行
- **機種名**：8 个不同机型（1 行 nan 被跳过）
  - 例：iPhone17 Pro 256GB, iPhone17 Pro 512GB, iPhone17 Pro 1TB, iPhone17 Pro Max 256/512/1TB/2TB, iPhone17 256GB
- 未包含：iPhone 17 512GB、iPhone Air 等

### 2.3 根因

**数据源本身只覆盖部分机型**，对应 8 个 (model, cap)，合计 26 个 part_number：

- iPhone 17 256GB：5 色
- iPhone 17 Pro 256/512/1TB：各 3 色 → 9
- iPhone 17 Pro Max 256/512/1TB/2TB：各 3 色 → 12  
- **合计**：5 + 9 + 12 = **26**

因此，输出 26 行是符合当前数据与清洗逻辑的，并非错误。

---

## 三、Shop13（家電市場）— 输出 224 行

### 3.1 清洗器逻辑

- 使用 **B 类模板**：`clean_with_model_capacity_matching`
- 与 shop2/3/4/9 等 C 类不同，**不解析颜色减价**，仅按 (model, cap) 展开为全色
- 对每一行输入：从 `買取商品2` 解析 (model, cap)，从 `新品価格` 取价 → 为该 (model, cap) 的 **所有** part_number 各输出一行（价格相同）

### 3.2 数据情况

- **输入行数**：约 312 行
- **買取商品2**：163 个不同值
- **格式**：`"iPhone 14 Pro Max 128GB SIMフリー [スペースブラック]"` 等，**一行对应一种颜色**  
  即：同一 (model, cap) 有多行（每种颜色一行）

### 3.3 根因

**同一 (model, cap) 在输入中被重复展开**：

- 每个输入行解析为 (model, cap) 后，都会展开为该 (model, cap) 的 **全部** part_number
- 对同一 (model, cap)，有多行输入（如 3 种颜色各 1 行）时，会重复展开 3 次
- 结果：64 行有效输入 → 224 行输出，但 **唯一 part_number 仅 43 个**
- `Duplicate (pn, price, ts)` 约 148 行，即同一 (part_number, price_new, recorded_at) 多次输出

数据格式为「按颜色分行」，而 `clean_with_model_capacity_matching` 按行展开、**不按 (model, cap) 去重**，导致同一 part_number 被重复输出。

---

## 四、结论对照

| 项目 | shop9 | shop13 |
|------|-------|--------|
| 输出行数 | 26 | 224 |
| 唯一 part_number | 26 | 43 |
| 与 43 行差异原因 | 数据仅含 8 机型 → 26 色 | 按行展开且无去重 → 43 色重复约 5.2 次 |
| 清洗逻辑 | C 类，按颜色减价展开 | B 类，仅按 (model, cap) 全色展开 |

---

## 五、总结

1. **shop9**：26 行是合理的。数据只覆盖 8 个机型、26 个 part_number，清洗器按现有逻辑正确输出。
2. **shop13**：224 行来自逻辑设计。`clean_with_model_capacity_matching` 对每一行都做全色展开，shop13 的「一行一色」数据导致同一 (model, cap) 被多次展开，产生大量重复的 part_number 行。
