# Shop 两阶段处理逻辑对比总结

各 shop 清洗器的两阶段流水线（`_match_*` → `expand_match_tokens` → `match_tokens_to_specs`）对比分析。

---

## 一、流程概览

| Shop | 店铺名 | 文本来源 | 全色前置 | Fragment | 特殊预处理 |
|------|--------|----------|----------|----------|------------|
| **shop2** | 海峡通信 | data5 | ✅ | 单段 | +++/、→\n |
| **shop3** | 買取一丁目 | 减价1 | ✅ | 单段 | — |
| **shop4** | モバイルミックス | data+data11 | ✅ | 多行block | _collect_block_segments, 円/ 分割 |
| **shop7** | 買取ホムラ | data2(下一行) | ✅ | 单行 | 下一行无价格=颜色行 |
| **shop9** | アキモバ | 買取価格+色・詳細等 | ✅ | 合并2列 | _clean_shop9_text |
| **shop11** | モバステ | caution_empty | ✅ | 单段 | 去括号备注 |
| **shop12** | トゥインクル | 備考1+買取価格 | ✅ | 合并/单段 | 去開封行 |
| **shop14** | 買取楽園 | 减价条件×3列 | ✅ | **多fragment** | 3列分别match后合并 |
| **shop16** | 携帯空間 | 買取価格 | ❌ | 单段 | 去基础价前缀、换行→/ |
| **shop17** | ゲストモバイル | 色減額 | ❌ | 单段 | _pick_unopened_section【未開封】 |

---

## 二、核心组件对比

### 2.1 全色 (all_delta) 检测

| 有全色前置 | 无全色前置 |
|------------|------------|
| shop2, 3, 4, 7, 9, 11, 12, 14 | shop16, shop17 |

- **有**：`_detect_all_delta(text)` → 匹配 `全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?`，有则插入 `("全色", agg_all_delta)` 到 delta_specs 首位
- **无**：shop16/17 未实现该逻辑，可能是数据中无全色表述

### 2.2 文本预处理函数

| Shop | 函数 | 作用 |
|------|------|------|
| shop2 | `_clean_color_text_shop2` | +++/、→\n，归一化空白 |
| shop3 | `_clean_color_text_shop3` | 空白归一化，normalize_text_basic |
| shop4 | `_clean_block_text` | 空白归一化，用于单段 |
| shop7 | `_clean_color_text_shop7` | 空白归一化 |
| shop9 | `_clean_shop9_text` | 合并買取価格+色・詳細等 |
| shop11 | `_clean_caution_frag` | 去括号备注，归一化 |
| shop12 | `_normalize_remark_text` + `_clean_remark_frag` | 去開封行 / 清理片段 |
| shop14 | `_clean_remark_frag` | 清理 remark 片段 |
| shop16 | `_normalize_price_text_shop16` | 换行→/，压缩空白 |
| shop17 | `_normalize_color_text_shop17` | remove_newlines=False, collapse_spaces=False |

### 2.3 SPLIT_TOKENS_RE（part 分割）

| Shop | 模式 | 说明 |
|------|------|------|
| shop2 | `[／/、，,・]\|(?:\s*[;；]\s*)\|\n` | 含・、； |
| shop3 | 同上 | 与 shop2 一致 |
| shop4 | `[／/、，]\|(?:\s*;\s*)\|\n` | 不含・ |
| shop7 | 同上 | 与 shop4 一致 |
| shop9 | `[／/、，]\|(?:\s*;\s*)\|;\|；\|\n` | 多显式分号 |
| shop11 | `[／/、，]\|(?:\s*;\s*)\|\n` | 与 shop4 一致 |
| shop12 | `[／/、，]\|(?:\s*;\s*)\|\n` | 同上 |
| shop14 | `\s*(?:、\|，\|／\|/\|;\|；)\s*` | 不含 \n |
| shop16 | LABEL_SPLIT_RE_shop16 | `[／/、，,]\|(?:\s*;\s*)` |
| shop17 | LABEL_SPLIT_RE_shop17 | `[／/、]\|(?:\s*;\s*)\|\n` |

### 2.4 正则模式 (NONE_RE / DELTA_RE / ABS_RE)

**共同点**：所有 shop 均有 NONE_RE（なし）、DELTA_RE（±金额）、ABS_RE（￥金额），结构一致。

**差异**：

| Shop | DELTA_RE | LOOSE fallback |
|------|----------|----------------|
| shop3 | 单 pattern | ✅ COLOR_DELTA_RE_shop3_LOOSE |
| shop2,4,7,9,11,12,14,16,17 | 单 pattern | ❌ |

- shop3 使用 STRICT→LOOSE 双模式，其余多为单一 DELTA_RE

### 2.5 阶段 1 匹配顺序

统一顺序：**NONE_RE** → **ABS_RE** → **DELTA_RE** → **pending_labels**

- `pending_labels`：仅标签无金额时先挂起，等下一 part 的金额再绑定

### 2.6 format_hint 分支（DELTA_RE）

| sign | sep | hint |
|------|-----|------|
| 有 ± | — | FORMAT_HINT_SIGNED |
| 无，sep 为 - | — | FORMAT_HINT_SEP_MINUS |
| 无，sep 为 ： | — | FORMAT_HINT_COLON_PREFIX |
| 无 sign/sep | — | FORMAT_HINT_PLAIN_DIGITS |

---

## 三、数据源与结构差异

### 3.1 单行/单段

- **shop2**：data5 单单元格
- **shop3**：减价1 单单元格
- **shop7**：下一行 data2（无价格为颜色行）
- **shop11**：caution_empty 单单元格
- **shop16**：買取価格 单单元格
- **shop17**：色減額 单单元格

### 3.2 多列合并

- **shop9**：買取価格 + 色・詳細等 拼接
- **shop12**：備考1 + 買取価格

### 3.3 多行 block

- **shop4**：`_collect_block_segments` 按机种 block 收集多行，按 `円/` 分割成段

### 3.4 多 fragment

- **shop14**：减价条件、减价条件2、23432 三列各自 `_match_shop14`，tokens 合并后再 expand + match_tokens_to_specs；全色在任一 frag 或 combined 中检测

---

## 四、基准价 (base_price) 来源

| Shop | 来源 |
|------|------|
| shop2 | extract_price_yen(data3) |
| shop3 | extract_price_yen(data5) |
| shop4 | `_find_base_price` 回溯上一行/上 3 行 |
| shop7 | extract_price_yen(data3) |
| shop9 | extract_price_yen(買取価格) |
| shop11 | extract_price_yen(price_unopened) |
| shop12 | extract_price_yen(買取価格) |
| shop14 | to_int_yen(price2) |
| shop16 | `_extract_base_price_shop16` 从買取価格文本提取 |
| shop17 | extract_price_yen(新未開封品) |

---

## 五、expand_match_tokens 配置

| Shop | enable_adaptive |
|------|-----------------|
| shop2 | True |
| shop3 | True |
| shop4 | True |
| shop7 | True |
| shop9 | True |
| shop11 | True |
| shop12 | True |
| shop14 | True |
| shop16 | ENABLE_ADAPTIVE_SPLIT_SHOP16 (env, 默认 true) |
| shop17 | ENABLE_ADAPTIVE_SPLIT_SHOP17 (env, 默认 true) |

---

## 六、resolve_color_prices 差异

| Shop | emit_default_rows | skip_non_positive |
|------|-------------------|-------------------|
| shop2 | True | **True** |
| 其他 | True | False (默认) |

- shop2 对非正价格有特殊策略

---

## 七、特有逻辑摘要

| Shop | 特有逻辑 |
|------|----------|
| shop2 | SIMfree+未開封过滤，列名小写，lenient，cmap_filtered |
| shop4 | block 结构、`_find_base_price`、`_collect_block_segments`、`円/` 分割 |
| shop7 | 颜色行=下一行 data2 且无价格，`_norm_model_for_shop7` |
| shop9 | 合并 2 列，`_direct_abs_overrides`（已注释） |
| shop12 | 去開封行预处理 |
| shop14 | 3 列 fragment 分别 match 再合并 |
| shop16 | 基础价从文本提取、去前缀，`_GROUP_SHARED_DELTA_RE` |
| shop17 | `_pick_unopened_section`【未開封】段落提取 |

---

## 八、统一性与建议

### 已统一

- NONE_RE / DELTA_RE / ABS_RE 结构
- `expand_match_tokens` + `match_tokens_to_specs` 调用方式
- `_label_matches_color_unified` 颜色匹配
- `_is_plausible_color_label_*` 过滤逻辑
- `_normalize_label_*` 标签归一化

### 可改进

1. **shop16/17 全色**：若实际数据有「全色±N」，可补充 `_detect_all_delta`。
2. **shop3 LOOSE**：其余 shop 若遇宽松格式，可考虑增加 LOOSE fallback。
3. **SPLIT_TOKENS_RE**：各 shop 略有差异，可按数据格式收敛到少数几种。
4. **环境变量**：shop16/17 的 `SHOP*_ADAPTIVE_SPLIT` 可考虑统一为 cleaner_tools 配置。
