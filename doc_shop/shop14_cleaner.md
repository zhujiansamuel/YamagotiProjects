# Shop14 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop14_cleaner.py`
> 店铺名称: 買取楽園

---

## 一、总流程图

整个 shop14 清洗器的核心入口是 `clean_shop14(df)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\nname / data6 / price2 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| B2[模糊解析备注列名\n_resolve_remark_cols\n减价条件 / 减价条件2 / 23432]
    B2 --> C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建颜色映射表\n_build_color_map]
    D --> E[逐行遍历 DataFrame]

    E --> F{data6 列\n是否含未開封?}
    F -->|否| E
    F -->|是| G[型号标准化\n_normalize_model_generic]

    G --> H{型号/容量\n能否解析?}
    H -->|否| E
    H -->|是| I[在 color_map 中\n查找该机型]

    I --> J{color_map\n是否存在?}
    J -->|否| E
    J -->|是| K[解析基准价格\nto_int_yen price2]

    K --> L{基准价格\n是否有效?}
    L -->|否| E
    L -->|是| M[读取3个备注列\n清洗合并文本片段]

    M --> N[逐列调用 LangExtract\n_shop14_extract_rules_with_langextract]
    N --> O[汇总提取结果\nagg_all_delta / agg_abs / agg_delta]

    O --> P{逐列均无结果\n且合并串非空?}
    P -->|是| P1[用合并串再跑一次 LangExtract]
    P -->|否| Q{全色规则\nagg_all_delta?}
    P1 --> Q

    Q -->|是| R[所有颜色统一价格\nprice = base + all_delta]
    Q -->|否| S[逐颜色匹配 abs/delta\n_label_matches_color_unified]

    S --> T[计算每个颜色的最终价格\n绝对价优先 > base+delta > base]
    R --> U[生成输出行\npart_number / shop_name / price_new / recorded_at]
    T --> U
    U --> E

    E -->|遍历结束| V[组装输出 DataFrame]
    V --> W[去除空值 / 类型转换]
    W --> X[输出: 标准化 DataFrame\npart_number, shop_name, price_new, recorded_at]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop14(df)"]

    clean --> resolve["_resolve_remark_cols(df)"]
    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> buildcm["_build_color_map(info_df)"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> toint["to_int_yen(val)"]
    clean --> cleanfrag["_clean_remark_frag(val)"]
    clean --> lxrules["_shop14_extract_rules_with_langextract(text)"]
    clean --> labelmatch["_label_matches_color_unified(label, color_raw, color_norm)"]
    clean --> parsedt["parse_dt_aware(val)"]

    lxrules --> prompt["_shop14_lx_prompt_and_examples()"]
    lxrules --> cleanfrag
    lxrules --> splitpairs["_split_color_amount_pairs_multi(txt)"]
    lxrules --> coerce["_coerce_amount_yen(v)"]
    lxrules --> splitlbl["_split_labels(labels_str)"]
    lxrules --> lblfallback["_labels_from_text_fallback(txt)"]
    lxrules --> lxextract["langextract.extract()"]

    splitpairs --> coerce
    splitpairs --> pairre["PAIR_RE_MULTI 正则"]

    resolve --> normcol["_norm_colname(x)"]
    buildcm --> normmod
    labelmatch --> norm["_norm(s)"]
    labelmatch --> familysyn["FAMILY_SYNONYMS_shop14 字典查表"]

    clean --> hasall["_has_all_colors(text)"]
    clean --> extractdelta["_extract_color_deltas_shop14(text)"]
    clean --> extractabs["_extract_color_abs_prices(text)"]
    extractdelta --> normlbl["_norm_label(lbl)"]
    extractabs --> normlbl
    clean --> repair["_repair_abs_delta_from_compound_text(text)"]
    repair --> splitlbl
    repair --> striplbl["_strip_label_delims(s)"]
```

### 2.2 核心函数详细说明

#### `clean_shop14(df: pd.DataFrame) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `name`, `data6`, `price2`, `time-scraped` 列的 DataFrame，以及可选的备注列 (`减价条件`, `减价条件2`, `23432`)
- **输出**: 包含 `part_number`, `shop_name`("買取楽園"), `price_new`, `recorded_at` 列的 DataFrame
- **关键逻辑**: 逐行遍历，先过滤 `data6` 含"未開封"的行，再对每个备注列分别调用 LangExtract 提取规则，最后按"全色优先 > 绝对价优先 > base+delta > base"策略计算价格

#### `_resolve_remark_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]`
- **作用**: 模糊匹配备注列名，兼容 BOM 前缀、全角空格等列名差异
- **策略**: 先精确匹配（去 BOM/空格归一化后），再包含匹配（fuzzy）
- **返回**: `{"减价条件": 实际列名或None, "减价条件2": ..., "23432": ...}`

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ->Pro) / 紧凑写法展开 (17pro->17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB->GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_shop14_extract_rules_with_langextract(text: str) -> Dict`
- **作用**: 使用 LangExtract(Ollama) 从文本中抽取价格规则，带 `@lru_cache(maxsize=4096)` 缓存
- **返回结构**:
  ```python
  {
      "all_delta": Optional[int],      # 全色统一调整额
      "abs":   List[(label, price)],    # 绝对价规则
      "delta": List[(label, delta)],    # 差额规则
      "raw":   List[dict],              # 原始 extraction 列表
  }
  ```
- **三类 extraction_class**: `all_colors`, `abs_group`, `delta_group`
- **流程**:

```mermaid
flowchart TD
    A["_shop14_extract_rules_with_langextract(text)"] --> B["_clean_remark_frag(text)"]
    B --> C{清洗后为空?}
    C -->|是| D["返回空结果 {}"]
    C -->|否| E["调用 lx.extract()"]

    E --> E1["配置参数:\nmodel_id = gemma3:1b\nmodel_url = localhost:11434\nprompt = 价格规则抽取提示词\nexamples = 7个 few-shot 示例"]

    E1 --> F{调用成功?}
    F -->|TypeError| F1[兼容旧版 API 重试]
    F1 --> G[遍历 extractions]
    F -->|成功| G

    G --> H{extraction_text\n含多组颜色+金额?}
    H -->|是| I["_split_color_amount_pairs_multi\n拆分多对并推断 abs/delta"]
    H -->|否| J{class 含 all\n或 text 含 全色?}

    J -->|是| K["设置 all_delta"]
    J -->|否| L[提取 labels + amount_yen]

    L --> M{labels 为空?}
    M -->|是| N["_labels_from_text_fallback\n从 extraction_text 兜底"]
    M -->|否| O{根据 class 或\namount 大小推断类型}
    N --> O

    O -->|abs| P["追加到 abs_list"]
    O -->|delta| Q["追加到 delta_list"]
    I --> G
    K --> G
    P --> G
    Q --> G

    G -->|遍历完| R["返回 {all_delta, abs, delta, raw}"]
```

#### `_shop14_lx_prompt_and_examples() -> Tuple[str, List]`
- **作用**: 返回 LangExtract 的 prompt 和 7 个 few-shot 示例（带 `@lru_cache(maxsize=1)` 只初始化一次）
- **7 个 few-shot 示例**:
  1. `"青 229,500"` -> abs_group
  2. `"橙 -2500"` -> delta_group
  3. `"全色 -3,000円"` -> all_colors
  4. `"青/銀 229,500"` -> abs_group (多标签)
  5. `"橙/銀 -2,500円"` -> delta_group (多标签)
  6. `"青 229,500\n橙 -2500"` -> abs_group + delta_group (混合)
  7. `"全色"` -> all_colors (无金额, amount_yen=0)

#### `_split_color_amount_pairs_multi(txt: str) -> List[Tuple[str, int]]`
- **作用**: 处理复合文本如 `"橙227000、青228000"`，拆分为多个 (label, amount) 对
- **返回条件**: 仅当检测到 >=2 个"颜色+金额"对时返回非空列表
- **示例**:
  - `"橙227000、青228000"` -> `[("橙", 227000), ("青", 228000)]`
  - `"青229500/銀228000"` -> `[("青", 229500), ("銀", 228000)]`

#### `_repair_abs_delta_from_compound_text(text: str) -> Tuple[List, List]`
- **作用**: 从复合文本串中恢复绝对价和差额列表
- **判断逻辑**: 有显式正负号 -> delta_list；无符号 -> abs_list
- **返回**: `(abs_list, delta_list)`

#### `_extract_color_deltas_shop14(text: str) -> List[Tuple[str, int]]`
- **作用**: 从备注文本中提取 (label_raw, delta_int) 差额对
- **分隔策略**: 使用安全切分 `_SPLIT_TOKENS_SAFE_RE`，避免拆千位分隔符 (如 `2,000`)

#### `_extract_color_abs_prices(text: str) -> List[Tuple[str, int]]`
- **作用**: 从文本中抽取绝对价格 (label_raw, abs_price)
- **特殊处理**: 跳过含 +/- 符号的片段（留给差额解析器处理）；支持多标签共用一个金额（如 `"青/銀327000"`）

#### `_has_all_colors(text: str) -> Optional[int]`
- **作用**: 检测文本是否包含"全色"关键词
- **返回**: 含金额返回 delta 整数，仅含"全色"返回 0，未出现返回 None

#### `_label_matches_color_unified(label_raw, color_raw, color_norm) -> bool`
- **作用**: 判断提取到的颜色标签是否匹配 info 表中的某个颜色
- **匹配策略** (三级宽松匹配):

```mermaid
flowchart TD
    A["输入: label_raw, color_raw, color_norm"] --> B{精确匹配?\nlabel归一 == color_norm}
    B -->|是| Z[返回 True]
    B -->|否| C{子串匹配?\nlabel_raw in color_raw}
    C -->|是| Z
    C -->|否| D[查 FAMILY_SYNONYMS_shop14\n颜色家族同义词表]
    D --> E{label 在字典 key 中?}
    E -->|是| F[获取同义词列表]
    E -->|否| G[反向查: 遍历所有家族\n找包含 label 的条目]
    F --> H{同义词中任一\n出现在 color_raw 中?}
    G --> H
    H -->|是| Z
    H -->|否| Y[返回 False]
```

#### `_build_color_map(info_df) -> Dict`
- **作用**: 构建 `(model_norm, capacity_gb) -> {color_norm: (part_number, color_raw)}` 映射
- **数据源**: iphone17_info 参考表

#### `_clean_remark_frag(x) -> str`
- **作用**: 清洗单个备注字段：去 BOM/全角空格/NBSP，合并多余空白，过滤 nan

#### `_norm_colname(x) -> str`
- **作用**: 归一化列名：去 BOM 前缀、全角空格转半角、去两端空白

#### `_coerce_amount_yen(v) -> Optional[int]`
- **作用**: 将 LLM attributes 或文本中的金额字符串转为整数（支持符号、逗号、円、¥ 等格式）

#### `_split_labels(labels: str) -> List[str]`
- **作用**: 将 `"青/銀"` / `"青、銀"` / `"青 銀"` 等拆为列表 `["青", "銀"]`

#### `_labels_from_text_fallback(extraction_text: str) -> str`
- **作用**: 当 LLM 未返回 labels 属性时，从 extraction_text 中去掉金额部分和"全色"，剩余部分当作颜色标签

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: name, data6, price2, time-scraped\n+ 备注列: 减价条件, 减价条件2, 23432"]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color"]
    end

    subgraph 中间数据结构
        REMAP["remark_cols_map\n{逻辑列名 -> 实际列名}\n经 _resolve_remark_cols 模糊解析"]
        CMAP["color_map 字典\n(model_norm, cap_gb) ->\n  color_norm: part_number, color_raw"]
        FRAGS["frags 字典\n{减价条件: 清洗文本,\n 减价条件2: 清洗文本,\n 23432: 清洗文本}"]
        LXOUT["LangExtract 提取结果\nagg_all_delta: Optional int\nagg_abs: List of label+price\nagg_delta: List of label+delta"]
        CABS["color_abs 字典\n{color_norm: abs_price}"]
        CDELTA["color_deltas 字典\n{color_norm: delta}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> CMAP
    RAW --> REMAP
    RAW -->|"逐行读取"| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["data6 过滤: 仅保留含未開封的行"]
        P2["name -> model_norm + cap_gb"]
        P3["price2 -> base_price via to_int_yen"]
        P4["3个备注列 -> frags 清洗文本"]
        P5["逐列 LangExtract -> agg 汇总"]
        P6["label 匹配 color_map -> color_abs / color_deltas"]
        P7["价格计算: abs优先 > base+delta > base"]
    end

    REMAP --> PROC
    CMAP --> PROC
    PROC --> FRAGS
    FRAGS --> LXOUT
    LXOUT --> CABS
    LXOUT --> CDELTA
    CABS --> OUT
    CDELTA --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  name         = "iPhone17 Pro Max 256GB SIMフリー"
  data6        = "未開封"
  price2       = "230,000"
  time-scraped = "2025-06-01 12:00:00"
  减价条件      = ""
  减价条件2     = "橙 -2500"
  23432        = "青 229,500"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 过滤 + 型号解析"]
        T0["data6='未開封' -> 通过过滤"]
        T1["'iPhone17 Pro Max 256GB SIMフリー'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step2["Step 2: 查询 color_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 cmap_all| T5["{\n  'ブラックチタニウム': ('MYW23J/A', ...),\n  'ホワイトチタニウム': ('MYW53J/A', ...),\n  'ナチュラルチタニウム': ('MYW83J/A', ...),\n  ...\n}"]
    end

    subgraph Step3["Step 3: 基准价格"]
        T6["price2 = '230,000'"]
        T6 -->|to_int_yen| T7["230000"]
    end

    subgraph Step4["Step 4: 备注列清洗 + LangExtract"]
        T8a["减价条件 = '' -> 跳过"]
        T8b["减价条件2 = '橙 -2500'"]
        T8c["23432 = '青 229,500'"]
        T8b -->|_shop14_extract_rules_with_langextract| T9b["delta=[('橙', -2500)]"]
        T8c -->|_shop14_extract_rules_with_langextract| T9c["abs=[('青', 229500)]"]
        T9b --> T10["汇总: agg_abs=[('青',229500)]\nagg_delta=[('橙',-2500)]"]
        T9c --> T10
    end

    subgraph Step5["Step 5: 颜色匹配 + 价格计算"]
        T13["对 color_map 中每个颜色:"]
        T13 --> T14["青系 -> abs匹配 label '青' -> price=229500"]
        T13 --> T15["橙系 -> delta匹配 label '橙' -> price=230000+(-2500)=227500"]
        T13 --> T16["其他颜色 -> 未匹配 -> price=230000(base)"]
    end

    subgraph Step6["Step 6: 输出行"]
        T17["{\n  part_number: 'MYWxxx',\n  shop_name: '買取楽園',\n  price_new: 229500,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYWyyy',\n  shop_name: '買取楽園',\n  price_new: 227500,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
```

### 3.3 LangExtract 三类提取策略

```mermaid
flowchart TD
    INPUT["备注列原始文本\n(减价条件 / 减价条件2 / 23432)"]

    INPUT --> CLEAN["_clean_remark_frag 清洗"]
    CLEAN --> LX["_shop14_extract_rules_with_langextract\n(带 @lru_cache 4096 缓存)"]

    LX --> CLASS{"extraction_class?"}

    CLASS -->|all_colors| AC["全色规则\n所有颜色统一调整\nprice = base + amount_yen"]
    CLASS -->|abs_group| AG["绝对价规则\n指定颜色使用绝对价\nprice = amount_yen"]
    CLASS -->|delta_group| DG["差额规则\n指定颜色在基准上加减\nprice = base + amount_yen"]

    subgraph 多对检测
        MP["_split_color_amount_pairs_multi\n如 '橙227000、青228000'"]
        MP --> MPJ{所有金额 >= 20000?}
        MPJ -->|是| MPA["当作 abs_group"]
        MPJ -->|否| MPD{所有金额 <= 20000?}
        MPD -->|是| MPDD["当作 delta_group"]
        MPD -->|否| MPMIX["按多数判断"]
    end

    LX --> MP
```

### 3.4 价格计算优先级策略

```mermaid
flowchart TD
    START["汇总后的提取结果"]
    START --> Q1{agg_all_delta\n不为 None?}

    Q1 -->|是| ALL["全色统一价格\n对 color_map 中所有颜色:\nprice = base_price + all_delta"]

    Q1 -->|否| Q2["逐颜色匹配\n_label_matches_color_unified"]

    Q2 --> MATCH["对 color_map 中每个颜色:"]
    MATCH --> Q3{color_norm\n在 color_abs 中?}

    Q3 -->|是| ABS["使用绝对价\nprice = color_abs 中的值"]
    Q3 -->|否| Q4{color_norm\n在 color_deltas 中?}

    Q4 -->|是| DELTA["使用差额\nprice = base_price + delta"]
    Q4 -->|否| BASE["使用基准价\nprice = base_price"]
```

### 3.5 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY_SYNONYMS_shop14
        BLUE["blue 家族\nブルー / 青"]
        BLACK["black 家族\nブラック / 黒"]
        WHITE["white 家族\nホワイト / 白"]
        GREEN["green 家族\nグリーン / 緑"]
        RED["red 家族\nレッド / 赤"]
        ORANGE["orange 家族\nオレンジ / 橙"]
        SILVER["silver 家族\nシルバー / 銀"]
        GOLD["gold 家族\nゴールド / 金"]
        OTHER["... 其他家族\npink / purple / yellow / gray / natural"]
    end

    LABEL["提取到的 label\n如: '青'"]
    COLOR["info表中的 color\n如: 'ディープブルー'"]

    LABEL -->|"查 FAMILY_SYNONYMS"| BLUE
    BLUE -->|"同义词 'ブルー' in 'ディープブルー'"| MATCH["匹配成功!"]
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`。

| 配置项/环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EXTRACTION_MODE` | `"regex"` | regex / llm / auto（cleaner_tools） |
| `OLLAMA_URL` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `SHOP14_DEBUG` | `"True"` (启用) | 是否启用 debug 打印输出 |
| `IPHONE17_INFO_CSV` | 自动推断路径 (从 `__file__` 往上两级的 `data/iphone17_info.csv`) | iphone17_info 参考文件路径 |
| `lru_cache(maxsize=4096)` | 4096 | `_shop14_extract_rules_with_langextract` 的缓存大小 |
| `lru_cache(maxsize=1)` | 1 | `_shop14_lx_prompt_and_examples` 的缓存大小（只初始化一次） |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max`, `iPhone16Plus` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
| `COLOR_DELTA_RE_shop14` | `(?P<label>[^：:\-\s/、／]+)\s*(?P<sep>[：:\-])\s*(?P<sign>[+\-−－])?\s*(?P<amount>\d[\d,]*)\s*(円)?` | 匹配"颜色±金额"差额 | `ブルー：-2,000円`, `青-3000`, `銀+1000` |
| `_SPLIT_TOKENS_SAFE_RE` | `[／/、，]\|(?<!\d),(?!\d)\|(?:\s+\+\s+)\|(?:\s*;\s*)` | 安全拆分多条目（避免拆千位逗号） | `青-3000、橙+1000` |
| `_COLOR_ABS_PRICE_RE` | `(?P<label>...)(?:¥\|￥)?\s*(?P<amount>\d{1,3}(?:[,]\d{3})*\|\d+)\s*(?:円)?` | 匹配"颜色+绝对价格" | `青229,500`, `コズミックオレンジ227000` |
| `PAIR_RE_MULTI` | `([^\d¥円,，＋+－\-−\s]+)\s*([+\-−－]?\s*\d[\d,，]*)` | 从复合文本中拆多个颜色+金额对 | `橙227000、青228000` |
| `_PAIR_GROUP_RE_shop14` | `(?P<labels>[^\d¥￥円:+\-−－＋]+?)\s*(?P<sign>[+\-−－＋])?\s*(?P<amount>...)` | 从复合串中恢复 abs/delta 列表 | `青/銀229000、橙227000` |
| `SPLIT_TOKENS_RE` | `[／/、，,]\|(?:\s+\+\s+)\|(?:\s*;\s*)` | 基础分隔符拆分 | `青/銀+1000` |
