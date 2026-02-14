# Shop15 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop15_cleaner.py`
> 店铺名称: 買取当番

---

## 一、总流程图

整个 shop15 清洗器的核心入口是 `clean_shop15(df, debug)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。与 shop17 不同，shop15 的 `price` 列同时包含基准价格和颜色规格（差额或绝对价），需要通过 LLM 提取两类信息（`base_price` + `color_spec`）。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\nprice / data2 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建颜色映射表\n_build_color_map]
    D --> E[逐行遍历 DataFrame]

    E --> F{data2 列是否为空?}
    F -->|空| E
    F -->|非空| G[型号标准化\n_normalize_model_generic]

    G --> H{型号/容量\n能否解析?}
    H -->|否| E
    H -->|是| I[在 cmap_all 中\n查找该机型+容量]

    I --> J{color_map\n是否存在?}
    J -->|否| E
    J -->|是| K[LLM 解析 price 列\n_parse_shop15_price_via_langextract\n提取 base_price + specs]

    K --> L[Post-LLM 纠偏\n_coerce_specs_shop15\n_augment_multi_label_block_specs_shop15]
    L --> M[构建颜色价格映射\n_build_color_prices_from_specs_shop15]

    M --> N[遍历 color_map 中每个颜色\n生成输出行]
    N --> O[part_number / shop_name / price_new / recorded_at]
    O --> E

    E -->|遍历结束| P[组装输出 DataFrame]
    P --> Q[去除空值 / 类型转换]
    Q --> R[去重: 同一 part_number+shop_name\n只保留最新 recorded_at]
    R --> S[输出: 标准化 DataFrame\npart_number, shop_name, price_new, recorded_at]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop15(df, debug)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> buildcm["_build_color_map(info_df)"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> parseprice["_parse_shop15_price_via_langextract(price_text, ...)"]
    clean --> buildcp["_build_color_prices_from_specs_shop15(color_map, base, specs)"]
    clean --> parsedt["parse_dt_aware(val)"]
    clean --> debugchk["_shop15_debug_enabled(debug)"]

    parseprice --> cached["_parse_shop15_price_via_langextract_cached(s, model_id, model_url)\n@lru_cache(maxsize=4096)"]
    parseprice --> extractbase["_extract_base_price_at_start(text)\n(LLM 未给 base 时兜底)"]
    parseprice --> coerce["_coerce_specs_shop15(price_text, base, specs)"]
    parseprice --> augment["_augment_multi_label_block_specs_shop15(price_text, specs)"]

    cached --> lxextract["lx.extract()\nLangExtract 调用"]
    cached --> examples["_shop15_langextract_examples()\n5个 few-shot 示例"]
    cached --> iterext["_iter_extractions_in_order(result)"]
    cached --> parseyen["_parse_signed_int_yen(s)"]
    cached --> cleanlbl["_clean_label_shop15(label)"]

    coerce --> signed["_extract_signed_amount_after_label_shop15(price_text, label)"]

    augment --> splitlbl["_split_color_labels_shop15(label_blob)"]
    augment --> cleanlbl

    buildcp --> labelmatch["_label_matches_color_unified(label, color_raw, color_norm)"]
    labelmatch --> familysyn["FAMILY_SYNONYMS 字典查表"]

    buildcm --> normmod
```

### 2.2 核心函数详细说明

#### `clean_shop15(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `price`, `data2`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`("買取当番"), `price_new`, `recorded_at` 列的 DataFrame
- **去重策略**: 同一 `(part_number, shop_name)` 只保留最新 `recorded_at`

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_parse_shop15_price_via_langextract(price_text, model_id, model_url, debug) -> Tuple[Optional[int], List[Tuple[str, str, int]]]`
- **作用**: 解析 price 列的调度函数，整合 LLM 结果 + 后处理纠偏
- **返回**: `(base_price, specs)` 其中 specs 为 `[(label, kind, yen_value), ...]`

```mermaid
flowchart TD
    A["_parse_shop15_price_via_langextract(price_text)"] --> B["_parse_shop15_price_via_langextract_cached()\n@lru_cache(4096)"]
    B --> C{LLM 返回\nbase_price?}
    C -->|None| D["_extract_base_price_at_start()\n正则兜底提取 base"]
    C -->|有值| E[保留 LLM 的 base_price]
    D --> F["_coerce_specs_shop15()\n纠偏: abs负数→delta\n原文验证覆盖"]
    E --> F
    F --> G["_augment_multi_label_block_specs_shop15()\n多标签块增强/覆盖"]
    G --> H["返回 (base_price, specs)"]
```

#### `_parse_shop15_price_via_langextract_cached(price_text, model_id, model_url) -> Tuple[Optional[int], List[Tuple[str, str, int]]]`
- **作用**: 带 LRU 缓存的 LangExtract 调用核心
- **缓存**: `@lru_cache(maxsize=4096)` 避免相同 price_text 重复调用 LLM
- **提取两类实体**:
  - `extraction_class="base_price"` → 基准价格
  - `extraction_class="color_spec"` → 颜色规格 (带 `kind` 属性: `"delta"` 或 `"abs"`)

```mermaid
flowchart TD
    A[输入 price_text] --> B{LangExtract 可用?}
    B -->|否| C["返回 (None, [])"]
    B -->|是| D["构建 5 个 few-shot 示例\n_shop15_langextract_examples()"]
    D --> E["调用 lx.extract()"]

    E --> E1["配置参数:\nmodel_id = gemma3:1b\nmodel_url = localhost:11434\ntemperature = 0.0\nprompt = SHOP15_PRICE_PROMPT\nexamples = 5个 few-shot 示例\nfence_output = False\nuse_schema_constraints = False"]

    E1 --> F{调用成功?}
    F -->|TypeError| F2["兼容调用: 去掉 temperature 参数"]
    F -->|其他异常| C
    F -->|成功| G["_iter_extractions_in_order(result)\n按文本位置排序"]
    F2 --> G

    G --> H[遍历 extractions]
    H --> I{extraction_class?}
    I -->|base_price| J["提取 yen 属性\n_parse_signed_int_yen → base_price"]
    I -->|color_spec| K["提取 label / kind / yen\n_clean_label_shop15 + _parse_signed_int_yen"]
    I -->|其他| H

    K --> L{kind 合法?\ndelta 或 abs}
    L -->|否| M["根据 yen 字符串中\n是否有 +/- 推断 kind"]
    L -->|是| N["添加到 specs\n(label, kind, value)"]
    M --> N
    J --> H
    N --> H

    H -->|遍历完| O["返回 (base_price, specs)"]
```

#### `_coerce_specs_shop15(price_text, base_price, specs, debug) -> List[Tuple[str, str, int]]`
- **作用**: 对 LLM 输出的 specs 做纠偏（针对小模型不稳定输出）
- **纠偏规则**:

```mermaid
flowchart TD
    A[遍历 specs 中每个 spec] --> B{规则1: kind=abs\n且 value 小于 0?}
    B -->|是| C["强制 kind → delta"]
    B -->|否| D[保持不变]
    C --> E{规则2: 原文中 label 后\n出现 +/- 金额?}
    D --> E
    E -->|是| F["强制 kind → delta\n用原文的 signed 金额覆盖 value"]
    E -->|否| G[保持当前值]
    F --> H[输出纠偏后的 spec]
    G --> H
```

#### `_augment_multi_label_block_specs_shop15(price_text, specs, debug) -> List[Tuple[str, str, int]]`
- **作用**: 处理"多颜色共享一个差额"的表达模式
- **典型输入**: `"オレンジ、ブルー-1000円"`, `"シルバー、ブルー-3000円"`
- **流程**:

```mermaid
flowchart TD
    A[输入 price_text] --> B["用 MULTI_LABEL_DELTA_BLOCK_RE_shop15\n正则查找所有 block"]
    B --> C[遍历每个 block]
    C --> D["提取 label_blob / sign / amount"]
    D --> E["_split_color_labels_shop15(label_blob)\n按 、/／・ 等分隔符拆分"]
    E --> F[遍历每个 label]
    F --> G{specs 中已有\n该 label?}
    G -->|是| H["覆盖为 (label, delta, value)\n纠正 LLM 的错误"]
    G -->|否| I["新增 (label, delta, value)"]
    H --> F
    I --> F
    F -->|遍历完| C
    C -->|遍历完| J[返回增强后的 specs]
```

#### `_build_color_prices_from_specs_shop15(color_map, base_price, specs, debug) -> Tuple[Dict, List, List]`
- **作用**: 将 specs 映射到 info 表中的颜色，计算每个颜色的最终价格
- **价格计算逻辑**:

```mermaid
flowchart TD
    A[输入: color_map / base_price / specs] --> B{base_price\n是否存在?}
    B -->|是| C["初始化所有颜色价格\n= base_price"]
    B -->|否| D["color_prices 为空字典"]
    C --> E[遍历 specs]
    D --> E
    E --> F{kind?}
    F -->|abs| G["color_prices 直接覆盖\n= value"]
    F -->|delta| H{base_price 存在?}
    H -->|是| I["color_prices = base + delta"]
    H -->|否| J[跳过该颜色]
    G --> K[继续遍历]
    I --> K
    J --> K
    K --> E
    E -->|遍历完| L["返回 (color_prices, hit_log, unmatched)"]
```

#### `_label_matches_color_unified(label_raw, color_raw, color_norm) -> bool`
- **作用**: 判断提取到的颜色标签是否匹配 info 表中的某个颜色
- **匹配策略** (三级宽松匹配):

```mermaid
flowchart TD
    A["输入: label_raw, color_raw, color_norm"] --> B{精确匹配?\nlabel归一 == color_norm}
    B -->|是| Z[返回 True]
    B -->|否| C{子串匹配?\nlabel_raw in color_raw}
    C -->|是| Z
    C -->|否| D["查 FAMILY_SYNONYMS\n颜色家族同义词表"]
    D --> E{label 的小写\n在 FAMILY_SYNONYMS 中?}
    E -->|是| F[获取日文同义词列表]
    E -->|否| G[尝试 label_norm 查表]
    F --> H{同义词中任一\n出现在 color_raw 中?}
    G --> H
    H -->|是| Z
    H -->|否| Y[返回 False]
```

#### `_build_color_map(info_df) -> Dict[Tuple, Dict]`
- **作用**: 构建 `(model_norm, capacity_gb) -> {color_norm: (part_number, color_raw)}` 映射
- **数据源**: iphone17_info 参考表

#### `_clean_label_shop15(label: str) -> str`
- **作用**: 清理颜色标签文本
- **处理**: 全角空白→半角 / 多余空白合并 / 去掉粘着的分隔符 (`:：-/／、,，・` 等)

#### `_parse_signed_int_yen(s: object) -> Optional[int]`
- **作用**: 将各种格式的日元金额字符串解析为带符号整数
- **支持**: `'229,000'` / `'229,000円'` / `'-1000'` / `'-1,000円'` / `'+2000円'`
- **处理**: 全角符号→半角 (`＋→+`, `−→-`, `－→-`)

#### `_extract_base_price_at_start(text: object) -> Optional[int]`
- **作用**: 从文本开头提取基准价格 (正则兜底)
- **正则**: `_BASE_YEN_AT_START_RE` 只匹配开头位置，避免将颜色后的价格误识别为 base

#### `_iter_extractions_in_order(result) -> List`
- **作用**: 将 LangExtract 输出按文本位置排序
- **排序优先级**: `char_interval.start_pos` > `extraction_index` > 默认顺序

#### `_split_color_labels_shop15(label_blob: str) -> List[str]`
- **作用**: 将颜色标签串拆分为单个标签列表
- **分隔符**: `、` / `,` / `，` / `／` / `/` / `・` / `&` / `＆`

#### `_extract_signed_amount_after_label_shop15(price_text, label) -> Optional[int]`
- **作用**: 在原文中查找某颜色标签后紧跟的带符号金额
- **用途**: `_coerce_specs_shop15` 中用于从原文验证/覆盖 LLM 的输出

#### `_shop15_debug_enabled(debug: bool) -> bool`
- **作用**: 判断 debug 模式是否开启
- **来源**: 函数参数 `debug=True` 或环境变量 `SHOP15_DEBUG` 为 `1/true/yes/y/on`

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: price, data2, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        CMAP["cmap_all 字典\n(model_norm, cap_gb) → {\n  color_norm: (part_number, color_raw)\n}"]
        SPECS["specs 列表\n[(label, kind, yen_value), ...]\nkind: delta 或 abs\n如: [('ブルー','delta',-1000), ('シルバー','abs',229000)]"]
        CP["color_prices 字典\n{color_norm: final_price}\n如: {'ブルー': 179000, 'シルバー': 229000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> CMAP
    RAW -->|"逐行读取"| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["data2 → model_norm + cap_gb"]
        P2["price → LLM 提取 → base_price + specs"]
        P3["specs → 纠偏 + 多标签增强"]
        P4["specs + color_map + base → color_prices"]
        P5["color_prices → 逐颜色生成输出行"]
    end

    CMAP --> PROC
    PROC --> SPECS
    SPECS --> CP
    CP --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  data2        = "iPhone17 Pro Max 256GB"
  price        = "207,000円　オレンジ、ブルー-1000円"
  time-scraped = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 型号解析 (data2)"]
        T1["'iPhone17 Pro Max 256GB'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step2["Step 2: 查询 color_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 cmap_all| T5["{\n  'ブラックチタニウム': ('MYW23J/A', 'ブラックチタニウム'),\n  'ホワイトチタニウム': ('MYW53J/A', 'ホワイトチタニウム'),\n  ...\n}"]
    end

    subgraph Step3["Step 3: LLM 解析 price 列"]
        T6["'207,000円　オレンジ、ブルー-1000円'"]
        T6 -->|"lx.extract()\nbase_price 类"| T7["base_price = 207000"]
        T6 -->|"lx.extract()\ncolor_spec 类"| T8["specs = [\n  ('オレンジ', 'delta', -1000),\n  ('ブルー', 'delta', -1000)\n]"]
    end

    subgraph Step4["Step 4: Post-LLM 纠偏"]
        T9["_coerce_specs_shop15:\n检查原文中 label 后 +/- 金额 → 确认/覆盖"]
        T10["_augment_multi_label_block_specs_shop15:\n匹配 'オレンジ、ブルー-1000円' block\n→ 确认两个 label 共享 delta=-1000"]
    end

    subgraph Step5["Step 5: 颜色匹配 + 价格计算"]
        T11["对 color_map 中每个颜色:\n初始 → 全部 = base_price(207000)"]
        T11 --> T12["オレンジ 命中某颜色 → 207000 + (-1000) = 206000"]
        T11 --> T13["ブルー 命中某颜色 → 207000 + (-1000) = 206000"]
        T11 --> T14["未命中颜色 → 保持 base 207000"]
    end

    subgraph Step6["Step 6: 输出行"]
        T15["{\n  part_number: 'MYW23J/A',\n  shop_name: '買取当番',\n  price_new: 207000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYWXXX',\n  shop_name: '買取当番',\n  price_new: 206000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
```

### 3.3 Price 列解析策略: LLM 提取 + 多层纠偏

shop15 的 `price` 列同时包含基准价和颜色规格信息，解析流程如下:

```mermaid
flowchart TD
    INPUT["price 列原始文本\n如: '207,000円　オレンジ、ブルー-1000円'"]

    INPUT --> LLM["LLM 解析器\nlx.extract() + gemma3:1b"]

    LLM --> BASE_CHECK{LLM 返回了\nbase_price?}
    BASE_CHECK -->|是| SPECS["使用 LLM base_price"]
    BASE_CHECK -->|否| REGEX_BASE["正则兜底\n_extract_base_price_at_start()\n从开头匹配 N円"]
    REGEX_BASE --> SPECS

    SPECS --> COERCE["纠偏层1: _coerce_specs_shop15\n- abs且负数 → 改为 delta\n- 原文验证: label后有+/-金额 → 覆盖"]

    COERCE --> AUGMENT["纠偏层2: _augment_multi_label_block_specs_shop15\n- 正则匹配多标签 block\n- 拆分并覆盖/新增 specs"]

    AUGMENT --> FINAL["最终 (base_price, specs)"]

    subgraph LLM解析器详细
        L1["prompt: SHOP15_PRICE_PROMPT\n提取两类实体"]
        L2["extraction_class = base_price\n→ 基准价格"]
        L3["extraction_class = color_spec\n→ attributes: kind(delta/abs) + yen"]
        L4["examples: 5个 few-shot 示例"]
        L5["model: gemma3:1b (本地 Ollama)"]
        L6["temperature: 0.0 (确定性输出)"]
        L7["缓存: @lru_cache(maxsize=4096)"]
    end
```

### 3.4 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY_SYNONYMS
        BLUE["blue 家族\nブルー"]
        BLACK["black 家族\nブラック / 黒"]
        WHITE["white 家族\nホワイト / 白"]
        GREEN["green 家族\nグリーン / 緑"]
        RED["red 家族\nレッド / 赤"]
        PINK["pink 家族\nピンク"]
        PURPLE["purple 家族\nパープル / 紫"]
        YELLOW["yellow 家族\nイエロー / 黄"]
        GOLD["gold 家族\nゴールド"]
        SILVER["silver 家族\nシルバー"]
        GRAY["gray 家族\nグレー / グレイ / 灰"]
        NATURAL["natural 家族\nナチュラル"]
    end

    LABEL["提取到的 label\n如: 'ブルー'"]
    COLOR["info表中的 color\n如: 'マリンブルー'"]

    LABEL -->|"查 FAMILY_SYNONYMS"| BLUE
    BLUE -->|"同义词 'ブルー' in 'マリンブルー'"| MATCH["匹配成功!"]
```

### 3.5 LLM 提取类对比: base_price vs color_spec

```mermaid
flowchart LR
    subgraph base_price["extraction_class = base_price"]
        BP1["extraction_text = '207,000円'"]
        BP2["attributes = {yen: '207000'}"]
        BP3["用途: 确定基准价格"]
    end

    subgraph color_spec["extraction_class = color_spec"]
        CS1["extraction_text = 'ブルー'"]
        CS2["attributes = {\n  kind: 'delta' 或 'abs',\n  yen: '-1000' 或 '229000'\n}"]
        CS3["kind=delta: 相对差额\nfinal = base + delta"]
        CS4["kind=abs: 绝对价格\nfinal = value"]
    end
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EXTRACTION_MODE` | `"regex"` | regex / llm / auto（cleaner_tools） |
| `OLLAMA_URL` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `SHOP15_DEBUG` | `""` (关闭) | 是否启用 debug 输出 (`1/true/yes/y/on` 启用) |
| `IPHONE17_INFO_CSV` | 自动推断路径 | iphone17_info 参考文件路径 |

**LangExtract 调用参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_id` | `gemma3:1b` | 本地 Ollama 小模型 |
| `model_url` | `http://localhost:11434` | Ollama 服务端口 |
| `temperature` | `0.0` | 确定性输出，减少随机性 |
| `fence_output` | `False` | 不使用 fence 输出格式 |
| `use_schema_constraints` | `False` | 不使用 schema 约束 |
| `@lru_cache maxsize` | `4096` | 缓存最多 4096 个不同的 price_text 调用结果 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_BASE_YEN_AT_START_RE` | `^\s*(?:￥\|¥)?\s*(\d[\d,]*)\s*円?` | 从文本开头提取基准价格 (避免误抓颜色后的价格) | `207,000円`, `¥180,000` |
| `BASE_YEN_AT_START_RE_shop15` | `^\s*(?:￥\|¥)?\s*(\d[\d,]*)\s*円?` | 同上，shop15 专用别名 | `230,500円` |
| `FIRST_YEN_RE` | `(?:￥\|¥)?\s*(\d[\d,]*)\s*円?` | 抓取文本中第一个日元金额 (不限位置) | `207,000円` |
| `FIRST_YEN_RE_shop15` | `(\d[\d,]*)\s*円` | 抓取 price 中第一个 "N円" | `207,000円` |
| `COLOR_DELTA_IN_PRICE_RE_shop15` | `(?P<label>[^\d円¥]+?)\s*(?P<sep>[：:\-])?\s*(?P<sign>[+\-−－])?\s*(?P<amount>\d[\d,]*)\s*(?:円)?` | 匹配"颜色标签±金额"模式 | `ブルー-1000円`, `シルバー+2,000円` |
| `COLOR_ENTRY_RE_shop15` | 同 `COLOR_DELTA_IN_PRICE_RE_shop15` | 提取颜色条目 (区分 delta/abs) | `ブルー229,000円`, `シルバー-3,000円` |
| `MULTI_LABEL_DELTA_BLOCK_RE_shop15` | `(?P<label_blob>[^\d円¥]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>\d[\d,]*)\s*円?` | 匹配多颜色共享差额的 block | `オレンジ、ブルー-1000円` |
| `_LABEL_LIST_SPLIT_RE_shop15` | `\s*(?:、\|,\|，\|／\|/\|・\|&\|＆)\s*` | 拆分多颜色标签串 | `オレンジ、ブルー` → `['オレンジ','ブルー']` |
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
