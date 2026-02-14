# Shop3 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop3_cleaner.py`
> 店铺名称: 買取一丁目

---

## 一、总流程图

整个 shop3 清洗器的核心入口是 `clean_shop3(df, debug, debug_limit)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\ntitle / data5 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| B2[过滤 time-scraped 为空的行]
    B2 --> C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建颜色映射表\n_build_color_map]
    D --> E0[批量预处理列\nmodel_norm / cap_gb / base_price / recorded_at]
    E0 --> E1{减价1 列\n是否存在?}
    E1 -->|是| E2[对 unique 减价1 文本\n批量调用 _extract_color_deltas_shop3\n构建 delta_map]
    E1 -->|否| E3["deltas_series 全为空列表 []"]
    E2 --> E[逐行遍历 DataFrame]
    E3 --> E

    E --> F{model_norm\n是否为空?}
    F -->|空| E
    F -->|非空| G{capacity_gb\n是否为空?}

    G -->|空| E
    G -->|非空| H{base_price\n是否有效?}

    H -->|否| E
    H -->|是| I[在 color_map 中\n查找该机型+容量]

    I --> J{color_map\n是否存在?}
    J -->|否| E
    J -->|是| K[取该行已提取的 deltas]

    K --> L[颜色标签匹配\n_label_matches_color_unified\n构建 per_color_delta]
    L --> M[对 color_map 中每个颜色\nprice = base_price + delta]
    M --> N[生成输出行\npart_number / shop_name / price_new / recorded_at]
    N --> E

    E -->|遍历结束| O[组装输出 DataFrame]
    O --> P[去除空值 / 类型转换]
    P --> Q["输出: 标准化 DataFrame\npart_number, shop_name(買取一丁目), price_new, recorded_at"]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop3(df, debug, debug_limit)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> buildcm["_build_color_map(info_df)"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> pricefn["extract_price_yen(x)"]
    clean --> extract["_extract_color_deltas_shop3(text)"]
    clean --> labelmatch["_label_matches_color_unified(label, color_raw, color_norm)"]
    clean --> parsedt["parse_dt_aware(val)"]

    extract --> llm["_extract_color_deltas_shop3_llm_cached(text)"]
    extract --> regex["_extract_color_deltas_shop3_regex(text)"]

    llm --> signdelta["_single_signed_delta_from_text(text)"]
    llm --> infersign["_infer_default_sign_from_text(text)"]
    llm --> iterext["_iter_extractions_from_langextract_result(result)"]
    llm --> cleanlbl["_clean_label_token(tok)"]
    llm --> parsedeltallm["_parse_delta_int_llm(x, default_sign)"]
    llm --> lxextract["lx.extract()"]

    signdelta --> signedamts["_extract_signed_amounts_from_text(text)"]
    infersign --> signedamts
    signedamts --> normamt["_normalize_amount_text(s)"]
    parsedeltallm --> normamt

    regex --> normamt
    regex --> cleanlbl

    pricefn --> toint["to_int_yen(val)"]
    buildcm --> normmod
    labelmatch --> familysyn["FAMILY_SYNONYMS_shop3 字典查表"]
```

### 2.2 核心函数详细说明

#### `clean_shop3(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `title`, `data5`, `time-scraped` 列的 DataFrame（可选列 `减价1`）
- **输出**: 包含 `part_number`, `shop_name`, `price_new`, `recorded_at` 列的 DataFrame
- **debug 模式**: 仅对"减价1"中可抽出颜色差额的行打印对照信息（最多 debug_limit 条）

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `extract_price_yen(x: object) -> Optional[int]`
- **作用**: 将 data5 列的原始价格文本转为整数 (JPY)
- **处理**: 预期形如 `'¥177,000'`；兼容 `'～177,000円'` / `'10.5万'` 等；去除修饰词（"新品/未開封"等），然后调用 `to_int_yen` 取区间最大值

#### `_extract_color_deltas_shop3(text: str) -> List[Tuple[str, int]]`
- **作用**: 提取颜色差额的调度函数，采用 **LLM 优先、正则兜底** 策略

```mermaid
flowchart TD
    A["_extract_color_deltas_shop3(text)"] --> B{EXTRACTION_MODE\n已启用 且 lx 可用?}
    B -->|否| C["_extract_color_deltas_shop3_regex(text)"]
    B -->|是| D["try: _extract_color_deltas_shop3_llm_cached(text)"]
    D --> E{调用成功?}
    E -->|是| F[返回 LLM 结果]
    E -->|异常| C
    C --> G[返回正则结果]
```

#### `_extract_color_deltas_shop3_llm_cached(text: str) -> Tuple[Tuple[str, int], ...]`
- **作用**: LLM 版颜色差额提取（带 `@lru_cache(maxsize=4096)` 缓存）
- **流程**:

```mermaid
flowchart TD
    A[输入 text] --> B{文本是否为空?\n或不含数字/符号?}
    B -->|是| C["返回空 tuple()"]
    B -->|否| D{lx 是否可用?}
    D -->|否| C
    D -->|是| E["推断全局符号:\ndelta_global = _single_signed_delta_from_text\ndefault_sign = _infer_default_sign_from_text"]
    E --> F["调用 lx.extract()\nmodel_id = SHOP3_OLLAMA_MODEL_ID\nmodel_url = SHOP3_OLLAMA_URL\nprompt = _SHOP3_COLOR_DELTA_PROMPT\nexamples = 6个 few-shot 示例"]
    F --> G[遍历 extractions]
    G --> H{extraction_class\n== color_delta?}
    H -->|否| G
    H -->|是| I["提取 label = extraction_text"]
    I --> J{delta_global\n是否非 None?}
    J -->|是| K["直接赋值:\nmp[label] = delta_global"]
    J -->|否| L["从 attributes 提取 delta_yen\n_parse_delta_int_llm(delta_raw, default_sign)"]
    L --> M{delta 解析成功?}
    M -->|否| N["尝试 attrs['delta'] 或 attrs['amount']"]
    N --> O{仍然失败?}
    O -->|是| G
    O -->|否| K2["mp[label] = delta"]
    M -->|是| K2
    K --> G
    K2 --> G
    G -->|遍历完| P["返回 tuple(mp.items())"]
```

#### `_extract_color_deltas_shop3_regex(text: str) -> List[Tuple[str, int]]`
- **作用**: 正则版颜色差额提取（LLM 失败或不可用时的后备方案）
- **流程**:

```mermaid
flowchart TD
    A[输入 text] --> B["全角→半角转换\n_FZ_TO_HZ_TRANS"]
    B --> C["先用 _DELTA_PATTERN_STRICT 匹配"]
    C --> D{strict 有结果?}
    D -->|是| E[对每个匹配提取\nlabels / sign / amount]
    D -->|否| F["改用 _DELTA_PATTERN_LOOSE 匹配"]
    F --> E
    E --> G["按 _LABEL_SPLIT_RE 拆分\nlabels_part 得到多个 token"]
    G --> H["对每个 token:\n_clean_label_token 清洗"]
    H --> I["合并 sign + amount\n→ delta (int)"]
    I --> J["添加 (label, delta)"]
    J --> K["返回 [(label, delta), ...]"]
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
    C -->|否| D["查 FAMILY_SYNONYMS_shop3\n颜色家族同义词表"]
    D --> E{label 在家族表中?}
    E -->|是| F[获取同义词列表]
    E -->|否| G["反向查: 遍历所有家族\n找包含 label 的条目"]
    F --> H{同义词中任一\n出现在 color_raw 中?}
    G --> H
    H -->|是| Z
    H -->|否| Y[返回 False]
```

#### `_build_color_map(info_df) -> Dict`
- **作用**: 构建 `(model_norm, capacity_gb) -> {color_norm: (part_number, color_raw)}` 映射
- **数据源**: iphone17_info 参考表
- **处理**: 对 model_name 做 `_normalize_model_generic`，对 color 做 `_norm`

#### `extract_price_yen(x: object) -> Optional[int]`
- **作用**: 从 data5 列解析基准价格
- **处理**: 去除修饰词 (新品/未開封 等)，然后调用 `to_int_yen`

#### `_infer_default_sign_from_text(text: str) -> Optional[int]`
- **作用**: 分析原文中所有带符号金额，若全为负返回 `-1`，全为正返回 `+1`，混合返回 `None`
- **用途**: 为 LLM 解析的 delta 提供默认符号方向

#### `_single_signed_delta_from_text(text: str) -> Optional[int]`
- **作用**: 若原文只出现一种带符号金额（可能重复），返回该值；否则返回 `None`
- **用途**: 当原文只有一个 delta 值时，直接覆盖所有 label 的 delta（delta_global）

#### `_parse_delta_int_llm(x: object, default_sign: Optional[int]) -> Optional[int]`
- **作用**: 解析 LLM 输出的 delta 值，支持 int / float / 字符串
- **特殊逻辑**: 若无显式符号则按 `default_sign` 赋符号；若显式符号与 `default_sign` 冲突则以 `default_sign` 为准

#### `_clean_label_token(tok: str) -> str`
- **作用**: 清洗标签文本，去除括号内内容

#### `_normalize_amount_text(s: str) -> Optional[int]`
- **作用**: 把全角数字/标点转半角，提取数字部分转为 int

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: title, data5, 减价1, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        CMAP["color_map 字典\n(model_norm, cap_gb) → {\n  color_norm: (part_number, color_raw)\n}"]
        DELTAS["deltas_series\n每行: [(label_raw, delta_int), ...]\n如: [('ブルー', -1000), ('シルバー', -1000)]"]
        PCD["per_color_delta 字典\n{color_norm: delta_int}\n如: {'ブルー': -1000, 'シルバー': -1000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name(買取一丁目), price_new, recorded_at"]
    end

    INFO --> CMAP
    RAW --> |"批量预处理"| BATCH

    subgraph BATCH[批量列处理]
        direction TB
        B1["title → model_norm (via _normalize_model_generic)"]
        B2["title → cap_gb (via _parse_capacity_gb)"]
        B3["data5 → base_price (via extract_price_yen)"]
        B4["time-scraped → recorded_at (via parse_dt_aware)"]
        B5["减价1 → deltas_series (via _extract_color_deltas_shop3)"]
    end

    BATCH --> DELTAS
    CMAP --> PROC
    DELTAS --> PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["跳过无效行: model/cap/price 为空"]
        P2["查 color_map 获取该机型所有颜色"]
        P3["deltas + color_map → per_color_delta"]
        P4["base_price + delta → price_new"]
    end

    PROC --> PCD
    PCD --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  title        = "iPhone17 Pro Max 256GB SIMフリー"
  data5        = "¥177,000"
  减价1        = "ブルー、シルバー　-1000"
  time-scraped = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 型号解析 (title 列)"]
        T1["'iPhone17 Pro Max 256GB SIMフリー'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step2["Step 2: 查询 color_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 color_maps| T5["{\n  'ブラックチタニウム': ('MYW23J/A', 'ブラックチタニウム'),\n  'ホワイトチタニウム': ('MYW53J/A', 'ホワイトチタニウム'),\n  'ナチュラルチタニウム': ('MYW83J/A', 'ナチュラルチタニウム'),\n  ...\n}"]
    end

    subgraph Step3["Step 3: 基准价格 (data5 列)"]
        T6["'¥177,000'"]
        T6 -->|"extract_price_yen → to_int_yen"| T7["177000"]
    end

    subgraph Step4["Step 4: 颜色差额提取 (减价1 列)"]
        T8["'ブルー、シルバー　-1000'"]
        T8 -->|"_extract_color_deltas_shop3 (LLM 优先)"| T8a["LLM 路径"]
        T8a -->|"delta_global = _single_signed_delta_from_text"| T8b["delta_global = -1000\n(原文只有一种带符号金额)"]
        T8b -->|"lx.extract → 提取 label: ブルー, シルバー"| T8c["所有 label 统一使用 delta_global"]
        T8c --> T12["[('ブルー', -1000), ('シルバー', -1000)]"]
    end

    subgraph Step5["Step 5: 颜色匹配 + 价格计算"]
        T13["对 color_map 中每个颜色匹配 deltas:"]
        T13 --> T14["ブラックチタニウム → 未匹配 → delta=0 → price=177000"]
        T13 --> T15["ホワイトチタニウム → 未匹配 → delta=0 → price=177000"]
        T13 --> T16["(假设存在ブルー系) → 匹配 'ブルー' → delta=-1000 → price=176000"]
        T13 --> T17["(假设存在シルバー系) → 匹配 'シルバー' → delta=-1000 → price=176000"]
    end

    subgraph Step6["Step 6: 输出行"]
        T18["{\n  part_number: 'MYW23J/A',\n  shop_name: '買取一丁目',\n  price_new: 177000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYWXXX',\n  shop_name: '買取一丁目',\n  price_new: 176000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
```

### 3.3 颜色差额提取 - LLM vs 正则 策略

```mermaid
flowchart TD
    INPUT["减价1 原始文本"]

    INPUT --> CHECK1{EXTRACTION_MODE\n开启 且 lx 可用?}
    CHECK1 -->|否| REGEX["正则解析器\n_extract_color_deltas_shop3_regex"]
    CHECK1 -->|是| LLM["LLM 解析器\n_extract_color_deltas_shop3_llm_cached\n(带 @lru_cache maxsize=4096)"]
    LLM --> CHECK2{调用成功?}
    CHECK2 -->|异常| REGEX
    CHECK2 -->|是| USE_LLM["使用 LLM 结果\n(处理复杂/非标准格式)"]

    REGEX --> USE_REGEX["使用正则结果\n(快速 & 稳定)"]

    subgraph LLM解析器详细
        L0["全局符号推断:\n_single_signed_delta_from_text → delta_global\n_infer_default_sign_from_text → default_sign"]
        L1["prompt: 买取表色減額/减价备注解析专用提示词"]
        L2["examples: 6个 few-shot 示例"]
        L3["model: SHOP3_OLLAMA_MODEL_ID (默认 gemma3:1b)"]
        L4["model_url: SHOP3_OLLAMA_URL (默认 localhost:11434)"]
        L5["输出: extraction_class=color_delta\nextraction_text=颜色标签\nattributes={delta_yen}"]
        L6["符号修正逻辑:\n若 delta_global 存在 → 覆盖所有 label 的 delta\n否则 → _parse_delta_int_llm 结合 default_sign 修正"]
    end

    subgraph 正则解析器详细
        R1["_DELTA_PATTERN_STRICT: 先尝试严格模式"]
        R2["_DELTA_PATTERN_LOOSE: strict 无结果时启用宽松模式"]
        R3["_LABEL_SPLIT_RE: 按分隔符拆分多标签"]
        R4["_clean_label_token: 清洗标签"]
    end
```

### 3.4 全局符号修正机制 (delta_global / default_sign)

```mermaid
flowchart TD
    TEXT["减价1 原文\n如: 'ブルー、シルバー -1000'"]

    TEXT --> A["_extract_signed_amounts_from_text\n提取所有带符号金额"]
    A --> B["结果: [-1000]"]

    B --> C["_single_signed_delta_from_text"]
    B --> D["_infer_default_sign_from_text"]

    C --> C1{unique 值集合\n大小 == 1?}
    C1 -->|是| C2["delta_global = -1000\n(所有 label 直接用此值)"]
    C1 -->|否| C3["delta_global = None"]

    D --> D1{全为负?}
    D1 -->|是| D2["default_sign = -1"]
    D1 -->|否| D3{全为正?}
    D3 -->|是| D4["default_sign = +1"]
    D3 -->|否| D5["default_sign = None (混合)"]

    C2 --> LLM["LLM 解析时:\n若 delta_global 非 None\n→ 直接 mp[label] = delta_global"]
    D2 --> LLM2["LLM 解析时:\n若 delta_global 为 None\n→ _parse_delta_int_llm 按 default_sign 修正符号"]
```

### 3.5 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY_SYNONYMS_shop3
        BLUE["blue 家族\nブルー / 青 / ディープブルー"]
        SILVER["silver 家族\nシルバー / 銀"]
    end

    LABEL["提取到的 label\n如: 'ブルー'"]
    COLOR["info表中的 color\n如: 'ディープブルー'"]

    LABEL -->|"查 FAMILY_SYNONYMS_shop3"| BLUE
    BLUE -->|"同义词 'ブルー' / 'ディープブルー' → 'ディープブルー' in color_raw"| MATCH["匹配成功!"]
```

---

## 四、配置项说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EXTRACTION_MODE` | `"regex"` | regex / llm / auto（cleaner_tools） |
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `OLLAMA_URL` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `IPHONE17_INFO_CSV` | 自动推断路径 (`data/iphone17_info.csv`) | iphone17_info 参考文件路径 |

| 函数参数 | 默认值 | 说明 |
|---------|--------|------|
| `clean_shop3(debug=)` | `True` | 是否启用 debug 打印 |
| `clean_shop3(debug_limit=)` | `30` | debug 打印最大行数 |
| `@lru_cache(maxsize=)` | `4096` | LLM 调用缓存大小（`_extract_color_deltas_shop3_llm_cached`） |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_DELTA_PATTERN_STRICT` | `(?P<labels>[^+\-−－\d¥￥円]+?)(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?` | 严格模式：匹配"颜色标签±金额"（标签中不含数字/符号） | `ブルー-1000`, `シルバー-3,000` |
| `_DELTA_PATTERN_LOOSE` | `(?P<labels>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?` | 宽松模式：标签允许包含片假名/汉字/空格等（strict 无结果时启用） | `ディープブルー-3,000`, `ブラック、ブルー-4000` |
| `_LABEL_SPLIT_RE` | `[／/、，,・\s；;]+` | 拆分多标签共用金额的标签部分 | `ブルー、シルバー` → `['ブルー', 'シルバー']` |
| `_SIGNED_AMOUNT_PAT` | `([+\-−－])\s*([0-9０-９][0-9０-９,，]*)` | 从原文中提取所有带符号金额（用于 delta_global/default_sign 推断） | `-1000`, `−3,000`, `＋２,０００` |
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max`, `iPhone16Plus` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
| `_FZ_TO_HZ_TRANS` | 全角→半角映射表 | 统一全角数字/标点为半角 | `０`→`0`, `，`→`,`, `－`→`-`, `＋`→`+` |
