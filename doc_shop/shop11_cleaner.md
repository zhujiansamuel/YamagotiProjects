# Shop11 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop11_cleaner.py`
> 店铺名称: モバステ

---

## 一、总流程图

整个 shop11 清洗器的核心入口是 `clean_shop11(df, debug, debug_limit)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。与 shop17 不同的是，shop11 同时对**型号/容量**和**颜色差额**使用 LLM 解析，且两处均有正则回退。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\nstorage_name / price_unopened / caution_empty / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建颜色映射表\n_build_color_map]
    D --> D2[推导 valid_models 列表\n约束 LLM 输出范围]
    D2 --> E[逐行遍历 DataFrame]

    E --> F{storage_name\n是否为空?}
    F -->|空| E
    F -->|非空| G["LLM 解析型号+容量\n_lx_parse_storage_shop11"]

    G --> H{LLM 结果\nmodel/cap 有效?}
    H -->|否| H2["正则回退\n_normalize_model_generic\n_parse_capacity_gb"]
    H -->|是| I
    H2 --> H3{正则结果有效?}
    H3 -->|否| E
    H3 -->|是| I[在 color_map 中\n查找该机型]

    I --> J{color_map\n是否存在?}
    J -->|否| J2["二次规范化尝试\n_normalize_model_generic(model_norm)"]
    J -->|是| K
    J2 --> J3{二次查找成功?}
    J3 -->|否| E
    J3 -->|是| K[解析基准价格\nto_int_yen_shop11]

    K --> L{基准价格\n是否有效?}
    L -->|否| E
    L -->|是| M["LLM 解析颜色差额\n_lx_parse_color_deltas_shop11"]

    M --> N{LLM 结果\n是否为空?}
    N -->|非空| O[使用 LLM 结果]
    N -->|空且有文本| P["正则回退\n_extract_color_deltas_shop11\n+ _label_matches_color_unified"]
    P --> O

    O --> Q[计算每个颜色的最终价格\nprice = base_price + delta]
    Q --> R[生成输出行\npart_number / shop_name / price_new / recorded_at]
    R --> E

    E -->|遍历结束| S[组装输出 DataFrame]
    S --> T[去除空值 / 类型转换]
    T --> U["输出: 标准化 DataFrame\npart_number, shop_name(モバステ), price_new, recorded_at"]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop11(df, debug, debug_limit)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> buildcm["_build_color_map(info_df)"]
    clean --> lxstorage["_lx_parse_storage_shop11(storage, valid_models)"]
    clean --> normmod["_normalize_model_generic(text)\n(回退)"]
    clean --> parsecap["_parse_capacity_gb(text)\n(回退)"]
    clean --> toint["to_int_yen_shop11(val)"]
    clean --> lxcolor["_lx_parse_color_deltas_shop11(caution, avail_colors)"]
    clean --> regexdelta["_extract_color_deltas_shop11(text)\n(回退)"]
    clean --> labelmatch["_label_matches_color_unified(label, color_raw, color_norm)"]
    clean --> normnum["_normalize_number_text(txt)"]

    lxstorage --> storemats["_shop11_lx_storage_materials(valid_models)"]
    lxstorage --> lxextract["_lx_extract_ollama(text, prompt, examples)"]
    lxstorage --> normmod2["_normalize_model_generic(mn)\n(二次规范化)"]
    lxstorage --> coerce["_coerce_int(val)"]

    lxcolor --> colormats["_shop11_lx_color_materials()"]
    lxcolor --> lxextract
    lxcolor --> coerce
    lxcolor --> labelmatch

    lxextract --> modelcfg["_shop11_model_config()"]
    lxextract --> lxlib["langextract.extract()"]

    modelcfg --> formatjson["FormatType.JSON"]

    regexdelta --> normnum
    regexdelta --> colorgrp["_COLOR_GROUP_RE"]
    regexdelta --> colorfb["_COLOR_GROUP_FALLBACK_RE"]
    regexdelta --> colorsep["_COLOR_SEP_SPLIT_RE"]

    buildcm --> normmod
    labelmatch --> family["FAMILY 同义词字典"]
```

### 2.2 核心函数详细说明

#### `clean_shop11(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `storage_name`, `price_unopened`, `caution_empty`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`, `price_new`, `recorded_at` 列的 DataFrame
- **debug 模式**: 根据 `caution_empty` 中的颜色关键词匹配，选取最多 `debug_limit` 行打印详细调试信息

#### `_lx_parse_storage_shop11(storage: str, valid_models: Tuple[str, ...]) -> Tuple[str, Optional[int], trace]`
- **作用**: LLM 解析型号名与容量（第一次 LLM 调用）
- **缓存**: `@lru_cache(maxsize=4096)`
- **输入文本格式**: `"STORAGE: {storage}"`
- **extraction_classes**: `device_model`（属性: `model_norm`）、`storage_capacity`（属性: `capacity_gb`）
- **流程**:

```mermaid
flowchart TD
    A["输入: storage 文本"] --> B{langextract 可用?}
    B -->|否| C["返回空结果 ('', None, ())"]
    B -->|是| D["构造输入: 'STORAGE: {storage}'"]
    D --> E["获取 prompt + examples\n_shop11_lx_storage_materials(valid_models)"]
    E --> F["调用 _lx_extract_ollama()"]
    F --> G[遍历 extractions]
    G --> H{extraction_class\n== device_model?}
    H -->|是| I["提取 model_norm\n再走 _normalize_model_generic 规范化"]
    H -->|否| J{extraction_class\n== storage_capacity?}
    J -->|是| K["提取 capacity_gb\n_coerce_int 转整数"]
    J -->|否| G
    I --> G
    K --> G
    G -->|遍历完| L["返回 (model_norm, cap_gb, trace)"]
```

#### `_lx_parse_color_deltas_shop11(caution: str, available_colors: Tuple[str, ...]) -> Tuple[deltas_items, trace]`
- **作用**: LLM 解析颜色差额（第二次 LLM 调用）
- **缓存**: `@lru_cache(maxsize=4096)`
- **输入文本格式**: `"CAUTION: {caution}\nAVAILABLE_COLORS: {c1 | c2 | ...}"`
- **extraction_class**: `color_delta`（属性: `delta_yen`）
- **流程**:

```mermaid
flowchart TD
    A["输入: caution 文本 + available_colors"] --> B{langextract 可用?}
    B -->|否| C["返回空结果 ((), ())"]
    B -->|是| D["构造输入:\n'CAUTION: {caution}\nAVAILABLE_COLORS: c1 | c2 | ...'"]
    D --> E["获取 prompt + examples\n_shop11_lx_color_materials()"]
    E --> F["调用 _lx_extract_ollama()"]
    F --> G[遍历 extractions]
    G --> H{extraction_class\n== color_delta?}
    H -->|否| G
    H -->|是| I["提取 delta_yen\n_coerce_int 转整数"]
    I --> J{delta 有效\n且 extraction_text 非空?}
    J -->|否| G
    J -->|是| K{extraction_text\n在 available_colors 中?}
    K -->|是| L["直接记录 (color, delta)"]
    K -->|否| M["遍历 available_colors\n用 _label_matches_color_unified 匹配"]
    M --> L
    L --> G
    G -->|遍历完| N["返回 (deltas_items, trace)\n同色多次取最后值"]
```

#### `_lx_extract_ollama(text: str, prompt: str, examples: list)`
- **作用**: LangExtract 调用封装，兼容新旧两种 API
- **策略**:

```mermaid
flowchart TD
    A["输入: text, prompt, examples"] --> B{langextract 可用?}
    B -->|否| C[返回 None]
    B -->|是| D["获取 ModelConfig\n_shop11_model_config()"]
    D --> E{config 不为 None?}
    E -->|是| F["尝试新版 API\nlx.extract(config=cfg, ...)"]
    F --> G{成功?}
    G -->|是| H[返回 result]
    G -->|TypeError/Exception| I["回退到旧版 API"]
    E -->|否| I
    I --> J["lx.extract(\nlanguage_model_type=OllamaLanguageModel,\nmodel_id=..., model_url=..., ...)"]
    J --> K{成功?}
    K -->|是| H
    K -->|否| C
```

#### `_shop11_model_config()`
- **作用**: 构建 LangExtract `ModelConfig` 对象
- **缓存**: `@lru_cache(maxsize=1)`
- **配置**: `model_id`, `model_url`, `temperature`, `timeout`, `max_tokens`, `FormatType.JSON`

#### `_shop11_lx_storage_materials(valid_models: Tuple[str, ...]) -> (prompt, examples)`
- **作用**: 构建 storage 解析的 LLM prompt 和 3 个 few-shot 示例
- **缓存**: `@lru_cache(maxsize=8)`
- **prompt 特征**: 要求 `model_norm` 必须严格等于 `valid_models` 列表中的值
- **示例**:
  1. `"iPhone17 Pro Max 256GB"` -> device_model + storage_capacity
  2. `"17pro 1TB"` -> 1TB 换算为 1024GB
  3. `"iPhone17 プロ 512GB"` -> 日文别名处理

#### `_shop11_lx_color_materials() -> (prompt, examples)`
- **作用**: 构建颜色差额解析的 LLM prompt 和 3 个 few-shot 示例
- **缓存**: `@lru_cache(maxsize=1)`
- **prompt 特征**: 输入包含 `AVAILABLE_COLORS` 行；支持 `全色`/`すべて`/`全カラー` 全色规则；同色多次取最后值
- **示例**:
  1. `"ブルー、ブラック：-2,000円(未開封)"` -> 两色各 -2000
  2. `"全色:+1,000円"` -> 所有颜色 +1000
  3. `"シルバー・ブルー：-１０００円"` -> 全角数字处理

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ->Pro, エア->Air) / 紧凑写法展开 (17pro->17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB->GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_extract_color_deltas_shop11(text: str) -> List[Tuple[str, int]]`
- **作用**: 正则版颜色差额提取（LLM 失败时的回退方案）
- **流程**:

```mermaid
flowchart TD
    A[输入 text] --> B{text 为空?}
    B -->|是| C["返回空列表 []"]
    B -->|否| D["去括号备注\n_normalize_number_text 半角化"]
    D --> E["主匹配: _COLOR_GROUP_RE\n'labels：+/-amount'"]
    E --> F["按 _COLOR_SEP_SPLIT_RE 拆分 labels\n／ / 、 , ・ 空格"]
    F --> G["收集 (label, delta) 对"]
    G --> H["回退匹配: _COLOR_GROUP_FALLBACK_RE\n'labels +/-amount' (无冒号)"]
    H --> I["继续收集 (label, delta) 对"]
    I --> J["去重: 同 label 保留最后出现的 delta"]
    J --> K["返回 [(label, delta), ...]"]
```

#### `_label_matches_color_unified(label_raw, color_raw, color_norm) -> bool`
- **作用**: 判断提取到的颜色标签是否匹配 info 表中的某个颜色
- **匹配策略** (四级宽松匹配):

```mermaid
flowchart TD
    A["输入: label_raw, color_raw, color_norm"] --> B{精确匹配?\nlabel归一 == color_norm}
    B -->|是| Z[返回 True]
    B -->|否| C{子串匹配?\nlabel_raw in color_raw}
    C -->|是| Z
    C -->|否| D["分割匹配:\n按 _COLOR_SEP_SPLIT_RE 拆分 label"]
    D --> E{任一片段\n是 color_raw 的子串\n或归一 == color_norm?}
    E -->|是| Z
    E -->|否| F[查 FAMILY 同义词字典\n六个颜色家族]
    F --> G{label 在家族表中?}
    G -->|是| H[获取同义词列表]
    G -->|否| I[反向查: 遍历所有家族\n找包含 label 的条目]
    H --> J{同义词中任一\n出现在 color_raw 中?}
    I --> J
    J -->|是| Z
    J -->|否| Y[返回 False]
```

#### `_build_color_map(info_df) -> Dict`
- **作用**: 构建 `(model_norm, capacity_gb) -> {color_norm: (part_number, color_raw)}` 映射
- **数据源**: iphone17_info 参考表

#### `to_int_yen_shop11(v) -> Optional[int]`
- **作用**: 将各种形式的日元表示解析为 int
- **支持**: `"1,000"`, `"1,000円"`, `"¥1,000"`, `"１，０００"`, 全角数字、带括号备注等

#### `_normalize_number_text(txt: str) -> str`
- **作用**: 全角数字/标点转半角
- **映射**: `０-９` -> `0-9` / `，` -> `,` / `：` -> `:` / `－` -> `-` / `＋` -> `+` / `¥￥` -> 删除

#### `_coerce_int(v) -> Optional[int]`
- **作用**: 安全地将各种类型值转换为 int
- **支持**: `"1,000"`, `"-1000"`, float, int

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: storage_name, price_unopened,\ncaution_empty, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name,\ncapacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        CMAP["color_map 字典\n(model_norm, cap_gb) -> {\n  color_norm: (part_number, color_raw)\n}"]
        VMOD["valid_models 元组\n约束 LLM 输出的合法机型列表"]
        DELTAS["color_deltas 字典\n{color_norm: delta_int}\n如: {'ブルー': -1000, 'ブラック': -2000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name(モバステ),\nprice_new, recorded_at"]
    end

    INFO --> CMAP
    INFO --> VMOD
    RAW -->|逐行读取| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["storage_name → LLM解析 → model_norm + cap_gb\n(失败则正则回退)"]
        P2["price_unopened → to_int_yen_shop11 → base_price"]
        P3["caution_empty → LLM解析 → color_deltas\n(失败则正则回退 + 颜色匹配)"]
        P4["base_price + delta → price_new (每色一行)"]
    end

    CMAP --> PROC
    VMOD --> PROC
    PROC --> DELTAS
    DELTAS --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  storage_name   = "iPhone17 Pro Max 256GB"
  price_unopened = "206,000"
  caution_empty  = "シルバー・ブルー：-1,000円(未開封)"
  time-scraped   = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: LLM 型号/容量解析"]
        T1["'iPhone17 Pro Max 256GB'"]
        T1 -->|"_lx_parse_storage_shop11\n输入: 'STORAGE: iPhone17 Pro Max 256GB'"| T2["LLM 返回:\ndevice_model: model_norm='iPhone 17 Pro Max'\nstorage_capacity: capacity_gb=256"]
        T2 -->|"_normalize_model_generic 二次规范化"| T3["model_norm='iPhone 17 Pro Max'\ncap_gb=256"]
    end

    subgraph Step2["Step 2: 查询 color_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 cmap_all| T5["{\n  'ブラックチタニウム': ('MYW23J/A', 'ブラックチタニウム'),\n  'ホワイトチタニウム': ('MYW53J/A', 'ホワイトチタニウム'),\n  'ナチュラルチタニウム': ('MYW83J/A', 'ナチュラルチタニウム'),\n  ...\n}"]
    end

    subgraph Step3["Step 3: 基准价格"]
        T6["'206,000'"]
        T6 -->|to_int_yen_shop11| T7["206000"]
    end

    subgraph Step4["Step 4: LLM 颜色差额解析"]
        T8["caution: 'シルバー・ブルー：-1,000円(未開封)'\navailable_colors: ('ブラックチタニウム', 'ホワイトチタニウム', ...)"]
        T8 -->|"_lx_parse_color_deltas_shop11\n输入: 'CAUTION: ...\nAVAILABLE_COLORS: ...'"|T9["LLM 返回:\ncolor_delta: 'シルバー' delta_yen=-1000\ncolor_delta: 'ブルー' delta_yen=-1000"]
        T9 -->|"_label_matches_color_unified\n将 'シルバー' 匹配到对应 available_color"| T10["color_deltas = {\n  'ホワイトチタニウム': -1000,\n  ...matched colors...\n}"]
    end

    subgraph Step5["Step 5: 价格计算 (每色一行)"]
        T11["对 color_map 中每个颜色:"]
        T11 --> T12["有 delta 的颜色 → price = 206000 + (-1000) = 205000"]
        T11 --> T13["无 delta 的颜色 → price = 206000 + 0 = 206000"]
    end

    subgraph Step6["Step 6: 输出行"]
        T14["{\n  part_number: 'MYW23J/A',\n  shop_name: 'モバステ',\n  price_new: 206000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYW53J/A',\n  shop_name: 'モバステ',\n  price_new: 205000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
```

### 3.3 型号/容量解析 - LLM vs 正则策略

```mermaid
flowchart TD
    INPUT["storage_name 原始文本"]

    INPUT --> LLM["LLM 解析器 (优先)\n_lx_parse_storage_shop11"]
    LLM --> CHECK{model_norm 和\ncap_gb 均有效?}
    CHECK -->|是| USE_LLM["使用 LLM 结果\n(支持日文别名/非标准写法)"]
    CHECK -->|否| REGEX["正则回退\n_normalize_model_generic\n+ _parse_capacity_gb"]
    REGEX --> CHECK2{正则结果有效?}
    CHECK2 -->|是| USE_REGEX["使用正则结果"]
    CHECK2 -->|否| SKIP["跳过该行"]

    subgraph LLM解析器详细
        L1["prompt: 严格解析器\n要求 model_norm 必须在 valid_models 列表中"]
        L2["examples: 3个 few-shot 示例\n覆盖标准写法/紧凑写法/日文写法"]
        L3["extraction_classes:\ndevice_model + storage_capacity"]
        L4["model: 本地 Ollama (默认 gemma3:1b)"]
    end

    subgraph 正则解析器详细
        R1["_NUM_MODEL_PAT: 匹配 iPhone + 数字 + 后缀"]
        R2["_AIR_PAT: 匹配 iPhone Air"]
        R3["日文别名替换: プロ→Pro / エア→Air 等"]
        R4["容量提取: 支持 GB/TB 格式"]
    end
```

### 3.4 颜色差额解析 - LLM vs 正则策略

```mermaid
flowchart TD
    INPUT["caution_empty 原始文本\n+ available_colors"]

    INPUT --> LLM["LLM 解析器 (优先)\n_lx_parse_color_deltas_shop11"]
    LLM --> CHECK{结果非空?}
    CHECK -->|是| USE_LLM["使用 LLM 结果\n(处理全色/全角/复杂格式)"]
    CHECK -->|空且有文本| REGEX["正则回退\n_extract_color_deltas_shop11\n+ _label_matches_color_unified"]
    REGEX --> USE_REGEX["使用正则结果"]

    subgraph LLM解析器详细
        L1["prompt: 买取表色delta解析专用提示词"]
        L2["examples: 3个 few-shot 示例"]
        L3["model: 本地 Ollama (默认 gemma3:1b)"]
        L4["temperature: 0.0 (确定性输出)"]
        L5["输入包含 AVAILABLE_COLORS 行\n约束输出颜色范围"]
        L6["输出: extraction_class=color_delta\nattributes={delta_yen}"]
    end

    subgraph 正则解析器详细
        R1["_COLOR_GROUP_RE: 匹配 'labels：+/-amount'\n(带冒号的主匹配)"]
        R2["_COLOR_GROUP_FALLBACK_RE: 匹配 'labels +/-amount'\n(无冒号的回退匹配)"]
        R3["_COLOR_SEP_SPLIT_RE: 拆分多颜色标签\n/ 、 ・ 空格 等"]
        R4["结果通过 _label_matches_color_unified\n匹配到 color_map 中的合法颜色"]
    end
```

### 3.5 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY["FAMILY 同义词字典 (shop11)"]
        BLUE["blue 家族\nブルー / 青 / blue"]
        SILVER["silver 家族\nシルバー / 銀 / silver"]
        BLACK["black 家族\nブラック / 黒 / black"]
        WHITE["white 家族\nホワイト / 白 / white"]
        GOLD["gold 家族\nゴールド / 金 / gold"]
        ORANGE["orange 家族\nオレンジ / 橙"]
    end

    LABEL["提取到的 label\n如: 'シルバー'"]
    COLOR["info表中的 color\n如: 'ホワイトチタニウム'"]

    LABEL -->|"查 FAMILY 同义词"| SILVER
    SILVER -->|"同义词 'シルバー' 不在 'ホワイトチタニウム' 中"| NOMATCH["不匹配"]

    LABEL2["提取到的 label\n如: 'ブルー'"]
    COLOR2["info表中的 color\n如: 'マリンブルー'"]

    LABEL2 -->|"子串匹配: 'ブルー' in 'マリンブルー'"| MATCH["匹配成功!"]
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`。shop11 专用 LLM 参数保留。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EXTRACTION_MODE` | `"regex"` | regex / llm / auto（cleaner_tools） |
| `OLLAMA_URL` / `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `SHOP11_OLLAMA_TEMPERATURE` | `"0.0"` | LLM 推理温度 |
| `SHOP11_OLLAMA_TIMEOUT` | `"180"` | LLM 请求超时时间 (秒) |
| `SHOP11_OLLAMA_MAX_TOKENS` | `"512"` | LLM 最大输出 token 数 |
| `IPHONE17_INFO_CSV` | 自动推断路径 (`data/iphone17_info.csv`) | iphone17_info 文件路径 |

**ModelConfig 参数汇总** (`_shop11_model_config`):

| 参数 | 值来源 | 说明 |
|------|--------|------|
| `model_id` | `SHOP11_OLLAMA_MODEL_ID` | 模型标识 |
| `model_url` | `SHOP11_OLLAMA_URL` | 服务端点 |
| `temperature` | `SHOP11_OLLAMA_TEMPERATURE` | 推理温度 |
| `timeout` | `SHOP11_OLLAMA_TIMEOUT` | 超时 (秒) |
| `max_tokens` | `SHOP11_OLLAMA_MAX_TOKENS` | 最大输出 token |
| `format_type` | `FormatType.JSON` | 强制 JSON 输出格式，减少解析失败 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max`, `iPhone17Pro` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
| `_COLOR_GROUP_RE` | `labels[：:]\s*sign?\s*amount` | 主颜色差额匹配 (带冒号) | `シルバー・ブルー：-1,000円` |
| `_COLOR_GROUP_FALLBACK_RE` | `labels\s*sign\s*amount` | 回退颜色差额匹配 (无冒号) | `ブルー -4000` |
| `_COLOR_SEP_SPLIT_RE` | `[／/、，,・\s]+` | 拆分多颜色标签 | `シルバー・ブルー` -> `['シルバー', 'ブルー']` |
| 容量 TB 匹配 | `(\d+(?:\.\d+)?)\s*TB` | 提取 TB 容量 (换算为 GB) | `1TB` -> `1024` |
| 容量 GB 匹配 | `(\d{2,4})\s*GB` | 提取 GB 容量 | `256GB` -> `256` |
| 数字后补空格 | `(\d{2})(?=[A-Za-z])` | 紧凑写法展开 | `17pro` -> `17 pro` |
| 日元金额提取 | `([+\-−－]?)\s*(?:¥\|￥)?\s*([\d][\d,]*)` | `to_int_yen_shop11` 金额解析 | `¥206,000`, `-1,000円` |
| 括号备注去除 | `\（.*?\）\|\(.*?\)` | 去除括号内注释 | `(未開封)` -> 删除 |
| SIM 信息去除 | `SIMフリ[ーｰ–-]?\|シムフリ[ーｰ–-]?\|sim\s*free` | 去除 SIM 噪声 | `SIMフリー` -> 删除 |
