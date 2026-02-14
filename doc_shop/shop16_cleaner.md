# Shop16 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop16_cleaner.py`
> 店铺名称: 携帯空間

---

## 一、总流程图

整个 shop16 清洗器的核心入口是 `clean_shop16(df, debug)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\niPhone 17 Pro Max / 説明1 / 買取価格 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建颜色映射表\n_build_color_map]
    D --> E[逐行遍历 DataFrame]

    E --> F{説明1 或 型号列\n包含 未開封?}
    F -->|否| E
    F -->|是| G[型号标准化\n_normalize_model_generic]

    G --> H{型号/容量\n能否解析?}
    H -->|否| E
    H -->|是| I[在 color_map 中\n查找该机型]

    I --> J{color_map\n是否存在?}
    J -->|否| E
    J -->|是| K[价格文本标准化\n_normalize_price_text_shop16]

    K --> L[LLM 抽取\n_lx_extract_price_parts_shop16]
    L --> L1{LLM 成功?}
    L1 -->|是| M[获取 base_price / deltas / absps]
    L1 -->|否| M1[标记 llm_ok=False\n回退旧逻辑]

    M --> N[Guardrail A: 基础价纯文本检测\n_is_base_only_price_text]
    M1 --> N
    N -->|纯基础价| N1[丢弃所有 color 抽取]
    N -->|非纯基础价| O[Guardrail B: 共享差价纠错\n_extract_shared_delta_map_shop16]

    N1 --> O
    O --> P[Guardrail C: 证据过滤\nlabel+金额必须在原文出现]
    P --> Q{llm_ok=False 且\n无颜色信息?}
    Q -->|是| Q1[正则回退\n_extract_color_deltas_shop16\n_extract_color_abs_prices_shop16]
    Q -->|否| R[标签映射到 color_norm\n_label_matches_color_unified]
    Q1 --> R

    R --> S[计算每个颜色的最终价格\nabs优先 否则 base + delta]
    S --> T[生成输出行\npart_number / shop_name / price_new / recorded_at]
    T --> E

    E -->|遍历结束| U[组装输出 DataFrame]
    U --> V[去除空值 / 类型转换]
    V --> W[输出: 标准化 DataFrame\npart_number, shop_name, price_new, recorded_at]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop16(df, debug)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> buildcm["_build_color_map(info_df)"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> normprice["_normalize_price_text_shop16(s)"]
    clean --> lxextract["_lx_extract_price_parts_shop16(price_text)"]
    clean --> baseonly["_is_base_only_price_text(text)"]
    clean --> shareddelta["_extract_shared_delta_map_shop16(text)"]
    clean --> labelmatch["_label_matches_color_unified(label, color_raw, color_norm)"]
    clean --> parsedt["parse_dt_aware(val)"]
    clean --> normlbl["_normalize_label_shop16(lbl)"]

    lxextract --> examples["_shop16_price_examples()"]
    lxextract --> lxlib["langextract.extract()"]
    lxextract --> toint["to_int_yen(val)"]
    lxextract --> tosigned["_to_signed_int_yen(val)"]
    lxextract --> splitlbl["_split_labels_shop16(lbl)"]

    clean -->|"llm_ok=False 回退"| regexdelta["_extract_color_deltas_shop16(text)"]
    clean -->|"llm_ok=False 回退"| regexabs["_extract_color_abs_prices_shop16(text)"]
    clean --> baseprice["_extract_base_price_shop16(text)"]

    regexdelta --> toint
    regexabs --> toint
    shareddelta --> normlbl
    shareddelta --> toint

    buildcm --> normmod
    labelmatch --> familysyn["cleaner_tools 颜色家族同义词"]
    splitlbl --> normlbl
```

### 2.2 核心函数详细说明

#### `clean_shop16(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `iPhone 17 Pro Max`, `説明1`, `買取価格`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`, `price_new`, `recorded_at` 列的 DataFrame
- **特殊逻辑**: 仅处理 `説明1` 或型号列中包含 `未開封` 的行

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_normalize_price_text_shop16(s: object) -> str`
- **作用**: 统一价格文本格式
- **处理**: 全角空格/不间断空格→半角空格 / 换行符→` / ` / 压缩多余空白 / 合并重复分隔符

```mermaid
flowchart TD
    A["输入: '￥86100\\n黒:-1000円'"] --> B["替换 \\u3000/\\xa0/\\t → 空格"]
    B --> C["换行 → ' / '"]
    C --> D["压缩连续空白"]
    D --> E["合并重复 ' / '"]
    E --> F["输出: '￥86100 / 黒:-1000円'"]
```

#### `_lx_extract_price_parts_shop16(price_text: str) -> Tuple`
- **作用**: 核心 LLM 抽取函数，使用 LangExtract + Ollama 从价格文本中提取结构化信息
- **装饰器**: `@lru_cache(maxsize=4096)` 缓存相同输入的结果
- **返回**: `(base_price, deltas, abs_prices, debug_extractions)`
- **三个抽取类**:
  - `base_price`: 基础价格（如 `"86,100円"`）
  - `color_delta`: 颜色差价（如 `"黒:-1000円"`）
  - `color_abs`: 颜色绝对价（如 `"黒￥86100"`）

```mermaid
flowchart TD
    A["输入 price_text"] --> B{文本为空?}
    B -->|是| C["返回 (None, [], [], [])"]
    B -->|否| D["导入 langextract"]
    D --> E["获取 6 个 few-shot 示例\n_shop16_price_examples()"]
    E --> F["调用 lx.extract()\nmodel_id=OLLAMA_MODEL_ID\nmodel_url=OLLAMA_URL\nprompt=SHOP16_PRICE_PROMPT"]

    F --> G["遍历 extractions"]
    G --> H{extraction_class?}

    H -->|base_price| I["提取 amount_yen → base_price\n仅取第一个"]
    H -->|color_delta| J["提取 color_label + delta_yen\n_split_labels_shop16 拆分多标签\n_to_signed_int_yen 解析带符号金额"]
    H -->|color_abs| K["提取 color_label + amount_yen\n_split_labels_shop16 拆分多标签"]

    I --> L["返回\n(base_price, deltas, abs_prices, debug_extractions)"]
    J --> L
    K --> L
```

#### `_extract_color_deltas_shop16(text: str) -> List[Tuple[str, int]]`
- **作用**: 正则版颜色差额提取（LLM 失败时的回退方案）
- **流程**:

```mermaid
flowchart TD
    A["输入 text"] --> B["FIRST_YEN_RE 去掉基础价前缀"]
    B --> C["SPLIT_TOKENS_RE 按分隔符拆分"]
    C --> D["逐段遍历"]
    D --> E{COLOR_DELTA_RE\n匹配成功?}
    E -->|是| F["提取 label + sign + amount\n添加到结果"]
    E -->|否| G["暂存为 pending_labels\n等待后续金额"]
    F --> H{有 pending_labels?}
    H -->|是| I["为所有 pending_labels\n应用相同金额"]
    H -->|否| D
    I --> D
    G --> D
    D -->|遍历完| J["返回结果列表"]
```

#### `_extract_color_abs_prices_shop16(text: str) -> List[Tuple[str, int]]`
- **作用**: 正则版颜色绝对价提取
- **匹配模式**: `COLOR_ABS_RE` 匹配 `"颜色名￥金额"` 格式

#### `_is_base_only_price_text(price_text_norm: str) -> bool`
- **作用**: Guardrail A - 检测文本是否仅包含一个基础价格（无颜色信息）
- **逻辑**: 正则匹配 `_BASE_ONLY_RE`，若整段文本只是 `"￥86100"` 或 `"86,100円"` 则返回 True
- **效果**: 丢弃所有 LLM 抽出的颜色信息（防止幻觉）

#### `_extract_shared_delta_map_shop16(price_text_norm: str) -> Dict[str, int]`
- **作用**: Guardrail B - 提取共享差价（如 `"オレンジ/青 -1500"`）用于纠正 LLM 错误
- **流程**:

```mermaid
flowchart TD
    A["输入 price_text_norm"] --> B["FIRST_YEN_RE 去掉基础价前缀"]
    B --> C["_GROUP_SHARED_DELTA_RE 匹配\n多颜色共享差价模式"]
    C --> D["拆分 labels 段\n按 ／/、，, 分割"]
    D --> E["逐标签归一化\n_normalize_label_shop16"]
    E --> F["返回 Dict\n如 {オレンジ: -1500, 青: -1500}"]
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
    C -->|否| D[查 FAMILY_SYNONYMS_shop16\n颜色家族同义词表]
    D --> E{label 在家族表中?}
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

#### `_normalize_label_shop16(lbl: str) -> str`
- **作用**: 归一化颜色标签
- **处理**: 去掉空白字符 / 去掉尾部 `カラー` `色` / 去掉黏在标签末尾的金额符号

#### `_split_labels_shop16(lbl: str) -> List[str]`
- **作用**: 拆分多标签字符串（如 `"青/オレンジ"` → `["青", "オレンジ"]`）

#### `_to_signed_int_yen(x: object) -> Optional[int]`
- **作用**: 将带符号的金额文本解析为有符号整数
- **优先级**: 先查找带符号数字（差价），再取最后一个无符号数字（兜底）

#### `_extract_base_price_shop16(text: str) -> Optional[int]`
- **作用**: 提取基础价格（LLM 抽取失败时的兜底）
- **逻辑**: 先用 `FIRST_YEN_RE` 匹配，失败则直接用 `to_int_yen` 兜底

#### `_shop16_price_examples() -> List[ExampleData]`
- **作用**: 提供 6 个 few-shot 示例用于 LangExtract 抽取
- **覆盖场景**:
  1. 基础价 + 多颜色差价（正负号）
  2. 基础价 + 多颜色共享差价（`青/オレンジ -5000円`）
  3. 纯颜色绝对价（`黒￥86100/青￥87100`）
  4. 基础价 + 差价为0 + 差价为负（`ホワイト +0円／ブラック -3000円`）
  5. 基础价 + 全角冒号差价（`ブルー：+2,000円`）
  6. 基础价 + 换行后差价（`￥197000\n\nオレンジ-1000`）

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: iPhone 17 Pro Max, 説明1, 買取価格, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        CMAP["color_map 字典\n(model_norm, cap_gb) → {\n  color_norm: (part_number, color_raw)\n}"]
        LLM_OUT["LLM 抽取结果\nbase_price: Optional int\ndeltas: list of (label, delta)\nabsps: list of (label, abs_price)"]
        CDM["color_delta_map 字典\n{color_norm: delta_int}"]
        CAM["color_abs_map 字典\n{color_norm: abs_price_int}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> CMAP
    RAW -->|"逐行读取"| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["説明1 / 型号列 → 未開封 过滤"]
        P2["iPhone 17 Pro Max → model_norm + cap_gb"]
        P3["買取価格 → _normalize_price_text_shop16 → price_text"]
        P4["price_text → _lx_extract_price_parts_shop16 → LLM 抽取"]
        P5["Guardrail A/B/C → 过滤和纠错"]
        P6["标签匹配 → color_delta_map / color_abs_map"]
        P7["abs优先 否则 base+delta → price_new"]
    end

    CMAP --> PROC
    PROC --> LLM_OUT
    LLM_OUT --> CDM
    LLM_OUT --> CAM
    CDM --> OUT
    CAM --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  iPhone 17 Pro Max = "iPhone17 Pro Max 256GB SIMフリー"
  説明1            = "新品未開封"
  買取価格         = "￥186000\nオレンジ/青 -1500円"
  time-scraped     = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 未開封 过滤"]
        S1A["説明1 = '新品未開封'"]
        S1A --> S1B["包含 '未開封' → 继续处理"]
    end

    subgraph Step2["Step 2: 型号解析"]
        T1["'iPhone17 Pro Max 256GB SIMフリー'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step3["Step 3: 查询 color_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 cmap_all| T5["{\n  'ブラックチタニウム': ('MYW23J/A', ...),\n  'ホワイトチタニウム': ('MYW53J/A', ...),\n  'ナチュラルチタニウム': ('MYW83J/A', ...),\n  ...\n}"]
    end

    subgraph Step4["Step 4: 价格文本标准化"]
        T6["'￥186000\\nオレンジ/青 -1500円'"]
        T6 -->|_normalize_price_text_shop16| T7["'￥186000 / オレンジ/青 -1500円'"]
    end

    subgraph Step5["Step 5: LLM 抽取 + 三重 Guardrail"]
        T8["_lx_extract_price_parts_shop16"]
        T8 --> T9["base_price = 186000"]
        T8 --> T10["deltas = [('オレンジ', -1500), ('青', -1500)]"]
        T10 --> T11["Guardrail A: 非纯基础价 → 保留"]
        T11 --> T12["Guardrail B: 共享差价纠错\nshared_delta_map = {オレンジ: -1500, 青: -1500}\n纠正后: deltas 不变"]
        T12 --> T13["Guardrail C: 证据过滤\n'オレンジ' in 原文 ✓ / '1500' in 原文 ✓\n'青' in 原文 ✓ / '1500' in 原文 ✓\n→ 全部保留"]
    end

    subgraph Step6["Step 6: 标签匹配 + 价格计算"]
        T14["对 color_map 中每个颜色:"]
        T14 --> T15["匹配到 'オレンジ' → delta=-1500 → price=184500"]
        T14 --> T16["匹配到 '青' 相关 → delta=-1500 → price=184500"]
        T14 --> T17["未匹配颜色 → delta=0 → price=186000"]
    end

    subgraph Step7["Step 7: 输出行"]
        T18["{\n  part_number: 'MYW23J/A',\n  shop_name: '携帯空間',\n  price_new: 186000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYWXXX',\n  shop_name: '携帯空間',\n  price_new: 184500,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
    Step6 --> Step7
```

### 3.3 LLM 优先、正则回退策略

```mermaid
flowchart TD
    INPUT["買取価格 原始文本"]

    INPUT --> NORM["_normalize_price_text_shop16\n换行 → ' / '"]
    NORM --> LLM["LLM 抽取器\n_lx_extract_price_parts_shop16\n(LangExtract + Ollama)"]

    LLM --> CHECK_OK{llm_ok?}
    CHECK_OK -->|是| GUARD["三重 Guardrail 过滤"]
    CHECK_OK -->|否| FALLBACK["正则回退\n_extract_color_deltas_shop16\n_extract_color_abs_prices_shop16"]

    GUARD --> CHECK_EMPTY{过滤后\n有颜色信息?}
    CHECK_EMPTY -->|是| USE_LLM["使用 LLM 结果\n(经 Guardrail 验证)"]
    CHECK_EMPTY -->|否| USE_LLM

    FALLBACK --> USE_REGEX["使用正则结果\n(llm_ok=False 时的容错)"]

    subgraph LLM抽取器详细
        L1["prompt: SHOP16_PRICE_PROMPT\n买取価格解析专用提示词"]
        L2["examples: 6 个 few-shot 示例\n覆盖 base_price / color_delta / color_abs"]
        L3["model: OLLAMA_MODEL_ID (默认 gemma3:1b)"]
        L4["model_url: OLLAMA_URL (默认 localhost:11434)"]
        L5["@lru_cache(maxsize=4096) 缓存"]
        L6["extraction_passes=1\nmax_char_buffer=300"]
    end

    subgraph 三重Guardrail详细
        GA["Guardrail A: _is_base_only_price_text\n纯基础价文本 → 丢弃所有颜色抽取"]
        GB["Guardrail B: _extract_shared_delta_map_shop16\n共享差价正则纠错"]
        GC["Guardrail C: 证据过滤\nlabel + 金额绝对值 必须在原文出现"]
    end
```

### 3.4 三重 Guardrail 防幻觉机制

```mermaid
flowchart TD
    LLM_RESULT["LLM 抽取结果\nbase_price / deltas / absps"]

    LLM_RESULT --> GA{"Guardrail A\n_is_base_only_price_text?"}
    GA -->|"文本仅含基础价\n如 '￥86100'"| GA_ACT["丢弃全部 deltas 和 absps\n防止 LLM 凭空编造颜色"]
    GA -->|"含颜色信息"| GB

    GA_ACT --> GB["Guardrail B\n_extract_shared_delta_map_shop16"]
    GB --> GB_CHECK{发现共享差价\n且 deltas 非空?}
    GB_CHECK -->|是| GB_ACT["用正则确定性证据\n逐标签纠正 delta 值\n如: LLM 抽 '青:+1500' → 纠正为 '青:-1500'"]
    GB_CHECK -->|否| GC

    GB_ACT --> GC["Guardrail C\n逐条证据过滤"]
    GC --> GC_D["遍历 deltas:\n1. label 必须在 price_text 中出现\n2. abs(delta) 数字必须在原文出现"]
    GC --> GC_A["遍历 absps:\n1. label 必须在 price_text 中出现\n2. amount 数字必须在原文出现"]

    GC_D --> FINAL["过滤后的 deltas / absps"]
    GC_A --> FINAL
```

### 3.5 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY_SYNONYMS_shop16
        BLUE["blue 家族\nブルー / 青 / マリン"]
        BLACK["black 家族\nブラック / 黒"]
        WHITE["white 家族\nホワイト / 白"]
        GREEN["green 家族\nグリーン / 緑"]
        RED["red 家族\nレッド / 赤"]
        GOLD["gold 家族\nゴールド / 金"]
        SILVER["silver 家族\nシルバー / 銀"]
        GRAY["gray 家族\nグレー / グレイ / 灰"]
        OTHER["... 其他家族\nyellow / orange / natural"]
    end

    LABEL["提取到的 label\n如: 'ブルー'"]
    COLOR["info表中的 color\n如: 'マリンブルー'"]

    LABEL -->|"查 FAMILY_SYNONYMS_shop16"| BLUE
    BLUE -->|"同义词 'ブルー' in 'マリンブルー'"| MATCH["匹配成功!"]
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`，各 shop 通用。

| 环境变量 / 常量 | 默认值 | 说明 |
|---------|--------|------|
| `OLLAMA_URL` / `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `OLLAMA_MODEL_ID` / `OLLAMA_MODEL` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `EXTRACTION_MODE` | `"regex"` | 抽取模式：regex / llm / auto（cleaner_tools） |
| `MODEL_COL` | `"iPhone 17 Pro Max"` | 输入 DataFrame 的型号列名 |
| `DESC_COL` | `"説明1"` | 输入 DataFrame 的描述/备注列名 |
| `PRICE_COL` | `"買取価格"` | 输入 DataFrame 的价格列名 |
| `IPHONE17_INFO_CSV` | 自动推断路径 | iphone17_info 文件路径（环境变量 `IPHONE17_INFO_CSV`） |
| `lru_cache maxsize` | `4096` | `_lx_extract_price_parts_shop16` 的缓存大小 |
| `extraction_passes` | `1` | LangExtract 抽取轮数 |
| `max_char_buffer` | `300` | LangExtract 最大字符缓冲 |
| `fence_output` | `False` | LangExtract 是否使用 fence 输出 |
| `use_schema_constraints` | `False` | LangExtract 是否使用 schema 约束 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `COLOR_DELTA_RE` | `label[：:\-]\s*[+\-−－]?\s*amount円` | 匹配"颜色±金额"差价 | `黒:-1,000円`, `青:+500円` |
| `COLOR_ABS_RE` | `label\s*￥\s*amount` | 匹配"颜色￥绝对价" | `黒￥86100`, `青￥87100` |
| `FIRST_YEN_RE` | `(?:￥\|¥)?\s*(\d[\d,]*)\s*円?` | 提取第一个日元金额（基础价） | `￥86100`, `86,100円` |
| `SPLIT_TOKENS_RE` | `[／/、，,]\|(?:\s*;\s*)` | 拆分多个颜色条目 | `黒:-1000／青:+500` |
| `_BASE_ONLY_RE` | `^\s*(?:￥\|¥)?\s*\d[\d,]*\s*(?:円)?\s*$` | 检测纯基础价文本（Guardrail A） | `￥86100`, `86,100円` |
| `_GROUP_SHARED_DELTA_RE` | `labels\s*[+\-−－]\s*amount(?:円)?` | 匹配多颜色共享差价（Guardrail B） | `オレンジ/青 -1500円` |
| `_TRAILING_AMOUNT_IN_LABEL_RE` | `[：:]?\s*￥?\s*[+\-−－]?\s*\d[\d,]*\s*(?:円)?\s*$` | 去掉标签末尾黏附的金额 | `ブルー:-1000円` → `ブルー` |
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
