# Shop4 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop4_cleaner.py`
> 店铺名称: モバイルミックス (Mobile Mix)

---

## 一、总流程图

整个 shop4 清洗器的核心入口是 `clean_shop4(df, debug, debug_limit)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

shop4 采用 **块（block）处理** 模式：`data11` 列非空的行标记一个新机型块的开始，后续行直到下一个 `data11` 非空行之前均属于同一块。基准价格通过向上回溯（而非当前行）获取，颜色差额则从整个块中提取。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\ndata / data11 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建 pn_map 映射表\nmodel_norm + cap_gb → color_norm → part_number]
    D --> E[逐行遍历 DataFrame]

    E --> F{data11 列\n是否非空?}
    F -->|空| E
    F -->|非空 → 机型行| G[型号标准化\n_normalize_model_generic]

    G --> H{型号/容量\n能否解析?}
    H -->|否 → SKIP| E
    H -->|是| I[在 pn_map 中\n查找该 key]

    I --> J{pn_map\n是否存在?}
    J -->|否 → SKIP| E
    J -->|是| K[回溯查找基准价格\n_find_base_price\n向上最多 3 行]

    K --> L{基准价格\n是否有效?}
    L -->|否 → SKIP| E
    L -->|是| M[收集颜色差额\n_collect_adjustments_shop4\nLLM 优先 / regex 兜底]

    M --> N{adjustments\n包含 ALL 键?}
    N -->|是| O[所有颜色统一使用\nALL delta\nprice = base + ALL_delta]
    N -->|否| P[按 color_norm 匹配\n各颜色各自 delta\nprice = base + delta]

    O --> Q[生成输出行\npart_number / shop_name / price_new / recorded_at]
    P --> Q
    Q --> E

    E -->|遍历结束| R[组装输出 DataFrame]
    R --> S[去除空值 / 类型转换]
    S --> T[输出: 标准化 DataFrame\npart_number, shop_name, price_new, recorded_at]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop4(df, debug, debug_limit)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> parsedt["parse_dt_aware(val)"]
    clean --> findprice["_find_base_price(df, idx)"]
    clean --> collect["_collect_adjustments_shop4(df, start_idx)"]
    clean --> nextmodel["_next_model_idx(start)"]

    findprice --> toint["to_int_yen(val)"]
    collect --> llm["_collect_adjustments_shop4_llm(df, start_idx)"]
    collect --> regex["_parse_color_delta_shop4_regex(line)"]

    llm --> lxextract["_lx_extract_color_deltas(text)"]
    llm --> getstart["_get_start_pos(extraction)"]
    llm --> splitlbl["_split_labels(label)"]
    llm --> coerce["_coerce_int_maybe(v)"]
    llm --> norm["_norm(s)"]

    lxextract --> lx["lx.extract()\nLangExtract API"]
    lxextract --> prompt["_SHOP4_LE_PROMPT"]
    lxextract --> examples["_SHOP4_LE_EXAMPLES\n6 个 few-shot 示例"]

    regex --> normamt["_normalize_amount_text(s)"]
    regex --> splitlbl
    regex --> toint
    regex --> norm

    coerce --> normamt

    load --> normmod
```

### 2.2 核心函数详细说明

#### `clean_shop4(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `data`, `data11`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`("モバイルミックス"), `price_new`, `recorded_at` 列的 DataFrame
- **处理流程**:
  1. 校验必要列是否存在
  2. 加载 iphone17_info 并构建 `pn_map` (型号+容量 → 颜色→品番) 映射
  3. 逐行遍历，仅在 `data11` 非空时触发处理（标识一个新的机型块）
  4. 对每个机型块：解析型号/容量 → 查 pn_map → 回溯基准价 → 收集颜色差额 → 输出行

#### `_find_base_price(df: pd.DataFrame, idx: int) -> Optional[int]`
- **作用**: 向上回溯查找基准价格
- **流程**:

```mermaid
flowchart TD
    A["输入: df, idx (机型行索引)"] --> B["j = idx - 1"]
    B --> C{j >= 0 且\nj >= idx - 3?}
    C -->|否| D["返回 None"]
    C -->|是| E["读取 df.data 第 j 行"]
    E --> F{"包含 '円' 或\n匹配数字模式?"}
    F -->|否| G["j = j - 1"] --> C
    F -->|是| H["to_int_yen 解析"]
    H --> I{解析成功?}
    I -->|是| J["返回 price (int)"]
    I -->|否| G
```

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_collect_adjustments_shop4(df: pd.DataFrame, start_idx: int) -> Dict[str, int]`
- **作用**: 颜色差额提取的统一入口，采用 **LLM 优先、regex 兜底** 策略

```mermaid
flowchart TD
    A["_collect_adjustments_shop4(df, start_idx)"] --> B{SHOP4_USE_LLM\n且 lx 可用?}
    B -->|是| C["_collect_adjustments_shop4_llm(df, start_idx)"]
    C --> D{LLM 调用成功?}
    D -->|是| E[返回 LLM 结果]
    D -->|异常| F["regex 兜底"]
    B -->|否| F
    F --> G["逐行遍历 block\n_parse_color_delta_shop4_regex(line)"]
    G --> H["返回 regex 结果\nDict: color_norm/ALL → delta"]
```

#### `_collect_adjustments_shop4_llm(df: pd.DataFrame, start_idx: int) -> Dict[str, int]`
- **作用**: 用 LangExtract 一次性解析整个机型块的所有颜色±金额
- **返回**: `{ color_norm | "ALL" : delta_int }`
- **流程**:

```mermaid
flowchart TD
    A[输入: df, start_idx] --> B["收集 block 文本\nstart_idx → 下一个 data11 非空行前"]
    B --> C["将各行拼接为 block_text\n用换行符连接"]
    C --> D["记录 line0 起止位置\n用于判断同行全色"]
    D --> E["_lx_extract_color_deltas(block_text)"]
    E --> F{返回结果\n是否为空?}
    F -->|空| G["返回空字典 {}"]
    F -->|非空| H["按 _get_start_pos 排序\n保持出现顺序"]
    H --> I[遍历每个 extraction]
    I --> J{extraction_class\n== color_delta?}
    J -->|否| I
    J -->|是| K["提取 label + delta_yen\n从 extraction_text / attributes"]
    K --> L["_coerce_int_maybe(delta_yen)"]
    L --> M{delta 有效?}
    M -->|否| N{"label 含全色\n且无数字?"}
    N -->|是| O["delta = 0"]
    N -->|否| I
    M -->|是| P["检查是否同行全色\nstart_pos 在 line0 范围内"]
    O --> P
    P --> Q["_split_labels 拆分复合标签"]
    Q --> R{"含全色?"}
    R -->|是| S["result['ALL'] = delta"]
    R -->|否| T["result[label_norm] = delta"]
    S --> I
    T --> I
    I -->|遍历完| U["如 global_all_delta 非空\n覆盖 result['ALL']"]
    U --> V["返回 result"]
```

#### `_lx_extract_color_deltas(text: str) -> List`
- **作用**: 对文本做一次 LangExtract 抽取，返回 extractions 列表
- **配置参数**:
  - `model_id`: SHOP4_OLLAMA_MODEL_ID (默认 gemma3:1b)
  - `model_url`: SHOP4_OLLAMA_URL (默认 localhost:11434)
  - `temperature`: 0.0 (确定性输出)
  - `prompt`: _SHOP4_LE_PROMPT (专用日语 iPhone 价格表解析提示词)
  - `examples`: 6 个 few-shot 示例
  - `extraction_passes`: 1
  - `max_char_buffer`: 1500

#### `_parse_color_delta_shop4_regex(line: str) -> Optional[List[Tuple[str, int]]]`
- **作用**: 正则版单行颜色差额提取（regex 兜底时逐行调用）
- **流程**:

```mermaid
flowchart TD
    A["输入: line (单行文本)"] --> B{line 为空?}
    B -->|是| C["返回 None"]
    B -->|否| D{"line == '全色'?"}
    D -->|是| E["返回 [('全色', 0)]"]
    D -->|否| F["尝试 _COLOR_DELTA_RE.match"]
    F --> G{匹配成功?}
    G -->|是| H["提取 label / sign / amount"]
    H --> I["LABEL_SPLIT_RE 拆分复合标签"]
    I --> J["返回 [(label, delta), ...]"]
    G -->|否| K["正则搜索金额模式\n([+-])?数字+円?"]
    K --> L{找到金额?}
    L -->|否| M{"包含全色?"}
    M -->|是| E
    M -->|否| C
    L -->|是| N["截取金额前文本作为 label"]
    N --> O{label 非空?}
    O -->|否| C
    O -->|是| I
```

#### `_get_start_pos(extraction) -> int`
- **作用**: 从 LangExtract 的 extraction 对象中提取 `char_interval` 的起始位置
- **兼容性**: 支持属性访问 (`start_pos`/`start`/`begin`) 和字典访问两种方式

#### `_split_labels(label: str) -> List[str]`
- **作用**: 按分隔符 (`／ / 、 ， , ・ 空格`) 拆分复合颜色标签
- **示例**: `"シルバー/ディープブルー"` → `["シルバー", "ディープブルー"]`

#### `_coerce_int_maybe(v) -> Optional[int]`
- **作用**: 宽容地将各种类型值转为 int（支持全角负号/半角负号/字符串数字）
- **处理链**: 判断类型 → 提取符号 → `_normalize_amount_text` 解析数字部分 → 返回带符号整数

#### `_normalize_amount_text(s: str) -> Optional[int]`
- **作用**: 将含全角数字/逗号/货币符号的金额文本转为半角纯数字 int
- **处理**: 先做全角→半角转换表 (`_FZ_TO_HZ_TRANS`)，再正则提取数字串

#### `_norm(s: str) -> str`
- **作用**: 简单文本归一化，strip 空白
- **返回**: `(s or "").strip()`

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: data, data11, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        PNMAP["pn_map 字典\n(model_norm, cap_gb) → {\n  color_norm: part_number\n}"]
        ADJ["adjustments 字典\n{ color_norm: delta_int }\n或 { 'ALL': delta_int }\n如: {'シルバー': -1000, 'ALL': -2000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> PNMAP
    RAW -->|"逐行遍历\ndata11 非空触发"| PROC

    subgraph PROC[块处理流程]
        direction TB
        P1["data11 → model_norm + cap_gb"]
        P2["回溯 data 列 → base_price (int)"]
        P3["block 全文 → adjustments (LLM/regex)"]
        P4["adjustments 含 ALL?\n→ 全色统一价 / 各色独立价"]
        P5["base_price + delta → price_new"]
    end

    PNMAP --> PROC
    PROC --> ADJ
    ADJ --> OUT
```

### 3.2 块（Block）结构示意

shop4 的数据以"块"为单位组织。`data11` 列非空标记一个新机型块的开始，基准价格在机型行 **上方** 回溯获取。

```
行号  data11              data
───── ──────────────────  ─────────────────────────
 i-2  (空)                "150,000円"          ← 基准价格（回溯找到）
 i-1  (空)                "..."
 i    "iPhone17 Pro 256"  "シルバー/ブルー-1,000円"  ← 机型行（block 起点）
 i+1  (空)                "ディープブルー-3,000円"    ← block 内容行
 i+2  (空)                "全色-2,000円"            ← block 内容行
 i+3  "iPhone17 Plus 128" "..."                     ← 下一个 block 起点
```

### 3.3 单行数据处理示例

以一个实际机型块为例，展示完整的数据转换过程:

```
输入块:
  行 i-1: data = "180,000円"           (基准价格行)
  行 i  : data11 = "iPhone17 Pro Max 256GB SIMフリー"
           data  = "シルバー-1,000円"
  行 i+1: data  = "ディープブルー-3,000円"
  time-scraped = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 型号/容量解析"]
        T1["data11 = 'iPhone17 Pro Max 256GB SIMフリー'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step2["Step 2: 查询 pn_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 pn_map| T5["{\n  'ブラックチタニウム': 'MYW23J/A',\n  'ホワイトチタニウム': 'MYW53J/A',\n  'ナチュラルチタニウム': 'MYW83J/A',\n  'サンドチタニウム': 'MYWF3J/A',\n  ...\n}"]
    end

    subgraph Step3["Step 3: 基准价格回溯"]
        T6["_find_base_price(df, i)"]
        T6 -->|"回溯 i-1 行: '180,000円'"| T7["to_int_yen → 180000"]
    end

    subgraph Step4["Step 4: 颜色差额收集"]
        T8["block_text =\n'シルバー-1,000円\nディープブルー-3,000円'"]
        T8 -->|"_collect_adjustments_shop4\n(LLM 优先)"| T9["{\n  'シルバー': -1000,\n  'ディープブルー': -3000\n}"]
    end

    subgraph Step5["Step 5: 价格计算 (无 ALL 键)"]
        T10["对 pn_map 中每个颜色:"]
        T10 --> T11["ブラックチタニウム → 无匹配 → delta=0 → price=180000"]
        T10 --> T12["ホワイトチタニウム → 无匹配 → delta=0 → price=180000"]
        T10 --> T13["... 其他未匹配颜色 → delta=0 → price=180000"]
    end

    subgraph Step6["Step 6: 输出行"]
        T14["{\n  part_number: 'MYW23J/A',\n  shop_name: 'モバイルミックス',\n  price_new: 180000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYW53J/A',\n  shop_name: 'モバイルミックス',\n  price_new: 180000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
```

### 3.4 含"全色/ALL"的处理示例

当 adjustments 中包含 `"ALL"` 键时，所有颜色统一使用该 delta：

```
输入块:
  行 i-1: data = "150,000円"
  行 i  : data11 = "iPhone17 Plus 128GB"
           data  = "全色-2,000円"

adjustments = {"ALL": -2000}
```

```mermaid
flowchart TD
    A["base_price = 150000"] --> B{"adjustments 含 ALL?"}
    B -->|是| C["final_price = 150000 + (-2000) = 148000"]
    C --> D["所有颜色统一输出 price_new = 148000"]
    D --> E["part_number: XXX, price_new: 148000\npart_number: YYY, price_new: 148000\npart_number: ZZZ, price_new: 148000\n..."]
```

### 3.5 颜色差额提取 - LLM vs Regex 策略

```mermaid
flowchart TD
    INPUT["block 文本\n(从机型行到下一个 data11 前)"]

    INPUT --> DISPATCH["_collect_adjustments_shop4"]
    DISPATCH --> CHECK{SHOP4_USE_LLM\n且 lx 可用?}

    CHECK -->|是| LLM["LLM 解析器\n_collect_adjustments_shop4_llm"]
    LLM --> LLM_OK{调用成功?}
    LLM_OK -->|是| USE_LLM["使用 LLM 结果\n(整个 block 一次性解析)"]
    LLM_OK -->|异常| REGEX["regex 兜底"]

    CHECK -->|否| REGEX
    REGEX --> SCAN["逐行遍历 block\n_parse_color_delta_shop4_regex(line)"]
    SCAN --> USE_REGEX["使用 regex 结果\n(逐行独立解析后合并)"]

    subgraph LLM解析器详细
        L1["prompt: 日语 iPhone 价格表颜色差额抽取提示词"]
        L2["examples: 6个 few-shot 示例\n涵盖单色/多色/全色/全角/零差额"]
        L3["model: gemma3:1b (本地 Ollama)"]
        L4["temperature: 0.0 (确定性输出)"]
        L5["输出: extraction_class=color_delta\nextraction_text=颜色名\nattributes={delta_yen: int}"]
    end

    subgraph Regex解析器详细
        R1["_COLOR_DELTA_RE: 匹配 'label ± amount 円'"]
        R2["全角→半角归一化"]
        R3["LABEL_SPLIT_RE: 拆分复合标签"]
        R4["全色 特殊处理: delta=0"]
    end
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EXTRACTION_MODE` | `"regex"` | regex / llm / auto（cleaner_tools） |
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `OLLAMA_URL` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `IPHONE17_INFO_CSV` | 自动推断路径 (`data/iphone17_info.csv`) | iphone17_info 参考表文件路径 |

> **注**: `debug` 和 `debug_limit` 为函数参数（非环境变量），分别控制是否打印调试信息和最大打印机型数（默认 30）。

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_COLOR_DELTA_RE` | `^\s*(?P<label>全色\|[\S　 ]*?[^\s　])\s*(?P<sign>[+\-−－])?\s*(?P<amount>\d[\d,]*)\s*円?\s*$` | 匹配"颜色标签±金额"格式的完整行 | `シルバー-1,000円`, `全色+0円`, `ディープブルー-3000` |
| `LABEL_SPLIT_RE` | `[／/、，,・\s]+` | 拆分复合颜色标签 | `シルバー/ディープブルー` → `["シルバー", "ディープブルー"]` |
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号 iPhone 机型 | `iPhone 17 Pro Max`, `iPhone17Pro` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air 机型 | `iPhone Air` |
| `_FZ_TO_HZ_TRANS` | 全角→半角转换表 | 将全角数字/符号/货币符号统一为半角 | `０` → `0`, `－` → `-`, `￥` → (删除) |
| `_DEBUG_HINT_PAT` | `(全色\|シルバー\|ブルー\|...\|[+-]\s*\d\|[\d,]+\s*円)` | 调试用：判断行是否包含颜色/价格相关内容 | `シルバー-1,000円`, `全色`, `150,000円` |
| 金额回退正则 (内联) | `([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)\s*円?` | `_parse_color_delta_shop4_regex` 中 `_COLOR_DELTA_RE` 不匹配时的回退金额搜索 | `＋０円`, `-3,000円` |
| 基准价格行检测 (内联) | `\d[\d,]*` | `_find_base_price` 中判断回溯行是否含金额 | `180,000`, `150000円` |
