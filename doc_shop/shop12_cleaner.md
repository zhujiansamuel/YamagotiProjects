# Shop12 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop12_cleaner.py`
> 店铺名称: トゥインクル (Twinkle)

---

## 一、总流程图

整个 shop12 清洗器的核心入口是 `clean_shop12(df)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\nモデルナンバー / 備考1 / 買取価格 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]
    C --> D[构建颜色映射表 cmap_all\n逐行归一化 model_name / capacity / color]
    D --> E[逐行遍历 DataFrame]

    E --> F{買取価格\n是否有效?}
    F -->|无效| E
    F -->|有效| G[型号标准化\n_normalize_model_generic]

    G --> H{型号/容量\n能否解析?}
    H -->|否| E
    H -->|是| I[在 cmap_all 中\n查找该机型+容量]

    I --> J{color_map\n是否存在?}
    J -->|否| E
    J -->|是| K[预处理備考1\n_normalize_remark_for_llm\n去除開封相关行]

    K --> L[LLM 规则提取\n_parse_rules_with_langextract]
    L --> M{abs_list 和 delta_list\n是否都为空?}
    M -->|是且有备注| N[正则回退\n_fallback_parse_rules]
    M -->|否| O[颜色标签匹配\n_label_matches_color_unified]
    N --> O

    O --> P{有 ALL 差额?}
    P -->|是| Q[所有颜色统一价\nprice = base + ALL_delta]
    P -->|否| R{有绝对价?}
    R -->|是| S[绝对价覆盖\n匹配色用绝对价\n未匹配色用 base]
    R -->|否| T[普通差额\nprice = base + delta]

    Q --> U[生成输出行\npart_number / shop_name / price_new / recorded_at]
    S --> U
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
    clean["clean_shop12(df)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> toint["to_int_yen(val)"]
    clean --> normremark["_normalize_remark_for_llm(remark_raw)"]
    clean --> llm["_parse_rules_with_langextract(remark)"]
    clean --> fallback["_fallback_parse_rules(text)"]
    clean --> labelmatch["_label_matches_color_unified(label, color_raw, color_norm)"]
    clean --> parsedt["parse_dt_aware(val)"]

    llm --> lxextract["langextract.extract()"]
    llm --> lxexamples["_lx_examples()"]
    llm --> normamt["_norm_amount_to_int(s)"]

    fallback --> normamt
    fallback --> fallback_abs["_FALLBACK_ABS_RE 正则"]
    fallback --> fallback_delta["_FALLBACK_DELTA_RE 正则"]

    labelmatch --> entojp["EN_TO_JP 字典查表"]
    labelmatch --> norm["_norm(s)"]

    normremark --> reopening["去除開封相关行"]

    load --> normmod
```

### 2.2 核心函数详细说明

#### `clean_shop12(df: pd.DataFrame) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `モデルナンバー`, `備考1`, `買取価格`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`("トゥインクル"), `price_new`, `recorded_at` 列的 DataFrame
- **价格输出策略**: ALL差额优先 > 绝对价覆盖 > 普通差额

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro, エア→Air) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_normalize_remark_for_llm(remark_raw: str) -> str`
- **作用**: 对備考1原始文本进行预处理，去除開封相关内容，只保留新品价规则文本
- **流程**:

```mermaid
flowchart TD
    A[输入 remark_raw] --> B["在開封关键词前插入换行\n※開封品 / 開封品 / 開封済 / 開封"]
    B --> C[按换行符拆分为多行]
    C --> D[逐行过滤]
    D --> E{行中包含\n開封 关键词?}
    E -->|是| F[丢弃此行]
    E -->|否| G[保留此行]
    F --> D
    G --> D
    D -->|遍历完| H["合并保留行\n返回 remark_for_llm"]
```

#### `_parse_rules_with_langextract(remark_for_llm: str) -> Tuple[abs_list, delta_list, llm_dbg]`
- **作用**: 使用 LangExtract + Ollama 从備考1文本中提取颜色价格规则，带 `@lru_cache(maxsize=8192)` 缓存
- **返回**:
  - `abs_list`: `[(label_raw, absolute_price_yen), ...]` 绝对价列表
  - `delta_list`: `[(label_raw, delta_yen), ...]` 差额列表
  - `llm_dbg`: `[(effective_class, extraction_text, attrs), ...]` 调试信息
- **两种提取类型**:
  - `"delta"`: 颜色±金额，如 `orange-1000円`
  - `"abs_price"`: 颜色绝对价，如 `Silver ¥230,500`
- **流程**:

```mermaid
flowchart TD
    A[输入 remark_for_llm] --> B{文本是否为空?}
    B -->|是| C["返回 [], [], []"]
    B -->|否| D["调用 lx.extract()"]

    D --> D1["配置参数:\nmodel_id = gemma3:1b\nmodel_url = localhost:11434\ntemperature = 0.0\nprompt = _LX_PROMPT\nexamples = 4个 few-shot 示例\nmax_char_buffer = 2000"]

    D1 --> E{调用成功?}
    E -->|异常| F["打印警告\n返回 [], [], []"]
    E -->|成功| G[遍历 extractions]

    G --> H[获取 cls_raw / txt / attrs]
    H --> I["effective_class 判定"]

    I --> I1{txt 中有\n+/- 号+数字?}
    I1 -->|是| I2["effective_cls = delta"]
    I1 -->|否| I3{txt 中有\n¥/￥/円?}
    I3 -->|是| I4["effective_cls = abs_price"]
    I3 -->|否| I5["effective_cls = cls_raw 或 delta"]

    I2 --> J[解析 label]
    I4 --> J
    I5 --> J

    J --> K{label 非空?}
    K -->|否| G
    K -->|是| L{effective_cls\n类型?}

    L -->|abs_price| M["从 attrs 中提取 price_yen\n兜底取 delta_yen 或 txt\n_norm_amount_to_int 解析"]
    L -->|delta| N["从 attrs 中提取 delta_yen\n兜底从 txt 正则解析符号+金额"]

    M --> O["添加到 abs_list"]
    N --> P["添加到 delta_list"]
    O --> G
    P --> G

    G -->|遍历完| Q[同一 label 去重\n后者覆盖前者]
    Q --> R["返回 abs_list, delta_list, llm_dbg"]
```

#### `_fallback_parse_rules(text: str) -> Tuple[abs_list, delta_list]`
- **作用**: LLM 提取为空时的正则回退解析器
- **流程**:

```mermaid
flowchart TD
    A[输入 text] --> B[按换行拆分为多行]
    B --> C[逐行处理]
    C --> D{行中包含 全色?}
    D -->|是| E["解析全色差额\n添加到 delta_list"]
    D -->|否| F["_FALLBACK_ABS_RE 匹配\n颜色标签 + 金额"]
    F --> G["_FALLBACK_DELTA_RE 匹配\n颜色标签 + 正负号 + 金额"]
    G --> H[按分隔符拆分标签\n逐个添加到对应列表]
    E --> C
    H --> C
    C -->|遍历完| I["返回 abs_list, delta_list"]
```

#### `_label_matches_color_unified(label_raw, color_raw, color_norm) -> bool`
- **作用**: 判断提取到的颜色标签是否匹配 info 表中的某个颜色
- **匹配策略** (多级宽松匹配):

```mermaid
flowchart TD
    A["输入: label_raw, color_raw, color_norm"] --> B{EN_TO_JP 英文匹配?\nlabel小写 在 EN_TO_JP 中}
    B -->|是| B2{对应日文同义词\n出现在 color_raw 中?}
    B2 -->|是| Z[返回 True]
    B2 -->|否| C
    B -->|否| C{精确匹配?\n_norm label == color_norm}
    C -->|是| Z
    C -->|否| D{子串匹配?\nlabel_raw in color_raw\n或互相包含}
    D -->|是| Z
    D -->|否| E{去空格子串匹配?\nln_short in cn_short\n或 cn_short in ln_short}
    E -->|是| Z
    E -->|否| Y[返回 False]
```

#### `_normalize_remark_for_llm(remark_raw: str) -> str`
- **作用**: 去除備考1中与"開封"相关的行，只保留新品价规则文本供 LLM 处理
- **处理**: 在開封关键词前强行插入换行 → 按行过滤掉包含開封的行 → 合并剩余行

#### `_norm_amount_to_int(s: str) -> Optional[int]`
- **作用**: 将包含全角数字、货币符号的金额字符串统一转为整数
- **处理**: 全角→半角 / 去除 ¥/￥ 符号 / 去除逗号 / 提取数字

#### `_load_iphone17_info_df_from_db() -> pd.DataFrame`
- **作用**: 加载 iphone17_info 参考表 (CSV 或 Excel)
- **数据源**: Django settings 路径 > 环境变量 `IPHONE17_INFO_CSV` > 默认推断路径

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: モデルナンバー, 備考1, 買取価格, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color"]
    end

    subgraph 中间数据结构
        CMAP["cmap_all 字典\n(model_norm, cap_gb) -> {\n  color_norm: (part_number, color_raw)\n}"]
        ABSLIST["abs_list\n[(label_raw, absolute_price_yen), ...]\n如: [('Silver', 230500)]"]
        DELTALIST["delta_list\n[(label_raw, delta_yen), ...]\n如: [('orange', -1000)]"]
        CABSMAP["color_abs_map\n{color_norm: absolute_price}\n如: {'シルバー': 230500}"]
        CDELTAMAP["color_delta_map\n{color_norm: delta / 'ALL': delta}\n如: {'オレンジ': -1000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> CMAP
    RAW -->|"逐行读取"| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["モデルナンバー -> model_norm + cap_gb"]
        P2["買取価格 -> base_price (int)"]
        P3["備考1 -> _normalize_remark_for_llm -> remark_for_llm"]
        P4["remark_for_llm -> LLM提取 -> abs_list + delta_list"]
        P5["LLM为空? -> _fallback_parse_rules"]
        P6["label + color_map -> color_abs_map / color_delta_map"]
        P7["base_price + delta 或 abs_price -> price_new"]
    end

    CMAP --> PROC
    PROC --> ABSLIST
    PROC --> DELTALIST
    ABSLIST --> CABSMAP
    DELTALIST --> CDELTAMAP
    CABSMAP --> OUT
    CDELTAMAP --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  モデルナンバー = "iPhone17 Pro Max 256GB SIMフリー"
  備考1          = "orange-1000円※開封品は-5000円"
  買取価格       = "180,000"
  time-scraped   = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 基准价格"]
        T1["'180,000'"]
        T1 -->|to_int_yen| T2["180000"]
    end

    subgraph Step2["Step 2: 型号解析"]
        T3["'iPhone17 Pro Max 256GB SIMフリー'"]
        T3 -->|_normalize_model_generic| T4["'iPhone 17 Pro Max'"]
        T3 -->|_parse_capacity_gb| T5["256"]
    end

    subgraph Step3["Step 3: 查询 cmap_all"]
        T6["key = ('iPhone 17 Pro Max', 256)"]
        T6 -->|查 cmap_all| T7["{\n  'ブラックチタニウム': ('MYW23J/A', 'ブラックチタニウム'),\n  'ホワイトチタニウム': ('MYW53J/A', 'ホワイトチタニウム'),\n  ...\n}"]
    end

    subgraph Step4["Step 4: 備考1 预处理"]
        T8["'orange-1000円※開封品は-5000円'"]
        T8 -->|_normalize_remark_for_llm| T9["'orange-1000円'"]
        T8a["在※開封品前插入换行\n-> 'orange-1000円\\n※開封品は-5000円'\n-> 过滤掉開封行"]
    end

    subgraph Step5["Step 5: LLM 规则提取"]
        T10["'orange-1000円'"]
        T10 -->|_parse_rules_with_langextract| T11["abs_list = []\ndelta_list = [('orange', -1000)]\nllm_dbg = [('delta', 'orange-1000円', ...)]"]
        T10a["effective_class 判定:\ntxt='orange-1000円'\nhas_sign=True -> effective_cls='delta'"]
    end

    subgraph Step6["Step 6: 颜色匹配"]
        T12["label='orange' 匹配 color_map"]
        T12 --> T13["EN_TO_JP['orange'] = ['オレンジ', '橙']\n在 color_raw 中查找匹配"]
        T13 --> T14["若匹配到 -> color_delta_map['オレンジ'] = -1000\n若未匹配到 -> 跳过"]
    end

    subgraph Step7["Step 7: 价格计算 + 输出"]
        T15["对 color_map 中每个颜色:"]
        T15 --> T16["オレンジ: delta=-1000 -> price=180000+(-1000)=179000"]
        T15 --> T17["其他颜色: delta=0 -> price=180000"]
        T15 --> T18["输出行:\n{\n  part_number: 'MYWXXX',\n  shop_name: 'トゥインクル',\n  price_new: 179000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
    Step6 --> Step7
```

### 3.3 LLM 提取 + 正则回退策略

```mermaid
flowchart TD
    INPUT["備考1 原始文本"]

    INPUT --> PREPROC["预处理: _normalize_remark_for_llm\n去除開封相关行"]
    PREPROC --> LLM["LLM 解析器\n_parse_rules_with_langextract\n(LangExtract + Ollama gemma3:1b)\n带 @lru_cache(8192) 缓存"]

    LLM --> CHECK{abs_list 和\ndelta_list\n都为空?}
    CHECK -->|否| USE_LLM["使用 LLM 结果"]
    CHECK -->|是且有备注| REGEX["正则回退解析器\n_fallback_parse_rules"]
    CHECK -->|是且无备注| EMPTY["无规则\n所有颜色使用 base_price"]
    REGEX --> USE_REGEX["使用正则结果"]

    subgraph LLM解析器详细
        L1["prompt: 備考1 颜色价格规则解析专用提示词"]
        L2["examples: 4个 few-shot 示例\n- delta: orange-1000円\n- delta: Orange-2000円\n- abs_price: Silver ¥230,500 / Blue ¥229,000\n- delta: Blue-4000円 / Black-4000円"]
        L3["model: gemma3:1b (本地 Ollama)"]
        L4["temperature: 0.0 (确定性输出)"]
        L5["extraction_class:\n  delta -> attributes={color_label, delta_yen}\n  abs_price -> attributes={color_label, price_yen}"]
    end

    subgraph effective_class判定逻辑
        EC1["从 extraction_text 判断:"]
        EC2["has_sign: +/-号+数字 -> delta"]
        EC3["has_currency: ¥/￥/円 -> abs_price"]
        EC4["兜底: 用 LLM 原始 class"]
        EC1 --> EC2
        EC2 --> EC3
        EC3 --> EC4
    end

    subgraph 正则回退解析器详细
        R1["_FALLBACK_ABS_RE: 匹配 颜色标签 + 金额"]
        R2["_FALLBACK_DELTA_RE: 匹配 颜色标签 + ±金额"]
        R3["全色 特殊处理"]
    end
```

### 3.4 价格输出优先级策略

```mermaid
flowchart TD
    INPUT["color_abs_map + color_delta_map\n+ base_price"]

    INPUT --> P1{color_delta_map\n中有 ALL?}
    P1 -->|是| P1A["所有颜色统一价\nprice = base_price + ALL_delta\n对 color_map 每个颜色输出同一价格"]
    P1 -->|否| P2{color_abs_map\n非空?}
    P2 -->|是| P2A["绝对价覆盖模式:\n匹配色 -> 使用绝对价\n未匹配色 -> 使用 base_price"]
    P2 -->|否| P3["普通差额模式:\n对每个颜色查 color_delta_map\nprice = base_price + delta\n未匹配色 delta=0"]

    P1A --> OUT["输出行"]
    P2A --> OUT
    P3 --> OUT
```

### 3.5 颜色标签匹配机制 (EN_TO_JP + 多级匹配)

```mermaid
flowchart LR
    subgraph EN_TO_JP
        SILVER["silver -> シルバー / 銀"]
        BLUE["blue -> ブルー / 青 / ディープブルー"]
        BLACK["black -> ブラック / 黒"]
        WHITE["white -> ホワイト / 白"]
        GOLD["gold -> ゴールド / 金"]
        GREEN["green -> グリーン / 緑"]
        RED["red -> レッド / 赤"]
        PINK["pink -> ピンク"]
        PURPLE["purple -> パープル / 紫"]
        YELLOW["yellow -> イエロー / 黄"]
        ORANGE["orange -> オレンジ / 橙"]
        GRAY["gray -> グレー / グレイ / 灰"]
        NATURAL["natural -> ナチュラル"]
    end

    LABEL["LLM/正则提取的 label\n如: 'orange'"]
    COLOR["info表中的 color_raw\n如: 'オレンジ'"]

    LABEL -->|"label.lower() 查 EN_TO_JP"| ORANGE
    ORANGE -->|"同义词 'オレンジ' in color_raw 'オレンジ'"| MATCH["匹配成功!"]
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `EXTRACTION_MODE` | `"regex"` | regex / llm / auto（cleaner_tools） |
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID（cleaner_tools） |
| `OLLAMA_URL` / `OLLAMA_HOST` | `"http://localhost:11434"` | Ollama 服务地址（cleaner_tools） |
| `SHOP12_DEBUG` | `"1"` (启用) | 是否启用 debug 打印 |
| `SHOP12_LLM_TEMPERATURE` | `"0.0"` | LLM 推理温度 (确定性输出) |
| `SHOP12_LLM_TIMEOUT` | `"120"` | LLM 请求超时秒数 |
| `SHOP12_LLM_NUM_CTX` | `"4096"` | LLM 上下文窗口大小 |
| `IPHONE17_INFO_CSV` | 自动推断路径 | iphone17_info 参考文件路径 |
| `EXTERNAL_IPHONE17_INFO_PATH` | (Django settings) | Django 环境下的参考文件路径 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_FALLBACK_ABS_RE` | `(?P<labels>[^\d¥￥円:：/、，,;；※]+?)\s*(?:[:：]?\s*)?(?:¥\|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?` | 回退: 匹配颜色标签+绝对金额 | `Silver ¥230,500`, `シルバー230500円` |
| `_FALLBACK_DELTA_RE` | `(?P<labels>[^+\-−－\d¥￥円/、，,;；※]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?` | 回退: 匹配颜色标签±差额 | `orange-1000円`, `Blue+2000円` |
| `_SPLIT_SEPS` | `[／/、，,・\s]+` | 分隔多个颜色标签 | `Silver／Blue`, `シルバー、ブルー` |
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max`, `iPhone16Plus` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
| 開封分割正则 | `(※\s*開封品\|※\s*開封\|開封品\|開封済\|開封)` | 在開封关键词前插入换行 | `※開封品は-5000円` |
| effective_class: has_sign | `[+\-−－]\s*[０-９0-9]` | 判断文本中是否有正负号+数字 → delta | `orange-1000円` |
| effective_class: has_currency | `[¥￥円]` | 判断文本中是否有日元符号 → abs_price | `Silver ¥230,500` |
| 全色解析 | `全色\s*[：:\-]?\s*([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)?` | 回退: 解析全色差额 | `全色-2000円`, `全色：+1000` |
