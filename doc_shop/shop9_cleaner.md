# Shop9 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop9_cleaner.py`
> 店铺名称: アキモバ

---

## 一、总流程图

整个 shop9 清洗器的核心入口是 `clean_shop9(df, debug, debug_limit)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。与 shop17 不同，shop9 采用 **LLM 优先、正则兜底** 策略，且区分绝对价格 (abs) 和差额 (delta) 两种提取结果。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\n機種名 / 買取価格 / 色・詳細等 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_for_shop2]
    C --> D[构建 pn_map\nmodel_norm + cap_gb → color_to_pn]
    D --> E[逐行遍历 DataFrame]

    E --> F{型号/容量\n能否解析?}
    F -->|否| E
    F -->|是| G[在 pn_map 中\n查找该机型]

    G --> H{pn_map\n是否存在?}
    H -->|否| E
    H -->|是| I[解析基准价格\nto_int_yen: 買取価格 or 色・詳細等]

    I --> J["拼接输入文本\n買取価格: {price}\n色・詳細等: {detail}"]
    J --> K{USE_LLM\n启用?}

    K -->|是| L["LLM 抽取\n_llm_extract_rules_cached\n返回 (abs_map, delta_map)"]
    K -->|否| M["正则回退抽取\n_extract_abs_prices_regex\n_extract_deltas_regex"]

    L --> N{abs_map 和\ndelta_map 都为空?}
    N -->|是| M
    N -->|否| O[正则后修正\n_direct_abs_overrides_for_row]

    M --> O

    O --> P[输出优先级决策]
    P --> Q[生成输出行\npart_number / shop_name / price_new / recorded_at]
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
    clean["clean_shop9(df, debug, debug_limit)"]

    clean --> load["_load_iphone17_info_df_for_shop2()"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> toint["to_int_yen(val)"]
    clean --> parsedt["parse_dt_aware(val)"]
    clean --> llmcache["_llm_extract_rules_cached(price, detail, avail_colors)"]
    clean --> regexabs["_extract_abs_prices_regex(text)"]
    clean --> regexdelta["_extract_deltas_regex(text)"]
    clean --> directabs["_direct_abs_overrides_for_row(text, color_to_pn, synonyms)"]
    clean --> mapcolor["_map_to_available_color(raw, avail_set)"]

    llmcache --> buildaliases["_build_color_aliases(available_colors)"]
    llmcache --> examples["_shop9_lx_examples()"]
    llmcache --> lxextract["langextract.extract()"]
    llmcache --> bucket["_bucket_amount(cls_norm, ex_text, amt)"]
    llmcache --> mapcolor
    llmcache --> coerce["_coerce_signed_int(x)"]
    llmcache --> normcls["_norm_cls(x)"]

    directabs --> extractafter["_extract_amount_after_alias(text, alias)"]
    directabs --> normamount["_norm_amount_to_int(x)"]

    buildaliases --> synonyms["SYNONYM_LOOKUP 字典查表"]
    mapcolor --> synonyms
    mapcolor --> norm["_norm(s)"]

    regexabs --> normamount
    regexdelta --> normamount
```

### 2.2 核心函数详细说明

#### `clean_shop9(df: pd.DataFrame, debug: bool, debug_limit: int) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `機種名`, `買取価格`, `色・詳細等`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`("アキモバ"), `price_new`, `recorded_at` 列的 DataFrame
- **debug 模式**: 通过 COLOR_PAT / DISCOUNT_PAT / ABS_PRICE_PAT 三个正则筛选"疑似有颜色价格信息"的行，最多打印 debug_limit 行

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_llm_extract_rules_cached(price_text, detail_text, avail_colors_key) -> Tuple[Dict, Dict]`
- **作用**: LLM 核心抽取函数，带 `@lru_cache(maxsize=4096)` 缓存
- **返回**: `(abs_map, delta_map)` 双映射字典
- **流程**:

```mermaid
flowchart TD
    A["_llm_extract_rules_cached(price_text, detail_text, avail_colors_key)"] --> B["拼接输入文本\n'買取価格: {price}\n色・詳細等: {detail}'"]
    B --> C["构建动态 prompt\n包含 AVAILABLE_COLORS 和 COLOR_ALIASES"]
    C --> D["调用 lx.extract()\n5 个 few-shot 示例"]

    D --> D1["配置参数:\nmodel_id = gemma3:1b\nmodel_url = localhost:11434\ntemperature = 0.0\nfence_output = False\nuse_schema_constraints = False"]

    D1 --> E{调用成功?}
    E -->|异常| F["返回 ({}, {})"]
    E -->|成功| G[遍历 extractions]

    G --> H[提取 extraction_class / attributes / extraction_text]
    H --> I["_norm_cls 标准化类名"]
    I --> J["_coerce_signed_int 解析 amount_yen"]
    J --> K{amount 有效?}
    K -->|否| G
    K -->|是| L["_bucket_amount 分类\n→ 'abs' 或 'delta'"]

    L --> M[遍历 colors 列表]
    M --> N["_map_to_available_color\n映射到可用颜色"]
    N --> O{映射成功?}
    O -->|否| M
    O -->|是| P{bucket == abs?}
    P -->|是| Q["abs_map[color] = amount"]
    P -->|否| R["delta_map[color] = amount"]
    Q --> M
    R --> M

    M -->|遍历完| S["返回 (abs_map, delta_map)"]
```

#### `_bucket_amount(cls_norm: str, ex_text: str, amt: int) -> str`
- **作用**: 将 LLM 提取结果分类为 "abs"（绝对价）或 "delta"（差额）
- **判定逻辑**:

```mermaid
flowchart TD
    A["_bucket_amount(cls_norm, ex_text, amt)"] --> B{amt 为 None?}
    B -->|是| Z["返回 'delta'"]
    B -->|否| C{amt < 0?}
    C -->|是| Z
    C -->|否| D{DELTA_HINT_RE\n匹配 ex_text?}
    D -->|是| Z
    D -->|否| E{"|amt| >= ABS_LIKE_MIN\n(默认 50000)?"}
    E -->|是| Y["返回 'abs'"]
    E -->|否| F{cls_norm 属于\nabs 类名集?}
    F -->|是| Y
    F -->|否| G{cls_norm 属于\ndelta 类名集?}
    G -->|是| Z
    G -->|否| Z
```

#### `_map_to_available_color(raw_color: str, available_set: set) -> Optional[str]`
- **作用**: 将 LLM/正则提取到的颜色标签映射到 pn_map 中的可用颜色
- **匹配策略** (四级宽松匹配):

```mermaid
flowchart TD
    A["输入: raw_color, available_set"] --> A0{ALL / 全色?}
    A0 -->|是| Z0["返回 'ALL'"]
    A0 -->|否| B{精确匹配?\nraw_color in available_set}
    B -->|是| Z[返回匹配颜色]
    B -->|否| C{归一等价匹配?\n_norm 比较}
    C -->|是| Z
    C -->|否| D[查 SYNONYM_LOOKUP\n颜色同义词表]
    D --> E{同义词中任一\n命中 available_set?}
    E -->|是| Z
    E -->|否| F{子串包含匹配?\nrcn in cn or cn in rcn}
    F -->|是| Z
    F -->|否| Y[返回 None]
```

#### `_direct_abs_overrides_for_row(raw_color_text, color_to_pn, synonym_lookup) -> Dict[str, int]`
- **作用**: 正则后修正层，直接扫描原始文本中"颜色别名 + 紧随数字"模式，覆盖 abs_map
- **流程**:

```mermaid
flowchart TD
    A["输入: raw_color_text, color_to_pn, synonym_lookup"] --> B[遍历 color_to_pn 中每个颜色]
    B --> C["构建别名集合\n自身 + SYNONYM_LOOKUP 中的同义词"]
    C --> D[对每个别名调用\n_extract_amount_after_alias]
    D --> E{找到金额且\n>= ABS_MIN_YEN?}
    E -->|是| F["overrides[color_norm] = amount"]
    E -->|否| G[跳过]
    F --> B
    G --> B
    B -->|遍历完| H["返回 overrides\n覆盖写入 abs_map"]
```

#### `_shop9_lx_examples() -> List[ExampleData]`
- **作用**: 返回 5 个 few-shot 示例，带 `@lru_cache(maxsize=1)` 缓存
- **示例场景覆盖**:
  1. 多颜色共享一个绝对价 + 另一颜色不同绝对价
  2. 每颜色独立 delta (正/负)
  3. 全色 delta ("全色-500円")
  4. 纯绝对价无基准价 ("買取価格: -")
  5. 紧凑格式多颜色多价格 ("橙,銀230,500/青229,000")

#### `_build_color_aliases(available_colors) -> Dict[str, List[str]]`
- **作用**: 为当前行的可用颜色列表构建别名映射，注入到 LLM prompt 中
- **数据源**: SYNONYM_LOOKUP (由 FAMILY_SYNONYMS_SHOP9 展开)

#### `_coerce_signed_int(x) -> Optional[int]`
- **作用**: 将各种格式的数字文本转为带符号整数
- **处理**: 全角→半角 / 千分位忽略 / 正负号识别 / 容错终止

#### `_extract_abs_prices_regex(text) -> List[Tuple[str, int]]`
- **作用**: 正则回退版绝对价提取
- **正则**: ABS_PRICE_RE 匹配 "标签 + 金额" 模式

#### `_extract_deltas_regex(text) -> List[Tuple[str, int]]`
- **作用**: 正则回退版差额提取
- **正则**: DELTA_RE 匹配 "标签 +/- 金额" 模式
- **特殊**: "全色" 关键词触发 ALL delta 回退

#### `_norm(s: str) -> str` (内部版)
- **作用**: 局部文本归一化 (全角→半角数字、空白统一、小写化)

#### `_norm_cls(x: str) -> str`
- **作用**: 标准化 extraction_class 名称 (小写 + 分隔符统一为下划线)

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: 機種名, 買取価格, 色・詳細等, time-scraped"]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        PNMAP["pn_map 字典\n(model_norm, cap_gb) → {\n  color_norm: part_number\n}"]
        SYNMAP["SYNONYM_LOOKUP 字典\n颜色 → [同义词列表]\n由 FAMILY_SYNONYMS_SHOP9 展开"]
        ABSMAP["abs_map 字典\n{color_norm or 'ALL': amount_yen}\n绝对价映射"]
        DELTAMAP["delta_map 字典\n{color_norm or 'ALL': signed_delta}\n差额映射"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> PNMAP
    RAW --> |"逐行读取"| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["機種名 → model_norm + cap_gb"]
        P2["買取価格 → base_price (int)"]
        P3["拼接 price + detail → LLM 输入"]
        P4["LLM 抽取 → abs_map + delta_map"]
        P5["正则回退 (若 LLM 无结果)"]
        P6["_direct_abs_overrides 后修正"]
        P7["优先级决策 → price_new"]
    end

    PNMAP --> PROC
    SYNMAP --> PROC
    PROC --> ABSMAP
    PROC --> DELTAMAP
    ABSMAP --> OUT
    DELTAMAP --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  機種名       = "iPhone17 Pro Max 256GB SIMフリー"
  買取価格     = "195,500円"
  色・詳細等   = "未開 橙194,500/青,銀195,500"
  time-scraped = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 型号解析"]
        T1["'iPhone17 Pro Max 256GB SIMフリー'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step2["Step 2: 查询 pn_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 pn_map| T5["{\n  'ブラック': 'MYW23J/A',\n  'ホワイト': 'MYW53J/A',\n  'ディープブルー': 'MYW83J/A',\n  'コズミックオレンジ': 'MYWF3J/A',\n  'シルバー': 'MYWG3J/A'\n}"]
    end

    subgraph Step3["Step 3: 基准价格"]
        T6["'195,500円'"]
        T6 -->|to_int_yen| T7["195500"]
    end

    subgraph Step4["Step 4: LLM 抽取"]
        T8["拼接: '買取価格: 195,500円\n色・詳細等: 未開 橙194,500/青,銀195,500'"]
        T8 -->|"_llm_extract_rules_cached\n+ 5 few-shot examples"| T9["extractions:\n1) abs_price: colors=['橙'] amount=194500\n2) abs_price: colors=['青','銀'] amount=195500"]
        T9 -->|"_bucket_amount\n194500 >= 50000 → abs\n195500 >= 50000 → abs"| T10["abs_map = {\n  'コズミックオレンジ': 194500,\n  'ディープブルー': 195500,\n  'シルバー': 195500\n}\ndelta_map = {}"]
    end

    subgraph Step5["Step 5: 正则后修正"]
        T11["_direct_abs_overrides_for_row\n扫描原文: '橙194,500' → 194500\n'青...195,500' → 跳过(已有)\n'銀195,500' → 195500"]
        T11 --> T12["abs_map 不变\n(已被 LLM 正确覆盖)"]
    end

    subgraph Step6["Step 6: 输出优先级决策"]
        T13["abs_map 非空 → 走 per-color abs 分支"]
        T13 --> T14["コズミックオレンジ → abs 194500"]
        T13 --> T15["ディープブルー → abs 195500"]
        T13 --> T16["シルバー → abs 195500"]
        T13 --> T17["ブラック → 无 abs → base_price 195500"]
        T13 --> T18["ホワイト → 无 abs → base_price 195500"]
    end

    subgraph Step7["Step 7: 输出行"]
        T19["{\n  part_number: 'MYWF3J/A', shop_name: 'アキモバ',\n  price_new: 194500, recorded_at: ...\n},\n{\n  part_number: 'MYW83J/A', shop_name: 'アキモバ',\n  price_new: 195500, recorded_at: ...\n},\n{\n  part_number: 'MYWG3J/A', shop_name: 'アキモバ',\n  price_new: 195500, recorded_at: ...\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
    Step6 --> Step7
```

### 3.3 LLM 优先 + 正则兜底 + 后修正 策略

```mermaid
flowchart TD
    INPUT["買取価格 + 色・詳細等 原始文本"]

    INPUT --> LLM["LLM 解析器\n_llm_extract_rules_cached\n(LangExtract + Ollama)"]
    LLM --> CHECK{abs_map 和\ndelta_map 都为空?}
    CHECK -->|否| OVERRIDE["正则后修正\n_direct_abs_overrides_for_row"]
    CHECK -->|是| REGEX["正则回退解析器\n_extract_abs_prices_regex\n_extract_deltas_regex"]
    REGEX --> OVERRIDE

    OVERRIDE --> DECISION["输出优先级决策"]

    subgraph LLM解析器详细
        L1["prompt: 动态生成\n包含 AVAILABLE_COLORS / COLOR_ALIASES"]
        L2["examples: 5个 few-shot 示例"]
        L3["model: gemma3:1b (本地 Ollama)"]
        L4["temperature: 0.0 (确定性输出)"]
        L5["缓存: @lru_cache maxsize=4096"]
        L6["输出分类: _bucket_amount\nabs_price → abs_map\ndelta → delta_map"]
    end

    subgraph 正则回退详细
        R1["ABS_PRICE_RE: 标签 + 金额"]
        R2["DELTA_RE: 标签 +/- 金额"]
        R3["SPLIT_SEPS: 分隔拆分"]
        R4["_match_label_to_colnorm: 宽松匹配"]
    end

    subgraph 后修正详细
        O1["_direct_abs_overrides_for_row"]
        O2["逐颜色扫描原文"]
        O3["alias + 紧随数字 → abs 覆盖"]
        O4["仅接受 >= ABS_MIN_YEN"]
    end
```

### 3.4 输出优先级决策

```mermaid
flowchart TD
    START["abs_map + delta_map 就绪"] --> A{"delta_map 含 'ALL'?"}
    A -->|是| A1["所有颜色: price = base_price + delta_ALL"]
    A -->|否| B{"abs_map 含 'ALL'?"}
    B -->|是| B1["所有颜色: price = abs_ALL"]
    B -->|否| C{abs_map 非空?}
    C -->|是| C1["有 abs 的颜色: price = abs_map[color]\n无 abs 的颜色: price = base_price (回退)"]
    C -->|否| D{base_price 有效?}
    D -->|否| D1["跳过该行"]
    D -->|是| E["所有颜色: price = base_price + delta_map.get(color, 0)"]

    style A1 fill:#e6ffe6
    style B1 fill:#e6ffe6
    style C1 fill:#e6ffe6
    style E fill:#e6ffe6
    style D1 fill:#ffe6e6
```

### 3.5 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY_SYNONYMS_SHOP9
        BLUE["blue 家族\nブルー / 青 / ディープブルー / ディープ ブルー"]
        SILVER["silver 家族\nシルバー / 銀"]
        BLACK["black 家族\nブラック / 黒"]
        ORANGE["orange 家族\nオレンジ / 橙 / コズミックオレンジ"]
        WHITE["white 家族\nホワイト / 白"]
    end

    LABEL["LLM 提取到的 label\n如: '橙'"]
    COLOR["pn_map 中的 color\n如: 'コズミックオレンジ'"]

    LABEL -->|"查 SYNONYM_LOOKUP"| ORANGE
    ORANGE -->|"同义词 '橙' → 'コズミックオレンジ'"| MATCH["匹配成功!"]

    subgraph SYNONYM_LOOKUP展开规则
        S1["FAMILY_SYNONYMS_SHOP9 的每个 key→values"]
        S2["双向扩展: key 自身也加入 values 的同义词"]
        S3["去重保序: dict.fromkeys"]
    end
```

---

## 四、配置项说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SHOP9_USE_LLM` | `"1"` (启用) | 是否启用 LLM 抽取；设为 `"0"` / `"false"` / `"no"` 则关闭 |
| `SHOP9_OLLAMA_HOST` | `"http://localhost:11434"` | Ollama 服务地址 (回退到 `OLLAMA_HOST`) |
| `SHOP9_LX_MODEL_ID` | `"gemma3:1b"` | LangExtract 使用的 Ollama 模型 ID (回退到 `SHOP9_LLM_MODEL_ID`) |
| `SHOP9_LLM_TEMPERATURE` | `"0.0"` | LLM 温度参数，0.0 为确定性输出 |
| `SHOP9_ABS_LIKE_MIN` | `"50000"` | 绝对价量级阈值：金额 >= 此值且无 delta 线索则分类为 abs |
| `SHOP9_ALLOW_REGEX_FALLBACK` | `"1"` (启用) | LLM 无结果时是否允许正则回退；设为 `"0"` / `"false"` 则禁用 |
| `IPHONE17_INFO_CSV` | 自动推断路径 | iphone17_info 文件路径 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_NUM_MODEL_PAT` | `(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配数字代号机型 | `iPhone 17 Pro Max`, `iPhone17ProMax` |
| `_AIR_PAT` | `(iPhone)\s*(Air)(?:\s*(Pro\s*Max\|Pro\|Plus\|mini))?` | 匹配 iPhone Air | `iPhone Air` |
| `DELTA_HINT_RE` | `(?:[+\-−－]\|値下げ\|値引\|割引\|円引\|OFF\|オフ\|減額)` | 判断文本是否含 delta 线索 | `値下げ`, `-2000`, `OFF` |
| `ABS_PRICE_RE` | `(?P<labels>[^0-9...]+?)\s*(?:¥\|￥)?\s*(?P<amount>[0-9...][0-9,]*)\s*(?:円)?` | 正则回退：匹配"标签 + 绝对金额" | `青 229,000円`, `ブルー：229000` |
| `DELTA_RE` | `(?P<labels>...)\s*[：:\-]?\s*(?P<sign>[+\-−－])\s*(?:¥\|￥)?\s*(?P<amount>...)\s*(?:円)?` | 正则回退：匹配"标签 +/- 差额" | `ブラック -2,000円`, `シルバー:+1000` |
| `SPLIT_SEPS` | `[/／、，,;；\s]+` | 拆分多个颜色条目 | `橙194,500/青195,500` |
| `_extract_amount_after_alias` (内部) | `{alias}\s*(?:¥\|￥)?\s*([0-9][0-9,]*)` | 后修正：在别名后提取紧随的绝对价 | `橙194,500`, `シルバー 195500` |
| `COLOR_PAT` (debug) | `(ブラック\|ホワイト\|ブルー\|...\|Titanium)` | Debug 模式：筛选含颜色关键词的行 | `ブラック`, `Silver` |
| `DISCOUNT_PAT` (debug) | `(値下げ\|値引\|割引\|円引\|OFF\|オフ\|[+\-]\s*[0-9])` | Debug 模式：筛选含折扣关键词的行 | `値下げ`, `-500` |
| `ABS_PRICE_PAT` (debug) | `(?:¥\|￥)?\s*[0-9]{2,3}(?:[0-9,]{3,})\s*(?:円)?` | Debug 模式：筛选含绝对价格式的行 | `¥195,500円`, `230000` |
