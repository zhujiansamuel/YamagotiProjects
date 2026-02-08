# Shop17 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop17_cleaner.py`
> 店铺名称: ゲストモバイル (Guest Mobile)

---

## 一、总流程图

整个 shop17 清洗器的核心入口是 `clean_shop17(df)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\ntype / 新未開封品 / 色減額 / time-scraped]
    B -->|缺列| B1[抛出 ValueError]
    B -->|通过| C[加载 iphone17_info 参考表\n_load_iphone17_info_df_for_shop2]
    C --> D[构建颜色映射表\n_build_color_map_shop17]
    D --> E[逐行遍历 DataFrame]

    E --> F{type 列是否为空?}
    F -->|空| E
    F -->|非空| G[型号标准化\n_normalize_model_generic]

    G --> H{型号/容量\n能否解析?}
    H -->|否| E
    H -->|是| I[在 color_map 中\n查找该机型]

    I --> J{color_map\n是否存在?}
    J -->|否| E
    J -->|是| K[解析基准价格\nto_int_yen 新未開封品]

    K --> L{基准价格\n是否有效?}
    L -->|否| E
    L -->|是| M[提取颜色差额\n_extract_color_deltas_shop17]

    M --> N[颜色标签匹配\n_label_matches_color_shop17]
    N --> O[计算每个颜色的最终价格\nprice = base_price + delta]
    O --> P[生成输出行\npart_number / shop_name / price_new / recorded_at]
    P --> E

    E -->|遍历结束| Q[组装输出 DataFrame]
    Q --> R[去除空值 / 类型转换]
    R --> S[输出: 标准化 DataFrame\npart_number, shop_name, price_new, recorded_at]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop17(df)"]

    clean --> load["_load_iphone17_info_df_for_shop2()"]
    clean --> buildcm["_build_color_map_shop17(info_df)"]
    clean --> normmod["_normalize_model_generic(text)"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> toint["to_int_yen(val)"]
    clean --> extract["_extract_color_deltas_shop17(text)"]
    clean --> labelmatch["_label_matches_color_shop17(label, color_raw, color_norm)"]
    clean --> parsedt["parse_dt_aware(val)"]

    extract --> regex["_extract_color_deltas_shop17_regex(text)"]
    extract --> llm["_extract_color_deltas_shop17_llm(text)"]

    regex --> normcolor["_normalize_color_text_shop17(s)"]
    regex --> unopened["_pick_unopened_section(text)"]
    regex --> normlbl["_normalize_label_shop17(lbl)"]
    regex --> plausible["_is_plausible_color_label_shop17(label)"]

    llm --> normcolor
    llm --> unopened
    llm --> plausible
    llm --> examples["_get_color_delta_examples_shop17()"]
    llm --> parsedelta["_parse_delta_attr_to_int(val)"]
    llm --> lxextract["langextract.extract()"]

    buildcm --> normmod
    labelmatch --> familysyn["FAMILY_SYNONYMS_shop17 字典查表"]
```

### 2.2 核心函数详细说明

#### `clean_shop17(df: pd.DataFrame) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `type`, `新未開封品`, `色減額`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`, `price_new`, `recorded_at` 列的 DataFrame

#### `_normalize_model_generic(text: str) -> str`
- **作用**: 将各种型号写法统一为标准格式
- **处理**: 日文别名转英文 (プロ→Pro) / 紧凑写法展开 (17pro→17 Pro) / 去噪 (容量/SIM信息)
- **输出**: 如 `"iPhone 17 Pro Max"`, `"iPhone Air"`, `"iPhone 16 Plus"`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB→GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_extract_color_deltas_shop17(text: str) -> List[Tuple[str, int]]`
- **作用**: 提取颜色差额的调度函数，采用 **正则优先、LLM 兜底** 策略

```mermaid
flowchart TD
    A["_extract_color_deltas_shop17(text)"] --> B["_extract_color_deltas_shop17_regex(text)"]
    B --> C{正则结果\n是否为空?}
    C -->|非空| D[返回正则结果]
    C -->|空| E["_extract_color_deltas_shop17_llm(text)"]
    E --> F[返回 LLM 结果]
```

#### `_extract_color_deltas_shop17_regex(text: str) -> List[Tuple[str, int]]`
- **作用**: 正则版颜色差额提取
- **流程**:

```mermaid
flowchart TD
    A[输入 text] --> B[文本预处理\n_normalize_color_text_shop17\n_pick_unopened_section]
    B --> C{包含 色減額?}
    C -->|是| D["截取 '色減額' 之后的部分"]
    C -->|否| E[使用原文]
    D --> F{整段是否为\nなし/減額なし?}
    E --> F
    F -->|是| G["返回空列表 []"]
    F -->|否| H["按分隔符拆分\n／ / 、 ; \\n"]
    H --> I[逐段匹配]
    I --> J{匹配 COLOR_NONE_RE?\n如 シルバーなし}
    J -->|是| K["添加 (label, 0)"]
    J -->|否| L{匹配 COLOR_DELTA_RE?\n如 ブルー-1000}
    L -->|是| M["添加 (label, delta)"]
    L -->|否| N[跳过此段]
    K --> I
    M --> I
    N --> I
    I -->|遍历完| O["返回 [(label, delta), ...]"]
```

#### `_extract_color_deltas_shop17_llm(text: str) -> List[Tuple[str, int]]`
- **作用**: LLM 版颜色差额提取 (正则失败时的后备方案)
- **流程**:

```mermaid
flowchart TD
    A[输入 text] --> B{LangExtract 可用?\nUSE_LLM 开启?}
    B -->|否| C["返回空列表 []"]
    B -->|是| D[文本预处理\n_normalize_color_text_shop17\n_pick_unopened_section]
    D --> E{整段是否为\nなし/減額なし?}
    E -->|是| C
    E -->|否| F["调用 lx.extract()"]

    F --> F1["配置参数:\nmodel_id = gemma3:1b\nmodel_url = localhost:11434\ntemperature = 0.0\nprompt = COLOR_DELTA_PROMPT_SHOP17\nexamples = 3个 few-shot 示例"]

    F1 --> G{调用成功?}
    G -->|异常| H[打印错误\n返回空列表]
    G -->|成功| I[遍历 extractions]
    I --> J{extraction_class\n== color_delta?}
    J -->|否| I
    J -->|是| K[提取 color / delta\n从 attributes 字典]
    K --> L{color 是否\n合理颜色名?}
    L -->|否| I
    L -->|是| M{delta 能否\n解析为整数?}
    M -->|否| N[尝试从 extraction_text\n正则提取数字]
    M -->|是| O["添加到结果 (color, delta)"]
    N --> P{提取成功?}
    P -->|否| I
    P -->|是| O
    O --> I
    I -->|遍历完| Q["返回 [(color, delta), ...]"]
```

#### `_label_matches_color_shop17(label_raw, color_raw, color_norm) -> bool`
- **作用**: 判断提取到的颜色标签是否匹配 info 表中的某个颜色
- **匹配策略** (三级宽松匹配):

```mermaid
flowchart TD
    A["输入: label_raw, color_raw, color_norm"] --> B{精确匹配?\nlabel归一 == color_norm}
    B -->|是| Z[返回 True]
    B -->|否| C{子串匹配?\nlabel_raw in color_raw}
    C -->|是| Z
    C -->|否| D[查 FAMILY_SYNONYMS_shop17\n颜色家族同义词表]
    D --> E{label 在家族表中?}
    E -->|是| F[获取同义词列表]
    E -->|否| G[反向查: 遍历所有家族\n找包含 label 的条目]
    F --> H{同义词中任一\n出现在 color_raw 中?}
    G --> H
    H -->|是| Z
    H -->|否| Y[返回 False]
```

#### `_build_color_map_shop17(info_df) -> Dict`
- **作用**: 构建 `(model_norm, capacity_gb) -> {color_norm: (part_number, color_raw)}` 映射
- **数据源**: iphone17_info 参考表

#### `_normalize_color_text_shop17(s: str) -> str`
- **作用**: 统一色減額文本中的特殊字符
- **处理**: 全角数字→半角 / 全角逗号→半角 / 各种 dash 统一为 `-` / 全角斜杠→半角 / 多余空白清理

#### `_is_plausible_color_label_shop17(label: str) -> bool`
- **作用**: 过滤非颜色名标签
- **排除规则**: 以 △/▲ 开头 / 包含数字 / 长度 >16 / 包含关键词 (利用制限, 保証, 郵送 等)

#### `_pick_unopened_section(text: str) -> str`
- **作用**: 如果文本中包含 `【未開封】`，则只取该段内容

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: type, 新未開封品, 色減額, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 中间数据结构
        CMAP["color_map 字典\n(model_norm, cap_gb) → {\n  color_norm: (part_number, color_raw)\n}"]
        DELTAS["labels_and_deltas\n[(label_raw, delta_int), ...]\n如: [('シルバー', 0), ('ブルー', -1000)]"]
        CD["color_deltas 字典\n{color_norm: delta_int}\n如: {'シルバー': 0, 'ブルー': -1000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    INFO --> CMAP
    RAW --> |"逐行读取"| PROC

    subgraph PROC[逐行处理]
        direction TB
        P1["type → model_norm + cap_gb"]
        P2["新未開封品 → base_price (int)"]
        P3["色減額 → labels_and_deltas"]
        P4["labels + color_map → color_deltas"]
        P5["base_price + delta → price_new"]
    end

    CMAP --> PROC
    PROC --> DELTAS
    DELTAS --> CD
    CD --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  type         = "iPhone17 Pro Max 256GB SIMフリー"
  新未開封品    = "180,000"
  色減額       = "色減額:シルバーなし/ブルー-1000\n\n郵送は翌日着のみ保証\n\n△減額なし"
  time-scraped = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 型号解析"]
        T1["'iPhone17 Pro Max 256GB SIMフリー'"]
        T1 -->|_normalize_model_generic| T2["'iPhone 17 Pro Max'"]
        T1 -->|_parse_capacity_gb| T3["256"]
    end

    subgraph Step2["Step 2: 查询 color_map"]
        T4["key = ('iPhone 17 Pro Max', 256)"]
        T4 -->|查 cmap_all| T5["{\n  'ブラックチタニウム': ('MYW23J/A', 'ブラックチタニウム'),\n  'ホワイトチタニウム': ('MYW53J/A', 'ホワイトチタニウム'),\n  'ナチュラルチタニウム': ('MYW83J/A', 'ナチュラルチタニウム'),\n  'サンドチタニウム': ('MYWF3J/A', 'サンドチタニウム'),\n  ...\n}"]
    end

    subgraph Step3["Step 3: 基准价格"]
        T6["'180,000'"]
        T6 -->|to_int_yen| T7["180000"]
    end

    subgraph Step4["Step 4: 颜色差额提取"]
        T8["'色減額:シルバーなし/ブルー-1000\\n\\n郵送は翌日着のみ保証\\n\\n△減額なし'"]
        T8 -->|_normalize_color_text_shop17| T9["'色減額:シルバーなし/ブルー-1000 郵送は翌日着のみ保証 △減額なし'"]
        T9 -->|"截取 '色減額:' 后部分"| T10["'シルバーなし/ブルー-1000 ...'"]
        T10 -->|"按 / 拆分"| T11["['シルバーなし', 'ブルー-1000 ...']"]
        T11 -->|正则匹配| T12["[('シルバー', 0), ('ブルー', -1000)]"]
    end

    subgraph Step5["Step 5: 颜色匹配 + 价格计算"]
        T13["对 color_map 中每个颜色:"]
        T13 --> T14["シルバー → 匹配 label 'シルバー' → delta=0 → price=180000"]
        T13 --> T15["ブルー → 匹配 label 'ブルー' → delta=-1000 → price=179000"]
        T13 --> T16["其他颜色 → 未匹配 → delta=0 → price=180000"]
    end

    subgraph Step6["Step 6: 输出行"]
        T17["{\n  part_number: 'MYW53J/A',\n  shop_name: 'ゲストモバイル',\n  price_new: 180000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: 'MYWXXX',\n  shop_name: 'ゲストモバイル',\n  price_new: 179000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
```

### 3.3 颜色差额提取 - 正则 vs LLM 策略

```mermaid
flowchart TD
    INPUT["色減額 原始文本"]

    INPUT --> REGEX["正则解析器"]
    REGEX --> CHECK{结果非空?}
    CHECK -->|是| USE_REGEX["使用正则结果\n(快速 & 稳定)"]
    CHECK -->|否| LLM["LLM 解析器\n(LangExtract + Ollama gemma3:1b)"]
    LLM --> USE_LLM["使用 LLM 结果\n(处理复杂/非标准格式)"]

    subgraph 正则解析器详细
        R1["COLOR_NONE_RE: 匹配 '颜色名なし' → delta=0"]
        R2["COLOR_DELTA_RE: 匹配 '颜色名±金额' → delta=±N"]
    end

    subgraph LLM解析器详细
        L1["prompt: 买取表色減額解析专用提示词"]
        L2["examples: 3个 few-shot 示例"]
        L3["model: gemma3:1b (本地 Ollama)"]
        L4["temperature: 0.0 (确定性输出)"]
        L5["输出: extraction_class=color_delta\nattributes={color, delta, raw}"]
    end
```

### 3.4 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph FAMILY_SYNONYMS_shop17
        BLUE["blue 家族\nブルー / 青 / ミッドナイト / マリン / ミストブルー"]
        BLACK["black 家族\nブラック / 黒"]
        WHITE["white 家族\nホワイト / 白 / スターライト"]
        GOLD["gold 家族\nゴールド / 金 / ライトゴールド"]
        GREEN["green 家族\nグリーン / 緑 / セージ"]
        OTHER["... 其他家族 (silver/orange/pink/yellow/purple/natural/spaceblack)"]
    end

    LABEL["提取到的 label\n如: 'ブルー'"]
    COLOR["info表中的 color\n如: 'マリンブルー'"]

    LABEL -->|"查 FAMILY_SYNONYMS"| BLUE
    BLUE -->|"同义词 'ブルー' in 'マリンブルー'"| MATCH["匹配成功!"]
```

---

## 四、配置项说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SHOP17_USE_LLM` | `"1"` (启用) | 是否启用 LLM 兜底 |
| `SHOP17_LX_MODEL_ID` | `"gemma3:1b"` | Ollama 模型 ID |
| `SHOP17_LX_MODEL_URL` | `"http://localhost:11434"` | Ollama 服务地址 |
| `DEBUG_SHOP17_MAX_ROWS` | `20` | Debug 输出最大行数 |
| `DEBUG_SHOP17_SHOW_ALL_COLORS` | `""` (关闭) | 是否显示所有颜色的 debug 信息 |
| `IPHONE17_INFO_CSV` | 自动推断路径 | iphone17_info 文件路径 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `COLOR_NONE_RE_shop17` | `label...なし` | 匹配"颜色无减额" | `シルバーなし`, `クラウドホワイト：なし` |
| `COLOR_DELTA_RE_shop17` | `label[：:-]?[+-]?amount` | 匹配"颜色±金额" | `ブルー-1000`, `スカイブルー: -3,000` |
| `SPLIT_TOKENS_RE_shop17` | `[／/、;\n]` | 拆分多个颜色条目 | `シルバーなし／ブルー-1000` |
| `_NUM_MODEL_PAT` | `iPhone\s*\d{2}...` | 匹配数字代号机型 | `iPhone 17 Pro Max` |
| `_AIR_PAT` | `iPhone\s*Air...` | 匹配 iPhone Air | `iPhone Air` |
