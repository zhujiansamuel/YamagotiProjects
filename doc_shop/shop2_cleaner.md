# Shop2 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop2_cleaner.py`
> 店铺名称: 海峡通信

---

## 一、总流程图

整个 shop2 清洗器的核心入口是 `clean_shop2(shop2_df, debug, debug_limit)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\ndata2-1 / data2-2 / data3 / data5 / time-scraped]
    B -->|缺列| B1[自动补 None 保持兼容]
    B -->|通过| C[过滤目标行\ndata2-2 含 simfree AND 未開封]
    C -->|无匹配行| C1[返回空 DataFrame]
    C -->|有匹配行| D[加载 iphone17_info 参考表\n_load_iphone17_info_df_from_db]

    D --> E[逐行遍历 DataFrame]

    E --> F{data2-1 是否为空?}
    F -->|空| E
    F -->|非空| G[容量解析\n_parse_capacity_gb]

    G --> H{容量\n能否解析?}
    H -->|否| E
    H -->|是| I[机型宽松匹配\n_pick_model_name_loose]

    I --> J{model_name\n是否匹配?}
    J -->|否| E
    J -->|是| K[在 info 中查找\n该机型+容量的子集 sub]

    K --> L{sub\n是否为空?}
    L -->|是| E
    L -->|否| M[解析基准价格\n_parse_yen data3]

    M --> N{基准价格\n是否有效?}
    N -->|否| E
    N -->|是| O[解析颜色减价规则\n_parse_adjust_rule data5]

    O --> P[遍历 sub 中每个颜色]
    P --> Q[计算每个颜色的最终价格\n_apply_adjust_with_trace\nprice = base_price + adj]

    Q --> R{price > 0?}
    R -->|否| P
    R -->|是| S[生成输出行\npart_number / shop_name / price_new / recorded_at]
    S --> P

    P -->|遍历结束| E
    E -->|遍历结束| T[组装输出 DataFrame]
    T --> U[输出: 标准化 DataFrame\npart_number, shop_name, price_new, recorded_at]
```

---

## 二、函数流程图

### 2.1 函数调用关系总览

```mermaid
flowchart LR
    clean["clean_shop2(shop2_df, debug, debug_limit)"]

    clean --> load["_load_iphone17_info_df_from_db()"]
    clean --> parsecap["_parse_capacity_gb(text)"]
    clean --> pickmodel["_pick_model_name_loose(model_token, iphone17_df)"]
    clean --> parseyen["_parse_yen(val)"]
    clean --> parserule["_parse_adjust_rule(val)"]
    clean --> applyadj["_apply_adjust_with_trace(color_name, rules)"]
    clean --> parsedt["parse_dt_aware(val)"]

    parserule --> llm["_parse_adjust_rule_llm(rule_text)\n@lru_cache(1024)"]
    parserule --> simpleall["_parse_adjust_rule_simple_all(val)"]

    llm --> lxextract["lx.extract()\nLangExtract + Ollama"]
    llm --> astext["_as_text(val)"]
    llm --> tokenparse["_parse_rule_token_simple(token)"]
    llm --> coerceint["_coerce_int(val)"]
    llm --> regexfb["_parse_adjust_rule_regex(val)\n正则回退"]

    simpleall --> tokenparse
    simpleall --> astext

    regexfb --> astext

    applyadj --> matchcolor["_match_color_group(group, color_name)"]
    pickmodel --> normtoken["_norm_model_token(s)"]
```

### 2.2 核心函数详细说明

#### `clean_shop2(shop2_df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame`
- **作用**: 清洗器主入口，将原始爬取数据转化为标准四列输出
- **输入**: 包含 `data2-1`, `data2-2`, `data3`, `data5`, `time-scraped` 列的 DataFrame
- **输出**: 包含 `part_number`, `shop_name`, `price_new`, `recorded_at` 列的 DataFrame
- **过滤条件**: data2-2 必须同时包含 `"simfree"` 和 `"未開封"`
- **debug 模式**: 选出含颜色减价规则的行，输出对照信息（最多 debug_limit 行）

#### `_pick_model_name_loose(model_token: str, iphone17_df: pd.DataFrame) -> Optional[str]`
- **作用**: 在 iphone17_info 的 model_name 列中宽松匹配输入的机型文本
- **匹配策略**:

```mermaid
flowchart TD
    A["输入: model_token"] --> B["_norm_model_token()\n小写化 / 去符号 / 仅保留 a-z0-9 空格"]
    B --> C["获取 iphone17_df 中所有\nmodel_name 候选（去重）"]
    C --> D["逐个候选做包含匹配\ntoken in norm(m) 或 norm(m) in token"]
    D --> E{命中数量?}
    E -->|0| F[返回 None]
    E -->|1| G[返回该命中项]
    E -->|多个| H["返回最长的命中项\n（更具体的优先）"]
```

#### `_parse_adjust_rule(val) -> dict`
- **作用**: 解析颜色减价规则的调度函数，采用 **LLM 优先、正则补全** 策略

```mermaid
flowchart TD
    A["_parse_adjust_rule(val)"] --> B["_as_text(val)\n文本规范化"]
    B --> C{文本为空?}
    C -->|是| D["返回空字典 {}"]
    C -->|否| E["_parse_adjust_rule_llm(s)\nLLM 解析（带缓存）"]
    E --> F["_parse_adjust_rule_simple_all(s)\n保守正则解析"]
    F --> G["合并结果\nmerged = dict(llm_rules)\nfor k,v in supplement:\n  merged.setdefault(k, v)\nLLM 结果优先"]
    G --> H["返回 merged"]
```

#### `_parse_adjust_rule_llm(rule_text: str) -> dict`
- **作用**: 使用 LangExtract + Ollama 本地 LLM 解析颜色减价规则
- **缓存**: `@lru_cache(maxsize=1024)`，对相同 rule_text 避免重复调用 LLM
- **流程**:

```mermaid
flowchart TD
    A["输入: rule_text"] --> B{LangExtract 可用?}
    B -->|否| C["回退: _parse_adjust_rule_regex(s)"]
    B -->|是| D["调用 lx.extract()"]

    D --> D1["配置参数:\nmodel_id = gemma3:1b\nmodel_url = localhost:11434\nprompt = _COLOR_RULE_PROMPT\nexamples = 2个 few-shot 示例\nfence_output = False\nuse_schema_constraints = False"]

    D1 --> E{调用成功?}
    E -->|异常| C
    E -->|成功| F["将 result 转为 dict"]
    F --> G["遍历 extractions"]
    G --> H["优先从 extraction_text\n按行用 _parse_rule_token_simple 解析"]
    H --> I["再用 attributes 中的\ngroup_label + delta_yen 兜底"]
    I --> J{解析出规则?}
    J -->|否| C
    J -->|是| K["返回 rules 字典"]
```

#### `_parse_adjust_rule_simple_all(val) -> dict`
- **作用**: 保守补全解析，按分隔符拆开后逐段用 `_parse_rule_token_simple` 解析
- **用途**: 补齐 LLM 可能漏掉的规则；不覆盖 LLM 已有的 key（通过 `setdefault` 合并）
- **分隔符**: `+++`, `++`, `+`, 全角加号, 换行, 顿号, 逗号

#### `_parse_rule_token_simple(token: str) -> Optional[Tuple[str, int]]`
- **作用**: 解析单条规则 token，如 `'黒-2000'` -> `('黒', -2000)`
- **流程**: 从末尾向前找数字串 -> 找符号 (+/-) -> 取出组名 (group)

#### `_parse_adjust_rule_regex(val) -> dict`
- **作用**: 旧版纯正则解析（fallback），用于 LLM 不可用或 LLM 解析为空时的回退
- **处理**: 按 `+++`, 逗号, 空格拆分，逐段正则匹配 `(.+?)-(\d+)`

#### `_match_color_group(group: str, color_name: str) -> Tuple[bool, str]`
- **作用**: 判断减价规则中的"组名"是否匹配 info 表中的某个颜色名
- **前处理**: 去除组名末尾的"系"/"色"后缀
- **匹配策略** (家族映射):

```mermaid
flowchart TD
    A["输入: group, color_name"] --> A1["去除 group 末尾的 系/色 后缀"]
    A1 --> B{group 属于哪个家族?}

    B -->|"青/ブルー/ミストブルー/ディープブルー/スカイブルー"| C1["检查 color_name 含 ブルー"]
    B -->|"銀/シルバー"| C2["检查 color_name 含 シルバー"]
    B -->|"黒/ブラック"| C3["检查 color_name 含 ブラック/黒/ミッドナイト"]
    B -->|"白/ホワイト"| C4["检查 color_name 含 ホワイト/白/シルバー"]
    B -->|"金/ゴールド"| C5["检查 color_name 含 ゴールド"]
    B -->|"橙/オレンジ"| C6["检查 color_name 含 オレンジ/橙"]
    B -->|"其他非空 group"| C7["Fallback: 检查 group in color_name\n子串匹配"]
    B -->|"空 group"| C8["返回 (False, '')"]

    C1 --> Z["返回 (is_match, reason)"]
    C2 --> Z
    C3 --> Z
    C4 --> Z
    C5 --> Z
    C6 --> Z
    C7 --> Z
```

#### `_apply_adjust_with_trace(color_name: str, rules: dict) -> Tuple[int, list[dict]]`
- **作用**: 对指定颜色名应用所有减价规则，返回调整总额和 trace 信息
- **输出**: `(adjust_sum, [{"group": "青", "delta": -1000, "reason": "contains ブルー"}, ...])`
- **特点**: 累加所有命中规则的 delta 值，trace 记录每条命中的详细信息

#### `_parse_yen(val) -> Optional[int]`
- **作用**: 将各种日元价格写法转为整数
- **处理**: `'¥177,000'` / `'177,000円'` / `'177000'` -> `177000`

#### `_parse_capacity_gb(text: str) -> Optional[int]`
- **作用**: 从文本中提取容量 (GB)
- **处理**: 支持 TB->GB 换算 (1TB=1024GB)，支持 `"256GB"`, `"1TB"` 等格式

#### `_load_iphone17_info_df_from_db() -> pd.DataFrame`
- **作用**: 读取 iphone17_info 参考表
- **数据源**: Django settings / 环境变量 `IPHONE17_INFO_CSV` / 自动推断路径
- **输出列**: `part_number`, `model_name`, `capacity_gb`, `color`，若检测到 jan 列则额外返回

#### `_as_text(val) -> str`
- **作用**: 将可能为 NaN/None/数字/字符串的输入统一规范为字符串
- **排除**: `nan`, `none`, `null` 均返回空字符串

#### `_coerce_int(val) -> Optional[int]`
- **作用**: 将 int/float/str 数字（含日元符号、逗号、全角符号）稳健转为 int
- **处理**: 去除逗号/円/¥，全角符号转半角，正则提取数字

---

## 三、数据流程图

### 3.1 整体数据流

```mermaid
flowchart TD
    subgraph 输入数据
        RAW["原始爬取 DataFrame\n列: data2-1, data2-2, data3, data5, time-scraped, ..."]
        INFO["iphone17_info.csv\n列: part_number, model_name, capacity_gb, color, (jan)"]
    end

    subgraph 过滤阶段
        FILTER["data2-2 过滤\n必须含 simfree AND 未開封"]
    end

    subgraph 中间数据结构
        MODEL["model_name + cap_gb\n由 data2-1 解析得到"]
        SUB["sub DataFrame\ninfo 中匹配 model_name + cap_gb 的子集"]
        RULES["rules 字典\n{group_label: delta_yen}\n如: {'青': -1000, '銀': -5000}"]
    end

    subgraph 输出数据
        OUT["标准化 DataFrame\n列: part_number, shop_name, price_new, recorded_at"]
    end

    RAW --> FILTER
    FILTER --> |"逐行读取"| PROC
    INFO --> SUB

    subgraph PROC[逐行处理]
        direction TB
        P1["data2-1 → model_name + cap_gb"]
        P2["data3 → base_price (int)"]
        P3["data5 → rules (LLM + 正则)"]
        P4["sub 中每个颜色 + rules → adj"]
        P5["base_price + adj → price_new"]
    end

    PROC --> MODEL
    MODEL --> SUB
    PROC --> RULES
    SUB --> OUT
    RULES --> OUT
```

### 3.2 单行数据处理示例

以一行实际数据为例，展示完整的数据转换过程:

```
输入行:
  data2-1      = "iPhone17 Pro Max 256GB"
  data2-2      = "simfree+未開封"
  data3        = "180,000"
  data5        = "青-1000"
  time-scraped = "2025-06-01 12:00:00"
```

```mermaid
flowchart TD
    subgraph Step1["Step 1: 过滤检查"]
        T0["data2-2 = 'simfree+未開封'"]
        T0 -->|"含 simfree AND 未開封"| T0a["通过过滤 ✓"]
    end

    subgraph Step2["Step 2: 机型 + 容量解析"]
        T1["data2-1 = 'iPhone17 Pro Max 256GB'"]
        T1 -->|_parse_capacity_gb| T2["cap_gb = 256"]
        T1 -->|_pick_model_name_loose| T3["model_name = 'iPhone 17 Pro Max'"]
    end

    subgraph Step3["Step 3: 查询 info 子集"]
        T4["key = (model_name='iPhone 17 Pro Max', cap_gb=256)"]
        T4 -->|查 info| T5["sub = {\n  'ブラックチタニウム': 'MYW23J/A',\n  'ホワイトチタニウム': 'MYW53J/A',\n  'ナチュラルチタニウム': 'MYW83J/A',\n  'サンドチタニウム': 'MYWF3J/A',\n  ...\n}"]
    end

    subgraph Step4["Step 4: 基准价格"]
        T6["data3 = '180,000'"]
        T6 -->|_parse_yen| T7["base_price = 180000"]
    end

    subgraph Step5["Step 5: 颜色减价规则解析"]
        T8["data5 = '青-1000'"]
        T8 -->|"_parse_adjust_rule\n(LLM优先 + 正则补全)"| T9["rules = {'青': -1000}"]
    end

    subgraph Step6["Step 6: 逐颜色计算价格"]
        T10["对 sub 中每个颜色:"]
        T10 --> T11["ブラックチタニウム\n→ _match_color_group('青', 'ブラックチタニウム')\n→ 不含ブルー → adj=0\n→ price=180000"]
        T10 --> T12["ホワイトチタニウム\n→ adj=0 → price=180000"]
        T10 --> T13["(假设含ブルー的颜色)\n→ _match_color_group('青', '...ブルー...')\n→ 含ブルー → adj=-1000\n→ price=179000"]
    end

    subgraph Step7["Step 7: 输出行"]
        T14["{\n  part_number: 'MYW23J/A',\n  shop_name: '海峡通信',\n  price_new: 180000,\n  recorded_at: datetime(...)\n},\n{\n  part_number: '...',\n  shop_name: '海峡通信',\n  price_new: 179000,\n  recorded_at: datetime(...)\n},\n..."]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6
    Step6 --> Step7
```

### 3.3 颜色减价规则解析 - LLM 优先 + 正则补全策略

```mermaid
flowchart TD
    INPUT["data5 原始文本"]

    INPUT --> RULE["_parse_adjust_rule(val)"]
    RULE --> LLM["_parse_adjust_rule_llm(s)\nLangExtract + Ollama gemma3:1b\n带 @lru_cache(1024) 缓存"]
    RULE --> SIMPLE["_parse_adjust_rule_simple_all(s)\n保守正则逐段解析"]

    LLM --> MERGE["合并结果\nmerged = dict(llm_rules)"]
    SIMPLE --> MERGE
    MERGE --> SETDEFAULT["for k,v in supplement:\n  merged.setdefault(k, v)\nLLM 结果优先\n正则仅补齐缺失 key"]
    SETDEFAULT --> RESULT["最终 rules 字典"]

    subgraph LLM解析器详细
        L1["prompt: _COLOR_RULE_PROMPT\n买取表色减额解析专用提示词"]
        L2["examples: 2个 few-shot 示例\n示例1: '青-1000' → color_rule\n示例2: '銀-5000+++青-3000' → 2条 color_rule"]
        L3["model: gemma3:1b (本地 Ollama)"]
        L4["解析: 优先从 extraction_text 按行解析\n再用 attributes 兜底"]
        L5["失败回退: _parse_adjust_rule_regex"]
    end

    subgraph 正则补全解析器详细
        S1["分隔符拆分:\n+++ / ++ / + / 换行 / 顿号 / 逗号"]
        S2["逐段: _parse_rule_token_simple\n从末尾找数字 → 找符号 → 取组名"]
    end
```

### 3.4 颜色家族匹配机制

```mermaid
flowchart LR
    subgraph 颜色家族映射表["_match_color_group 家族映射"]
        BLUE["Blue 系\n青 / ブルー / ミストブルー\nディープブルー / スカイブルー\n→ 检查含 ブルー"]
        SILVER["Silver 系\n銀 / シルバー\n→ 检查含 シルバー"]
        BLACK["Black 系\n黒 / ブラック\n→ 检查含 ブラック/黒/ミッドナイト"]
        WHITE["White 系\n白 / ホワイト\n→ 检查含 ホワイト/白/シルバー"]
        GOLD["Gold 系\n金 / ゴールド\n→ 检查含 ゴールド"]
        ORANGE["Orange 系\n橙 / オレンジ\n→ 检查含 オレンジ/橙"]
        FALLBACK["Fallback\n其他非空 group\n→ 子串匹配 group in color_name"]
    end

    LABEL["规则中的 group\n如: '青'"]
    COLOR["info表中的 color\n如: 'マリンブルー'"]

    LABEL -->|"查家族映射"| BLUE
    BLUE -->|"ブルー in マリンブルー"| MATCH["匹配成功!\nadj = delta"]
```

---

## 四、配置项说明

OLLAMA 与 EXTRACTION_MODE 配置已统一迁移至 `cleaner_tools.py`。

| 配置项 | 值 | 类型 | 说明 |
|--------|-----|------|------|
| `OLLAMA_MODEL_ID` | `"gemma3:1b"` | cleaner_tools | Ollama 本地 LLM 模型名 |
| `OLLAMA_URL` | `"http://localhost:11434"` | cleaner_tools | Ollama 服务地址 |
| `EXTRACTION_MODE` | `"regex"` | cleaner_tools | regex / llm / auto |
| `@lru_cache(maxsize=1024)` | 1024 | 硬编码 | `_parse_adjust_rule_llm` 的缓存大小 |
| `SHOP` | `"海峡通信"` | 硬编码 | 输出 DataFrame 的 shop_name 值 |
| `debug` | `True` | 函数参数 | 是否输出调试信息 |
| `debug_limit` | `30` | 函数参数 | Debug 输出最大行数 |
| `IPHONE17_INFO_CSV` | 自动推断路径 | 环境变量 | iphone17_info 文件路径（优先 Django settings） |
| `lx.extract` 参数 `fence_output` | `False` | 硬编码 | LangExtract 不使用 fence 输出 |
| `lx.extract` 参数 `use_schema_constraints` | `False` | 硬编码 | LangExtract 不使用 schema 约束 |

---

## 五、关键正则表达式

| 名称 | 模式 | 用途 | 示例匹配 |
|------|------|------|---------|
| `_YEN_RE` | `[^\d]+` | 去除价格中的非数字字符 | `'¥177,000'` → `'177000'` |
| `_norm_model_token` 内 | `iphone\s*` → `iphone ` | 规范化 iPhone 前缀的空格 | `iPhone17` → `iphone 17` |
| `_norm_model_token` 内 | `[^0-9a-z\s+]` → `""` | 仅保留字母数字和空格 | 去除 `フリー` 等日文 |
| `_parse_capacity_gb` TB | `(\d+(?:\.\d+)?)\s*TB` | 匹配 TB 容量 | `1TB` → `1024` |
| `_parse_capacity_gb` GB | `(\d{2,4})\s*GB` | 匹配 GB 容量 | `256GB` → `256` |
| `_parse_adjust_rule_regex` 分隔 | `\+{1,3}\|[,、，\s]+` | 拆分多条减价规则 | `銀-5000+++青-3000` |
| `_parse_adjust_rule_regex` 规则 | `(.+?)-(\d+)` | 匹配单条"组名-金额"规则 | `青-1000` → `('青', -1000)` |
| `_RULE_PAT` (debug) | `(青\|銀\|黒\|白\|橙\|ブルー\|シルバー\|ブラック\|ホワイト\|オレンジ).{0,6}-\d+` | Debug 模式: 检测 data5 是否含颜色减价规则 | `青-1000`, `シルバー-5000` |
| `_INT_RE` | `[+-]?\d+` | 从字符串中提取整数 | `'-1000円'` → `-1000` |
| `_match_color_group` 后缀清理 | `(系\|色)$` | 去除组名末尾的"系"/"色" | `青系` → `青`, `黒色` → `黒` |
| iphone17_info 文件扩展名 | `\.(xlsx\|xlsm\|xls\|ods)$` | 判断参考表是否为 Excel 格式 | `iphone17_info.xlsx` |
