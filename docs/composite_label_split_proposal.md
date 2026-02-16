# 复合标签渐进式分割方案（最终版）

## 📋 背景与问题

当前 shop2 使用固定的分割正则 `LABEL_SPLIT_RE_shop2 = r"[／/、，,・\s]+"`，但实际数据中：
- 分割符号可能变化（如 `&`、`;`、`|` 等），导致复合标签无法正确分割
- 可能存在未被识别的复合标签，造成颜色遗漏
- 需要验证分割结果是否为有效颜色，避免提取无关文本
- 需要检测原文中提到但未被提取的颜色（特别是有价格信息的）

## 🎯 核心思路

利用以下已有资源：
1. **iPhone 颜色种类有限**：每个机型的颜色都在数据库 `color_map` 中
2. **颜色同义词已知**：`FAMILY_SYNONYMS_COLOR` + `_label_matches_color_unified` 可验证
3. **现有日志字段**：复用 "no match" 等字段，不新增日志结构

## ⚖️ 设计决策（已确认）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **未提取颜色处理** | 记录到现有 "no match" 日志，如有价格信息则尝试提取 | 复用现有日志结构，减少复杂度 |
| **停止策略** | 全匹配提前结束 OR 遍历所有策略 | 直观明确，避免置信度计算 |
| **性能优化** | 暂不考虑 | 先验证正确性，后续再优化 |
| **部署路径** | shop17 试点 → 其他清洗器 | shop17 作为测试床，验证后推广 |

## 🔧 方案设计

### 阶段 1: 渐进式分割尝试

使用多个正则表达式逐次尝试分割，从严格到宽松：

```python
# 分割策略列表（按严格程度排序）
LABEL_SPLIT_STRATEGIES = [
    {
        "name": "standard",  # 标准分割（shop2/4/7/11/12 通用）
        "regex": re.compile(r"[／/、，,・\s]+"),
        "description": "斜杠、顿号、逗号、中点、空格"
    },
    {
        "name": "with_semicolon",  # 包含分号（shop3/9/14）
        "regex": re.compile(r"[／/、，,・\s;；]+"),
        "description": "标准 + 分号"
    },
    {
        "name": "with_ampersand",  # 包含 & 符号（shop15）
        "regex": re.compile(r"[／/、，,・\s&＆]+"),
        "description": "标准 + & 符号"
    },
    {
        "name": "with_pipe",  # 包含竖线
        "regex": re.compile(r"[／/、，,・\s|｜]+"),
        "description": "标准 + 竖线"
    },
    {
        "name": "aggressive",  # 激进模式（所有常见分隔符）
        "regex": re.compile(r"[／/、，,・\s;；&＆|｜]+"),
        "description": "所有常见分隔符"
    },
]
```

### 阶段 2: 颜色有效性验证

对每个分割结果，使用现有的匹配机制验证：

```python
def validate_split_labels(
    labels: List[str],
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType,
) -> Tuple[List[str], List[str]]:
    """
    验证分割后的标签是否为有效颜色。

    返回:
        (valid_labels, invalid_labels)
    """
    valid = []
    invalid = []

    for label in labels:
        label_cleaned = label.strip()
        if not label_cleaned:
            continue

        # 尝试匹配任意数据库中的颜色
        matched = False
        for color_norm, (pn, color_raw) in color_map.items():
            if label_matcher(label_cleaned, color_raw, color_norm):
                matched = True
                break

        if matched:
            valid.append(label_cleaned)
        else:
            invalid.append(label_cleaned)

    return valid, invalid
```

### 阶段 3: 未提取颜色检测（带价格信息识别）

检测数据库中存在但未被提取的颜色，特别关注带价格信息的颜色：

```python
def detect_missing_colors_with_price(
    extracted_labels: List[str],
    original_text: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType,
) -> List[Dict]:
    """
    检测原文中可能存在但未被提取的颜色。
    如果颜色后面跟着价格/减价信息，会标记为"应提取"。

    返回:
        [
            {
                "color_norm": ...,
                "color_raw": ...,
                "found_synonym": ...,
                "has_price_info": bool,  # 是否有价格信息
                "price_pattern": str,     # 价格模式（如 "-1000", "+500"）
                "should_extract": bool,   # 是否应该提取
            },
            ...
        ]
    """
    # 价格模式：查找颜色后面的 +/-数字
    price_pattern = re.compile(r"([+\-＋－])[\s]*(\d+)")

    missing = []
    text_lower = original_text.lower()

    for color_norm, (pn, color_raw) in color_map.items():
        # 检查该颜色是否已被提取
        already_extracted = any(
            label_matcher(label, color_raw, color_norm)
            for label in extracted_labels
        )

        if already_extracted:
            continue

        # 检查原文中是否包含该颜色的任何同义词
        synonyms = SYNONYM_LOOKUP_NORM.get(color_norm, [])
        found_synonym = None
        found_position = -1

        # 1. 检查 color_raw 原文
        if color_raw in original_text:
            found_synonym = color_raw
            found_position = original_text.index(color_raw)
        elif color_raw.lower() in text_lower:
            found_synonym = color_raw
            found_position = text_lower.index(color_raw.lower())
        # 2. 检查同义词
        elif synonyms:
            for syn in synonyms:
                if syn in original_text:
                    found_synonym = syn
                    found_position = original_text.index(syn)
                    break
                elif syn.lower() in text_lower:
                    found_synonym = syn
                    found_position = text_lower.index(syn.lower())
                    break

        if found_synonym and found_position >= 0:
            # 检查颜色后面是否有价格信息
            text_after = original_text[found_position:found_position + 50]  # 取后面50字符
            price_match = price_pattern.search(text_after)

            has_price = price_match is not None
            price_str = price_match.group(0) if has_price else None

            missing.append({
                "color_norm": color_norm,
                "color_raw": color_raw,
                "part_number": pn,
                "found_synonym": found_synonym,
                "has_price_info": has_price,
                "price_pattern": price_str,
                "should_extract": has_price,  # 有价格信息则应该提取
                "context": text_after[:30],   # 上下文（用于日志）
            })

    return missing
```

### 阶段 4: 自适应分割策略选择（全匹配优先停止）

```python
def split_composite_label_adaptive(
    label_text: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType,
) -> Dict:
    """
    自适应分割复合标签，使用"全匹配优先停止"策略。

    停止条件：
    1. 如果某个策略匹配到了该机型的所有颜色，立即停止并返回
    2. 否则尝试完所有策略后，返回匹配颜色最多的结果

    返回:
        {
            "strategy_used": str,           # 使用的策略名称
            "labels": List[str],            # 有效标签列表
            "matched_color_count": int,     # 匹配到的颜色数量
            "total_colors_in_catalog": int, # 该机型总颜色数
            "is_full_match": bool,          # 是否全匹配
            "invalid_parts": List[str],     # 无效部分
            "missing_colors": List[Dict],   # 潜在未提取的颜色
        }
    """
    total_colors = len(color_map)
    best_result = {
        "strategy_used": "none",
        "labels": [],
        "matched_color_count": 0,
        "total_colors_in_catalog": total_colors,
        "is_full_match": False,
        "invalid_parts": [],
        "missing_colors": [],
    }

    # 尝试各种分割策略
    for strategy in LABEL_SPLIT_STRATEGIES:
        # 1. 分割
        parts = [
            p.strip()
            for p in strategy["regex"].split(label_text)
            if p.strip()
        ]

        if not parts:
            continue

        # 2. 验证并收集匹配的颜色
        valid_labels = []
        invalid_parts = []
        matched_colors = set()  # 使用 set 避免重复计数

        for label in parts:
            label_cleaned = label.strip()
            if not label_cleaned:
                continue

            # 尝试匹配任意数据库中的颜色
            matched = False
            for color_norm, (pn, color_raw) in color_map.items():
                if label_matcher(label_cleaned, color_raw, color_norm):
                    valid_labels.append(label_cleaned)
                    matched_colors.add(color_norm)
                    matched = True
                    break

            if not matched:
                invalid_parts.append(label_cleaned)

        matched_count = len(matched_colors)

        # 3. 如果匹配到所有颜色，立即返回（提前终止）
        if matched_count == total_colors:
            return {
                "strategy_used": strategy["name"],
                "labels": valid_labels,
                "matched_color_count": matched_count,
                "total_colors_in_catalog": total_colors,
                "is_full_match": True,
                "invalid_parts": invalid_parts,
                "missing_colors": [],  # 全匹配时无遗漏颜色
            }

        # 4. 如果这个策略更好（匹配更多颜色），更新最佳结果
        if matched_count > best_result["matched_color_count"]:
            best_result = {
                "strategy_used": strategy["name"],
                "labels": valid_labels,
                "matched_color_count": matched_count,
                "total_colors_in_catalog": total_colors,
                "is_full_match": False,
                "invalid_parts": invalid_parts,
                "missing_colors": [],
            }

    # 5. 检测潜在遗漏的颜色（仅当非全匹配时）
    best_result["missing_colors"] = detect_missing_colors(
        best_result["labels"],
        label_text,
        color_map,
        label_matcher,
    )

    return best_result
```

## 📊 日志与监控（复用现有字段）

### 设计原则
- **复用现有日志结构**：使用 "no match" 等现有事件类型
- **不新增字段**：扩展现有字段的信息内容
- **向后兼容**：确保现有日志分析工具仍可工作

### 日志事件类型

```python
# 1. 分割策略选择日志（DEBUG 级别）
logger.debug(
    "Composite label split",
    extra={
        "event_type": "composite_label_split",
        "strategy_used": result["strategy_used"],
        "original_text": label_text,
        "valid_labels": result["labels"],
        "matched_color_count": result["matched_color_count"],
        "total_colors_in_catalog": result["total_colors_in_catalog"],
        "is_full_match": result["is_full_match"],
        "invalid_parts": result["invalid_parts"],  # 无效部分也在这里记录
    }
)

# 2. 未提取颜色记录（复用 "no match" 事件）
# 注意：这里复用现有的 "no match" 事件类型，只是扩展了信息
for missing_color in result["missing_colors"]:
    if missing_color["should_extract"]:  # 仅记录应该提取的（有价格信息）
        logger.warning(
            "Color mentioned in text but not extracted",
            extra={
                "event_type": "no_match",  # 复用现有事件类型
                "label_raw": missing_color["found_synonym"],
                "label_type": "missing_with_price",  # 扩展：标记类型
                "color_norm": missing_color["color_norm"],
                "color_raw": missing_color["color_raw"],
                "part_number": missing_color["part_number"],
                "price_pattern": missing_color["price_pattern"],
                "context": missing_color["context"],
                "model_name": model_name,
                "capacity_gb": capacity_gb,
                "row_index": row_index,
            }
        )
    else:  # 没有价格信息的，记录为 INFO
        logger.info(
            "Color mentioned in text (no price info)",
            extra={
                "event_type": "no_match",
                "label_raw": missing_color["found_synonym"],
                "label_type": "missing_no_price",
                "color_norm": missing_color["color_norm"],
                "context": missing_color["context"],
            }
        )

# 3. 全匹配成功日志（INFO 级别）
if result["is_full_match"]:
    logger.info(
        "Full color match achieved",
        extra={
            "event_type": "composite_label_full_match",
            "strategy_used": result["strategy_used"],
            "matched_colors": result["matched_color_count"],
            "model_name": model_name,
            "capacity_gb": capacity_gb,
        }
    )
```

## 🔄 集成到现有代码

### 修改 `_parse_rule_token_simple`

```python
def _parse_rule_token_simple(
    token: str,
    color_map: Optional[Dict[str, Tuple[str, str]]] = None,
    label_matcher: Optional[LabelMatcherType] = None,
    enable_adaptive: bool = False,
) -> List[Tuple[str, int]]:
    """
    解析单条规则 token，支持自适应复合标签分割。

    参数:
        token: 规则文本（如 '青/オレンジ-2000'）
        color_map: 颜色映射表（可选，用于验证）
        label_matcher: 颜色匹配函数（可选，用于验证）
        enable_adaptive: 是否启用自适应分割（默认 False）
    """
    s = safe_to_text(token)
    if not s:
        return []

    # ... 提取数字和符号的逻辑（与现有代码相同）...

    group = s[:k].strip().strip(" :：\t")
    if not group:
        return []

    amt = sign * int(num_str)

    # 标准分割（向后兼容）
    if not enable_adaptive or not color_map or not label_matcher:
        labels = [
            lbl.strip()
            for lbl in LABEL_SPLIT_RE_shop2.split(group)
            if lbl.strip()
        ]
        return [(lbl, amt) for lbl in labels]

    # 自适应分割（新功能）
    result = split_composite_label_adaptive(
        group, color_map, label_matcher
    )

    return [(lbl, amt) for lbl in result["labels"]]
```

## 📈 渐进式部署策略（shop17 先行）

### 为什么选择 shop17 作为试点？
1. **代表性**：shop17 通常有复杂的标签格式
2. **影响范围可控**：单一清洗器便于问题定位
3. **可对比性**：可与其他清洗器横向对比效果

### Phase 1: shop17 实现与验证（2-4周）

**目标**：在 shop17 中完整实现自适应分割，收集数据验证效果

```python
# 在 shop17_cleaner.py 中实现
ENABLE_ADAPTIVE_SPLIT_SHOP17 = os.getenv("SHOP17_ADAPTIVE_SPLIT", "true").lower() == "true"

def _parse_rule_token_shop17(
    token: str,
    color_map: Dict[str, Tuple[str, str]],
) -> List[Tuple[str, int]]:
    """shop17 专用解析函数，内置自适应分割"""
    if not ENABLE_ADAPTIVE_SPLIT_SHOP17:
        # 回退到标准分割
        return _parse_rule_token_simple_standard(token)

    # 使用自适应分割
    result = split_composite_label_adaptive(
        token, color_map, _label_matches_color_unified
    )

    # 详细日志（用于效果评估）
    logger.debug("shop17 adaptive split", extra={
        "original_token": token,
        "strategy_used": result["strategy_used"],
        "matched_colors": result["matched_color_count"],
        "is_full_match": result["is_full_match"],
        "missing_colors": result["missing_colors"],
    })

    return [(lbl, amt) for lbl in result["labels"]]
```

**验证指标**：
- 提取的颜色数量变化
- 全匹配率（is_full_match）
- 新发现的带价格的遗漏颜色数量
- 无效部分数量

### Phase 2: 效果评估与调优（1-2周）

**数据收集**：
```bash
# 运行 shop17 清洗并收集日志
python -m AppleStockChecker.utils.external_ingest.clean --shop shop17

# 分析日志
grep "composite_label_split" logs/cleaner.log | jq '.'
grep "no_match.*missing_with_price" logs/cleaner.log | jq '.'
```

**关键问题**：
1. 全匹配率是否提升？
2. 是否发现了之前遗漏的带价格颜色？
3. 是否有误判（提取了不应该提取的）？
4. 哪个策略使用最频繁？

### Phase 3: 推广到其他清洗器（按需）

**推广优先级**：
```python
# 根据 shop17 的效果，按优先级推广
ROLLOUT_PRIORITY = [
    "shop17",  # ✅ 已完成
    "shop2",   # 优先级 1：复合标签问题明显
    "shop3",   # 优先级 2：有分号分隔符
    "shop15",  # 优先级 2：有 & 分隔符
    # ... 其他清洗器按需添加
]
```

**统一接口**（便于推广）：
```python
# 在 cleaner_tools.py 中添加统一函数
def parse_rule_token_adaptive(
    token: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType = _label_matches_color_unified,
    enable_adaptive: bool = True,
) -> List[Tuple[str, int]]:
    """
    统一的自适应规则解析函数，供所有清洗器使用。

    参数:
        token: 规则文本
        color_map: 颜色映射表
        label_matcher: 颜色匹配函数
        enable_adaptive: 是否启用自适应分割
    """
    if not enable_adaptive:
        return _parse_rule_token_simple_standard(token)

    # 提取数字和符号
    s = safe_to_text(token)
    if not s:
        return []

    # ... (数字提取逻辑) ...

    # 自适应分割
    result = split_composite_label_adaptive(
        group, color_map, label_matcher
    )

    return [(lbl, amt) for lbl in result["labels"]]
```

### Phase 4: 全面监控与维护

**持续监控指标**：
- 每日全匹配率趋势
- 每周新发现的遗漏颜色数量
- 策略使用分布

**告警设置**：
```python
# 如果某天的全匹配率大幅下降，发送告警
if daily_full_match_rate < baseline_rate * 0.8:
    send_alert("Adaptive split performance degradation detected")
```

## 🎯 方案优势

| 优势 | 说明 | 实现方式 |
|------|------|----------|
| **渐进性** | 多策略逐步尝试，从严格到宽松 | 5个分割策略按序尝试 |
| **完整性** | 检测全匹配，避免遗漏颜色 | 全匹配提前停止机制 |
| **智能性** | 识别带价格的遗漏颜色 | 价格模式正则匹配 |
| **兼容性** | 复用现有日志结构 | 使用 "no match" 等现有字段 |
| **可验证** | 基于数据库和同义词验证 | `_label_matches_color_unified` |
| **可推广** | 统一接口便于推广 | `parse_rule_token_adaptive` |
| **可控性** | shop17 试点，风险可控 | 渐进式部署策略 |

## 📝 示例场景（最终版）

### 场景 1: 标准分隔符 + 全匹配

```
输入: "青/オレンジ-2000"
机型颜色: ["青", "オレンジ"]  # 该机型只有这两种颜色

执行过程:
- 策略 1 (standard): 分割为 ["青", "オレンジ"]
- 验证: 匹配到 2/2 颜色
- 结果: 全匹配！提前停止

输出:
{
    "strategy_used": "standard",
    "labels": ["青", "オレンジ"],
    "matched_color_count": 2,
    "total_colors_in_catalog": 2,
    "is_full_match": True,  ← 全匹配
    "invalid_parts": [],
    "missing_colors": []
}

日志: INFO - Full color match achieved (strategy=standard)
```

### 场景 2: 非标准分隔符 + 策略降级

```
输入: "シルバー&ゴールド&ブラック-3000"
机型颜色: ["シルバー", "ゴールド", "ブラック", "ホワイト"]  # 4种颜色

执行过程:
- 策略 1 (standard): 无法分割（不包含 &）
- 策略 2 (with_semicolon): 无法分割
- 策略 3 (with_ampersand): 分割为 ["シルバー", "ゴールド", "ブラック"]
- 验证: 匹配到 3/4 颜色
- 继续尝试策略 4, 5（未找到更好结果）

输出:
{
    "strategy_used": "with_ampersand",
    "matched_color_count": 3,
    "total_colors_in_catalog": 4,
    "is_full_match": False,
    "missing_colors": [
        {
            "color_norm": "white",
            "found_synonym": None,  # 原文中未提到
            "has_price_info": False
        }
    ]
}

日志: DEBUG - Composite label split (strategy=with_ampersand, matched=3/4)
```

### 场景 3: 遗漏颜色检测（带价格信息）

```
输入: "青-2000、銀-1500（黒は-3000円、在庫少）"
机型颜色: ["青", "銀", "黒"]

执行过程:
- 策略 1 (standard): 分割为 ["青", "銀", "（黒は"]
- 验证: 匹配到 2/3 颜色（"（黒は" 无法匹配）
- 检测遗漏: 发现 "黒" + "-3000" 模式

输出:
{
    "strategy_used": "standard",
    "matched_color_count": 2,
    "missing_colors": [
        {
            "color_norm": "black",
            "found_synonym": "黒",
            "has_price_info": True,  ← 有价格信息！
            "price_pattern": "-3000",
            "should_extract": True,  ← 应该提取
            "context": "黒は-3000円、在庫少"
        }
    ]
}

日志: WARNING - Color mentioned in text but not extracted
      (event_type=no_match, label_type=missing_with_price,
       color_norm=black, price_pattern=-3000)
```

### 场景 4: 遗漏颜色（无价格信息）

```
输入: "青-2000（黒は在庫なし）"

执行过程:
- 提取: ["青"]
- 检测遗漏: 发现 "黒" 但无价格模式

输出:
{
    "missing_colors": [
        {
            "color_norm": "black",
            "found_synonym": "黒",
            "has_price_info": False,  ← 无价格信息
            "should_extract": False,  ← 不应提取
            "context": "黒は在庫なし）"
        }
    ]
}

日志: INFO - Color mentioned in text (no price info)
      (event_type=no_match, label_type=missing_no_price)
```

### 场景 5: 无效部分过滤

```
输入: "青/備考/オレンジ/注意事項-2000"

执行过程:
- 分割: ["青", "備考", "オレンジ", "注意事項"]
- 验证: 只有 "青" 和 "オレンジ" 匹配

输出:
{
    "labels": ["青", "オレンジ"],
    "invalid_parts": ["備考", "注意事項"],  ← 无效部分
    "matched_color_count": 2
}

日志: DEBUG - Composite label split
      (valid_labels=["青","オレンジ"], invalid_parts=["備考","注意事項"])
```

## ✅ 实施清单（shop17 试点）

### 第1步：基础函数实现（cleaner_tools.py）

- [ ] 实现 `validate_split_labels()` - 验证分割结果
- [ ] 实现 `detect_missing_colors_with_price()` - 检测遗漏颜色
- [ ] 实现 `split_composite_label_adaptive()` - 自适应分割核心
- [ ] 添加 `LABEL_SPLIT_STRATEGIES` 配置列表
- [ ] 编写单元测试（至少覆盖 5 个场景）

### 第2步：shop17 集成（shop17_cleaner.py）

- [ ] 修改 `_parse_rule_token_shop17()` 调用自适应分割
- [ ] 添加环境变量开关 `SHOP17_ADAPTIVE_SPLIT`
- [ ] 集成日志记录（复用 "no match" 字段）
- [ ] 更新 shop17 的文档注释

### 第3步：测试与验证

- [ ] 准备 shop17 测试数据集（包含各种场景）
- [ ] 运行清洗并收集日志
- [ ] 分析全匹配率变化
- [ ] 检查新发现的带价格遗漏颜色
- [ ] 检查是否有误判

### 第4步：文档与推广

- [ ] 更新 shop17 清洗器文档
- [ ] 记录效果数据（作为推广依据）
- [ ] 准备推广方案（shop2, shop3, shop15 等）
- [ ] 编写推广教程（如何在其他清洗器中使用）

## 🔍 未来改进方向

### 短期（3-6个月）
1. **统计分析**: 收集各策略使用频率，优化策略顺序
2. **性能优化**: 如需要，添加缓存和提前终止
3. **边界案例**: 处理更多特殊情况（如嵌套括号、多行文本）

### 中期（6-12个月）
1. **上下文感知**: 结合机型信息预测可能的颜色
2. **模糊匹配**: 容忍拼写错误（如 "オレジ" → "オレンジ"）
3. **分割符号学习**: 自动发现新的分隔符模式

### 长期（1年+）
1. **机器学习增强**: 使用历史数据训练分割模型
2. **多语言支持**: 支持英文、中文等其他语言
3. **自适应阈值**: 根据历史数据自动调整匹配策略

## 📚 参考资料

### 相关代码文件
- `cleaner_tools.py`: 颜色匹配、同义词、日志工具
- `shop17_cleaner.py`: shop17 清洗器（试点实现）
- `shop2_cleaner.py`: shop2 清洗器（已支持基础复合标签）

### 相关文档
- 颜色同义词表: `FAMILY_SYNONYMS_COLOR`
- 颜色匹配策略: `_label_matches_color_unified()`
- 现有分割正则: `LABEL_SPLIT_RE_shop*`

### 测试数据示例
```python
# 可在单元测试中使用
TEST_CASES = [
    # (输入, 机型颜色, 期望输出)
    ("青/オレンジ-2000", ["青", "オレンジ"], {"is_full_match": True}),
    ("シルバー&ゴールド-3000", ["シルバー", "ゴールド"], {"strategy": "with_ampersand"}),
    ("青-2000（黒は-3000円）", ["青", "黒"], {"missing_with_price": ["黒"]}),
]
```

---

**文档版本**: v2.0 (最终版)
**最后更新**: 2026-02-16
**状态**: ✅ 方案已确认，待实施
