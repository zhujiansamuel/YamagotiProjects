# 复合标签渐进式分割方案

## 📋 背景

当前 shop2 使用固定的分割正则 `LABEL_SPLIT_RE_shop2 = r"[／/、，,・\s]+"`，但实际数据中：
- 分割符号可能变化（如 `&`、`;`、`|` 等）
- 可能存在未被分割正则识别的复合标签
- 需要验证分割结果是否为有效颜色

## 🎯 核心思路

利用以下两个事实：
1. **iPhone 颜色种类有限**：每个机型的颜色都在数据库 `color_map` 中
2. **颜色同义词已知**：`FAMILY_SYNONYMS_COLOR` + `_label_matches_color_unified` 可验证

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

### 阶段 3: 未提取颜色检测

检测数据库中存在但未被提取的颜色：

```python
def detect_missing_colors(
    extracted_labels: List[str],
    original_text: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType,
) -> List[Dict]:
    """
    检测原文中可能存在但未被提取的颜色。

    返回:
        [{"color_norm": ..., "color_raw": ..., "found_in_text": ...}, ...]
    """
    missing = []
    text_lower = original_text.lower()
    text_norm = _norm_strip(original_text)

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

        # 1. 检查 color_raw 原文
        if color_raw.lower() in text_lower or color_raw in original_text:
            found_synonym = color_raw
        # 2. 检查同义词
        elif synonyms:
            for syn in synonyms:
                syn_lower = syn.lower()
                if syn_lower in text_lower or syn in original_text:
                    found_synonym = syn
                    break

        if found_synonym:
            missing.append({
                "color_norm": color_norm,
                "color_raw": color_raw,
                "part_number": pn,
                "found_synonym": found_synonym,
                "in_original_text": original_text,
            })

    return missing
```

### 阶段 4: 自适应分割策略选择

```python
def split_composite_label_adaptive(
    label_text: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType,
) -> Dict:
    """
    自适应分割复合标签，返回最佳分割结果。

    返回:
        {
            "strategy_used": str,  # 使用的策略名称
            "labels": List[str],   # 有效标签列表
            "invalid_parts": List[str],  # 无效部分
            "missing_colors": List[Dict],  # 潜在未提取的颜色
            "confidence": float,  # 置信度 (0-1)
        }
    """
    best_result = {
        "strategy_used": "none",
        "labels": [],
        "invalid_parts": [],
        "missing_colors": [],
        "confidence": 0.0,
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

        # 2. 验证
        valid_labels, invalid_parts = validate_split_labels(
            parts, color_map, label_matcher
        )

        # 3. 计算置信度
        total_parts = len(parts)
        valid_parts = len(valid_labels)
        confidence = valid_parts / total_parts if total_parts > 0 else 0.0

        # 4. 如果这个策略更好，更新最佳结果
        if confidence > best_result["confidence"]:
            best_result = {
                "strategy_used": strategy["name"],
                "labels": valid_labels,
                "invalid_parts": invalid_parts,
                "missing_colors": [],
                "confidence": confidence,
            }

    # 5. 检测潜在遗漏的颜色
    best_result["missing_colors"] = detect_missing_colors(
        best_result["labels"],
        label_text,
        color_map,
        label_matcher,
    )

    return best_result
```

## 📊 日志与监控

### 日志事件类型

```python
# 1. 分割策略选择日志
logger.debug(
    "Composite label split",
    extra={
        "event_type": "composite_label_split",
        "strategy_used": result["strategy_used"],
        "original_text": label_text,
        "valid_labels": result["labels"],
        "invalid_parts": result["invalid_parts"],
        "confidence": result["confidence"],
    }
)

# 2. 未提取颜色警告
if result["missing_colors"]:
    logger.warning(
        "Potential missing colors detected",
        extra={
            "event_type": "missing_colors_detected",
            "original_text": label_text,
            "extracted_labels": result["labels"],
            "missing_colors": result["missing_colors"],
            "model_name": model_name,
            "capacity_gb": capacity_gb,
        }
    )

# 3. 无效部分警告
if result["invalid_parts"]:
    logger.warning(
        "Invalid label parts after split",
        extra={
            "event_type": "invalid_label_parts",
            "original_text": label_text,
            "invalid_parts": result["invalid_parts"],
            "strategy_used": result["strategy_used"],
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

## 📈 渐进式部署策略

### Phase 1: 观察模式（只记录，不改变行为）

```python
# 在 shop2_cleaner.py 中添加配置
ENABLE_ADAPTIVE_SPLIT_LOG = os.getenv("SHOP2_ADAPTIVE_SPLIT_LOG", "false").lower() == "true"
ENABLE_ADAPTIVE_SPLIT = os.getenv("SHOP2_ADAPTIVE_SPLIT", "false").lower() == "true"

if ENABLE_ADAPTIVE_SPLIT_LOG:
    # 运行自适应分割但只记录差异，不改变输出
    standard_result = _parse_rule_token_simple(token, enable_adaptive=False)
    adaptive_result = _parse_rule_token_simple(
        token, color_map, label_matcher, enable_adaptive=True
    )

    if standard_result != adaptive_result:
        logger.info(
            "Adaptive split difference detected",
            extra={
                "original_token": token,
                "standard_result": standard_result,
                "adaptive_result": adaptive_result,
            }
        )
```

### Phase 2: A/B 测试模式

```python
# 随机选择 10% 的行使用自适应分割
import random
use_adaptive = ENABLE_ADAPTIVE_SPLIT or (random.random() < 0.1)
```

### Phase 3: 全面启用

```python
# 默认启用自适应分割
ENABLE_ADAPTIVE_SPLIT = os.getenv("SHOP2_ADAPTIVE_SPLIT", "true").lower() == "true"
```

## 🎯 优势总结

1. **渐进性**: 多个分割策略从严格到宽松逐步尝试
2. **验证性**: 使用数据库和同义词验证分割结果
3. **可观测性**: 详细的日志记录分割策略选择和遗漏颜色
4. **向后兼容**: 可通过配置开关逐步启用
5. **自适应性**: 自动选择最佳分割策略
6. **完整性检测**: 主动发现潜在遗漏的颜色

## 📝 示例场景

### 场景 1: 标准分隔符

```
输入: "青/オレンジ-2000"
策略: standard
结果: [("青", -2000), ("オレンジ", -2000)]
置信度: 1.0
```

### 场景 2: 非标准分隔符

```
输入: "シルバー&ゴールド-3000"
策略: with_ampersand
结果: [("シルバー", -3000), ("ゴールド", -3000)]
置信度: 1.0
遗漏颜色: []
```

### 场景 3: 潜在遗漏颜色

```
输入: "青-2000（黒は在庫なし）"
策略: standard
结果: [("青", -2000)]
置信度: 1.0
遗漏颜色: [
    {
        "color_norm": "black",
        "found_synonym": "黒",
        "in_original_text": "青-2000（黒は在庫なし）"
    }
]
警告: 检测到文本中提到"黒"但未提取
```

### 场景 4: 无效部分检测

```
输入: "青/備考/オレンジ-2000"
策略: standard
结果: [("青", -2000), ("オレンジ", -2000)]
无效部分: ["備考"]
置信度: 0.67
```

## 🔍 未来改进方向

1. **机器学习增强**: 使用历史数据训练分割模型
2. **上下文感知**: 结合机型信息预测可能的颜色
3. **分割符号学习**: 自动发现新的分隔符模式
4. **模糊匹配**: 容忍拼写错误（如 "オレジ" → "オレンジ"）
