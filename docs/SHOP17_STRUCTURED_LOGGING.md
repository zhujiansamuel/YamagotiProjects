# Shop17 结构化日志改造文档

## 概述

本文档说明 shop17 清洗器的结构化日志改造，目的是为 ELK 栈提供易于分析的 JSON 格式日志。

## 改造内容

### 1. 日志配置 (settings.py)

#### 新增配置项

```python
# 日志目录
SHOP_CLEANERS_LOG_DIR = LOG_DIR / "shop_cleaners"

# JSON formatter (使用 python-json-logger)
'formatters': {
    'json': {
        '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
        'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
    },
}

# 文件 handler (按天轮转，保留14天)
'shop17_file_json': {
    'class': 'logging.handlers.TimedRotatingFileHandler',
    'filename': 'logs/shop_cleaners/shop17.log',
    'when': 'midnight',
    'interval': 1,
    'backupCount': 14,
    'formatter': 'json',
    'level': 'DEBUG',
}

# Logger 配置
'AppleStockChecker.utils.external_ingest.shop17_cleaner': {
    'handlers': ['console', 'shop17_file_json'],
    'level': os.getenv('SHOP17_LOG_LEVEL', 'DEBUG'),
    'propagate': False,
}
```

#### 日志输出策略

- **控制台**: INFO 级别，简洁格式（人类可读）
- **文件**: DEBUG 级别，JSON 格式（供 ELK 采集）

### 2. 代码改造 (shop17_cleaner.py)

#### 移除的功能

- `DEBUG_SHOP17` 环境变量控制
- `_dbg_print()` 和 `_dbg_block()` 函数
- 所有 `print()` 语句

#### 新增的日志层级

##### 层级 1: 清洗器生命周期（INFO）

**cleaner_start** - 清洗器启动
```json
{
  "event_type": "cleaner_start",
  "shop_name": "ゲストモバイル",
  "cleaner_name": "shop17",
  "input_rows": 50,
  "start_time": "2026-02-09 10:30:00"
}
```

**cleaner_complete** - 清洗器完成
```json
{
  "event_type": "cleaner_complete",
  "shop_name": "ゲストモバイル",
  "input_rows": 50,
  "output_records": 200,
  "elapsed_seconds": 5.23
}
```

##### 层级 2: 行级概览（INFO）

**row_processing_summary** - 每行处理的概览
```json
{
  "event_type": "row_processing_summary",
  "row_index": 42,
  "model_text": "iPhone Air 256GB",
  "model_norm": "iPhone Air",
  "capacity_gb": 256,
  "base_price": 120000,
  "color_discount_raw_preview": "色減額:シルバーなし/ブルー-1000...",
  "extraction_method": "regex",
  "labels_extracted_count": 2,
  "colors_in_catalog": 4,
  "colors_matched_count": 2,
  "output_records_count": 4,
  "has_discounted_colors": true,
  "min_delta": -1000,
  "max_delta": 0
}
```

##### 层级 3: 提取结果（DEBUG）

**extraction_result** - 正则/LLM 提取的结果
```json
{
  "event_type": "extraction_result",
  "row_index": 42,
  "model_norm": "iPhone Air",
  "capacity_gb": 256,
  "color_discount_raw": "色減額:シルバーなし/ブルー-1000...",
  "color_discount_raw_full": "完整原始内容...",
  "extraction_method": "regex",
  "labels_and_deltas": [
    {"label": "シルバー", "delta": 0},
    {"label": "ブルー", "delta": -1000}
  ],
  "available_colors": [
    {"color_norm": "スペースブラック", "part_number": "ABC123"},
    {"color_norm": "クラウドホワイト", "part_number": "ABC124"}
  ]
}
```

##### 层级 4: 颜色匹配详情（DEBUG）

**label_matching** - 每个 label 的匹配结果
```json
{
  "event_type": "label_matching",
  "row_index": 42,
  "label": "ブルー",
  "delta": -1000,
  "matched_colors": ["スカイブルー"],
  "matched_part_numbers": ["ABC125"],
  "match_count": 1
}
```

##### 层级 5: 输出记录（DEBUG）

**output_record** - 每条输出记录（扁平化）
```json
{
  "event_type": "output_record",
  "row_index": 42,
  "part_number": "ABC125",
  "color_norm": "スカイブルー",
  "base_price": 120000,
  "delta": -1000,
  "final_price": 119000,
  "delta_source": "matched_label",
  "matched_label": "ブルー"
}
```

##### 层级 6: 异常情况（WARNING/ERROR）

**llm_extraction_error** - LLM 提取失败
```json
{
  "event_type": "llm_extraction_error",
  "error": "Connection timeout",
  "model_id": "gemma3:1b",
  "text_preview": "..."
}
```

**label_no_match** - Label 未匹配到颜色
```json
{
  "event_type": "label_no_match",
  "row_index": 42,
  "label": "グレー",
  "delta": -2000,
  "available_colors": ["スペースブラック", "クラウドホワイト"]
}
```

## 使用指南

### 运行测试

```bash
cd /home/samuelzhu/YamagotiProjects
python test_shop17_logging.py
```

### 查看日志

**控制台输出** (简洁):
```
INFO Starting shop17 cleaner
INFO Row 1: iPhone Air 256GB → 2 labels, 2 colors matched, 4 records output
INFO Shop17 cleaner completed
```

**日志文件** (详细 JSON):
```bash
tail -f logs/shop_cleaners/shop17.log
# 或查看最近的日志
tail -20 logs/shop_cleaners/shop17.log | jq .
```

### 环境变量控制

```bash
# 设置日志级别（默认 DEBUG）
export SHOP17_LOG_LEVEL=INFO

# 运行 Django
python manage.py runserver
```

## ELK 集成

### Filebeat 配置示例

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /home/samuelzhu/YamagotiProjects/logs/shop_cleaners/*.log
    json.keys_under_root: true
    json.add_error_key: true
    fields:
      log_type: shop_cleaner
      environment: production

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "shop-cleaners-%{+yyyy.MM.dd}"
```

### Kibana 查询示例

#### 1. 验证提取准确性
```
event_type: extraction_result AND labels_and_deltas.label: "ブルー"
```

#### 2. 查看颜色匹配情况
```
event_type: label_matching AND matched_colors: "スカイブルー"
```

#### 3. 统计价格分布
```
event_type: output_record
→ 按 delta 字段聚合，创建直方图
```

#### 4. 发现未匹配的 label
```
event_type: label_no_match
→ 按 label 字段分组统计
```

#### 5. 监控异常情况
```
levelname: WARNING OR levelname: ERROR
```

#### 6. 分析处理性能
```
event_type: cleaner_complete
→ 按 elapsed_seconds 字段绘制趋势图
```

## Kibana Dashboard 建议

### 1. 概览仪表盘
- 总处理行数 (cleaner_start.input_rows)
- 总输出记录数 (cleaner_complete.output_records)
- 平均处理时间 (cleaner_complete.elapsed_seconds)
- 错误率 (ERROR / 总日志数)

### 2. 色減額解析仪表盘
- 提取方法分布 (extraction_method: regex vs llm)
- Label 匹配率 (matched / total labels)
- 未匹配的 label Top 10
- Delta 分布直方图

### 3. 异常监控仪表盘
- LLM 提取失败次数
- 未匹配 label 列表
- 错误日志时间线

## 字段索引

### 核心字段
- `event_type`: 事件类型（用于过滤）
- `row_index`: 行索引（用于关联）
- `model_norm`: 标准化型号
- `capacity_gb`: 容量
- `color_norm`: 标准化颜色

### 色減額相关
- `color_discount_raw`: 原始 "色減額" 字段
- `color_discount_raw_full`: 完整原始内容
- `color_discount_normalized`: 标准化后的内容
- `labels_and_deltas`: 提取的 label 和 delta 数组

### 匹配相关
- `matched_colors`: 匹配到的颜色列表
- `matched_part_numbers`: 匹配到的 part_number 列表
- `match_count`: 匹配数量

### 价格相关
- `base_price`: 基础价格
- `delta`: 价格差额
- `final_price`: 最终价格
- `delta_source`: delta 来源（matched_label / default_zero）

## 后续扩展

### 为其他 shop cleaners 添加结构化日志

1. 在 settings.py 添加对应的 handler:
```python
'shop2_file_json': {
    'class': 'logging.handlers.TimedRotatingFileHandler',
    'filename': str(SHOP_CLEANERS_LOG_DIR / 'shop2.log'),
    ...
}
```

2. 在 settings.py 添加对应的 logger:
```python
'AppleStockChecker.utils.external_ingest.shop2_cleaner': {
    'handlers': ['console', 'shop2_file_json'],
    ...
}
```

3. 在清洗器代码中按 shop17 的模式添加结构化日志

### 性能优化

如果日志量太大影响性能，可以考虑：

1. **采样日志**: 只详细记录部分行
2. **异步日志**: 使用 QueueHandler
3. **调整级别**: 生产环境设置为 INFO
4. **缩短保留期**: backupCount 改为 7 天

## 常见问题

### Q: 日志文件没有创建？
A: 检查 `logs/shop_cleaners/` 目录权限，确保 Django 进程有写入权限。

### Q: JSON 日志格式不正确？
A: 确保 `python-json-logger` 已安装（requirements.txt 中已包含）。

### Q: 控制台输出太多？
A: 将 console handler 的 level 改为 WARNING。

### Q: 想看更详细的匹配过程？
A: 设置环境变量 `export SHOP17_LOG_LEVEL=DEBUG` 并重启。

## 总结

通过这次改造，shop17 清洗器现在可以：

1. ✅ 输出结构化的 JSON 日志供 ELK 分析
2. ✅ 在控制台显示简洁的人类可读日志
3. ✅ 详细记录"色減額"的解析过程
4. ✅ 按时间轮转日志，自动保留14天
5. ✅ 通过环境变量灵活控制日志级别
6. ✅ 为每个处理阶段提供专门的事件类型

这为后续的日志分析、问题排查和性能监控提供了坚实的基础。
