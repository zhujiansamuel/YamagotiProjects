# Shop10 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop10_cleaner.py`
> 店铺名称: ドラゴンモバイル

---

## 一、总流程图

整个 shop10 清洗器从 data2 / price / time-scraped 列解析机型与容量，映射到 part_number 并展开所有颜色。

```mermaid
flowchart TD
    A[输入: data2 / price / time-scraped] --> B[_load_iphone17_info_df 机型信息]
    B --> C[校验必要列]
    C --> D[_normalize_model_generic 机型归一化]
    D --> E[_parse_capacity_gb 容量解析]
    E --> F[extract_price_yen 价格提取]
    F --> G[(model_norm, cap_gb) 匹配 info 表]
    G --> H[取该机型所有颜色 part_number 展开]
    H --> I[clean_shop10 主函数]
    I --> J[输出: part_number, shop_name, price_new, recorded_at]
```

---

## 二、核心函数说明

| 函数 | 作用 |
|------|------|
| `_load_iphone17_info_df()` | 读取 CSV/Excel 机型信息（本地，非 cleaner_tools 数据库） |
| `_normalize_model_generic(text)` | 机型归一化（cleaner_tools） |
| `_parse_capacity_gb(text)` | 容量解析（cleaner_tools） |
| `extract_price_yen(raw)` | 价格提取（cleaner_tools） |

---

## 三、配置项说明

shop10 无 OLLAMA / EXTRACTION_MODE 配置。机型信息来自 `IPHONE17_INFO_CSV` 或 `EXTERNAL_IPHONE17_INFO_PATH`。
