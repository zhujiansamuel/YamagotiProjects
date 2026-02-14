# Shop1 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop1_cleaner.py`
> 店铺名称: 買取商店

---

## 一、总流程图

整个 shop1 清洗器的核心入口是 `clean_shop1(df)` 函数，从原始爬取的 DataFrame 到输出标准化的买取价格 DataFrame。以 JAN 码为主要映射依据。

```mermaid
flowchart TD
    A[输入: 爬取原始 DataFrame] --> B[校验必要列\nJAN / price / time-scraped 或 JSON 列]
    B --> C[_iter_records 规范化记录]
    C --> D[逐条遍历]
    D --> E{_extract_jan_digits\n能提取 JAN?}
    E -->|否| D
    E -->|是| F[jan_map 中查 part_number]
    F --> G{part_number\n存在?}
    G -->|否| D
    G -->|是| H[to_int_yen 解析价格]
    H --> I{price_new 有效?}
    I -->|否| D
    I -->|是| J[生成输出行]
    J --> D
    D -->|遍历结束| K[组装输出 DataFrame]
    K --> L[输出: part_number, shop_name, price_new, recorded_at]
```

---

## 二、核心函数说明

| 函数 | 作用 |
|------|------|
| `_iter_records(df)` | 产出规范化记录，兼容直列（JAN/price/time-scraped）与 JSON 列两种输入格式 |
| `_extract_jan_digits(s)` | 从 JAN 字段提取 13 位数字（cleaner_tools） |
| `_build_jan_map(info_df)` | 构建 JAN → part_number 映射（cleaner_tools） |
| `clean_shop1(df)` | 主入口，仅输出 `_load_iphone17_info_df_from_db()` 中存在的机型 |

---

## 三、配置项说明

shop1 无独立 OLLAMA/EXTRACTION_MODE 配置，机型信息来自 `cleaner_tools._load_iphone17_info_df_from_db()`。
