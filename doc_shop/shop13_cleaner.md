# Shop13 清洗器详细流程说明

> 文件路径: `AppleStockChecker/utils/external_ingest/shop_cleaners_split/shop13_cleaner.py`
> 店铺名称: 家電市場

---

## 一、总流程图

整个 shop13 清洗器从「新品価格」「買取商品2」「time-scraped」解析机型、容量，通过 `_load_iphone17_info_df_from_db()` 取该机型所有颜色的 PN 展开。

```mermaid
flowchart TD
    A[输入: 新品価格 / 買取商品2 / time-scraped] --> B[校验必要列]
    B --> C[_load_iphone17_info_df_from_db]
    C --> D[_normalize_model_generic 机型归一化]
    D --> E[_parse_capacity_gb 容量解析]
    E --> F[extract_price_yen 价格提取]
    F --> G[(model_norm, cap_gb) 匹配 info 表]
    G --> H[取该机型所有颜色 part_number 展开]
    H --> I[clean_shop13 主函数]
    I --> J[输出: part_number, shop_name, price_new, recorded_at]
```

---

## 二、核心函数说明

| 函数 | 作用 |
|------|------|
| `_load_iphone17_info_df_from_db()` | 机型信息（cleaner_tools） |
| `_normalize_model_generic(text)` | 机型归一化（cleaner_tools） |
| `_parse_capacity_gb(text)` | 容量解析（cleaner_tools） |
| `extract_price_yen(raw)` | 价格提取（cleaner_tools） |

---

## 三、配置项说明

shop13 无 OLLAMA / EXTRACTION_MODE 配置。
