# 随机抽取数据清洗验证操作指南

本文档说明如何从 `shop-data/` 随机抽取 Excel 文件，验证各 shop 清洗器是否正常工作。

---

## 一、数据来源

| 项目 | 说明 |
|------|------|
| **目录** | `shop-data/{shopN}/`，如 `shop-data/shop13/` |
| **格式** | `.xlsx` 文件 |
| **命名** | 通常为 `{id}-{date}-shop{N}.xlsx` |

**说明**：shop7 数据本身有缺失，已从验证范围排除。

---

## 二、环境要求

- Python 3（含 pandas、openpyxl）
- Django 项目已配置（需连接数据库以加载 iPhone 机型 catalog）
- 激活虚拟环境（若有）

---

## 三、验证脚本

| 项目 | 说明 |
|------|------|
| **路径** | `scripts/verify_shops_sample.py` |
| **入口** | `run_cleaner`（与生产流程一致，含输出去重） |
| **抽样** | 每个 shop 随机抽取 **5** 个文件 |

### 支持的 shop

| shop | 店铺名 |
|------|--------|
| shop2 | 海峡通信 |
| shop3 | 買取一丁目 |
| shop4 | モバイルミックス |
| shop8 | 買取wiki |
| shop9 | アキモバ |
| shop10 | ドラゴンモバイル |
| shop12 | トゥインクル |
| shop13 | 家電市場 |
| shop14 | 買取楽園 |
| shop15 | 買取当番 |
| shop16 | 携帯空間 |
| shop17 | ゲストモバイル |

---

## 四、执行步骤

### 4.1 本地（虚拟环境已激活）

在**项目根目录**执行：

```bash
# 验证全部 shop
python scripts/verify_shops_sample.py

# 验证指定 shop
python scripts/verify_shops_sample.py shop13 shop8 shop10
```

### 4.2 Docker 环境

```bash
docker compose exec web python3 /app/scripts/verify_shops_sample.py

# 指定 shop
docker compose exec web python3 /app/scripts/verify_shops_sample.py shop13
```

---

## 五、输出说明

### 5.1 单文件行

```
00037610889-2026-01-02-shop13.xlsx: in=312 out=43 OK | 存在多档价格，颜色减价信息有体现, 输出 43 个 part_number
```

| 字段 | 含义 |
|------|------|
| `in` | 输入 Excel 行数 |
| `out` | 清洗后输出行数 |
| `OK` / `WARN` | 通过 / 有告警 |
| `存在多档价格` | 输出中价格有差异，颜色减价信息被正确解析 |

### 5.2 结论行

```
结论: 通过
```
或
```
结论: 有问题
```

### 5.3 汇总

```
======================================================================
汇总: 12/12 shop 验证通过
======================================================================
```

---

## 六、验证内容

脚本会检查：

1. **输出存在**：是否有 DataFrame 输出
2. **列结构**：必含 `part_number`, `shop_name`, `price_new`, `recorded_at`
3. **非空**：part_number、price_new 无空值
4. **价格区间**：price_new 在 30,000～500,000 日元
5. **颜色减价**：同文件内不同 part_number 应有价格差异（体现颜色差价）

---

## 七、常见问题

| 情况 | 可能原因 |
|------|----------|
| 目录不存在 | 确认 `shop-data/{shopN}/` 存在且含 `.xlsx` |
| 无 xlsx 文件 | 该 shop 数据目录为空或格式不符 |
| 错误 / traceback | 数据库未连、列缺失、清洗器异常 |
| out=0 | 输入无匹配机型，或必选列缺失 |
| price_new 不合理 | 个别样本价格异常（如 0、超出区间） |

---

## 八、复现报告

验证完成后，可参考 [shops_verification_report.md](shops_verification_report.md) 的格式整理结果，并更新执行日期与结论。
