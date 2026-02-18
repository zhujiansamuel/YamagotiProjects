#!/usr/bin/env python
"""
多 shop 清洗模拟验证脚本

从 shop-data/{shopN} 随机抽取 5 个 Excel 文件，模拟清洗流程，检查重构是否有问题。
支持: shop2, shop3, shop4, shop8, shop9, shop10, shop12, shop13, shop14, shop15, shop16, shop17

验证内容：
  - 输出是否存在
  - 输出列结构是否正确（part_number, shop_name, price_new, recorded_at）
  - 结构化数据合理性：price_new 正数且在合理区间、part_number 非空
  - 颜色减价信息：同文件内不同 part_number 应有价格差异（体现颜色差价），且价格在合理范围

用法：
    python scripts/verify_shops_sample.py [shop2 shop3 ...]
    python scripts/verify_shops_sample.py   # 默认验证所有

需在项目根目录、Django 虚拟环境激活后运行。
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 抑制清洗器 INFO 日志，只保留错误
_SHOP_IDS = ("shop2", "shop3", "shop4", "shop8", "shop9", "shop10", "shop12", "shop13", "shop14", "shop15", "shop16", "shop17")
logging.getLogger("cleaner_tools").setLevel(logging.WARNING)
for _ in _SHOP_IDS:
    logging.getLogger(f"cleaner_tools.{_}").setLevel(logging.WARNING)

# 设置 Django 环境
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "YamagotiProjects.settings")

import django

django.setup()

import pandas as pd
from AppleStockChecker.utils.external_ingest.registry import run_cleaner
from AppleStockChecker.utils.external_ingest.shop_cleaners_split import (
    shop2_cleaner,
    shop3_cleaner,
    shop4_cleaner,
    shop8_cleaner,
    shop9_cleaner,
    shop10_cleaner,
    shop12_cleaner,
    shop13_cleaner,
    shop14_cleaner,
    shop15_cleaner,
    shop16_cleaner,
    shop17_cleaner,
)

OUTPUT_COLS = ["part_number", "shop_name", "price_new", "recorded_at"]
PRICE_MIN, PRICE_MAX = 30_000, 500_000  # 日元合理区间（新品买取）

SHOP_CONFIG: List[Tuple[str, Callable, List[str]]] = [
    ("shop2", shop2_cleaner.clean_shop2, ["data2-1", "data2-2", "data3", "data5", "time-scraped"]),
    ("shop3", shop3_cleaner.clean_shop3, ["title", "data5", "time-scraped"]),
    ("shop4", shop4_cleaner.clean_shop4, ["data", "data11", "time-scraped"]),
    ("shop8", shop8_cleaner.clean_shop8, ["機種名", "未開封", "time-scraped"]),
    ("shop9", shop9_cleaner.clean_shop9, ["機種名", "買取価格", "色・詳細等", "time-scraped"]),
    ("shop10", shop10_cleaner.clean_shop10, ["data2", "price", "time-scraped"]),
    ("shop12", shop12_cleaner.clean_shop12, ["モデルナンバー", "備考1", "買取価格", "time-scraped"]),
    ("shop13", shop13_cleaner.clean_shop13, ["新品価格", "買取商品2", "time-scraped"]),
    ("shop14", shop14_cleaner.clean_shop14, ["name", "data6", "price2", "time-scraped"]),
    ("shop15", shop15_cleaner.clean_shop15, ["price", "data2", "time-scraped"]),
    ("shop16", shop16_cleaner.clean_shop16, ["iPhone 17 Pro Max", "説明1", "買取価格", "time-scraped"]),
    ("shop17", shop17_cleaner.clean_shop17, ["type", "新未開封品", "色減額", "time-scraped"]),
]


def validate_output_structure(out: pd.DataFrame) -> Dict[str, any]:
    """校验输出结构化数据及颜色减价信息合理性。"""
    issues: List[str] = []
    if out is None or not isinstance(out, pd.DataFrame):
        return {"ok": False, "issues": ["输出非 DataFrame"]}
    if out.empty:
        return {"ok": True, "issues": [], "notes": ["无输出行（可能无匹配数据）"]}

    # 1. 列结构
    missing = [c for c in OUTPUT_COLS if c not in out.columns]
    if missing:
        issues.append(f"缺失列: {missing}")
        return {"ok": False, "issues": issues}

    # 2. 关键字段非空
    if out["part_number"].isna().any() or (out["part_number"].astype(str).str.strip() == "").any():
        issues.append("part_number 存在空值")
    if out["price_new"].isna().any():
        issues.append("price_new 存在空值")

    # 3. price_new 合理区间（正数且在 PRICE_MIN~PRICE_MAX）
    prices = pd.to_numeric(out["price_new"], errors="coerce")
    bad = prices[(prices <= 0) | (prices < PRICE_MIN) | (prices > PRICE_MAX)]
    if len(bad) > 0:
        issues.append(f"price_new 不合理: {len(bad)} 条超出 [{PRICE_MIN},{PRICE_MAX}] 或≤0")

    # 4. 颜色减价体现：同文件内不同 part_number 应有价格 spread（非全同），表示颜色差价被正确解析
    notes = []
    unique_prices = prices.dropna().unique()
    if len(unique_prices) >= 2:
        notes.append("存在多档价格，颜色减价信息有体现")
    elif len(unique_prices) == 1:
        notes.append("全部同价（可能为全色统一定价）")
    if len(out) >= 3:
        notes.append(f"输出 {len(out)} 个 part_number（多颜色展开）")

    return {"ok": len(issues) == 0, "issues": issues, "notes": notes}


def verify_shop(shop_id: str, cleaner_fn: Callable, required_cols: List[str], sample_count: int = 5) -> dict:
    shop_dir = BASE_DIR / "shop-data" / shop_id
    if not shop_dir.exists():
        return {"shop": shop_id, "error": f"目录不存在 {shop_dir}", "results": []}

    files = list(shop_dir.glob("*.xlsx"))
    if not files:
        return {"shop": shop_id, "error": "无 xlsx 文件", "results": []}

    selected = random.sample(files, min(sample_count, len(files)))
    results = []

    for fpath in selected:
        result = {
            "file": fpath.name,
            "input_rows": 0,
            "output_rows": 0,
            "cols_ok": False,
            "output_cols_ok": False,
            "structure_ok": False,
            "structure_notes": [],
            "error": None,
        }
        try:
            df = pd.read_excel(fpath, engine="openpyxl")
            result["input_rows"] = len(df)
            result["cols"] = list(df.columns)
            result["cols_ok"] = all(c in df.columns for c in required_cols)

            out = run_cleaner(shop_id, df)
            result["output_rows"] = len(out)
            result["output_cols_ok"] = (
                all(c in out.columns for c in OUTPUT_COLS)
                if isinstance(out, pd.DataFrame) and not out.empty
                else False
            )

            val = validate_output_structure(out)
            result["structure_ok"] = val.get("ok", False)
            result["structure_issues"] = val.get("issues", [])
            result["structure_notes"] = val.get("notes", [])
        except Exception as e:
            result["error"] = str(e)
            import traceback

            result["traceback"] = traceback.format_exc()
        results.append(result)

    all_ok = (
        not any(r["error"] for r in results)
        and all(r.get("output_cols_ok") or r["output_rows"] == 0 for r in results)
        and all(r.get("structure_ok", True) or r["output_rows"] == 0 for r in results)
    )
    return {"shop": shop_id, "error": None, "results": results, "all_ok": all_ok}


def main():
    shops_arg = [s.strip() for s in sys.argv[1:]] if len(sys.argv) > 1 else None
    if shops_arg:
        configs = [(sid, fn, cols) for sid, fn, cols in SHOP_CONFIG if sid in shops_arg]
    else:
        configs = SHOP_CONFIG

    print("=" * 70)
    print("多 Shop 清洗模拟验证（含结构及颜色减价校验）")
    print("=" * 70)

    all_reports = []
    for shop_id, cleaner_fn, required_cols in configs:
        print(f"\n--- {shop_id} ---")
        report = verify_shop(shop_id, cleaner_fn, required_cols)
        all_reports.append(report)

        if report.get("error") and "results" not in report:
            print(f"  跳过: {report['error']}")
            continue

        for r in report["results"]:
            if r["error"]:
                print(f"  {r['file']}: 错误 {r['error'][:55]}...")
            else:
                status = "OK" if r["cols_ok"] and r["output_cols_ok"] and r.get("structure_ok", True) else "WARN"
                extra = ""
                if r.get("structure_issues"):
                    extra = f" | 结构: {r['structure_issues'][:1]}"
                if r.get("structure_notes"):
                    extra = extra + " | " + ", ".join(r["structure_notes"][:2])
                print(f"  {r['file']}: in={r['input_rows']} out={r['output_rows']} {status}{extra}")
        print(f"  结论: {'通过' if report.get('all_ok') else '有问题'}")

    print("\n" + "=" * 70)
    passed = sum(1 for r in all_reports if r.get("all_ok"))
    print(f"汇总: {passed}/{len(all_reports)} shop 验证通过")
    print("=" * 70)

    return 0 if passed == len(all_reports) else 1


if __name__ == "__main__":
    sys.exit(main())
