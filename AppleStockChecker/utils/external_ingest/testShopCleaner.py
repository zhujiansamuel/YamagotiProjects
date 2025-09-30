import argparse
import importlib.util
import os
import re
import sys
from datetime import datetime

import pandas as pd


# ---------------- helper (与清洗器同逻辑的小工具，用于断言) ----------------
YEN_RE = re.compile(r"[^\d]+")

def parse_yen(val):
    if val is None: return None
    s = str(val).strip()
    if not s: return None
    s = YEN_RE.sub("", s)
    if not s: return None
    try:
        return int(s)
    except Exception:
        return None

def norm(s): return (s or "").strip()

def parse_capacity_gb(text):
    if not text: return None
    m = re.search(r"(\d+)\s*(TB|GB)", str(text), flags=re.IGNORECASE)
    if not m: return None
    qty = int(m.group(1)); unit = m.group(2).upper()
    return qty * 1024 if unit == "TB" else qty

def parse_rule(s: str) -> dict:
    """'青-1000' / '銀-5000+++青-5000' -> {'青': -1000, '銀': -5000}"""
    rules = {}
    if not s: return rules
    parts = re.split(r"\+{1,3}|[,、，\s]+", str(s))
    for p in parts:
        p = p.strip()
        if not p: continue
        m = re.match(r"(.+?)-(\d+)", p)
        if not m: continue
        rules[m.group(1).strip()] = -int(m.group(2))
    return rules

def adjust_for_color(color_name: str, rules: dict) -> int:
    c = color_name or ""
    adj = 0
    for group, delta in rules.items():
        g = group.strip()
        if g in ("青", "ブルー"):
            if "ブルー" in c: adj += delta
        elif g in ("銀", "シルバー"):
            if "シルバー" in c: adj += delta
        else:
            if g and g in c: adj += delta
    return adj

def is_simfree_unopened(s: str) -> bool:
    s = (s or "").lower()
    return ("simfree" in s) and ("未開封" in s) and ("開封" not in s)


# ---------------- dynamic import base_cleaners ----------------
def import_cleaners(cleaners_path: str = None):
    """
    优先 import 系统中的 base_cleaners；若提供 --cleaners 路径，则动态加载该文件。
    """
    if cleaners_path:
        spec = importlib.util.spec_from_file_location("base_cleaners", cleaners_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    else:
        return __import__("base_cleaners")


# ---------------- main test routine ----------------
def main():
    ap = argparse.ArgumentParser(description="Validate shop2 cleaner")
    ap.add_argument("--shop2", required=True, help="shop2.csv path")
    ap.add_argument("--info", required=True, help="iphone17_info.csv path")
    ap.add_argument("--cleaners", help="base_cleaners.py path (optional)")
    ap.add_argument("-o", "--output", help="output cleaned csv (optional)")
    args = ap.parse_args()

    # 1) import cleaner
    cleaners = import_cleaners(args.cleaners)
    if hasattr(cleaners, "CLEANER_REGISTRY") and "shop2" in cleaners.CLEANER_REGISTRY:
        cleaner = cleaners.CLEANER_REGISTRY["shop2"]
    elif hasattr(cleaners, "clean_shop2"):
        cleaner = cleaners.clean_shop2
    else:
        print("[ERROR] shop2 cleaner not found in base_cleaners")
        sys.exit(1)

    # 2) load CSVs
    shop2_df = pd.read_csv(args.shop2)
    info_df  = pd.read_csv(args.info)

    # 3) run cleaner
    out = cleaner(shop2_df, info_df)
    print("\n=== cleaner output (head) ===")
    print(out.head(10))
    print(f"\nrows: {len(out)}")

    # 4) basic assertions
    ok = True

    # a) 全部 shop_name = 海峡通信
    if not out.empty:
        shops = out["shop_name"].dropna().unique().tolist()
        if shops != ["海峡通信"]:
            ok = False
            print("[FAIL] shop_name unexpected:", shops)
        else:
            print("[OK] shop_name all '海峡通信'")

    # b) price_new > 0
    if (out["price_new"] <= 0).any():
        bad = out[out["price_new"] <= 0]
        ok = False
        print(f"[FAIL] price_new <= 0 rows: {len(bad)}")
    else:
        print("[OK] all price_new > 0")

    # c) recorded_at 可解析
    def parse_dt(x):
        try:
            return pd.to_datetime(x, utc=True, errors="coerce")
        except Exception:
            return pd.NaT

    bad_dt = out["recorded_at"].apply(parse_dt).isna().sum()
    if bad_dt > 0:
        ok = False
        print(f"[WARN] recorded_at parse failed count: {bad_dt}")
    else:
        print("[OK] recorded_at parseable")

    # 5) rule validation (抽样): 当 data5 有 “青-/銀-” 等，验证对应颜色价格低于基准
    # 将原始 shop2_df 过滤到 simfree 未開封
    s2 = shop2_df.copy()
    s2.columns = [c.strip().lower() for c in s2.columns]
    for c in ["data2-1", "data2-2", "data3", "data5", "time-scraped"]:
        if c not in s2.columns: s2[c] = None
    s2f = s2[s2["data2-2"].apply(is_simfree_unopened)].copy()

    # 基于 info_df 找机型+容量的颜色组 -> part_numbers
    info = info_df.copy()
    info["model_name"] = info["model_name"].apply(norm)
    info["color"] = info["color"].apply(norm)

    # 从有 data5 规则的行里抽 5 条做检查
    rule_rows = s2f[s2f["data5"].astype(str).str.contains(r"(青-|銀-|シルバー-)", na=False)].head(5)

    def find_model_loose(modelcap):
        token = (modelcap or "").lower()
        token = re.sub(r"iphone\s*", "iphone ", token)
        token = re.sub(r"[^0-9a-z\s+]", "", token)
        token = re.sub(r"\s+", " ", token).strip()
        # 简单包含匹配
        cands = list(dict.fromkeys(info["model_name"].dropna().tolist()))
        def nm(m):
            mm = (m or "").lower()
            mm = re.sub(r"iphone\s*", "iphone ", mm)
            mm = re.sub(r"[^0-9a-z\s+]", "", mm)
            mm = re.sub(r"\s+", " ", mm).strip()
            return mm
        hits = [m for m in cands if token in nm(m) or nm(m) in token]
        if hits: return sorted(hits, key=lambda m: len(m), reverse=True)[0]
        return None

    if not rule_rows.empty and not out.empty:
        mismatches = 0
        for _, r in rule_rows.iterrows():
            base_price = parse_yen(r.get("data3"))
            rules = parse_rule(r.get("data5"))
            modelcap = norm(r.get("data2-1"))
            cap = parse_capacity_gb(modelcap)
            mdl = find_model_loose(modelcap)
            if not base_price or not cap or not mdl:
                continue
            sub = info[(info["model_name"]==mdl) & (info["capacity_gb"]==cap)]
            if sub.empty: continue

            # 取这批 part 的输出价格
            parts = set(sub["part_number"].dropna().tolist())
            out_sub = out[out["part_number"].isin(parts)]
            if out_sub.empty:
                continue

            # 检查 蓝/银 系列的价格 < 其他颜色（或 < 基础价）
            lowered_ok = False
            for _, it in sub.iterrows():
                color = norm(it["color"])
                pn = it["part_number"]
                adj = adjust_for_color(color, rules)
                row_p = out_sub[out_sub["part_number"]==pn]["price_new"]
                if row_p.empty: continue
                price = int(row_p.iloc[0])
                if adj < 0:
                    # 预期降价
                    if price <= base_price + adj + 1:   # 允许 ±1 的四舍五入误差
                        lowered_ok = True
                        break
            if not lowered_ok:
                mismatches += 1

        if mismatches == 0:
            print("[OK] rules (data5) sampled check passed (blue/silver reduced)")
        else:
            print(f"[WARN] rules (data5) sampled check found {mismatches} potential mismatches")
    else:
        print("[INFO] no rows with data5 rules to sample, or no output rows")

    # 6) export if requested
    if args.output:
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"[SAVE] output -> {args.output}")

    print("\n=== done ===")
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
