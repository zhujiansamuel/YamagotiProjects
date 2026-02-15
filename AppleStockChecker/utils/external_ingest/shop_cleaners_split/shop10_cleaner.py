from __future__ import annotations
"""
shop10 清洗器 — ドラゴンモバイル

  原始 DataFrame（data2 / price / time-scraped）
    │
    ├─ _load_iphone17_info_df()   ← Step 1: 机型信息（model_name_norm, capacity_gb）
    ├─ _normalize_model_generic() ← Step 2: 机型归一化（cleaner_tools）
    ├─ _parse_capacity_gb()       ← Step 3: 容量解析（cleaner_tools）
    ├─ extract_price_yen()        ← Step 4: 价格提取（cleaner_tools）
    └─ clean_shop10()             ← Step 5: 主函数，输出 part_number / price_new / recorded_at
"""
from typing import List, Optional
from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _normalize_model_generic, extract_price_yen, assemble_output_df, validate_columns, _load_info_df_from_csv
import re
import pandas as pd
import time

def clean_shop10(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    print("##shop10:ドラゴンモバイル---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    info_df = _load_info_df_from_csv(
        required_cols={"part_number", "model_name", "capacity_gb"},
        output_cols=["part_number", "model_name_norm", "capacity_gb"],
        add_model_norm=True,
    )

    validate_columns(df, ["data2", "price", "time-scraped"],
                     cleaner_name="shop10", shop_name="ドラゴンモバイル")

    # 解析
    model_norm = df["data2"].map(_normalize_model_generic)
    cap_gb     = df["data2"].map(_parse_capacity_gb)

    price_new   = df["price"].map(extract_price_yen)
    recorded_at = df["time-scraped"].map(parse_dt_aware)

    groups = (
        info_df.groupby(["model_name_norm", "capacity_gb"])["part_number"]
        .apply(list).to_dict()
    )

    # ---- DEBUG: 仅打印“疑似包含颜色/减价信息”的行（限制条数）----
    debug_pos_set = set()
    if debug:
        _COLOR_DISCOUNT_PAT = re.compile(
            r"(ブラック|ホワイト|ブルー|グリーン|ピンク|レッド|イエロー|パープル|ゴールド|シルバー|"
            r"グラファイト|ミッドナイト|スターライト|ナチュラル|チタニウム|チタン|"
            r"Black|White|Blue|Green|Pink|Red|Yellow|Purple|Gold|Silver|Titanium|"
            r"値下げ|値引|割引|円引|OFF|オフ|[-−–]\s*\d|\d+\s*円\s*(?:引|OFF|オフ))",
            re.I
        )

        s_data2  = df["data2"].fillna("").astype(str)
        s_price  = df["price"].fillna("").astype(str)
        mask = s_data2.str.contains(_COLOR_DISCOUNT_PAT, na=False) | s_price.str.contains(_COLOR_DISCOUNT_PAT, na=False)

        # 取前 debug_limit 条“命中”的行（按位置）
        hit_cnt = 0
        for pos, hit in enumerate(mask.to_numpy()):
            if hit:
                debug_pos_set.add(pos)
                hit_cnt += 1
                if hit_cnt >= debug_limit:
                    break

        print(f"[shop10 debug] total_rows={len(df)}, hit_rows={int(mask.sum())}, print_rows={len(debug_pos_set)}")

    rows = []
    for i in range(len(df)):
        raw_data2 = df["data2"].iat[i]
        raw_price = df["price"].iat[i]

        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p = price_new.iat[i]
        t = recorded_at.iat[i]

        # 先准备匹配信息（便于 debug 输出）
        key = None
        pn_list = []
        if m and (not pd.isna(c)):
            key = (m, int(c))
            pn_list = groups.get(key, [])

        def _dbg_print(reason: str | None = None):
            # 只对命中行打印（避免刷屏）
            if not debug or i not in debug_pos_set:
                return
            print("\n[shop10 debug] row_pos=", i)
            print("  data2(raw):", repr(raw_data2))
            print("  price(raw):", repr(raw_price))
            print("  model_norm:", repr(m))
            print("  capacity_gb:", repr(c))
            print("  price_new:", repr(p))
            print("  recorded_at:", repr(t))
            print("  match_key:", repr(key))
            print("  part_numbers:", pn_list[:10], f"(len={len(pn_list)})")
            if reason:
                print("  SKIP_REASON:", reason)

        # 过滤逻辑（保持原逻辑不变，只是在 skip 时打印原因）
        if not m:
            _dbg_print("model_norm 为空（无法识别机型）")
            continue
        if pd.isna(c):
            _dbg_print("capacity_gb 为空（无法识别容量）")
            continue
        if p is None:
            _dbg_print("price_new 为空（价格无法转为日元整数）")
            continue
        if not pn_list:
            _dbg_print("未匹配到 part_number（groups 中无对应机型+容量）")
            continue

        # 通过过滤：也打印一次“提取&匹配结果”
        _dbg_print(None)

        for pn in pn_list:
            rows.append({
                "part_number": str(pn),
                "shop_name": "ドラゴンモバイル",
                "price_new": int(p),
                "recorded_at": t,
            })

            # 如果你希望看到“最终写入行”，可打开下面这个（同样只对命中行打印）
            if debug and i in debug_pos_set:
                print("  -> OUT_ROW:", {"part_number": str(pn), "price_new": int(p)})

    out = assemble_output_df(rows, coerce_price=False)

    if debug:
        print(f"\n[shop10 debug] out_rows={len(out)}  out_head=\n", out.head(10).to_string(index=False))

    return out
