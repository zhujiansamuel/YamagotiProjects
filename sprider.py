
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iPhone 17 / iPhone Air 買取価格 収集スクリプト（初版骨架）

機能概要:
- 指定の4サイト (shop2~shop5) から iPhone 17 系列 & iPhone Air の
  「未開封 / SIMフリー」二手買取価格を取得し、入力CSVの shop2~shop5 列に書き込みます。
- 入力CSVには model_name, capacity_gb, color の組合せが全て既に存在する前提です。
- 価格が「範囲(例: 100,000～110,000)」の場合は平均値を算出し、末尾に * を付けて出力します。

対応サイト:
- shop2: https://www.mobile-ichiban.com/Prod/1/01
- shop3: https://www.1-chome.com/mobile?category=RGNg976kptBN7UjF
         API: https://www.1-chome.com/api/keitai/listPage?accCode=&page=1&size=24&keyword=&isImpo=false&isCampaign=false&cateCode=RGNg976kptBN7UjF&kbNames=&isImpoCate=true
- shop4: https://mobile-mix.jp/?category=7  (未開封は /html/body/table/tbody/tr[2]/td[1]/span)
         ※色区分なし(同一model+capacityで全色同一価格)
- shop5: https://www.morimori-kaitori.jp/category/price-list/0301066
         ※新品未開封(SIMフリー)のみ抽出

使用方法:
    python fetch_prices.py --in input.csv --out output.csv

必要ライブラリ:
    pip install requests pandas beautifulsoup4 lxml

注意:
- 各サイトのDOMやAPI仕様変更に備え、パーサ関数にはTODOコメントを多めに入れています。
- 実運用前に、実際のHTML/APIレスポンスを確認し、セレクタ/キー名を調整してください。
"""

import re
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from lxml import html

# -------------------------
# 定数/設定
# -------------------------

SHOP2_URL = "https://www.mobile-ichiban.com/Prod/1/01"
SHOP3_API = ("https://www.1-chome.com/api/keitai/listPage"
             "?accCode=&page={page}&size={size}&keyword=&isImpo=false&isCampaign=false"
             "&cateCode=RGNg976kptBN7UjF&kbNames=&isImpoCate=true")
SHOP4_URL = "https://mobile-mix.jp/?category=7"
SHOP5_URL = "https://www.morimori-kaitori.jp/category/price-list/0301066"

# 対象モデル名の正規化: 網羅は適宜拡張
MODEL_ALIASES = {
    # 左=入力CSVの表記例, 右=社内統一表記
    "iphone 17": "iPhone 17",
    "iphone17": "iPhone 17",
    "iphone 17 pro": "iPhone 17 Pro",
    "iphone 17 pro max": "iPhone 17 Pro Max",
    "iphone air": "iPhone Air",
    "iphone17 pro": "iPhone 17 Pro",
    "iphone17 pro max": "iPhone 17 Pro Max",
    "iphoneair": "iPhone Air",
}

# 容量正規化 (文字 -> int GB)
CAPACITY_ALIASES = {
    "1t": 1024, "1tb": 1024, "1024": 1024,
    "2t": 2048, "2tb": 2048, "2048": 2048,
    "256": 256, "512": 512,
}

# 価格抽出に使う正規表現
PRICE_RE = re.compile(r"([\d,]+)\s*(?:円)?")
RANGE_SEP_RE = re.compile(r"[～~\-–—]+")

# HTTP セッション
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
})

# -------------------------
# ユーティリティ
# -------------------------

def normalize_model_name(name: str) -> str:
    """モデル名のゆらぎを正規化。"""
    key = name.strip().lower()
    return MODEL_ALIASES.get(key, name.strip())

def normalize_capacity_gb(cap) -> Optional[int]:
    """容量をGB整数に正規化。"""
    if cap is None:
        return None
    if isinstance(cap, (int, float)):
        return int(cap)
    s = str(cap).strip().lower().replace(" ", "")
    s = s.replace("gb", "")
    return CAPACITY_ALIASES.get(s, safe_int(s))

def safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None

def parse_price_text_to_value_with_flag(text: str) -> Tuple[Optional[int], bool]:
    """
    テキストから価格(円)を整数で抽出。
    - 単一価格: 110,000 → (110000, False)
    - 範囲価格: 100,000～110,000 → (105000, True)  # 平均 & 星印フラグ
    戻り値: (price_int or None, is_averaged_range)
    """
    if not text:
        return None, False

    # 範囲を分割
    if RANGE_SEP_RE.search(text):
        parts = RANGE_SEP_RE.split(text)
        vals = []
        for p in parts:
            m = PRICE_RE.search(p)
            if m:
                vals.append(int(m.group(1).replace(",", "")))
        if len(vals) >= 2:
            avg = round(sum(vals[:2]) / 2)
            return avg, True

    # 単一値
    m = PRICE_RE.search(text)
    if m:
        return int(m.group(1).replace(",", "")), False

    return None, False

def price_to_output_str(price: Optional[int], averaged: bool) -> str:
    if price is None:
        return ""
    return f"{price}{'*' if averaged else ''}"

def fill_prices_for_all_colors(df: pd.DataFrame, key_cols: Tuple[str, str], price_col: str, value: str):
    """
    同一 (model_name, capacity_gb) の全色行に一括で価格を適用。
    df: 入力CSVデータ
    key_cols: ('model_name', 'capacity_gb')
    price_col: 'shopX'
    value: 書き込み用の価格文字列
    """
    m_col, c_col = key_cols
    model, capacity = key_cols
    mask = (df[m_col] == model) & (df[c_col] == capacity)
    df.loc[mask, price_col] = value

# -------------------------
# ページ/データ取得
# -------------------------

def get_html(url: str, retries: int = 3, timeout: int = 20) -> Optional[str]:
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as e:
            time.sleep(1 + i)
    return None

def get_json(url: str, retries: int = 3, timeout: int = 20) -> Optional[dict]:
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            time.sleep(1 + i)
    return None

# -------------------------
# 各ショップのスクレイパ
# -------------------------

def scrape_shop2() -> Dict[Tuple[str, int, Optional[str]], str]:
    """
    mobile-ichiban (shop2)
    戻り値: {(model_name, capacity_gb, color_or_None): "price_str", ...}
    備考:
      - 実HTML構造に合わせて選択子/抽出ロジックを調整してください。
      - 色がない場合は color_or_None=None を返し、後段で全色に適用します。
    """
    html_text = get_html(SHOP2_URL)
    if not html_text:
        return {}

    soup = BeautifulSoup(html_text, "lxml")
    out: Dict[Tuple[str, int, Optional[str]], str] = {}

    # TODO: 以下はダミー例。実サイトのカード/表構造に合わせて修正。
    # 例: 商品カード class="prod-card"
    for card in soup.select(".prod-card"):
        title = card.select_one(".prod-title")
        price = card.select_one(".price")
        if not title or not price:
            continue

        name = normalize_model_name(title.get_text(strip=True))
        # 容量をタイトルから抽出する例: "iPhone 17 256GB"
        cap_match = re.search(r"(\d{3,4})\s*GB", title.get_text("", strip=True), flags=re.I)
        capacity = normalize_capacity_gb(cap_match.group(1)) if cap_match else None

        price_int, averaged = parse_price_text_to_value_with_flag(price.get_text(strip=True))
        price_str = price_to_output_str(price_int, averaged)

        if name and capacity:
            # 色区分が無い想定 → color=None
            out[(name, capacity, None)] = price_str

    return out

def scrape_shop3() -> Dict[Tuple[str, int, Optional[str]], str]:
    """
    1-chome (shop3) - API 取得
    戻り値: {(model_name, capacity_gb, color_or_None): "price_str", ...}
    備考:
      - ページングあり。pageを回して全件取得。
      - 実レスポンスのJSONキーに合わせて field 名を調整してください。
    """
    out: Dict[Tuple[str, int, Optional[str]], str] = {}

    page = 1
    size = 100  # できるだけ大きめで
    while True:
        url = SHOP3_API.format(page=page, size=size)
        data = get_json(url)
        if not data:
            break

        # TODO: 実際のJSON構造に合わせて以下を修正
        # 想定: data = {"data": {"list": [{ "name": "...", "capacity": "256", "color": "ブラック", "price": "110,000" }, ...]}}
        items = None
        if isinstance(data, dict):
            items = data.get("data", {}).get("list") or data.get("list") or data.get("rows")

        if not items:
            break

        for it in items:
            title = it.get("name") or it.get("title") or ""
            name = normalize_model_name(title)
            capacity = normalize_capacity_gb(it.get("capacity") or "")
            color = (it.get("color") or "").strip() or None
            price_text = it.get("price") or it.get("maxPrice") or it.get("minPrice") or ""
            price_int, averaged = parse_price_text_to_value_with_flag(str(price_text))
            price_str = price_to_output_str(price_int, averaged)

            if name and capacity:
                out[(name, capacity, color)] = price_str

        # ページ末尾判定 (総数・ページ数・hasNext 等に合わせて調整)
        has_next = False
        # 例: total / page / size から計算
        total = (data.get("data") or {}).get("total") if isinstance(data, dict) else None
        if total and (page * size) < int(total):
            has_next = True

        if not has_next:
            break
        page += 1

    return out

def scrape_shop4() -> Dict[Tuple[str, int, Optional[str]], str]:
    """
    mobile-mix (shop4)
    - /html/body/table/tbody/tr[2]/td[1]/span に「未開封」表記があり、その行の価格を採用。
    - 色区分なし → (model, capacity, None) で返す。
    """
    html_text = get_html(SHOP4_URL)
    if not html_text:
        return {}

    tree = html.fromstring(html_text)
    out: Dict[Tuple[str, int, Optional[str]], str] = {}

    # TODO: 実際の表構造に合わせてループ/列インデックスを調整
    # 例: テーブルの各行(tr)を走査し、「未開封」スパンが tr[2]/td[1]/span にある想定
    rows = tree.xpath("//table//tr")
    for tr in rows:
        # 「未開封」チェック（指定のXPath）
        unopened = tr.xpath("./td[1]/span/text()")
        if not unopened:
            continue
        if "未開封" not in "".join([t.strip() for t in unopened]):
            continue

        # 例: 同じ行に '機種名 容量' と '価格' がある想定
        title_text = "".join(tr.xpath(".//td[2]//text()")).strip()
        price_text = "".join(tr.xpath(".//td[3]//text()")).strip()

        name = normalize_model_name(title_text)
        cap_match = re.search(r"(\d{3,4})\s*GB", title_text, flags=re.I)
        capacity = normalize_capacity_gb(cap_match.group(1)) if cap_match else None

        price_int, averaged = parse_price_text_to_value_with_flag(price_text)
        price_str = price_to_output_str(price_int, averaged)

        if name and capacity:
            out[(name, capacity, None)] = price_str

    return out

def scrape_shop5() -> Dict[Tuple[str, int, Optional[str]], str]:
    """
    morimori-kaitori (shop5)
    - 新品未開封(SIMフリー)の価格のみ抽出。
    - 色区分はサイト仕様に応じて有無が変わるため、基本は None とし、分かれる場合は色文字列を入れる。
    """
    html_text = get_html(SHOP5_URL)
    if not html_text:
        return {}

    soup = BeautifulSoup(html_text, "lxml")
    out: Dict[Tuple[str, int, Optional[str]], str] = {}

    # TODO: 実HTMLに合わせて調整。以下は例。
    # 例: 商品毎のブロックに「新品 / 中古 / A品 / B品」などが並ぶ。
    for block in soup.select(".price-item, .item, .kaitori-item"):
        title = block.select_one(".title, h3, .item-title")
        if not title:
            continue
        title_text = title.get_text(strip=True)
        name = normalize_model_name(title_text)
        cap_match = re.search(r"(\d{3,4})\s*GB", title_text, flags=re.I)
        capacity = normalize_capacity_gb(cap_match.group(1)) if cap_match else None

        # ラベルに「新品」「未開封」「SIMフリー」等が含まれる要素を優先
        price_node = None
        # 例: ラベル+価格の行を探索
        for row in block.select(".row, .price-row, tr"):
            label = row.get_text(" ", strip=True)
            if "新品" in label and ("未開封" in label or "未開封品" in label) and ("SIMフリー" in label or "SIMフリー" in label):
                price_node = row
                break

        if not price_node:
            # 次善: 「新品」かつ「未開封」だけでも可
            for row in block.select(".row, .price-row, tr"):
                label = row.get_text(" ", strip=True)
                if "新品" in label and ("未開封" in label or "未開封品" in label):
                    price_node = row
                    break

        if not price_node:
            continue

        # 価格抽出（例: 同一row内の .price, td:last-child 等）
        price_text = None
        cand = price_node.select_one(".price, .value, td:last-child")
        if cand:
            price_text = cand.get_text(strip=True)
        else:
            price_text = price_node.get_text(" ", strip=True)

        price_int, averaged = parse_price_text_to_value_with_flag(price_text or "")
        price_str = price_to_output_str(price_int, averaged)

        if name and capacity:
            out[(name, capacity, None)] = price_str

    return out

# -------------------------
# CSV 統合ロジック
# -------------------------

def merge_into_csv(df: pd.DataFrame,
                   shop_data: Dict[Tuple[str, int, Optional[str]], str],
                   shop_col: str) -> pd.DataFrame:
    """
    shop_data を入力CSV df にマージして shop_col に書き込む。
    - キーは (model_name, capacity_gb, color or None)
    - color=None のデータは当該 (model, capacity) の全色行へブロードキャスト
    """
    # 事前正規化
    df["model_name"] = df["model_name"].astype(str).map(normalize_model_name)
    df["capacity_gb"] = df["capacity_gb"].map(normalize_capacity_gb)

    # 1) color が指定されたキーを先に反映（より具体的）
    for (model, cap, color), price in shop_data.items():
        if not price:
            continue
        if color:  # 色指定の行のみ反映
            mask = (
                (df["model_name"] == model) &
                (df["capacity_gb"] == cap) &
                (df["color"].astype(str).str.strip() == str(color).strip())
            )
            df.loc[mask, shop_col] = price

    # 2) color=None のキーは (model, cap) 全色に一括適用（未設定のみ上書き or 常に上書き選択）
    for (model, cap, color), price in shop_data.items():
        if not price or color is not None:
            continue
        mask = (
            (df["model_name"] == model) &
            (df["capacity_gb"] == cap)
        )
        # 既に値が入っている場合は維持したいなら以下で空セルのみ更新:
        # empty_mask = df[shop_col].isna() | (df[shop_col].astype(str).str.strip() == "")
        # df.loc[mask & empty_mask, shop_col] = price
        df.loc[mask, shop_col] = price

    return df

# -------------------------
# メイン処理
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch iPhone 17/Air buyback prices and fill CSV.")
    ap.add_argument("--in", dest="in_csv", required=True, help="入力CSVのパス")
    ap.add_argument("--out", dest="out_csv", required=True, help="出力CSVのパス")
    ap.add_argument("--shops", nargs="*", default=["shop2", "shop3", "shop4", "shop5"],
                    help="実行対象のショップ列 (例: shop2 shop3)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # shop 列が無ければ作成
    for col in ["shop2", "shop3", "shop4", "shop5"]:
        if col not in df.columns:
            df[col] = ""

    if "shop2" in args.shops:
        print("[shop2] fetching...")
        data2 = scrape_shop2()
        df = merge_into_csv(df, data2, "shop2")

    if "shop3" in args.shops:
        print("[shop3] fetching...")
        data3 = scrape_shop3()
        df = merge_into_csv(df, data3, "shop3")

    if "shop4" in args.shops:
        print("[shop4] fetching...")
        data4 = scrape_shop4()
        df = merge_into_csv(df, data4, "shop4")

    if "shop5" in args.shops:
        print("[shop5] fetching...")
        data5 = scrape_shop5()
        df = merge_into_csv(df, data5, "shop5")

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Saved: {args.out_csv}")

if __name__ == "__main__":
    main()
