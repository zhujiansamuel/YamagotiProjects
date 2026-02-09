from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Tuple
from urllib.parse import urlparse
from typing import Dict, Optional, List, Iterable, Union
import os, re, json, pathlib
from datetime import datetime
import pytz
import time
from functools import lru_cache
try:
    import langextract as lx
    _LANGEXTRACT_OK = True
except Exception:
    lx = None
    _LANGEXTRACT_OK = False
SHOP15_OLLAMA_URL_DEFAULT = os.getenv("SHOP15_OLLAMA_URL", "http://localhost:11434")
SHOP15_OLLAMA_MODEL_DEFAULT = os.getenv("SHOP15_OLLAMA_MODEL", "gemma3:1b")
# 基准价只从开头抓（避免把“ブルー229,000円”的229,000误当 base）
_BASE_YEN_AT_START_RE = re.compile(r"^\s*(?:￥|¥|\u00a5)?\s*(\d[\d,]*)\s*円?")


_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)

FAMILY_SYNONYMS = {
    "blue": ["ブルー"],
    "black": ["ブラック", "黒"],
    "white": ["ホワイト", "白"],
    "green": ["グリーン", "緑"],
    "red": ["レッド", "赤"],
    "pink": ["ピンク"],
    "purple": ["パープル", "紫"],
    "yellow": ["イエロー", "黄"],
    "gold": ["ゴールド"],
    "silver": ["シルバー"],
    "gray": ["グレー", "グレイ", "灰"],
    "natural": ["ナチュラル"],
}

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

BASE_YEN_AT_START_RE_shop15 = re.compile(r"^\s*(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

COLOR_DELTA_IN_PRICE_RE_shop15 = re.compile(
    r"""(?P<label>[^\d円¥]+?)\s*             # 标签或标签列表（允许包含 、/／・ 等）
        (?P<sep>[：:\-])? \s*                # 分隔符（可无）
        (?P<sign>[+\-−－])? \s*              # 正负号（可无）
        (?P<amount>\d[\d,]*) \s* (?:円)?     # 金额（可跟円）
    """,
    re.UNICODE | re.VERBOSE,
)



def _parse_signed_int_yen(s: object) -> Optional[int]:
    """
    解析：'229,000' / '229,000円' / '-1000' / '-1,000円' / '+2000円'
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None

    # 统一符号
    t = t.replace("＋", "+").replace("−", "-").replace("－", "-")
    sign = 1
    if t.startswith("+"):
        t = t[1:].strip()
    elif t.startswith("-"):
        sign = -1
        t = t[1:].strip()

    # 去掉非数字/逗号
    t = re.sub(r"[^\d,]", "", t)
    if not t:
        return None
    try:
        return sign * int(t.replace(",", ""))
    except Exception:
        return None

def _extract_base_price_at_start(text: object) -> Optional[int]:
    if text is None:
        return None
    s = str(text)
    m = _BASE_YEN_AT_START_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None

def _clean_label_shop15(label: str) -> str:
    if not label:
        return ""
    s = str(label).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # 去掉可能粘着的分隔符
    s = s.strip(" 　:：-‐‑–—/／、,，・")
    return s

def _iter_extractions_in_order(result) -> List:
    """
    LangExtract 输出顺序在不同 provider 下可能不严格保证，这里尽量按文本位置排序。
    """
    exts = list(getattr(result, "extractions", []) or [])

    def key(e):
        ci = getattr(e, "char_interval", None)
        sp = getattr(ci, "start_pos", None) if ci is not None else None
        if sp is not None:
            return (0, int(sp))
        idx = getattr(e, "extraction_index", None)
        if idx is not None:
            return (1, int(idx))
        return (2, 0)

    return sorted(exts, key=key)

# -----------------------------
# LLM 解析 prompt + few-shot
# -----------------------------
SHOP15_PRICE_PROMPT = (
    "You parse Japanese iPhone buyback 'price' strings.\n"
    "Extract:\n"
    "1) base price (the first yen price at the beginning of the string).\n"
    "   Return ONE extraction:\n"
    "   - extraction_class = \"base_price\"\n"
    "   - extraction_text = exact substring including 円 (e.g., \"230,500円\")\n"
    "   - attributes = {\"yen\": \"230500\"}\n"
    "\n"
    "2) color-specific rules. For each color label, return ONE extraction:\n"
    "   - extraction_class = \"color_spec\"\n"
    "   - extraction_text = exact color label substring (e.g., \"ブルー\")\n"
    "   - attributes must include:\n"
    "       kind: \"delta\" or \"abs\"\n"
    "       yen: integer yen string. For delta use signed string like \"-1000\" or \"2000\". For abs use \"229000\".\n"
    "\n"
    "Rules:\n"
    "- If a color label is followed by +/− amount (e.g., ブルー-1000円, シルバー+2,000円) => kind=\"delta\".\n"
    "- If a color label is followed by a price WITHOUT +/− (e.g., ブルー229,000円, ブルー:229,000円, シルバー 229,000円) => kind=\"abs\".\n"
    "- If multiple color labels are listed before the same amount using separators (、/／・,&), apply the same rule to each label.\n"
    "- Return entities in order of appearance. Use exact text; do not paraphrase.\n"
)

def _shop15_langextract_examples():
    # 延迟构建（避免没装 langextract 时 import 失败）
    return [
        lx.data.ExampleData(
            text="207,000円　オレンジ、ブルー-1000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="207,000円",
                    attributes={"yen": "207000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="オレンジ",
                    attributes={"kind": "delta", "yen": "-1000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "delta", "yen": "-1000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="230,500円　ブルー229,000円　シルバー　229,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="230,500円",
                    attributes={"yen": "230500"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "abs", "yen": "229000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "abs", "yen": "229000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="213,500円　ブルー-9,000円　シルバー-7,500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="213,500円",
                    attributes={"yen": "213500"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "delta", "yen": "-9000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "delta", "yen": "-7500"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="180,000円 シルバー+2,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="180,000円",
                    attributes={"yen": "180000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "delta", "yen": "2000"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="263,000円　ブルー-3,000円　シルバー　-3,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="base_price",
                    extraction_text="263,000円",
                    attributes={"yen": "263000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="ブルー",
                    attributes={"kind": "delta", "yen": "-3000"},
                ),
                lx.data.Extraction(
                    extraction_class="color_spec",
                    extraction_text="シルバー",
                    attributes={"kind": "delta", "yen": "-3000"},
                ),
            ],
        ),
    ]
def _extract_signed_amount_after_label_shop15(price_text: str, label: str) -> Optional[int]:
    """
    在原文中查找: <label> [可选:：] 空白? (+/-) 金额
    如: 'シルバー　-3,000円' / 'ブルー:+2,000円'
    返回 signed int（-3000 / +2000），找不到返回 None。
    """
    if not price_text or not label:
        return None
    s = str(price_text).replace("\u3000", " ")
    lab = _clean_label_shop15(label)
    if not lab:
        return None

    # 找到 label 出现位置（取第一个命中即可）
    idx = s.find(lab)
    if idx < 0:
        return None
    window = s[idx: idx + 40]  # 足够覆盖 "label  -3,000円"

    m = re.match(
        re.escape(lab) + r"\s*(?:[：:])?\s*([+\-−－])\s*(\d[\d,]*)",
        window
    )
    if not m:
        return None

    sign = m.group(1)
    amt = int(m.group(2).replace(",", ""))
    if sign in ("-", "−", "－"):
        amt = -amt
    return amt


def _coerce_specs_shop15(price_text: str, base_price: Optional[int], specs: List[Tuple[str, str, int]], debug: bool = False):
    """
    纠错策略（针对小模型不稳定输出）：
      - kind=abs 且 value<0 => 强制改成 delta
      - 若原文里 label 后出现 +/- 金额 => 强制 delta，并用原文的 signed 金额覆盖 value
    """
    fixed: List[Tuple[str, str, int]] = []
    for (label, kind, value) in specs:
        orig = (label, kind, value)
        kind2, value2 = kind, value

        # 规则1：abs 不应该是负数，直接改为 delta
        if kind2 == "abs" and value2 < 0:
            kind2 = "delta"

        # 规则2：从原文补判（支持“シルバー  -3,000円”这种带空白的写法）
        signed_ctx = _extract_signed_amount_after_label_shop15(price_text, label)
        if signed_ctx is not None:
            kind2 = "delta"
            value2 = int(signed_ctx)

        if _shop15_debug_enabled(debug) and (orig != (label, kind2, value2)):
            print(f"[shop15][lx] coerce spec {orig} -> {(label, kind2, value2)}")

        fixed.append((label, kind2, value2))
    return fixed

# 颜色列表 + 差额 的 block，例如:
# "オレンジ、ブルー-1000円", "シルバー、ブルー-3000円"
MULTI_LABEL_DELTA_BLOCK_RE_shop15 = re.compile(
    r"""
    (?P<label_blob>[^\d円¥]+?)     # 一个或多个颜色标签（可包含 、／/・ 等）
    \s*
    (?P<sign>[+\-−－])            # + 或 -
    \s*
    (?P<amount>\d[\d,]*)          # 金额
    \s*円?
    """,
    re.UNICODE | re.VERBOSE,
)

def _augment_multi_label_block_specs_shop15(
    price_text: str,
    specs: List[Tuple[str, str, int]],
    debug: bool = False,
) -> List[Tuple[str, str, int]]:
    """
    处理形如: 'オレンジ、ブルー-1000円', 'シルバー、ブルー-3000円' 这种“多个颜色共享一个差额”的表达。

    规则：
      - 在 price_text 里用 MULTI_LABEL_DELTA_BLOCK_RE_shop15 找到所有 block
      - label_blob 用 _split_color_labels_shop15 拆成 ['オレンジ','ブルー'] 这类列表
      - 对每个 label:
          * 强制 kind='delta'
          * value = sign (+/-) * amount
          * 若 specs 中已有该 label 的条目 => 覆盖为这个 delta（纠正 LLM）
          * 若 specs 中没有该 label => 新增一条 delta
      - 多个 block 顺序执行，后面的覆盖前面的（和你 _build_color_prices_from_specs_shop15 的逻辑一致）
    """
    if not price_text:
        return specs

    s = str(price_text)
    new_specs: List[Tuple[str, str, int]] = list(specs)

    for m in MULTI_LABEL_DELTA_BLOCK_RE_shop15.finditer(s):
        label_blob = m.group("label_blob") or ""
        sign = m.group("sign")
        amount_str = m.group("amount")

        # 金额解析失败直接跳过
        try:
            amt = int(amount_str.replace(",", ""))
        except Exception:
            continue

        # 根据符号决定正负
        value = -amt if sign in ("-", "−", "－") else amt

        labels = _split_color_labels_shop15(label_blob)

        for lab in labels:
            lab_clean = _clean_label_shop15(lab)
            if not lab_clean:
                continue

            found = False
            for idx, (lbl_old, kind_old, val_old) in enumerate(new_specs):
                if lbl_old == lab_clean:
                    old = new_specs[idx]
                    new_specs[idx] = (lab_clean, "delta", int(value))
                    found = True
                    if _shop15_debug_enabled(debug) and old != new_specs[idx]:
                        print(
                            f"[shop15][lx] multi-label block override "
                            f"{old} -> {new_specs[idx]} from block {m.group(0)!r}"
                        )
                    break

            if not found:
                new_specs.append((lab_clean, "delta", int(value)))
                if _shop15_debug_enabled(debug):
                    print(
                        f"[shop15][lx] multi-label block add "
                        f"{new_specs[-1]} from block {m.group(0)!r}"
                    )

    if _shop15_debug_enabled(debug):
        print(f"[shop15][lx] specs after multi-label augment = {new_specs}")

    return new_specs


@lru_cache(maxsize=4096)
def _parse_shop15_price_via_langextract_cached(
    price_text: str,
    model_id: str,
    model_url: str,
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    """
    返回:
      base_price: Optional[int]
      specs: List[(label, kind, yen_value)]
        kind in {"delta","abs"}
        yen_value: delta 为 signed, abs 为正数
    """
    if not _LANGEXTRACT_OK:
        return None, []

    examples = _shop15_langextract_examples()

    # 低温度尽量稳定
    try:
        result = lx.extract(
            text_or_documents=price_text,
            prompt_description=SHOP15_PRICE_PROMPT,
            examples=examples,
            model_id=model_id,
            model_url=model_url,
            fence_output=False,
            use_schema_constraints=False,
            temperature=0.0,
        )
    except TypeError:
        # 兼容不同版本签名
        result = lx.extract(
            text_or_documents=price_text,
            prompt_description=SHOP15_PRICE_PROMPT,
            examples=examples,
            model_id=model_id,
            model_url=model_url,
            fence_output=False,
            use_schema_constraints=False,
        )

    base_price = None
    specs: List[Tuple[str, str, int]] = []

    for ext in _iter_extractions_in_order(result):
        cls = (getattr(ext, "extraction_class", "") or "").strip().lower()
        txt = (getattr(ext, "extraction_text", "") or "").strip()
        attrs = getattr(ext, "attributes", {}) or {}

        if cls == "base_price":
            yen = attrs.get("yen")
            v = _parse_signed_int_yen(yen if yen is not None else txt)
            if v is not None:
                base_price = int(v)
            continue

        if cls == "color_spec":
            label = _clean_label_shop15(txt)
            if not label:
                continue
            kind = str(attrs.get("kind", "")).strip().lower()
            yen_raw = attrs.get("yen")
            v = _parse_signed_int_yen(yen_raw)
            if v is None:
                # 最后兜底：如果没给 yen，就从 extraction_text 尝试（通常不会）
                v = _parse_signed_int_yen(txt)

            if v is None:
                continue

            if kind not in {"delta", "abs"}:
                # 轻量兜底：有负号/加号更可能是 delta
                ys = str(yen_raw) if yen_raw is not None else ""
                ys = ys.replace("＋", "+").replace("−", "-").replace("－", "-")
                kind = "delta" if ("-" in ys or "+" in ys) else "abs"

            specs.append((label, kind, int(v)))
            continue

    return base_price, specs


def _parse_shop15_price_via_langextract(
    price_text: object,
    model_id: str = SHOP15_OLLAMA_MODEL_DEFAULT,
    model_url: str = SHOP15_OLLAMA_URL_DEFAULT,
    debug: bool = False,
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    if price_text is None:
        return None, []
    s = str(price_text)

    base_price, specs = (None, [])
    try:
        base_price, specs = _parse_shop15_price_via_langextract_cached(s, model_id, model_url)
    except Exception as e:
        if _shop15_debug_enabled(debug):
            print(f"[shop15][lx] ERROR: {e!r} -> fallback base-only")

    # 兜底：LLM 没给 base，就自己从开头 regex 抓
    if base_price is None:
        base_price = _extract_base_price_at_start(s)

    # 第一步：对单 label 的错标做纠偏（abs 负数 -> delta, 从原文补 +/-）
    specs = _coerce_specs_shop15(s, base_price, specs, debug=debug)

    # 第二步：对 "シルバー、ブルー-3000円" / "オレンジ、ブルー-1000円" 这类多颜色 block 做增强/覆盖
    specs = _augment_multi_label_block_specs_shop15(s, specs, debug=debug)

    if _shop15_debug_enabled(debug):
        # print(f"[shop15][lx]    price_raw  = {s!r}")
        # print(f"[shop15][lx]    base_price = {base_price}")
        print(f"[shop15][lx]    specs      = {specs}")

    return base_price, specs


def _build_color_prices_from_specs_shop15(
    color_map: Dict[str, Tuple[str, str]],
    base_price: Optional[int],
    specs: List[Tuple[str, str, int]],
    debug: bool = False,
) -> Tuple[Dict[str, int], List[Tuple[str, str, str, int]], List[Tuple[str, str, int]]]:
    """
    specs: [(label, kind, value)]
    kind = "abs"  => final_price = value
    kind = "delta"=> final_price = base + value
    未命中颜色：
      - 有 base => base
      - 无 base => 不产出该颜色价格（避免误写）
    """
    hit_log: List[Tuple[str, str, str, int]] = []
    unmatched = list(specs)

    color_prices: Dict[str, int] = {}
    if base_price is not None:
        for col_norm in color_map.keys():
            color_prices[col_norm] = int(base_price)

    for (label, kind, value) in specs:
        matched_any = False
        for col_norm, (_pn, col_raw) in color_map.items():
            if _label_matches_color(label, col_raw, col_norm):
                matched_any = True
                hit_log.append((label, col_raw, kind, int(value)))
                if kind == "abs":
                    color_prices[col_norm] = int(value)
                else:
                    if base_price is None:
                        continue
                    color_prices[col_norm] = int(base_price + int(value))

        if matched_any:
            for idx, t in enumerate(unmatched):
                if t == (label, kind, value):
                    unmatched.pop(idx)
                    break

    if _shop15_debug_enabled(debug):
        print(f"[shop15][price] hit_log    = {hit_log}")
        if unmatched:
            print(f"[shop15][price] unmatched_specs={unmatched}")

    return color_prices, hit_log, unmatched

# 建议加在模块里（任意位置，只要在 clean_shop15 调用前即可）
def _shop15_debug_enabled(debug: bool = False) -> bool:
    if debug:
        return True
    v = os.getenv("SHOP15_DEBUG", "")
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _norm(s: str) -> str:
    return (s or "").strip()

def _load_iphone17_info_df_for_shop2() -> pd.DataFrame:
    """
    读取 iphone17_info，并尽量保留 jan 列以供 shop1 做 JAN→PN 映射。
    输出列至少包含：part_number, model_name, capacity_gb, color，
    若检测到任何 jan 列，则额外返回标准化列 'jan'。
    """
    try:
        from django.conf import settings
        p = getattr(settings, "EXTERNAL_IPHONE17_INFO_PATH", None)
        if p:
            path = str(p)
        else:
            raise AttributeError
    except Exception:
        path = os.getenv("IPHONE17_INFO_CSV") or str(Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv")

    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError(f"未找到 iphone17_info：{pth}")

    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(pth), re.I):
        df = pd.read_excel(pth)
    else:
        df = pd.read_csv(pth, encoding="utf-8-sig")

    need = {"part_number", "model_name", "capacity_gb", "color"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"iphone17_info 缺少必要列：{missing}")

    df = df.copy()
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number", "color"])

    # ★ 检测并标准化 jan 列（尽最大可能适配命名）
    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):  # 兼容 'JANコード' 等
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))  # 去重保序

    cols = ["part_number", "model_name", "capacity_gb", "color"]
    if jan_candidates:
        src = jan_candidates[0]
        df["jan"] = df[src]
        cols.append("jan")

    return df[cols]

def _normalize_model_generic(text: str) -> str:
    """
    统一型号主体：
      - iPhone17/16 + 后缀（Pro/Pro Max/Plus/mini）
      - iPhone Air（含“17 air”→ Air）
      - 允许紧凑写法：17pro / 17promax / 16Pro / 16Plus ...
    输出：'iPhone 17 Pro Max' / 'iPhone 17 Pro' / 'iPhone Air' / ...
    """
    if not text:
        return ""

def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    宽松匹配一个 'label_raw' 是否命中某个颜色（color_raw / color_norm）。
    优先：
      - 归一化后完全相等
      - label_raw 子串包含于 color_raw
      - 英文族名（如 Blue）映射到日文家族词，并判断是否是 color_raw 的子串
    """
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    key = label_raw.strip().lower()
    if key in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[key]:
            if jp in str(color_raw):
                return True
    # 也尝试 label_norm 的英文键
    if label_norm in FAMILY_SYNONYMS:
        for jp in FAMILY_SYNONYMS[label_norm]:
            if jp in str(color_raw):
                return True
    return False

FIRST_YEN_RE_shop15 = re.compile(r"(\d[\d,]*)\s*円")  # 抓取 price 中第一个 “N円”（作为基准价）

# 允许 label 里包含 "、/／・" 等，整体抓出来后再拆分
COLOR_ENTRY_RE_shop15 = re.compile(
    r"""(?P<label>[^\d円¥]+?)\s*             # 标签或标签列表（可含 、/／・）
        (?P<sep>[：:\-])? \s*                # 可选分隔符
        (?P<sign>[+\-−－])? \s*              # 可选正负号
        (?P<amount>\d[\d,]*) \s* (?:円)?     # 金额
    """,
    re.UNICODE | re.VERBOSE,
)

_LABEL_LIST_SPLIT_RE_shop15 = re.compile(r"\s*(?:、|,|，|／|/|・|&|＆)\s*")

def _extract_base_price_shop15(text: str, debug: bool = False) -> Optional[int]:
    if not text:
        return None
    s = str(text)
    m = BASE_YEN_AT_START_RE_shop15.search(s)
    if _shop15_debug_enabled(debug):
        match = m.group(0) if m else None
        print(f"[shop15][base] raw={s!r}")
        print(f"[shop15][base] START_BASE match={match!r}")
    if not m:
        return None
    val = to_int_yen(m.group(1))
    if _shop15_debug_enabled(debug):
        print(f"[shop15][base] base extracted -> {val}")
    return int(val) if val is not None else None

def _split_color_labels_shop15(label_blob: str) -> List[str]:
    if not label_blob:
        return []
    s = str(label_blob).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" 　:：-‐‑–—/／、,，・")
    parts = [p.strip() for p in _LABEL_LIST_SPLIT_RE_shop15.split(s) if p.strip()]
    return parts or [s]
def _extract_base_price(text: str, debug: bool = False) -> Optional[int]:
    if not text:
        if _shop15_debug_enabled(debug):
            print("[shop15][base] text is empty -> None")
        return None

    s = str(text)
    m = FIRST_YEN_RE.search(s)
    if _shop15_debug_enabled(debug):
        print(f"[shop15][base] raw={s!r}")
        print(f"[shop15][base] FIRST_YEN_RE match={(m.group(0) if m else None)!r}")

    if not m:
        val = to_int_yen(s)
        if _shop15_debug_enabled(debug):
            print(f"[shop15][base] fallback to_int_yen -> {val}")
        return val

    val = to_int_yen(m.group(1))
    if _shop15_debug_enabled(debug):
        print(f"[shop15][base] base extracted -> {val}")
    return val

def _extract_color_deltas_from_price(text: str, debug: bool = False) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not text:
        if _shop15_debug_enabled(debug):
            print("[shop15][delta] text is empty -> []")
        return out

    s = str(text)
    m0 = FIRST_YEN_RE_shop15.search(s)
    tail = s[m0.end():] if m0 else s

    if _shop15_debug_enabled(debug):
        print(f"[shop15][delta] raw={s!r}")
        print(f"[shop15][delta] base_match={repr(m0.group(0)) if m0 else None}")
        print(f"[shop15][delta] tail_for_parse={tail!r}")

    for m in COLOR_DELTA_IN_PRICE_RE_shop15.finditer(tail):
        label_blob = (m.group("label") or "").strip()
        if not label_blob:
            continue

        sep = m.group("sep")
        sign = m.group("sign")
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            if _shop15_debug_enabled(debug):
                print(f"[shop15][delta] hit={m.group(0)!r} but amount parse failed -> skip")
            continue

        if sign:
            negative = sign in ("-", "−", "－")
        else:
            negative = sep in ("-", "−", "－") if sep else False

        delta = -int(amt) if negative else int(amt)

        labels = _split_color_labels_shop15(label_blob)
        for lab in labels:
            out.append((lab, delta))

        if _shop15_debug_enabled(debug):
            print(
                f"[shop15][delta] hit={m.group(0)!r} | label_blob={label_blob!r} -> labels={labels} "
                f"| sep={sep!r}, sign={sign!r}, amount={m.group('amount')!r} -> delta={delta}"
            )

    if _shop15_debug_enabled(debug):
        print(f"[shop15][delta] extracted labels_and_deltas={out}")
    return out

def _extract_color_specs_from_price_shop15(text: str, debug: bool = False) -> List[Tuple[str, str, int]]:
    """
    返回: [(label, kind, value), ...]
      kind = "delta" 表示差额（相对 base）
      kind = "abs"   表示绝对价（直接覆盖）
    规则：
      - 有 sign（+/-）=> delta
      - 或 sep 为 '-' 且无 sign => delta（负号）
      - 其他（sep None / ':' / '：' 且无 sign）=> abs
    """
    out: List[Tuple[str, str, int]] = []
    if not text:
        return out

    s = str(text)
    m0 = BASE_YEN_AT_START_RE_shop15.search(s)
    tail = s[m0.end():] if m0 else s  # base 存在就跳过 base 部分

    if _shop15_debug_enabled(debug):
        print(f"[shop15][spec] raw={s!r}")
        print(f"[shop15][spec] tail_for_parse={tail!r}")

    for m in COLOR_ENTRY_RE_shop15.finditer(tail):
        label_blob = (m.group("label") or "").strip()
        if not label_blob:
            continue

        sep = m.group("sep")
        sign = m.group("sign")
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        amt = int(amt)

        # 判定 delta or abs
        if sign:
            kind = "delta"
            negative = sign in ("-", "−", "－")
            value = -amt if negative else amt
        elif sep in ("-", "−", "－"):
            kind = "delta"
            value = -amt
        else:
            kind = "abs"
            value = amt

        labels = _split_color_labels_shop15(label_blob)
        for lab in labels:
            out.append((lab, kind, value))

        if _shop15_debug_enabled(debug):
            print(f"[shop15][spec] hit={m.group(0)!r} | labels={labels} | sep={sep!r}, sign={sign!r} -> kind={kind}, value={value}")

    if _shop15_debug_enabled(debug):
        print(f"[shop15][spec] extracted specs={out}")
    return out



def _build_color_deltas_shop15(
    color_map: Dict[str, Tuple[str, str]],
    labels_and_deltas: List[Tuple[str, int]],
) -> Tuple[Dict[str, int], List[Tuple[str, str, int]], List[str]]:
    """
    return:
      - color_deltas: {color_norm: delta}
      - hit_log: [(label_raw, color_raw, delta), ...]
      - unmatched_labels: [label_raw, ...]
    """
    color_deltas: Dict[str, int] = {}
    hit_log: List[Tuple[str, str, int]] = []
    unmatched = {lbl for (lbl, _) in labels_and_deltas}

    for col_norm, (pn, col_raw) in color_map.items():
        for label_raw, delta in labels_and_deltas:
            if _label_matches_color(label_raw, col_raw, col_norm):
                color_deltas[col_norm] = delta
                hit_log.append((label_raw, col_raw, delta))
                unmatched.discard(label_raw)

    return color_deltas, hit_log, sorted(unmatched)

def _build_color_map_shop15(info_df: pd.DataFrame) -> Dict[tuple, Dict[str, Tuple[str, str]]]:
    """
    (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }
    """
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df["color_norm"] = df["color"].map(lambda x: _norm(str(x)))
    cmap: Dict[tuple, Dict[str, Tuple[str, str]]] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]; cap = r["capacity_gb"]
        if not m or pd.isna(cap):
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        cmap[key][_norm(str(r["color"]))] = (str(r["part_number"]), str(r["color"]))
    return cmap

def _build_color_prices_shop15(
    color_map: Dict[str, Tuple[str, str]],
    base_price: Optional[int],
    specs: List[Tuple[str, str, int]],
) -> Tuple[Dict[str, int], List[Tuple[str, str, str, int]], List[Tuple[str, str, int]]]:
    """
    return:
      - color_prices: {color_norm: final_price}
      - hit_log: [(label, color_raw, kind, value), ...]
      - unmatched_specs: [(label, kind, value), ...]
    逻辑：
      - abs 命中 => final_price 直接覆盖
      - delta 命中 => base_price + delta（要求 base_price 非空）
      - 未命中：如果 base_price 有 => final_price = base_price；否则不生成该颜色价格
    """
    hit_log: List[Tuple[str, str, str, int]] = []
    unmatched = specs[:]

    # 先初始化为 base（如果有）
    color_prices: Dict[str, int] = {}
    if base_price is not None:
        for col_norm in color_map.keys():
            color_prices[col_norm] = int(base_price)

    # 应用 specs（按出现顺序；后者覆盖前者）
    for label, kind, value in specs:
        matched_any = False
        for col_norm, (_pn, col_raw) in color_map.items():
            if _label_matches_color(label, col_raw, col_norm):
                matched_any = True
                hit_log.append((label, col_raw, kind, value))

                if kind == "abs":
                    color_prices[col_norm] = int(value)
                else:
                    if base_price is None:
                        # 没 base 时无法应用 delta：跳过
                        continue
                    color_prices[col_norm] = int(base_price + int(value))

        if matched_any:
            # 从 unmatched 移除一个对应项（可能重复，按值移除一个）
            for idx, t in enumerate(unmatched):
                if t == (label, kind, value):
                    unmatched.pop(idx)
                    break

    return color_prices, hit_log, unmatched


def clean_shop15(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    debug = _shop15_debug_enabled(debug)

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"shop15:買取当番---------->进入清洗器时间: {now}")

    for c in ["price", "data2", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop15 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop15(info_df)

    rows: List[dict] = []

    for i, row in df.iterrows():
        print("----------------------------------------------------------")
        model_text = str(row.get("data2") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None:
            if debug:
                print(f"[shop15][row {i}] skip: model/cap parse failed | data2={model_text!r}")
            continue

        key = (model_norm, int(cap_gb))
        color_map = cmap_all.get(key)
        if not color_map:
            if debug:
                print(f"[shop15][row {i}] skip: no color_map for key={key} | data2={model_text!r}")
            continue

        price_text = row.get("price")
        if debug:
            print(f"[shop15][row {i}] data2      = {model_text!r}")
            # print(f"[shop15][row {i}] parsed -> model_norm={model_norm!r}, cap_gb={int(cap_gb)}")
            print(f"[shop15][row {i}] price_raw  = {str(price_text)!r}")
            # print(f"[shop15][row {i}] known_colors_in_info={[cr for (_, cr) in color_map.values()]}")

        # 关键：用 LangExtract + Ollama 解析 price
        base_price, specs = _parse_shop15_price_via_langextract(
            price_text,
            model_id=SHOP15_OLLAMA_MODEL_DEFAULT,
            model_url=SHOP15_OLLAMA_URL_DEFAULT,
            debug=debug,
        )

        # 应用到 “信息表里存在的所有颜色”
        color_prices, hit_log, unmatched_specs = _build_color_prices_from_specs_shop15(
            color_map=color_map,
            base_price=base_price,
            specs=specs,
            debug=debug,
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        # 产出行：未命中颜色 => base（若 base 存在）；否则跳过
        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm not in color_prices:
                continue
            rows.append(
                {
                    "part_number": str(pn),
                    "shop_name": "買取当番",
                    "price_new": int(color_prices[col_norm]),
                    "recorded_at": rec_at,
                }
            )

        if debug:
            preview = []
            for col_norm, (pn, col_raw) in color_map.items():
                if col_norm in color_prices:
                    preview.append((col_raw, pn, color_prices[col_norm]))
            print(f"[shop15][row {i}] final      = {preview}")

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if out.empty:
        return out

    out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
    out["part_number"] = out["part_number"].astype(str)
    out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    # “有历史则更新”：同一 (part_number, shop_name) 只保留最新 recorded_at
    out = (
        out.sort_values(["part_number", "shop_name", "recorded_at"])
          .drop_duplicates(subset=["part_number", "shop_name"], keep="last")
          .reset_index(drop=True)
    )

    return out
