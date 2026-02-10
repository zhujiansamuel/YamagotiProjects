from __future__ import annotations

"""
shop15 清洗器 — 買取当番

  原始文本（price 列）
    │
    ├─ _extract_base_price_at_start()             ← Step 2: 提取基础价
    │
    ├─ _extract_price_parts_shop15_dispatch()      ← Step 9: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_price_parts_shop15_regex()       ← Step 6: 正则提取 specs
    │   │
    │   └─ llm 路径:
    │       ├─ _parse_shop15_price_via_langextract()     ← Step 7: LLM 核心提取
    │       └─ _coerce_specs / _augment_multi_label      ← Step 8: 纠错/增强
    │
    ├─ _build_color_prices_from_specs_shop15()     ← Step 10: specs → 最终价格
    │
    ├─ _label_matches_color()                      ← Step 4: 标签→颜色匹配
    │
    └─ clean_shop15()                              ← Step 11: 主函数，生成输出行
"""

import logging
import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
)

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

try:
    import langextract as lx
    _LANGEXTRACT_OK = True
except Exception:
    lx = None
    _LANGEXTRACT_OK = False

SHOP15_OLLAMA_URL_DEFAULT = os.getenv("SHOP15_OLLAMA_URL", "http://localhost:11434")
SHOP15_OLLAMA_MODEL_DEFAULT = os.getenv("SHOP15_OLLAMA_MODEL", "gemma3:1b")

SHOP15_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

MODEL_COL = "data2"
PRICE_COL = "price"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

def _truncate_for_log(s: str, n: int = 200) -> str:
    """截断长字符串，保留前 n 个字符，用于日志显示"""
    if s is None:
        return ""
    t = str(s)
    if len(t) <= n:
        return t
    return t[:n] + f"... (truncated, total_length={len(t)})"

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

# ----------------------------------------------------------------------
# Step 2: 基础价提取
# ----------------------------------------------------------------------

# 基准价只从开头抓（避免把"ブルー229,000円"的229,000误当 base）
_BASE_YEN_AT_START_RE = re.compile(r"^\s*(?:￥|¥|\u00a5)?\s*(\d[\d,]*)\s*円?")

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

# ----------------------------------------------------------------------
# Step 3: 标签归一化 & 拆分
# ----------------------------------------------------------------------

def _norm(s: str) -> str:
    return (s or "").strip()

def _clean_label_shop15(label: str) -> str:
    if not label:
        return ""
    s = str(label).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # 去掉可能粘着的分隔符
    s = s.strip(" 　:：-‐‑–—/／、,，・")
    return s

_LABEL_LIST_SPLIT_RE_shop15 = re.compile(r"\s*(?:、|,|，|／|/|・|&|＆)\s*")

def _split_color_labels_shop15(label_blob: str) -> List[str]:
    if not label_blob:
        return []
    s = str(label_blob).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" 　:：-‐‑–—/／、,，・")
    parts = [p.strip() for p in _LABEL_LIST_SPLIT_RE_shop15.split(s) if p.strip()]
    return parts or [s]

# ----------------------------------------------------------------------
# Step 4: 颜色家族同义词 & 匹配
# ----------------------------------------------------------------------

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

def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    宽松匹配：精确(归一) | 原文子串 | 英文族名→日文家族词
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

# ----------------------------------------------------------------------
# Step 5: 正则模式定义
# ----------------------------------------------------------------------

BASE_YEN_AT_START_RE_shop15 = re.compile(r"^\s*(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")

# 允许 label 里包含 "、/／・" 等，整体抓出来后再拆分
COLOR_ENTRY_RE_shop15 = re.compile(
    r"""(?P<label>[^\d円¥]+?)\s*             # 标签或标签列表（可含 、/／・）
        (?P<sep>[：:\-])? \s*                # 可选分隔符
        (?P<sign>[+\-−－])? \s*              # 可选正负号
        (?P<amount>\d[\d,]*) \s* (?:円)?     # 金额
    """,
    re.UNICODE | re.VERBOSE,
)

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

# ----------------------------------------------------------------------
# Step 6: 正则提取函数
# ----------------------------------------------------------------------

def _extract_price_parts_shop15_regex(
    price_text: str,
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    """
    纯正则版：从 price_text 中提取 (base_price, specs)。
    specs = [(label, kind, value)]  kind ∈ {"delta", "abs"}
    """
    if not price_text:
        return None, []

    s = str(price_text).replace("\u3000", " ")
    base_price = _extract_base_price_at_start(s)

    # 跳过开头的基础价部分
    m0 = _BASE_YEN_AT_START_RE.search(s)
    tail = s[m0.end():] if m0 else s

    if not tail.strip():
        return base_price, []

    specs: List[Tuple[str, str, int]] = []

    for m in COLOR_ENTRY_RE_shop15.finditer(tail):
        label_blob = m.group("label") or ""
        sep = m.group("sep")
        sign = m.group("sign")
        amount_str = m.group("amount")

        amt = _parse_signed_int_yen(amount_str)
        if amt is None:
            continue
        amt = abs(amt)

        # 判断 delta vs abs
        if sign:
            negative = sign in ("-", "−", "－")
            kind = "delta"
            value = -amt if negative else amt
        elif sep and sep in ("-", "−", "－"):
            kind = "delta"
            value = -amt
        else:
            # 无符号 → abs（如 "ブルー229,000円"）
            kind = "abs"
            value = amt

        # 拆分 label blob（如 "オレンジ、ブルー"）
        labels = _split_color_labels_shop15(label_blob)
        for lab in labels:
            lab_clean = _clean_label_shop15(lab)
            if lab_clean:
                specs.append((lab_clean, kind, value))

    return base_price, specs

# ----------------------------------------------------------------------
# Step 7: LLM 配置 & 核心提取函数
# ----------------------------------------------------------------------

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
                # 最后兜底：如果没给 yen，就从 extraction_text 尝试
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

# ----------------------------------------------------------------------
# Step 8: LLM 纠错 & 增强（仅 LLM 路径使用）
# ----------------------------------------------------------------------

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

def _coerce_specs_shop15(
    price_text: str, base_price: Optional[int],
    specs: List[Tuple[str, str, int]],
) -> List[Tuple[str, str, int]]:
    """
    纠错策略（针对小模型不稳定输出）：
      - kind=abs 且 value<0 => 强制改成 delta
      - 若原文里 label 后出现 +/- 金额 => 强制 delta，并用原文的 signed 金额覆盖 value
    """
    fixed: List[Tuple[str, str, int]] = []
    for (label, kind, value) in specs:
        kind2, value2 = kind, value

        # 规则1：abs 不应该是负数，直接改为 delta
        if kind2 == "abs" and value2 < 0:
            kind2 = "delta"

        # 规则2：从原文补判（支持"シルバー  -3,000円"这种带空白的写法）
        signed_ctx = _extract_signed_amount_after_label_shop15(price_text, label)
        if signed_ctx is not None:
            kind2 = "delta"
            value2 = int(signed_ctx)

        fixed.append((label, kind2, value2))
    return fixed

def _augment_multi_label_block_specs_shop15(
    price_text: str,
    specs: List[Tuple[str, str, int]],
) -> List[Tuple[str, str, int]]:
    """
    处理形如: 'オレンジ、ブルー-1000円', 'シルバー、ブルー-3000円' 这种"多个颜色共享一个差额"的表达。

    规则：
      - 在 price_text 里用 MULTI_LABEL_DELTA_BLOCK_RE_shop15 找到所有 block
      - label_blob 用 _split_color_labels_shop15 拆成 ['オレンジ','ブルー'] 这类列表
      - 对每个 label:
          * 强制 kind='delta'
          * value = sign (+/-) * amount
          * 若 specs 中已有该 label 的条目 => 覆盖为这个 delta（纠正 LLM）
          * 若 specs 中没有该 label => 新增一条 delta
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
                    new_specs[idx] = (lab_clean, "delta", int(value))
                    found = True
                    break

            if not found:
                new_specs.append((lab_clean, "delta", int(value)))

    return new_specs

def _extract_price_parts_shop15_llm_with_guardrails(
    price_text: str, idx: object = None,
) -> Tuple[Optional[int], List[Tuple[str, str, int]]]:
    """
    LLM 提取 + 纠错（coerce + augment）。仅 LLM 路径使用。
    返回 (base_price, specs)。
    """
    base_price, specs = None, []
    llm_ok = False

    try:
        base_price, specs = _parse_shop15_price_via_langextract_cached(
            str(price_text),
            SHOP15_OLLAMA_MODEL_DEFAULT,
            SHOP15_OLLAMA_URL_DEFAULT,
        )
        llm_ok = True
    except Exception as e:
        logger.warning(
            "LangExtract extraction failed",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": "買取当番",
                "cleaner_name": "shop15",
                "error": str(e),
                "error_type": type(e).__name__,
                "model_id": SHOP15_OLLAMA_MODEL_DEFAULT,
                "model_url": SHOP15_OLLAMA_URL_DEFAULT,
                "row_index": idx,
                "text_length": len(price_text),
                "text_preview": _truncate_for_log(price_text, 100),
            }
        )

    # 兜底：LLM 没给 base，就自己从开头 regex 抓
    if base_price is None:
        base_price = _extract_base_price_at_start(price_text)

    # 纠错1：对单 label 的错标做纠偏（abs 负数 -> delta, 从原文补 +/-）
    specs = _coerce_specs_shop15(price_text, base_price, specs)

    # 纠错2：对 "シルバー、ブルー-3000円" 这类多颜色 block 做增强/覆盖
    specs = _augment_multi_label_block_specs_shop15(price_text, specs)

    # LLM 完全失败且无 specs 时，回退到正则
    if (not llm_ok) and (not specs):
        _, specs = _extract_price_parts_shop15_regex(price_text)

    return base_price, specs

# ----------------------------------------------------------------------
# Step 9: 提取模式调度
# ----------------------------------------------------------------------

def _extract_price_parts_shop15_dispatch(
    price_text: str, idx: object = None,
) -> Tuple[Optional[int], List[Tuple[str, str, int]], str]:
    """
    根据 SHOP15_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + 纠错
      - "auto":  正则优先，正则无 specs 时 LLM 兜底

    返回 (base_price, specs, extraction_method)
    """
    mode = SHOP15_EXTRACTION_MODE

    if mode == "regex":
        bp, specs = _extract_price_parts_shop15_regex(price_text)
        return bp, specs, "regex"

    if mode == "llm":
        bp, specs = _extract_price_parts_shop15_llm_with_guardrails(
            price_text, idx=idx,
        )
        return bp, specs, "llm"

    # ---- auto: 正则优先，正则无 specs 时 LLM 兜底 ----
    bp_re, specs_re = _extract_price_parts_shop15_regex(price_text)
    if specs_re:
        return bp_re, specs_re, "regex"

    bp_llm, specs_llm = _extract_price_parts_shop15_llm_with_guardrails(
        price_text, idx=idx,
    )
    # LLM 的 base_price 优先，其次正则的
    bp_final = bp_llm if bp_llm is not None else bp_re
    return bp_final, specs_llm, "llm"

# ----------------------------------------------------------------------
# Step 10: specs → 最终颜色价格
# ----------------------------------------------------------------------

def _build_color_prices_from_specs_shop15(
    color_map: Dict[str, Tuple[str, str]],
    base_price: Optional[int],
    specs: List[Tuple[str, str, int]],
) -> Tuple[Dict[str, int], List[Tuple[str, str, str, int]], List[Tuple[str, str, int]]]:
    """
    specs: [(label, kind, value)]
    kind = "abs"  => final_price = value
    kind = "delta"=> final_price = base + value
    未命中颜色：
      - 有 base => base
      - 无 base => 不产出该颜色价格（避免误写）

    返回: (color_prices, hit_log, unmatched_specs)
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
            for i, t in enumerate(unmatched):
                if t == (label, kind, value):
                    unmatched.pop(i)
                    break

    return color_prices, hit_log, unmatched

# ----------------------------------------------------------------------
# Step 11: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop15(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop15 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息
    CLEANER_NAME = "shop15"
    SHOP_NAME = "買取当番"

    logger.info(
        "Starting shop15 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    for c in [PRICE_COL, MODEL_COL, "time-scraped"]:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "missing_column": c,
                    "available_columns": list(df.columns),
                }
            )
            raise ValueError(f"shop15 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

    for i, row in df.iterrows():
        current_row_records: List[dict] = []
        model_text = str(row.get(MODEL_COL) or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None:
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        price_text = row.get(PRICE_COL)
        price_text_s = "" if price_text is None else str(price_text)

        # 根据 SHOP15_EXTRACTION_MODE 提取价格信息
        base_price, specs, extraction_method = _extract_price_parts_shop15_dispatch(
            price_text_s, idx=i,
        )

        # DEBUG: 记录提取结果
        available_colors_list = [
            {"color_norm": cn, "part_number": pn, "color_raw": cr}
            for cn, (pn, cr) in color_map.items()
        ]

        _log_seq += 1
        logger.debug(
            "Extraction result",
            extra={
                "event_type": "extraction_result",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(i),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "price_text_raw": _truncate_for_log(price_text_s, 200),
                "price_text_raw_full": price_text_s,
                "extraction_method": extraction_method,
                "specs": [
                    {"label": label, "kind": kind, "value": value}
                    for label, kind, value in specs
                ],
                "specs_count": len(specs),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_map),
            }
        )

        # 应用 specs 到颜色映射
        color_prices, hit_log, unmatched_specs = _build_color_prices_from_specs_shop15(
            color_map=color_map,
            base_price=base_price,
            specs=specs,
        )

        # DEBUG: label 匹配日志
        for (label, col_raw_hit, kind, value) in hit_log:
            _log_seq += 1
            logger.debug(
                f"Label matched: {label} -> {col_raw_hit}",
                extra={
                    "event_type": "label_matching",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(i),
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "base_price": base_price,
                    "label": label,
                    "color_raw_hit": col_raw_hit,
                    "kind": kind,
                    "value": value,
                    "price_text_raw_full": price_text_s,
                }
            )

        # WARNING: 未匹配到任何颜色的 specs
        for (label, kind, value) in unmatched_specs:
            _log_seq += 1
            logger.warning(
                f"Label not matched: {label}",
                extra={
                    "event_type": "label_no_match",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(i),
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "label": label,
                    "kind": kind,
                    "value": value,
                    "available_colors": [cn for cn in color_map.keys()],
                    "price_text_raw_full": price_text_s,
                }
            )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        # 生成输出记录
        output_records = []
        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm not in color_prices:
                continue

            final_price = int(color_prices[col_norm])
            delta = final_price - base_price if base_price is not None else 0
            delta_source = "spec_matched" if any(
                _label_matches_color(lab, col_raw, col_norm) for (lab, _, _) in specs
            ) else "default_base"

            # DEBUG: 每条输出记录
            _log_seq += 1
            logger.debug(
                f"Output record: {pn}",
                extra={
                    "event_type": "output_record",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(i),
                    "model_text": model_text,
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "part_number": pn,
                    "color_norm": col_norm,
                    "color_raw": col_raw,
                    "base_price": base_price,
                    "delta": delta,
                    "final_price": final_price,
                    "delta_source": delta_source,
                    "recorded_at": str(rec_at) if rec_at else None,
                    "price_text_raw_full": price_text_s,
                }
            )

            output_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": delta,
                "final_price": final_price,
                "delta_source": delta_source,
            })

            rows.append({
                "part_number": str(pn),
                "shop_name": SHOP_NAME,
                "price_new": final_price,
                "recorded_at": rec_at,
            })

            current_row_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": delta,
                "final_price": final_price,
                "recorded_at": rec_at,
                "delta_source": delta_source,
            })

        # DEBUG: 行级详细汇总
        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(i),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "price_text_raw_full": price_text_s,
                "current_row_records": [
                    {"pn": r["part_number"], "color": r["color_norm"], "delta": r["delta"], "final_price": r["final_price"], "src": r["delta_source"]}
                    for r in current_row_records
                ],
            }
        )

        # INFO: 行级概览（简洁）
        specs_matched = len(hit_log)
        all_deltas = [cp - base_price for cp in color_prices.values() if base_price is not None]

        _log_seq += 1
        logger.info(
            f"Row {i:<3d} | {model_text:<28s} | specs: {len(specs):<2d} | matched: {specs_matched:<2d} | records: {len(output_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(i),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "price_text_raw_preview": _truncate_for_log(price_text_s, 100),
                "extraction_method": extraction_method,
                "specs_count": len(specs),
                "specs_matched_count": specs_matched,
                "unmatched_count": len(unmatched_specs),
                "colors_in_catalog": len(color_map),
                "output_records_count": len(output_records),
                "has_discounted_colors": any(d != 0 for d in all_deltas),
                "min_delta": min(all_deltas) if all_deltas else 0,
                "max_delta": max(all_deltas) if all_deltas else 0,
            }
        )

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if out.empty:
        elapsed_time = time.time() - start_time
        logger.info(
            "Shop15 cleaner completed (empty)",
            extra={
                "event_type": "cleaner_complete",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "input_rows": len(df),
                "output_records": 0,
                "elapsed_seconds": round(elapsed_time, 2),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        )
        return out

    out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
    out["part_number"] = out["part_number"].astype(str)
    out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    # "有历史则更新"：同一 (part_number, shop_name) 只保留最新 recorded_at
    out = (
        out.sort_values(["part_number", "shop_name", "recorded_at"])
          .drop_duplicates(subset=["part_number", "shop_name"], keep="last")
          .reset_index(drop=True)
    )

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop15 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "output_records": len(out),
            "elapsed_seconds": round(elapsed_time, 2),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    return out
