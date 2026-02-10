"""
shop14_cleaner  —  買取楽園

数据处理流程:
  raw DataFrame
    │
    ├─ Step 1  列校验 & remark列解析
    ├─ Step 2  行级过滤（未開封 + model/cap/color_map 匹配）
    ├─ Step 3  base_price 提取
    ├─ Step 4  remark文本归一化（3列合并）
    ├─ Step 5  价格规则抽取 dispatch（regex / llm / auto）
    │           ├─ regex路径: _extract_rules_shop14_regex()
    │           └─ llm路径:   _extract_rules_shop14_llm_with_guardrails()
    ├─ Step 6  全色处理（all_delta 快捷路径）
    ├─ Step 7  label → color 匹配（家族同义词）
    ├─ Step 8  价格计算（abs优先 > base+delta > base）
    └─ Step 9  输出 DataFrame 组装
"""
from __future__ import annotations

import logging
import os
import re
import time
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: 配置常量
# ---------------------------------------------------------------------------
SHOP14_EXTRACTION_MODE = os.getenv("SHOP14_EXTRACTION_MODE", "auto")  # "regex" | "llm" | "auto"

SHOP14_OLLAMA_URL = os.getenv("SHOP14_OLLAMA_URL", "http://localhost:11434")
SHOP14_LLM_MODEL_ID = os.getenv("SHOP14_LLM_MODEL_ID", "gemma3:1b")

# ---------------------------------------------------------------------------
# Step 2: 文本归一化 helpers
# ---------------------------------------------------------------------------

_norm = _norm_strip


def _norm_label(lbl: str) -> str:
    """去除空白并统一全角空格/NBSP，保留原文字顺序用作匹配用 key"""
    if lbl is None:
        return ""
    s = str(lbl)
    s = s.strip().replace("\u3000", " ").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_remark_frag(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.lstrip("\ufeff").replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_colname(x) -> str:
    s = str(x or "")
    s = s.lstrip("\ufeff")
    s = s.replace("\u3000", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _split_labels(labels: str) -> List[str]:
    s = str(labels or "").strip()
    if not s:
        return []
    parts = re.split(r"[／/、，,;；\s]+", s)
    return [p.strip() for p in parts if p and p.strip()]


def _coerce_amount_yen(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return int(v)
        except Exception:
            return None

    s = str(v).strip()
    if not s:
        return None

    sign = 1
    if s[:1] in {"+", "＋"}:
        s = s[1:].strip()
    elif s[:1] in {"-", "−", "－"}:
        sign = -1
        s = s[1:].strip()

    n = to_int_yen(s)
    if n is None:
        s2 = re.sub(r"[^\d]", "", s)
        if not s2:
            return None
        try:
            n = int(s2)
        except Exception:
            return None

    return sign * int(n)


def _labels_from_text_fallback(extraction_text: str) -> str:
    t = str(extraction_text or "")
    t = t.replace("全色", "")
    t = re.sub(r"(?:[+\-−－])?\s*(?:¥|￥)?\s*\d[\d,，]*\s*(?:円)?", "", t)
    t = t.strip()
    return t


def _strip_label_delims(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"^[／/、，,;；\s]+", "", s)
    s = re.sub(r"[／/、，,;；\s]+$", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Step 3: 正则模式定义
# ---------------------------------------------------------------------------

COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)

_SPLIT_TOKENS_SAFE_RE = re.compile(
    r"""
    [／/、，]
    |(?<!\d),(?!\d)
    |(?:\s+\+\s+)
    |(?:\s*;\s*)
    """,
    re.UNICODE | re.VERBOSE,
)

_COLOR_ABS_PRICE_RE = re.compile(
    r"""^\s*
        (?P<label>[^：:\-\s/、／¥円]+?)
        \s*(?:[:：]?\s*)
        (?:¥|￥)?\s*
        (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})*|\d+)
        \s*(?:円)?\s*$
    """,
    re.UNICODE | re.VERBOSE,
)

_PAIR_GROUP_RE_shop14 = re.compile(
    r"""
    (?P<labels>[^\d¥￥円:+\-−－＋]+?)
    \s*(?:[:：]\s*)?
    (?P<sign>[+\-−－＋])?
    \s*(?:¥|￥)?\s*
    (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})+|\d+)
    \s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

PAIR_RE_MULTI = re.compile(
    r"([^\d¥円,，＋+－\-−\s]+)\s*([+\-−－]?\s*\d[\d,，]*)"
)

# ---------------------------------------------------------------------------
# Step 4: 颜色家族同义词
# ---------------------------------------------------------------------------

_FAMILY_TOKENS = {
    "blue":   ["blue", "ブルー", "青"],
    "black":  ["black", "ブラック", "黒"],
    "white":  ["white", "ホワイト", "白"],
    "green":  ["green", "グリーン", "緑"],
    "red":    ["red", "レッド", "赤"],
    "pink":   ["pink", "ピンク"],
    "purple": ["purple", "パープル", "紫"],
    "yellow": ["yellow", "イエロー", "黄"],
    "orange": ["orange", "オレンジ", "橙"],
    "silver": ["silver", "シルバー", "銀"],
    "gold":   ["gold", "ゴールド", "金"],
    "gray":   ["gray", "grey", "グレー", "グレイ", "灰"],
    "natural":["natural", "ナチュラル"],
}

FAMILY_SYNONYMS_shop14: Dict[str, List[str]] = {}
for _fam, _toks in _FAMILY_TOKENS.items():
    for _t in _toks:
        FAMILY_SYNONYMS_shop14[str(_t).lower()] = _toks


def _label_matches_color_shop14(label_raw: str, color_raw: str, color_norm: str) -> bool:
    label_norm = _norm(label_raw)
    if not label_norm:
        return False

    color_raw_s = str(color_raw or "")
    color_norm_s = str(color_norm or "")

    label_l = label_norm.lower()
    color_raw_l = color_raw_s.lower()
    color_norm_l = color_norm_s.lower()

    if label_l == color_norm_l:
        return True

    if label_l and (label_l in color_raw_l):
        return True

    keys = {label_raw.strip().lower(), label_norm, label_raw.strip()}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop14:
            candidates.update(FAMILY_SYNONYMS_shop14[k])

    if not candidates:
        for k, toks in FAMILY_SYNONYMS_shop14.items():
            if k and (k == label_l or k in label_l):
                candidates.update(toks)
                break

    return any(str(tok).lower() in color_raw_l for tok in candidates)


# ---------------------------------------------------------------------------
# Step 5: remark列解析
# ---------------------------------------------------------------------------

def _resolve_remark_cols(df: "pd.DataFrame") -> Dict[str, Optional[str]]:
    want = ["减价条件", "减价条件2", "23432"]
    norm_map = {_norm_colname(c): c for c in df.columns}

    resolved: Dict[str, Optional[str]] = {w: None for w in want}
    for w in want:
        nw = _norm_colname(w)
        if nw in norm_map:
            resolved[w] = norm_map[nw]
            continue
        for nc, ac in norm_map.items():
            if nw and (nw in nc):
                resolved[w] = ac
                break
    return resolved


# ---------------------------------------------------------------------------
# Step 6: multi-pair 拆分 helper
# ---------------------------------------------------------------------------

def _split_color_amount_pairs_multi(txt: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not txt:
        return out
    s = str(txt)

    for label, amt_s in PAIR_RE_MULTI.findall(s):
        label = label.lstrip("、/,／，,;；").strip()
        if not label:
            continue
        amt = _coerce_amount_yen(amt_s)
        if amt is None:
            continue
        out.append((label, amt))

    if len(out) >= 2:
        return out
    return []


# ---------------------------------------------------------------------------
# Step 7-A: 纯正则抽取路径
# ---------------------------------------------------------------------------

def _extract_rules_shop14_regex(
    text: str,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]]]]:
    """
    纯正则从 remark 文本中抽取颜色价格规则。
    返回: {"all_delta": Optional[int], "abs": [...], "delta": [...]}
    """
    out: Dict[str, Union[Optional[int], List[Tuple[str, int]]]] = {
        "all_delta": None, "abs": [], "delta": [],
    }
    s = _clean_remark_frag(text)
    if not s:
        return out

    # 全色检测
    m_all = re.search(r"全色\s*(?:[+\-−－])?\s*(\d[\d,]*)\s*(?:円)?", s)
    if m_all:
        out["all_delta"] = _coerce_amount_yen(m_all.group(0).replace("全色", "").strip()) or 0
        return out
    if "全色" in s:
        out["all_delta"] = 0
        return out

    # 先尝试 multi-pair 解析
    multi = _split_color_amount_pairs_multi(s)
    if multi:
        vals_abs = [abs(v) for _, v in multi]
        if all(v >= 20000 for v in vals_abs):
            out["abs"] = [(lb, abs(v)) for lb, v in multi]
        else:
            out["delta"] = list(multi)
        return out

    # PAIR_GROUP 正则
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []

    for m in _PAIR_GROUP_RE_shop14.finditer(s):
        labels_raw = _strip_label_delims(m.group("labels"))
        sign_str = m.group("sign") or ""
        amt_str = m.group("amount")
        amt = _coerce_amount_yen(amt_str)
        if amt is None:
            continue

        has_sign = sign_str in {"+", "-", "−", "－", "＋"}
        if has_sign:
            if sign_str in {"-", "−", "－"}:
                amt = -abs(amt)
            for lb in _split_labels(labels_raw):
                delta_list.append((lb, amt))
        else:
            if abs(amt) >= 20000:
                for lb in _split_labels(labels_raw):
                    abs_list.append((lb, abs(amt)))
            else:
                for lb in _split_labels(labels_raw):
                    delta_list.append((lb, amt))

    out["abs"] = abs_list
    out["delta"] = delta_list
    return out


# ---------------------------------------------------------------------------
# Step 7-B: LangExtract (LLM) 路径
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shop14_lx_prompt_and_examples():
    import langextract as lx

    prompt = textwrap.dedent(
        """\
        你是信息抽取系统。请从输入文本中抽取"按颜色的价格规则（円）"。

        规则类型只有三类：
        1) all_colors：文本出现"全色"，可选跟金额（例如"全色 -3000""全色 3000円"）。
           表示所有颜色统一调整：final = base + amount_yen。若没写金额，amount_yen=0。
        2) abs_group：颜色标签(一个或多个)后面出现一个金额（例如"青 229,500""青/銀 229500円"）。
           表示这些颜色的最终价格等于该绝对金额。
        3) delta_group：颜色标签(一个或多个)后面出现带正负号的金额（例如"橙 -2500""銀+1000"）。
           表示这些颜色在基准价上加上差价（可为负）。

        分隔符可能是空格、换行、"/""／""、"","";"等。多个颜色可能共用同一个金额（例如"青/銀 229,500"），
        这种情况请把 attributes.labels 写成 "青/銀"（原样即可）。

        输出要求（非常重要）：
        - Use exact text for extraction_text（必须是原文连续子串，不要改写）。
        - 只抽取原文明确出现的规则，不要推断/补全。
        - attributes.amount_yen 必须是纯整数（去掉逗号/円/¥），差价允许负数。
        - attributes.labels：颜色标签，字符串（单色就写单个；多色就用原文分隔符，如 "青/銀"）。
        """
    )

    examples = [
        lx.data.ExampleData(
            text="青 229,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青 229,500",
                    attributes={"labels": "青", "amount_yen": "229500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="橙 -2500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙 -2500",
                    attributes={"labels": "橙", "amount_yen": "-2500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="全色 -3,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="all_colors",
                    extraction_text="全色 -3,000円",
                    attributes={"amount_yen": "-3000"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="青/銀 229,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青/銀 229,500",
                    attributes={"labels": "青/銀", "amount_yen": "229500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="橙/銀 -2,500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙/銀 -2,500円",
                    attributes={"labels": "橙/銀", "amount_yen": "-2500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="青 229,500\n橙 -2500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青 229,500",
                    attributes={"labels": "青", "amount_yen": "229500"},
                ),
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙 -2500",
                    attributes={"labels": "橙", "amount_yen": "-2500"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="全色",
            extractions=[
                lx.data.Extraction(
                    extraction_class="all_colors",
                    extraction_text="全色",
                    attributes={"amount_yen": "0"},
                )
            ],
        ),
    ]

    return prompt, examples


@lru_cache(maxsize=4096)
def _shop14_extract_rules_with_langextract(
    text: str,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]], List[dict]]]:
    """
    用 LangExtract(Ollama) 抽取规则。
    返回: {"all_delta": Optional[int], "abs": [...], "delta": [...], "raw": [...]}
    """
    out: Dict = {"all_delta": None, "abs": [], "delta": [], "raw": []}
    s = _clean_remark_frag(text)
    if not s:
        return out

    import langextract as lx

    prompt, examples = _shop14_lx_prompt_and_examples()

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=SHOP14_LLM_MODEL_ID,
            model_url=SHOP14_OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except TypeError:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            model_id=SHOP14_LLM_MODEL_ID,
            model_url=SHOP14_OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

    all_delta: Optional[int] = None
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []

    for e in (getattr(result, "extractions", None) or []):
        cls = str(getattr(e, "extraction_class", "") or "").strip()
        txt = str(getattr(e, "extraction_text", "") or "")
        attrs = getattr(e, "attributes", {}) or {}

        out["raw"].append({"class": cls, "text": txt, "attributes": attrs})

        cls_l = cls.lower().strip()

        # multi-pair 检测
        multi_pairs = _split_color_amount_pairs_multi(txt)
        if multi_pairs:
            vals_abs = [abs(v) for _, v in multi_pairs]
            kind: Optional[str] = None
            if "abs" in cls_l:
                kind = "abs"
            elif "delta" in cls_l or "diff" in cls_l:
                kind = "delta"
            else:
                if all(v >= 20000 for v in vals_abs):
                    kind = "abs"
                elif all(v <= 20000 for v in vals_abs):
                    kind = "delta"
                else:
                    big = sum(1 for v in vals_abs if v >= 20000)
                    kind = "abs" if big >= len(vals_abs) / 2.0 else "delta"

            for label, amt in multi_pairs:
                if kind == "abs":
                    abs_list.append((label, abs(int(amt))))
                else:
                    delta_list.append((label, int(amt)))

            logger.debug(
                "[LangExtract-multi] multi-pair detected",
                extra={
                    "event_type": "llm_multi_pair",
                    "shop_name": "買取楽園",
                    "cleaner_name": "shop14",
                    "extraction_text": _truncate_for_log(txt),
                    "kind": kind,
                    "pairs": str(multi_pairs),
                },
            )
            continue

        # 全色
        amount = None
        if isinstance(attrs, dict):
            amount = _coerce_amount_yen(attrs.get("amount_yen")) or _coerce_amount_yen(
                attrs.get("amount")
            )
        if amount is None:
            amount = _coerce_amount_yen(txt)

        if ("all" in cls_l) or ("全色" in txt):
            all_delta = int(amount) if amount is not None else 0
            continue

        # 普通 abs/delta
        labels_str = ""
        if isinstance(attrs, dict):
            labels_str = str(attrs.get("labels") or attrs.get("label") or "").strip()
        if not labels_str:
            labels_str = _labels_from_text_fallback(txt)

        labels = _split_labels(labels_str)

        kind = None
        if "abs" in cls_l:
            kind = "abs"
        elif "delta" in cls_l or "diff" in cls_l:
            kind = "delta"
        else:
            if amount is not None and abs(int(amount)) >= 20000:
                kind = "abs"
            elif amount is not None:
                kind = "delta"

        if not kind or amount is None or not labels:
            continue

        if kind == "abs":
            v = abs(int(amount))
            for lb in labels:
                abs_list.append((lb, v))
        else:
            v = int(amount)
            for lb in labels:
                delta_list.append((lb, v))

    out["all_delta"] = all_delta
    out["abs"] = abs_list
    out["delta"] = delta_list
    return out


def _extract_rules_shop14_llm_with_guardrails(
    text: str,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]]]]:
    """LLM抽取 + Guardrails（仅LLM路径应用）"""
    try:
        parsed = _shop14_extract_rules_with_langextract(text)
    except Exception as exc:
        logger.warning(
            "LLM extraction failed, returning empty",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": "買取楽園",
                "cleaner_name": "shop14",
                "error": str(exc),
                "text_snippet": _truncate_for_log(text, 120),
                "model_id": SHOP14_LLM_MODEL_ID,
                "model_url": SHOP14_OLLAMA_URL,
            },
        )
        return {"all_delta": None, "abs": [], "delta": []}

    return {
        "all_delta": parsed.get("all_delta"),
        "abs": parsed.get("abs", []),
        "delta": parsed.get("delta", []),
    }


# ---------------------------------------------------------------------------
# Step 7-C: Dispatch（三模式路由）
# ---------------------------------------------------------------------------

def _extract_rules_shop14_dispatch(
    text: str,
    mode: str = SHOP14_EXTRACTION_MODE,
) -> Tuple[Dict[str, Union[Optional[int], List[Tuple[str, int]]]], str]:
    """
    三模式路由。
    返回: (parsed_dict, extraction_method)
    parsed_dict = {"all_delta": ..., "abs": [...], "delta": [...]}
    """
    if mode == "regex":
        parsed = _extract_rules_shop14_regex(text)
        return parsed, "regex"

    if mode == "llm":
        parsed = _extract_rules_shop14_llm_with_guardrails(text)
        return parsed, "llm"

    # auto: regex first, LLM fallback
    parsed = _extract_rules_shop14_regex(text)
    has_results = (
        parsed.get("all_delta") is not None
        or parsed.get("abs")
        or parsed.get("delta")
    )
    if has_results:
        return parsed, "regex"

    parsed = _extract_rules_shop14_llm_with_guardrails(text)
    return parsed, "llm"


# ---------------------------------------------------------------------------
# Step 8: 主清洗函数
# ---------------------------------------------------------------------------

def clean_shop14(df: "pd.DataFrame", debug: bool = True) -> "pd.DataFrame":
    t_start = time.time()
    log_seq = 0

    logger.info(
        "shop14 cleaner started",
        extra={
            "event_type": "cleaner_start",
            "shop_name": "買取楽園",
            "cleaner_name": "shop14",
            "log_seq": log_seq,
            "input_rows": len(df),
            "extraction_mode": SHOP14_EXTRACTION_MODE,
        },
    )
    log_seq += 1

    # ---- 列校验 ----
    for c in ["name", "data6", "price2", "time-scraped"]:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": "買取楽園",
                    "cleaner_name": "shop14",
                    "log_seq": log_seq,
                    "column": c,
                },
            )
            log_seq += 1
            raise ValueError(f"shop14 清洗器缺少必要列：{c}")

    remark_cols_map = _resolve_remark_cols(df)

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        status = str(row.get("data6") or "")
        if "未開封" not in status:
            continue

        model_text = str(row.get("name") or "").strip()
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

        base_price = to_int_yen(row.get("price2"))
        if base_price is None:
            continue
        base_price = int(base_price)

        rec_at = parse_dt_aware(row.get("time-scraped"))

        # ---- remark 3列读取 ----
        frags: Dict[str, str] = {}
        for logical in ("减价条件", "减价条件2", "23432"):
            actual = remark_cols_map.get(logical)
            raw_val = row.get(actual) if actual else None
            frags[logical] = _clean_remark_frag(raw_val)

        combined = " ".join([v for v in frags.values() if v]).strip()

        # ---- 逐列抽取 + 聚合 ----
        agg_all_delta: Optional[int] = None
        agg_abs: List[Tuple[str, int]] = []
        agg_delta: List[Tuple[str, int]] = []
        extraction_method = "none"

        for col, frag in frags.items():
            if not frag:
                continue
            parsed, method = _extract_rules_shop14_dispatch(frag)

            if parsed.get("all_delta") is not None:
                agg_all_delta = int(parsed["all_delta"])
            agg_abs.extend(parsed.get("abs") or [])
            agg_delta.extend(parsed.get("delta") or [])
            extraction_method = method

        # 兜底：逐列都没抽到，合并串再跑一次
        if combined and (agg_all_delta is None) and (not agg_abs) and (not agg_delta):
            parsed2, method2 = _extract_rules_shop14_dispatch(combined)
            if parsed2.get("all_delta") is not None:
                agg_all_delta = int(parsed2["all_delta"])
            agg_abs.extend(parsed2.get("abs") or [])
            agg_delta.extend(parsed2.get("delta") or [])
            extraction_method = method2

        logger.debug(
            "extraction result",
            extra={
                "event_type": "extraction_result",
                "shop_name": "買取楽園",
                "cleaner_name": "shop14",
                "log_seq": log_seq,
                "row_idx": idx,
                "model": model_norm,
                "cap_gb": cap_gb,
                "base_price": base_price,
                "all_delta": agg_all_delta,
                "abs_count": len(agg_abs),
                "delta_count": len(agg_delta),
                "extraction_method": extraction_method,
                "combined_text": _truncate_for_log(combined, 120),
            },
        )
        log_seq += 1

        # ---- 全色快捷路径 ----
        if agg_all_delta is not None:
            final_price = base_price + int(agg_all_delta)

            for _col_norm, (pn, _raw) in color_map.items():
                rows.append(
                    {
                        "part_number": pn,
                        "shop_name": "買取楽園",
                        "price_new": int(final_price),
                        "recorded_at": rec_at,
                    }
                )
                logger.debug(
                    "output record (all_delta)",
                    extra={
                        "event_type": "output_record",
                        "shop_name": "買取楽園",
                        "cleaner_name": "shop14",
                        "log_seq": log_seq,
                        "part_number": pn,
                        "price": int(final_price),
                        "reason": f"all_delta({agg_all_delta})",
                    },
                )
                log_seq += 1
            continue

        # ---- label → color 匹配 ----
        color_abs: Dict[str, int] = {}
        color_deltas: Dict[str, int] = {}

        if agg_abs:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, abs_price in agg_abs:
                    if _label_matches_color_shop14(label_raw, col_raw, col_norm):
                        color_abs[col_norm] = int(abs_price)
                        logger.debug(
                            "label matched (abs)",
                            extra={
                                "event_type": "label_matching",
                                "shop_name": "買取楽園",
                                "cleaner_name": "shop14",
                                "log_seq": log_seq,
                                "label": label_raw,
                                "color_raw": col_raw,
                                "match_type": "abs",
                                "value": abs_price,
                            },
                        )
                        log_seq += 1

        if agg_delta:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in agg_delta:
                    if _label_matches_color_shop14(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = int(delta)
                        logger.debug(
                            "label matched (delta)",
                            extra={
                                "event_type": "label_matching",
                                "shop_name": "買取楽園",
                                "cleaner_name": "shop14",
                                "log_seq": log_seq,
                                "label": label_raw,
                                "color_raw": col_raw,
                                "match_type": "delta",
                                "value": delta,
                            },
                        )
                        log_seq += 1

        # ---- 各色价格计算 ----
        row_count = 0
        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm in color_abs:
                price_val = int(color_abs[col_norm])
                reason = "abs"
            else:
                d = int(color_deltas.get(col_norm, 0))
                price_val = int(base_price + d)
                reason = f"base+delta({d})" if col_norm in color_deltas else "base"

            rows.append(
                {
                    "part_number": pn,
                    "shop_name": "買取楽園",
                    "price_new": price_val,
                    "recorded_at": rec_at,
                }
            )
            row_count += 1

            logger.debug(
                "output record",
                extra={
                    "event_type": "output_record",
                    "shop_name": "買取楽園",
                    "cleaner_name": "shop14",
                    "log_seq": log_seq,
                    "part_number": pn,
                    "color": col_raw,
                    "price": price_val,
                    "reason": reason,
                },
            )
            log_seq += 1

        logger.debug(
            "row processing summary",
            extra={
                "event_type": "row_processing_summary",
                "shop_name": "買取楽園",
                "cleaner_name": "shop14",
                "log_seq": log_seq,
                "row_idx": idx,
                "model": model_norm,
                "records_produced": row_count,
            },
        )
        log_seq += 1

    # ---- 输出 DataFrame 组装 ----
    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed = round(time.time() - t_start, 2)
    logger.info(
        "shop14 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": "買取楽園",
            "cleaner_name": "shop14",
            "log_seq": log_seq,
            "output_rows": len(out),
            "elapsed_seconds": elapsed,
        },
    )

    return out
