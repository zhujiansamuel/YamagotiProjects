"""
shop14_cleaner  —  買取楽園

数据处理流程:
  raw DataFrame
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ Step 1  列校验 & remark列解析
    ├─ Step 2  行级过滤（未開封 + model/cap/color_map 匹配）
    ├─ Step 3  base_price 提取
    ├─ Step 4  remark文本归一化（3列合并）
    ├─ Step 5  价格规则抽取 dispatch（EXTRACTION_MODE: regex / llm / auto）
    │           ├─ regex路径: _extract_specs_shop14_regex()
    │           └─ llm路径:   _extract_specs_shop14_llm()
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
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    LABEL_SPLIT_RE_shop14,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop14"
SHOP_NAME = "買取楽園"

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
    parts = LABEL_SPLIT_RE_shop14.split(s)
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
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ---------------------------------------------------------------------------
# 原 shop14 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

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

def _extract_specs_shop14_regex(
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
def _extract_specs_shop14_lx_prompt():
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
def _extract_specs_shop14_llm_core(
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

    prompt, examples = _extract_specs_shop14_lx_prompt()

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except TypeError:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
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
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
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


def _extract_specs_shop14_llm(
    text: str,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]]]]:
    """LLM抽取 + Guardrails（仅LLM路径应用）"""
    try:
        parsed = _extract_specs_shop14_llm_core(text)
    except Exception as exc:
        logger.warning(
            "LLM extraction failed, returning empty",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "error": str(exc),
                "text_snippet": _truncate_for_log(text, 120),
                "model_id": OLLAMA_MODEL_ID,
                "model_url": OLLAMA_URL,
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

def _extract_specs_shop14_parse(
    text: str,
    mode: str = EXTRACTION_MODE,
) -> Tuple[Dict[str, Union[Optional[int], List[Tuple[str, int]]]], str]:
    """
    单 fragment 三模式路由。
    返回: (parsed_dict, extraction_method)
    parsed_dict = {"all_delta": ..., "abs": [...], "delta": [...]}
    """
    if mode == "regex":
        parsed = _extract_specs_shop14_regex(text)
        return parsed, "regex"

    if mode == "llm":
        parsed = _extract_specs_shop14_llm(text)
        return parsed, "llm"

    # auto: regex first, LLM fallback
    parsed = _extract_specs_shop14_regex(text)
    has_results = (
        parsed.get("all_delta") is not None
        or parsed.get("abs")
        or parsed.get("delta")
    )
    if has_results:
        return parsed, "regex"

    parsed = _extract_specs_shop14_llm(text)
    return parsed, "llm"


def _extract_specs_shop14_dispatch(
    frags: Dict[str, str],
    combined: str,
    *,
    base_price: int,
    source_text_raw: str,
) -> PriceDecomposition:
    """
    多 fragment 聚合 + PriceDecomposition 構築。

    frags の各値を _extract_specs_shop14_parse() で解析し結果を集約。
    全 fragment が空の場合は combined テキストで再試行。
    """
    agg_all_delta: Optional[int] = None
    agg_abs: List[Tuple[str, int]] = []
    agg_delta: List[Tuple[str, int]] = []
    extraction_method = "none"

    for _col, frag in frags.items():
        if not frag:
            continue
        parsed, method = _extract_specs_shop14_parse(frag)
        if parsed.get("all_delta") is not None:
            agg_all_delta = int(parsed["all_delta"])
        agg_abs.extend(parsed.get("abs") or [])
        agg_delta.extend(parsed.get("delta") or [])
        extraction_method = method

    # 兜底：逐列都没抽到，合并串再跑一次
    if combined and (agg_all_delta is None) and (not agg_abs) and (not agg_delta):
        parsed2, method2 = _extract_specs_shop14_parse(combined)
        if parsed2.get("all_delta") is not None:
            agg_all_delta = int(parsed2["all_delta"])
        agg_abs.extend(parsed2.get("abs") or [])
        agg_delta.extend(parsed2.get("delta") or [])
        extraction_method = method2

    # ---- 全色预处理 ----
    delta_specs: List[Tuple[str, int]] = list(agg_delta)
    abs_specs: List[Tuple[str, int]] = list(agg_abs)

    if agg_all_delta is not None:
        delta_specs = [("全色", agg_all_delta)]
        abs_specs = []

    return PriceDecomposition(
        base_price=base_price,
        delta_specs=delta_specs,
        abs_specs=abs_specs,
        extraction_method=extraction_method,
        source_text_raw=source_text_raw,
    )


# ---------------------------------------------------------------------------
# Step 8: 主清洗函数
# ---------------------------------------------------------------------------

def clean_shop14(df: "pd.DataFrame", debug: bool = True) -> "pd.DataFrame":
    t_start = time.time()
    _log_seq = 0

    logger.info(
        "shop14 cleaner started",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "log_seq": _log_seq,
            "input_rows": len(df),
            "extraction_mode": EXTRACTION_MODE,
        },
    )
    _log_seq += 1

    # ---- 列校验 ----
    for c in ["name", "data6", "price2", "time-scraped"]:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "log_seq": _log_seq,
                    "column": c,
                },
            )
            _log_seq += 1
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

        # ---- 提取 + 聚合 ----
        decomp = _extract_specs_shop14_dispatch(
            frags, combined,
            base_price=base_price,
            source_text_raw=combined,
        )
        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=True,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

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
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "log_seq": _log_seq,
            "output_rows": len(out),
            "elapsed_seconds": elapsed,
        },
    )

    return out
