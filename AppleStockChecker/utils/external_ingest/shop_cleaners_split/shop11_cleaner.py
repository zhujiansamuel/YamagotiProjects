from __future__ import annotations

"""
shop11 清洗器 — モバステ

  原始文本（storage_name / price_unopened / caution_empty）
    │
    ├─ _normalize_number_text()              ← Step 1: 全角→半角归一化
    │
    ├─ to_int_yen_shop11()                   ← Step 2: 日元价格解析
    │
    ├─ _lx_parse_storage_shop11()            ← Step 3: LLM 机型/容量解析
    │   └─ fallback: _normalize_model_generic + _parse_capacity_gb
    │
    ├─ _extract_color_deltas_shop11_dispatch()  ← Step 7: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_color_deltas_shop11_regex()   ← Step 5: 正则提取差价
    │   │
    │   └─ llm 路径:
    │       ├─ _lx_parse_color_deltas_shop11()        ← Step 6a: LLM 核心提取
    │       └─ Guardrails (delta 合理性检查)            ← Step 6b: 防幻觉过滤
    │
    ├─ _label_matches_color_shop11()         ← Step 4: 标签→颜色匹配
    │
    └─ clean_shop11()                        ← Step 8: 主函数，生成输出行
"""

import logging
import os
import re
import textwrap
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as dateparser

from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
    _FZ_TO_HZ_TRANS,
)

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

SHOP11_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SHOP11_OLLAMA_MODEL_ID = os.getenv("SHOP11_OLLAMA_MODEL_ID", "gemma3:1b")

SHOP11_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

def _coerce_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int,)):
            return int(v)
        s = str(v).strip()
        if not s:
            return None
        # 允许 "1,000" / "-1000" 之类
        s = s.replace(",", "")
        return int(float(s))
    except Exception:
        return None

# ----------------------------------------------------------------------
# Step 1: 全角→半角归一化
# ----------------------------------------------------------------------

def _normalize_number_text(txt: str) -> str:
    if txt is None:
        return ""
    return str(txt).translate(_FZ_TO_HZ_TRANS).strip()

# ----------------------------------------------------------------------
# Step 2: 日元价格解析
# ----------------------------------------------------------------------

def to_int_yen_shop11(v) -> Optional[int]:
    """
    将各种形式的日元表示解析为 int（日元），若无法解析返回 None。
    支持样例：
      "1,000" "1,000円" "¥1,000" "１，０００" "1000" 以及带空格的混合形式
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # 去掉括号内的备注
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()

    # 半角映射（数字/标点）
    trans_map = str.maketrans({
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        '，': ',', '．': '.', '－': '-', '＋': '+', '¥': '', '￥': '',
    })
    s2 = s.translate(trans_map)

    m = re.search(r"([+\-−－]?)\s*(?:¥|￥)?\s*([\d][\d,]*)", s2)
    if not m:
        m2 = re.search(r"([\d][\d,]*)", s2)
        if not m2:
            return None
        amt_txt = m2.group(1)
        sign = ""
    else:
        sign = m.group(1) or ""
        amt_txt = m.group(2)

    amt_digits = re.sub(r"[^\d]", "", amt_txt or "")
    if not amt_digits:
        return None
    try:
        val = int(amt_digits)
    except Exception:
        return None

    if sign in ("-", "−", "－"):
        val = -val
    return val

# ----------------------------------------------------------------------
# Step 3: LLM 机型/容量解析（storage_name -> model_norm, cap_gb）
# ----------------------------------------------------------------------

try:
    import langextract as lx
except Exception:  # 允许在未安装时仍可跑 fallback
    lx = None

@lru_cache(maxsize=1)
def _shop11_model_config():
    """
    LangExtract 新版推荐的 ModelConfig 方式；若你的 langextract 版本不支持，会在调用处兜底到旧参数。
    """
    if lx is None:
        return None
    provider_kwargs = {
        "model_url": SHOP11_OLLAMA_URL,
        "temperature": float(os.getenv("SHOP11_OLLAMA_TEMPERATURE", "0.0")),
        "timeout": int(os.getenv("SHOP11_OLLAMA_TIMEOUT", "180")),
        "max_tokens": int(os.getenv("SHOP11_OLLAMA_MAX_TOKENS", "512")),
    }
    # JSON mode（能显著降低本地模型"夹杂解释文字"导致的解析失败）
    try:
        provider_kwargs["format_type"] = lx.data.FormatType.JSON
    except Exception:
        pass
    try:
        return lx.factory.ModelConfig(model_id=SHOP11_OLLAMA_MODEL_ID, provider_kwargs=provider_kwargs)
    except Exception:
        return None

def _lx_extract_ollama(text: str, prompt: str, examples: list):
    """
    返回 result 对象；失败返回 None
    """
    if lx is None:
        return None

    cfg = _shop11_model_config()
    try:
        if cfg is not None:
            return lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                config=cfg,
                fence_output=True,
                use_schema_constraints=False,
            )
    except TypeError:
        # 老版本 langextract 不支持 config=...
        pass
    except Exception:
        # config 路径失败则也尝试旧参数路径
        pass

    # 旧参数路径（v1.0.4 一类写法）
    try:
        return lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=SHOP11_OLLAMA_MODEL_ID,
            model_url=SHOP11_OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except Exception:
        return None

@lru_cache(maxsize=8)
def _shop11_lx_storage_materials(valid_models: Tuple[str, ...]):
    """
    storage_name 解析：device_model + storage_capacity
    """
    model_list = "\n".join(f"- {m}" for m in valid_models if m)

    prompt = textwrap.dedent(f"""\
        You are a strict parser.

        Input format:
          STORAGE: <text>

        Extract up to 2 items:
          1) device_model:
             - extraction_text must be an exact substring from STORAGE (do not invent text).
             - attributes must include: {{"model_norm": "<normalized model>"}}
             - model_norm MUST exactly equal one of:
{model_list}

          2) storage_capacity:
             - extraction_text must be an exact substring from STORAGE containing GB or TB (e.g., "256GB", "1TB").
             - attributes must include: {{"capacity_gb": <int>}}
             - Convert TB to GB using 1TB = 1024GB.

        If you cannot determine a field, do not output that extraction.
    """)

    examples = [
        lx.data.ExampleData(
            text="STORAGE: iPhone17 Pro Max 256GB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="iPhone17 Pro Max",
                    attributes={"model_norm": "iPhone 17 Pro Max"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="256GB",
                    attributes={"capacity_gb": 256},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="STORAGE: 17pro 1TB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="17pro",
                    attributes={"model_norm": "iPhone 17 Pro"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="1TB",
                    attributes={"capacity_gb": 1024},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="STORAGE: iPhone17 プロ 512GB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="iPhone17 プロ",
                    attributes={"model_norm": "iPhone 17 Pro"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="512GB",
                    attributes={"capacity_gb": 512},
                ),
            ],
        ),
    ]
    return prompt, examples

@lru_cache(maxsize=4096)
def _lx_parse_storage_shop11(storage: str, valid_models: Tuple[str, ...]) -> Tuple[str, Optional[int], Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]]:
    """
    返回 (model_norm, cap_gb, trace)
    trace: (class, extraction_text, sorted(attributes.items()) )
    """
    if not storage or lx is None:
        return "", None, tuple()

    prompt, examples = _shop11_lx_storage_materials(valid_models)
    txt = f"STORAGE: {storage}"

    res = _lx_extract_ollama(txt, prompt, examples)
    extrs = getattr(res, "extractions", None) or []

    model_norm = ""
    cap_gb: Optional[int] = None
    trace = []

    for e in extrs:
        cls = str(getattr(e, "extraction_class", "") or "")
        et = str(getattr(e, "extraction_text", "") or "")
        attrs = getattr(e, "attributes", None) or {}
        attrs_items = tuple(sorted((str(k), str(v)) for k, v in attrs.items()))
        trace.append((cls, et, attrs_items))

        if cls == "device_model":
            mn = (attrs.get("model_norm") or "").strip()
            if mn:
                # 再走一次你现有的规范化，确保和 info_df 的 key 完全一致
                model_norm = _normalize_model_generic(mn) or mn
        elif cls == "storage_capacity":
            cap_gb = _coerce_int(attrs.get("capacity_gb"))

    return model_norm, cap_gb, tuple(trace)

# ----------------------------------------------------------------------
# Step 4: 颜色标签匹配
# ----------------------------------------------------------------------

_COLOR_SEP_SPLIT_RE = re.compile(r"[／/、，,・\s]+")

FAMILY_SYNONYMS_shop11 = {
    "blue": ["ブルー", "青", "blue"],
    "silver": ["シルバー", "銀", "silver"],
    "black": ["ブラック", "黒", "black"],
    "white": ["ホワイト", "白", "white"],
    "gold": ["ゴールド", "金", "gold"],
    "orange": ["オレンジ", "橙"],
}

def _label_matches_color_shop11(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    匹配策略（宽容）：
      - 归一化后（去空白、半角/全角数字转换）相等；
      - label_raw 为 color_raw 的子串；
      - label 在常见家族词典里，家族任一词出现在 color_raw 即匹配。
    """
    if not label_raw or not color_raw:
        return False
    lbl = str(label_raw).strip()
    cr = str(color_raw).strip()

    # 规范化：半角化 + 去两端空白
    lbl_norm = _norm(lbl)
    cr_norm = color_norm  # 传入时应已是 _norm(color_raw)

    # 1) 精确归一化相等
    if lbl_norm == cr_norm:
        return True

    # 2) 原文子串（精确子串）
    if lbl in cr:
        return True

    # 3) 分割后任一片段是 color_raw 的子串（处理 "シルバー SV" vs "シルバー" 之类）
    for tok in _COLOR_SEP_SPLIT_RE.split(lbl):
        tok = tok.strip()
        if not tok:
            continue
        if tok in cr:
            return True
        if _norm(tok) == cr_norm:
            return True

    # 4) 家族同义词
    k = lbl.strip().lower()
    for fam_tokens in FAMILY_SYNONYMS_shop11.values():
        if k in [t.lower() for t in fam_tokens] or any(tok in lbl for tok in fam_tokens):
            for tok in fam_tokens:
                if tok in cr:
                    return True

    return False

# ----------------------------------------------------------------------
# Step 5: 正则提取颜色差价
# ----------------------------------------------------------------------

_COLOR_GROUP_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # 最多 80 char 的 label group（不会以数字或 +/- 开头）
    [：:]\s*                               # 必须有 冒号（：或:）作为分隔（这是最常见的情形）
    (?P<sign>[+\-−－]?)\s*                 # 可选 +/-
    (?P<amount>[\d０-９,，]+)              # 金额（含全角数字与逗号）
    (?:\s*円|\s*¥|\s*￥)?                  # 可选货币符
    """,
    re.UNICODE | re.VERBOSE,
)

_COLOR_GROUP_FALLBACK_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # label group（保守）
    [\s]*?(?P<sign>[+\-−－])\s*
    (?P<amount>[\d０-９,，]+)
    (?:\s*円|\s*¥|\s*￥)?
    """,
    re.UNICODE | re.VERBOSE,
)

def _extract_color_deltas_shop11_regex(text: str) -> List[Tuple[str, int]]:
    """
    纯正则版颜色差额解析，返回 [(label_raw, delta_int), ...]
    支持：
      - "シルバー・ブルー：-1,000円(未開封)" -> ('シルバー', -1000), ('ブルー', -1000)
      - "ブルー、ブラック：-2,000円(未開封)"
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text).strip()
    # 去掉括号内备注 (未開封) 等
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()
    if not s:
        return out

    s_norm = _normalize_number_text(s)

    # 1) 主匹配：labelGroup：+/-?amount
    for m in _COLOR_GROUP_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = _normalize_number_text(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        # 把 labels 按常见分隔符拆成多个 label
        for lbl in _COLOR_SEP_SPLIT_RE.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 2) 回退匹配（如 "ブルー -4000"）
    for m in _COLOR_GROUP_FALLBACK_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = _normalize_number_text(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        for lbl in _COLOR_SEP_SPLIT_RE.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 去重：若同 label 多次解析到，以最后一个为准（保留最后出现的 delta）
    if out:
        tmp: Dict[str, int] = {}
        for lbl, d in out:
            tmp[lbl] = d
        return list(tmp.items())

    return out

# ----------------------------------------------------------------------
# Step 6: LLM + Guardrails 颜色差价提取
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shop11_lx_color_materials():
    """
    caution_empty 解析：color_delta（对 AVAILABLE_COLORS 里的颜色逐一给 delta）
    """
    prompt = textwrap.dedent("""\
        You are a strict parser.

        Input format:
          CAUTION: <text>
          AVAILABLE_COLORS: <c1 | c2 | c3 ...>

        Task:
          Extract color price deltas relative to the base unopened price.

        Output extractions:
          - extraction_class: "color_delta"
          - extraction_text: MUST be EXACTLY one color string from AVAILABLE_COLORS (copy it exactly).
          - attributes: {"delta_yen": <int>}

        Parsing rules:
          - "+2000円" => 2000, "-1,000円" => -1000 (JPY).
          - If CAUTION says "全色" or "すべて" or "全カラー", apply the same delta to ALL AVAILABLE_COLORS.
          - If a color has no delta info, do not output it.
          - If multiple deltas exist for same color, the last one wins.
          - Ignore notes in parentheses like "(未開封)".
    """)

    examples = [
        lx.data.ExampleData(
            text="CAUTION: ブルー、ブラック：-2,000円(未開封)\nAVAILABLE_COLORS: ブルー | ブラック | シルバー",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": -2000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="CAUTION: 全色:+1,000円\nAVAILABLE_COLORS: ブルー | ブラック",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": 1000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"delta_yen": 1000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="CAUTION: シルバー・ブルー：-１０００円\nAVAILABLE_COLORS: ブルー | ブラック | シルバー",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -1000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": -1000},
                ),
            ],
        ),
    ]
    return prompt, examples

@lru_cache(maxsize=4096)
def _lx_parse_color_deltas_shop11(
    caution: str,
    available_colors: Tuple[str, ...],
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]]:
    """
    返回 (deltas_items, trace)
    deltas_items: tuple of (color, delta_yen)  —— 最后出现覆盖前面
    """
    if lx is None:
        return tuple(), tuple()

    prompt, examples = _shop11_lx_color_materials()
    avail_line = " | ".join([c for c in available_colors if c])
    txt = f"CAUTION: {caution or ''}\nAVAILABLE_COLORS: {avail_line}"

    res = _lx_extract_ollama(txt, prompt, examples)
    extrs = getattr(res, "extractions", None) or []

    tmp: Dict[str, int] = {}
    trace = []

    for e in extrs:
        cls = str(getattr(e, "extraction_class", "") or "")
        et = str(getattr(e, "extraction_text", "") or "").strip()
        attrs = getattr(e, "attributes", None) or {}
        attrs_items = tuple(sorted((str(k), str(v)) for k, v in attrs.items()))
        trace.append((cls, et, attrs_items))

        if cls != "color_delta":
            continue

        delta = _coerce_int(attrs.get("delta_yen"))
        if delta is None or not et:
            continue

        # 目标：et 必须是 available_colors 之一；若不完全一致，fallback 做一层匹配
        if et in available_colors:
            tmp[et] = int(delta)
            continue

        # fallback：用 label 匹配逻辑把 et 贴到合法颜色上
        for c in available_colors:
            if _label_matches_color_shop11(et, c, _norm(c)):
                tmp[c] = int(delta)

    return tuple(tmp.items()), tuple(trace)

def _extract_color_deltas_shop11_llm_with_guardrails(
    caution_txt: str,
    available_colors: Tuple[str, ...],
    color_map: Dict[str, Tuple[str, str]],
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
) -> Dict[str, int]:
    """
    LLM 提取 + Guardrails（仅 LLM 路径使用）。
    返回 {color_norm: delta_int}
    """
    color_deltas: Dict[str, int] = {}
    llm_ok = False

    try:
        deltas_items, deltas_trace = _lx_parse_color_deltas_shop11(caution_txt, available_colors)
        color_deltas = dict(deltas_items)
        llm_ok = True
    except Exception as e:
        llm_ok = False
        log_extra = {
            "event_type": "llm_extraction_error",
            "error": str(e),
            "error_type": type(e).__name__,
            "model_id": SHOP11_OLLAMA_MODEL_ID,
            "model_url": SHOP11_OLLAMA_URL,
            "text_length": len(caution_txt),
            "text_preview": _truncate_for_log(caution_txt, 100),
        }
        if shop_name:
            log_extra["shop_name"] = shop_name
        if cleaner_name:
            log_extra["cleaner_name"] = cleaner_name
        if row_context:
            log_extra.update(row_context)
        logger.warning(
            "LangExtract color delta extraction failed",
            extra=log_extra,
        )

    # Guardrail A: delta 合理性检查 — 过滤掉不在 available_colors 中的键
    if color_deltas:
        filtered: Dict[str, int] = {}
        for cn, dv in color_deltas.items():
            if cn in available_colors:
                filtered[cn] = dv
        color_deltas = filtered

    # LLM 完全失败且无结果时，回退到正则
    if (not llm_ok) and (not color_deltas) and caution_txt.strip():
        deltas_fb = _extract_color_deltas_shop11_regex(caution_txt)
        if deltas_fb:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in deltas_fb:
                    if _label_matches_color_shop11(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = int(delta)

    return color_deltas

# ----------------------------------------------------------------------
# Step 7: 提取模式调度
# ----------------------------------------------------------------------

def _extract_color_deltas_shop11_dispatch(
    caution_txt: str,
    available_colors: Tuple[str, ...],
    color_map: Dict[str, Tuple[str, str]],
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
) -> Tuple[Dict[str, int], str]:
    """
    根据 SHOP11_EXTRACTION_MODE 决定颜色差价提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 (color_deltas_dict, extraction_method)
    """
    mode = SHOP11_EXTRACTION_MODE

    if mode == "regex":
        color_deltas: Dict[str, int] = {}
        deltas_re = _extract_color_deltas_shop11_regex(caution_txt)
        if deltas_re:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in deltas_re:
                    if _label_matches_color_shop11(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = int(delta)
        return color_deltas, "regex"

    if mode == "llm":
        color_deltas = _extract_color_deltas_shop11_llm_with_guardrails(
            caution_txt, available_colors, color_map,
            shop_name=shop_name, cleaner_name=cleaner_name, row_context=row_context,
        )
        return color_deltas, "llm"

    # ---- auto: 正则優先，正则無結果时 LLM 兜底 ----
    color_deltas_re: Dict[str, int] = {}
    deltas_re = _extract_color_deltas_shop11_regex(caution_txt)
    if deltas_re:
        for col_norm, (pn, col_raw) in color_map.items():
            for label_raw, delta in deltas_re:
                if _label_matches_color_shop11(label_raw, col_raw, col_norm):
                    color_deltas_re[col_norm] = int(delta)

    if color_deltas_re:
        return color_deltas_re, "regex"

    color_deltas_llm = _extract_color_deltas_shop11_llm_with_guardrails(
        caution_txt, available_colors, color_map,
        shop_name=shop_name, cleaner_name=cleaner_name, row_context=row_context,
    )
    return color_deltas_llm, "llm"

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop11(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop11 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop11"
    SHOP_NAME = "モバステ"

    logger.info(
        "Starting shop11 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )

    need_cols = ["storage_name", "price_unopened", "caution_empty", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            logger.error(
                f"Missing required column: {c}",
                extra={
                    "event_type": "validation_error",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "missing_column": c,
                    "available_columns": list(df.columns),
                },
            )
            raise ValueError(f"shop11 清洗器缺少必要列：{c}")

    df2 = df.copy().reset_index(drop=True)

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    # 用 info_df 推导"允许的规范化机型"，用来约束 LLM 输出，减少 key 对不上
    valid_models = tuple(
        sorted({m for m in info_df["model_name"].map(_normalize_model_generic).tolist() if m})
    )

    rows: List[dict] = []

    for i, row in df2.iterrows():
        current_row_records: List[dict] = []
        storage_raw = row.get("storage_name")
        price_raw = row.get("price_unopened")
        caution_raw = row.get("caution_empty")
        time_raw = row.get("time-scraped")

        storage = str(storage_raw or "").strip()
        if not storage:
            continue

        # 1) 先走 LLM 解析（失败则 fallback 到 regex）
        model_norm, cap_gb, storage_trace = _lx_parse_storage_shop11(storage, valid_models)

        if not model_norm or cap_gb is None:
            # fallback：regex
            model_norm_fb = _normalize_model_generic(storage)
            cap_fb = _parse_capacity_gb(storage)
            if model_norm_fb and cap_fb is not None:
                model_norm, cap_gb = model_norm_fb, int(cap_fb)

        if not model_norm or cap_gb is None:
            continue

        cap_gb = int(cap_gb)
        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)

        # 若 key 对不上，再做一次"保险规范化"尝试（减少 LLM 输出微小差异导致 miss）
        if not color_map:
            model_norm2 = _normalize_model_generic(model_norm) or model_norm
            key2 = (model_norm2, cap_gb)
            color_map = cmap_all.get(key2)
            if color_map:
                key = key2
                model_norm = model_norm2

        if not color_map:
            continue

        base_price = to_int_yen_shop11(price_raw)
        if base_price is None:
            continue

        # recorded_at（保持原有逻辑）
        rec_at_raw = time_raw
        try:
            recorded_at = dateparser.parse(str(rec_at_raw)) if pd.notna(rec_at_raw) else None
        except Exception:
            recorded_at = rec_at_raw

        # 2) 颜色差额：根据 SHOP11_EXTRACTION_MODE 调度
        avail_colors = tuple(color_map.keys())
        caution_txt = _normalize_number_text(str(caution_raw or ""))

        # 构建行级上下文，用于传递给下级函数和日志
        row_context = {
            "row_index": int(i),
            "storage_raw": str(storage_raw or ""),
            "model_norm": model_norm,
            "capacity_gb": cap_gb,
            "base_price": int(base_price),
        }

        color_deltas, extraction_method = _extract_color_deltas_shop11_dispatch(
            caution_txt, avail_colors, color_map,
            shop_name=SHOP_NAME, cleaner_name=CLEANER_NAME, row_context=row_context,
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
                "storage_raw": _truncate_for_log(str(storage_raw or ""), 200),
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": int(base_price),
                "caution_raw": _truncate_for_log(str(caution_raw or ""), 200),
                "caution_raw_full": str(caution_raw or ""),
                "caution_normalized": _truncate_for_log(caution_txt, 200),
                "extraction_method": extraction_method,
                "color_deltas": [
                    {"color": cn, "delta": dv}
                    for cn, dv in color_deltas.items()
                ],
                "color_deltas_count": len(color_deltas),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_map),
            },
        )

        # label -> color 命中日志（逐 delta 记录）
        if color_deltas:
            for cn, dv in color_deltas.items():
                _log_seq += 1
                logger.debug(
                    f"Label matching: {cn}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(i),
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "base_price": int(base_price),
                        "color_norm": cn,
                        "delta": dv,
                        "match_count": 1,
                        "caution_raw_full": str(caution_raw or ""),
                    },
                )

        # 3) 输出：每个颜色一行
        output_records = []
        for col_norm, (pn, col_raw) in color_map.items():
            delta = int(color_deltas.get(col_norm, 0))
            price_new = int(base_price + delta)

            # 判断 delta 来源
            delta_source = "matched_label" if col_norm in color_deltas else "default_zero"

            # DEBUG: 每条输出记录单独一条日志（扁平化）
            _log_seq += 1
            logger.debug(
                f"Output record: {pn}",
                extra={
                    "event_type": "output_record",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(i),
                    "storage_raw": str(storage_raw or ""),
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "part_number": pn,
                    "color_norm": col_norm,
                    "color_raw": col_raw,
                    "base_price": int(base_price),
                    "delta": delta,
                    "final_price": price_new,
                    "delta_source": delta_source,
                    "recorded_at": str(recorded_at) if recorded_at else None,
                    "caution_raw_full": str(caution_raw or ""),
                },
            )

            output_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": delta,
                "final_price": price_new,
                "delta_source": delta_source,
            })

            rows.append({
                "part_number": pn,
                "shop_name": SHOP_NAME,
                "price_new": price_new,
                "recorded_at": recorded_at,
            })

            current_row_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": delta,
                "final_price": price_new,
                "recorded_at": recorded_at,
                "delta_source": delta_source,
            })

        # DEBUG: 行级详细汇总
        all_deltas_values = list(color_deltas.values())
        colors_matched = len(color_deltas)

        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(i),
                "storage_raw": str(storage_raw or ""),
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": int(base_price),
                "caution_raw_full": str(caution_raw or ""),
                "current_row_records": [
                    {"pn": r["part_number"], "color": r["color_norm"], "delta": r["delta"], "final_price": r["final_price"], "src": r["delta_source"]}
                    for r in current_row_records
                ],
            },
        )

        # INFO: 行级概览（简洁）
        _log_seq += 1
        logger.info(
            f"Row {i:<3d} | {storage:<28s} | deltas: {len(color_deltas):<2d} | matched: {colors_matched:<2d} | records: {len(output_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(i),
                "storage_raw": _truncate_for_log(str(storage_raw or ""), 100),
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": int(base_price),
                "caution_raw_preview": _truncate_for_log(caution_txt, 100),
                "extraction_method": extraction_method,
                "color_deltas_count": len(color_deltas),
                "colors_in_catalog": len(color_map),
                "colors_matched_count": colors_matched,
                "output_records_count": len(output_records),
                "has_discounted_colors": any(d != 0 for d in all_deltas_values),
                "min_delta": min(all_deltas_values) if all_deltas_values else 0,
                "max_delta": max(all_deltas_values) if all_deltas_values else 0,
            },
        )

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop11 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "output_records": len(out),
            "elapsed_seconds": round(elapsed_time, 2),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )

    return out
