from __future__ import annotations

"""
shop3 清洗器 — 買取一丁目

  原始文本（title / data5 / 减价1）
    │
    ├─ _normalize_model_generic()          ← Step 1: 机型归一化（外部工具）
    │
    ├─ _parse_capacity_gb()                ← Step 2: 容量解析（外部工具）
    │
    ├─ _price_from_shop3()                 ← Step 3: 基础价提取
    │
    ├─ _extract_color_deltas_shop3_dispatch()  ← Step 6: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_color_deltas_shop3_regex()   ← Step 4: 正则提取差价
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_color_deltas_shop3_llm_cached()  ← Step 5a: LLM 核心提取
    │       └─ Guardrails (_parse_delta_int_llm,          ← Step 5b: 防幻觉过滤
    │           _infer_default_sign_from_text)
    │
    ├─ _label_matches_color_shop3()        ← Step 7: 标签→颜色匹配
    │
    └─ clean_shop3()                       ← Step 8: 主函数，生成输出行
"""

import logging
import os
import re
import textwrap
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

SHOP3_OLLAMA_URL = os.getenv("SHOP3_OLLAMA_URL", "http://localhost:11434")
SHOP3_OLLAMA_MODEL_ID = os.getenv("SHOP3_OLLAMA_MODEL_ID", "gemma3:1b")
SHOP3_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

# --- LangExtract (LLM 抽取) ---
try:
    import langextract as lx
    _HAS_LANGEXTRACT = True
except Exception:
    lx = None
    _HAS_LANGEXTRACT = False

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

def _norm(s: str) -> str:
    return (s or "").strip()

# ----------------------------------------------------------------------
# Step 1-2: 文本归一化 & 金额解析
# ----------------------------------------------------------------------

_FZ_TO_HZ_TRANS = str.maketrans({
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    '，': ',', '．': '.', '：': ':', '（': '(', '）': ')',
    '　': ' ', '－': '-', '＋': '+', '¥': '', '￥': '',
})

def _normalize_amount_text(s: str) -> Optional[int]:
    """
    把全角数字/标点转半角，去掉非数字字符后尝试转换为 int。
    返回 None 表示无法解析。
    """
    if s is None:
        return None
    t = str(s).translate(_FZ_TO_HZ_TRANS)
    m = re.search(r"([0-9][0-9,]*)", t)
    if not m:
        return None
    numtxt = m.group(1).replace(",", "")
    try:
        return int(numtxt)
    except Exception:
        return None

def _clean_label_token(tok: str) -> str:
    if tok is None:
        return ""
    t = str(tok).strip()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"（.*?）", "", t)
    return t.strip()

# ----------------------------------------------------------------------
# Step 3: 基础价提取
# ----------------------------------------------------------------------

def _price_from_shop3(x: object) -> Optional[int]:
    """
    data5 -> price_new
    - 预期形如 '¥177,000'；也兼容 '～177,000円'/'10.5万' 等；取区间最大值
    - 去除可能出现的修饰词（"新品/未開封"等）
    """
    if x is None:
        return None
    s = str(x)
    s = (s.replace("新品", "")
           .replace("新\u54c1", "")
           .replace("未開封", "")
           .replace("未开封", ""))
    return to_int_yen(s)

# ----------------------------------------------------------------------
# Step 4: 正则提取差价
# ----------------------------------------------------------------------

_LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s；;]+")

_SIGNED_AMOUNT_PAT = re.compile(r"([+\-−－])\s*([0-9０-９][0-9０-９,，]*)")

_DELTA_PATTERN_STRICT = re.compile(
    r"""(?P<labels>[^+\-−－\d¥￥円]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

_DELTA_PATTERN_LOOSE = re.compile(
    r"""(?P<labels>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

def _extract_color_deltas_shop3_regex(text: str) -> List[Tuple[str, int]]:
    """
    从 text 中提取 (label_raw, delta_int) 多条记录，支持多标签共用金额的写法。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text).translate(_FZ_TO_HZ_TRANS).strip()
    for m in _DELTA_PATTERN_STRICT.finditer(s):
        labels_part = m.group("labels")
        sign = m.group("sign")
        amt_txt = m.group("amount")
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            continue
        if sign in ("-", "−", "－"):
            amt = -amt
        toks = [t for t in _LABEL_SPLIT_RE.split(labels_part) if t and t.strip()]
        for tok in toks:
            lbl = _clean_label_token(tok)
            if lbl:
                out.append((lbl, int(amt)))

    if not out:
        for m in _DELTA_PATTERN_LOOSE.finditer(s):
            labels_part = m.group("labels")
            sign = m.group("sign")
            amt_txt = m.group("amount")
            amt = _normalize_amount_text(amt_txt)
            if amt is None:
                continue
            if sign in ("-", "−", "－"):
                amt = -amt
            toks = [t for t in _LABEL_SPLIT_RE.split(labels_part) if t and t.strip()]
            for tok in toks:
                lbl = _clean_label_token(tok)
                if lbl:
                    out.append((lbl, int(amt)))

    return out

# ----------------------------------------------------------------------
# Step 5a: LLM 核心提取（带缓存）
# ----------------------------------------------------------------------

def _extract_signed_amounts_from_text(text: str) -> List[int]:
    """
    从原文中提取所有 "+/- 金额"（单位：JPY），例如 '-1500'、'−3,000'、'＋２,０００'
    """
    s = (text or "").translate(_FZ_TO_HZ_TRANS)
    s = s.replace("−", "-")  # U+2212
    out: List[int] = []
    for m in _SIGNED_AMOUNT_PAT.finditer(s):
        sign = m.group(1)
        amt = _normalize_amount_text(m.group(2))
        if amt is None:
            continue
        out.append(int(amt) if sign in ("+", "＋") else -int(amt))
    return out

def _single_signed_delta_from_text(text: str) -> Optional[int]:
    """
    若原文只出现一种带符号金额（可能重复出现），返回它；否则返回 None
    """
    vals = _extract_signed_amounts_from_text(text)
    if not vals:
        return None
    return vals[0] if len(set(vals)) == 1 else None

def _infer_default_sign_from_text(text: str) -> Optional[int]:
    """
    若原文只出现"全为负"或"全为正"的带符号金额，返回其符号方向（-1 / +1）；混合则 None
    """
    vals = _extract_signed_amounts_from_text(text)
    if not vals:
        return None
    has_pos = any(v > 0 for v in vals)
    has_neg = any(v < 0 for v in vals)
    if has_neg and not has_pos:
        return -1
    if has_pos and not has_neg:
        return 1
    return None

def _parse_delta_int_llm(x: object, default_sign: Optional[int]) -> Optional[int]:
    """
    解析 LLM 输出的 delta。
    - 若无符号且 default_sign 可推断，则按 default_sign 赋符号
    - 若有符号但与 default_sign 冲突（原文只有一种符号方向），以原文为准修正
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None

    # 数值类型：可能没有"显式符号"，按 default_sign 修正方向
    if isinstance(x, int):
        if default_sign is not None and x != 0 and (x > 0) != (default_sign > 0):
            return int(default_sign * abs(x))
        return int(x)
    if isinstance(x, float):
        v = int(round(x))
        if default_sign is not None and v != 0 and (v > 0) != (default_sign > 0):
            return int(default_sign * abs(v))
        return v

    s = str(x).strip().translate(_FZ_TO_HZ_TRANS).replace("−", "-").strip()
    if not s:
        return None

    explicit_sign: Optional[int] = None
    if s[0] in {"+", "-"}:
        explicit_sign = 1 if s[0] == "+" else -1
        s_num = s[1:].strip()
    else:
        s_num = s

    amt = _normalize_amount_text(s_num)
    if amt is None:
        return None

    # 无显式符号：必须借助原文 default_sign，否则不解析（避免误判为正）
    if explicit_sign is None:
        if default_sign is None:
            return None
        return int(default_sign * amt)

    # 有显式符号但原文只有一种符号方向：以原文为准
    if default_sign is not None and explicit_sign != default_sign:
        explicit_sign = default_sign

    return int(explicit_sign * amt)

# LLM prompt & examples (only available when langextract is installed)
if _HAS_LANGEXTRACT:
    _SHOP3_COLOR_DELTA_PROMPT = textwrap.dedent("""\
        你将看到一个很短的"备注/减价1"字符串，来源于日本二手回收价格表。
        目标：抽取"颜色标签 -> 价格差额(人民币/日元中的日元JPY)"的映射。
        输出要求（非常重要）：
        - 只抽取与"颜色差额/加价/减价"相关的信息；忽略与差额无关的描述（如 新品/未開封/SIMフリー 等）。
        - 每个颜色标签单独输出一条 Extraction。
        - extraction_class 固定为 "color_delta"。
        - extraction_text 必须是输入文本中的"原样子串"，只包含颜色标签本身（不要带分隔符、不要翻译、不要归一化）。
        - attributes 必须包含键 "delta_yen"，值为整数（可为字符串形式的整数也可），单位 JPY：
            * "-3000""−3,000""－３,０００"代表 -3000
            * "+2000""＋２,０００"代表 +2000
        - 如果文本里没有任何明确的 +/- 金额，则返回空的 extractions。
    """)

    _SHOP3_COLOR_DELTA_EXAMPLES = [
        lx.data.ExampleData(
            text="ブルー、シルバー　-1000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "-1000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="シルバー", attributes={"delta_yen": "-1000"}),
            ],
        ),
        lx.data.ExampleData(
            text="シルバー-3,000/ディープブルー-3,000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="シルバー", attributes={"delta_yen": "-3000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ディープブルー", attributes={"delta_yen": "-3000"}),
            ],
        ),
        lx.data.ExampleData(
            text="シルバー　-3,000 ブルー -3,000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="シルバー", attributes={"delta_yen": "-3000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "-3000"}),
            ],
        ),
        lx.data.ExampleData(
            text="ブラック、ブルー　-4000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブラック", attributes={"delta_yen": "-4000"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "-4000"}),
            ],
        ),
        lx.data.ExampleData(
            text="ブルー +2000",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ブルー", attributes={"delta_yen": "+2000"}),
            ],
        ),
        lx.data.ExampleData(
            text="オレンジ、ディープブルー-1500",
            extractions=[
                lx.data.Extraction(extraction_class="color_delta", extraction_text="オレンジ",
                                   attributes={"delta_yen": "-1500"}),
                lx.data.Extraction(extraction_class="color_delta", extraction_text="ディープブルー",
                                   attributes={"delta_yen": "-1500"}),
            ],
        ),
    ]
else:
    _SHOP3_COLOR_DELTA_PROMPT = ""
    _SHOP3_COLOR_DELTA_EXAMPLES = []

def _iter_extractions_from_langextract_result(result) -> List[object]:
    """
    lx.extract(text) 可能返回 AnnotatedDocument，也可能在未来版本/调用方式里返回 list；
    这里统一成 list[Extraction]。
    """
    if result is None:
        return []
    if isinstance(result, list):
        out = []
        for doc in result:
            out.extend(getattr(doc, "extractions", []) or [])
        return out
    return list(getattr(result, "extractions", []) or [])

@lru_cache(maxsize=4096)
def _extract_color_deltas_shop3_llm_cached(text: str) -> Tuple[Tuple[str, int], ...]:
    s = (text or "").strip()
    if not s:
        return tuple()

    if not re.search(r"[0-9０-９+\-−－]", s):
        return tuple()

    if lx is None:
        return tuple()

    # 原文全局 delta / 默认符号
    delta_global = _single_signed_delta_from_text(s)     # 例如：仅出现 -1500 => -1500
    default_sign = _infer_default_sign_from_text(s)      # 例如：仅出现负号 => -1

    result = lx.extract(
        text_or_documents=s,
        prompt_description=_SHOP3_COLOR_DELTA_PROMPT,
        examples=_SHOP3_COLOR_DELTA_EXAMPLES,
        model_id=SHOP3_OLLAMA_MODEL_ID,
        model_url=SHOP3_OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
    )

    mp: Dict[str, int] = {}
    for ext in _iter_extractions_from_langextract_result(result):
        if getattr(ext, "extraction_class", None) != "color_delta":
            continue

        label = _clean_label_token(getattr(ext, "extraction_text", "") or "")
        if not label:
            continue

        # 若原文只有一种带符号金额，直接覆盖所有 label 的 delta
        if delta_global is not None:
            mp[label] = int(delta_global)
            continue

        attrs = getattr(ext, "attributes", None) or {}
        delta_raw = attrs.get("delta_yen")
        delta = _parse_delta_int_llm(delta_raw, default_sign)

        if delta is None:
            delta = _parse_delta_int_llm(attrs.get("delta") or attrs.get("amount"), default_sign)

        if delta is None:
            continue

        mp[label] = int(delta)

    return tuple(mp.items())

# ----------------------------------------------------------------------
# Step 5b: LLM + Guardrails（仅 LLM 路径使用）
# ----------------------------------------------------------------------

def _extract_color_deltas_shop3_llm_with_guardrails(
    text: str,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
) -> List[Tuple[str, int]]:
    """
    LLM 提取 + guardrails（_parse_delta_int_llm, _infer_default_sign_from_text）。
    仅 LLM 路径使用。
    """
    s = (text or "").strip()
    if not s:
        return []

    if not _HAS_LANGEXTRACT or lx is None:
        return []

    try:
        result = list(_extract_color_deltas_shop3_llm_cached(s))
    except Exception as e:
        log_extra = {
            "event_type": "llm_extraction_error",
            "error": str(e),
            "error_type": type(e).__name__,
            "model_id": SHOP3_OLLAMA_MODEL_ID,
            "model_url": SHOP3_OLLAMA_URL,
            "text_length": len(s),
            "text_preview": _truncate_for_log(s, 100),
        }
        if shop_name:
            log_extra["shop_name"] = shop_name
        if cleaner_name:
            log_extra["cleaner_name"] = cleaner_name
        if row_context:
            log_extra.update(row_context)

        logger.warning(
            "LangExtract extraction failed",
            extra=log_extra,
        )
        return []

    return result

# ----------------------------------------------------------------------
# Step 6: 提取模式调度
# ----------------------------------------------------------------------

def _extract_color_deltas_shop3_dispatch(
    text: str,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
) -> Tuple[List[Tuple[str, int]], str]:
    """
    根据 SHOP3_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 (labels_and_deltas, extraction_method)
    """
    mode = SHOP3_EXTRACTION_MODE

    if mode == "regex":
        return _extract_color_deltas_shop3_regex(text), "regex"

    if mode == "llm":
        result = _extract_color_deltas_shop3_llm_with_guardrails(
            text, shop_name, cleaner_name, row_context,
        )
        return result, "llm"

    # ---- auto: 正则优先，正则无颜色结果时 LLM 兜底 ----
    regex_res = _extract_color_deltas_shop3_regex(text)
    if regex_res:
        return regex_res, "regex"

    llm_res = _extract_color_deltas_shop3_llm_with_guardrails(
        text, shop_name, cleaner_name, row_context,
    )
    return llm_res, "llm"

# ----------------------------------------------------------------------
# Step 7: 颜色家族同义词 & 匹配
# ----------------------------------------------------------------------

FAMILY_SYNONYMS_shop3 = {
    "blue": ["ブルー", "青", "ディープブルー"],
    "ブルー": ["ブルー", "青", "ディープブルー"],
    "青": ["ブルー", "青", "ディープブルー"],
    "ディープブルー": ["ディープブルー", "ブルー", "青"],
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],
}

def _label_matches_color_shop3(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配：归一等值 | 文本子串 | 家族词命中"""
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True

    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop3:
            candidates.update(FAMILY_SYNONYMS_shop3[k])
    if not candidates:
        for _, toks in FAMILY_SYNONYMS_shop3.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break
    return any(tok in str(color_raw) for tok in candidates)

# ----------------------------------------------------------------------
# Step 8: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop3(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop3 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop3"
    SHOP_NAME = "買取一丁目"

    logger.info(
        "Starting shop3 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )

    need_cols = ["title", "data5", "time-scraped"]
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
            raise ValueError(f"shop3 清洗器缺少必要列：{c}")

    src = df.copy()
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok].reset_index(drop=True)
    if src.empty:
        elapsed_time = time.time() - start_time
        logger.info(
            "Shop3 cleaner completed (empty input after time filter)",
            extra={
                "event_type": "cleaner_complete",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "input_rows": len(df),
                "output_records": 0,
                "elapsed_seconds": round(elapsed_time, 2),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            },
        )
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_from_db()
    color_maps = _build_color_map(info_df)

    model_norm = src["title"].map(_normalize_model_generic)
    cap_gb     = src["title"].map(_parse_capacity_gb)

    try:
        base_price = src["data5"].map(_price_from_shop3)
    except Exception:
        base_price = src["data5"].map(to_int_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    remark = src["减价1"] if "减价1" in src.columns else None

    # 预提取 remark 文本（用于后续 dispatch 调用）
    if remark is not None:
        remark_text = remark.fillna("").astype(str)
    else:
        remark_text = pd.Series(["" for _ in range(len(src))])

    rows: List[dict] = []

    for i in range(len(src)):
        current_row_records: List[dict] = []
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p0 = base_price.iat[i]
        t  = recorded_at.iat[i]

        rem_text = str(remark.iat[i]) if remark is not None else ""

        if not m:
            continue
        if pd.isna(c):
            continue
        if p0 is None:
            continue

        key = (m, int(c))
        cmap = color_maps.get(key)
        if not cmap:
            continue

        # 构建行级上下文，用于传递给下级函数和日志
        row_context = {
            "row_index": i,
            "model_text": str(src["title"].iat[i]),
            "model_norm": m,
            "capacity_gb": int(c),
            "base_price": int(p0),
        }

        # 根据 SHOP3_EXTRACTION_MODE 提取颜色差额（regex / llm / auto）
        deltas, extraction_method = _extract_color_deltas_shop3_dispatch(
            rem_text,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            row_context=row_context,
        )

        # DEBUG: 记录提取结果
        available_colors_list = [
            {"color_norm": cn, "part_number": pn, "color_raw": cr}
            for cn, (pn, cr) in cmap.items()
        ]

        _log_seq += 1
        logger.debug(
            "Extraction result",
            extra={
                "event_type": "extraction_result",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": str(src["title"].iat[i]),
                "model_norm": m,
                "capacity_gb": int(c),
                "base_price": int(p0),
                "color_discount_raw": _truncate_for_log(rem_text, 200),
                "color_discount_raw_full": rem_text,
                "extraction_method": extraction_method,
                "labels_and_deltas": [
                    {"label": label, "delta": delta}
                    for label, delta in deltas
                ],
                "labels_extracted_count": len(deltas),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(cmap),
            },
        )

        # label -> color 命中 & 计算 delta
        per_color_delta: Dict[str, int] = {}
        per_color_trace: Dict[str, Dict[str, object]] = {}

        if deltas:
            for col_norm, (pn, col_raw) in cmap.items():
                for label_raw, delta in deltas:
                    if _label_matches_color_shop3(label_raw, col_raw, col_norm):
                        per_color_delta[col_norm] = int(delta)
                        per_color_trace[col_norm] = {
                            "matched_label": label_raw,
                            "delta": int(delta),
                            "color_raw": col_raw,
                        }

        # 详细的 label 匹配日志
        if deltas:
            for label_raw, delta in deltas:
                matched_colors = []
                matched_pns = []
                for col_norm, (pn, col_raw) in cmap.items():
                    if _label_matches_color_shop3(label_raw, col_raw, col_norm):
                        matched_colors.append(col_norm)
                        matched_pns.append(pn)

                _log_seq += 1
                logger.debug(
                    f"Label matching: {label_raw}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": i,
                        "model_norm": m,
                        "capacity_gb": int(c),
                        "base_price": int(p0),
                        "label": label_raw,
                        "delta": delta,
                        "matched_colors": matched_colors,
                        "matched_part_numbers": matched_pns,
                        "match_count": len(matched_colors),
                        "color_discount_raw_full": rem_text,
                        "labels_and_deltas": [
                            {"label": lb, "delta": d}
                            for lb, d in deltas
                        ],
                    },
                )

                # WARNING: 如果 label 没有匹配到任何颜色
                if not matched_colors:
                    _log_seq += 1
                    logger.warning(
                        f"Label not matched: {label_raw}",
                        extra={
                            "event_type": "label_no_match",
                            "log_seq": _log_seq,
                            "shop_name": SHOP_NAME,
                            "cleaner_name": CLEANER_NAME,
                            "row_index": i,
                            "model_norm": m,
                            "capacity_gb": int(c),
                            "base_price": int(p0),
                            "label": label_raw,
                            "delta": delta,
                            "available_colors": [cn for cn in cmap.keys()],
                            "color_discount_raw_full": rem_text,
                            "labels_and_deltas": [
                                {"label": lb, "delta": d}
                                for lb, d in deltas
                            ],
                        },
                    )

        # 生成输出记录
        output_records = []
        for col_norm, (pn, col_raw) in cmap.items():
            d = per_color_delta.get(col_norm, 0)
            price_val = int(p0) + int(d)
            delta_source = "matched_label" if col_norm in per_color_delta else "default_zero"

            # 找到匹配的 label（用于日志）
            matched_label = None
            if delta_source == "matched_label":
                trace = per_color_trace.get(col_norm)
                if trace:
                    matched_label = trace.get("matched_label")

            # 每条输出记录单独一条日志（扁平化）
            _log_seq += 1
            logger.debug(
                f"Output record: {pn}",
                extra={
                    "event_type": "output_record",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": i,
                    "model_text": str(src["title"].iat[i]),
                    "model_norm": m,
                    "capacity_gb": int(c),
                    "part_number": pn,
                    "color_norm": col_norm,
                    "color_raw": col_raw,
                    "base_price": int(p0),
                    "delta": d,
                    "final_price": int(price_val),
                    "delta_source": delta_source,
                    "matched_label": matched_label,
                    "recorded_at": str(t) if t else None,
                    "color_discount_raw_full": rem_text,
                    "labels_and_deltas": [
                        {"label": lb, "delta": dd}
                        for lb, dd in deltas
                    ],
                },
            )

            output_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": d,
                "final_price": int(price_val),
                "delta_source": delta_source,
            })

            rows.append({
                "part_number": str(pn),
                "shop_name": SHOP_NAME,
                "price_new": int(price_val),
                "recorded_at": t,
            })

            current_row_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": d,
                "final_price": int(price_val),
                "recorded_at": t,
                "delta_source": delta_source,
            })

        # 行级概览
        colors_matched = len([cn for cn in cmap.keys() if cn in per_color_delta])
        all_deltas_values = list(per_color_delta.values())

        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": str(src["title"].iat[i]),
                "model_norm": m,
                "capacity_gb": int(c),
                "base_price": int(p0),
                "color_discount_raw_full": rem_text,
                "current_row_records": [
                    {"pn": r["part_number"], "color": r["color_norm"], "delta": r["delta"], "final_price": r["final_price"], "src": r["delta_source"]}
                    for r in current_row_records
                ],
            },
        )

        _log_seq += 1
        logger.info(
            f"Row {i:<3d} | {str(src['title'].iat[i]):<28s} | labels: {len(deltas):<2d} | matched: {colors_matched:<2d} | records: {len(output_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": i,
                "model_text": str(src["title"].iat[i]),
                "model_norm": m,
                "capacity_gb": int(c),
                "base_price": int(p0),
                "color_discount_raw_preview": _truncate_for_log(rem_text, 100),
                "extraction_method": extraction_method,
                "labels_extracted_count": len(deltas),
                "colors_in_catalog": len(cmap),
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

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop3 cleaner completed",
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
