from __future__ import annotations

"""
shop2 清洗器 — 海峡通信

  原始文本（data2-1 / data2-2 / data3 / data5）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _is_target()                          ← Step 1: SIMfree+未開封 过滤
    │
    ├─ extract_price_yen()                   ← Step 2: 基础价(data3)解析（cleaner_tools 统一）
    │
    ├─ _normalize_model_generic()            ← Step 3: 机型规范化（cleaner_tools 统一）
    │
    ├─ _parse_capacity_gb()                  ← Step 4: 容量解析
    │
    ├─ _extract_specs_shop2_dispatch()   ← Step 7: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop2_regex()   ← Step 5: 正则提取规则
    │   │
    │   └─ llm 路径:
    │       ├─ _extract_specs_shop2_llm_core()     ← Step 6a: LLM 核心提取
    │       ├─ Guardrail A: label 原文校验         ← Step 6b: 防幻觉过滤
    │       ├─ Guardrail B: amount 原文校验        ← Step 6b: 防幻觉过滤
    │       └─ _extract_specs_shop2_regex()   ← Step 6c: 正则补全
    │
    ├─ _label_matches_color_unified()         ← Step 8: 颜色匹配（cleaner_tools 统一）
    │
    ├─ resolve_color_prices()               ← Step 9: 统一定价流程（cleaner_tools）
    │
    └─ clean_shop2()                         ← Step 10: 主函数，生成输出行
"""

import logging
import os
import re
import time
import textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import (
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _build_color_map,
    _load_iphone17_info_df_from_db,
    _truncate_for_log,
    _label_matches_color_unified,
    safe_to_text,
    PriceDecomposition,
    resolve_color_prices,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    log_row_skip,
    validate_columns,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop2"
SHOP_NAME = "海峡通信"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

# LangExtract + Ollama (本地 LLM) 集成，OLLAMA 配置见 cleaner_tools
try:
    import langextract as lx
    _HAS_LANGEXTRACT = True
except Exception:
    lx = None
    _HAS_LANGEXTRACT = False

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

def _norm(s: str) -> str:
    """shop2 专用：strip 归一化，用于 model/color/part_number 等字段"""
    return (s or "").strip()


# ----------------------------------------------------------------------
# Step 1: SIMfree+未開封 过滤
# ----------------------------------------------------------------------

def _is_target(s: str) -> bool:
    s = (s or "").lower()
    return ("simfree" in s) and ("未開封" in s)

# ----------------------------------------------------------------------
# Step 3: 颜色匹配（使用 cleaner_tools 统一实现）
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Step 4: 规则解析辅助函数
# ----------------------------------------------------------------------

_INT_RE = re.compile(r"[+-]?\d+")

def _coerce_int(val) -> Optional[int]:
    """把 int/float/str 的数字（含 '円'、'¥'、逗号、全角符号）稳健转成 int。"""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass

    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    s = s.replace(",", "").replace("円", "").replace("¥", "")
    s = s.replace("−", "-").replace("－", "-").replace("＋", "+")
    m = _INT_RE.search(s)
    if not m:
        return None
    return int(m.group(0))

_SIGN_MINUS = {"-", "−", "－", "–", "—", "―"}
_SIGN_PLUS = {"+", "＋"}

def _parse_rule_token_simple(token: str) -> Optional[Tuple[str, int]]:
    """
    解析单条规则 token，例如：
      '黒-2000' / '青-2000円' / '銀 +3000' -> ('黒', -2000) / ('青', -2000) / ('銀', 3000)
    """
    s = safe_to_text(token)
    if not s:
        return None

    # 从末尾找数字串
    i = len(s) - 1
    while i >= 0 and not s[i].isdigit():
        i -= 1
    if i < 0:
        return None

    j = i
    while j >= 0 and s[j].isdigit():
        j -= 1
    num_str = s[j + 1 : i + 1]
    if not num_str:
        return None

    # 数字前找 +/- 符号（允许中间有空格）
    k = j
    while k >= 0 and s[k].isspace():
        k -= 1
    if k < 0:
        return None

    sign_ch = s[k]
    if sign_ch in _SIGN_PLUS:
        sign = 1
    elif sign_ch in _SIGN_MINUS:
        sign = -1
    else:
        return None

    group = s[:k].strip().strip(" :：\t")
    if not group:
        return None

    return group, sign * int(num_str)

# ----------------------------------------------------------------------
# Step 5: 纯正则版规则提取
# ----------------------------------------------------------------------

def _extract_specs_shop2_regex(val) -> dict:
    """
    对原始 data5 做正则解析：
    - 按分隔符拆开（换行/+++ / + / 逗号等），逐段用 _parse_rule_token_simple 解析
    - 也尝试旧版 regex 模式作为补充
    """
    s = safe_to_text(val)
    if not s:
        return {}

    # 方法 A: _parse_adjust_rule_simple_all 逻辑
    t = s
    for rep in ("+++", "++", "+", "＋＋＋", "＋＋", "＋", "\r"):
        t = t.replace(rep, "\n")
    for sep in ("、", "，", ","):
        t = t.replace(sep, "\n")

    rules: dict[str, int] = {}
    for line in t.splitlines():
        parsed = _parse_rule_token_simple(line)
        if parsed:
            g, d = parsed
            rules[g] = d

    # 方法 B: 旧版正则（fallback 补充）
    if not rules:
        parts = re.split(r"\+{1,3}|[,、，\s]+", s)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = re.match(r"(.+?)-(\d+)", p)
            if not m:
                continue
            group = m.group(1).strip()
            amt = -int(m.group(2))
            rules[group] = amt

    return rules

# ----------------------------------------------------------------------
# Step 6: LLM + Guardrails 版规则提取
# ----------------------------------------------------------------------

if _HAS_LANGEXTRACT:
    _COLOR_RULE_PROMPT = textwrap.dedent(
        """\
        あなたは中古スマホ買取表の「色ごとの減額条件」を解析するツールです。
        入力は data5 列に入っている短い日本語テキストです。例:
        - "青-1000"
        - "銀-5000+++青-3000"
        - "青-1000円\n※開封品 ¥183,000"
        など、色名と金額（減額/増額）が混在して書かれています。

        タスク:
        - data5 の中から「色グループ」と「基準価格からの差額（円）」をすべて抽出してください。
        - 減額は負の値、増額は正の値とします。
        - 抽出対象は、基準価格(data3)に対する相対額だけです。開封品価格など他の情報は無視してください。

        出力スキーマ:
        - 抽出するエンティティはすべて extraction_class="color_rule" とします。
        - 各 color_rule の attributes には次のキーを入れてください:
          - "group_label": 文字列。元テキスト中の色グループ名（例: "青", "銀", "スペースブラック", "全色"）
          - "delta_yen": 整数。基準価格からの差額（円）。減額は負の値、増額は正の値。

        注意:
        - "青-1000" や "銀-5000" のような書き方は「基準価格から 1000 円/5000 円減額」を意味します。
        - "青+2000" のような表現があれば、それは「基準価格から 2000 円増額」です。
        - テキストの中に色の情報がなく、金額だけの場合は無視してください。
        - 解釈に迷う場合は、その項目を抽出しないでください（安全側）。
        """
    )

    _COLOR_RULE_EXAMPLES: List = [
        lx.data.ExampleData(
            text="青-1000\n※開封品 ¥183,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="青-1000",
                    attributes={"group_label": "青", "delta_yen": -1000},
                )
            ],
        ),
        lx.data.ExampleData(
            text="銀-5000+++青-3000\n※開封品 ¥183,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="銀-5000",
                    attributes={"group_label": "銀", "delta_yen": -5000},
                ),
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="青-3000",
                    attributes={"group_label": "青", "delta_yen": -3000},
                ),
            ],
        ),
    ]
else:
    _COLOR_RULE_PROMPT = ""
    _COLOR_RULE_EXAMPLES = []

import json

@lru_cache(maxsize=1024)
def _extract_specs_shop2_llm_core(rule_text: str) -> dict:
    """LLM 核心提取（无 guardrails），结果被缓存。"""
    s = (rule_text or "").strip()
    if not s:
        return {}

    if not _HAS_LANGEXTRACT:
        return _extract_specs_shop2_regex(s)

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=_COLOR_RULE_PROMPT,
            examples=_COLOR_RULE_EXAMPLES,
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

        doc = result.to_dict() if hasattr(result, "to_dict") else json.loads(
            json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o)))
        )

        rules: dict[str, int] = {}

        for ext in doc.get("extractions", []) or []:
            attrs = ext.get("attributes") or {}
            extraction_text = safe_to_text(ext.get("extraction_text"))

            # 1) 优先从 extraction_text 按行解析（更贴近原文，且可一次吃掉多条）
            if extraction_text:
                for piece in extraction_text.replace("\r", "\n").split("\n"):
                    parsed = _parse_rule_token_simple(piece)
                    if parsed:
                        g, d = parsed
                        rules[g] = d

            # 2) 再用 attributes 兜底（处理 extraction_text 不含金额的情况）
            group_label = safe_to_text(attrs.get("group_label"))
            delta = _coerce_int(attrs.get("delta_yen"))
            if group_label and (delta is not None):
                rules[group_label] = int(delta)

        # LLM 一条都没解析出来就回退
        if not rules:
            return _extract_specs_shop2_regex(s)

        return rules

    except Exception:
        return _extract_specs_shop2_regex(s)

def _extract_specs_shop2_llm(
    val,
    row_index: object = None,
) -> dict:
    """
    LLM 提取 + Guardrail A/B + 正则补全（仅 LLM 路径使用）。
    Guardrails:
      A) group_label 必须在原文中出现
      B) delta 金額の绝对值必须在原文中出现
    然后用正则结果补全 LLM 漏掉的 key。
    """
    s = safe_to_text(val)
    if not s:
        return {}

    llm_ok = False
    llm_rules: dict = {}
    try:
        llm_rules = _extract_specs_shop2_llm_core(s)
        llm_ok = True
    except Exception as e:
        logger.warning(
            "LangExtract extraction failed",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "error": str(e),
                "error_type": type(e).__name__,
                "model_id": OLLAMA_MODEL_ID,
                "model_url": OLLAMA_URL,
                "row_index": row_index,
                "text_length": len(s),
                "text_preview": _truncate_for_log(s, 100),
            },
        )

    # Guardrail A & B: label/amount 必须在原文出现
    text_no_commas = s.replace(",", "")
    filtered_rules: dict[str, int] = {}
    for group_label, delta in llm_rules.items():
        # Guardrail A: group_label 在原文中
        if group_label not in s:
            continue
        # Guardrail B: delta 金额绝对值在原文中
        if str(abs(int(delta))) not in text_no_commas:
            continue
        filtered_rules[group_label] = int(delta)

    # 正则补全：LLM 漏掉的 key 用正则结果补齐
    supplement = _extract_specs_shop2_regex(s)
    merged = dict(filtered_rules)
    for k, v in supplement.items():
        merged.setdefault(k, v)

    # LLM 完全失败时，回退到纯正则
    if (not llm_ok) and (not merged):
        return _extract_specs_shop2_regex(s)

    return merged

# ----------------------------------------------------------------------
# Step 7: 提取模式调度
# ----------------------------------------------------------------------

def _extract_specs_shop2_dispatch(
    val,
    *,
    base_price: int,
    source_text_raw: str,
    row_index: object = None,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无结果时 LLM + Guardrails 兜底

    返回 PriceDecomposition
    """
    mode = EXTRACTION_MODE

    if mode == "regex":
        rules = _extract_specs_shop2_regex(val)
        method = "regex"
    elif mode == "llm":
        rules = _extract_specs_shop2_llm(val, row_index=row_index)
        method = "llm"
    else:
        # ---- auto: 正则優先，正則無結果時 LLM 兜底 ----
        rules = _extract_specs_shop2_regex(val)
        if rules:
            method = "regex"
        else:
            rules = _extract_specs_shop2_llm(val, row_index=row_index)
            method = "llm"

    return PriceDecomposition(
        base_price=base_price,
        delta_specs=list(rules.items()),
        abs_specs=[],
        extraction_method=method,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 10: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop2(shop2_df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(shop2_df), log_seq=_log_seq, extraction_mode=EXTRACTION_MODE)
    _log_seq += 1

    # 统一列名（小写）
    df = shop2_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 必要列存在性检查（若缺则补 None，保持兼容）
    _log_seq = validate_columns(df, ["data2-1", "data2-2", "data3", "data5", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq, lenient=True)

    # 只保留 simfree 未開封
    df = df[df["data2-2"].apply(_is_target)].copy().reset_index(drop=True)
    if df.empty:
        log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(shop2_df), output_records=0, start_time=start_time, log_seq=_log_seq)
        return pd.DataFrame(
            columns=["part_number", "shop_name", "price_new", "recorded_at"]
        )

    # iphone17_df 预处理
    info = _load_iphone17_info_df_from_db()
    if "capacity_gb" not in info.columns:
        logger.error(
            "Missing required column: capacity_gb in iphone17_info",
            extra={
                "event_type": "validation_error",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "missing_column": "capacity_gb",
            },
        )
        _log_seq += 1
        raise ValueError("iphone17_info.csv 需要包含 capacity_gb 列")
    info["color"] = info["color"].apply(_norm)
    color_maps = _build_color_map(info)

    logger.debug(
        "After filter",
        extra={
            "event_type": "cleaner_start",
            "log_seq": _log_seq,
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "total_rows_after_filter": len(df),
        },
    )
    _log_seq += 1

    out_rows: list[dict] = []

    for pos, row in enumerate(df.to_dict("records")):
        rec_raw = row.get("time-scraped")
        recorded_at = parse_dt_aware(rec_raw)

        raw_modelcap = _norm(row.get("data2-1"))
        raw_targetflag = row.get("data2-2")
        raw_price = row.get("data3")
        raw_rule = row.get("data5")

        if not raw_modelcap:
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="data2-1 empty")
            continue

        cap_gb = _parse_capacity_gb(raw_modelcap)
        if not cap_gb:
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="capacity_gb parse failed",
                         data2_1_raw=_truncate_for_log(raw_modelcap, 100))
            continue

        model_name = _normalize_model_generic(raw_modelcap)
        if not model_name:
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="model_name normalization failed",
                         data2_1_raw=_truncate_for_log(raw_modelcap, 100))
            continue

        key = (model_name, int(cap_gb))
        cmap = color_maps.get(key)
        if not cmap:
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="no info match",
                         model_name=model_name, capacity_gb=cap_gb)
            continue

        base_price = extract_price_yen(raw_price)
        if base_price is None:
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=pos, skip_reason="base_price parse failed",
                         data3_raw=_truncate_for_log(str(raw_price), 100))
            continue

        # ---- 提取 ----
        raw_rule_s = safe_to_text(raw_rule)

        decomp = _extract_specs_shop2_dispatch(
            raw_rule,
            base_price=base_price,
            source_text_raw=raw_rule_s,
            row_index=pos,
        )

        # ---- 过滤空 part_number ----
        cmap_filtered = {cn: (pn, cr) for cn, (pn, cr) in cmap.items() if pn}

        new_rows, _log_seq = resolve_color_prices(
            decomp,
            cmap_filtered,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=recorded_at,
            emit_default_rows=True,
            skip_non_positive=True,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=pos,
            model_text=raw_modelcap,
            model_norm=model_name,
            capacity_gb=cap_gb,
        )
        out_rows.extend(new_rows)

    if not out_rows:
        log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(shop2_df), output_records=0, start_time=start_time, log_seq=_log_seq)
        return pd.DataFrame(
            columns=["part_number", "shop_name", "price_new", "recorded_at"]
        )

    out = assemble_output_df(out_rows, coerce_price=False)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(shop2_df), output_records=len(out), start_time=start_time, log_seq=_log_seq)

    return out
