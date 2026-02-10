from __future__ import annotations

"""
shop2 清洗器 — 海峡通信

  原始文本（data2-1 / data2-2 / data3 / data5）
    │
    ├─ _is_target()                          ← Step 1: SIMfree+未開封 过滤
    │
    ├─ _parse_yen()                          ← Step 2: 基础价(data3)解析
    │
    ├─ _pick_model_name_loose()              ← Step 3: 机型宽松匹配
    │
    ├─ _parse_capacity_gb()                  ← Step 4: 容量解析
    │
    ├─ _parse_adjust_rule_shop2_dispatch()   ← Step 7: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   └─ _parse_adjust_rule_shop2_regex()   ← Step 5: 正则提取规则
    │   │
    │   └─ llm 路径:
    │       ├─ _parse_adjust_rule_llm()           ← Step 6a: LLM 核心提取
    │       ├─ Guardrail A: label 原文校验         ← Step 6b: 防幻觉过滤
    │       ├─ Guardrail B: amount 原文校验        ← Step 6b: 防幻觉过滤
    │       └─ _parse_adjust_rule_shop2_regex()   ← Step 6c: 正则补全
    │
    ├─ _match_color_group()                  ← Step 8: 颜色组匹配
    │
    ├─ _apply_adjust_with_trace()            ← Step 9: 应用减额规则
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
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _load_iphone17_info_df_from_db

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

SHOP2_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

# LangExtract + Ollama (本地 LLM) 集成
try:
    import langextract as lx
    _HAS_LANGEXTRACT = True
except Exception:
    lx = None
    _HAS_LANGEXTRACT = False

_LANGEXTRACT_MODEL_ID = os.getenv("SHOP2_LX_MODEL_ID", "gemma3:1b")
_LANGEXTRACT_MODEL_URL = os.getenv("SHOP2_LX_MODEL_URL", "http://localhost:11434")

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

_YEN_RE = re.compile(r"[^\d]+")

def _parse_yen(val) -> int | None:
    """'¥177,000' / '177,000円' / '177000' -> 177000"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = _YEN_RE.sub("", s)
    if not s:
        return None
    try:
        n = int(s)
        return n
    except Exception:
        return None

def _norm(s: str) -> str:
    return (s or "").strip()

def _as_text(val) -> str:
    """
    把 data5 这种可能为 NaN/None/数字/字符串 的输入，统一规范成可解析的字符串。
    """
    if val is None:
        return ""

    # pandas 的 NaN / NA
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass

    s = str(val).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

# ----------------------------------------------------------------------
# Step 1: SIMfree+未開封 过滤
# ----------------------------------------------------------------------

def _is_target(s: str) -> bool:
    s = (s or "").lower()
    return ("simfree" in s) and ("未開封" in s)

# ----------------------------------------------------------------------
# Step 2: 机型宽松匹配
# ----------------------------------------------------------------------

def _norm_model_token(s: str) -> str:
    """
    将 data2-1 的机型片段"宽松"规范化（仅用于和 iphone17_info 里的 model_name 做宽松匹配）
    规则：小写、去空格、去多余符号
    """
    s = (s or "").lower()
    s = re.sub(r"iphone\s*", "iphone ", s)
    s = re.sub(r"[^0-9a-z\s+]", "", s)  # 仅保留 a-z0-9 和空格
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_model_name_loose(model_token: str, iphone17_df: pd.DataFrame) -> str | None:
    """
    宽松匹配：在 iphone17_df['model_name'] 中选与 token 最匹配的项（不严格 Fuzzy，先简单包含匹配）
    """
    token = _norm_model_token(model_token)
    if not token:
        return None
    # 候选（去重）
    candidates = list(
        dict.fromkeys([_norm(x) for x in iphone17_df["model_name"].dropna().tolist()])
    )

    def norm_m(m: str) -> str:
        return _norm_model_token(m)

    # 简单策略：同样规范化后，包含则命中
    hits = [m for m in candidates if token in norm_m(m) or norm_m(m) in token]
    if len(hits) == 1:
        return hits[0]
    # 多命中时偏向更长的 model_name（更具体）
    if hits:
        return sorted(hits, key=lambda m: len(m), reverse=True)[0]
    return None

# ----------------------------------------------------------------------
# Step 3: 颜色组匹配逻辑（黒 -> ブラック 等）
# ----------------------------------------------------------------------

def _match_color_group(group: str, color_name: str) -> tuple[bool, str]:
    """
    返回 (is_match, reason)
    集中管理 data5 里"组名"到实际颜色名的映射规则。
    """
    g = (group or "").strip()
    c = color_name or ""

    # 常见后缀清理：青系 / 橙色 / 黒色 等
    # （避免 LLM 或简单解析把"色/系"也带进 group_label）
    g = re.sub(r"(系|色)$", "", g).strip()

    # Blue 系
    if g in ("青", "ブルー", "ミストブルー", "ディープブルー", "スカイブルー"):
        return ("ブルー" in c), "contains ブルー"

    # Silver 系
    if g in ("銀", "シルバー"):
        return ("シルバー" in c), "contains シルバー"

    # Black 系 —— 黒 也命中 スペースブラック
    if g in ("黒", "ブラック"):
        return (
            ("ブラック" in c) or ("黒" in c) or ("ミッドナイト" in c),
            "contains ブラック/黒/ミッドナイト",
        )

    # White 系（如果你不希望"白"覆盖银色，可以去掉 "シルバー"）
    if g in ("白", "ホワイト"):
        return (
            ("ホワイト" in c) or ("白" in c) or ("シルバー" in c),
            "contains ホワイト/白/シルバー",
        )

    # Gold 系（可选）
    if g in ("金", "ゴールド"):
        return ("ゴールド" in c), "contains ゴールド"

    # Orange 系
    if g in ("橙", "オレンジ"):
        return (
            ("オレンジ" in c) or ("橙" in c),
            "contains オレンジ/橙",
        )

    # Fallback: 允许 data5 里直接写具体颜色（例如 "コズミックオレンジ-3000"）
    if g:
        return (g in c), "substring match"

    return False, ""

def _apply_adjust_with_trace(color_name: str, rules: dict) -> tuple[int, list[dict]]:
    """
    返回：(adjust_sum, trace)
    trace 项示例：{"group":"青","delta":-1000,"reason":"contains ブルー"}
    """
    c = color_name or ""
    adjust = 0
    trace: list[dict] = []
    for group, delta in (rules or {}).items():
        ok, reason = _match_color_group(group, c)
        if ok:
            adjust += int(delta)
            trace.append(
                {"group": (group or "").strip(), "delta": int(delta), "reason": reason}
            )
    return adjust, trace

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
    s = _as_text(token)
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

def _parse_adjust_rule_shop2_regex(val) -> dict:
    """
    对原始 data5 做正则解析：
    - 按分隔符拆开（换行/+++ / + / 逗号等），逐段用 _parse_rule_token_simple 解析
    - 也尝试旧版 regex 模式作为补充
    """
    s = _as_text(val)
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
def _parse_adjust_rule_llm_core(rule_text: str) -> dict:
    """LLM 核心提取（无 guardrails），结果被缓存。"""
    s = (rule_text or "").strip()
    if not s:
        return {}

    if not _HAS_LANGEXTRACT:
        return _parse_adjust_rule_shop2_regex(s)

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=_COLOR_RULE_PROMPT,
            examples=_COLOR_RULE_EXAMPLES,
            model_id=_LANGEXTRACT_MODEL_ID,
            model_url=_LANGEXTRACT_MODEL_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

        doc = result.to_dict() if hasattr(result, "to_dict") else json.loads(
            json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o)))
        )

        rules: dict[str, int] = {}

        for ext in doc.get("extractions", []) or []:
            attrs = ext.get("attributes") or {}
            extraction_text = _as_text(ext.get("extraction_text"))

            # 1) 优先从 extraction_text 按行解析（更贴近原文，且可一次吃掉多条）
            if extraction_text:
                for piece in extraction_text.replace("\r", "\n").split("\n"):
                    parsed = _parse_rule_token_simple(piece)
                    if parsed:
                        g, d = parsed
                        rules[g] = d

            # 2) 再用 attributes 兜底（处理 extraction_text 不含金额的情况）
            group_label = _as_text(attrs.get("group_label"))
            delta = _coerce_int(attrs.get("delta_yen"))
            if group_label and (delta is not None):
                rules[group_label] = int(delta)

        # LLM 一条都没解析出来就回退
        if not rules:
            return _parse_adjust_rule_shop2_regex(s)

        return rules

    except Exception:
        return _parse_adjust_rule_shop2_regex(s)

def _parse_adjust_rule_shop2_llm(
    val,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
) -> dict:
    """
    LLM 提取 + Guardrail A/B + 正则补全（仅 LLM 路径使用）。
    Guardrails:
      A) group_label 必须在原文中出现
      B) delta 金额的绝对值必须在原文中出现
    然后用正则结果补全 LLM 漏掉的 key。
    """
    s = _as_text(val)
    if not s:
        return {}

    llm_ok = False
    llm_rules: dict = {}
    try:
        llm_rules = _parse_adjust_rule_llm_core(s)
        llm_ok = True
    except Exception as e:
        log_extra = {
            "event_type": "llm_extraction_error",
            "error": str(e),
            "error_type": type(e).__name__,
            "model_id": _LANGEXTRACT_MODEL_ID,
            "model_url": _LANGEXTRACT_MODEL_URL,
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
    supplement = _parse_adjust_rule_shop2_regex(s)
    merged = dict(filtered_rules)
    for k, v in supplement.items():
        merged.setdefault(k, v)

    # LLM 完全失败时，回退到纯正则
    if (not llm_ok) and (not merged):
        return _parse_adjust_rule_shop2_regex(s)

    return merged

# ----------------------------------------------------------------------
# Step 7: 提取模式调度
# ----------------------------------------------------------------------

def _parse_adjust_rule_shop2_dispatch(
    val,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None,
) -> Tuple[dict, str]:
    """
    根据 SHOP2_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无结果时 LLM + Guardrails 兜底

    返回 (rules, extraction_method)
    """
    mode = SHOP2_EXTRACTION_MODE

    if mode == "regex":
        rules = _parse_adjust_rule_shop2_regex(val)
        return rules, "regex"

    if mode == "llm":
        rules = _parse_adjust_rule_shop2_llm(
            val,
            shop_name=shop_name,
            cleaner_name=cleaner_name,
            row_context=row_context,
        )
        return rules, "llm"

    # ---- auto: 正则优先，正则无结果时 LLM 兜底 ----
    regex_rules = _parse_adjust_rule_shop2_regex(val)
    if regex_rules:
        return regex_rules, "regex"

    llm_rules = _parse_adjust_rule_shop2_llm(
        val,
        shop_name=shop_name,
        cleaner_name=cleaner_name,
        row_context=row_context,
    )
    return llm_rules, "llm"

# ----------------------------------------------------------------------
# Step 10: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop2(shop2_df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop2 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop2"
    SHOP_NAME = "海峡通信"

    logger.info(
        "Starting shop2 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(shop2_df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    # 统一列名（小写）
    df = shop2_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 必要列存在性检查（若缺则补 None，保持兼容）
    need_cols = ["data2-1", "data2-2", "data3", "data5", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            _log_seq += 1
            logger.warning(
                f"Missing column, filling with None: {c}",
                extra={
                    "event_type": "validation_error",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "missing_column": c,
                    "available_columns": list(df.columns),
                }
            )
            df[c] = None

    # 只保留 simfree 未開封
    df = df[df["data2-2"].apply(_is_target)].copy().reset_index(drop=True)
    if df.empty:
        elapsed_time = time.time() - start_time
        logger.info(
            "Shop2 cleaner completed (no target rows)",
            extra={
                "event_type": "cleaner_complete",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "input_rows": len(shop2_df),
                "output_records": 0,
                "elapsed_seconds": round(elapsed_time, 2),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        )
        return pd.DataFrame(
            columns=["part_number", "shop_name", "price_new", "recorded_at"]
        )

    # iphone17_df 预处理
    info = _load_iphone17_info_df_from_db()
    if "capacity_gb" not in info.columns:
        _log_seq += 1
        logger.error(
            "Missing required column: capacity_gb in iphone17_info",
            extra={
                "event_type": "validation_error",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "missing_column": "capacity_gb",
            }
        )
        raise ValueError("iphone17_info.csv 需要包含 capacity_gb 列")
    info["color"] = info["color"].apply(_norm)

    _log_seq += 1
    logger.debug(
        "After filter",
        extra={
            "event_type": "cleaner_start",
            "log_seq": _log_seq,
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "total_rows_after_filter": len(df),
        }
    )

    out_rows: list[dict] = []

    for pos, row in enumerate(df.to_dict("records")):
        current_row_records: list[dict] = []
        rec_raw = row.get("time-scraped")
        recorded_at = parse_dt_aware(rec_raw)

        raw_modelcap = _norm(row.get("data2-1"))
        raw_targetflag = row.get("data2-2")
        raw_price = row.get("data3")
        raw_rule = row.get("data5")

        if not raw_modelcap:
            logger.debug(
                "Skipping row: data2-1 is empty",
                extra={
                    "event_type": "row_processing_summary",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": pos,
                    "skip_reason": "data2-1 empty",
                }
            )
            continue

        cap_gb = _parse_capacity_gb(raw_modelcap)
        if not cap_gb:
            logger.debug(
                "Skipping row: capacity_gb parse failed",
                extra={
                    "event_type": "row_processing_summary",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": pos,
                    "skip_reason": "capacity_gb parse failed",
                    "data2_1_raw": _truncate_for_log(raw_modelcap, 100),
                }
            )
            continue

        model_name = _pick_model_name_loose(raw_modelcap, info)
        if not model_name:
            logger.debug(
                "Skipping row: model_name loose match failed",
                extra={
                    "event_type": "row_processing_summary",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": pos,
                    "skip_reason": "model_name loose match failed",
                    "data2_1_raw": _truncate_for_log(raw_modelcap, 100),
                }
            )
            continue

        sub = info[
            (info["model_name"] == model_name) & (info["capacity_gb"] == cap_gb)
        ].copy()
        if sub.empty:
            logger.debug(
                "Skipping row: no info match for model+capacity",
                extra={
                    "event_type": "row_processing_summary",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": pos,
                    "skip_reason": "no info match",
                    "model_name": model_name,
                    "capacity_gb": cap_gb,
                }
            )
            continue

        base_price = _parse_yen(raw_price)
        if base_price is None:
            logger.debug(
                "Skipping row: base_price parse failed",
                extra={
                    "event_type": "row_processing_summary",
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": pos,
                    "skip_reason": "base_price parse failed",
                    "data3_raw": _truncate_for_log(str(raw_price), 100),
                }
            )
            continue

        # 构建行级上下文，用于传递给下级函数和日志
        row_context = {
            "row_index": pos,
            "model_name": model_name,
            "capacity_gb": cap_gb,
            "base_price": base_price,
        }

        # 提取颜色减价规则（dispatch: regex / llm / auto）
        rules, extraction_method = _parse_adjust_rule_shop2_dispatch(
            raw_rule,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            row_context=row_context,
        )

        raw_rule_s = _as_text(raw_rule)

        # DEBUG: 记录提取结果
        available_colors_list = sub["color"].dropna().astype(str).unique().tolist()

        _log_seq += 1
        logger.debug(
            "Extraction result",
            extra={
                "event_type": "extraction_result",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": pos,
                "data2_1_raw": _truncate_for_log(raw_modelcap, 200),
                "model_name": model_name,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "data5_raw": _truncate_for_log(raw_rule_s, 200),
                "data5_raw_full": raw_rule_s,
                "extraction_method": extraction_method,
                "parsed_rules": {k: v for k, v in rules.items()},
                "rules_count": len(rules),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(sub),
            }
        )

        used_groups: set[str] = set()
        output_records: list[dict] = []

        for _, it in sub.iterrows():
            part = _norm(it.get("part_number"))
            color = _norm(it.get("color"))
            if not part:
                logger.debug(
                    "Skip item: empty part_number",
                    extra={
                        "event_type": "output_record",
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": pos,
                        "color": color,
                        "skip_reason": "empty part_number",
                    }
                )
                continue

            adj, trace = _apply_adjust_with_trace(color, rules)
            for tr in trace:
                used_groups.add(tr["group"])

            # 判断 delta 来源
            delta_source = "matched_label" if trace else "default_zero"

            # label matching 日志
            if trace:
                matched_groups = [tr["group"] for tr in trace]
                _log_seq += 1
                logger.debug(
                    f"Label matching: {color}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": pos,
                        "model_name": model_name,
                        "capacity_gb": cap_gb,
                        "base_price": base_price,
                        "color": color,
                        "part_number": part,
                        "adjustment": adj,
                        "matched_groups": matched_groups,
                        "trace": trace,
                        "data5_raw_full": raw_rule_s,
                    }
                )

            price = int(base_price + adj)
            if price <= 0:
                logger.warning(
                    f"Skipping item: price <= 0",
                    extra={
                        "event_type": "output_record",
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": pos,
                        "color": color,
                        "part_number": part,
                        "base_price": base_price,
                        "adjustment": adj,
                        "final_price": price,
                        "skip_reason": "price <= 0",
                    }
                )
                continue

            # DEBUG: 每条输出记录单独一条日志
            _log_seq += 1
            logger.debug(
                f"Output record: {part}",
                extra={
                    "event_type": "output_record",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": pos,
                    "model_name": model_name,
                    "capacity_gb": cap_gb,
                    "part_number": part,
                    "color": color,
                    "base_price": base_price,
                    "adjustment": adj,
                    "final_price": price,
                    "delta_source": delta_source,
                    "trace": trace,
                    "recorded_at": str(recorded_at) if recorded_at else None,
                    "data5_raw_full": raw_rule_s,
                }
            )

            output_records.append({
                "part_number": part,
                "color": color,
                "adjustment": adj,
                "final_price": price,
                "delta_source": delta_source,
            })

            out_rows.append(
                {
                    "part_number": part,
                    "shop_name": SHOP_NAME,
                    "price_new": price,
                    "recorded_at": recorded_at,
                }
            )

            current_row_records.append({
                "part_number": part,
                "color": color,
                "adjustment": adj,
                "final_price": price,
                "recorded_at": recorded_at,
                "delta_source": delta_source,
            })

        # WARNING: rules 未命中任何颜色
        if rules:
            unused = [g for g in rules.keys() if g not in used_groups]
            if unused:
                _log_seq += 1
                logger.warning(
                    f"Unmatched rule groups: {unused}",
                    extra={
                        "event_type": "label_no_match",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": pos,
                        "model_name": model_name,
                        "capacity_gb": cap_gb,
                        "unmatched_groups": unused,
                        "available_colors": available_colors_list,
                        "data5_raw_full": raw_rule_s,
                        "parsed_rules": {k: v for k, v in rules.items()},
                    }
                )

        # DEBUG: 行级详细汇总
        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": pos,
                "model_name": model_name,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "data5_raw_full": raw_rule_s,
                "current_row_records": [
                    {"pn": r["part_number"], "color": r["color"], "adj": r["adjustment"], "final_price": r["final_price"], "src": r["delta_source"]}
                    for r in current_row_records
                ],
            }
        )

        # INFO: 行级概览（简洁）
        colors_matched = len(used_groups)
        _log_seq += 1
        logger.info(
            f"Row {pos:<3d} | {raw_modelcap:<28s} | rules: {len(rules):<2d} | matched: {colors_matched:<2d} | records: {len(output_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": pos,
                "model_name": model_name,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "data5_raw_preview": _truncate_for_log(raw_rule_s, 100),
                "extraction_method": extraction_method,
                "rules_count": len(rules),
                "colors_in_catalog": len(sub),
                "colors_matched_count": colors_matched,
                "output_records_count": len(output_records),
                "has_discounted_colors": any(v != 0 for v in rules.values()) if rules else False,
                "min_delta": min(rules.values()) if rules else 0,
                "max_delta": max(rules.values()) if rules else 0,
            }
        )

    if not out_rows:
        elapsed_time = time.time() - start_time
        logger.info(
            "Shop2 cleaner completed (no output rows)",
            extra={
                "event_type": "cleaner_complete",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "input_rows": len(shop2_df),
                "output_records": 0,
                "elapsed_seconds": round(elapsed_time, 2),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        )
        return pd.DataFrame(
            columns=["part_number", "shop_name", "price_new", "recorded_at"]
        )

    out = pd.DataFrame(
        out_rows, columns=["part_number", "shop_name", "price_new", "recorded_at"]
    )

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop2 cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(shop2_df),
            "output_records": len(out),
            "elapsed_seconds": round(elapsed_time, 2),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    return out
