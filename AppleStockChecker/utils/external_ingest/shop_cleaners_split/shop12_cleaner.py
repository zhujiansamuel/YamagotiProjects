from __future__ import annotations

"""
shop12 清洗器 — トゥインクル

  原始文本（備考1 + 買取価格）
    │
    ├─ _normalize_remark_for_llm()              ← Step 1: 去除開封行，预处理備考1
    │
    ├─ _norm_amount_to_int()                    ← Step 2: 统一全角数字→int
    │
    ├─ _extract_price_parts_shop12_dispatch()   ← Step 5: 模式调度
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_price_parts_shop12_regex()    ← Step 3: 正则提取 (abs + delta)
    │   │       └─ _fallback_parse_rules()            ← 核心正则: _FALLBACK_ABS_RE / _FALLBACK_DELTA_RE
    │   │
    │   └─ llm 路径:
    │       └─ _extract_price_parts_shop12_llm_with_guardrails()  ← Step 4: LLM 提取 + 防幻觉
    │           └─ _parse_rules_with_langextract()    ← LLM 核心: effective_class 修正 + 去重
    │
    ├─ _label_matches_color()                   ← Step 6: 标签→颜色匹配 (EN_TO_JP)
    │
    └─ clean_shop12()                           ← Step 7: 主函数，生成输出行
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
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
)

# ----------------------------------------------------------------------
# 初始化 logger
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

SHOP12_EXTRACTION_MODE = "auto"  # "regex" | "llm" | "auto"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------

_norm = _norm_strip

# ----------------------------------------------------------------------
# Step 1: 備考1 文本预处理
# ----------------------------------------------------------------------

def _normalize_remark_for_llm(remark_raw: str) -> str:
    """
    - 把与"開封/開封品/※開封/開封済"粘在同一行的内容拆行；
    - 去掉所有"開封"行，只保留可用于新品价规则的行；
    - 最终返回喂给 LLM 的文本（可能是多行）。
    """
    if not remark_raw:
        return ""
    s = str(remark_raw)

    # 关键：把"※開封品"等前面强行插入换行（解决: Orange-2000円※開封品...）
    s = re.sub(r"(※\s*開封品|※\s*開封|開封品|開封済|開封)", r"\n\1", s)

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", s) if ln is not None and ln.strip()]
    keep: List[str] = []
    for ln in lines:
        if ("開封" in ln) or ("開封品" in ln) or ("※開封" in ln) or ("開封済" in ln):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()

# ----------------------------------------------------------------------
# Step 2: 数字归一化（含全角）
# ----------------------------------------------------------------------

def _norm_amount_to_int(s: str) -> Optional[int]:
    if s is None:
        return None
    tt = str(s).replace("　", " ").replace("，", ",").replace("．", ".")
    tt = tt.translate(str.maketrans({
        '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
        '－':'-','＋':'+','¥':'','￥':''
    }))
    m = re.search(r"([0-9][0-9,]*)", tt)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None

# ----------------------------------------------------------------------
# Step 3: 正则提取函数
# ----------------------------------------------------------------------

_FALLBACK_ABS_RE = re.compile(
    r"""(?P<labels>[^\d¥￥円:：/、，,;；※]+?)\s*(?:[:：]?\s*)?(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
    re.UNICODE | re.VERBOSE,
)
_FALLBACK_DELTA_RE = re.compile(
    r"""(?P<labels>[^+\-−－\d¥￥円/、，,;；※]+?)\s*(?P<sign>[+\-−－])\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
    re.UNICODE | re.VERBOSE,
)
_SPLIT_SEPS = r"[／/、，,・\s]+"

def _fallback_parse_rules(text: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []
    if not text:
        return abs_list, delta_list

    for ln in re.split(r"[\r\n]+", str(text)):
        ln = (ln or "").strip()
        if not ln:
            continue

        # 全色
        if "全色" in ln:
            m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)?", ln)
            if m:
                sign = m.group(1) or "+"
                amt = _norm_amount_to_int(m.group(2) or "0") or 0
                delta_list.append(("全色", -amt if sign in ("-", "−", "－") else amt))
            else:
                delta_list.append(("全色", 0))
            continue

        for m in _FALLBACK_ABS_RE.finditer(ln):
            amt = _norm_amount_to_int(m.group("amount"))
            if amt is None:
                continue
            labels_part = m.group("labels") or ""
            toks = [t.strip() for t in re.split(_SPLIT_SEPS, labels_part) if t.strip()]
            for tok in toks:
                if tok:
                    abs_list.append((tok, int(amt)))

        for m in _FALLBACK_DELTA_RE.finditer(ln):
            amt = _norm_amount_to_int(m.group("amount"))
            if amt is None:
                continue
            sign = m.group("sign") or "+"
            delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
            labels_part = m.group("labels") or ""
            toks = [t.strip() for t in re.split(_SPLIT_SEPS, labels_part) if t.strip()]
            for tok in toks:
                if tok:
                    delta_list.append((tok, delta))

    return abs_list, delta_list

def _extract_price_parts_shop12_regex(
    remark_for_llm: str,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    纯正则版：从预处理后的備考1文本中提取 (abs_list, delta_list)。
    """
    return _fallback_parse_rules(remark_for_llm)

# ----------------------------------------------------------------------
# Step 4: LLM 配置 & 核心提取函数
# ----------------------------------------------------------------------

_LX_PROMPT = textwrap.dedent(r"""
你要从输入文本（備考1）中抽取"颜色对应的价格规则"。只抽取以下两类：

1) delta（差额）
- 形式：<颜色标签><+或-><金额>円
- 例：orange-1000円  => delta_yen=-1000, color_label="orange"
- 例：Blue+2000円    => delta_yen=+2000, color_label="Blue"
- 例：全色-2000円     => delta_yen=-2000, color_label="全色"

2) abs_price（绝对价）
- 形式：<颜色标签> ¥<金額> 或 <颜色标签> <金額>円
- 例：Silver ¥230,500 => price_yen=230500, color_label="Silver"

规则：
- extraction_text 必须是输入里的"原文片段"，不要改写/不要翻译。
- 如果一行里有多种颜色分别给价或给差额，要分别输出多条 extraction。
- 如果文本里出现"開封/開封品/※開封/開封済"，这些内容不参与抽取（可以忽略）。
- 就算文本非常短（例如仅有 'orange-1000円'），只要存在规则也必须抽取出来。
""").strip()

def _lx_examples():
    import langextract as lx
    return [
        lx.data.ExampleData(
            text="orange-1000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="orange-1000円",
                    attributes={"color_label": "orange", "delta_yen": -1000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Orange-2000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="Orange-2000円",
                    attributes={"color_label": "Orange", "delta_yen": -2000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Silver ¥230,500\nBlue ¥229,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="Silver ¥230,500",
                    attributes={"color_label": "Silver", "price_yen": 230500},
                ),
                lx.data.Extraction(
                    extraction_class="abs_price",
                    extraction_text="Blue ¥229,000",
                    attributes={"color_label": "Blue", "price_yen": 229000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Blue-4000円\nBlack-4000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="Blue-4000円",
                    attributes={"color_label": "Blue", "delta_yen": -4000},
                ),
                lx.data.Extraction(
                    extraction_class="delta",
                    extraction_text="Black-4000円",
                    attributes={"color_label": "Black", "delta_yen": -4000},
                ),
            ],
        ),

    ]

@lru_cache(maxsize=8192)
def _parse_rules_with_langextract(remark_for_llm: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, str, dict]]]:
    """
    返回:
      abs_list  = [(label_raw, absolute_price_yen), ...]
      delta_list= [(label_raw, delta_yen), ...]
      llm_dbg   = [(effective_class, extraction_text, attrs), ...]
    这里的 effective_class 是经过"带货币符号/有正负号"的规则修正后的结果，
    不再完全相信 LLM 的 extraction_class。
    """
    remark_for_llm = (remark_for_llm or "").strip()
    if not remark_for_llm:
        return [], [], []

    try:
        import langextract as lx

        model_id = os.getenv("SHOP12_OLLAMA_MODEL_ID") or os.getenv("OLLAMA_MODEL_ID") or "gemma3:1b"
        model_url = os.getenv("SHOP12_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"

        llm_input = "色別価格ルール:\n" + remark_for_llm

        res = lx.extract(
            text_or_documents=llm_input,
            prompt_description=_LX_PROMPT,
            examples=_lx_examples(),
            model_id=model_id,
            model_url=model_url,
            temperature=float(os.getenv("SHOP12_LLM_TEMPERATURE", "0.0")),
            fence_output=False,
            use_schema_constraints=False,
            max_char_buffer=2000,
            language_model_params={
                "timeout": int(os.getenv("SHOP12_LLM_TIMEOUT", "120")),
                "num_ctx": int(os.getenv("SHOP12_LLM_NUM_CTX", "4096")),
            },
        )

        exts = getattr(res, "extractions", []) or []
        llm_dbg: List[Tuple[str, str, dict]] = []
        abs_list: List[Tuple[str, int]] = []
        delta_list: List[Tuple[str, int]] = []

        for e in exts:
            cls_raw = (getattr(e, "extraction_class", "") or "").strip()
            txt = getattr(e, "extraction_text", "") or ""
            attrs = dict(getattr(e, "attributes", {}) or {})

            # ---------- 关键逻辑：由文本内容来决定 effective_class ----------
            # 有 + / - 号 + 数字 => delta
            has_sign = bool(re.search(r"[+\-−－]\s*[０-９0-9]", txt))
            # 有 日元符号(¥ / ￥ / 円) => 价格
            has_currency = bool(re.search(r"[¥￥円]", txt))

            if has_sign:
                # 像 "Orange-1000円" / "全色-2000円"
                effective_cls = "delta"
            elif has_currency:
                # 像 "Orange ¥193,500" / "Silver 230,500円"
                effective_cls = "abs_price"
            else:
                # 兜底：用 LLM 原始的分类
                effective_cls = cls_raw or "delta"

            llm_dbg.append((effective_cls, txt, attrs))

            # ---------- 解析 label ----------
            label = (
                str(attrs.get("color_label") or attrs.get("color") or attrs.get("colour") or "")
                .strip()
            )
            if not label:
                # 再粗暴一点：如果前面有一串非数字非货币符号，就当颜色
                m_lbl = re.match(r"^[^\d0-9¥￥円+\-−－]+", txt)
                if m_lbl:
                    label = m_lbl.group(0).strip()
            if not label:
                continue

            # ---------- abs_price 逻辑 ----------
            if effective_cls == "abs_price":
                # LLM 可能给的是 price_yen，也可能误放在 delta_yen，统一兜一下
                raw_price = attrs.get("price_yen")
                if raw_price is None:
                    raw_price = attrs.get("delta_yen")
                if raw_price is None:
                    raw_price = txt
                price_i = _norm_amount_to_int(raw_price)
                if price_i is None:
                    price_i = _norm_amount_to_int(txt)
                if price_i is None:
                    continue
                # 绝对价一律用正数
                price_i = abs(int(price_i))
                abs_list.append((label, price_i))
                continue

            # ---------- delta 逻辑 ----------
            if effective_cls == "delta":
                raw_delta = attrs.get("delta_yen")
                delta_i: Optional[int] = None

                if isinstance(raw_delta, (int,)):
                    delta_i = int(raw_delta)
                else:
                    # 先从属性里解析
                    if raw_delta is not None:
                        delta_i = _norm_amount_to_int(raw_delta)

                # 属性里拿不到，再从文本里按 "符号 + 金额" 模式解析
                if delta_i is None:
                    m = re.search(r"([+\-−－])\s*([０-９0-9][０-９0-9,，]*)", txt)
                    if m:
                        sign = m.group(1)
                        amt = _norm_amount_to_int(m.group(2))
                        if amt is not None:
                            delta_i = -amt if sign in ("-", "−", "－") else amt

                if delta_i is None:
                    continue
                delta_list.append((label, int(delta_i)))
                continue

        # 同一 label 去重（后者覆盖前者）
        if abs_list:
            tmp = {}
            for k, v in abs_list:
                tmp[k] = v
            abs_list = list(tmp.items())
        if delta_list:
            tmp = {}
            for k, v in delta_list:
                tmp[k] = v
            delta_list = list(tmp.items())

        return abs_list, delta_list, llm_dbg

    except Exception as e:
        logger.warning(
            "LangExtract extraction failed",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": "トゥインクル",
                "cleaner_name": "shop12",
                "error": str(e),
                "error_type": type(e).__name__,
                "text_length": len(remark_for_llm),
                "text_preview": _truncate_for_log(remark_for_llm, 100),
            }
        )
        return [], [], []

def _extract_price_parts_shop12_llm_with_guardrails(
    remark_for_llm: str,
    idx: object = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    LLM 提取 + Guardrails（仅 LLM 路径使用）。
    Guardrail: effective_class 修正 + 去重（内置于 _parse_rules_with_langextract）。
    LLM 失败时回退到正则。
    """
    abs_list, delta_list, _llm_dbg = _parse_rules_with_langextract(remark_for_llm)

    # Guardrail: LLM 完全失败（空结果）时，回退到正则
    if not abs_list and not delta_list and remark_for_llm:
        logger.debug(
            "LLM returned empty, falling back to regex",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": "トゥインクル",
                "cleaner_name": "shop12",
                "row_index": idx,
                "remark_preview": _truncate_for_log(remark_for_llm, 100),
            }
        )
        f_abs, f_delta = _fallback_parse_rules(remark_for_llm)
        if f_abs or f_delta:
            abs_list, delta_list = f_abs, f_delta

    return abs_list, delta_list

# ----------------------------------------------------------------------
# Step 5: 提取模式调度
# ----------------------------------------------------------------------

def _extract_price_parts_shop12_dispatch(
    remark_for_llm: str, idx: object = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], str]:
    """
    根据 SHOP12_EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 (abs_list, delta_list, extraction_method)
    """
    mode = SHOP12_EXTRACTION_MODE

    if mode == "regex":
        abs_list, delta_list = _extract_price_parts_shop12_regex(remark_for_llm)
        return abs_list, delta_list, "regex"

    if mode == "llm":
        abs_list, delta_list = _extract_price_parts_shop12_llm_with_guardrails(
            remark_for_llm, idx=idx,
        )
        return abs_list, delta_list, "llm"

    # ---- auto: 正则优先，正则无颜色结果时 LLM 兜底 ----
    abs_re, delta_re = _extract_price_parts_shop12_regex(remark_for_llm)
    if abs_re or delta_re:
        return abs_re, delta_re, "regex"

    abs_llm, delta_llm = _extract_price_parts_shop12_llm_with_guardrails(
        remark_for_llm, idx=idx,
    )
    return abs_llm, delta_llm, "llm"

# ----------------------------------------------------------------------
# Step 6: 颜色匹配
# ----------------------------------------------------------------------

EN_TO_JP = {
    "silver": ["シルバー", "銀"],
    "blue":   ["ブルー", "青", "ディープブルー"],
    "black":  ["ブラック", "黒"],
    "white":  ["ホワイト", "白"],
    "gold":   ["ゴールド", "金"],
    "green":  ["グリーン", "緑"],
    "red":    ["レッド", "赤"],
    "pink":   ["ピンク"],
    "purple": ["パープル", "紫"],
    "yellow": ["イエロー", "黄"],
    "orange": ["オレンジ", "橙"],
    "gray":   ["グレー", "グレイ", "灰"],
    "natural":["ナチュラル"],
}

def _label_matches_color(label_raw: str, color_raw: str, color_norm: str) -> bool:
    if not label_raw or not color_raw:
        return False
    lbl_raw = str(label_raw).strip()
    cr_raw  = str(color_raw).strip()

    # 英文直译
    label_lower = lbl_raw.lower()
    if label_lower in EN_TO_JP:
        for jp in EN_TO_JP[label_lower]:
            if jp in cr_raw:
                return True
            if _norm(jp) == color_norm:
                return True

    ln = _norm(lbl_raw)
    cn = color_norm
    if ln == cn:
        return True
    if lbl_raw in cr_raw or ln in cn or cn in ln:
        return True

    ln_short = re.sub(r"[\s\u3000]+", "", ln)
    cn_short = re.sub(r"[\s\u3000]+", "", cn)
    return bool(ln_short and (ln_short in cn_short or cn_short in ln_short))

# ----------------------------------------------------------------------
# Step 7: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop12(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop12 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop12"
    SHOP_NAME = "トゥインクル"

    logger.info(
        "Starting shop12 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    for c in ["モデルナンバー", "備考1", "買取価格", "time-scraped"]:
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
            raise ValueError(f"shop12 清洗器缺少必要列：{c}")

    # 载入 info 表并构建 (model_norm, cap)-> {color_norm: (pn, color_raw)}
    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        current_row_records: List[dict] = []
        price_base = to_int_yen(row.get("買取価格"))
        if price_base is None:
            continue

        model_text = str(row.get("モデルナンバー") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            _log_seq += 1
            logger.debug(
                f"Skip row: model/cap parse failed: {model_text!r}",
                extra={
                    "event_type": "row_processing_summary",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(idx),
                    "model_text": model_text,
                }
            )
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            _log_seq += 1
            logger.debug(
                f"Skip row: info no key={key}",
                extra={
                    "event_type": "row_processing_summary",
                    "log_seq": _log_seq,
                    "shop_name": SHOP_NAME,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(idx),
                    "model_text": model_text,
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                }
            )
            continue

        remark_raw = row.get("備考1") or ""
        remark_for_llm = _normalize_remark_for_llm(remark_raw)

        # 根据 SHOP12_EXTRACTION_MODE 提取价格信息（regex / llm / auto）
        abs_list, delta_list, extraction_method = _extract_price_parts_shop12_dispatch(
            remark_for_llm, idx=idx,
        )

        # 构建行级上下文
        row_context = {
            "row_index": int(idx),
            "model_text": model_text,
            "model_norm": model_norm,
            "capacity_gb": cap_gb,
            "base_price": int(price_base),
        }

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
                "row_index": int(idx),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": int(price_base),
                "remark_raw": _truncate_for_log(str(remark_raw), 200),
                "remark_raw_full": str(remark_raw),
                "remark_for_llm": _truncate_for_log(remark_for_llm, 200),
                "extraction_method": extraction_method,
                "abs_list": [
                    {"label": label, "amount": amt}
                    for label, amt in abs_list
                ],
                "delta_list": [
                    {"label": label, "delta": delta}
                    for label, delta in delta_list
                ],
                "abs_count": len(abs_list),
                "delta_count": len(delta_list),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_map),
            }
        )

        # label -> color_norm
        color_abs_map: Dict[str, int] = {}
        color_delta_map: Dict[str, int] = {}

        for label_raw, amt in abs_list:
            matched = None
            matched_pn = None
            for col_norm, (pn, col_raw) in color_map.items():
                if _label_matches_color(label_raw, col_raw, col_norm):
                    matched = col_norm
                    matched_pn = pn
                    break

            _log_seq += 1
            if matched:
                color_abs_map[matched] = int(amt)
                logger.debug(
                    f"Label matching (abs): {label_raw}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "label": label_raw,
                        "abs_price": amt,
                        "match_type": "abs",
                        "matched_colors": [matched],
                        "matched_part_numbers": [matched_pn],
                        "match_count": 1,
                        "remark_raw_full": str(remark_raw),
                    }
                )
            else:
                logger.warning(
                    f"Label not matched (abs): {label_raw}",
                    extra={
                        "event_type": "label_no_match",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "label": label_raw,
                        "abs_price": amt,
                        "match_type": "abs",
                        "available_colors": [cn for cn in color_map.keys()],
                        "remark_raw_full": str(remark_raw),
                    }
                )

        for label_raw, delta in delta_list:
            if str(label_raw).strip() in {"全色", "ALL"}:
                color_delta_map["ALL"] = int(delta)
                _log_seq += 1
                logger.debug(
                    f"Label matching (delta): ALL = {delta}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "label": "ALL",
                        "delta": delta,
                        "match_type": "delta",
                        "matched_colors": ["ALL"],
                        "match_count": 1,
                    }
                )
                continue
            matched = None
            matched_pn = None
            for col_norm, (pn, col_raw) in color_map.items():
                if _label_matches_color(label_raw, col_raw, col_norm):
                    matched = col_norm
                    matched_pn = pn
                    break

            _log_seq += 1
            if matched:
                color_delta_map[matched] = int(delta)
                logger.debug(
                    f"Label matching (delta): {label_raw}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "label": label_raw,
                        "delta": delta,
                        "match_type": "delta",
                        "matched_colors": [matched],
                        "matched_part_numbers": [matched_pn],
                        "match_count": 1,
                        "remark_raw_full": str(remark_raw),
                    }
                )
            else:
                logger.warning(
                    f"Label not matched (delta): {label_raw}",
                    extra={
                        "event_type": "label_no_match",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "label": label_raw,
                        "delta": delta,
                        "match_type": "delta",
                        "available_colors": [cn for cn in color_map.keys()],
                        "remark_raw_full": str(remark_raw),
                    }
                )

        recorded_at = parse_dt_aware(row.get("time-scraped"))

        # 生成输出记录
        output_records = []

        # 输出：ALL 差额优先
        if "ALL" in color_delta_map:
            final_price = int(price_base + color_delta_map["ALL"])
            for col_norm, (pn, _) in color_map.items():
                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_text": model_text,
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "base_price": int(price_base),
                        "delta": color_delta_map["ALL"],
                        "final_price": final_price,
                        "delta_source": "ALL",
                        "recorded_at": str(recorded_at) if recorded_at else None,
                        "remark_raw_full": str(remark_raw),
                    }
                )

                output_records.append({
                    "part_number": pn,
                    "color_norm": col_norm,
                    "delta": color_delta_map["ALL"],
                    "final_price": final_price,
                    "delta_source": "ALL",
                })

                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": final_price, "recorded_at": recorded_at})

                current_row_records.append({
                    "part_number": pn,
                    "color_norm": col_norm,
                    "delta": color_delta_map["ALL"],
                    "final_price": final_price,
                    "recorded_at": recorded_at,
                    "delta_source": "ALL",
                })

        # 绝对价覆盖
        elif color_abs_map:
            for col_norm, (pn, _) in color_map.items():
                if col_norm in color_abs_map:
                    val = int(color_abs_map[col_norm])
                    delta_source = "abs_price"
                    delta = 0
                else:
                    val = int(price_base)
                    delta_source = "base_fallback"
                    delta = 0

                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_text": model_text,
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "base_price": int(price_base),
                        "delta": delta,
                        "final_price": val,
                        "delta_source": delta_source,
                        "recorded_at": str(recorded_at) if recorded_at else None,
                        "remark_raw_full": str(remark_raw),
                    }
                )

                output_records.append({
                    "part_number": pn,
                    "color_norm": col_norm,
                    "delta": delta,
                    "final_price": val,
                    "delta_source": delta_source,
                })

                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": val, "recorded_at": recorded_at})

                current_row_records.append({
                    "part_number": pn,
                    "color_norm": col_norm,
                    "delta": delta,
                    "final_price": val,
                    "recorded_at": recorded_at,
                    "delta_source": delta_source,
                })

        # 普通差额
        else:
            for col_norm, (pn, _) in color_map.items():
                delta = color_delta_map.get(col_norm, 0)
                val = int(price_base + delta)
                delta_source = "matched_label" if col_norm in color_delta_map else "default_zero"

                _log_seq += 1
                logger.debug(
                    f"Output record: {pn}",
                    extra={
                        "event_type": "output_record",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_text": model_text,
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "base_price": int(price_base),
                        "delta": delta,
                        "final_price": val,
                        "delta_source": delta_source,
                        "recorded_at": str(recorded_at) if recorded_at else None,
                        "remark_raw_full": str(remark_raw),
                    }
                )

                output_records.append({
                    "part_number": pn,
                    "color_norm": col_norm,
                    "delta": delta,
                    "final_price": val,
                    "delta_source": delta_source,
                })

                rows.append({"part_number": pn, "shop_name": SHOP_NAME, "price_new": val, "recorded_at": recorded_at})

                current_row_records.append({
                    "part_number": pn,
                    "color_norm": col_norm,
                    "delta": delta,
                    "final_price": val,
                    "recorded_at": recorded_at,
                    "delta_source": delta_source,
                })

        # DEBUG: 行级详细汇总
        all_deltas_values = list(color_delta_map.values())
        colors_matched = len(color_delta_map) + len(color_abs_map)

        _log_seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(idx),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": int(price_base),
                "remark_raw_full": str(remark_raw),
                "current_row_records": [
                    {"pn": r["part_number"], "color": r["color_norm"], "delta": r["delta"], "final_price": r["final_price"], "src": r["delta_source"]}
                    for r in current_row_records
                ],
            }
        )

        # INFO: 行级概览（简洁）
        _log_seq += 1
        logger.info(
            f"Row {idx:<3d} | {model_text:<28s} | deltas: {len(delta_list):<2d} | abs: {len(abs_list):<2d} | matched: {colors_matched:<2d} | records: {len(output_records):<2d} | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(idx),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": int(price_base),
                "remark_raw_preview": _truncate_for_log(str(remark_raw), 100),
                "extraction_method": extraction_method,
                "deltas_extracted_count": len(delta_list),
                "abs_prices_extracted_count": len(abs_list),
                "colors_in_catalog": len(color_map),
                "colors_matched_count": colors_matched,
                "output_records_count": len(output_records),
                "has_discounted_colors": any(d != 0 for d in all_deltas_values),
                "min_delta": min(all_deltas_values) if all_deltas_values else 0,
                "max_delta": max(all_deltas_values) if all_deltas_values else 0,
            }
        )

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed_time = time.time() - start_time
    logger.info(
        "Shop12 cleaner completed",
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
