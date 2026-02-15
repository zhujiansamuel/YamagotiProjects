from __future__ import annotations

"""
shop12 清洗器 — トゥインクル

  原始文本（備考1 + 買取価格）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _normalize_remark_for_llm()              ← Step 1: 去除開封行，预处理備考1
    │
    ├─ _norm_amount_to_int()                    ← Step 2: 统一全角数字→int
    │
    ├─ _extract_specs_shop12_dispatch()   ← Step 5: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   └─ _extract_specs_shop12_regex()    ← Step 3: 正则提取 (abs + delta)
    │   │       └─ _fallback_parse_rules()            ← 核心正则: _FALLBACK_ABS_RE / _FALLBACK_DELTA_RE
    │   │
    │   └─ llm 路径:
    │       └─ _extract_specs_shop12_llm()  ← Step 4: LLM 提取 + 防幻觉
    │           └─ _extract_specs_shop12_llm_core()    ← LLM 核心: effective_class 修正 + 去重
    │
    ├─ _label_matches_color_unified()           ← Step 6: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop12()                           ← Step 7: 主函数，生成输出行
"""

import logging
import os
import re
import time
import textwrap
from functools import lru_cache
from typing import List, Optional, Tuple

import pandas as pd

from ...external_ingest.helpers import parse_dt_aware
from ..cleaner_tools import (
    extract_price_yen,
    _parse_capacity_gb,
    _normalize_model_generic,
    _load_iphone17_info_df_from_db,
    _build_color_map,
    _truncate_for_log,
    _norm_strip,
    normalize_text_basic,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    assemble_output_df,
    log_cleaner_start,
    log_cleaner_complete,
    validate_columns,
    log_row_skip,
    LABEL_SPLIT_RE_shop12,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

CLEANER_NAME = "shop12"
SHOP_NAME = "トゥインクル"

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
    """
    解析金额文本，使用通用规范化函数
    """
    if s is None:
        return None
    # 使用通用规范化（全角→半角 + 去换行 + 合并空格）
    tt = normalize_text_basic(str(s))
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
# LABEL_SPLIT_RE_shop12: 从 cleaner_tools 导入

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
            toks = [t.strip() for t in LABEL_SPLIT_RE_shop12.split(labels_part) if t.strip()]
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
            toks = [t.strip() for t in LABEL_SPLIT_RE_shop12.split(labels_part) if t.strip()]
            for tok in toks:
                if tok:
                    delta_list.append((tok, delta))

    return abs_list, delta_list

def _extract_specs_shop12_regex(
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
def _extract_specs_shop12_llm_core(remark_for_llm: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, str, dict]]]:
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

        llm_input = "色別価格ルール:\n" + remark_for_llm

        res = lx.extract(
            text_or_documents=llm_input,
            prompt_description=_LX_PROMPT,
            examples=_lx_examples(),
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
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
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "error": str(e),
                "error_type": type(e).__name__,
                "text_length": len(remark_for_llm),
                "text_preview": _truncate_for_log(remark_for_llm, 100),
            }
        )
        return [], [], []

def _extract_specs_shop12_llm(
    remark_for_llm: str,
    idx: object = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    LLM 提取 + Guardrails（仅 LLM 路径使用）。
    Guardrail: effective_class 修正 + 去重（内置于 _extract_specs_shop12_llm_core）。
    LLM 失败时回退到正则。
    """
    abs_list, delta_list, _llm_dbg = _extract_specs_shop12_llm_core(remark_for_llm)

    # Guardrail: LLM 完全失败（空结果）时，回退到正则
    if not abs_list and not delta_list and remark_for_llm:
        logger.debug(
            "LLM returned empty, falling back to regex",
            extra={
                "event_type": "llm_extraction_error",
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
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

def _extract_specs_shop12_dispatch(
    remark_for_llm: str,
    *,
    base_price: int,
    source_text_raw: str,
    idx: object = None,
) -> PriceDecomposition:
    """
    根据 EXTRACTION_MODE 决定提取方式：
      - "regex": 只用正则
      - "llm":   只用 LLM + Guardrails
      - "auto":  正则优先，正则无颜色结果时 LLM + Guardrails 兜底

    返回 PriceDecomposition
    """
    mode = EXTRACTION_MODE

    if mode == "regex":
        abs_list, delta_list = _extract_specs_shop12_regex(remark_for_llm)
        method = "regex"
    elif mode == "llm":
        abs_list, delta_list = _extract_specs_shop12_llm(
            remark_for_llm, idx=idx,
        )
        method = "llm"
    else:
        # ---- auto: 正则优先，正则无颜色结果时 LLM 兜底 ----
        abs_list, delta_list = _extract_specs_shop12_regex(remark_for_llm)
        if abs_list or delta_list:
            method = "regex"
        else:
            abs_list, delta_list = _extract_specs_shop12_llm(
                remark_for_llm, idx=idx,
            )
            method = "llm"

    # ---- "全色" delta → 覆盖全部，清空 abs ----
    delta_specs: List[Tuple[str, int]] = []
    abs_specs: List[Tuple[str, int]] = list(abs_list)

    for label_raw, delta_val in delta_list:
        if str(label_raw).strip() in {"全色", "ALL"}:
            delta_specs = [("全色", int(delta_val))]
            abs_specs = []
            break
        delta_specs.append((label_raw, int(delta_val)))
    else:
        delta_specs = [(lb, int(d)) for lb, d in delta_list]

    return PriceDecomposition(
        base_price=base_price,
        delta_specs=delta_specs,
        abs_specs=abs_specs,
        extraction_method=method,
        source_text_raw=source_text_raw,
    )

# ----------------------------------------------------------------------
# Step 6: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop12 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 7: 清洗主函数
# ----------------------------------------------------------------------

def clean_shop12(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    t_start = time.time()
    _log_seq = 0

    log_cleaner_start(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), log_seq=_log_seq, extraction_mode=EXTRACTION_MODE)
    _log_seq += 1

    _log_seq = validate_columns(df, ["モデルナンバー", "備考1", "買取価格", "time-scraped"],
                                cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                                logger=logger, log_seq=_log_seq)

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        base_price = extract_price_yen(row.get("買取価格"))
        if base_price is None:
            continue
        base_price = int(base_price)

        model_text = str(row.get("モデルナンバー") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            _log_seq += 1
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=idx, skip_reason="model_or_cap_parse_failed", log_seq=_log_seq,
                         model_text=model_text)
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            _log_seq += 1
            log_row_skip(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
                         row_index=idx, skip_reason="no_info_key", log_seq=_log_seq,
                         model_text=model_text, model_norm=model_norm, capacity_gb=cap_gb)
            continue

        remark_raw = row.get("備考1") or ""
        remark_for_llm = _normalize_remark_for_llm(remark_raw)
        source_text_raw_full = str(remark_raw)

        decomp = _extract_specs_shop12_dispatch(
            remark_for_llm,
            base_price=base_price,
            source_text_raw=source_text_raw_full,
            idx=idx,
        )

        rec_at = parse_dt_aware(row.get("time-scraped"))

        new_rows, _log_seq = resolve_color_prices(
            decomp, color_map, _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            logger=logger,
            log_seq_start=_log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    # ---- 输出 DataFrame 组装 ----
    out = assemble_output_df(rows)

    log_cleaner_complete(logger, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME, input_rows=len(df), output_records=len(out), start_time=t_start, log_seq=_log_seq)

    return out
