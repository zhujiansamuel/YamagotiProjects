from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _load_iphone17_info_df_from_db,
    _parse_capacity_gb,
    _truncate_for_log,
    _normalize_model_generic,
    _build_color_map,
    normalize_text_basic,
    extract_price_yen,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    LABEL_SPLIT_RE_shop17 as SPLIT_TOKENS_RE_shop17,
    OLLAMA_URL,
    OLLAMA_MODEL_ID,
    EXTRACTION_MODE,
)
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
import pytz
import time
import textwrap
import logging



"""
shop17 清洗器 — ゲストモバイル

  原始文本（type / 新未開封品 / 色減額）
    │ 配置: EXTRACTION_MODE / OLLAMA_URL / OLLAMA_MODEL_ID (cleaner_tools)
    │
    ├─ _normalize_model_generic() / _parse_capacity_gb()  ← Step 1: 机型・容量解析（cleaner_tools）
    │
    ├─ extract_price_yen()         ← Step 2: 基础价提取（cleaner_tools）
    │
    ├─ _extract_color_deltas_shop17()  ← Step 3: 模式调度（EXTRACTION_MODE）
    │   │
    │   ├─ regex 路径:
    │   │   ├─ _pick_unopened_section()     ← 提取【未開封】段
    │   │   ├─ _normalize_color_text_shop17()  ← 归一化
    │   │   ├─ SPLIT_TOKENS_RE 拆分          ← 分割多条目
    │   │   └─ COLOR_NONE_RE / COLOR_DELTA_RE  ← なし模式・金额模式
    │   │
    │   └─ llm 路径:
    │       └─ _extract_color_deltas_shop17_llm()  ← LangExtract 核心提取
    │
    ├─ _label_matches_color_unified()  ← Step 4: 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop17()              ← Step 5: 主函数，生成输出行
"""

# 初始化 logger
logger = logging.getLogger(__name__)

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

SHOP_NAME_OVERRIDE: Optional[str] = "ゲストモバイル"

# ----------------------------------------------------------------------
# 正则表达式与辅助函数（按处理流程排列）
# ----------------------------------------------------------------------

# ── Step 1: 提取【未開封】段落 ──
def _pick_unopened_section(text: str) -> str:
    """若包含【未開封】…，取该段直到下一个 '【' 或行末；否则返回原文。"""
    if not text:
        return ""
    s = str(text)
    m = re.search(r"【\s*未開封\s*】(.*?)(?=【|$)", s, flags=re.DOTALL)
    return m.group(1) if m else s

# ── Step 2: 归一化色減額文本 ──
def _normalize_color_text_shop17(s: str) -> str:
    """
    统一色減額文本里的全角数字/逗号/各种 dash，顺便清理空白。
    使用通用规范化函数（全角→半角）。
    保留换行与空白结构（remove_newlines=False, collapse_spaces=False），
    以便 SPLIT_TOKENS_RE 能按 \\n 正确切分多段。
    """
    if s is None:
        return ""
    # 色減額 split 前保留换行，否则「ブルー-1000」与「△減額なし」会合并到同一 part
    return normalize_text_basic(
        str(s), remove_newlines=False, collapse_spaces=False
    )

# ── Step 3: 归一化颜色标签（清除空白） ──
def _normalize_label_shop17(lbl: str) -> str:
    return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

# ── Step 4: 验证颜色标签合理性 ──
_BAD_LABEL_WORDS_shop17 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")

def _is_plausible_color_label_shop17(label: str) -> bool:
    """过滤掉明显不是"颜色名"的 label（比如 利用制限△ / 保証開始3か月未満 等）。"""
    label = _normalize_label_shop17(label)
    if not label:
        return False
    if label.startswith(("△", "▲")):
        return False
    if re.search(r"\d", label):
        return False
    if len(label) > 16:
        return False
    if any(w in label for w in _BAD_LABEL_WORDS_shop17):
        return False
    return True

# ── Step 5: 分割多颜色条目 ──
# SPLIT_TOKENS_RE_shop17: 从 cleaner_tools.LABEL_SPLIT_RE_shop17 导入

# ── Step 6: 匹配无减额颜色（なし模式） ──
COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# ── Step 7: 匹配有金额减额的颜色 ──
COLOR_DELTA_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# ----------------------------------------------------------------------
# LangExtract / Ollama 集成配置
# ----------------------------------------------------------------------

try:
    import langextract as lx
    from langextract.data import ExampleData, Extraction
    _HAS_LANGEXTRACT = True
except Exception:
    lx = None
    ExampleData = None
    Extraction = None
    _HAS_LANGEXTRACT = False

# ----------------------------------------------------------------------
# 颜色匹配函数
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop17 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

def _extract_color_deltas_shop17_regex(text: str) -> List[Tuple[str, int]]:
    """
    正则版提取 [(label_raw, delta_int)]，作为 LLM 的 fallback，也可以单独使用。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = _normalize_color_text_shop17(_pick_unopened_section(str(text)))

    if "色減額" in s:
        s = s.split("色減額", 1)[-1].lstrip(":：")

    # 整段就是「なし/減額なし」-> 无色差额
    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        return out

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop17.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    for part in parts:
        # 「シルバーなし」/「クラウドホワイト：なし」
        m0 = COLOR_NONE_RE_shop17.search(part)
        if m0:
            label = _normalize_label_shop17(m0.group("label"))
            if _is_plausible_color_label_shop17(label):
                out.append((label, 0))
            continue

        # 「ブルー-1000」「スカイブルー: -3,000」 等
        for m in COLOR_DELTA_RE_shop17.finditer(part):
            label = _normalize_label_shop17(m.group("label"))
            if not _is_plausible_color_label_shop17(label):
                continue
            sep = m.group("sep")
            sign = m.group("sign")
            amt = to_int_yen(m.group("amount"))
            if amt is None:
                continue
            if sign:
                negative = sign in ("-", "−", "－")
            else:
                negative = sep in ("-", "−", "－") if sep else False
            delta = -int(amt) if negative else int(amt)
            out.append((label, delta))

    return out

# ----------------------------------------------------------------------
# LangExtract + Ollama: LLM 驱动的颜色差额抽取
# ----------------------------------------------------------------------
COLOR_DELTA_PROMPT_SHOP17 = textwrap.dedent("""
あなたは中古iPhone買取表の「色減額」欄を解析するアシスタントです。
入力は1つのセルのテキストです。この中には色ごとの減額情報のほかに、
「郵送は翌日着のみ保証」「持ち込みのみ保証」「利用制限△-10000」などの
色と関係ない条件も含まれます。

タスク:
- 色名ごとの減額（または増額）だけを抽出してください。
- 色名の例: スカイブルー, スペースブラック, クラウドホワイト, ライトゴールド, シルバー, ブルー など。
- 「利用制限△-10000」や「保証開始3か月未満減額なし」など、色と無関係な金額・文言は無視してください。
- 「色名なし」(例: シルバーなし) はその色の delta=0 として扱います。
- 色名が付いていない「減額なし」(例: △減額なし) は無視します。

出力ポリシー:
- extraction_class は必ず "color_delta" にしてください。
- extraction_text には、表に書かれている「色と金額のフレーズ全体」
  （例: "スカイブルー-3,000", "クラウドホワイト：なし", "シルバーなし"）をそのまま入れてください。
- attributes には必ず次のキーを入れてください:
  - "color": 色名だけ（例: "スカイブルー"）
  - "delta": その色の価格差（整数。値引きは負の数。例: -3000）
  - "raw": 抜き出した元の部分文字列（extraction_text と同じでもよい）

その他ルール:
- 価格は円単位で扱い、「円」「,」などは無視して整数に変換してください。
- 色名が複数ある場合は、それぞれ1つずつ color_delta を出力してください。
- 文章内の改行や空行は無視して構いません。
""").strip()

@lru_cache()
def _get_color_delta_examples_shop17() -> List[ExampleData]:
    if not _HAS_LANGEXTRACT:
        return []

    examples: List[ExampleData] = []

    # Example 0: スカイブルーのみ
    examples.append(
        ExampleData(
            text="色減額:スカイブルー-3,000\n\n郵送は翌日着のみ保証\n\n利用制限△-10000",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー-3,000",
                    attributes={
                        "color": "スカイブルー",
                        "delta": "-3000",
                        "raw": "スカイブルー-3,000",
                    },
                )
            ],
        )
    )

    # Example 1: 2 色 + 利用制限△
    examples.append(
        ExampleData(
            text="色減額:スカイブルー-4,000/スペースブラック-4,000\n\n持ち込みのみ保証\n\n利用制限△-10000",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー-4,000",
                    attributes={
                        "color": "スカイブルー",
                        "delta": "-4000",
                        "raw": "スカイブルー-4,000",
                    },
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スペースブラック-4,000",
                    attributes={
                        "color": "スペースブラック",
                        "delta": "-4000",
                        "raw": "スペースブラック-4,000",
                    },
                ),
            ],
        )
    )

    # Example 2: 你这条问题里的真实样例
    examples.append(
        ExampleData(
            text="色減額:シルバーなし/ブルー-1000\n\n郵送は翌日着のみ保証\n\n△減額なし 保証開始3か月未満減額なし",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバーなし",
                    attributes={
                        "color": "シルバー",
                        "delta": "0",
                        "raw": "シルバーなし",
                    },
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー-1000",
                    attributes={
                        "color": "ブルー",
                        "delta": "-1000",
                        "raw": "ブルー-1000",
                    },
                ),
            ],
        )
    )

    return examples

def _parse_delta_attr_to_int(val) -> Optional[int]:
    if val is None:
        return None
    s = str(val)
    s = s.replace("円", "").replace(",", "").replace(" ", "").replace("　", "")
    s = s.replace("−", "-").replace("－", "-")
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None

def _extract_color_deltas_shop17_llm(
    text: str,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None
) -> List[Tuple[str, int]]:
    if not _HAS_LANGEXTRACT:
        return []
    if not text or not str(text).strip():
        return []

    s = _normalize_color_text_shop17(_pick_unopened_section(str(text)))

    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        return []

    import langextract as lx

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=COLOR_DELTA_PROMPT_SHOP17,
            examples=_get_color_delta_examples_shop17(),
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            temperature=0.0,
            fence_output=False,
            use_schema_constraints=False,
            # 这里是关键：关闭 few-shot 对齐校验，避免 WARNING
            prompt_validation_level="OFF",
            prompt_validation_strict=False,
        )
    except Exception as e:
        log_extra = {
            "event_type": "llm_extraction_error",
            "error": str(e),
            "error_type": type(e).__name__,
            "model_id": OLLAMA_MODEL_ID,
            "model_url": OLLAMA_URL,
            "text_length": len(s),
            "text_preview": _truncate_for_log(s, 100),
        }
        # 添加上下文信息（如果提供）
        if shop_name:
            log_extra["shop_name"] = shop_name
        if cleaner_name:
            log_extra["cleaner_name"] = cleaner_name
        if row_context:
            log_extra.update(row_context)

        logger.warning(
            "LangExtract extraction failed",
            extra=log_extra
        )
        return []

    out: List[Tuple[str, int]] = []
    extractions = getattr(result, "extractions", None) or []
    for ext in extractions:
        try:
            if ext.extraction_class != "color_delta":
                continue
            attrs = ext.attributes or {}
            color = (attrs.get("color") or ext.extraction_text or "").strip()
            if not _is_plausible_color_label_shop17(color):
                continue
            delta_int = _parse_delta_attr_to_int(attrs.get("delta"))
            if delta_int is None:
                # fallback：从 extraction_text 再捞一次金额
                txt = (ext.extraction_text or "").strip()
                m = re.search(r"([+\-−－]?\d[\d,]*)", txt)
                if m:
                    delta_int = _parse_delta_attr_to_int(m.group(1))
            if delta_int is None:
                continue
            out.append((color, delta_int))
        except Exception:
            continue

    return out

def _extract_color_deltas_shop17(
    text: str,
    shop_name: Optional[str] = None,
    cleaner_name: Optional[str] = None,
    row_context: Optional[Dict] = None
) -> List[Tuple[str, int]]:
    """
    根据 EXTRACTION_MODE 决定提取方式：
    - "regex": 只用正则
    - "llm":   只用 LLM
    - "auto":  正则优先，正则无结果时 LLM 兜底
    """
    if EXTRACTION_MODE == "regex":
        return _extract_color_deltas_shop17_regex(text)
    elif EXTRACTION_MODE == "llm":
        return _extract_color_deltas_shop17_llm(text, shop_name, cleaner_name, row_context)
    else:  # auto
        regex_res = _extract_color_deltas_shop17_regex(text)
        if regex_res:
            return regex_res
        return _extract_color_deltas_shop17_llm(text, shop_name, cleaner_name, row_context)
# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------
def clean_shop17(df: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()
    _log_seq = 0  # 日志序号：同一次 clean_shop17 调用内单调递增，用于 ELK 排序

    # 定义清洗器级别的上下文信息，将被所有下级日志继承
    CLEANER_NAME = "shop17"
    SHOP_NAME = "ゲストモバイル"

    logger.info(
        "Starting shop17 cleaner",
        extra={
            "event_type": "cleaner_start",
            "shop_name": SHOP_NAME,
            "cleaner_name": CLEANER_NAME,
            "input_rows": len(df),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )

    for c in ["type", "新未開封品", "色減額", "time-scraped"]:
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
            raise ValueError(f"shop17 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_from_db()
    cmap_all = _build_color_map(info_df)
    rows: List[dict] = []

    for idx, row in df.iterrows():
        model_text = str(row.get("type") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        base_price = extract_price_yen(row.get("新未開封品"))
        if base_price is None:
            continue
        base_price = int(base_price)

        raw_color = row.get("色減額")
        raw_color_s = "" if raw_color is None else str(raw_color)

        # 构建行级上下文，用于传递给下级函数和日志
        row_context = {
            "row_index": int(idx),
            "model_text": model_text,
            "model_norm": model_norm,
            "capacity_gb": cap_gb,
            "base_price": base_price,
        }

        # 提取颜色差额
        labels_and_deltas = _extract_color_deltas_shop17(
            raw_color_s,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            row_context=row_context
        )

        # 判断使用的提取方法
        if not labels_and_deltas:
            extraction_method = "none"
        elif EXTRACTION_MODE in ("regex", "llm"):
            extraction_method = EXTRACTION_MODE
        else:  # auto: 需要判断结果来自哪个方法
            regex_result = _extract_color_deltas_shop17_regex(raw_color_s)
            extraction_method = "regex" if regex_result else "llm"

        shop_name = SHOP_NAME_OVERRIDE or (urlparse(str(row.get("web-scraper-start-url") or "")).netloc or "shop17")
        rec_at = parse_dt_aware(row.get("time-scraped"))

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=labels_and_deltas,
            abs_specs=[],
            extraction_method=extraction_method,
            source_text_raw=raw_color_s,
        )

        new_rows, _log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=shop_name,
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

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    elapsed_time = time.time() - start_time
    logger.info(
        f"Shop17 cleaner completed",
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
