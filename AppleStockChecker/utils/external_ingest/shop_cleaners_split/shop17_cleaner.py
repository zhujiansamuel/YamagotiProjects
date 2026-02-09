from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _load_iphone17_info_df_from_db
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

# 初始化 logger
logger = logging.getLogger(__name__)




# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

_DASH_TRANS_shop17 = str.maketrans({
    "−": "-",  # U+2212
    "－": "-",  # U+FF0D
    "‐": "-",  # U+2010
    "‑": "-",  # U+2011
    "‒": "-",  # U+2012
    "–": "-",  # U+2013
    "—": "-",  # U+2014
})
_FW_DIGITS_TRANS_shop17 = str.maketrans("０１２３４５６７８９", "0123456789")

_BAD_LABEL_WORDS_shop17 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")

def _normalize_color_text_shop17(s: str) -> str:
    """统一色減額文本里的全角数字/逗号/各种 dash，顺便清理空白。"""
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(_FW_DIGITS_TRANS_shop17)
    s = s.replace("，", ",")
    s = s.translate(_DASH_TRANS_shop17)
    s = s.replace("／", "/")
    s = re.sub(r"[ \t\u3000\xa0]+", " ", s)
    return s.strip()

def _is_plausible_color_label_shop17(label: str) -> bool:
    """过滤掉明显不是“颜色名”的 label（比如 利用制限△ / 保証開始3か月未満 等）。"""
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

# 调试辅助函数：截断长字符串用于日志显示
def _truncate_for_log(s: str, n: int = 200) -> str:
    """截断长字符串，保留前 n 个字符，用于日志显示"""
    if s is None:
        return ""
    t = str(s)
    if len(t) <= n:
        return t
    return t[:n] + f"... (truncated, total_length={len(t)})"

# ----------------------------------------------------------------------
# LangExtract / Ollama 集成配置
# ----------------------------------------------------------------------

COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_DELTA_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

try:
    import langextract as lx
    from langextract.data import ExampleData, Extraction
    _HAS_LANGEXTRACT = True
except Exception:
    lx = None
    ExampleData = None
    Extraction = None
    _HAS_LANGEXTRACT = False

LX_SHOP17_MODEL_ID = os.getenv("SHOP17_LX_MODEL_ID", "gemma3:1b")
LX_SHOP17_MODEL_URL = os.getenv("SHOP17_LX_MODEL_URL", "http://localhost:11434")
USE_LLM_FOR_SHOP17 = str(os.getenv("SHOP17_USE_LLM", "1")).strip().lower() not in {"0", "false", "no"}

_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)

def _normalize_model_generic(text: str) -> str:
    """
    统一型号主体：
      - iPhone17/16 + 后缀（Pro/Pro Max/Plus/mini）
      - iPhone Air（含“17 air”→ Air）
      - 允许紧凑写法：17pro / 17promax / 16Pro / 16Plus ...
    输出：'iPhone 17 Pro Max' / 'iPhone 17 Pro' / 'iPhone Air' / ...
    """
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    # 日文别名到英文
    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini")
           .replace("エアー", "Air")
           .replace("エア", "Air"))

    # 数字后紧跟英文：17pro -> 17 pro
    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)

    # 标准化后缀大小写
    t = re.sub(r"(?i)\bpro\s*max\b", "Pro Max", t)
    t = re.sub(r"(?i)\bpro\b", "Pro", t)
    t = re.sub(r"(?i)\bplus\b", "Plus", t)
    t = re.sub(r"(?i)\bmini\b", "mini", t)

    # 若没有 iPhone 前缀但出现纯数字代号，补上
    if "iPhone" not in t and re.search(r"\b1[0-9]\b", t):
        t = re.sub(r"\b(1[0-9])\b", r"iPhone \1", t, count=1)

    # 特例：'17 air' → iPhone Air（防止被当成 iPhone 17）
    t = re.sub(r"(?i)\biPhone\s+17\s+Air\b", "iPhone Air", t)

    # 去容量/SIM/括号噪声
    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 1) 数字代号机型
    m = _NUM_MODEL_PAT.search(t)
    if m:
        base = f"{m.group(1)} {m.group(2)}"
        suf  = (m.group(3) or "").strip()
        return f"{base} {suf}".strip()

    # 2) Air
    m2 = _AIR_PAT.search(t)
    if m2:
        return "iPhone Air"

    return ""

def _parse_capacity_gb(text: str) -> Optional[int]:
    if not text:
        return None
    t = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", t, flags=re.I)
    if m:
        return int(round(float(m.group(1)) * 1024))
    m = re.search(r"(\d{2,4})\s*GB", t, flags=re.I)
    if m:
        return int(m.group(1))
    return None

SHOP_NAME_OVERRIDE: Optional[str] = "ゲストモバイル"  # 例如 "ゲストモバイル"

FAMILY_SYNONYMS_shop17 = {
    # blue 家族
    "blue": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "ブルー": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "青": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "ミッドナイト": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],
    "マリン": ["ブルー", "青", "ミッドナイト", "マリン", "ミストブルー"],

    # black
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],

    # white / starlight
    "white": ["ホワイト", "白", "スターライト", "Starlight", "starlight"],
    "ホワイト": ["ホワイト", "白", "スターライト"],
    "白": ["ホワイト", "白", "スターライト"],
    "スターライト": ["ホワイト", "白", "スターライト"],
    "starlight": ["ホワイト", "白", "スターライト"],

    # silver
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],

    # gold / light gold
    "gold": ["ゴールド", "金", "ライトゴールド"],
    "ゴールド": ["ゴールド", "金", "ライトゴールド"],
    "ライトゴールド": ["ゴールド", "金", "ライトゴールド"],

    # orange
    "orange": ["オレンジ", "橙"],
    "オレンジ": ["オレンジ", "橙"],
    "橙": ["オレンジ", "橙"],

    # green / セージ
    "green": ["グリーン", "緑", "セージ"],
    "グリーン": ["グリーン", "緑", "セージ"],
    "緑": ["グリーン", "緑", "セージ"],
    "セージ": ["グリーン", "緑", "セージ"],

    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],

    # yellow
    "yellow": ["イエロー", "黄", "黄色"],
    "イエロー": ["イエロー", "黄", "黄色"],
    "黄": ["イエロー", "黄", "黄色"],
    "黄色": ["イエロー", "黄", "黄色"],

    # purple / lavender
    "purple": ["パープル", "紫", "ラベンダー"],
    "パープル": ["パープル", "紫", "ラベンダー"],
    "紫": ["パープル", "紫", "ラベンダー"],
    "ラベンダー": ["パープル", "紫", "ラベンダー"],

    # natural
    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],

    # space black
    "spaceblack": ["スペースブラック"],
    "スペースブラック": ["スペースブラック"],
}
def _norm(s: str) -> str:
    return (s or "").strip()

def _label_matches_color_shop17(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """宽松匹配：精确(归一) | 原文子串 | 颜色家族关键词命中"""
    label_norm = _norm(label_raw)
    if label_norm == color_norm:
        return True
    if label_raw and str(label_raw) in str(color_raw):
        return True
    keys = {label_raw.strip(), label_raw.strip().lower(), label_norm}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop17:
            candidates.update(FAMILY_SYNONYMS_shop17[k])
    if not candidates:
        for _, toks in FAMILY_SYNONYMS_shop17.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in toks):
                candidates.update(toks)
                break
    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map_shop17(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """(model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }"""
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df["color_norm"] = df["color"].map(lambda x: _norm(str(x)))
    cmap: Dict[Tuple[str, int], Dict[str, Tuple[str, str]]] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        if not m or pd.isna(cap):
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        cmap[key][_norm(str(r["color"]))] = (str(r["part_number"]), str(r["color"]))
    return cmap

def _pick_unopened_section(text: str) -> str:
    """若包含【未開封】…，取该段直到下一个 '【' 或行末；否则返回原文。"""
    if not text:
        return ""
    s = str(text)
    m = re.search(r"【\s*未開封\s*】(.*?)(?=【|$)", s, flags=re.DOTALL)
    return m.group(1) if m else s

# ----------------------------------------------------------------------
# 原来的 正则解析器（保留为 fallback）
# ----------------------------------------------------------------------
SPLIT_TOKENS_RE_shop17 = re.compile(r"[／/、]|(?:\s*;\s*)|\n")

def _normalize_label_shop17(lbl: str) -> str:
    return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

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
    if not _HAS_LANGEXTRACT or not USE_LLM_FOR_SHOP17:
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
            model_id=LX_SHOP17_MODEL_ID,
            model_url=LX_SHOP17_MODEL_URL,
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
            "model_id": LX_SHOP17_MODEL_ID,
            "model_url": LX_SHOP17_MODEL_URL,
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
    优先使用正则（对像「シルバーなし/ブルー-1000」这种结构非常稳定），
    当正则完全解析不到任何颜色差额时，再让 LangExtract + Ollama 兜底。
    """
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
    cmap_all = _build_color_map_shop17(info_df)
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

        base_price = to_int_yen(row.get("新未開封品"))
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
        extraction_method = "none"
        if labels_and_deltas:
            # 检查是否来自正则
            regex_result = _extract_color_deltas_shop17_regex(raw_color_s)
            if regex_result:
                extraction_method = "regex"
            else:
                extraction_method = "llm"

        # DEBUG: 记录提取结果
        available_colors_list = [
            {
                "color_norm": cn,
                "part_number": pn,
                "color_raw": cr
            }
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
                "base_price": base_price,
                "color_discount_raw": _truncate_for_log(raw_color_s, 200),
                "color_discount_raw_full": raw_color_s,  # 完整版本
                "color_discount_normalized": _truncate_for_log(_normalize_color_text_shop17(raw_color_s), 200),
                "extraction_method": extraction_method,
                "labels_and_deltas": [
                    {"label": label, "delta": delta}
                    for label, delta in labels_and_deltas
                ],
                "labels_extracted_count": len(labels_and_deltas),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_map),
            }
        )

        # label -> color 命中 & 计算 delta
        color_deltas: Dict[str, int] = {}

        if labels_and_deltas:
            for label_raw, delta in labels_and_deltas:
                matched_colors = []
                matched_pns = []

                for col_norm, (pn, col_raw) in color_map.items():
                    if _label_matches_color_shop17(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta
                        matched_colors.append(col_norm)
                        matched_pns.append(pn)

                # DEBUG: 详细的 label 匹配日志
                _log_seq += 1
                logger.debug(
                    f"Label matching: {label_raw}",
                    extra={
                        "event_type": "label_matching",
                        "log_seq": _log_seq,
                        "shop_name": SHOP_NAME,
                        "cleaner_name": CLEANER_NAME,
                        "row_index": int(idx),
                        "model_text": model_text,
                        "model_norm": model_norm,
                        "capacity_gb": cap_gb,
                        "base_price": base_price,
                        "label": label_raw,
                        "delta": delta,
                        "matched_colors": matched_colors,
                        "matched_part_numbers": matched_pns,
                        "match_count": len(matched_colors),
                    }
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
                            "row_index": int(idx),
                            "model_text": model_text,
                            "model_norm": model_norm,
                            "capacity_gb": cap_gb,
                            "base_price": base_price,
                            "label": label_raw,
                            "delta": delta,
                            "available_colors": [cn for cn in color_map.keys()],
                        }
                    )

        shop_name = SHOP_NAME_OVERRIDE or (urlparse(str(row.get("web-scraper-start-url") or "")).netloc or "shop17")
        rec_at = parse_dt_aware(row.get("time-scraped"))

        # 生成输出记录
        output_records = []
        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_deltas.get(col_norm, 0)
            price = int(base_price + delta)

            # 判断 delta 来源
            delta_source = "matched_label" if col_norm in color_deltas else "default_zero"
            matched_label = None
            if delta_source == "matched_label":
                # 找到匹配的 label
                for label, d in labels_and_deltas:
                    if color_deltas.get(col_norm) == d:
                        for test_col_norm, (test_pn, test_col_raw) in color_map.items():
                            if test_col_norm == col_norm and _label_matches_color_shop17(label, test_col_raw, col_norm):
                                matched_label = label
                                break
                        if matched_label:
                            break

            # DEBUG: 每条输出记录单独一条日志（扁平化）
            _log_seq += 1
            logger.debug(
                f"Output record: {pn}",
                extra={
                    "event_type": "output_record",
                    "log_seq": _log_seq,
                    "shop_name": shop_name,
                    "cleaner_name": CLEANER_NAME,
                    "row_index": int(idx),
                    "model_text": model_text,
                    "model_norm": model_norm,
                    "capacity_gb": cap_gb,
                    "part_number": pn,
                    "color_norm": col_norm,
                    "color_raw": col_raw,
                    "base_price": base_price,
                    "delta": delta,
                    "final_price": price,
                    "delta_source": delta_source,
                    "matched_label": matched_label,
                    "recorded_at": str(rec_at) if rec_at else None,
                }
            )

            output_records.append({
                "part_number": pn,
                "color_norm": col_norm,
                "delta": delta,
                "final_price": price,
            })

            rows.append({
                "part_number": pn,
                "shop_name": shop_name,
                "price_new": price,
                "recorded_at": rec_at,
            })

        # INFO: 行级概览（汇总信息）
        colors_matched = len([cn for cn in color_map.keys() if cn in color_deltas])
        deltas = [d for d in color_deltas.values()]

        _log_seq += 1
        logger.info(
            f"Row {idx}: {model_text} → {len(labels_and_deltas)} labels, {colors_matched} colors matched, {len(output_records)} records output",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": SHOP_NAME,
                "cleaner_name": CLEANER_NAME,
                "row_index": int(idx),
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": cap_gb,
                "base_price": base_price,
                "color_discount_raw_preview": _truncate_for_log(raw_color_s, 100),
                "extraction_method": extraction_method,
                "labels_extracted_count": len(labels_and_deltas),
                "colors_in_catalog": len(color_map),
                "colors_matched_count": colors_matched,
                "output_records_count": len(output_records),
                "has_discounted_colors": any(d != 0 for d in deltas),
                "min_delta": min(deltas) if deltas else 0,
                "max_delta": max(deltas) if deltas else 0,
            }
        )

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
