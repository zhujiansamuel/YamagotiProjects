from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
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


DEBUG_SHOP17 = str(os.getenv("DEBUG_SHOP17", "")).strip().lower() in {"1", "true", "yes", "on"}
DEBUG_SHOP17_MAX_ROWS = int(os.getenv("DEBUG_SHOP17_MAX_ROWS", "20") or 20)
DEBUG_SHOP17_SHOW_ALL_COLORS = str(os.getenv("DEBUG_SHOP17_SHOW_ALL_COLORS", "")).strip().lower() in {"1", "true", "yes", "on"}

def _dbg_short(s: str, n: int = 400) -> str:
    if s is None:
        return ""
    t = str(s)
    if len(t) <= n:
        return t
    return t[:n] + f"... <len={len(t)}>"

def _dbg_print(ctx: str, msg: str, enabled: bool):
    if enabled:
        print(f"[shop17 DEBUG] {ctx} {msg}")

def _dbg_block(ctx: str, title: str, body: str, enabled: bool):
    if not enabled:
        return
    print("=" * 90)
    print(f"[shop17 DEBUG] {ctx} {title}")
    print("-" * 90)
    print(body)
    print("=" * 90)

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
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(_FW_DIGITS_TRANS_shop17)
    s = s.replace("，", ",").replace("／", "/")
    s = s.translate(_DASH_TRANS_shop17)
    s = re.sub(r"[ \t\u3000\xa0]+", " ", s)
    return s.strip()

def _is_plausible_color_label_shop17(label: str) -> bool:
    # 依赖你已有的 _normalize_label_shop17
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


def _load_iphone17_info_df_for_shop2() -> pd.DataFrame:
    """
    读取 iphone17_info，并尽量保留 jan 列以供 shop1 做 JAN→PN 映射。
    输出列至少包含：part_number, model_name, capacity_gb, color，
    若检测到任何 jan 列，则额外返回标准化列 'jan'。
    """
    try:
        from django.conf import settings
        p = getattr(settings, "EXTERNAL_IPHONE17_INFO_PATH", None)
        if p:
            path = str(p)
        else:
            raise AttributeError
    except Exception:
        path = os.getenv("IPHONE17_INFO_CSV") or str(
            Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv"
        )

    pth = Path(path)
    if not pth.exists():
        raise FileNotFoundError(f"未找到 iphone17_info：{pth}")

    if re.search(r"\.(xlsx|xlsm|xls|ods)$", str(pth), re.I):
        df = pd.read_excel(pth)
    else:
        df = pd.read_csv(pth, encoding="utf-8-sig")

    need = {"part_number", "model_name", "capacity_gb", "color"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"iphone17_info 缺少必要列：{missing}")

    df = df.copy()
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number", "color"])

    # ★ 检测并标准化 jan 列（尽最大可能适配命名）
    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):  # 兼容 'JANコード' 等
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))  # 去重保序

    cols = ["part_number", "model_name", "capacity_gb", "color"]
    if jan_candidates:
        src = jan_candidates[0]
        df["jan"] = df[src]
        cols.append("jan")

    return df[cols]

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
COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_NONE_RE_shop17 = re.compile(
    r"""(?P<label>[^：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])\s*
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

SPLIT_TOKENS_RE_shop17 = re.compile(r"[／/、]|(?:\s*;\s*)|\n")

def _normalize_label_shop17(lbl: str) -> str:
    return re.sub(r"[\s\u3000\xa0]+", "", lbl or "")

def _extract_color_deltas_shop17_regex(text: str, *, ctx: str = "", dbg: bool = False) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    if not text:
        _dbg_print(ctx, "[regex] empty text -> []", dbg)
        return out

    s0 = _pick_unopened_section(str(text))
    s = _normalize_color_text_shop17(s0)

    if "色減額" in s:
        s = s.split("色減額", 1)[-1].lstrip(":：")

    _dbg_block(ctx, "[regex] input", f"raw:\n{_dbg_short(s0)}\n\nnormalized:\n{_dbg_short(s)}", dbg)

    if re.fullmatch(r"\s*(?:なし|減額なし)\s*", s):
        _dbg_print(ctx, "[regex] whole segment is 'なし/減額なし' -> []", dbg)
        return out

    parts = [p.strip() for p in SPLIT_TOKENS_RE_shop17.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    _dbg_print(ctx, f"[regex] parts={parts}", dbg)

    for part in parts:
        m0 = COLOR_NONE_RE_shop17.search(part)
        if m0:
            label = _normalize_label_shop17(m0.group("label"))
            if _is_plausible_color_label_shop17(label):
                out.append((label, 0))
                _dbg_print(ctx, f"[regex] NONE hit: label={label!r} delta=0 from part={part!r}", dbg)
            else:
                _dbg_print(ctx, f"[regex] NONE ignored (implausible): label={label!r} part={part!r}", dbg)
            continue

        for m in COLOR_DELTA_RE_shop17.finditer(part):
            label = _normalize_label_shop17(m.group("label"))
            if not _is_plausible_color_label_shop17(label):
                _dbg_print(ctx, f"[regex] DELTA ignored (implausible): label={label!r} part={part!r}", dbg)
                continue

            sep = m.group("sep")
            sign = m.group("sign")
            amt = to_int_yen(m.group("amount"))
            if amt is None:
                _dbg_print(ctx, f"[regex] DELTA parse amount failed: amount={m.group('amount')!r} part={part!r}", dbg)
                continue

            negative = (sign == "-") if sign else (sep == "-" if sep else False)
            delta = -int(amt) if negative else int(amt)
            out.append((label, delta))
            _dbg_print(ctx, f"[regex] DELTA hit: label={label!r} delta={delta} from part={part!r}", dbg)

    _dbg_print(ctx, f"[regex] result={out}", dbg)
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
- 色名の例: スカイブルー, スペースブラック, クラウドホワイト, ライトゴールド など。
- 「利用制限△-10000」や「郵送は翌日着のみ保証」など、色と無関係な金額・文言は無視してください。
- 「なし」「減額なし」はその色の delta=0 として扱ってください。

出力ポリシー:
- extraction_class は必ず "color_delta" にしてください。
- extraction_text は「色名そのもの」（例: "スカイブルー"）だけにしてください。
- attributes には必ず次のキーを入れてください:
  - "delta": その色の価格差（整数。値引きは負の数。例: -3000）
  - "raw": 元の部分文字列（例: "スカイブルー-3,000"）
- 価格は円単位で扱い、「円」「,」などは無視して整数に変換してください。
- 色名が複数ある場合は、それぞれ1つずつ color_delta を出力してください。
- 色に関係しない金額（例: 利用制限△-10000）は一切出力しないでください。
追加ルール:
- 「色名なし」(例: シルバーなし) はその色の delta=0 として扱う。
- 「△減額なし」「保証開始3か月未満減額なし」など、色名が直前に無い「減額なし」は無視する。
- 入力には改行/空行が含まれる。改行は無視してよい。
""").strip()

@lru_cache()
def _get_color_delta_examples_shop17() -> List[ExampleData]:
    """
    few-shot 例。LangExtract が色名 + delta の抽出パターンを学習する。
    """
    if not _HAS_LANGEXTRACT:
        return []

    examples: List[ExampleData] = []

    # Example 1: あなたが提供したケースそのまま
    examples.append(
        ExampleData(
            text="色減額:スカイブルー-3,000\n\n郵送は翌日着のみ保証\n\n利用制限△-10000",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー",
                    attributes={
                        "delta": "-3000",
                        "raw": "スカイブルー-3,000",
                    },
                )
            ],
        )
    )
    examples.append(
        ExampleData(
            text="色減額:シルバーなし/ブルー-1000\n\n郵送は翌日着のみ保証\n\n△減額なし 保証開始3か月未満減額なし",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta": "0", "raw": "シルバーなし"},
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta": "-1000", "raw": "ブルー-1000"},
                ),
            ],
        )
    )

    # Example 2: 2 色 + 利用制限△
    examples.append(
        ExampleData(
            text="色減額:スカイブルー-4,000/スペースブラック-4,000\n\n持ち込みのみ保証\n\n利用制限△-10000",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー",
                    attributes={
                        "delta": "-4000",
                        "raw": "スカイブルー-4,000",
                    },
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スペースブラック",
                    attributes={
                        "delta": "-4000",
                        "raw": "スペースブラック-4,000",
                    },
                ),
            ],
        )
    )

    # Example 3: 「なし」パターン
    examples.append(
        ExampleData(
            text="色減額:スカイブルー-3,000/クラウドホワイト：なし",
            extractions=[
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="スカイブルー",
                    attributes={
                        "delta": "-3000",
                        "raw": "スカイブルー-3,000",
                    },
                ),
                Extraction(
                    extraction_class="color_delta",
                    extraction_text="クラウドホワイト",
                    attributes={
                        "delta": "0",
                        "raw": "クラウドホワイト：なし",
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
    # 统一成整数
    s = s.replace("円", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.replace("　", "")
    s = s.replace("−", "-").replace("－", "-")
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None

def _extract_color_deltas_shop17_llm(text: str, *, ctx: str = "", dbg: bool = False) -> List[Tuple[str, int]]:
    if not _HAS_LANGEXTRACT or not USE_LLM_FOR_SHOP17:
        _dbg_print(ctx, "[llm] disabled or no langextract -> []", dbg)
        return []

    raw0 = _normalize_color_text_shop17(_pick_unopened_section(str(text or "")))
    if not raw0:
        _dbg_print(ctx, "[llm] empty after normalize -> []", dbg)
        return []

    if re.fullmatch(r"(?:なし|減額なし)", re.sub(r"\s+", "", raw0)):
        _dbg_print(ctx, "[llm] whole segment is 'なし/減額なし' -> []", dbg)
        return []

    # 候选：优先色減額后的第一段/第一行，最后再用全量
    candidates: List[str] = []
    if "色減額" in raw0:
        after = raw0.split("色減額", 1)[-1].lstrip(":：").strip()
        first_block = re.split(r"\n\s*\n", after, maxsplit=1)[0].strip()
        if first_block:
            candidates.append(first_block)
            first_line = first_block.split("\n", 1)[0].strip()
            if first_line and first_line != first_block:
                candidates.append(first_line)
    candidates.append(raw0)

    # 去重保序
    seen = set()
    deduped: List[str] = []
    for c in candidates:
        c = _normalize_color_text_shop17(c)
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)

    _dbg_block(ctx, "[llm] candidates", "\n\n".join(f"- { _dbg_short(x) }" for x in deduped), dbg)

    last_err: Optional[Exception] = None

    for s in deduped:
        if re.fullmatch(r"(?:なし|減額なし)", re.sub(r"\s+", "", s)):
            continue

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
            )
        except Exception as e:
            last_err = e
            _dbg_print(ctx, f"[llm] call failed: {e}", dbg)
            continue

        extractions = getattr(result, "extractions", None) or []
        if dbg:
            _dbg_print(ctx, f"[llm] raw extractions count={len(extractions)}", dbg)
            for ext in extractions[:20]:
                _dbg_print(
                    ctx,
                    f"[llm] ext: class={getattr(ext,'extraction_class',None)!r}, text={getattr(ext,'extraction_text',None)!r}, attrs={getattr(ext,'attributes',None)!r}",
                    dbg
                )

        out: List[Tuple[str, int]] = []
        for ext in extractions:
            try:
                if getattr(ext, "extraction_class", None) != "color_delta":
                    continue

                label = (getattr(ext, "extraction_text", "") or "").strip()
                label = re.sub(r"(?:減額)?なし$", "", label).strip()
                label = re.sub(r"[:：\-/／]+$", "", label).strip()
                if not _is_plausible_color_label_shop17(label):
                    _dbg_print(ctx, f"[llm] ignore label (implausible): {label!r}", dbg)
                    continue

                attrs = getattr(ext, "attributes", None) or {}
                delta_int = _parse_delta_attr_to_int(attrs.get("delta"))

                # raw 兜底
                if delta_int is None:
                    raw_piece = attrs.get("raw")
                    if raw_piece:
                        rp = _normalize_color_text_shop17(raw_piece)
                        if "なし" in rp:
                            delta_int = 0
                        else:
                            m = re.search(r"([+-])\s*(\d[\d,]*)", rp)
                            if m:
                                amt = to_int_yen(m.group(2))
                                if amt is not None:
                                    delta_int = int(amt) * (-1 if m.group(1) == "-" else 1)

                if delta_int is None:
                    _dbg_print(ctx, f"[llm] ignore ext (no delta): label={label!r} attrs={attrs!r}", dbg)
                    continue

                out.append((_normalize_label_shop17(label), int(delta_int)))
            except Exception as e:
                _dbg_print(ctx, f"[llm] parse ext error: {e}", dbg)
                continue

        if out:
            merged: Dict[str, int] = {}
            for lbl, d in out:
                merged[lbl] = d
            final = list(merged.items())
            _dbg_print(ctx, f"[llm] result={final}", dbg)
            return final

        _dbg_print(ctx, "[llm] no usable output for this candidate", dbg)

    if last_err is not None:
        print(f"[shop17][LangExtract] extraction error: {last_err}")
    return []

def _extract_color_deltas_shop17(text: str, *, ctx: str = "", dbg: bool = False) -> List[Tuple[str, int]]:
    llm_res = _extract_color_deltas_shop17_llm(text, ctx=ctx, dbg=dbg)
    if llm_res:
        _dbg_print(ctx, f"[extract] path=LLM labels_and_deltas={llm_res}", dbg)
        return llm_res

    rx = _extract_color_deltas_shop17_regex(text, ctx=ctx, dbg=dbg)
    _dbg_print(ctx, f"[extract] path=REGEX labels_and_deltas={rx}", dbg)
    return rx
# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------
def clean_shop17(df: pd.DataFrame) -> pd.DataFrame:
    print("shop17:ゲストモバイル---------->进入清洗器时间：",
          time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    for c in ["type", "新未開封品", "色減額", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop17 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop17(info_df)
    rows: List[dict] = []

    dbg_printed = 0

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

        # 仅对前 N 行开启详细 debug（避免刷屏）
        dbg_row = DEBUG_SHOP17 and (dbg_printed < DEBUG_SHOP17_MAX_ROWS) and bool(raw_color_s.strip())
        if dbg_row:
            dbg_printed += 1

        ctx = f"row={idx} | type={model_text} | norm={model_norm} | cap={cap_gb} | base={base_price}"

        if dbg_row:
            _dbg_block(
                ctx, "INPUT 色減額",
                f"raw:\n{_dbg_short(raw_color_s)}\n\nnormalized:\n{_dbg_short(_normalize_color_text_shop17(raw_color_s))}",
                dbg_row
            )
            _dbg_print(ctx, f"color_map({len(color_map)}): " +
                       ", ".join([f"{cn}->{pn}" for cn, (pn, cr) in list(color_map.items())]),
                       dbg_row)

        labels_and_deltas = _extract_color_deltas_shop17(raw_color_s, ctx=ctx, dbg=dbg_row)

        # label -> color 命中 & 计算 delta
        color_deltas: Dict[str, int] = {}
        if labels_and_deltas:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in labels_and_deltas:
                    if _label_matches_color_shop17(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = delta

        if dbg_row:
            _dbg_print(ctx, f"labels_and_deltas={labels_and_deltas}", dbg_row)
            _dbg_print(ctx, f"matched color_deltas={color_deltas}", dbg_row)

            # 展示每个 label 命中了哪些颜色（便于发现误匹配/漏匹配）
            for label_raw, delta in labels_and_deltas or []:
                hits = []
                for col_norm, (pn, col_raw) in color_map.items():
                    if _label_matches_color_shop17(label_raw, col_raw, col_norm):
                        hits.append(f"{col_norm}(pn={pn})")
                _dbg_print(ctx, f"label={label_raw!r} delta={delta} hits={hits}", dbg_row)

        shop_name = SHOP_NAME_OVERRIDE or (urlparse(str(row.get("web-scraper-start-url") or "")).netloc or "shop17")
        rec_at = parse_dt_aware(row.get("time-scraped"))

        for col_norm, (pn, col_raw) in color_map.items():
            delta = color_deltas.get(col_norm, 0)
            price = int(base_price + delta)

            if dbg_row and (DEBUG_SHOP17_SHOW_ALL_COLORS or delta != 0):
                _dbg_print(ctx, f"OUTPUT color={col_norm} pn={pn} base={base_price} delta={delta} => price={price}", dbg_row)

            rows.append({
                "part_number": pn,
                "shop_name": shop_name,
                "price_new": price,
                "recorded_at": rec_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
