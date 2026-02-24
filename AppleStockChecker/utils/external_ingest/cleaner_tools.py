# AppleStockChecker/utils/external_ingest/cleaner_tools.py
"""
清洗器通用工具模块
提供数据库访问、数据转换等通用功能
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Pattern
import logging
import os
import re
import time
import unicodedata

import pandas as pd

from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date


def to_int_yen(s: object) -> Optional[int]:
    if s is None: return None
    txt = str(s).strip()
    if not re.search(r"\d", txt): return None
    # 范围 "105,000～110,000"
    parts = re.split(r"[~～\-–—]", txt)
    candidates = []
    for p in parts:
        # 排除 12-14 位纯数字（像 JAN/电话）
        if re.fullmatch(r"\d{12,14}", p.strip()):
            continue
        digits = re.sub(r"[^\d万]", "", p)
        if not digits:
            continue
        if "万" in digits:
            m = re.search(r"([\d\.]+)万", digits)
            base = float(m.group(1)) if m else 0.0
            candidates.append(int(base * 10000))
        else:
            candidates.append(int(re.sub(r"[^\d]", "", digits)))
    if not candidates:
        return None
    val = max(candidates)
    # 合理区间过滤
    if val < 1000 or val > 5_000_000:
        return None
    return val


def parse_dt_aware(s: object) -> timezone.datetime:
    if not s:
        return timezone.now()
    txt = str(s).strip()
    dt = parse_datetime(txt)
    if dt is None:
        d = parse_date(txt)
        if d:
            dt = timezone.datetime(d.year, d.month, d.day)
    if dt is None:
        return timezone.now()
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


# ----------------------------------------------------------------------
# LLM/Ollama 与抽取模式配置（各 shop 通用）
# ----------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID") or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "regex")  # "regex" | "llm" | "auto"


# ----------------------------------------------------------------------
# langextract 统一导入（各 shop 共用）
# ----------------------------------------------------------------------
try:
    import langextract as lx
    HAS_LANGEXTRACT = True
except Exception:
    lx = None  # type: ignore[assignment]
    HAS_LANGEXTRACT = False


# ----------------------------------------------------------------------
# LLM 提取失败日志（各 shop 共用）
# ----------------------------------------------------------------------

def log_llm_extraction_error(
    logger: logging.Logger,
    *,
    cleaner_name: str,
    shop_name: str,
    error: Exception,
    text: str = "",
    row_index: object = None,
    text_preview_len: int = 100,
    **extra_fields: object,
) -> None:
    """
    LLM（LangExtract）提取失败时的统一 warning 日志。

    替代各 shop 清洗器中重复出现的:
        logger.warning("LangExtract extraction failed", extra={...})
    """
    extra: dict = {
        "event_type": "llm_extraction_error",
        "shop_name": shop_name,
        "cleaner_name": cleaner_name,
        "error": str(error),
        "error_type": type(error).__name__,
        "model_id": OLLAMA_MODEL_ID,
        "model_url": OLLAMA_URL,
        "text_length": len(text) if text else 0,
        "text_preview": _truncate_for_log(text, text_preview_len) if text else "",
    }
    if row_index is not None:
        extra["row_index"] = row_index
    extra.update(extra_fields)
    logger.warning("LangExtract extraction failed", extra=extra)


# ----------------------------------------------------------------------
# 绝对值价格提取（公共函数）
# ----------------------------------------------------------------------

def extract_price_yen(raw: object) -> Optional[int]:
    """
    从价格字段提取整数日元（绝对值价格）。

    统一处理流程：
      1. safe_to_text() 安全转字符串（兼容 NaN/None/数字等）
      2. normalize_text_basic() 全角→半角 + 去换行 + 合并空格
      3. 去除前导 '～' 及修饰词（新品/未開封 等）
      4. to_int_yen() 解析日元整数（支持区间取最大、万、逗号分隔、合理区间过滤）

    替代原各 shop 中的重复 wrapper：
      _price_from_shop3 / _price_from_shop5 / _price_from_shop6_data7 /
      _price_from_shop7 / _price_from_shop10 / _price_from_shop13 /
      _extract_price_new

    参数:
        raw: 价格字段的原始值（str / int / float / None / NaN 等）

    返回:
        Optional[int]: 解析出的日元整数，无法解析时返回 None
    """
    s = safe_to_text(raw)
    if not s:
        return None
    s = normalize_text_basic(s)
    s = s.lstrip("～")
    s = (s.replace("新品", "")
          .replace("新\u54c1", "")
          .replace("未開封", "")
          .replace("未开封", ""))
    return to_int_yen(s)


def _load_iphone17_info_df_from_db(
    *,
    add_model_norm: bool = False,
) -> pd.DataFrame:
    """
    从数据库中读取 iPhone 机型信息，返回 DataFrame

    输出列：part_number, model_name, capacity_gb, color, jan（如果 jan 字段有值）
    当 add_model_norm=True 时额外输出 model_name_norm（归一化型号，供 shop10 等使用）

    Parameters:
        add_model_norm: 是否添加 model_name_norm 列（默认 False，兼容既有调用方）

    Returns:
        pd.DataFrame: 包含 iPhone 机型信息的 DataFrame

    Raises:
        ValueError: 如果数据库中没有 iPhone 数据
    """
    from AppleStockChecker.models import Iphone

    # 查询所有 iPhone 数据，只选择需要的字段
    queryset = Iphone.objects.all().values(
        'part_number',
        'model_name',
        'capacity_gb',
        'color',
        'jan'
    )

    # 转换为 DataFrame
    df = pd.DataFrame.from_records(queryset)

    if df.empty:
        raise ValueError("数据库中没有 iPhone 数据，请先导入 iPhone 机型信息")

    # 确保数据类型正确
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")

    # 删除必要字段为空的行
    df = df.dropna(subset=["model_name", "capacity_gb", "part_number", "color"])

    # 处理 jan 列：如果所有 jan 都是空的，就删除这一列；否则保留
    if df["jan"].isna().all():
        df = df.drop(columns=["jan"])
        cols = ["part_number", "model_name", "capacity_gb", "color"]
    else:
        cols = ["part_number", "model_name", "capacity_gb", "color", "jan"]

    if add_model_norm:
        df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
        cols = ["part_number", "model_name", "model_name_norm", "capacity_gb", "color"]
        if "jan" in df.columns:
            cols.append("jan")

    return df[cols].reset_index(drop=True)


# 正则表达式模式用于型号匹配
_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)


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


def _normalize_model_generic(text: str) -> str:
    """
    统一型号主体：
      - iPhone17/16 + 后缀（Pro/Pro Max/Plus/mini）
      - iPhone Air（含"17 air"→ Air）
      - 允许紧凑写法：17pro / 17promax / 16Pro / 16Plus ...
    输出：'iPhone 17 Pro Max' / 'iPhone 17 Pro' / 'iPhone Air' / ...
    """
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    # 罕见写法归一：'i phone' / 'I Phone' → 'iPhone'
    t = re.sub(r"(?i)\bi\s+phone\b", "iPhone", t)

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


def _build_color_map(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """
    构建 (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) } 查找字典。

    各 shop 清洗器共用，用于按 (机型, 容量) 快速查找所有颜色变体及其 part_number。
    color_norm 使用 _norm_strip 归一化（去空白 + 转小写），与 _label_matches_color_unified 匹配逻辑一致。
    """
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    cmap: Dict[Tuple[str, int], Dict[str, Tuple[str, str]]] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        if not m or pd.isna(cap):
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        color_norm = _norm_strip(str(r["color"]))
        cmap[key][color_norm] = (str(r["part_number"]), str(r["color"]))
    return cmap


# ----------------------------------------------------------------------
# JAN 映射相关
# ----------------------------------------------------------------------
_JAN_RE = re.compile(r"(\d{8,})")


def _extract_jan_digits(v) -> Optional[str]:
    """从 JAN 字段值中提取连续 8+ 位数字。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    m = _JAN_RE.search(str(v))
    return m.group(1) if m else None


# ----------------------------------------------------------------------
# 共通辅助函数（各 shop 清洗器共用）
# ----------------------------------------------------------------------

def _truncate_for_log(s: str, n: int = 200) -> str:
    """截断长字符串，保留前 n 个字符，用于日志显示"""
    if s is None:
        return ""
    t = str(s)
    if len(t) <= n:
        return t
    return t[:n] + f"... (truncated, total_length={len(t)})"


def _norm_strip(s: str) -> str:
    """颜色匹配用归一化：去空格 + 转小写（用于 shop3/4/7/9/11/12/14/15/16/17）"""
    t = (s or "").strip()
    t = re.sub(r"[\s\u3000]+", "", t)  # 去除所有空白（含全角空格）
    return t.lower()


# ----------------------------------------------------------------------
# 颜色同义词表（合并各 shop，含带空格变体）
# ----------------------------------------------------------------------

FAMILY_SYNONYMS_COLOR: Dict[str, List[str]] = {
    # blue 家族（含 ディープ ブルー / ディープブルー / スカイブルー）
    "blue": ["ブルー", "青", "ディープブルー", "ディープ ブルー", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    "ブルー": ["ブルー", "青", "ディープブルー", "ディープ ブルー", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    "青": ["ブルー", "青", "ディープブルー", "ディープ ブルー", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    "ディープブルー": ["ブルー", "青", "ディープブルー", "ディープ ブルー", "スカイブルー"],
    "ディープ ブルー": ["ブルー", "青", "ディープブルー", "ディープ ブルー", "スカイブルー"],
    "スカイブルー": ["ブルー", "青", "ディープブルー", "ディープ ブルー", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    "ミッドナイト": ["ブルー", "青", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    "マリン": ["ブルー", "青", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    "ミストブルー": ["ブルー", "青", "スカイブルー", "ミッドナイト", "マリン", "ミストブルー"],
    # black（含 スペース ブラック / スペースブラック）
    "black": ["ブラック", "黒", "スペースブラック", "スペース ブラック"],
    "ブラック": ["ブラック", "黒", "スペースブラック", "スペース ブラック"],
    "黒": ["ブラック", "黒", "スペースブラック", "スペース ブラック"],
    "space black": ["スペースブラック", "スペース ブラック"],
    "spaceblack": ["スペースブラック", "スペース ブラック"],
    "スペースブラック": ["ブラック", "黒", "スペースブラック", "スペース ブラック"],
    "スペース ブラック": ["ブラック", "黒", "スペースブラック", "スペース ブラック"],
    # white / starlight
    "white": ["ホワイト", "白", "スターライト", "starlight"],
    "ホワイト": ["ホワイト", "白", "スターライト"],
    "白": ["ホワイト", "白", "スターライト"],
    "スターライト": ["ホワイト", "白", "スターライト"],
    "starlight": ["ホワイト", "白", "スターライト"],
    # silver
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],
    # gold
    "gold": ["ゴールド", "金", "ライトゴールド"],
    "ゴールド": ["ゴールド", "金", "ライトゴールド"],
    "金": ["ゴールド", "金", "ライトゴールド"],
    "ライトゴールド": ["ゴールド", "金", "ライトゴールド"],
    # orange
    "orange": ["オレンジ", "橙", "コズミックオレンジ"],
    "オレンジ": ["オレンジ", "橙", "コズミックオレンジ"],
    "橙": ["オレンジ", "橙", "コズミックオレンジ"],
    "コズミックオレンジ": ["オレンジ", "橙", "コズミックオレンジ"],
    # green
    "green": ["グリーン", "緑", "セージ"],
    "グリーン": ["グリーン", "緑", "セージ"],
    "緑": ["グリーン", "緑", "セージ"],
    "セージ": ["グリーン", "緑", "セージ"],
    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],
    # red
    "red": ["レッド", "赤"],
    "レッド": ["レッド", "赤"],
    "赤": ["レッド", "赤"],
    # yellow
    "yellow": ["イエロー", "黄", "黄色"],
    "イエロー": ["イエロー", "黄", "黄色"],
    "黄": ["イエロー", "黄", "黄色"],
    "黄色": ["イエロー", "黄", "黄色"],
    # purple
    "purple": ["パープル", "紫", "ラベンダー"],
    "パープル": ["パープル", "紫", "ラベンダー"],
    "紫": ["パープル", "紫", "ラベンダー"],
    "ラベンダー": ["パープル", "紫", "ラベンダー"],
    # natural
    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],
    # gray
    "gray": ["グレー", "グレイ", "灰"],
    "グレー": ["グレー", "グレイ", "灰"],
    "グレイ": ["グレー", "グレイ", "灰"],
    "灰": ["グレー", "グレイ", "灰"],
    # titanium
    "チタン": ["チタン", "チタニウム"],
    "チタニウム": ["チタン", "チタニウム"],
}


def build_synonym_lookup_norm(
    family_synonyms: Dict[str, List[str]],
    norm_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, List[str]]:
    """
    从 FAMILY_SYNONYMS 构建归一化版本的同义词 lookup。

    key 与 value 均经 norm_fn 变换（去空格、转小写），用于颜色匹配时以 _norm_strip 后的字符串查找。

    参数:
        family_synonyms: 原始同义词表 {key: [syn1, syn2, ...]}
        norm_fn: 归一化函数，默认 _norm_strip

    返回:
        {norm_key: [norm_syn1, norm_syn2, ...]} 每个 key/value 归一化后均作为 key
    """
    norm = norm_fn or _norm_strip
    out: Dict[str, List[str]] = {}

    for k, vs in family_synonyms.items():
        all_raw = list(dict.fromkeys([k] + vs))
        all_norm = list(dict.fromkeys([norm(x) for x in all_raw]))
        for x in all_raw:
            nx = norm(x)
            out.setdefault(nx, [])
            out[nx] = list(dict.fromkeys(out[nx] + all_norm))
    return out


# 预构建的去除空格版本同义词 lookup（供各 shop 颜色匹配使用）
SYNONYM_LOOKUP_NORM: Dict[str, List[str]] = build_synonym_lookup_norm(FAMILY_SYNONYMS_COLOR)


# ----------------------------------------------------------------------
# 拆分逻辑正则（按店铺号命名，便于对比）
# ----------------------------------------------------------------------
# 说明：各店铺用不同分隔符拆分复合颜色标签（如 "シルバー/ディープブルー" → ['シルバー','ディープブルー']）
# 分隔符差异：shop3/14 含 ;；；shop9/12 含 ;；；shop15 含 &＆；shop16/17 不含 ・
LABEL_SPLIT_RE_shop2 = re.compile(r"[／/、，,・\s]+")
LABEL_SPLIT_RE_shop3 = re.compile(r"[／/、，,・\s；;]+")
LABEL_SPLIT_RE_shop4 = re.compile(r"[／/、，,・\s]+")
LABEL_SPLIT_RE_shop7 = re.compile(r"[／/、，,・\s]+")
LABEL_SPLIT_RE_shop9 = re.compile(r"[/／、，,;；\s]+")
LABEL_SPLIT_RE_shop11 = re.compile(r"[／/、，,・\s]+")
LABEL_SPLIT_RE_shop12 = re.compile(r"[／/、，,・\s]+")
LABEL_SPLIT_RE_shop14 = re.compile(r"[／/、，,;；\s]+")
# 不含半角逗号，避免 "229,000円" 中的千位分隔符被误当作标签分隔
LABEL_SPLIT_RE_shop15 = re.compile(r"\s*(?:、|，|／|/|・|&|＆)\s*")
LABEL_SPLIT_RE_shop16 = re.compile(r"[／/、，,]|(?:\s*;\s*)")
LABEL_SPLIT_RE_shop16_SIMPLE = re.compile(r"[／/、，,]")
LABEL_SPLIT_RE_shop17 = re.compile(r"[／/、]|(?:\s*;\s*)|\n")

# ----------------------------------------------------------------------
# 两阶段流水线：MatchToken / format_hint / 语义映射
# ----------------------------------------------------------------------
# 阶段 1 输出 MatchToken（各 shop 实现）；阶段 2 match_tokens_to_specs 统一映射。
# 冲突去重优先级：after_yen > colon_prefix > plain_digits > signed > sep_minus > none

FORMAT_HINT_SIGNED = "signed"
FORMAT_HINT_SEP_MINUS = "sep_minus"
FORMAT_HINT_AFTER_YEN = "after_yen"
FORMAT_HINT_PLAIN_DIGITS = "plain_digits"
FORMAT_HINT_COLON_PREFIX = "colon_prefix"
FORMAT_HINT_NONE = "none"

# 优先级从高到低（用于冲突去重）
FORMAT_HINT_PRIORITY: Dict[str, int] = {
    FORMAT_HINT_AFTER_YEN: 5,
    FORMAT_HINT_COLON_PREFIX: 4,
    FORMAT_HINT_PLAIN_DIGITS: 3,
    FORMAT_HINT_SIGNED: 2,
    FORMAT_HINT_SEP_MINUS: 1,
    FORMAT_HINT_NONE: 0,
}

# 边界规则阈值
BOUNDARY_DELTA_MAX = 20000   # plain_digits 且 amount < 20000 → 视为 delta
BOUNDARY_ABS_MIN = 100000    # signed 且 amount > 100000 → 视为 abs


@dataclass
class MatchToken:
    """阶段 1 匹配输出，描述金额在原文中的呈现形式。"""
    label: str
    amount_int: int
    format_hint: str
    position: int = 0


def expand_match_tokens(
    tokens: List[MatchToken],
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: Callable[[str, str, str], bool],
    *,
    enable_adaptive: bool = True,
    logger: Optional[logging.Logger] = None,
    cleaner_name: str = "",
    shop_name: str = "",
) -> List[MatchToken]:
    """
    阶段 1 与阶段 2 之间的自适应分割：将复合标签展开为单标签。
    """
    out: List[MatchToken] = []
    for tok in tokens:
        if not enable_adaptive or not color_map:
            out.append(tok)
            continue
        adaptive = split_composite_label_adaptive(
            tok.label, color_map, label_matcher
        )
        labels = adaptive.get("labels") or []
        if not labels:
            labels = [tok.label]
        for i, lbl in enumerate(labels):
            out.append(MatchToken(
                label=lbl,
                amount_int=tok.amount_int,
                format_hint=tok.format_hint,
                position=tok.position * 1000 + i,
            ))
    return out


def match_tokens_to_specs(
    tokens: List[MatchToken],
    *,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    cleaner_name: str = "",
    shop_name: str = "",
    row_index: int = -1,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    阶段 2 语义映射：format_hint + 边界规则 → (deltas, abs_specs)。

    边界规则（带日志）:
      - plain_digits 且 amount < 20000 → 视为 delta
      - signed 且 amount > 100000 → 视为 abs
    """
    context = context or {}
    deltas: List[Tuple[str, int]] = []
    abs_specs: List[Tuple[str, int]] = []

    # 按 label 去重，保留 format_hint 优先级最高的
    by_label: Dict[str, MatchToken] = {}
    for tok in tokens:
        if not tok.label:
            continue
        existing = by_label.get(tok.label)
        pri = FORMAT_HINT_PRIORITY.get(tok.format_hint, -1)
        if existing is None or pri > FORMAT_HINT_PRIORITY.get(existing.format_hint, -1):
            by_label[tok.label] = tok

    for label, tok in by_label.items():
        kind: Optional[str] = None
        override_reason: Optional[str] = None

        # 默认映射
        if tok.format_hint == FORMAT_HINT_NONE:
            kind = "delta"
            value = 0
        elif tok.format_hint in (FORMAT_HINT_SIGNED, FORMAT_HINT_SEP_MINUS):
            kind = "delta"
            value = tok.amount_int
        elif tok.format_hint in (FORMAT_HINT_AFTER_YEN, FORMAT_HINT_COLON_PREFIX, FORMAT_HINT_PLAIN_DIGITS):
            kind = "abs"
            value = abs(tok.amount_int)

        # 边界规则覆盖
        if tok.format_hint == FORMAT_HINT_PLAIN_DIGITS and 0 < tok.amount_int < BOUNDARY_DELTA_MAX:
            kind = "delta"
            value = tok.amount_int  # 正数作为 delta（店铺可能写的是差额）
            override_reason = f"plain_digits amount {tok.amount_int} < {BOUNDARY_DELTA_MAX} → delta"
        elif tok.format_hint == FORMAT_HINT_SIGNED and abs(tok.amount_int) > BOUNDARY_ABS_MIN:
            kind = "abs"
            value = abs(tok.amount_int)
            override_reason = f"signed amount {abs(tok.amount_int)} > {BOUNDARY_ABS_MIN} → abs"

        if kind == "delta":
            deltas.append((label, value))
        elif kind == "abs":
            abs_specs.append((label, value))

        if override_reason and logger:
            logger.info(
                "format_hint boundary override",
                extra={
                    "event_type": "format_hint_boundary_override",
                    "label": label,
                    "format_hint": tok.format_hint,
                    "amount_int": tok.amount_int,
                    "override_reason": override_reason,
                    "cleaner_name": cleaner_name,
                    "shop_name": shop_name,
                    "row_index": row_index,
                },
            )

    return deltas, abs_specs


# ----------------------------------------------------------------------
# 自适应复合标签分割（shop17 试点）
# ----------------------------------------------------------------------
# 渐进式分割策略：从严格到宽松逐个尝试，使用"全匹配优先停止"策略

LABEL_SPLIT_STRATEGIES = [
    {
        "name": "standard",  # 标准分割（shop2/4/7/11/12 通用）
        "regex": re.compile(r"[／/、，,・\s]+"),
        "description": "斜杠、顿号、逗号、中点、空格"
    },
    {
        "name": "with_semicolon",  # 包含分号（shop3/9/14）
        "regex": re.compile(r"[／/、，,・\s;；]+"),
        "description": "标准 + 分号"
    },
    {
        "name": "with_ampersand",  # 包含 & 符号（shop15）
        "regex": re.compile(r"[／/、，,・\s&＆]+"),
        "description": "标准 + & 符号"
    },
    {
        "name": "with_pipe",  # 包含竖线
        "regex": re.compile(r"[／/、，,・\s|｜]+"),
        "description": "标准 + 竖线"
    },
    {
        "name": "aggressive",  # 激进模式（所有常见分隔符）
        "regex": re.compile(r"[／/、，,・\s;；&＆|｜]+"),
        "description": "所有常见分隔符"
    },
]


def validate_split_labels(
    labels: List[str],
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: Callable[[str, str, str], bool],
) -> Tuple[List[str], List[str], Set[str]]:
    """
    验证分割后的标签是否为有效颜色。

    参数:
        labels: 待验证的标签列表
        color_map: {color_norm: (part_number, color_raw)} 颜色映射表
        label_matcher: 颜色匹配函数 (label_raw, color_raw, color_norm) -> bool

    返回:
        (valid_labels, invalid_labels, matched_colors_set)
    """
    valid = []
    invalid = []
    matched_colors = set()

    for label in labels:
        label_cleaned = label.strip()
        if not label_cleaned:
            continue

        # 尝试匹配任意数据库中的颜色
        matched = False
        for color_norm, (pn, color_raw) in color_map.items():
            if label_matcher(label_cleaned, color_raw, color_norm):
                valid.append(label_cleaned)
                matched_colors.add(color_norm)
                matched = True
                break

        if not matched:
            invalid.append(label_cleaned)

    return valid, invalid, matched_colors


def detect_missing_colors_with_price(
    extracted_labels: List[str],
    original_text: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: Callable[[str, str, str], bool],
) -> List[Dict]:
    """
    检测原文中可能存在但未被提取的颜色。
    如果颜色后面跟着价格/减价信息，会标记为"应提取"。

    返回:
        [
            {
                "color_norm": ...,
                "color_raw": ...,
                "found_synonym": ...,
                "has_price_info": bool,
                "price_pattern": str,
                "should_extract": bool,
            },
            ...
        ]
    """
    # 价格模式：查找颜色后面的 +/-数字
    price_pattern = re.compile(r"([+\-＋－])[\s]*(\d+)")

    missing = []
    text_lower = original_text.lower()

    for color_norm, (pn, color_raw) in color_map.items():
        # 检查该颜色是否已被提取
        already_extracted = any(
            label_matcher(label, color_raw, color_norm)
            for label in extracted_labels
        )

        if already_extracted:
            continue

        # 检查原文中是否包含该颜色的任何同义词
        synonyms = SYNONYM_LOOKUP_NORM.get(color_norm, [])
        found_synonym = None
        found_position = -1

        # 1. 检查 color_raw 原文
        if color_raw in original_text:
            found_synonym = color_raw
            found_position = original_text.index(color_raw)
        elif color_raw.lower() in text_lower:
            found_synonym = color_raw
            found_position = text_lower.index(color_raw.lower())
        # 2. 检查同义词
        elif synonyms:
            for syn in synonyms:
                if syn in original_text:
                    found_synonym = syn
                    found_position = original_text.index(syn)
                    break
                elif syn.lower() in text_lower:
                    found_synonym = syn
                    found_position = text_lower.index(syn.lower())
                    break

        if found_synonym and found_position >= 0:
            # 检查颜色后面是否有价格信息
            text_after = original_text[found_position:found_position + 50]  # 取后面50字符
            price_match = price_pattern.search(text_after)

            has_price = price_match is not None
            price_str = price_match.group(0) if has_price else None

            missing.append({
                "color_norm": color_norm,
                "color_raw": color_raw,
                "part_number": pn,
                "found_synonym": found_synonym,
                "has_price_info": has_price,
                "price_pattern": price_str,
                "should_extract": has_price,  # 有价格信息则应该提取
                "context": text_after[:30],   # 上下文（用于日志）
            })

    return missing


def split_composite_label_adaptive(
    label_text: str,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: Callable[[str, str, str], bool],
) -> Dict:
    """
    自适应分割复合标签，使用"全匹配优先停止"策略。

    停止条件：
    1. 如果某个策略匹配到了该机型的所有颜色，立即停止并返回
    2. 否则尝试完所有策略后，返回匹配颜色最多的结果

    返回:
        {
            "strategy_used": str,           # 使用的策略名称
            "labels": List[str],            # 有效标签列表
            "matched_color_count": int,     # 匹配到的颜色数量
            "total_colors_in_catalog": int, # 该机型总颜色数
            "is_full_match": bool,          # 是否全匹配
            "invalid_parts": List[str],     # 无效部分
            "missing_colors": List[Dict],   # 潜在未提取的颜色
        }
    """
    total_colors = len(color_map)
    best_result = {
        "strategy_used": "none",
        "labels": [],
        "matched_color_count": 0,
        "total_colors_in_catalog": total_colors,
        "is_full_match": False,
        "invalid_parts": [],
        "missing_colors": [],
    }

    # 尝试各种分割策略
    for strategy in LABEL_SPLIT_STRATEGIES:
        # 1. 分割
        parts = [
            p.strip()
            for p in strategy["regex"].split(label_text)
            if p.strip()
        ]

        if not parts:
            continue

        # 2. 验证并收集匹配的颜色
        valid_labels, invalid_parts, matched_colors = validate_split_labels(
            parts, color_map, label_matcher
        )

        matched_count = len(matched_colors)

        # 3. 如果匹配到所有颜色，立即返回（提前终止）
        if matched_count == total_colors:
            return {
                "strategy_used": strategy["name"],
                "labels": valid_labels,
                "matched_color_count": matched_count,
                "total_colors_in_catalog": total_colors,
                "is_full_match": True,
                "invalid_parts": invalid_parts,
                "missing_colors": [],  # 全匹配时无遗漏颜色
            }

        # 4. 如果这个策略更好（匹配更多颜色），更新最佳结果
        if matched_count > best_result["matched_color_count"]:
            best_result = {
                "strategy_used": strategy["name"],
                "labels": valid_labels,
                "matched_color_count": matched_count,
                "total_colors_in_catalog": total_colors,
                "is_full_match": False,
                "invalid_parts": invalid_parts,
                "missing_colors": [],
            }

    # 5. 检测潜在遗漏的颜色（仅当非全匹配时）
    best_result["missing_colors"] = detect_missing_colors_with_price(
        best_result["labels"],
        label_text,
        color_map,
        label_matcher,
    )

    return best_result

# ----------------------------------------------------------------------
# 统一标签→颜色匹配函数（2025-02 替换各 shop 独立实现）
# ----------------------------------------------------------------------
# 合并 shop3/4/9/11/12/14/15/16/17 的 _label_matches_color 逻辑：
#   - shop3/4: 精确 + 原文子串 + color_raw_norm in candidates
#   - shop9: 去空白/连字符后双向包含兜底
#   - shop11: 分割多 token 匹配、分片同义词收集
#   - shop12: 归一化双向包含兜底
#   - shop14: 小写子串匹配 (label_norm in color_raw_l)
#   - shop15/16/17: any(tok in color_raw for tok in candidates)


def _label_matches_color_unified(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    统一标签→颜色匹配（供各 shop 共用）。

    匹配策略（按顺序）:
      0. 后缀清理：去除 "系/色"（如 "青系"→"青", "黒色"→"黒"）
      1. 精确归一化相等
      2. label_raw 为 color_raw 原文子串
      3. label_norm 为 color_raw 子串（小写，shop14）
      4. SYNONYM 同义词：color_norm / color_raw_norm / candidates 子串
      5. 归一化双向包含（shop12）
      6. 去空白/连字符后双向包含（shop9）

    注意：
      - 全色/ALL 由 resolve_color_prices 的 is_all 处理，此处不特殊返回
      - 标签分割应在各清洗器内部完成，传入此函数的 label_raw 应为单个标签
    """
    if not label_raw:
        return False

    # 0. 后缀清理：去除 "系/色"（shop2 数据常见 "青系" "黒色" 等写法）
    label_cleaned = re.sub(r"[系色]$", "", str(label_raw).strip()).strip()
    label_norm = _norm_strip(label_cleaned)
    color_raw_s = str(color_raw or "")
    color_raw_l = color_raw_s.lower()
    color_raw_norm = _norm_strip(color_raw_s)

    # 1. 精确
    if label_norm == color_norm:
        return True

    # 2. 原文子串 (label in color)
    if label_cleaned and label_cleaned in color_raw_s:
        return True

    # 3. shop14: label_norm in color (小写)
    if label_norm and label_norm in color_raw_l:
        return True

    # 4. SYNONYM
    candidates: set = set()
    if label_norm in SYNONYM_LOOKUP_NORM:
        candidates.update(SYNONYM_LOOKUP_NORM[label_norm])
    if candidates:
        if color_norm in candidates:
            return True
        if color_raw_norm in candidates:
            return True
        if any(tok in color_raw_l for tok in candidates):
            return True

    # 5. shop12: 归一化双向包含
    if label_norm and (label_norm in color_norm or color_norm in label_norm):
        return True

    # 6. shop9: 去空白/连字符后双向包含
    lr_short = re.sub(r"[\s\u3000\-]+", "", label_norm)
    cn_short = re.sub(r"[\s\u3000\-]+", "", color_norm)
    cr_short = re.sub(r"[\s\u3000\-]+", "", color_raw_norm)
    if lr_short and cn_short and (lr_short in cn_short or cn_short in lr_short):
        return True
    if lr_short and cr_short and (lr_short in cr_short or cr_short in lr_short):
        return True

    return False


# 阶段0：公用转换（各 shop 在取到原始文本后立即调用）
# 减号类 emoji → 标准减号；其他 emoji Unicode 块
_MINUS_EMOJI_PAT = re.compile(r"[\u2796]")  # HEAVY MINUS SIGN
_OTHER_EMOJI_PAT = re.compile(
    r"[\u2600-\u26FF\u2700-\u27BF"  # Misc Symbols, Dingbats
    r"\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF"  # Pictographs, Supplemental
    r"\u2300-\u23FF\u2B50\uFE00-\uFE0F\u200D\u200C]"
)


def normalize_text_stage0(text: str) -> str:
    """
    阶段0 公用转换：Unicode 归一化、HTML 清理、emoji 清理、去控制字符。
    各 shop 在取到原始文本后立即调用，再传入全色检测与阶段1。
    """
    if text is None or not isinstance(text, str):
        return "" if text is None else str(text)
    s = str(text)
    # 1. 去控制字符（保留 \\t \\n \\r）
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u0080-\u009f]", "", s)
    # 2. Unicode NFKC 归一化
    s = unicodedata.normalize("NFKC", s)
    # 3. HTML 标签清理
    s = re.sub(r"<[^>]+>", "", s)
    # 4. Emoji：减号类 → 标准减号
    s = _MINUS_EMOJI_PAT.sub("-", s)
    # 5. 其他 emoji → 删除
    s = _OTHER_EMOJI_PAT.sub("", s)
    return s


# 全角→半角 完整变换表（数字、标点、货币、日文符号）
_FZ_TO_HZ_TRANS = str.maketrans({
    # 数字
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    # 标点
    '，': ',', '．': '.', '：': ':', '；': ';',
    '（': '(', '）': ')', '「': '[', '」': ']',
    '『': '{', '』': '}', '【': '[', '】': ']',
    # 空格和连字符
    '　': ' ',   # 全角空格→半角空格
    '－': '-', '＋': '+', '／': '/', '＊': '*',
    # 货币符号（转为空）
    '¥': '', '￥': '',
    # Unicode 变体
    '−': '-',  # U+2212 MINUS SIGN
})


def normalize_text_basic(
    text: str,
    *,
    fullwidth_to_halfwidth: bool = True,
    remove_newlines: bool = True,
    collapse_spaces: bool = True,
    strip: bool = True
) -> str:
    """
    通用文本规范化（初步清洗）

    参数:
        text: 输入文本
        fullwidth_to_halfwidth: 全角→半角转换（数字、标点）
        remove_newlines: 去除换行符 (\\r\\n → 空格)
        collapse_spaces: 合并多个空格为一个
        strip: 去除首尾空白

    返回:
        规范化后的文本

    示例:
        >>> normalize_text_basic("iPhone　17　Pro\\n256GB")
        'iPhone 17 Pro 256GB'

        >>> normalize_text_basic("１２３，４５６円")
        '123,456円'
    """
    if text is None:
        return ""

    s = str(text)

    # 1. 全角→半角
    if fullwidth_to_halfwidth:
        s = s.translate(_FZ_TO_HZ_TRANS)

    # 2. 去除换行（转为空格，保持单词间隔）
    if remove_newlines:
        s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    # 3. 合并多个空格
    if collapse_spaces:
        s = re.sub(r"\s+", " ", s)

    # 4. Strip
    if strip:
        s = s.strip()

    return s


def safe_to_text(value) -> str:
    """
    安全地将任意值转为字符串，处理 NaN/None/空值

    参数:
        value: 任意类型的值（包括 pandas NA/NaT）

    返回:
        str: 转换后的字符串，异常值返回空字符串

    示例:
        >>> safe_to_text(None)
        ''
        >>> safe_to_text(pd.NA)
        ''
        >>> safe_to_text(123)
        '123'
        >>> safe_to_text("hello")
        'hello'
    """
    if value is None:
        return ""

    # pandas NA/NaT 处理
    if pd.isna(value):
        return ""

    # bool 类型特殊处理（避免 True → 'True'）
    if isinstance(value, bool):
        return ""

    return str(value)


def _normalize_amount_text(s: str) -> Optional[int]:
    """
    把全角数字/标点转半角，去掉非数字字符后尝试转换为 int。

    改进点：
    - 使用 normalize_text_basic 预处理（全角→半角 + 去换行 + 合并空格）
    - 支持更复杂的输入格式

    返回 None 表示无法解析。
    """
    if s is None:
        return None

    # 预处理：全角→半角 + 去换行 + 合并空格
    t = normalize_text_basic(str(s), strip=True)

    # 提取数字部分（支持逗号分隔）
    m = re.search(r"([0-9][0-9,]*)", t)
    if not m:
        return None

    numtxt = m.group(1).replace(",", "")
    try:
        return int(numtxt)
    except Exception:
        return None


# ======================================================================
# 公共类型强转函数（从 shop2/4/9/11/14 提取合并）
# ======================================================================

_INT_RE = re.compile(r"[+-]?\d+")

# 符号字符集（shop2 定义，多处引用）
SIGN_MINUS_CHARS = frozenset({"-", "−", "－", "–", "—", "―"})
SIGN_PLUS_CHARS = frozenset({"+", "＋"})


def coerce_int(val) -> Optional[int]:
    """
    把 int/float/str 的数字（含 '円'、'¥'、逗号、全角符号）稳健转成 int。

    合并自 shop2._coerce_int / shop11._coerce_int / shop4._coerce_int_maybe。
    统一处理：None / NaN / bool / int / float / 字符串（去逗号/去"円"/去全角符号）。
    """
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


def coerce_signed_int(x) -> Optional[int]:
    """
    全角数字/符号→半角，提取带符号整数。

    合并自 shop9._coerce_signed_int。
    处理：全角数字、全角 +/-、逗号分隔符。
    """
    if x is None:
        return None
    if isinstance(x, int) and not isinstance(x, bool):
        return int(x)

    s = str(x)
    # 全角数字/符号 -> 半角
    s = s.translate(str.maketrans("０１２３４５６７８９＋－−，", "0123456789+--,"))

    sign = 1
    digits = []
    sign_seen = False
    started = False
    for ch in s:
        if not started and not sign_seen and ch in "+-":
            sign_seen = True
            sign = -1 if ch == "-" else 1
            continue
        if ch.isdigit():
            started = True
            digits.append(ch)
            continue
        if started and ch in {",", " "}:
            # 千分位分隔符忽略
            continue
        if started:
            break

    if not digits:
        return None
    try:
        return sign * int("".join(digits))
    except Exception:
        return None


def coerce_amount_yen(v) -> Optional[int]:
    """
    带符号金额解析，支持 +/- 前缀和 to_int_yen 兜底。

    合并自 shop14._coerce_amount_yen。
    处理：符号前缀 → to_int_yen → 纯数字兜底。
    """
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
    if s[:1] in SIGN_PLUS_CHARS:
        s = s[1:].strip()
    elif s[:1] in SIGN_MINUS_CHARS:
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


# ======================================================================
# 公共提取模式调度（从 shop2/3/4/9/11/12/14 提取合并）
# ======================================================================

def _dispatch_extraction(
    mode: str,
    regex_fn: Callable,
    llm_fn: Optional[Callable] = None,
    *,
    has_result_fn: Optional[Callable] = None,
) -> tuple:
    """
    三模式（regex / llm / auto）提取调度的通用实现。

    合并自各 shop 原先的提取调度逻辑（regex/llm/auto if/elif/else）。
    当 llm_fn 为 None 时，仅使用正则路径（shop15/16/17 等）。

    参数:
        mode: "regex" | "llm" | "auto"（通常来自 EXTRACTION_MODE）
        regex_fn: 无参调用，返回提取结果（类型由 shop 自定）
        llm_fn: 无参调用，返回提取结果；None 时仅用 regex
        has_result_fn: 判断 regex 结果是否"有内容"的函数。
                       默认 bool(result)。用于 auto 模式下决定是否 fallback 到 LLM。

    返回:
        (result, method_str)
        method_str: "regex" | "llm"

    示例:
        result, method = _dispatch_extraction(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_regex(text),
            llm_fn=lambda: _extract_specs_llm(text, row_index=i),
        )
    """
    # llm_fn 为 None 时，仅正则路径（shop15/16/17 已移除 LLM）
    if llm_fn is None:
        return regex_fn(), "regex"

    check = has_result_fn or bool

    if mode == "regex":
        return regex_fn(), "regex"

    if mode == "llm":
        return llm_fn(), "llm"

    # auto: regex 优先，无结果时 LLM 兜底
    result = regex_fn()
    if check(result):
        return result, "regex"

    return llm_fn(), "llm"


# ======================================================================
# 公共提取模式调度 → PriceDecomposition（原各 shop dispatch 的公共部分）
# ======================================================================
# 说明：统一入口为 dispatch_extraction_to_price_decomposition，封装所有调度与后处理逻辑。
# 内部 helper 以 _ 前缀，仅供该函数调用。涵盖 shop2/3/4/9/11/12/14/15/16/17。


def _convert_all_color_maps_to_specs(
    delta_map: Dict[str, int],
    abs_map: Dict[str, int],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    将 map 格式（含 "ALL" 键）转为 (delta_specs, abs_specs) list 格式。

    合并自 shop9 的 abs_map / delta_map 处理逻辑。
    "ALL" 优先于 per-color 条目，优先级：delta_map["ALL"] > abs_map["ALL"]。

    参数:
        delta_map: { color_norm | "ALL" : delta_yen }
        abs_map: { color_norm | "ALL" : abs_yen }

    返回:
        (delta_specs, abs_specs)
    """
    if "ALL" in delta_map:
        return [("全色", int(delta_map["ALL"]))], []
    if "ALL" in abs_map:
        return [], [("全色", int(abs_map["ALL"]))]

    delta_specs = [(k, int(v)) for k, v in delta_map.items()]
    abs_specs = [(k, int(v)) for k, v in abs_map.items()]
    return delta_specs, abs_specs


def _merge_abs_overrides(
    abs_specs: List[Tuple[str, int]],
    overrides: Dict[str, int],
) -> List[Tuple[str, int]]:
    """
    将 overrides 合并进 abs_specs，同 label 时 override 覆盖原值。

    合并自 shop9 的 regex 路径后 _direct_abs_overrides_for_row 注入逻辑。

    参数:
        abs_specs: 原始 [(label, abs_yen), ...]
        overrides: { color_norm : abs_yen } 覆盖表

    返回:
        合并后的 abs_specs（override 的 label 在后，resolve_color_prices 中 per-color 会覆盖全色）
    """
    if not overrides:
        return list(abs_specs)
    seen = {str(lb).strip() for lb, _ in abs_specs}
    out = list(abs_specs)
    for col_norm, val in overrides.items():
        if col_norm and col_norm not in seen:
            out.append((col_norm, int(val)))
            seen.add(col_norm)
    return out


# ----------------------------------------------------------------------
# shop14: 多 fragment 聚合
# ----------------------------------------------------------------------

def _aggregate_fragment_extraction(
    frags: Dict[str, str],
    combined: str,
    parse_fn: Callable[[str], Tuple[Dict[str, Any], str]],
    *,
    all_delta_key: str = "all_delta",
    abs_key: str = "abs",
    delta_key: str = "delta",
) -> Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]], str]:
    """
    多 fragment 聚合提取。合并自 shop14 的 fragment 聚合逻辑。

    对 frags 各值依次调用 parse_fn(frag) -> (parsed_dict, method)，
    将 all_delta / abs / delta 聚合。若全部为空则对 combined 再跑一次 parse_fn。

    参数:
        frags: { logical_col : frag_text }
        combined: 合并后的兜底文本
        parse_fn: frag_text -> (parsed_dict, extraction_method)
        all_delta_key / abs_key / delta_key: parsed_dict 中的键名

    返回:
        (agg_all_delta, agg_abs, agg_delta, extraction_method)
    """
    agg_all_delta: Optional[int] = None
    agg_abs: List[Tuple[str, int]] = []
    agg_delta: List[Tuple[str, int]] = []
    method = "none"

    for _col, frag in frags.items():
        if not frag:
            continue
        parsed, m = parse_fn(frag)
        if parsed.get(all_delta_key) is not None:
            agg_all_delta = int(parsed[all_delta_key])
        agg_abs.extend(parsed.get(abs_key) or [])
        agg_delta.extend(parsed.get(delta_key) or [])
        method = m

    if combined and agg_all_delta is None and not agg_abs and not agg_delta:
        parsed2, method2 = parse_fn(combined)
        if parsed2.get(all_delta_key) is not None:
            agg_all_delta = int(parsed2[all_delta_key])
        agg_abs.extend(parsed2.get(abs_key) or [])
        agg_delta.extend(parsed2.get(delta_key) or [])
        method = method2

    return agg_all_delta, agg_abs, agg_delta, method


# ----------------------------------------------------------------------
# shop15/16: base_price 从结果提取 + LLM 返回 None 时 regex 回退
# ----------------------------------------------------------------------

def _apply_base_price_fallback_when_llm_none(
    raw_result: Any,
    method: str,
    regex_fn: Callable[[], Any],
    extract_base_fn: Callable[[Any], Optional[int]],
) -> Optional[int]:
    """
    当 method 为 "llm" 且 extract_base_fn(raw_result) 为 None 时，
    调用 regex_fn() 并用 extract_base_fn 提取 regex 的 base_price 作为回退。

    合并自 shop15、shop16 的 if method == "llm" and bp is None: bp = regex()[0]。

    参数:
        raw_result: dispatch_extraction 的原始结果
        method: "regex" | "llm"
        regex_fn: 无参可调用，返回 regex 原始结果
        extract_base_fn: raw -> Optional[int]，从原始结果提取 base_price

    返回:
        最终 base_price（含回退后的值）
    """
    base = extract_base_fn(raw_result)
    if method != "llm" or base is not None:
        return base
    regex_result = regex_fn()
    return extract_base_fn(regex_result)


# ----------------------------------------------------------------------
# 基础：list 格式全色归一化
# ----------------------------------------------------------------------

def _normalize_all_color_in_specs(
    delta_specs: List[Tuple[str, int]],
    abs_specs: List[Tuple[str, int]],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    「全色」/「ALL」归一化：当 delta 或 abs 中出现全色标签时，覆盖为单一全色条目并清空另一方。

    合并自 shop9、shop12、shop14 的重复逻辑。
    优先级：delta 全色 > abs 全色（与 shop9 的 if "ALL" in delta_map 优先一致）。

    参数:
        delta_specs: [(label, delta_yen), ...]
        abs_specs: [(label, abs_yen), ...]

    返回:
        (final_delta_specs, final_abs_specs)
    """
    _all_labels = frozenset({"全色", "ALL"})

    for label_raw, delta_val in delta_specs:
        if str(label_raw).strip() in _all_labels:
            return [("全色", int(delta_val))], []

    for label_raw, abs_val in abs_specs:
        if str(label_raw).strip() in _all_labels:
            return [], [("全色", int(abs_val))]

    return list(delta_specs), list(abs_specs)


def dispatch_extraction_to_price_decomposition(
    mode: Optional[str] = None,
    *,
    base_price: Optional[int] = None,
    source_text_raw: str = "",
    regex_fn: Optional[Callable[[], Any]] = None,
    llm_fn: Optional[Callable[[], Any]] = None,
    result_adapter: Optional[Callable[[Any], Union[
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]],
        Tuple[Optional[int], List[Tuple[str, int]], List[Tuple[str, int]]],
    ]]] = None,
    has_result_fn: Optional[Callable[[Any], bool]] = None,
    frags: Optional[Dict[str, str]] = None,
    combined: Optional[str] = None,
    parse_fn: Optional[Callable[[str], Tuple[Dict[str, Any], str]]] = None,
    all_delta_key: str = "all_delta",
    abs_key: str = "abs",
    delta_key: str = "delta",
    apply_all_color_override: bool = True,
    regex_post_hook: Optional[Callable[[], Dict[str, int]]] = None,
    result_is_maps: bool = False,
    base_price_when_none: Optional[int] = None,
    extract_base_from_result: Optional[Callable[[Any], Optional[int]]] = None,
    as_parse_fn: bool = False,
) -> Union[PriceDecomposition, Tuple[Dict[str, Any], str]]:
    """
    提取调度 → PriceDecomposition 的统一入口，封装所有模式与后处理逻辑。

    三种互斥模式：
    1) 单次提取 mode+regex_fn+llm_fn+result_adapter → PriceDecomposition
    2) Fragment 聚合 frags+combined+parse_fn → PriceDecomposition
    3) 单 fragment parse（as_parse_fn=True）mode+regex_fn+llm_fn → (parsed_dict, method)

    流程:
      1. 调用 dispatch_extraction(mode, regex_fn, llm_fn, has_result_fn)
      2. 使用 result_adapter 将原始结果转为 (delta_specs, abs_specs)
      3. 可选执行 normalize_all_color_in_specs（全色覆盖）
      4. 组装并返回 PriceDecomposition

    参数:
        mode: "regex" | "llm" | "auto"（通常取 EXTRACTION_MODE）
        regex_fn: 无参可调用，返回正则提取的原始结果
        llm_fn: 无参可调用，返回 LLM 提取的原始结果
        base_price: 基准价（日元），可为 None（仅 abs 路径时）
        source_text_raw: 提取来源的原始文本
        result_adapter: raw_result -> (delta_specs, abs_specs)
        has_result_fn: 判断 regex 结果是否“有内容”，用于 auto 模式
        apply_all_color_override: 是否执行全色归一化（默认 True）

    返回:
        PriceDecomposition

    示例（shop12 风格）:
        def _adapter(r):
            abs_list, delta_list = r
            return delta_list, abs_list

        decomp = dispatch_extraction_to_price_decomposition(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_shop12_regex(remark_for_llm),
            llm_fn=lambda: _extract_specs_shop12_llm_impl(remark_for_llm, idx=idx, fallback_parse_rules_fn=_fallback_parse_rules),
            base_price=base_price,
            source_text_raw=source_text_raw,
            result_adapter=_adapter,
            has_result_fn=lambda r: bool(r[0] or r[1]),
        )

    示例（shop14 单 fragment parse）:
        parse_fn = lambda t: dispatch_extraction_to_price_decomposition(
            EXTRACTION_MODE,
            regex_fn=lambda: _extract_specs_shop14_regex(t),
            llm_fn=lambda: _extract_specs_shop14_llm_impl(t, ...),
            has_result_fn=lambda r: r.get("all_delta") is not None or r.get("abs") or r.get("delta"),
            as_parse_fn=True,
        )
    """
    if as_parse_fn:
        if mode is None or regex_fn is None:
            raise ValueError("as_parse_fn 模式需 mode/regex_fn")
        raw_result, method = _dispatch_extraction(
            mode,
            regex_fn=regex_fn,
            llm_fn=llm_fn,
            has_result_fn=has_result_fn,
        )
        return (raw_result, method)

    if frags is not None and combined is not None and parse_fn is not None:
        agg_all_delta, agg_abs, agg_delta, method = _aggregate_fragment_extraction(
            frags, combined, parse_fn,
            all_delta_key=all_delta_key, abs_key=abs_key, delta_key=delta_key,
        )
        delta_specs = list(agg_delta)
        abs_specs = list(agg_abs)
        if agg_all_delta is not None:
            delta_specs = [("全色", agg_all_delta)]
            abs_specs = []
    else:
        if mode is None or regex_fn is None or result_adapter is None:
            raise ValueError(
                "单次提取需 mode/regex_fn/result_adapter；Fragment 模式需 frags/combined/parse_fn"
            )
        raw_result, method = _dispatch_extraction(
            mode,
            regex_fn=regex_fn,
            llm_fn=llm_fn,
            has_result_fn=has_result_fn,
        )
        adapted = result_adapter(raw_result)
        if len(adapted) == 3:
            base_from_result, delta_specs, abs_specs = adapted
            if base_from_result is not None:
                base_price = base_from_result
        else:
            delta_specs, abs_specs = adapted

        if result_is_maps and isinstance(raw_result, (tuple, list)) and len(raw_result) >= 2:
            abs_map, delta_map = raw_result[0], raw_result[1]
            delta_specs, abs_specs = _convert_all_color_maps_to_specs(delta_map, abs_map)

        if extract_base_from_result is not None:
            base_price = _apply_base_price_fallback_when_llm_none(
                raw_result, method, regex_fn, extract_base_from_result,
            )

        if method == "regex" and regex_post_hook is not None:
            overrides = regex_post_hook()
            if overrides:
                abs_specs = _merge_abs_overrides(abs_specs, overrides)

    if apply_all_color_override:
        delta_specs, abs_specs = _normalize_all_color_in_specs(delta_specs, abs_specs)

    final_base = base_price
    if final_base is None and base_price_when_none is not None:
        final_base = base_price_when_none
    if final_base is None and delta_specs:
        delta_specs = []

    return PriceDecomposition(
        base_price=final_base,
        delta_specs=[(lb, int(d)) for lb, d in delta_specs],
        abs_specs=[(lb, int(a)) for lb, a in abs_specs],
        extraction_method=method,
        source_text_raw=source_text_raw,
    )


# ======================================================================
# 公共正则模式 & 标签清洗（从 shop3/4/9/12 等提取合并）
# ======================================================================

def clean_label_token(tok: str) -> str:
    """
    清洗标签 token：去除括号内容（半角+全角），strip 空白。

    合并自 shop3._clean_label_token（shop3/4/12 等多处使用相同逻辑）。
    """
    if tok is None:
        return ""
    t = str(tok).strip()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"（.*?）", "", t)
    return t.strip()


# 颜色±金额 正则模式（各 shop 共用的基础变体）
# DELTA_PATTERN_STRICT: 严格模式 —— 标签后紧跟 ±金额（shop3/12 等使用）
DELTA_PATTERN_STRICT = re.compile(
    r"""(?P<labels>[^+\-−－\d¥￥円]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# DELTA_PATTERN_LOOSE: 宽松模式 —— 允许日文/汉字/连字符等标签字符（shop3 fallback）
DELTA_PATTERN_LOOSE = re.compile(
    r"""(?P<labels>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# ABS_PRICE_PATTERN: 绝对价格模式 —— 标签后跟金额，无 ± 符号（shop9/12 等使用）
ABS_PRICE_PATTERN = re.compile(
    r"""(?P<labels>[^0-9０-９¥￥円]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?""",
    re.I,
)

# SIGNED_AMOUNT_PATTERN: 提取文本中所有 ±金额（shop3 的 _extract_signed_amounts_from_text）
SIGNED_AMOUNT_PATTERN = re.compile(r"([+\-−－])\s*([0-9０-９][0-9０-９,，]*)")


# ======================================================================
# 公共 LLM Guardrails（从 shop2/3/4/12 提取合并）
# ======================================================================

def llm_guardrail_check(
    label: str,
    amount: int,
    source_text: str,
    *,
    check_label: bool = True,
    check_amount: bool = True,
) -> bool:
    """
    LLM 提取结果的通用防幻觉验证。

    合并自 shop2/3/4/12 中重复出现的 Guardrail A/B 逻辑。

    Guardrail A: label（颜色标签）必须在原文中出现
    Guardrail B: amount（金额）的绝对值必须在原文中出现（去逗号后比较）

    参数:
        label: LLM 提取的颜色标签
        amount: LLM 提取的金额（int）
        source_text: 原始文本
        check_label: 是否检查标签存在性（默认 True）
        check_amount: 是否检查金额存在性（默认 True）

    返回:
        True = 通过验证，False = 疑似幻觉应丢弃
    """
    if not source_text:
        return False

    # Guardrail A: label 在原文中
    if check_label and label:
        if label not in source_text:
            return False

    # Guardrail B: 金额绝对值在原文中（去逗号比较）
    if check_amount and amount is not None and amount != 0:
        text_no_commas = source_text.replace(",", "").replace("，", "")
        if str(abs(int(amount))) not in text_no_commas:
            return False

    return True


def apply_llm_guardrails(
    rules: dict,
    source_text: str,
    *,
    check_label: bool = True,
    check_amount: bool = True,
) -> dict:
    """
    对 {label: amount} 字典批量应用 LLM guardrails，返回过滤后的字典。

    合并自 shop2（llm_shop2）中的 LLM 过滤循环。

    参数:
        rules: {group_label: delta_yen} 字典
        source_text: 原始文本
        check_label/check_amount: 同 llm_guardrail_check

    返回:
        过滤后的 {label: amount} 字典
    """
    if not rules:
        return {}
    return {
        label: int(amount)
        for label, amount in rules.items()
        if llm_guardrail_check(
            label, amount, source_text,
            check_label=check_label,
            check_amount=check_amount,
        )
    }


def _build_jan_map(info_df: pd.DataFrame) -> Dict[str, str]:
    """
    构建 { jan_digits -> part_number } 映射。

    自动在 info_df 中查找名为 jan / jancode / jan_code 的列。
    若不存在则返回空字典。
    """
    jan_map: Dict[str, str] = {}
    jcol = None
    for c in info_df.columns:
        if str(c).strip().lower() in {"jan", "jancode", "jan_code"}:
            jcol = c
            break
    if not jcol:
        return jan_map
    for _, r in info_df.iterrows():
        jan_digits = _extract_jan_digits(r.get(jcol))
        pn = r.get("part_number")
        if jan_digits and pd.notna(pn):
            jan_map[str(jan_digits)] = str(pn)
    return jan_map


# ======================================================================
# 统一价格分解 & 下游匹配
# ======================================================================

# label_matcher 签名：(label_raw, color_raw, color_norm) -> bool
LabelMatcherType = Callable[[str, str, str], bool]

_ALL_COLOR_LABELS = frozenset({"全色", "ALL"})


@dataclass
class PriceDecomposition:
    """
    各 shop 提取层的统一输出结构。

    Attributes:
        base_price: 基准价（日元整数）
        delta_specs: [(label_raw, delta_int)] — 颜色级别的相对调整
        abs_specs: [(label_raw, abs_price)] — 颜色级别的绝对价格
        extraction_method: 提取方式标识 ("regex" / "llm" / "auto" / "none")
        source_text_raw: 提取来源的原始文本（用于日志）

    注意:
        - delta_specs 中若包含 "全色"/"ALL" 标签，应置于列表前部，
          以便后续的 per-color 条目可以覆盖它。
        - 独立型 shop（仅 delta）令 abs_specs 为空即可。
    """
    base_price: Optional[int] = None
    delta_specs: List[Tuple[str, int]] = field(default_factory=list)
    abs_specs: List[Tuple[str, int]] = field(default_factory=list)
    extraction_method: str = "none"
    source_text_raw: str = ""


def resolve_color_prices(
    decomp: PriceDecomposition,
    color_map: Dict[str, Tuple[str, str]],
    label_matcher: LabelMatcherType,
    *,
    shop_name: str,
    cleaner_name: str,
    recorded_at: object,
    emit_default_rows: bool = True,
    skip_non_positive: bool = False,
    logger: Optional[logging.Logger] = None,
    log_seq_start: int = 0,
    row_index: int = -1,
    model_text: str = "",
    model_norm: str = "",
    capacity_gb: int = 0,
) -> Tuple[List[dict], int]:
    """
    从 PriceDecomposition + color_map 生成输出行。

    统一的下游匹配 & 定价流程，替代各 shop 中 100~200 行的重复循环体。

    处理流程：
      1. 记录 extraction_result 日志
      2. label → color 匹配（delta + abs），"全色"/"ALL" 自动匹配所有颜色
      3. 计算最终价格，优先级：abs > delta > base_price
      4. 生成输出行并记录 output_record / row_processing_summary 日志

    参数:
        decomp: 价格分解结果
        color_map: {color_norm: (part_number, color_raw)} 颜色→PN 映射
        label_matcher: shop 级别的匹配函数 (label_raw, color_raw, color_norm) -> bool
        shop_name: 店铺名（用于输出行和日志）
        cleaner_name: 清洗器名（用于日志）
        recorded_at: 记录时间
        emit_default_rows: 未匹配颜色是否生成行（False → 仅输出有明确定价的颜色）
        skip_non_positive: 最终价格 <= 0 时使用基准价替代（True → 该颜色用 base_price，仅 warning 日志）
        logger: 日志器（None 则跳过所有日志）
        log_seq_start: 日志序号起始值
        row_index: 行号（用于日志）
        model_text / model_norm / capacity_gb: 用于日志上下文

    返回:
        (output_rows, log_seq_end)
        output_rows: [{"part_number", "shop_name", "price_new", "recorded_at"}]
        log_seq_end: 更新后的日志序号
    """
    _seq = log_seq_start
    base_price = decomp.base_price
    source_text_raw_full = decomp.source_text_raw
    extraction_method = decomp.extraction_method

    # 确保 "全色"/"ALL" 条目在前，per-color 条目在后可覆盖
    delta_specs = sorted(
        decomp.delta_specs,
        key=lambda x: 0 if str(x[0]).strip() in _ALL_COLOR_LABELS else 1,
    )
    abs_specs = list(decomp.abs_specs)

    # ── 1. extraction_result 日志 ─────────────────────────────────────
    if logger:
        available_colors_list = [
            {"color_norm": cn, "part_number": pn, "color_raw": cr}
            for cn, (pn, cr) in color_map.items()
        ]
        _seq += 1
        logger.debug(
            "Extraction result",
            extra={
                "event_type": "extraction_result",
                "log_seq": _seq,
                "shop_name": shop_name,
                "cleaner_name": cleaner_name,
                "row_index": row_index,
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": capacity_gb,
                "base_price": base_price,
                "source_text_raw": _truncate_for_log(source_text_raw_full, 200),
                "source_text_raw_full": (source_text_raw_full or "None"),
                "source_text_normalized": _truncate_for_log(
                    normalize_text_basic(source_text_raw_full) if source_text_raw_full else "", 200
                ),
                "extraction_method": extraction_method,
                "labels_and_deltas": [
                    {"label": lb, "delta": d} for lb, d in delta_specs
                ],
                "abs_prices": [
                    {"label": lb, "amount": amt} for lb, amt in abs_specs
                ],
                "labels_extracted_count": len(delta_specs),
                "abs_prices_count": len(abs_specs),
                "available_colors": available_colors_list,
                "colors_in_catalog": len(color_map),
            },
        )

    # 共通日志上下文（label_matching / label_no_match 共用）
    _log_ctx: dict = {
        "shop_name": shop_name,
        "cleaner_name": cleaner_name,
        "row_index": row_index,
        "model_text": model_text,
        "model_norm": model_norm,
        "capacity_gb": capacity_gb,
        "base_price": base_price,
        "source_text_raw_full": (source_text_raw_full or "None"),
        "labels_and_deltas": [
            {"label": lb, "delta": d} for lb, d in delta_specs
        ],
    }

    # ── 2. label → color 匹配 ────────────────────────────────────────
    color_delta_map: Dict[str, int] = {}
    color_delta_label_map: Dict[str, str] = {}
    color_abs_map: Dict[str, int] = {}
    color_abs_label_map: Dict[str, str] = {}

    # -- Delta 匹配 --
    for label_raw, delta_val in delta_specs:
        is_all = str(label_raw).strip() in _ALL_COLOR_LABELS
        matched_colors: List[str] = []
        matched_pns: List[str] = []

        for col_norm, (pn, col_raw) in color_map.items():
            if is_all or label_matcher(label_raw, col_raw, col_norm):
                color_delta_map[col_norm] = int(delta_val)
                color_delta_label_map[col_norm] = label_raw
                matched_colors.append(col_norm)
                matched_pns.append(pn)

        if logger:
            _seq += 1
            if matched_colors:
                logger.debug(
                    f"Label matching (delta): {label_raw}",
                    extra={
                        **_log_ctx,
                        "event_type": "label_matching",
                        "log_seq": _seq,
                        "label": label_raw,
                        "delta": delta_val,
                        "match_type": "delta",
                        "matched_colors": matched_colors,
                        "matched_part_numbers": matched_pns,
                        "match_count": len(matched_colors),
                    },
                )
            else:
                logger.warning(
                    f"Label not matched (delta): {label_raw}",
                    extra={
                        **_log_ctx,
                        "event_type": "label_no_match",
                        "log_seq": _seq,
                        "label": label_raw,
                        "delta": delta_val,
                        "match_type": "delta",
                        "available_colors": list(color_map.keys()),
                    },
                )

    # -- Abs 匹配 --
    for label_raw, abs_price in abs_specs:
        is_all = str(label_raw).strip() in _ALL_COLOR_LABELS
        matched_colors = []
        matched_pns = []

        for col_norm, (pn, col_raw) in color_map.items():
            if is_all or label_matcher(label_raw, col_raw, col_norm):
                color_abs_map[col_norm] = int(abs_price)
                color_abs_label_map[col_norm] = label_raw
                matched_colors.append(col_norm)
                matched_pns.append(pn)

        if logger:
            _seq += 1
            if matched_colors:
                logger.debug(
                    f"Label matching (abs): {label_raw}",
                    extra={
                        **_log_ctx,
                        "event_type": "label_matching",
                        "log_seq": _seq,
                        "label": label_raw,
                        "abs_price": abs_price,
                        "match_type": "abs",
                        "matched_colors": matched_colors,
                        "matched_part_numbers": matched_pns,
                        "match_count": len(matched_colors),
                    },
                )
            else:
                logger.warning(
                    f"Label not matched (abs): {label_raw}",
                    extra={
                        **_log_ctx,
                        "event_type": "label_no_match",
                        "log_seq": _seq,
                        "label": label_raw,
                        "abs_price": abs_price,
                        "match_type": "abs",
                        "available_colors": list(color_map.keys()),
                    },
                )

    # ── 3. 各色价格计算 + 输出行生成 ─────────────────────────────────
    output_rows: List[dict] = []
    current_row_records: List[dict] = []
    colors_matched = 0

    for col_norm, (pn, col_raw) in color_map.items():
        # 优先级：abs > delta > default(base_price)
        if col_norm in color_abs_map:
            effective_source = "abs_price"
            matched_label = color_abs_label_map[col_norm]
            spec_value = color_abs_map[col_norm]
            final_price = spec_value
        elif col_norm in color_delta_map:
            effective_source = "matched_label"
            matched_label = color_delta_label_map[col_norm]
            spec_value = color_delta_map[col_norm]
            if base_price is None:
                raise ValueError(
                    "resolve_color_prices: base_price is None but color has delta match. "
                    "Caller must clear delta_specs when base_price is None."
                )
            final_price = base_price + spec_value
        else:
            effective_source = "default_zero"
            matched_label = None
            spec_value = None
            final_price = base_price
            if base_price is None and emit_default_rows:
                raise ValueError(
                    "resolve_color_prices: base_price is None and emit_default_rows=True. "
                    "Cannot output default rows without base price."
                )

        if effective_source != "default_zero":
            colors_matched += 1

        # emit_default_rows=False → 未匹配颜色不生成行
        if not emit_default_rows and effective_source == "default_zero":
            continue

        # skip_non_positive 且 base_price 为空时，无法替代 ≤0 价格，则跳过
        if (
            skip_non_positive
            and final_price is not None
            and int(final_price) <= 0
            and base_price is None
        ):
            if logger:
                _seq += 1
                logger.warning(
                    "Skipping item: price <= 0 and base_price is None",
                    extra={
                        **_log_ctx,
                        "event_type": "output_record",
                        "log_seq": _seq,
                        "part_number": pn,
                        "color_norm": col_norm,
                        "spec_value": spec_value,
                        "final_price": int(final_price),
                        "skip_reason": "price <= 0, no base_price to substitute",
                    },
                )
            continue

        # 价格合理性校验：超出范围时用 base_price 替代，记 warning
        price_override_reason: Optional[str] = None
        original_final_price = int(final_price) if final_price is not None else None
        if final_price is not None and base_price is not None:
            fp = int(final_price)
            bp = int(base_price)
            if skip_non_positive and fp <= 0:
                final_price = base_price
                price_override_reason = "price <= 0"
            elif fp > int(bp * 1.5):
                final_price = base_price
                price_override_reason = "price > 1.5x base"
            elif fp < int(bp * 0.5):
                final_price = base_price
                price_override_reason = "price < 0.5x base"

        if price_override_reason and logger:
            _seq += 1
            logger.warning(
                f"Price override: {price_override_reason}, using base_price",
                extra={
                    **_log_ctx,
                    "event_type": "output_record",
                    "log_seq": _seq,
                    "part_number": pn,
                    "color_norm": col_norm,
                    "base_price": base_price,
                    "spec_value": spec_value,
                    "original_final_price": original_final_price,
                    "final_price": int(final_price),
                    "override_reason": price_override_reason,
                },
            )

        output_rows.append({
            "part_number": pn,
            "shop_name": shop_name,
            "price_new": int(final_price),
            "recorded_at": recorded_at,
        })

        current_row_records.append({
            "part_number": pn,
            "color_norm": col_norm,
            "final_price": int(final_price),
            "recorded_at": recorded_at,
            "effective_source": effective_source,
            "matched_label": matched_label,
            "spec_value": spec_value,
        })

        # output_record (DEBUG)
        if logger:
            _seq += 1
            logger.debug(
                f"Output record: {pn}",
                extra={
                    **_log_ctx,
                    "event_type": "output_record",
                    "log_seq": _seq,
                    "part_number": pn,
                    "color_norm": col_norm,
                    "color_raw": col_raw,
                    "final_price": int(final_price),
                    "effective_source": effective_source,
                    "matched_label": matched_label,
                    "spec_value": spec_value,
                    "recorded_at": str(recorded_at) if recorded_at else None,
                },
            )

    # ── 4. row_processing_summary 日志 ────────────────────────────────
    all_spec_values = [
        r["spec_value"] for r in current_row_records
        if r["spec_value"] is not None
    ]

    if logger:
        # DEBUG 级别：详细
        _seq += 1
        logger.debug(
            "Row summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _seq,
                "shop_name": shop_name,
                "cleaner_name": cleaner_name,
                "row_index": row_index,
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": capacity_gb,
                "base_price": base_price,
                "source_text_raw_full": (source_text_raw_full or "None"),
                "abs_applied_details": [
                    {
                        "pn": r["part_number"],
                        "color": r["color_norm"],
                        "final_price": r["final_price"],
                        "matched_label": r["matched_label"],
                        "spec_value": r["spec_value"],
                    }
                    for r in current_row_records
                    if r["effective_source"] == "abs_price"
                ],
                "delta_applied_details": [
                    {
                        "pn": r["part_number"],
                        "color": r["color_norm"],
                        "final_price": r["final_price"],
                        "matched_label": r["matched_label"],
                        "spec_value": r["spec_value"],
                    }
                    for r in current_row_records
                    if r["effective_source"] == "matched_label"
                ],
                "default_applied_pns": [
                    r["part_number"]
                    for r in current_row_records
                    if r["effective_source"] == "default_zero"
                ],
            },
        )

        # INFO 级别：一行摘要
        _seq += 1
        _model_display = model_text[:28] if len(model_text) > 28 else model_text
        logger.info(
            f"Row {row_index:<3d} | {_model_display:<28s}"
            f" | deltas: {len(delta_specs):<2d}"
            f" | abs: {len(abs_specs):<2d}"
            f" | matched: {colors_matched:<2d}"
            f" | records: {len(current_row_records):<2d}"
            f" | method: {extraction_method}",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _seq,
                "shop_name": shop_name,
                "cleaner_name": cleaner_name,
                "row_index": row_index,
                "model_text": model_text,
                "model_norm": model_norm,
                "capacity_gb": capacity_gb,
                "base_price": base_price,
                "source_text_raw_preview": _truncate_for_log(source_text_raw_full, 100),
                "extraction_method": extraction_method,
                "labels_extracted_count": len(delta_specs),
                "abs_prices_extracted_count": len(abs_specs),
                "colors_in_catalog": len(color_map),
                "colors_matched_count": colors_matched,
                "output_records_count": len(current_row_records),
                "has_discounted_colors": any(v != 0 for v in all_spec_values),
                "min_delta": min(all_spec_values) if all_spec_values else 0,
                "max_delta": max(all_spec_values) if all_spec_values else 0,
            },
        )

    return output_rows, _seq


# ======================================================================
# 1. DataFrame 输出组装
# ======================================================================

_OUTPUT_COLUMNS = ["part_number", "shop_name", "price_new", "recorded_at"]


def assemble_output_df(
    rows: List[dict],
    *,
    coerce_price: bool = True,
) -> pd.DataFrame:
    """
    将行列表组装为标准输出 DataFrame 并做统一的后处理。

    处理:
      1. 创建 DataFrame (columns = part_number, shop_name, price_new, recorded_at)
      2. dropna(subset=["part_number", "price_new"])
      3. part_number → str
      4. price_new → Int64 (可选, coerce_price=True 时)

    参数:
        rows: [{"part_number", "shop_name", "price_new", "recorded_at"}, ...]
        coerce_price: 是否将 price_new 转为 Int64 (默认 True)

    返回:
        pd.DataFrame — 列: part_number(str), shop_name, price_new(Int64), recorded_at
    """
    out = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        if coerce_price:
            out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out


def dedupe_output_keep_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 (part_number, shop_name) 去重，保留 recorded_at（time-scraped）最新的一行。
    供 run_cleaner 统一调用，对所有 shop 输出生效。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "part_number" not in df.columns or "recorded_at" not in df.columns:
        return df
    subset = ["part_number", "shop_name"] if "shop_name" in df.columns else ["part_number"]
    df_sorted = df.sort_values("recorded_at", ascending=True, na_position="last")
    return df_sorted.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)


# ======================================================================
# 2. 清洗器 开始/完了 日志
# ======================================================================

def log_cleaner_start(
    logger: logging.Logger,
    *,
    cleaner_name: str,
    shop_name: str,
    input_rows: int,
    log_seq: int = 0,
    extraction_mode: Optional[str] = None,
) -> None:
    """清洗器开始时的统一日志。"""
    extra: dict = {
        "event_type": "cleaner_start",
        "shop_name": shop_name,
        "cleaner_name": cleaner_name,
        "log_seq": log_seq,
        "input_rows": input_rows,
    }
    if extraction_mode is not None:
        extra["extraction_mode"] = extraction_mode
    logger.info(f"{cleaner_name} cleaner started", extra=extra)


def log_cleaner_complete(
    logger: logging.Logger,
    *,
    cleaner_name: str,
    shop_name: str,
    input_rows: int,
    output_records: int,
    start_time: float,
    log_seq: int = 0,
) -> None:
    """清洗器完了時的统一日志。"""
    elapsed = round(time.time() - start_time, 2)
    logger.info(
        f"{cleaner_name} cleaner completed",
        extra={
            "event_type": "cleaner_complete",
            "shop_name": shop_name,
            "cleaner_name": cleaner_name,
            "log_seq": log_seq,
            "input_rows": input_rows,
            "output_records": output_records,
            "elapsed_seconds": elapsed,
        },
    )


# ======================================================================
# 3. 行スキップ日志
# ======================================================================

def log_row_skip(
    logger: logging.Logger,
    *,
    cleaner_name: str,
    shop_name: str,
    row_index: int,
    skip_reason: str,
    log_seq: int = 0,
    **extra_fields: object,
) -> None:
    """行がスキップされた際の統一 DEBUG ログ。

    Parameters
    ----------
    extra_fields : keyword arguments
        追加コンテキスト (model_text, capacity_gb, data_raw, …) を
        そのまま extra dict にマージする。
    """
    extra: dict = {
        "event_type": "row_skip",
        "shop_name": shop_name,
        "cleaner_name": cleaner_name,
        "log_seq": log_seq,
        "row_index": row_index,
        "skip_reason": skip_reason,
    }
    extra.update(extra_fields)
    logger.debug(
        f"Row {row_index}: skip ({skip_reason})",
        extra=extra,
    )


# ======================================================================
# 4. 必須列バリデーション
# ======================================================================

def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    *,
    cleaner_name: str,
    shop_name: str,
    logger: Optional[logging.Logger] = None,
    log_seq: int = 0,
    lenient: bool = False,
) -> int:
    """必須列の存在を検証し、欠落時にログ出力 + エラーを投げる。

    Parameters
    ----------
    lenient : bool
        True の場合、欠落列を None で埋めて処理を続行する (shop2 方式)。
        False の場合 (デフォルト)、ValueError を送出する。

    Returns
    -------
    int
        更新後の log_seq (ログ出力した分だけインクリメント)。
    """
    seq = log_seq
    for c in required:
        if c not in df.columns:
            if logger is not None:
                if lenient:
                    logger.warning(
                        f"Missing column, filling with None: {c}",
                        extra={
                            "event_type": "validation_error",
                            "shop_name": shop_name,
                            "cleaner_name": cleaner_name,
                            "log_seq": seq,
                            "missing_column": c,
                            "available_columns": list(df.columns),
                        },
                    )
                else:
                    logger.error(
                        f"Missing required column: {c}",
                        extra={
                            "event_type": "validation_error",
                            "shop_name": shop_name,
                            "cleaner_name": cleaner_name,
                            "log_seq": seq,
                            "missing_column": c,
                            "available_columns": list(df.columns),
                        },
                    )
                seq += 1
            if lenient:
                df[c] = None
            else:
                raise ValueError(f"{cleaner_name} 清洗器缺少必要列：{c}")
    return seq


# ======================================================================
# 5. 清洗器主函数模板 — A类（JAN码直查）/ B类（型号容量匹配）
# ======================================================================

def clean_with_jan_matching(
    df: pd.DataFrame,
    *,
    cleaner_name: str,
    shop_name: str,
    iter_records_fn: Callable,
    jan_extractor_fn: Optional[Callable] = None,
    price_extractor_fn: Optional[Callable] = None,
    ts_extractor_fn: Optional[Callable] = None,
    row_filter_fn: Optional[Callable] = None,
    fallback_match_fn: Optional[Callable] = None,
    coerce_price: bool = True,
) -> pd.DataFrame:
    """
    A类清洗器通用模板：JAN码 → part_number 映射。

    Parameters
    ----------
    df : 原始 DataFrame
    cleaner_name : 清洗器标识（如 "shop1"）
    shop_name : 店铺显示名（如 "買取商店"）
    iter_records_fn : 接收 df，产出规范化记录字典的可迭代函数
        每条记录至少包含 "JAN" / "price" / "time-scraped" 键
    jan_extractor_fn : 从 JAN 字段提取数字的函数（默认 _extract_jan_digits）
    price_extractor_fn : 价格解析函数（默认 extract_price_yen）
    ts_extractor_fn : 时间解析函数（默认 parse_dt_aware）
    row_filter_fn : 可选的行级过滤（接收记录字典，返回 True 保留）
    fallback_match_fn : JAN 无法匹配时的回退函数
        签名: (record, info_df) -> Optional[str]  返回 part_number 或 None
    coerce_price : 是否对价格做 Int64 强制转换
    """
    _logger = logging.getLogger(f"cleaner_tools.{cleaner_name}")

    start_time = time.time()
    log_cleaner_start(_logger, cleaner_name=cleaner_name, shop_name=shop_name, input_rows=len(df))

    if df.empty:
        log_cleaner_complete(_logger, cleaner_name=cleaner_name, shop_name=shop_name,
                             input_rows=len(df), output_records=0, start_time=start_time)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    jan_ext = jan_extractor_fn or _extract_jan_digits
    price_ext = price_extractor_fn or extract_price_yen
    ts_ext = ts_extractor_fn or parse_dt_aware

    info_df = _load_iphone17_info_df_from_db()
    jan_map = _build_jan_map(info_df)

    rows: List[dict] = []

    for rec in iter_records_fn(df):
        if row_filter_fn and not row_filter_fn(rec):
            continue

        jan_digits = jan_ext(rec.get("JAN"))
        part_number: Optional[str] = None

        if jan_digits:
            part_number = jan_map.get(jan_digits)

        if not part_number and fallback_match_fn:
            part_number = fallback_match_fn(rec, info_df)

        if not part_number:
            continue

        price_new = price_ext(rec.get("price"))
        if price_new is None:
            continue

        recorded_at = ts_ext(rec.get("time-scraped"))

        rows.append({
            "part_number": str(part_number),
            "shop_name": shop_name,
            "price_new": int(price_new),
            "recorded_at": recorded_at,
        })

    out = assemble_output_df(rows, coerce_price=coerce_price)
    log_cleaner_complete(_logger, cleaner_name=cleaner_name, shop_name=shop_name,
                         input_rows=len(df), output_records=len(out), start_time=start_time)
    return out


def clean_with_model_capacity_matching(
    df: pd.DataFrame,
    *,
    cleaner_name: str,
    shop_name: str,
    model_col: str,
    price_col: str,
    ts_col: str = "time-scraped",
    required_cols: Optional[List[str]] = None,
    model_normalizer_fn: Optional[Callable] = None,
    price_extractor_fn: Optional[Callable] = None,
    pn_extractor_fn: Optional[Callable] = None,
    model_cap_color_extractor_fn: Optional[Callable[[str], Optional[Tuple[str, int, str]]]] = None,
    row_filter_fn: Optional[Callable] = None,
    add_model_norm: bool = False,
    coerce_price: bool = False,
) -> pd.DataFrame:
    """
    B类清洗器通用模板：(model, capacity) → 全色 part_number 展开。

    Parameters
    ----------
    df : 原始 DataFrame
    cleaner_name : 清洗器标识
    shop_name : 店铺显示名
    model_col : 型号文本所在列名
    price_col : 价格所在列名
    ts_col : 时间列名（默认 "time-scraped"）
    required_cols : 需要校验的列名列表（默认 [model_col, price_col, ts_col]）
    model_normalizer_fn : 机型归一化函数（默认 _normalize_model_generic）
    price_extractor_fn : 价格解析函数（默认 extract_price_yen）
    pn_extractor_fn : 可选的 part_number 直接提取函数
        签名: (row_text) -> Optional[str]
        若提供且返回非 None，输出 1 行
    model_cap_color_extractor_fn : 可选的 (model, cap, color) 提取函数
        签名: (model_col_text) -> Optional[(model_norm, cap_gb, color_raw)]
        若提供且返回非 None 且 color_map 匹配成功，输出 1 行（不展开全色）
        优先级高于 pn_extractor_fn
    row_filter_fn : 可选的行级过滤（接收 row Series/dict，返回 True 保留）
    add_model_norm : 传递给 _load_iphone17_info_df_from_db
    coerce_price : assemble_output_df 的参数
    """
    _logger = logging.getLogger(f"cleaner_tools.{cleaner_name}")

    start_time = time.time()
    log_cleaner_start(_logger, cleaner_name=cleaner_name, shop_name=shop_name, input_rows=len(df))

    cols_to_validate = required_cols or [model_col, price_col, ts_col]
    validate_columns(df, cols_to_validate, cleaner_name=cleaner_name, shop_name=shop_name)

    if df.empty:
        log_cleaner_complete(_logger, cleaner_name=cleaner_name, shop_name=shop_name,
                             input_rows=len(df), output_records=0, start_time=start_time)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    model_norm_fn = model_normalizer_fn or _normalize_model_generic
    price_ext = price_extractor_fn or extract_price_yen

    info_df = _load_iphone17_info_df_from_db(add_model_norm=add_model_norm)
    if not add_model_norm:
        info_df = info_df.copy()
        info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
        info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")

    groups = (
        info_df.groupby(["model_name_norm", "capacity_gb"])["part_number"]
        .apply(list).to_dict()
    )

    color_map: Optional[Dict] = None
    if model_cap_color_extractor_fn:
        color_map = _build_color_map(info_df)

    model_norm_series = df[model_col].map(model_norm_fn)
    cap_gb_series = df[model_col].map(_parse_capacity_gb)
    price_series = df[price_col].map(price_ext)
    recorded_at_series = df[ts_col].map(parse_dt_aware)

    rows: List[dict] = []
    _log_seq = 1
    for i in range(len(df)):
        if row_filter_fn and not row_filter_fn(df.iloc[i]):
            continue

        p = price_series.iat[i]
        t = recorded_at_series.iat[i]
        m_norm = model_norm_series.iat[i]
        c_gb = cap_gb_series.iat[i]

        if p is None:
            continue

        text = str(df[model_col].iat[i])
        _rows_start_idx = len(rows)
        _method = "none"

        # 1. [NEW] model_cap_color_extractor_fn: (model, cap, color) -> 1 PN
        if model_cap_color_extractor_fn and color_map:
            extracted = model_cap_color_extractor_fn(text)
            if extracted:
                m, c, color_raw = extracted
                key = (m, int(c))
                cmap_for_key = color_map.get(key)
                matched = False
                if cmap_for_key:
                    for color_norm, (pn, cr) in cmap_for_key.items():
                        if _label_matches_color_unified(color_raw, cr, color_norm):
                            rows.append({
                                "part_number": str(pn),
                                "shop_name": shop_name,
                                "price_new": int(p),
                                "recorded_at": t,
                            })
                            matched = True
                            break
                if matched:
                    _method = "model_cap_color"
                    # 这里记录日志并跳过后续匹配
                    _logger.debug(
                        f"Row {i} summary",
                        extra={
                            "event_type": "row_processing_summary",
                            "log_seq": _log_seq,
                            "shop_name": shop_name,
                            "cleaner_name": cleaner_name,
                            "row_index": i,
                            "model_text": _truncate_for_log(text, 100),
                            "model_norm": m,
                            "capacity_gb": int(c),
                            "base_price": int(p),
                            "source_text_raw_full": (text or "None"),
                            "extraction_method": _method,
                            "output_records_count": 1,
                        }
                    )
                    _log_seq += 1
                    continue

        # 2. pn_extractor_fn
        if pn_extractor_fn:
            pn_direct = pn_extractor_fn(text)
            if pn_direct:
                rows.append({
                    "part_number": str(pn_direct),
                    "shop_name": shop_name,
                    "price_new": int(p),
                    "recorded_at": t,
                })
                _method = "pn_direct"
                _logger.debug(
                    f"Row {i} summary",
                    extra={
                        "event_type": "row_processing_summary",
                        "log_seq": _log_seq,
                        "shop_name": shop_name,
                        "cleaner_name": cleaner_name,
                        "row_index": i,
                        "model_text": _truncate_for_log(text, 100),
                        "model_norm": m_norm,
                        "capacity_gb": int(c_gb) if not pd.isna(c_gb) else None,
                        "base_price": int(p),
                        "source_text_raw_full": (text or "None"),
                        "extraction_method": _method,
                        "output_records_count": 1,
                    }
                )
                _log_seq += 1
                continue

        # 3. 原有逻辑: (m, c) -> groups.get -> 全色展开
        if not m_norm or pd.isna(c_gb):
            continue

        pn_list = groups.get((m_norm, int(c_gb)), [])
        if not pn_list:
            continue

        for pn in pn_list:
            rows.append({
                "part_number": str(pn),
                "shop_name": shop_name,
                "price_new": int(p),
                "recorded_at": t,
            })
            _method = "model_cap_expansion"

        # 记录行处理概览 (针对方法 3)
        _logger.debug(
            f"Row {i} summary",
            extra={
                "event_type": "row_processing_summary",
                "log_seq": _log_seq,
                "shop_name": shop_name,
                "cleaner_name": cleaner_name,
                "row_index": i,
                "model_text": _truncate_for_log(text, 100),
                "model_norm": m_norm,
                "capacity_gb": int(c_gb) if not pd.isna(c_gb) else None,
                "base_price": int(p),
                "source_text_raw_full": (text or "None"),
                "extraction_method": _method,
                "output_records_count": len(rows) - _rows_start_idx,
            }
        )
        _log_seq += 1

    out = assemble_output_df(rows, coerce_price=coerce_price)
    log_cleaner_complete(_logger, cleaner_name=cleaner_name, shop_name=shop_name,
                         input_rows=len(df), output_records=len(out), start_time=start_time)
    return out


# ======================================================================
# 6. C类清洗器 setup / finalize — 颜色感知+价格分解公共骨架
# ======================================================================

@dataclass
class ColorCleanerContext:
    """C类清洗器的公共上下文，由 setup_color_cleaner 创建。"""
    cleaner_name: str
    shop_name: str
    start_time: float
    log_seq: int
    input_rows: int
    info_df: pd.DataFrame
    color_map: Dict
    logger: logging.Logger
    extraction_mode: Optional[str] = None


def setup_color_cleaner(
    df: pd.DataFrame,
    *,
    cleaner_name: str,
    shop_name: str,
    required_cols: List[str],
    extraction_mode: Optional[str] = None,
    lenient: bool = False,
) -> Tuple[Optional[ColorCleanerContext], Optional[pd.DataFrame]]:
    """
    C类清洗器的公共初始化：日志、列校验、空检查、加载参考数据。

    Returns
    -------
    (ctx, None) : 初始化成功，ctx 包含 info_df / color_map / log_seq 等
    (None, empty_df) : df 为空或校验后为空，直接返回 empty_df 即可
    """
    _logger = logging.getLogger(f"cleaner_tools.{cleaner_name}")
    start_time = time.time()
    _log_seq = 0

    log_cleaner_start(_logger, cleaner_name=cleaner_name, shop_name=shop_name,
                      input_rows=len(df), log_seq=_log_seq,
                      extraction_mode=extraction_mode)
    _log_seq += 1

    _log_seq = validate_columns(df, required_cols,
                                cleaner_name=cleaner_name, shop_name=shop_name,
                                logger=_logger, log_seq=_log_seq, lenient=lenient)

    if df.empty:
        log_cleaner_complete(_logger, cleaner_name=cleaner_name, shop_name=shop_name,
                             input_rows=len(df), output_records=0,
                             start_time=start_time, log_seq=_log_seq)
        return None, pd.DataFrame(columns=_OUTPUT_COLUMNS)

    info_df = _load_iphone17_info_df_from_db()
    color_map = _build_color_map(info_df)

    ctx = ColorCleanerContext(
        cleaner_name=cleaner_name,
        shop_name=shop_name,
        start_time=start_time,
        log_seq=_log_seq,
        input_rows=len(df),
        info_df=info_df,
        color_map=color_map,
        logger=_logger,
        extraction_mode=extraction_mode,
    )
    return ctx, None


def finalize_color_cleaner(
    ctx: ColorCleanerContext,
    rows: List[dict],
    *,
    coerce_price: bool = False,
) -> pd.DataFrame:
    """
    C类清洗器的公共收尾：组装输出 DataFrame + 完成日志。
    """
    out = assemble_output_df(rows, coerce_price=coerce_price)
    log_cleaner_complete(ctx.logger, cleaner_name=ctx.cleaner_name, shop_name=ctx.shop_name,
                         input_rows=ctx.input_rows, output_records=len(out),
                         start_time=ctx.start_time, log_seq=ctx.log_seq)
    return out


# ======================================================================
# 4. 通用两阶段清洗 Pipeline 组件
# ======================================================================

def detect_all_delta_unified(text: str, regex: Pattern) -> Optional[int]:
    """
    通用全色减额检测。
    归一化文本 -> 正则匹配 -> 提取金额。
    """
    s = normalize_text_basic(text)
    if not s:
        return None
    m = regex.search(s)
    if m:
        return coerce_amount_yen(m.group(0).replace("全色", "").strip()) or 0
    if "全色" in s:
        return 0
    return None


def clean_text_generic(text: str) -> str:
    """
    通用文本预处理：
    - safe_to_text
    - 基础归一化 (全角转半角, 合并空格)
    注意：默认不移除换行（remove_newlines=False），保留结构供 split 使用。
    """
    if not text:
        return ""
    s = safe_to_text(text)
    if not s or s.lower() == "nan":
        return ""
    # 调用基础归一化，保留换行
    return normalize_text_basic(s, remove_newlines=False, collapse_spaces=False)


# 通用 DELTA LOOSE fallback：STRICT 未匹配时使用，允许日文/汉字/连字符等（与 shop2/3 原 LOOSE 一致）
DELTA_RE_LOOSE_GENERIC = re.compile(
    r"""(?P<label>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

# 通用 ABS LOOSE fallback：STRICT 未匹配时使用，label 允许日文/汉字/连字符等（与 DELTA 分开，便于独立修改）
ABS_RE_LOOSE_GENERIC = re.compile(
    r"""(?P<label>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)\s*[￥¥]\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE | re.VERBOSE,
)


def match_tokens_generic(
    text: str,
    split_re: Pattern,
    none_re: Pattern,
    abs_re: Pattern,
    delta_re: Pattern,
    normalize_label_func: Callable[[str], str],
    is_plausible_label_func: Callable[[str], bool],
    delta_re_loose: Optional[Pattern] = None,
    use_delta_loose_fallback: bool = True,
    abs_re_loose: Optional[Pattern] = None,
    use_abs_loose_fallback: bool = True,
    preprocessor: Callable[[str], str] = clean_text_generic,
) -> List[MatchToken]:
    """
    通用阶段 1 匹配器：
    1. 预处理文本
    2. 按 split_re 分割
    3. 对每个 part 依次尝试匹配 NONE / ABS / DELTA（均为 STRICT 优先，LOOSE 兜底）
    4. 处理 pending_labels

    delta_re_loose: 自定义 DELTA LOOSE 模式；为 None 且 use_delta_loose_fallback=True 时使用 DELTA_RE_LOOSE_GENERIC
    use_delta_loose_fallback: 是否启用 DELTA LOOSE fallback（默认 True）
    abs_re_loose: 自定义 ABS LOOSE 模式；为 None 且 use_abs_loose_fallback=True 时使用 ABS_RE_LOOSE_GENERIC
    use_abs_loose_fallback: 是否启用 ABS LOOSE fallback（默认 True）
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = preprocessor(text)
    if not s:
        return tokens

    # split
    parts = [p.strip() for p in split_re.split(s) if p and p.strip()]
    if not parts:
        parts = [s.strip()]

    pending_labels: List[str] = []
    position = 0

    def _try_delta_patterns(part: str, patterns: List[Pattern]) -> bool:
        nonlocal position
        for pat in patterns:
            found_any = False
            for m in pat.finditer(part):
                label_raw = normalize_label_func(m.group("label"))
                if not is_plausible_label_func(label_raw):
                    continue
                
                # 提取金额相关
                sep = m.group("sep") if "sep" in m.groupdict() else None
                sign = m.group("sign") if "sign" in m.groupdict() else None
                amt = to_int_yen(m.group("amount"))
                if amt is None:
                    continue
                
                amt_val = int(amt)
                
                # 格式判定
                if sign:
                    negative = sign in ("-", "−", "－")
                    amount_int = -amt_val if negative else amt_val
                    hint = FORMAT_HINT_SIGNED
                elif sep and sep in ("-", "−", "－"):
                    amount_int = -amt_val
                    hint = FORMAT_HINT_SEP_MINUS
                elif sep and sep in ("：", ":"):
                    amount_int = amt_val
                    hint = FORMAT_HINT_COLON_PREFIX
                else:
                    amount_int = amt_val
                    hint = FORMAT_HINT_PLAIN_DIGITS

                # 添加当前 label
                tokens.append(MatchToken(
                    label=label_raw,
                    amount_int=amount_int,
                    format_hint=hint,
                    position=position,
                ))
                position += 1
                
                # 处理 pending labels（仅首个 match 时应用，后续 match 时 pending 已清空）
                for pl in pending_labels:
                    pl_norm = normalize_label_func(pl)
                    if pl_norm and is_plausible_label_func(pl_norm):
                        tokens.append(MatchToken(
                            label=pl_norm,
                            amount_int=amount_int,
                            format_hint=hint,
                            position=position,
                        ))
                        position += 1
                pending_labels.clear()
                found_any = True
            if found_any:
                return True
        return False

    delta_patterns = [delta_re]
    if delta_re_loose is not None:
        delta_patterns.append(delta_re_loose)
    elif use_delta_loose_fallback:
        delta_patterns.append(DELTA_RE_LOOSE_GENERIC)

    abs_patterns: List[Pattern] = [abs_re]
    if abs_re_loose is not None:
        abs_patterns.append(abs_re_loose)
    elif use_abs_loose_fallback:
        abs_patterns.append(ABS_RE_LOOSE_GENERIC)

    def _try_abs_patterns(part: str) -> bool:
        nonlocal position
        for pat in abs_patterns:
            found_any = False
            for m in pat.finditer(part):
                label_raw = normalize_label_func(m.group("label"))
                if not is_plausible_label_func(label_raw):
                    continue
                amt = to_int_yen(m.group("amount"))
                if amt is None:
                    continue
                tokens.append(MatchToken(
                    label=label_raw,
                    amount_int=int(amt),
                    format_hint=FORMAT_HINT_AFTER_YEN,
                    position=position,
                ))
                position += 1
                pending_labels.clear()
                found_any = True
            if found_any:
                return True
        return False

    for part in parts:
        # 1. NONE
        m0 = none_re.search(part)
        if m0:
            label_raw = normalize_label_func(m0.group("label"))
            if is_plausible_label_func(label_raw):
                tokens.append(MatchToken(
                    label=label_raw,
                    amount_int=0,
                    format_hint=FORMAT_HINT_NONE,
                    position=position,
                ))
                position += 1
            pending_labels.clear()
            continue

        # 2. ABS（STRICT 优先，LOOSE 兜底）
        if _try_abs_patterns(part):
            continue

        # 3. DELTA
        if _try_delta_patterns(part, delta_patterns):
            continue

        # 4. Pending Label (如果本 part 既没匹配到任何价格模式，则视为 label 候选)
        sub_parts = [t.strip() for t in split_re.split(part) if t and t.strip()]
        if not sub_parts:
            sub_parts = [part.strip()]
            
        for tok in sub_parts:
            tok_norm = normalize_label_func(tok)
            if tok_norm:
                pending_labels.append(tok)

    return tokens
