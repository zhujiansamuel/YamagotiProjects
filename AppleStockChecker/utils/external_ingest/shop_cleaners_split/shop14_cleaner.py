from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _normalize_model_generic
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Tuple
from urllib.parse import urlparse
from typing import Dict, Optional, List, Iterable, Union
import os, re, json, pathlib
from datetime import datetime
import pytz
import time
import os, re, time, textwrap
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

SHOP14_DEBUG = str(True)
SHOP14_OLLAMA_URL = os.getenv("SHOP14_OLLAMA_URL", "http://localhost:11434")
SHOP14_LLM_MODEL_ID = os.getenv("SHOP14_LLM_MODEL_ID", "gemma3:1b")

# 统一用“家族->tokens”，同时包含英文（小写）+ 日文/汉字
_FAMILY_TOKENS = {
    "blue":   ["blue", "ブルー", "青"],
    "black":  ["black", "ブラック", "黒"],
    "white":  ["white", "ホワイト", "白"],
    "green":  ["green", "グリーン", "緑"],
    "red":    ["red", "レッド", "赤"],
    "pink":   ["pink", "ピンク"],
    "purple": ["purple", "パープル", "紫"],
    "yellow": ["yellow", "イエロー", "黄"],
    "orange": ["orange", "オレンジ", "橙"],
    "silver": ["silver", "シルバー", "銀"],
    "gold":   ["gold", "ゴールド", "金"],
    "gray":   ["gray", "grey", "グレー", "グレイ", "灰"],
    "natural":["natural", "ナチュラル"],
}

# token -> 同族 tokens 的反向索引（全部转 lower 做 case-insensitive key）
FAMILY_SYNONYMS_shop14: Dict[str, List[str]] = {}
for fam, toks in _FAMILY_TOKENS.items():
    for t in toks:
        FAMILY_SYNONYMS_shop14[str(t).lower()] = toks

def _label_matches_color_shop14(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    改进点：
    - 英文大小写不敏感（Blue/blue 都能匹配）
    - 同义词族包含英文 token：青 -> blue -> 命中 "Blue Titanium"
    """
    label_norm = _norm(label_raw)
    if not label_norm:
        return False

    color_raw_s = str(color_raw or "")
    color_norm_s = str(color_norm or "")

    # case-insensitive（对英文很关键）
    label_l = label_norm.lower()
    color_raw_l = color_raw_s.lower()
    color_norm_l = color_norm_s.lower()

    # 1) 归一化相等（不区分大小写）
    if label_l == color_norm_l:
        return True

    # 2) 子串（不区分大小写）
    if label_l and (label_l in color_raw_l):
        return True

    # 3) 同义族：取出该 label 的族 tokens（包含英文/日文/汉字），逐个在 color_raw 里做 case-insensitive 子串匹配
    candidates = set(FAMILY_SYNONYMS_shop14.get(label_l, []))

    # 3.2 若 label 不是字典 key（例如带空格），尝试用“族内词”反查
    if not candidates:
        for k, toks in FAMILY_SYNONYMS_shop14.items():
            # k 是 lower 的 token
            if k and (k == label_l or k in label_l):
                candidates.update(toks)
                break

    return any(str(tok).lower() in color_raw_l for tok in candidates)

def _is_truthy(v) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

def _dprint(enabled: bool, *args, **kwargs) -> None:
    if enabled:
        print(*args, **kwargs)

def _norm(s: str) -> str:
    return (s or "").strip()

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
        path = os.getenv("IPHONE17_INFO_CSV") or str(Path(__file__).resolve().parents[2] / "data" / "iphone17_info.csv")

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

def _has_all_colors(text: str) -> Optional[int]:
    """
    若文本含“全色”，且可选出现 '全色 ± 金額'，返回统一 delta；
    若仅出现 '全色' 无金额，返回 0；
    若未出现 '全色'，返回 None。
    """
    if not text:
        return None
    s = str(text)
    if "全色" not in s:
        return None
    # 试图解析 "全色 ± n円"
    m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*(\d[\d,]*)\s*円", s)
    if m:
        sign = m.group(1) or "+"
        amt = to_int_yen(m.group(2)) or 0
        if sign in ("-", "−", "－"):
            amt = -amt
        return int(amt)
    return 0

COLOR_DELTA_RE_shop14 = re.compile(
    r"""(?P<label>[^：:\-\s/、／]+)\s*
        (?P<sep>[：:\-])\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(円)?
    """,
    re.UNICODE | re.VERBOSE,
)

SPLIT_TOKENS_RE = re.compile(r"[／/、，,]|(?:\s+\+\s+)|(?:\s*;\s*)")

FAMILY_SYNONYMS_shop14 = {
    # blue family
    "blue": ["ブルー", "青"],
    "ブルー": ["ブルー", "青"],
    "青": ["ブルー", "青"],

    # black
    "black": ["ブラック", "黒"],
    "ブラック": ["ブラック", "黒"],
    "黒": ["ブラック", "黒"],

    # white
    "white": ["ホワイト", "白"],
    "ホワイト": ["ホワイト", "白"],
    "白": ["ホワイト", "白"],

    # green
    "green": ["グリーン", "緑"],
    "グリーン": ["グリーン", "緑"],
    "緑": ["グリーン", "緑"],

    # red
    "red": ["レッド", "赤"],
    "レッド": ["レッド", "赤"],
    "赤": ["レッド", "赤"],

    # pink
    "pink": ["ピンク"],
    "ピンク": ["ピンク"],

    # purple
    "purple": ["パープル", "紫"],
    "パープル": ["パープル", "紫"],
    "紫": ["パープル", "紫"],

    # yellow
    "yellow": ["イエロー", "黄"],
    "イエロー": ["イエロー", "黄"],
    "黄": ["イエロー", "黄"],

    # orange / silver / gold / gray / natural
    "orange": ["オレンジ", "橙"],
    "オレンジ": ["オレンジ", "橙"],
    "橙": ["オレンジ", "橙"],

    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],

    "gold": ["ゴールド", "金"],
    "ゴールド": ["ゴールド", "金"],
    "金": ["ゴールド", "金"],

    "gray": ["グレー", "グレイ", "灰"],
    "グレー": ["グレー", "グレイ", "灰"],
    "グレイ": ["グレー", "グレイ", "灰"],
    "灰": ["グレー", "グレイ", "灰"],

    "natural": ["ナチュラル"],
    "ナチュラル": ["ナチュラル"],
}

_SPLIT_TOKENS_SAFE_RE = re.compile(
    r"""
    [／/、，]                 # 全角/斜杠类分隔符（始终切分）
    |(?<!\d),(?!\d)          # ASCII 逗号：仅当其两侧不是数字时切分（避免拆千位分隔）
    |(?:\s+\+\s+)            # " + " 形式
    |(?:\s*;\s*)             # 分号
    """,
    re.UNICODE | re.VERBOSE,
)

_COLOR_ABS_PRICE_RE = re.compile(
    r"""^\s*
        (?P<label>[^：:\-\s/、／¥円]+?)    # 颜色标签（非贪心，避免包含金额）
        \s*(?:[:：]?\s*)                   # 可选分隔符
        (?:¥|￥)?\s*                       # 可选货币符号
        (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})*|\d+)  # 支持千位逗号（ASCII or fullwidth）或无逗号数字
        \s*(?:円)?\s*$
    """,
    re.UNICODE | re.VERBOSE,
)

def _extract_color_deltas_shop14(text: str, *, debug: bool = True, ctx: str = "") -> List[Tuple[str, int]]:
    """
    从 '减价条件2' 提取若干 (label_raw, delta_int)。
    允许多组，使用 '/', '／', '、', ',', '，', ';' 等分隔。
    例：
      '青-3000'          -> [('青', -3000)]
      '橙/銀+1000'       -> [('橙', +1000), ('銀', +1000)]
      'ブルー：-2,000円' -> [('ブルー', -2000)]
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text)
    _dprint(debug, f"{ctx} [delta] raw='{s}'")

    # 用安全切分，避免把 2,000 拆开
    parts = [p.strip() for p in _SPLIT_TOKENS_SAFE_RE.split(s) if p and p.strip()]
    if not parts:
        parts = [s]

    _dprint(debug, f"{ctx} [delta] split_parts={parts}")

    for part in parts:
        m = COLOR_DELTA_RE_shop14.search(part)
        if not m:
            _dprint(debug, f"{ctx} [delta] no_match part='{part}'")
            continue

        label = (m.group("label") or "").strip()
        sep = m.group("sep")
        sign = m.group("sign")
        amt = to_int_yen(m.group("amount"))

        _dprint(
            debug,
            f"{ctx} [delta] matched part='{part}' -> label='{label}', sep='{sep}', sign='{sign}', amount_raw='{m.group('amount')}', amt={amt}"
        )

        if amt is None:
            continue

        if sign:
            negative = sign in ("-", "−", "－")
        else:
            negative = sep in ("-", "−", "－")

        delta = -int(amt) if negative else int(amt)
        out.append((label, delta))
        _dprint(debug, f"{ctx} [delta] parsed -> ({label!r}, {delta})")

    _dprint(debug, f"{ctx} [delta] out={out}")
    return out

def _label_matches_color_shop14(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    宽松匹配 label 是否命中颜色：
    1) 归一化精确相等
    2) label_raw 是 color_raw 的子串
    3) 同义族：label 无论是英文还是日文（如 “blue”“ブルー”“青”“銀”“橙”），
       都先取出该族的“日文关键词集合”，只要其中任意一个出现在 color_raw 中即命中。
    """
    label_norm = _norm(label_raw)

    # 1) 精确相等（归一化后）
    if label_norm == color_norm:
        return True

    # 2) 原文子串
    if label_raw and str(label_raw) in str(color_raw):
        return True

    # 3) 同义族匹配（正向键 + 反向值）
    # 3.1 直接以 label_raw/label_norm 作为键
    keys = {label_raw.strip().lower(), label_norm, label_raw.strip()}
    candidates = set()
    for k in keys:
        if k in FAMILY_SYNONYMS_shop14:
            candidates.update(FAMILY_SYNONYMS_shop14[k])

    # 3.2 若还没命中，将 label 当作“族内词”去反查家族，再收集该家族的全部关键词
    if not candidates:
        for fam, tokens in FAMILY_SYNONYMS_shop14.items():
            if any((t == label_raw) or (t == label_norm) or (t in str(label_raw)) for t in tokens):
                candidates.update(tokens)
                break

    # 家族里的任一关键词是 color_raw 的子串即可
    return any(tok in str(color_raw) for tok in candidates)

def _build_color_map_shop14(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """
    构建 (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }
    """
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

def _norm_label(lbl: str) -> str:
    """去除空白并统一全角空格/NBSP，保留原文字顺序用作匹配用 key"""
    if lbl is None:
        return ""
    s = str(lbl)
    # 去掉左右空白并规范全角空格为半角
    s = s.strip().replace("\u3000", " ").replace("\xa0", " ").strip()
    # 把中间多空格合并
    s = re.sub(r"\s+", " ", s)
    return s

def _clean_remark_frag(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.lstrip("\ufeff").replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _split_labels(labels: str) -> List[str]:
    """
    把 "青/銀" / "青、銀" / "青 銀" / "青,銀" 之类拆成 ["青","銀"]
    """
    s = str(labels or "").strip()
    if not s:
        return []
    # 统一常见分隔符
    parts = re.split(r"[／/、，,;；\s]+", s)
    return [p.strip() for p in parts if p and p.strip()]

def _coerce_amount_yen(v) -> Optional[int]:
    """
    把 amount_yen 从 LLM attributes / 文本里转成 int（支持：-2500、-2,500円、229,500 等）
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

    # 先拆符号（兼容全角减号）
    sign = 1
    if s[:1] in {"+", "＋"}:
        s = s[1:].strip()
    elif s[:1] in {"-", "−", "－"}:
        sign = -1
        s = s[1:].strip()

    # to_int_yen 通常能处理 "229,500" / "2,500円" / "¥229,500"
    n = to_int_yen(s)
    if n is None:
        # 兜底：只留数字再转
        s2 = re.sub(r"[^\d]", "", s)
        if not s2:
            return None
        try:
            n = int(s2)
        except Exception:
            return None

    return sign * int(n)

PAIR_RE_MULTI = re.compile(
    r"([^\d¥円,，＋+－\-−\s]+)\s*([+\-−－]?\s*\d[\d,，]*)"
)

def _split_color_amount_pairs_multi(txt: str) -> List[Tuple[str, int]]:
    """
    在单个文本里拆成多个 (label, amount) 对。
    仅当检测到 >=2 个“颜色+金额”对时才返回非空，用于解决：
      '橙227000、青228000' 被 LLM 当成一个 abs_group 的情况。
    例子：
      '橙227000、青228000'
      'コズミックオレンジ227000、ディープブルー228000'
      '青229500/銀228000'
      '橙 -2500、青 +3000'
    """
    out: List[Tuple[str, int]] = []
    if not txt:
        return out
    s = str(txt)

    for label, amt_s in PAIR_RE_MULTI.findall(s):
        # 去掉分隔符前缀：顿号/斜杠/逗号/分号等
        label = label.lstrip("、/,／，,;；").strip()
        if not label:
            continue
        amt = _coerce_amount_yen(amt_s)
        if amt is None:
            continue
        out.append((label, amt))

    # 只有发现了 2 个及以上 pair，我们才认为这是“多颜色规则”
    if len(out) >= 2:
        return out
    return []

def _labels_from_text_fallback(extraction_text: str) -> str:
    """
    当 LLM 没给 label/labels 时，用 extraction_text 兜底把金额去掉，剩下的当 labels。
    """
    t = str(extraction_text or "")
    # 去掉“全色”
    t = t.replace("全色", "")
    # 去掉金额段（可能带符号/¥/円/逗号）
    t = re.sub(r"(?:[+\-−－])?\s*(?:¥|￥)?\s*\d[\d,，]*\s*(?:円)?", "", t)
    t = t.strip()
    return t

def _extract_color_abs_prices(text: str, *, debug: bool = True, ctx: str = "") -> List[Tuple[str, int]]:
    """
    从 text 中抽取 (label_raw, abs_price) 绝对价。
    修复点：不会在数字千位分隔符处拆分（保留 229,000 完整）。
    支持多标签共用金额：'青/銀327000'、'青 銀 327000' 等。
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    pending_labels: List[str] = []

    s_all = str(text).strip()
    if s_all.lower() == "nan" or s_all == "":
        return out

    _dprint(debug, f"{ctx} [abs] raw='{s_all}'")

    parts = [p.strip() for p in _SPLIT_TOKENS_SAFE_RE.split(s_all) if p and p.strip()]
    if not parts:
        parts = [s_all]

    _dprint(debug, f"{ctx} [abs] split_parts={parts}")

    for part in parts:
        # 如果片段里同时含有 + 或 - （显式差额），跳过（差额解析会处理）
        if any(ch in part for ch in ("+", "-", "−", "－")):
            _dprint(debug, f"{ctx} [abs] skip(delta_like) part='{part}'")
            continue

        m = _COLOR_ABS_PRICE_RE.search(part)
        if m:
            label_raw = _norm_label(m.group("label"))
            amt_txt = m.group("amount")
            amt_clean = re.sub(r"[,\uFF0C]", "", amt_txt)

            _dprint(debug, f"{ctx} [abs] matched part='{part}' -> label='{label_raw}', amount_raw='{amt_txt}', amount_clean='{amt_clean}'")

            try:
                amt_val = int(amt_clean)
            except Exception:
                try:
                    amt_val = int(to_int_yen(amt_txt) or 0)
                except Exception:
                    _dprint(debug, f"{ctx} [abs] amount_parse_failed amount_raw='{amt_txt}'")
                    continue

            if label_raw:
                out.append((label_raw, amt_val))
                if pending_labels:
                    _dprint(debug, f"{ctx} [abs] apply pending_labels={pending_labels} with amt={amt_val}")
                for pl in pending_labels:
                    pln = _norm_label(pl)
                    if pln:
                        out.append((pln, amt_val))
                pending_labels = []
            continue

        # 没有找到金额：可能只是标签
        toks = [t for t in re.split(r"[／/、，;；,]", part) if t and t.strip()]
        for tok in toks:
            tok = _norm_label(tok)
            if tok:
                pending_labels.append(tok)
        _dprint(debug, f"{ctx} [abs] no_amount part='{part}', pending_labels={pending_labels}")

    _dprint(debug, f"{ctx} [abs] out={out}")
    return out

def _norm_colname(x) -> str:
    s = str(x or "")
    s = s.lstrip("\ufeff")                 # 去 BOM
    s = s.replace("\u3000", " ")           # 全角空格
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _resolve_remark_cols(df: "pd.DataFrame") -> Dict[str, Optional[str]]:
    """
    返回：{"减价条件": 实际列名或None, "减价条件2":..., "23432":...}
    - 先精确匹配（去 BOM/空格）
    - 再包含匹配（fuzzy）
    """
    want = ["减价条件", "减价条件2", "23432"]
    norm_map = {_norm_colname(c): c for c in df.columns}

    resolved: Dict[str, Optional[str]] = {w: None for w in want}
    for w in want:
        nw = _norm_colname(w)
        if nw in norm_map:
            resolved[w] = norm_map[nw]
            continue
        # fuzzy: 目标字符串出现在实际列名里
        for nc, ac in norm_map.items():
            if nw and (nw in nc):
                resolved[w] = ac
                break
    return resolved

@lru_cache(maxsize=1)
def _shop14_lx_prompt_and_examples():
    """
    LangExtract 的 prompt + few-shot examples（只初始化一次）。
    """
    import langextract as lx

    prompt = textwrap.dedent(
        """\
        你是信息抽取系统。请从输入文本中抽取“按颜色的价格规则（円）”。

        规则类型只有三类：
        1) all_colors：文本出现“全色”，可选跟金额（例如“全色 -3000”“全色 3000円”）。
           表示所有颜色统一调整：final = base + amount_yen。若没写金额，amount_yen=0。
        2) abs_group：颜色标签(一个或多个)后面出现一个金额（例如“青 229,500”“青/銀 229500円”）。
           表示这些颜色的最终价格等于该绝对金额。
        3) delta_group：颜色标签(一个或多个)后面出现带正负号的金额（例如“橙 -2500”“銀+1000”）。
           表示这些颜色在基准价上加上差价（可为负）。

        分隔符可能是空格、换行、“/”“／”“、”“,”“;”等。多个颜色可能共用同一个金额（例如“青/銀 229,500”），
        这种情况请把 attributes.labels 写成 "青/銀"（原样即可）。

        输出要求（非常重要）：
        - Use exact text for extraction_text（必须是原文连续子串，不要改写）。
        - 只抽取原文明确出现的规则，不要推断/补全。
        - attributes.amount_yen 必须是纯整数（去掉逗号/円/¥），差价允许负数。
        - attributes.labels：颜色标签，字符串（单色就写单个；多色就用原文分隔符，如 “青/銀”）。
        """
    )

    examples = [
        lx.data.ExampleData(
            text="青 229,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青 229,500",
                    attributes={"labels": "青", "amount_yen": "229500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="橙 -2500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙 -2500",
                    attributes={"labels": "橙", "amount_yen": "-2500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="全色 -3,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="all_colors",
                    extraction_text="全色 -3,000円",
                    attributes={"amount_yen": "-3000"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="青/銀 229,500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青/銀 229,500",
                    attributes={"labels": "青/銀", "amount_yen": "229500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="橙/銀 -2,500円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙/銀 -2,500円",
                    attributes={"labels": "橙/銀", "amount_yen": "-2500"},
                )
            ],
        ),
        lx.data.ExampleData(
            text="青 229,500\n橙 -2500",
            extractions=[
                lx.data.Extraction(
                    extraction_class="abs_group",
                    extraction_text="青 229,500",
                    attributes={"labels": "青", "amount_yen": "229500"},
                ),
                lx.data.Extraction(
                    extraction_class="delta_group",
                    extraction_text="橙 -2500",
                    attributes={"labels": "橙", "amount_yen": "-2500"},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="全色",
            extractions=[
                lx.data.Extraction(
                    extraction_class="all_colors",
                    extraction_text="全色",
                    attributes={"amount_yen": "0"},
                )
            ],
        ),
    ]

    return prompt, examples

_PAIR_GROUP_RE_shop14 = re.compile(
    r"""
    (?P<labels>[^\d¥￥円:+\-−－＋]+?)      # labels（允许包含 /、空格等，但不含数字/符号）
    \s*(?:[:：]\s*)?                     # 可选冒号
    (?P<sign>[+\-−－＋])?                 # 可选正负号（有则为 delta）
    \s*(?:¥|￥)?\s*
    (?P<amount>\d{1,3}(?:[,\uFF0C]\d{3})+|\d+)  # 金额：带千分位 或 纯数字
    \s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

def _strip_label_delims(s: str) -> str:
    s = str(s or "").strip()
    # 去掉匹配过程中可能带进来的前导分隔符（例如 "、青"）
    s = re.sub(r"^[／/、，,;；\s]+", "", s)
    s = re.sub(r"[／/、，,;；\s]+$", "", s)
    return s.strip()

def _repair_abs_delta_from_compound_text(text: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    从类似 '橙227000、青228000' / '青/銀229000、橙227000' / '橙-2500、青+1000'
    这种复合串中恢复为:
      abs=[('橙',227000), ('青',228000)] / delta=[...]
    """
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []
    s = str(text or "")
    if not s:
        return abs_list, delta_list

    for m in _PAIR_GROUP_RE_shop14.finditer(s):
        labels_raw = _strip_label_delims(m.group("labels"))
        sign = m.group("sign")
        amount_raw = m.group("amount")

        # 金额清洗
        amt_clean = re.sub(r"[,\uFF0C]", "", str(amount_raw))
        try:
            amt = int(amt_clean)
        except Exception:
            continue

        labels = _split_labels(labels_raw)  # 你已有：支持 /、空格、顿号等
        if not labels:
            continue

        if sign and sign in {"-", "−", "－"}:
            for lb in labels:
                delta_list.append((lb, -amt))
        elif sign and sign in {"+", "＋"}:
            for lb in labels:
                delta_list.append((lb, amt))
        else:
            for lb in labels:
                abs_list.append((lb, amt))

    return abs_list, delta_list

@lru_cache(maxsize=4096)
def _shop14_extract_rules_with_langextract(
    text: str,
) -> Dict[str, Union[Optional[int], List[Tuple[str, int]], List[dict]]]:
    """
    用 LangExtract(Ollama) 抽取规则。
    返回：
      {
        "all_delta": Optional[int],
        "abs":   List[(label_raw, abs_price)],
        "delta": List[(label_raw, delta)],
        "raw":   List[{"class","text","attributes"}],
      }
    """
    out = {"all_delta": None, "abs": [], "delta": [], "raw": []}
    s = _clean_remark_frag(text)
    if not s:
        return out

    import langextract as lx

    prompt, examples = _shop14_lx_prompt_and_examples()

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=SHOP14_LLM_MODEL_ID,
            model_url=SHOP14_OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except TypeError:
        # 兼容旧版 API
        result = lx.extract(
            text_or_documents=s,
            prompt_description=prompt,
            examples=examples,
            model_id=SHOP14_LLM_MODEL_ID,
            model_url=SHOP14_OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

    all_delta: Optional[int] = None
    abs_list: List[Tuple[str, int]] = []
    delta_list: List[Tuple[str, int]] = []

    for e in (getattr(result, "extractions", None) or []):
        cls = str(getattr(e, "extraction_class", "") or "").strip()
        txt = str(getattr(e, "extraction_text", "") or "")
        attrs = getattr(e, "attributes", {}) or {}

        out["raw"].append({"class": cls, "text": txt, "attributes": attrs})

        cls_l = cls.lower().strip()

        # ---------- 0) 先看能不能在 text 里拆出多组 “颜色+金额” ----------
        multi_pairs = _split_color_amount_pairs_multi(txt)
        if multi_pairs:
            # 推断这是 abs 还是 delta（全部金额都很大 -> abs；都很小 -> delta）
            vals_abs = [abs(v) for _, v in multi_pairs]
            kind: Optional[str] = None
            if "abs" in cls_l:
                kind = "abs"
            elif "delta" in cls_l or "diff" in cls_l:
                kind = "delta"
            else:
                if all(v >= 20000 for v in vals_abs):
                    kind = "abs"
                elif all(v <= 20000 for v in vals_abs):
                    kind = "delta"
                else:
                    big = sum(1 for v in vals_abs if v >= 20000)
                    kind = "abs" if big >= len(vals_abs) / 2.0 else "delta"

            for label, amt in multi_pairs:
                if kind == "abs":
                    abs_list.append((label, abs(int(amt))))
                else:
                    delta_list.append((label, int(amt)))

            if SHOP14_DEBUG:
                _dprint(
                    True,
                    f"[LangExtract-multi] txt='{txt}' -> kind={kind}, pairs={multi_pairs}",
                )
            # 已处理完这一条 extraction，继续下一条
            continue

        # ---------- 1) 全色 ----------
        amount = None
        if isinstance(attrs, dict):
            amount = _coerce_amount_yen(attrs.get("amount_yen")) or _coerce_amount_yen(
                attrs.get("amount")
            )
        if amount is None:
            amount = _coerce_amount_yen(txt)

        if ("all" in cls_l) or ("全色" in txt):
            all_delta = int(amount) if amount is not None else 0
            continue

        # ---------- 2) 普通 abs/delta ----------
        labels_str = ""
        if isinstance(attrs, dict):
            labels_str = str(attrs.get("labels") or attrs.get("label") or "").strip()
        if not labels_str:
            labels_str = _labels_from_text_fallback(txt)

        labels = _split_labels(labels_str)

        kind: Optional[str] = None
        if "abs" in cls_l:
            kind = "abs"
        elif "delta" in cls_l or "diff" in cls_l:
            kind = "delta"
        else:
            if amount is not None and abs(int(amount)) >= 20000:
                kind = "abs"
            elif amount is not None:
                kind = "delta"

        if not kind or amount is None or not labels:
            continue

        if kind == "abs":
            v = abs(int(amount))
            for lb in labels:
                abs_list.append((lb, v))
        else:
            v = int(amount)
            for lb in labels:
                delta_list.append((lb, v))

    out["all_delta"] = all_delta
    out["abs"] = abs_list
    out["delta"] = delta_list
    return out

def clean_shop14(df: "pd.DataFrame") -> "pd.DataFrame":
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"shop14:買取楽園---------->进入清洗器时间: {now}")

    # 必要列：remark 的三列允许缺失（缺失则当空），其余保持强校验
    for c in ["name", "data6", "price2", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop14 清洗器缺少必要列：{c}")

    # 这三个是你要求“同时监看”的列（存在则用，不存在也不报错）
    remark_cols = ["减价条件", "减价条件2", "23432"]
    remark_cols_map = _resolve_remark_cols(df)  # 关键：列名解析

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop14(info_df)

    rows: List[dict] = []

    for idx, row in df.iterrows():
        status = str(row.get("data6") or "")
        if "未開封" not in status:
            continue

        model_text = str(row.get("name") or "").strip()
        if not model_text:
            continue

        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None:
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)
        if not color_map:
            continue

        base_price = to_int_yen(row.get("price2"))
        if base_price is None:
            continue
        base_price = int(base_price)

        rec_at = parse_dt_aware(row.get("time-scraped"))

        # ===== 同时读取 3 列
        frags = {}
        for logical in ("减价条件", "减价条件2", "23432"):
            actual = remark_cols_map.get(logical)  # 实际列名 or None
            raw_val = row.get(actual) if actual else None
            frags[logical] = _clean_remark_frag(raw_val)

        combined = " ".join([v for v in frags.values() if v]).strip()

        # ===== 用 LangExtract(Ollama) 分别抽取，便于对照打印
        agg_all_delta: Optional[int] = None
        agg_abs: List[Tuple[str, int]] = []
        agg_delta: List[Tuple[str, int]] = []
        per_col_debug = {}

        for col, frag in frags.items():
            if not frag:
                continue
            parsed = _shop14_extract_rules_with_langextract(frag)
            per_col_debug[col] = parsed

            if parsed.get("all_delta") is not None:
                # 多列同时出现全色：以“后出现的列”为准（你也可改成 sum）
                agg_all_delta = int(parsed["all_delta"])

            agg_abs.extend(parsed.get("abs") or [])
            agg_delta.extend(parsed.get("delta") or [])

        # 兜底：如果逐列都没抽到，但合并串有内容，再跑一次合并串
        if combined and (agg_all_delta is None) and (not agg_abs) and (not agg_delta):
            parsed2 = _shop14_extract_rules_with_langextract(combined)
            per_col_debug["<combined>"] = parsed2
            if parsed2.get("all_delta") is not None:
                agg_all_delta = int(parsed2["all_delta"])
            agg_abs.extend(parsed2.get("abs") or [])
            agg_delta.extend(parsed2.get("delta") or [])

        # ====== print：对照 原始文字 vs 抽取结果
        if SHOP14_DEBUG and any(v for v in frags.values()):
            print(f"[{idx}] model='{model_text}' -> norm='{model_norm}', cap={cap_gb}, base_price={base_price}")
            for c in remark_cols:
                if c in df.columns:
                    print(f"[{idx}]        RAW {c:<5} = '{row.get(c):<20}'   | CLEAN = '{frags.get(c, '')}'")
            # for c, parsed in per_col_debug.items():
            #     print(f"[{idx}] LangExtract({c}) raw_extractions:")
            #     for ex in (parsed.get("raw") or []):
            #         print(f"      - class = {ex.get('class')} text = '{ex.get('text')}' attrs = {ex.get('attributes')}")
            #     print(f"[{idx}] LangExtract({c}) parsed -> all_delta={parsed.get('all_delta')}, abs={parsed.get('abs')}, delta={parsed.get('delta')}")

            # print(f"[{idx}] AGG -> all_delta={agg_all_delta}, abs={agg_abs}, delta={agg_delta}")
        # ====== 全色优先（保持你原逻辑）
        if agg_all_delta is not None:
            final_price = base_price + int(agg_all_delta)
            if SHOP14_DEBUG:
                print(f"[{idx}] 全色调整 detected: all_delta={agg_all_delta}, final_price={final_price}")

            for _col_norm, (pn, _raw) in color_map.items():
                rows.append(
                    {
                        "part_number": pn,
                        "shop_name": "買取楽園",
                        "price_new": int(final_price),
                        "recorded_at": rec_at,
                    }
                )
            continue

        # ====== 把 label -> 具体颜色（用你现有的 _label_matches_color_shop14）
        color_abs: Dict[str, int] = {}
        color_deltas: Dict[str, int] = {}

        if agg_abs:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, abs_price in agg_abs:
                    if _label_matches_color_shop14(label_raw, col_raw, col_norm):
                        color_abs[col_norm] = int(abs_price)
                        if SHOP14_DEBUG:
                            print(f"[{idx}] abs match -> label='{label_raw:<10}' hits color_raw='{col_raw:<10}' abs={abs_price}")

        if agg_delta:
            for col_norm, (pn, col_raw) in color_map.items():
                for label_raw, delta in agg_delta:
                    if _label_matches_color_shop14(label_raw, col_raw, col_norm):
                        color_deltas[col_norm] = int(delta)
                        if SHOP14_DEBUG:
                            print(f"[{idx}] delta match -> label='{label_raw:<10}' hits color_raw='{col_raw:<10}' delta={delta}")

        # ====== 生成各色价格：绝对价优先 > base+delta > base
        for col_norm, (pn, col_raw) in color_map.items():
            if col_norm in color_abs:
                price_val = int(color_abs[col_norm])
                reason = "abs"
            else:
                d = int(color_deltas.get(col_norm, 0))
                price_val = int(base_price + d)
                reason = f"base+delta({d})" if col_norm in color_deltas else "base"

            if SHOP14_DEBUG:
                print(f"[{idx}] ------> color='{col_raw:<10}' pn={pn} price={price_val:<7} reason={reason}")

            rows.append(
                {
                    "part_number": pn,
                    "shop_name": "買取楽園",
                    "price_new": price_val,
                    "recorded_at": rec_at,
                }
            )
        print("-----------------")

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out
