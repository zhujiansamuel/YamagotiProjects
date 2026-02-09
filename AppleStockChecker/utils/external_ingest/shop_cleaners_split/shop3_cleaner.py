from __future__ import annotations

import os
import re
import time
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb

# --- LangExtract (LLM 抽取) ---
try:
    import langextract as lx
except Exception:
    lx = None

SHOP3_OLLAMA_URL = os.getenv("SHOP3_OLLAMA_URL", "http://localhost:11434")
SHOP3_OLLAMA_MODEL_ID = os.getenv("SHOP3_OLLAMA_MODEL_ID", "gemma3:1b")
SHOP3_USE_LANGEXTRACT = os.getenv("SHOP3_USE_LANGEXTRACT", "1").strip().lower() not in {"0", "false", "no", "off"}

_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)

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

    # 检测并标准化 jan 列
    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))

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

    # 数值类型：可能没有“显式符号”，按 default_sign 修正方向
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


def _price_from_shop3(x: object) -> Optional[int]:
    """
    data5 -> price_new
    - 预期形如 '¥177,000'；也兼容 '～177,000円'/'10.5万' 等；取区间最大值
    - 去除可能出现的修饰词（“新品/未開封”等）
    """
    if x is None:
        return None
    s = str(x)
    s = (s.replace("新品", "")
           .replace("新\u54c1", "")
           .replace("未開封", "")
           .replace("未开封", ""))
    return to_int_yen(s)
_SIGNED_AMOUNT_PAT = re.compile(r"([+\-−－])\s*([0-9０-９][0-9０-９,，]*)")

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
    若原文只出现“全为负”或“全为正”的带符号金额，返回其符号方向（-1 / +1）；混合则 None
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

FAMILY_SYNONYMS_shop3 = {
    "blue": ["ブルー", "青", "ディープブルー"],
    "ブルー": ["ブルー", "青", "ディープブルー"],
    "青": ["ブルー", "青", "ディープブルー"],
    "ディープブルー": ["ディープブルー", "ブルー", "青"],
    "silver": ["シルバー", "銀"],
    "シルバー": ["シルバー", "銀"],
    "銀": ["シルバー", "銀"],
}

_LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s；;]+")

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

_DELTA_PATTERN_STRICT = re.compile(
    r"""(?P<labels>[^+\-−－\d¥￥円]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE
)

_DELTA_PATTERN_LOOSE = re.compile(
    r"""(?P<labels>[\u3000\u30A0-\u30FF\u4E00-\u9FFF\w\-\s\/、，,・]+?)
        (?P<sign>[+\-−－])\s*
        (?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE
)


def _clean_label_token(tok: str) -> str:
    if tok is None:
        return ""
    t = str(tok).strip()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"（.*?）", "", t)
    return t.strip()


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


# -------------------- 1) 原正则版：保留作为 fallback --------------------
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


# -------------------- 2) LangExtract + Ollama：主用 --------------------
if lx is not None:
    _SHOP3_COLOR_DELTA_PROMPT = textwrap.dedent("""\
        你将看到一个很短的“备注/减价1”字符串，来源于日本二手回收价格表。
        目标：抽取“颜色标签 -> 价格差额(人民币/日元中的日元JPY)”的映射。
        输出要求（非常重要）：
        - 只抽取与“颜色差额/加价/减价”相关的信息；忽略与差额无关的描述（如 新品/未開封/SIMフリー 等）。
        - 每个颜色标签单独输出一条 Extraction。
        - extraction_class 固定为 "color_delta"。
        - extraction_text 必须是输入文本中的“原样子串”，只包含颜色标签本身（不要带分隔符、不要翻译、不要归一化）。
        - attributes 必须包含键 "delta_yen"，值为整数（可为字符串形式的整数也可），单位 JPY：
            * “-3000”“−3,000”“－３,０００”代表 -3000
            * “+2000”“＋２,０００”代表 +2000
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


def _parse_delta_int(x: object) -> Optional[int]:
    """
    支持 int / "+2000" / "−3,000" / "－３,０００" 等，返回 int。
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(round(x))

    s = str(x).strip().translate(_FZ_TO_HZ_TRANS).strip()
    if not s:
        return None

    sign = 1
    if s[0] == "-":
        sign = -1
        s = s[1:].strip()
    elif s[0] == "+":
        sign = 1
        s = s[1:].strip()

    amt = _normalize_amount_text(s)
    if amt is None:
        return None
    return int(sign * amt)


@lru_cache(maxsize=4096)
def _extract_color_deltas_shop3_llm_cached(text: str) -> Tuple[Tuple[str, int], ...]:
    s = (text or "").strip()
    if not s:
        return tuple()

    if not re.search(r"[0-9０-９+\-−－]", s):
        return tuple()

    if lx is None:
        return tuple()

    # ★ NEW: 原文全局 delta / 默认符号
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

        # ★ NEW: 若原文只有一种带符号金额，直接覆盖所有 label 的 delta
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


def _extract_color_deltas_shop3(text: str) -> List[Tuple[str, int]]:
    """
    主入口：优先使用 LangExtract+Ollama；失败/不可用则回退正则。
    """
    s = (text or "").strip()
    if not s:
        return []

    if not SHOP3_USE_LANGEXTRACT or lx is None:
        return _extract_color_deltas_shop3_regex(s)

    try:
        return list(_extract_color_deltas_shop3_llm_cached(s))
    except Exception:
        return _extract_color_deltas_shop3_regex(s)


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


def _build_color_map_shop3(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
    """
    (model_norm, cap_gb) -> { color_norm: (part_number, color_raw) }
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


def clean_shop3(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print("shop4:モバイルミックス---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"shop3:買取一丁目---------->进入清洗器时间: {now}")
    """
    debug=True 时：
      - 仅对“减价1”中可抽出颜色差额的行（deltas 非空）打印对照信息（最多 debug_limit 条）
      - 打印原文（title/data5/减价1）与抽取结果（deltas、每色最终价）
    """

    need_cols = ["title", "data5", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop3 清洗器缺少必要列：{c}")

    src = df.copy()
    mask_time_ok = src["time-scraped"].astype(str).str.strip().ne("") & src["time-scraped"].notna()
    src = src[mask_time_ok].reset_index(drop=True)
    if src.empty:
        return pd.DataFrame(columns=["part_number", "shop_name", "price_new", "recorded_at"])

    info_df = _load_iphone17_info_df_for_shop2()
    color_maps = _build_color_map_shop3(info_df)

    model_norm = src["title"].map(_normalize_model_generic)
    cap_gb     = src["title"].map(_parse_capacity_gb)

    try:
        base_price = src["data5"].map(_price_from_shop3)
    except Exception:
        base_price = src["data5"].map(to_int_yen)
    recorded_at = src["time-scraped"].map(parse_dt_aware)

    remark = src["减价1"] if "减价1" in src.columns else None

    # -------------------- 关键修改：用 LangExtract/Ollama 抽取 deltas --------------------
    if remark is not None:
        remark_text = remark.fillna("").astype(str)
        # 只对 unique 文本调用（内部还有 lru_cache），减少请求次数
        uniq = remark_text.unique().tolist()
        delta_map = {t: _extract_color_deltas_shop3(t) for t in uniq}
        deltas_series = remark_text.map(delta_map)
    else:
        deltas_series = pd.Series([[] for _ in range(len(src))])

    # -------------------- DEBUG：挑选需要打印的行 --------------------
    debug_pos_set: set[int] = set()
    if debug:
        hit_cnt = 0
        for i in range(len(src)):
            if deltas_series.iat[i]:
                debug_pos_set.add(i)
                hit_cnt += 1
                if hit_cnt >= int(debug_limit):
                    break

        if not debug_pos_set:
            debug_pos_set = set(range(min(int(debug_limit), len(src))))

        print(f"[shop3 debug] total_rows={len(src)}, rows_with_deltas={int((deltas_series.map(bool)).sum())}, print_rows={len(debug_pos_set)}")

    def _dbg_on(pos: int) -> bool:
        return bool(debug) and (pos in debug_pos_set)

    def _dprint(pos: int, *args, **kwargs):
        if _dbg_on(pos):
            print(*args, **kwargs)

    rows: List[dict] = []

    for i in range(len(src)):
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        p0 = base_price.iat[i]
        t  = recorded_at.iat[i]

        rem_text = str(remark.iat[i]) if remark is not None else ""
        deltas = deltas_series.iat[i]

        if _dbg_on(i):
            # print("\n[shop3 debug] row_pos=", i)
            # print("  title(raw):", repr(src["title"].iat[i]))
            # print("  data5(raw):", repr(src["data5"].iat[i]))
            print(f"price_raw       : {repr(rem_text)}<-----------------")
            # print("  model_norm:", repr(m))
            # print("  capacity_gb:", repr(c))
            print(f"deltas_extracted: {deltas}<-----------------")
            print(f"base_price      : {repr(p0)}")
            # print("  recorded_at:", repr(t))

        if not m:
            _dprint(i, "  SKIP_REASON: model_norm 为空（无法识别机型）")
            continue
        if pd.isna(c):
            _dprint(i, "  SKIP_REASON: capacity_gb 为空（无法识别容量）")
            continue
        if p0 is None:
            _dprint(i, "  SKIP_REASON: base_price 为空（data5 无法解析价格）")
            continue

        key = (m, int(c))
        cmap = color_maps.get(key)
        if not cmap:
            _dprint(i, f"  SKIP_REASON: info 表中找不到该机型+容量 key={key}")
            continue

        # if _dbg_on(i):
            # sample_colors = [(cn, v[0], v[1]) for cn, v in list(cmap.items())[:10]]
            # print("  match_key:", repr(key))
            # print("  cmap_sample(color_norm, pn, color_raw):", sample_colors, f"(len={len(cmap)})")

        per_color_abs: Dict[str, int] = {}
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

        for col_norm, (pn, col_raw) in cmap.items():
            if col_norm in per_color_abs:
                price_val = per_color_abs[col_norm]
                reason = "ABS"
                trace = {"abs": price_val}
            else:
                d = per_color_delta.get(col_norm, 0)
                price_val = int(p0) + int(d)
                reason = "BASE+DELTA" if d else "BASE"
                trace = per_color_trace.get(col_norm, None)

            if _dbg_on(i):
                print(f"OUT_ITEM        : color_raw: {str(col_raw):<10},base: {int(p0):<7},--->  delta: {int(per_color_delta.get(col_norm, 0)):<7} ,final: {int(price_val):<7},reason: {reason},match_trace: {trace}")

                # print("  -> OUT_ITEM:", {
                #     "part_number": str(pn),
                #     "color_raw": str(col_raw),
                #     "base": int(p0),
                #     "delta": int(per_color_delta.get(col_norm, 0)),
                #     "final": int(price_val),
                #     "reason": reason,
                #     "match_trace": trace,
                # })

            rows.append({
                "part_number": str(pn),
                "shop_name": "買取一丁目",
                "price_new": int(price_val),
                "recorded_at": t,
            })

        if _dbg_on(i) and deltas:
            used_labels = {v.get("matched_label") for v in per_color_trace.values()}
            unused = [lbl for (lbl, _d) in deltas if lbl not in used_labels]
            if unused:
                print("  note: 以下 label 未命中任何颜色（可能需要补同义词/匹配规则）:", unused)

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)

    # if debug:
    #     print(f"\n[shop3 debug] out_rows={len(out)} head=\n{out.head(10).to_string(index=False)}")

    return out
