from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional,List
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb
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
from dateutil import parser as dateparser
import textwrap
from functools import lru_cache

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

    # ❗ 在“数字后立即跟英文”的位置补一个空格：17pro -> 17 pro
    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)

    # 标准化大小写/形态：pro-max / ProMax / promáx → Pro Max；pro → Pro；plus → Plus；mini → mini
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
        # 当前返回主体 'iPhone Air'；若以后真有 Air Plus 等可在此扩展
        return "iPhone Air"

    return ""

    t = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*TB", t, flags=re.I)
    if m:
        return int(round(float(m.group(1)) * 1024))
    m = re.search(r"(\d{2,4})\s*GB", t, flags=re.I)
    if m:
        return int(m.group(1))
    return None

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

def to_int_yen_shop11(v) -> Optional[int]:
    """
    将各种形式的日元表示解析为 int（日元），若无法解析返回 None。
    支持样例：
      "1,000" "1,000円" "¥1,000" "１，０００" "1000" 以及带空格的混合形式
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # 去掉括号内的备注
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()

    # 找到第一个数字段（允许全角数字与逗号/点），并取其附近
    # 首先把全角数字/逗号/点转换为半角
    # 半角映射（数字/标点）
    trans_map = str.maketrans({
        '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
        '，':',','．':'.','－':'-','＋':'+','¥':'','￥':''
    })
    s2 = s.translate(trans_map)

    # 移除非数字/逗号/点/+-/空格/円符号
    # 但先尝试用正则抓取像 -?¥?1,000 或 1,000 的金额
    m = re.search(r"([+\-−－]?)\s*(?:¥|￥)?\s*([\d][\d,]*)", s2)
    if not m:
        # 备用：从任何位置提取数字串
        m2 = re.search(r"([\d][\d,]*)", s2)
        if not m2:
            return None
        amt_txt = m2.group(1)
        sign = ""
    else:
        sign = m.group(1) or ""
        amt_txt = m.group(2)

    # 去逗号并转 int
    amt_digits = re.sub(r"[^\d]", "", amt_txt or "")
    if not amt_digits:
        return None
    try:
        val = int(amt_digits)
    except Exception:
        return None

    if sign in ("-", "−", "－"):
        val = -val
    return val

_FZ_TO_HZ_TRANS = str.maketrans({
    '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9',
    '，':',','．':'.','：':':','（':'(','）':')','　':' ','－':'-','＋':'+','¥':'','￥':''
})

def _normalize_number_text(txt: str) -> str:
    if txt is None:
        return ""
    return str(txt).translate(_FZ_TO_HZ_TRANS).strip()

_COLOR_SEP_SPLIT_RE = re.compile(r"[／/、，,・\s]+")  # split labels by these

_COLOR_GROUP_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # 最多 80 char 的 label group（不会以数字或 +/- 开头）
    [：:]\s*                               # 必须有 冒号（：或:）作为分隔（这是最常见的情形）
    (?P<sign>[+\-−－]?)\s*                 # 可选 +/-
    (?P<amount>[\d０-９,，]+)              # 金额（含全角数字与逗号）
    (?:\s*円|\s*¥|\s*￥)?                  # 可选货币符
    """,
    re.UNICODE | re.VERBOSE
)

_COLOR_GROUP_FALLBACK_RE = re.compile(
    r"""
    (?P<labels>[^+\-−－\d¥￥円()]{1,80}?)   # label group（保守）
    [\s]*?(?P<sign>[+\-−－])\s*
    (?P<amount>[\d０-９,，]+)
    (?:\s*円|\s*¥|\s*￥)?
    """,
    re.UNICODE | re.VERBOSE
)

def _extract_color_deltas_shop11(text: str) -> List[Tuple[str, int]]:
    """
    更鲁棒的颜色差额解析，返回 [(label_raw, delta_int), ...]
    支持：
      - "シルバー・ブルー：-1,000円(未開封)" -> ('シルバー', -1000), ('ブルー', -1000)
      - "ブルー、ブラック：-2,000円(未開封)"
      - "銀206000,青205500" 不在此函数处理（若需要可在外面加入绝对价解析）
    """
    out: List[Tuple[str, int]] = []
    if not text:
        return out

    s = str(text).strip()
    # 去掉括号内备注 (未開封) 等
    s = re.sub(r"\（.*?\）|\(.*?\)", "", s).strip()
    if not s:
        return out

    s_norm = _normalize_number_text(s)

    # 1) 主匹配：labelGroup：+/-?amount
    for m in _COLOR_GROUP_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = _normalize_number_text(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        # 把 labels 按常见分隔符拆成多个 label
        for lbl in _COLOR_SEP_SPLIT_RE.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 2) 回退匹配（如 "ブルー -4000"）
    for m in _COLOR_GROUP_FALLBACK_RE.finditer(s_norm):
        labels = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt_txt = m.group("amount") or ""
        amt_txt = _normalize_number_text(amt_txt)
        amt_digits = re.sub(r"[^\d]", "", amt_txt)
        if not amt_digits:
            continue
        amt = int(amt_digits)
        if sign in ("-", "−", "－"):
            amt = -amt
        for lbl in _COLOR_SEP_SPLIT_RE.split(labels):
            lbl = lbl.strip()
            if lbl:
                out.append((lbl, int(amt)))

    # 去重：若同 label 多次解析到，以最后一个为准（保留最后出现的 delta）
    if out:
        tmp: Dict[str, int] = {}
        for lbl, d in out:
            tmp[lbl] = d
        return list(tmp.items())

    return out

def _label_matches_color_shop11(label_raw: str, color_raw: str, color_norm: str) -> bool:
    """
    匹配策略（宽容）：
      - 归一化后（去空白、半角/全角数字转换）相等；
      - label_raw 为 color_raw 的子串；
      - label 在常见家族词典里，家族任一词出现在 color_raw 即匹配。
    """
    if not label_raw or not color_raw:
        return False
    lbl = str(label_raw).strip()
    cr = str(color_raw).strip()

    # 规范化：半角化 + 去两端空白
    lbl_norm = _norm(lbl)
    cr_norm = color_norm  # 传入时应已是 _norm(color_raw)

    # 1) 精确归一化相等
    if lbl_norm == cr_norm:
        return True

    # 2) 原文子串（精确子串）
    if lbl in cr:
        return True

    # 3) 分割后任一片段是 color_raw 的子串（处理 "シルバー SV" vs "シルバー" 之类）
    for tok in _COLOR_SEP_SPLIT_RE.split(lbl):
        tok = tok.strip()
        if not tok:
            continue
        if tok in cr:
            return True
        if _norm(tok) == cr_norm:
            return True

    # 4) 家族同义词（小词典，可按需扩充）
    FAMILY = {
        "blue": ["ブルー","青","blue"],
        "silver": ["シルバー","銀","silver"],
        "black": ["ブラック","黒","black"],
        "white": ["ホワイト","白","white"],
        "gold": ["ゴールド","金","gold"],
        "orange": ["オレンジ","橙"],
    }
    k = lbl.strip().lower()
    # 若 lbl 自身是家族词或家族内词，检查家族内的 token 是否出现在 color_raw
    for fam_tokens in FAMILY.values():
        if k in [t.lower() for t in fam_tokens] or any(tok in lbl for tok in fam_tokens):
            for tok in fam_tokens:
                if tok in cr:
                    return True

    return False

def _build_color_map_shop11(info_df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, Tuple[str, str]]]:
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

try:
    import langextract as lx
except Exception:  # 允许在未安装时仍可跑 fallback
    lx = None

SHOP11_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SHOP11_OLLAMA_MODEL_ID = os.getenv("SHOP11_OLLAMA_MODEL_ID", "gemma3:1b")


def _coerce_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int,)):
            return int(v)
        s = str(v).strip()
        if not s:
            return None
        # 允许 "1,000" / "-1000" 之类
        s = s.replace(",", "")
        return int(float(s))
    except Exception:
        return None


@lru_cache(maxsize=1)
def _shop11_model_config():
    """
    LangExtract 新版推荐的 ModelConfig 方式；若你的 langextract 版本不支持，会在调用处兜底到旧参数。
    """
    if lx is None:
        return None
    provider_kwargs = {
        "model_url": SHOP11_OLLAMA_URL,
        "temperature": float(os.getenv("SHOP11_OLLAMA_TEMPERATURE", "0.0")),
        "timeout": int(os.getenv("SHOP11_OLLAMA_TIMEOUT", "180")),
        "max_tokens": int(os.getenv("SHOP11_OLLAMA_MAX_TOKENS", "512")),
    }
    # JSON mode（能显著降低本地模型“夹杂解释文字”导致的解析失败）
    try:
        provider_kwargs["format_type"] = lx.data.FormatType.JSON
    except Exception:
        pass
    try:
        return lx.factory.ModelConfig(model_id=SHOP11_OLLAMA_MODEL_ID, provider_kwargs=provider_kwargs)
    except Exception:
        return None


def _lx_extract_ollama(text: str, prompt: str, examples: list):
    """
    返回 result 对象；失败返回 None
    """
    if lx is None:
        return None

    cfg = _shop11_model_config()
    try:
        if cfg is not None:
            return lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                config=cfg,
                fence_output=True,
                use_schema_constraints=False,
            )
    except TypeError:
        # 老版本 langextract 不支持 config=...
        pass
    except Exception:
        # config 路径失败则也尝试旧参数路径
        pass

    # 旧参数路径（v1.0.4 一类写法）
    try:
        return lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            language_model_type=lx.inference.OllamaLanguageModel,
            model_id=SHOP11_OLLAMA_MODEL_ID,
            model_url=SHOP11_OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )
    except Exception:
        return None


@lru_cache(maxsize=8)
def _shop11_lx_storage_materials(valid_models: Tuple[str, ...]):
    """
    storage_name 解析：device_model + storage_capacity
    """
    model_list = "\n".join(f"- {m}" for m in valid_models if m)

    prompt = textwrap.dedent(f"""\
        You are a strict parser.

        Input format:
          STORAGE: <text>

        Extract up to 2 items:
          1) device_model:
             - extraction_text must be an exact substring from STORAGE (do not invent text).
             - attributes must include: {{"model_norm": "<normalized model>"}}
             - model_norm MUST exactly equal one of:
{model_list}

          2) storage_capacity:
             - extraction_text must be an exact substring from STORAGE containing GB or TB (e.g., "256GB", "1TB").
             - attributes must include: {{"capacity_gb": <int>}}
             - Convert TB to GB using 1TB = 1024GB.

        If you cannot determine a field, do not output that extraction.
    """)

    examples = [
        lx.data.ExampleData(
            text="STORAGE: iPhone17 Pro Max 256GB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="iPhone17 Pro Max",
                    attributes={"model_norm": "iPhone 17 Pro Max"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="256GB",
                    attributes={"capacity_gb": 256},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="STORAGE: 17pro 1TB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="17pro",
                    attributes={"model_norm": "iPhone 17 Pro"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="1TB",
                    attributes={"capacity_gb": 1024},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="STORAGE: iPhone17 プロ 512GB",
            extractions=[
                lx.data.Extraction(
                    extraction_class="device_model",
                    extraction_text="iPhone17 プロ",
                    attributes={"model_norm": "iPhone 17 Pro"},
                ),
                lx.data.Extraction(
                    extraction_class="storage_capacity",
                    extraction_text="512GB",
                    attributes={"capacity_gb": 512},
                ),
            ],
        ),
    ]
    return prompt, examples


@lru_cache(maxsize=1)
def _shop11_lx_color_materials():
    """
    caution_empty 解析：color_delta（对 AVAILABLE_COLORS 里的颜色逐一给 delta）
    """
    prompt = textwrap.dedent("""\
        You are a strict parser.

        Input format:
          CAUTION: <text>
          AVAILABLE_COLORS: <c1 | c2 | c3 ...>

        Task:
          Extract color price deltas relative to the base unopened price.

        Output extractions:
          - extraction_class: "color_delta"
          - extraction_text: MUST be EXACTLY one color string from AVAILABLE_COLORS (copy it exactly).
          - attributes: {"delta_yen": <int>}

        Parsing rules:
          - "+2000円" => 2000, "-1,000円" => -1000 (JPY).
          - If CAUTION says "全色" or "すべて" or "全カラー", apply the same delta to ALL AVAILABLE_COLORS.
          - If a color has no delta info, do not output it.
          - If multiple deltas exist for same color, the last one wins.
          - Ignore notes in parentheses like "(未開封)".
    """)

    examples = [
        lx.data.ExampleData(
            text="CAUTION: ブルー、ブラック：-2,000円(未開封)\nAVAILABLE_COLORS: ブルー | ブラック | シルバー",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": -2000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="CAUTION: 全色:+1,000円\nAVAILABLE_COLORS: ブルー | ブラック",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": 1000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブラック",
                    attributes={"delta_yen": 1000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="CAUTION: シルバー・ブルー：-１０００円\nAVAILABLE_COLORS: ブルー | ブラック | シルバー",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -1000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": -1000},
                ),
            ],
        ),
    ]
    return prompt, examples


@lru_cache(maxsize=4096)
def _lx_parse_storage_shop11(storage: str, valid_models: Tuple[str, ...]) -> Tuple[str, Optional[int], Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]]:
    """
    返回 (model_norm, cap_gb, trace)
    trace: (class, extraction_text, sorted(attributes.items()) )
    """
    if not storage or lx is None:
        return "", None, tuple()

    prompt, examples = _shop11_lx_storage_materials(valid_models)
    txt = f"STORAGE: {storage}"

    res = _lx_extract_ollama(txt, prompt, examples)
    extrs = getattr(res, "extractions", None) or []

    model_norm = ""
    cap_gb: Optional[int] = None
    trace = []

    for e in extrs:
        cls = str(getattr(e, "extraction_class", "") or "")
        et = str(getattr(e, "extraction_text", "") or "")
        attrs = getattr(e, "attributes", None) or {}
        attrs_items = tuple(sorted((str(k), str(v)) for k, v in attrs.items()))
        trace.append((cls, et, attrs_items))

        if cls == "device_model":
            mn = (attrs.get("model_norm") or "").strip()
            if mn:
                # 再走一次你现有的规范化，确保和 info_df 的 key 完全一致
                model_norm = _normalize_model_generic(mn) or mn
        elif cls == "storage_capacity":
            cap_gb = _coerce_int(attrs.get("capacity_gb"))

    return model_norm, cap_gb, tuple(trace)


@lru_cache(maxsize=4096)
def _lx_parse_color_deltas_shop11(
    caution: str,
    available_colors: Tuple[str, ...],
) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]]:
    """
    返回 (deltas_items, trace)
    deltas_items: tuple of (color, delta_yen)  —— 最后出现覆盖前面
    """
    if lx is None:
        return tuple(), tuple()

    prompt, examples = _shop11_lx_color_materials()
    avail_line = " | ".join([c for c in available_colors if c])
    txt = f"CAUTION: {caution or ''}\nAVAILABLE_COLORS: {avail_line}"

    res = _lx_extract_ollama(txt, prompt, examples)
    extrs = getattr(res, "extractions", None) or []

    tmp: Dict[str, int] = {}
    trace = []

    for e in extrs:
        cls = str(getattr(e, "extraction_class", "") or "")
        et = str(getattr(e, "extraction_text", "") or "").strip()
        attrs = getattr(e, "attributes", None) or {}
        attrs_items = tuple(sorted((str(k), str(v)) for k, v in attrs.items()))
        trace.append((cls, et, attrs_items))

        if cls != "color_delta":
            continue

        delta = _coerce_int(attrs.get("delta_yen"))
        if delta is None or not et:
            continue

        # 目标：et 必须是 available_colors 之一；若不完全一致，fallback 做一层匹配
        if et in available_colors:
            tmp[et] = int(delta)
            continue

        # fallback：用你原有的 label 匹配逻辑把 et 贴到合法颜色上
        for c in available_colors:
            if _label_matches_color_shop11(et, c, _norm(c)):
                tmp[c] = int(delta)

    return tuple(tmp.items()), tuple(trace)




def clean_shop11(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    print("shop11:モバステ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    need_cols = ["storage_name", "price_unopened", "caution_empty", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"shop11 清洗器缺少必要列：{c}")

    df2 = df.copy().reset_index(drop=True)

    info_df = _load_iphone17_info_df_for_shop2()
    cmap_all = _build_color_map_shop11(info_df)

    # 用 info_df 推导“允许的规范化机型”，用来约束 LLM 输出，减少 key 对不上
    valid_models = tuple(
        sorted({m for m in info_df["model_name"].map(_normalize_model_generic).tolist() if m})
    )

    # DEBUG：只做轻量 hint 选择打印行（不在 debug 阶段对整列跑 LLM）
    debug_pos_set: set[int] = set()
    if debug:
        s_caution = df2["caution_empty"].fillna("").astype(str)
        _HINT_PAT = re.compile(
            r"(全色|シルバー|ブルー|ブラック|ホワイト|青|銀|黒|白|"
            r"[+\-−－]\s*[0-9０-９]|円|¥|￥|：|:)",
            re.I,
        )
        mask = s_caution.str.contains(_HINT_PAT, na=False)

        hit_cnt = 0
        for i in range(len(df2)):
            if bool(mask.iat[i]):
                debug_pos_set.add(i)
                hit_cnt += 1
                if hit_cnt >= int(debug_limit):
                    break

        if not debug_pos_set:
            debug_pos_set = set(range(min(int(debug_limit), len(df2))))

        print(f"[shop11 debug] total_rows={len(df2)}, print_rows={len(debug_pos_set)}, ollama={SHOP11_OLLAMA_URL}, model={SHOP11_OLLAMA_MODEL_ID}")

    def _dbg_on(pos: int) -> bool:
        return bool(debug) and (pos in debug_pos_set)

    rows: List[dict] = []

    for i, row in df2.iterrows():
        storage_raw = row.get("storage_name")
        price_raw = row.get("price_unopened")
        caution_raw = row.get("caution_empty")
        time_raw = row.get("time-scraped")

        storage = str(storage_raw or "").strip()
        if not storage:
            if _dbg_on(i):
                print("\n[shop11 debug] row_pos=", i, "SKIP_REASON: storage_name 为空")
            continue

        # 1) 先走 LLM 解析（失败则 fallback 到你现有 regex）
        model_norm, cap_gb, storage_trace = _lx_parse_storage_shop11(storage, valid_models)

        if not model_norm or cap_gb is None:
            # fallback：regex
            model_norm_fb = _normalize_model_generic(storage)
            cap_fb = _parse_capacity_gb(storage)
            if model_norm_fb and cap_fb is not None:
                model_norm, cap_gb = model_norm_fb, int(cap_fb)

        if not model_norm or cap_gb is None:
            if _dbg_on(i):
                print("\n[shop11 debug] row_pos=", i)
                print("  storage_name(raw):", repr(storage_raw))
                print("  LLM_trace:", storage_trace)
                print("  SKIP_REASON: model/cap 解析失败")
            continue

        cap_gb = int(cap_gb)
        key = (model_norm, cap_gb)
        color_map = cmap_all.get(key)

        # 若 key 对不上，再做一次“保险规范化”尝试（减少 LLM 输出微小差异导致 miss）
        if not color_map:
            model_norm2 = _normalize_model_generic(model_norm) or model_norm
            key2 = (model_norm2, cap_gb)
            color_map = cmap_all.get(key2)
            if color_map:
                key = key2
                model_norm = model_norm2

        if not color_map:
            if _dbg_on(i):
                print("\n[shop11 debug] row_pos=", i)
                print("  storage_name(raw):", repr(storage_raw))
                print("  model_norm/cap:", repr(model_norm), cap_gb, " key:", repr(key))
                print("  LLM_trace:", storage_trace)
                print("  SKIP_REASON: info 表中找不到该型号/容量映射")
            continue

        base_price = to_int_yen_shop11(price_raw)
        if base_price is None:
            if _dbg_on(i):
                print("\n[shop11 debug] row_pos=", i)
                print("  price_unopened(raw):", repr(price_raw))
                print("  SKIP_REASON: base_price 解析失败")
            continue

        # recorded_at（保持你原有逻辑）
        rec_at_raw = time_raw
        try:
            recorded_at = dateparser.parse(str(rec_at_raw)) if pd.notna(rec_at_raw) else None
        except Exception:
            recorded_at = rec_at_raw

        # 2) 颜色差额：LLM 优先；若为空且文本明显像差额规则，则 fallback 到 regex
        avail_colors = tuple(color_map.keys())
        caution_txt = _normalize_number_text(str(caution_raw or ""))  # 数字半角化，降低小模型误判

        deltas_items, deltas_trace = _lx_parse_color_deltas_shop11(caution_txt, avail_colors)
        color_deltas = dict(deltas_items)

        if not color_deltas and caution_txt.strip():
            # fallback：你现有的正则差额抽取（只在 LLM 没结果时用）
            deltas_fb = _extract_color_deltas_shop11(caution_txt)
            if deltas_fb:
                for col_norm, (pn, col_raw) in color_map.items():
                    for label_raw, delta in deltas_fb:
                        if _label_matches_color_shop11(label_raw, col_raw, col_norm):
                            color_deltas[col_norm] = int(delta)

        if _dbg_on(i):
            print("\n[shop11 debug] row_pos=", i)
            print("  storage_name(raw):", repr(storage_raw))
            print("  price_unopened(raw):", repr(price_raw))
            print("  caution_empty(raw):", repr(caution_raw))
            print("  time-scraped(raw):", repr(time_raw))
            print("  model_norm:", repr(model_norm), " cap_gb:", cap_gb, " key:", repr(key))
            print("  base_price:", base_price)
            print("  LLM_storage_trace:", storage_trace)
            print("  LLM_color_trace:", deltas_trace)
            print("  color_deltas:", color_deltas)

        # 3) 输出：每个颜色一行
        for col_norm, (pn, col_raw) in color_map.items():
            delta = int(color_deltas.get(col_norm, 0))
            price_new = int(base_price + delta)

            if _dbg_on(i):
                print("  -> OUT_ITEM:", {
                    "part_number": pn,
                    "color_raw": col_raw,
                    "base": int(base_price),
                    "delta": int(delta),
                    "final": int(price_new),
                })

            rows.append({
                "part_number": pn,
                "shop_name": "モバステ",
                "price_new": price_new,
                "recorded_at": recorded_at,
            })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    if debug:
        print(f"\n[shop11 debug] out_rows={len(out)} head=\n{out.head(10).to_string(index=False)}")

    return out