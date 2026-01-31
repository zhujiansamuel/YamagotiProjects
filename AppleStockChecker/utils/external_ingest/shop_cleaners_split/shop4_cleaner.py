from __future__ import annotations

import os
import re
import time
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd

from ...external_ingest.helpers import to_int_yen, parse_dt_aware


# ----------------------------
# 原有：机型/容量 normalization
# ----------------------------
_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)

def _find_base_price(df: pd.DataFrame, idx: int) -> Optional[int]:
    """
    按规范：机种行(data11非空)的上一行 data 是基准价。
    若上一行取不到，向上最多回溯 3 行找首个含“円”的金额。
    """
    for j in range(idx - 1, max(-1, idx - 4), -1):
        if j < 0:
            break
        s = str(df["data"].iat[j]) if "data" in df.columns else ""
        if s and ("円" in s or re.search(r"\d[\d,]*", s)):
            price = to_int_yen(s)
            if price:
                return int(price)
    return None


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

    jan_candidates = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"jan", "jancode", "jan_code", "jan13", "jan14"}:
            jan_candidates.append(c)
        elif "jan" in cl or "jan" in str(c):  # 兼容 'JANコード' 等
            jan_candidates.append(c)
    jan_candidates = list(dict.fromkeys(jan_candidates))

    cols = ["part_number", "model_name", "capacity_gb", "color"]
    if jan_candidates:
        src = jan_candidates[0]
        df["jan"] = df[src]
        cols.append("jan")

    return df[cols]


def _normalize_model_generic(text: str) -> str:
    if not text:
        return ""
    t = str(text).replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t)

    t = (t.replace("プロマックス", "Pro Max")
           .replace("プロ", "Pro")
           .replace("プラス", "Plus")
           .replace("ミニ", "mini")
           .replace("エアー", "Air")
           .replace("エア", "Air"))

    t = re.sub(r"(\d{2})(?=[A-Za-z])", r"\1 ", t)

    t = re.sub(r"(?i)\bpro\s*max\b", "Pro Max", t)
    t = re.sub(r"(?i)\bpro\b", "Pro", t)
    t = re.sub(r"(?i)\bplus\b", "Plus", t)
    t = re.sub(r"(?i)\bmini\b", "mini", t)

    if "iPhone" not in t and re.search(r"\b1[0-9]\b", t):
        t = re.sub(r"\b(1[0-9])\b", r"iPhone \1", t, count=1)

    t = re.sub(r"(?i)\biPhone\s+17\s+Air\b", "iPhone Air", t)

    t = re.sub(r"(\d+(?:\.\d+)?\s*TB|\d{2,4}\s*GB)", "", t, flags=re.I)
    t = re.sub(r"SIMフリ[ーｰ–-]?|シムフリ[ーｰ–-]?|sim\s*free", "", t, flags=re.I)
    t = re.sub(r"[（）\(\)\[\]【】].*?[（）\(\)\[\]【】]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    m = _NUM_MODEL_PAT.search(t)
    if m:
        base = f"{m.group(1)} {m.group(2)}"
        suf  = (m.group(3) or "").strip()
        return f"{base} {suf}".strip()

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


# ----------------------------
# LLM（LangExtract + Ollama）解析：颜色±金额
# ----------------------------
LABEL_SPLIT_RE = re.compile(r"[／/、，,・\s]+")

_FZ_TO_HZ_TRANS = str.maketrans({
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    '，': ',', '．': '.', '：': ':', '（': '(', '）': ')', '　': ' ',
    '－': '-', '＋': '+', '¥': '', '￥': ''
})


def _normalize_amount_text(s: str) -> Optional[int]:
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


def _coerce_int_maybe(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    sign = 1
    if s[0] in ("-", "−", "－"):
        sign = -1
    amt = _normalize_amount_text(s)
    if amt is None:
        return None
    return sign * int(amt)


def _split_labels(label: str) -> List[str]:
    return [p.strip() for p in LABEL_SPLIT_RE.split(label or "") if p and p.strip()]


# 可配置：默认用你给定的 Ollama
SHOP4_USE_LLM = os.getenv("SHOP4_USE_LLM", "1") not in {"0", "false", "False"}
SHOP4_OLLAMA_URL = os.getenv("SHOP4_OLLAMA_URL", "http://localhost:11434")
SHOP4_OLLAMA_MODEL_ID = os.getenv("SHOP4_OLLAMA_MODEL_ID", "gemma3:1b")

try:
    import langextract as lx
except Exception:
    lx = None


_SHOP4_LE_PROMPT = textwrap.dedent("""\
You are extracting structured information from a Japanese iPhone pricing table.

Input text contains one or more lines. A relevant line expresses:
- a color label (e.g., シルバー, ディープブルー) OR 全色 (means "all colors"),
- optionally followed by a signed yen adjustment amount.

Rules:
- Extract one item per color label.
- extraction_text MUST be the exact color label substring from the input (do not translate).
- attributes MUST include delta_yen as an integer (negative for discounts).
- Sign can be + or - and may include unicode minus characters: '−' or '－'.
- Amount may include commas and/or full-width digits.
- If a line indicates 全色 but has no amount, set delta_yen = 0.
- If a line does not express a color adjustment, output no extractions.
""").strip()

_SHOP4_LE_EXAMPLES = []
if lx is not None:
    _SHOP4_LE_EXAMPLES = [
        lx.data.ExampleData(
            text="シルバー-1,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -1000},
                )
            ],
        ),
        lx.data.ExampleData(
            text="シルバー/ディープブルー-3,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="シルバー",
                    attributes={"delta_yen": -3000},
                ),
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ディープブルー",
                    attributes={"delta_yen": -3000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="全色-2,000円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="全色",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="全色",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="全色",
                    attributes={"delta_yen": 0},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="ブルー ＋０円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="ブルー",
                    attributes={"delta_yen": 0},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="全色－２，０００円",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_delta",
                    extraction_text="全色",
                    attributes={"delta_yen": -2000},
                ),
            ],
        ),
    ]


def _lx_extract_color_deltas(text: str):
    """
    对 text 做一次 LangExtract 抽取，返回 result.extractions（若不可用则空列表）。
    """
    if not (SHOP4_USE_LLM and lx is not None and isinstance(text, str) and text.strip()):
        return []

    kwargs = dict(
        text_or_documents=text,
        prompt_description=_SHOP4_LE_PROMPT,
        examples=_SHOP4_LE_EXAMPLES,
        model_id=SHOP4_OLLAMA_MODEL_ID,
        model_url=SHOP4_OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
        extraction_passes=1,
        max_workers=1,
        max_char_buffer=1500,
        temperature=0.0,
        language_model_params={
            # 可选：让 Ollama 端更“稳”
            "timeout": 60,
            "keep_alive": 10 * 60,
        },
    )

    # 兼容不同版本：有的版本建议显式指定 OllamaLanguageModel
    try:
        if hasattr(lx, "inference") and hasattr(lx.inference, "OllamaLanguageModel"):
            kwargs["language_model_type"] = lx.inference.OllamaLanguageModel
    except Exception:
        pass

    try:
        result = lx.extract(**kwargs)
    except Exception:
        return []

    exts = getattr(result, "extractions", None)
    return list(exts) if exts else []


def _get_start_pos(extraction) -> int:
    ci = getattr(extraction, "char_interval", None)
    if ci is None:
        return 0
    for attr in ("start_pos", "start", "begin"):
        if hasattr(ci, attr):
            try:
                return int(getattr(ci, attr))
            except Exception:
                pass
    if isinstance(ci, dict):
        for k in ("start_pos", "start", "begin"):
            if k in ci:
                try:
                    return int(ci[k])
                except Exception:
                    pass
    return 0


def _collect_adjustments_shop4_llm(df: pd.DataFrame, start_idx: int) -> Dict[str, int]:
    """
    用 LangExtract 一次性解析“机种段落”(block)里的所有颜色±金额。
    返回：{ color_norm | "ALL" : delta_int }
    """
    lines: List[str] = []
    n = len(df)

    # 收集 block 文本：从 start_idx 到下一个 data11 非空前一行
    for j in range(start_idx, n):
        if j > start_idx:
            nxt_model = ""
            val = df["data11"].iat[j] if "data11" in df.columns else ""
            nxt_model = str(val) if val is not None else ""
            if nxt_model.strip():
                break
        raw = df["data"].iat[j] if "data" in df.columns else ""
        lines.append("" if raw is None else str(raw))

    if not lines:
        return {}

    block_text = "\n".join(lines)

    # 计算每一行在 block_text 的范围，用于识别“同一行(机种行)的全色”
    line0_start = 0
    line0_end = len(lines[0]) if lines else 0

    exts = _lx_extract_color_deltas(block_text)
    if not exts:
        return {}

    # 按出现顺序处理，保持覆盖逻辑一致
    exts = sorted(exts, key=_get_start_pos)

    result: Dict[str, int] = {}
    global_all_delta: Optional[int] = None

    for ex in exts:
        cls = str(getattr(ex, "extraction_class", "") or "").strip()
        if cls and cls.lower() not in {"color_delta", "colordelta", "color"}:
            # 防止模型乱输出其他类
            continue

        label = str(getattr(ex, "extraction_text", "") or "").strip()
        if not label:
            continue

        attrs = getattr(ex, "attributes", None)
        attrs = attrs if isinstance(attrs, dict) else {}
        delta = _coerce_int_maybe(attrs.get("delta_yen"))
        if delta is None:
            # 兜底：如果是“全色”且无金额，按 0
            if "全色" in label and not re.search(r"[0-9０-９]", block_text):
                delta = 0
            else:
                continue

        start_pos = _get_start_pos(ex)

        # same-line（机种行同一行 data）里的 全色：作为最高优先级的 ALL
        if "全色" in label and line0_start <= start_pos < max(line0_end, line0_start):
            global_all_delta = int(delta)

        # label 可能是复合项，拆分后分别写入
        for lbl in _split_labels(label):
            if "全色" in lbl:
                result["ALL"] = int(delta)
            else:
                result[_norm(lbl)] = int(delta)

    # 同行全色优先覆盖
    if global_all_delta is not None:
        result["ALL"] = int(global_all_delta)

    return result


# ---- 旧 regex 解析兜底（可选）----
_COLOR_DELTA_RE = re.compile(
    r"""^\s*
        (?P<label>全色|[\S　 ]*?[^\s　])
        \s*
        (?P<sign>[+\-−－])?
        \s*
        (?P<amount>\d[\d,]*)\s*円?
        \s*$
    """,
    re.VERBOSE,
)


def _parse_color_delta_shop4_regex(line: str) -> Optional[List[Tuple[str, int]]]:
    if not line or not isinstance(line, str):
        return None
    s = line.strip()
    if s == "全色" or s == "全 色":
        return [("全色", 0)]

    m = _COLOR_DELTA_RE.match(s)
    if not m:
        am = re.search(r"([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)\s*円?", s)
        if not am:
            if "全色" in s:
                return [("全色", 0)]
            return None
        sign = am.group(1) or "+"
        amt_txt = am.group(2)
        amt = _normalize_amount_text(amt_txt)
        if amt is None:
            try:
                amt = to_int_yen(amt_txt)
            except Exception:
                amt = None
        if amt is None:
            return None
        if sign in ("-", "−", "－"):
            amt = -int(amt)
        else:
            amt = int(amt)

        label_part = s[:am.start()].strip()
        if not label_part:
            return None
        labels = [p for p in LABEL_SPLIT_RE.split(label_part) if p]
        return [(lbl.strip(), int(amt)) for lbl in labels]

    label_raw = m.group("label").strip()
    sign = m.group("sign") or "+"
    amt_val = None
    try:
        amt_val = to_int_yen(m.group("amount"))
    except Exception:
        amt_val = None
    if amt_val is None:
        amt_val = _normalize_amount_text(m.group("amount"))
    if amt_val is None:
        return None

    amt_val = int(amt_val)
    if sign in ("-", "−", "－"):
        amt_val = -amt_val

    labels = [p for p in LABEL_SPLIT_RE.split(label_raw) if p]
    if not labels:
        return None
    return [(lbl.strip(), int(amt_val)) for lbl in labels]


def _collect_adjustments_shop4(df: pd.DataFrame, start_idx: int) -> Dict[str, int]:
    """
    统一入口：优先走 LLM；LLM 不可用/失败则走 regex 兜底。
    """
    if SHOP4_USE_LLM and lx is not None:
        try:
            return _collect_adjustments_shop4_llm(df, start_idx)
        except Exception:
            pass

    # regex 兜底：保持你原来的行为（逐行扫描）
    result: Dict[str, int] = {}
    n = len(df)
    for j in range(start_idx, n):
        nxt_model = ""
        if "data11" in df.columns:
            val = df["data11"].iat[j]
            nxt_model = str(val) if val is not None else ""
        if j > start_idx and nxt_model.strip():
            break

        line = ""
        if "data" in df.columns:
            val = df["data"].iat[j]
            line = str(val) if val is not None else ""

        parsed = _parse_color_delta_shop4_regex(line)
        if not parsed:
            continue

        for label, delta in parsed:
            if "全色" in label:
                result["ALL"] = int(delta)
            else:
                result[_norm(label)] = int(delta)
    return result


# ----------------------------
# shop4 主清洗器
# ----------------------------
def clean_shop4(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print("shop4:モバイルミックス---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"shop4:モバイルミックス---------->进入清洗器时间: {now}")

    for c in ["data", "data11", "time-scraped"]:
        if c not in df.columns:
            raise ValueError(f"shop4 清洗器缺少必要列：{c}")

    info_df = _load_iphone17_info_df_for_shop2().copy()
    info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
    info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")
    info_df["color_norm"] = info_df["color"].map(lambda x: _norm(str(x)))

    pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    for _, r in info_df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        col = r["color_norm"]
        pn = str(r["part_number"])
        if pd.isna(cap) or not m or not col:
            continue
        key = (m, int(cap))
        pn_map.setdefault(key, {})
        pn_map[key][col] = pn

    _DEBUG_HINT_PAT = re.compile(
        r"(全色|シルバー|ブルー|ディープブルー|ブラック|ホワイト|グリーン|ピンク|レッド|イエロー|パープル|"
        r"チタ|チタン|ナチュラル|ミッドナイト|スターライト|"
        r"[+\-−－]\s*\d|[\d,]+\s*円)",
        re.I,
    )

    def _next_model_idx(start: int) -> int:
        n = len(df)
        for k in range(start + 1, n):
            s = str(df["data11"].iat[k]) if df["data11"].iat[k] is not None else ""
            if s.strip():
                return k
        return n

    printed_models = 0
    rows: List[dict] = []
    n = len(df)

    for i in range(n):
        model_text = str(df["data11"].iat[i]) if df["data11"].iat[i] is not None else ""
        model_text = model_text.strip()
        if not model_text:
            continue

        block_end = _next_model_idx(i) - 1
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        rec_at = parse_dt_aware(df["time-scraped"].iat[i])

        def _maybe_print_header(reason: Optional[str] = None):
            nonlocal printed_models
            if not debug or printed_models >= int(debug_limit):
                return
            printed_models += 1

            # print("\n[shop4 debug] model_row_idx=", i, " block_end_idx=", block_end)
            print(f"[shop4 debug] model_row_idx= {i} ,block_end_idx={block_end}")
            print(f"data11(raw)      : {repr(model_text)}")
            # print("  data11(raw):", repr(model_text))
            # print("  model_norm:", repr(model_norm), " capacity_gb:", repr(cap_gb))
            # print("  recorded_at:", repr(rec_at))
            print(f"recorded_at      : {repr(rec_at)}")
            if reason:
                print(f"SKIP_REASON      : {reason}")

            # print("base_price_backtrack                ||")
            for j in range(i - 1, max(-1, i - 4), -1):
                if j < 0:
                    break
                raw = str(df["data"].iat[j]) if df["data"].iat[j] is not None else ""
                parsed = to_int_yen(raw) if raw else None
                # print(f"       idx={j} data(raw)={raw!r} -> parsed={parsed!r}")

            print("block_lines(color_delta_candidates) ||")
            for j in range(i, block_end + 1):
                raw = str(df["data"].iat[j]) if df["data"].iat[j] is not None else ""
                if not raw.strip():
                    continue
                if not _DEBUG_HINT_PAT.search(raw):
                    continue
                print(f"-----> idx={j} data(raw)={raw!r}")

        if not model_norm or pd.isna(cap_gb):
            if debug and printed_models < int(debug_limit):
                _maybe_print_header("model/cap 解析失败（data11 无法归一化或容量缺失）")
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            if debug and printed_models < int(debug_limit):
                has_hint = any(
                    _DEBUG_HINT_PAT.search(str(df["data"].iat[j]) if df["data"].iat[j] is not None else "")
                    for j in range(i, block_end + 1)
                )
                if has_hint:
                    _maybe_print_header(f"info 表没有该 (model,cap) key={key!r}")
            continue

        base_price = _find_base_price(df, i)
        if base_price is None:
            if debug and printed_models < int(debug_limit):
                has_hint = any(
                    _DEBUG_HINT_PAT.search(str(df["data"].iat[j]) if df["data"].iat[j] is not None else "")
                    for j in range(i, block_end + 1)
                )
                if has_hint:
                    _maybe_print_header("找不到可用基准价（上一行及回溯行均无法解析）")
            continue

        # ★ 调整：这里直接收集整个 block 的 ALL/颜色 delta（LLM 优先）
        adjustments = _collect_adjustments_shop4(df, i)

        if debug and printed_models < int(debug_limit):
            has_hint = bool(adjustments) or any(
                _DEBUG_HINT_PAT.search(str(df["data"].iat[j]) if df["data"].iat[j] is not None else "")
                for j in range(i, block_end + 1)
            )
            if has_hint:
                _maybe_print_header(None)
                print(f"adjustments(collected)  : {adjustments}")
                print(f"base_price(final)       : {base_price:<7}")
                # print("  base_price(final):", base_price)

                # print("  adjustments(collected):", adjustments)


                # print("  llm_enabled:", bool(SHOP4_USE_LLM and lx is not None),
                #       " model_id:", SHOP4_OLLAMA_MODEL_ID, " url:", SHOP4_OLLAMA_URL)

        if "ALL" in adjustments:
            final_price = int(base_price + adjustments["ALL"])
            for col_norm, pn in color_to_pn.items():
                rows.append({
                    "part_number": pn,
                    "shop_name": "モバイルミックス",
                    "price_new": int(final_price),
                    "recorded_at": rec_at,
                })
        else:
            for col_norm, pn in color_to_pn.items():
                delta = int(adjustments.get(col_norm, 0))
                final_price = int(base_price + delta)
                rows.append({
                    "part_number": pn,
                    "shop_name": "モバイルミックス",
                    "price_new": int(final_price),
                    "recorded_at": rec_at,
                })

    out = pd.DataFrame(rows, columns=["part_number", "shop_name", "price_new", "recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number", "price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")

    # if debug:
    #     print(f"\n[shop4 debug] out_rows={len(out)} head=\n{out.head(10).to_string(index=False)}")

    return out
