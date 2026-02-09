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

_NUM_MODEL_PAT = re.compile(r"(iPhone)\s*(\d{2})(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
_AIR_PAT = re.compile(r"(iPhone)\s*(Air)(?:\s*(Pro\s*Max|Pro|Plus|mini))?", re.I)
ABS_LIKE_MIN = int(os.getenv("SHOP9_ABS_LIKE_MIN", "50000"))  # iPhone17 绝对价量级阈值
DELTA_HINT_RE = re.compile(r"(?:[+\-−－]|値下げ|値引|割引|円引|OFF|オフ|減額)", re.I)





def _norm_cls(x: str) -> str:
    # 容错：abs price / abs-price / ABS_PRICE 统一
    s = (x or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    return s

def _bucket_amount(cls_norm: str, ex_text: str, amt: int) -> str:
    """
    返回 "abs" 或 "delta"
    规则：
      - 有负号/折扣词/加减符号 => delta
      - 金额量级很大(>=ABS_LIKE_MIN)且无加减线索 => abs（即使模型标成 delta）
      - 其余按 class；不认识则按金额量级兜底
    """
    tx = ex_text or ""
    if amt is None:
        return "delta"
    if amt < 0:
        return "delta"
    if DELTA_HINT_RE.search(tx):
        return "delta"
    if abs(amt) >= ABS_LIKE_MIN:
        return "abs"
    if cls_norm in {"abs_price", "abs", "absolute"}:
        return "abs"
    if cls_norm in {"delta", "delta_price", "adjust", "adjustment"}:
        return "delta"
    return "delta"


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




def clean_shop9(
    df: pd.DataFrame,
    debug: bool = True,
    debug_limit: int = 30,
) -> pd.DataFrame:
    import time
    import textwrap
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"shop9:アキモバ---------->进入清洗器时间: {now}")

    # print("shop9:アキモバ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    info_df = _load_iphone17_info_df_for_shop2()
    col_model = "機種名"
    col_price = "買取価格"
    col_color = "色・詳細等"
    col_time  = "time-scraped"

    for need in (col_model, col_price, col_color, col_time):
        if need not in df.columns:
            raise ValueError(f"shop9 清洗器缺少必要列：{need}")

    # =============== DEBUG 行选择（保留原逻辑） ===============
    debug_pos_set = set()
    if debug:
        COLOR_PAT = re.compile(
            r"(ブラック|ホワイト|ブルー|グリーン|ピンク|レッド|イエロー|パープル|オレンジ|"
            r"シルバー|ゴールド|グラファイト|ミッドナイト|スターライト|ナチュラル|"
            r"チタニウム|チタン|黒|白|青|緑|赤|黄|紫|橙|銀|金|"
            r"Black|White|Blue|Green|Pink|Red|Yellow|Purple|Orange|Silver|Gold|Titanium)",
            re.I,
        )
        DISCOUNT_PAT = re.compile(r"(値下げ|値引|割引|円引|OFF|オフ|[+＋\-−–－]\s*[０-９0-9])", re.I)
        ABS_PRICE_PAT = re.compile(r"(?:¥|￥)?\s*[０-９0-9]{2,3}(?:[０-９0-9,，]{3,})\s*(?:円)?")

        s_color_ser = df[col_color].fillna("").astype(str)
        s_price_ser = df[col_price].fillna("").astype(str)
        has_color = s_color_ser.str.contains(COLOR_PAT, na=False) | s_price_ser.str.contains(COLOR_PAT, na=False)
        has_discount = s_color_ser.str.contains(DISCOUNT_PAT, na=False) | s_price_ser.str.contains(DISCOUNT_PAT, na=False)
        has_abs_price = s_color_ser.str.contains(ABS_PRICE_PAT, na=False) | s_price_ser.str.contains(ABS_PRICE_PAT, na=False)
        mask = has_color & (has_discount | has_abs_price)

        hit_cnt = 0
        for pos, hit in enumerate(mask.to_numpy()):
            if hit:
                debug_pos_set.add(pos)
                hit_cnt += 1
                if hit_cnt >= int(debug_limit):
                    break
        if not debug_pos_set:
            debug_pos_set = set(range(min(int(debug_limit), len(df))))
        print(f"[shop9 debug] total_rows={len(df)}, hit_rows={int(mask.sum())}, print_rows={len(debug_pos_set)}")

    def _dbg_on(row_pos: int) -> bool:
        return bool(debug) and (row_pos in debug_pos_set)

    def _dprint(row_pos: int, *args, **kwargs):
        if _dbg_on(row_pos):
            print(*args, **kwargs)

    # =============== 同义表（原逻辑保留；LLM 产出不规范时兜底用） ===============
    FAMILY_SYNONYMS_SHOP9 = {
        "blue": ["ブルー", "青", "ディープブルー", "ディープ ブルー"],
        "ブルー": ["ブルー", "青", "ディープブルー"],
        "青": ["ブルー", "青", "ディープブルー"],
        "ディープブルー": ["ディープブルー", "ブルー", "青"],
        "silver": ["シルバー", "銀"],
        "シルバー": ["シルバー", "銀"],
        "銀": ["シルバー", "銀"],
        "black": ["ブラック", "黒"],
        "ブラック": ["ブラック", "黒"],
        "黒": ["ブラック", "黒"],
        "orange": ["オレンジ", "橙", "コズミックオレンジ"],
        "オレンジ": ["オレンジ", "橙", "コズミックオレンジ"],
        "橙": ["オレンジ", "橙", "コズミックオレンジ"],
        "コズミックオレンジ": ["コズミックオレンジ", "オレンジ", "橙", "orange"],
        "white": ["ホワイト", "白"],
        "ホワイト": ["ホワイト", "白"],
    }
    SYNONYM_LOOKUP = {}
    for k, vs in FAMILY_SYNONYMS_SHOP9.items():
        SYNONYM_LOOKUP[k] = list(dict.fromkeys(vs))
        for v in vs:
            SYNONYM_LOOKUP.setdefault(v, [])
            SYNONYM_LOOKUP[v] = list(dict.fromkeys(SYNONYM_LOOKUP[v] + vs + [k]))

    def _norm(s: str) -> str:
        if s is None:
            return ""
        t = str(s).strip().lower()
        t = t.replace("　", " ")
        t = re.sub(r"\s+", " ", t)
        # 全角数字转半角
        t = t.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        return t

    # =============== 无正则的金额解析（尽量少依赖 regex） ===============
    def _coerce_signed_int(x) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, (int,)) and not isinstance(x, bool):
            return int(x)

        s = str(x)
        # 全角数字/符号 -> 半角
        s = s.translate(str.maketrans("０１２３４５６７８９＋－−，", "0123456789+--,"))

        sign = 1
        digits = []
        sign_seen = False
        started = False
        for ch in s:
            if not started and not sign_seen and ch in "+-" :
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

    # =============== LLM 抽取：LangExtract + Ollama ===============
    # 开关：默认启用；你也可通过环境变量 SHOP9_USE_LLM=0 关闭
    USE_LLM = os.getenv("SHOP9_USE_LLM", "1").strip() not in {"0", "false", "False", "no", "NO"}

    OLLAMA_URL = os.getenv("SHOP9_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    LLM_MODEL_ID = os.getenv("SHOP9_LX_MODEL_ID") or os.getenv("SHOP9_LLM_MODEL_ID") or "gemma3:1b"
    LLM_TEMPERATURE = float(os.getenv("SHOP9_LLM_TEMPERATURE", "0.0") or "0.0")

    @lru_cache(maxsize=1)
    def _shop9_lx_examples():
        """
        Few-shot 示例：教模型识别
        - “多个颜色共享一个价格”
        - “全色 +/-”
        - “每色 +/-”
        """
        import langextract as lx

        return [
            lx.data.ExampleData(
                text="買取価格: 195,500円\n色・詳細等: 未開 橙194,500/青,銀195,500",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="abs_price",
                        extraction_text="橙194,500",
                        attributes={"colors": ["コズミックオレンジ"], "amount_yen": 194500},
                    ),
                    lx.data.Extraction(
                        extraction_class="abs_price",
                        extraction_text="青,銀195,500",
                        attributes={"colors": ["ディープブルー", "シルバー"], "amount_yen": 195500},
                    ),
                ],
            ),
            lx.data.ExampleData(
                text="買取価格: 200,000円\n色・詳細等: ブラック -2,000円 / シルバー:+1000",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="delta",
                        extraction_text="ブラック -2,000円",
                        attributes={"colors": ["ブラック"], "amount_yen": -2000},
                    ),
                    lx.data.Extraction(
                        extraction_class="delta",
                        extraction_text="シルバー:+1000",
                        attributes={"colors": ["シルバー"], "amount_yen": 1000},
                    ),
                ],
            ),
            lx.data.ExampleData(
                text="買取価格: 180,000円\n色・詳細等: 全色-500円",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="delta",
                        extraction_text="全色-500円",
                        attributes={"colors": ["ALL"], "amount_yen": -500},
                    ),
                ],
            ),
            lx.data.ExampleData(
                text="買取価格: -\n色・詳細等: ブルー：229,000円 シルバー：230000",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="abs_price",
                        extraction_text="ブルー：229,000円",
                        attributes={"colors": ["ブルー"], "amount_yen": 229000},
                    ),
                    lx.data.Extraction(
                        extraction_class="abs_price",
                        extraction_text="シルバー：230000",
                        attributes={"colors": ["シルバー"], "amount_yen": 230000},
                    ),
                ],
            ),
            lx.data.ExampleData(
                text="買取価格: 230,500円\n色・詳細等: 未開 橙,銀230,500/青229,000",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="abs_price",
                        extraction_text="橙,銀230,500",
                        attributes={"colors": ["橙", "銀"], "amount_yen": 230500},
                    ),
                    lx.data.Extraction(
                        extraction_class="abs_price",
                        extraction_text="青229,000",
                        attributes={"colors": ["青"], "amount_yen": 229000},
                    ),
                ],
            ),
        ]

    def _build_color_aliases(available_colors: List[str]) -> Dict[str, List[str]]:
        out = {}
        for c in available_colors:
            c0 = str(c).strip()
            if not c0:
                continue
            syns = SYNONYM_LOOKUP.get(c0, [])
            # 也把“自身”放进去
            out[c0] = list(dict.fromkeys([c0] + syns))[:20]
        return out

    def _map_to_available_color(raw_color: str, available_set: set) -> Optional[str]:
        if not raw_color:
            return None
        rc = str(raw_color).strip()
        if not rc:
            return None

        if rc.upper() == "ALL" or rc == "全色":
            return "ALL"

        if rc in available_set:
            return rc

        # 小写等价
        rcn = _norm(rc)
        for c in available_set:
            if _norm(c) == rcn:
                return c

        # 同义词兜底
        if rc in SYNONYM_LOOKUP:
            for syn in SYNONYM_LOOKUP[rc]:
                if syn in available_set:
                    return syn
                synn = _norm(syn)
                for c in available_set:
                    if _norm(c) == synn:
                        return c

        # 包含关系兜底
        for c in available_set:
            cn = _norm(c)
            if rcn and (rcn in cn or cn in rcn):
                return c

        return None

    @lru_cache(maxsize=4096)
    def _llm_extract_rules_cached(
        price_text: str,
        detail_text: str,
        avail_colors_key: Tuple[str, ...],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        返回:
          abs_map: {color_norm or 'ALL': amount_yen}
          delta_map: {color_norm or 'ALL': signed_delta_yen}
        """
        try:
            import langextract as lx
        except Exception:
            return {}, {}

        available_colors = list(avail_colors_key)
        aliases = _build_color_aliases(available_colors)

        # 输入拼接：让模型同时看到“基准价”和“详情”
        input_text = f"買取価格: {price_text}\n色・詳細等: {detail_text}"
        # ------------------------------------------------------------------------------------------
        prompt = textwrap.dedent(f"""\
        You are parsing Japanese iPhone buyback pricing notes.

        Goal:
        - Extract explicit color-scoped absolute prices and signed adjustments from the input.
        - Extract ONLY what is explicitly present. Do NOT infer missing prices or colors.

        How to interpret the format (VERY IMPORTANT):
        - The detail field (色・詳細等) may contain multiple independent groups separated by '/', '／', newline.
        - In each group, one amount (e.g. 230,500) applies to the color(s) listed immediately before it in that group.
        - Multiple colors in the same group can be separated by ',', '，', '、', or spaces. All those colors share the same amount in that group.
        - Example: "橙,銀230,500/青229,000" must produce TWO extractions:
          1) colors=["橙","銀"], amount_yen=230500
          2) colors=["青"], amount_yen=229000
        - Condition words are NOT colors: ignore terms like "未開", "未使用", "中古", "美品", etc.
        - When several colors and numbers appear in one sequence without separators
  (e.g. "橙193,500青193,500銀195,000"), each color MUST be paired with the closest number immediately following it.

        What to output:
        - extraction_class MUST be one of: "abs_price", "delta"
        - attributes.amount_yen MUST be an integer yen value (no commas). For delta, keep the sign (e.g. -2000).
        - attributes.colors MUST be a list of color labels AS THEY APPEAR IN THE INPUT (e.g. "青", "銀", "橙").
          You may also output "ALL" only when the text explicitly indicates all colors (e.g. "全色").
        - Do NOT drop a price mention just because it equals the base price shown in 買取価格.

        Normalization hints (for your reference):
        AVAILABLE_COLORS (system will map your labels to these):
        {json.dumps(available_colors, ensure_ascii=False)}

        COLOR_ALIASES (system will map using these aliases):
        {json.dumps(aliases, ensure_ascii=False)}
        """)
        # ------------------------------------------------------------------------------------------

        kw = dict(
            text_or_documents=input_text,
            prompt_description=prompt,
            examples=_shop9_lx_examples(),
            model_id=LLM_MODEL_ID,
            model_url=OLLAMA_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

        # 兼容不同版本参数签名：temperature 可能不被支持
        try:
            result = lx.extract(**kw, temperature=LLM_TEMPERATURE)
        except TypeError:
            result = lx.extract(**kw)
        except Exception:
            return {}, {}

        abs_map: Dict[str, int] = {}
        delta_map: Dict[str, int] = {}

        extractions = getattr(result, "extractions", None) or []
        avail_set = set(available_colors)

        for ex in extractions:
            cls_raw = str(getattr(ex, "extraction_class", "") or "")
            cls_norm = _norm_cls(cls_raw)
            attrs = getattr(ex, "attributes", None) or {}
            ex_text = str(getattr(ex, "extraction_text", "") or "")

            # 取 amount
            amt = attrs.get("amount_yen")
            amt_i = _coerce_signed_int(amt)
            if amt_i is None:
                amt_i = _coerce_signed_int(ex_text)
            if amt_i is None:
                continue

            # colors
            colors = attrs.get("colors") or attrs.get("color") or []
            if isinstance(colors, str):
                colors = [colors]
            if not isinstance(colors, list):
                colors = list(colors) if colors else []

            bucket = _bucket_amount(cls_norm, ex_text, int(amt_i))

            for c_raw in colors:
                mapped = _map_to_available_color(str(c_raw), avail_set)
                if not mapped:
                    continue
                if bucket == "abs":
                    abs_map[mapped] = int(amt_i)
                else:
                    delta_map[mapped] = int(amt_i)

        return abs_map, delta_map

    # ===============（可选）正则回退：保留你原函数以防 LLM 失败 ===============
    SPLIT_SEPS = r"[/／、，,;；\s]+"

    def _is_pure_number_token(tok: str) -> bool:
        if not tok:
            return False
        t = _norm(tok)
        t = t.replace(",", "").replace("，", "")
        return t.isdigit()

    def _norm_amount_to_int(x: str) -> Optional[int]:
        if not x:
            return None
        s = str(x).strip()
        s = s.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
        s = s.replace(",", "")
        if not s.isdigit():
            return None
        return int(s)

    ABS_MIN_YEN = int(os.getenv("SHOP9_ABS_LIKE_MIN", "50000"))  # 认为是绝对价的最小金额

    def _extract_amount_after_alias(text: str, alias: str) -> Optional[int]:
        """
        在 text 中查找形如 'alias 193,500' / 'alias193,500' / 'alias 193500円' 这种片段，
        只取 alias 后面“最近的那串数字”。
        不吃减价形式 'alias-500円'（中间有 '-'）。
        """
        if not text or not alias:
            return None
        s = str(text)

        # 允许 alias 后有若干空白，再跟可选的货币符号，再跟数字
        pat = re.compile(
            rf"{re.escape(alias)}\s*(?:¥|￥)?\s*([0-9０-９][0-9０-９,，]*)"
        )
        m = pat.search(s)
        if not m:
            return None
        return _norm_amount_to_int(m.group(1))

    def _direct_abs_overrides_for_row(
            raw_color_text: str,
            color_to_pn: Dict[str, str],
            synonym_lookup: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
        针对当前行，直接在 raw_color_text 里按“每个颜色的别名 -> 紧随其后的数字”扫描，
        得到 per-color 的绝对价覆盖表：{color_norm: amount_yen}。
        只接受金额 >= ABS_MIN_YEN，避免把 -500 / 500 之类 delta 当成 abs。
        """
        overrides: Dict[str, int] = {}
        if not raw_color_text:
            return overrides

        s = str(raw_color_text)
        for col_norm in color_to_pn.keys():
            # 构建该颜色的别名集合：自身 + 同义词
            aliases = {col_norm}
            for syn in synonym_lookup.get(col_norm, []):
                aliases.add(str(syn).strip())
            amt_for_color: Optional[int] = None
            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                val = _extract_amount_after_alias(s, alias)
                if val is not None and val >= ABS_MIN_YEN:
                    amt_for_color = val
                    break
            if amt_for_color is not None:
                overrides[col_norm] = int(amt_for_color)

        return overrides

    # 回退版 ABS / DELTA（修正：允许 “青,銀195,500” 这种逗号分隔标签进入 labels 再 split）
    ABS_PRICE_RE = re.compile(
        r"(?P<labels>[^0-9０-９¥￥円]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
        re.I,
    )
    DELTA_RE = re.compile(
        r"(?P<labels>[^0-9０-９¥￥円]+?)\s*[：:\-]?\s*(?P<sign>[+\-−－])\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
        re.I,
    )

    def _extract_abs_prices_regex(text: str) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        for m in ABS_PRICE_RE.finditer(s):
            labels_part = (m.group("labels") or "").strip()
            amt = _norm_amount_to_int(m.group("amount"))
            if amt is None:
                continue
            toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
            for tok in toks:
                if _is_pure_number_token(tok):
                    continue
                out.append((tok, int(amt)))
        return out

    def _extract_deltas_regex(text: str) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        if not text:
            return out
        s = str(text)
        for m in DELTA_RE.finditer(s):
            labels_part = m.group("labels") or ""
            sign = m.group("sign") or "+"
            amt = _norm_amount_to_int(m.group("amount"))
            if amt is None:
                continue
            delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
            toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
            for tok in toks:
                if _is_pure_number_token(tok):
                    continue
                out.append((tok, delta))
        if not out and "全色" in s:
            out.append(("全色", 0))
        return out

    # =============== 构建 pn_map（原逻辑） ===============
    info_df = info_df.copy()
    info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
    info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")
    info_df = info_df.dropna(subset=["model_name_norm", "capacity_gb", "part_number", "color"])

    pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    for _, r in info_df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        pn = str(r["part_number"]).strip()
        col = _norm(r["color"])
        if pd.isna(cap) or not m or not col:
            continue
        key = (m, int(cap))
        pn_map.setdefault(key, {})
        pn_map[key][col] = pn

    # =============== process rows ===============
    model_norm = df[col_model].map(_normalize_model_generic)
    cap_gb     = df[col_model].map(_parse_capacity_gb)
    recorded_at = df[col_time].map(lambda x: parse_dt_aware(x))

    rows = []
    for i in range(len(df)):
        raw_model = df[col_model].iat[i]
        m = model_norm.iat[i]
        c = cap_gb.iat[i]
        t = recorded_at.iat[i]
        raw_price_cell = df[col_price].iat[i]
        raw_color_cell = df[col_color].iat[i]
        print("|| || || || || || || || ||")
        print("\/ \/ \/ \/ \/ \/ \/ \/ \/")
        _dprint(i, f"[DEBUG row={i}] raw_model={raw_model!r} -> norm={m!r}, cap={c!r}, raw_price={raw_price_cell!r},     raw_color = {raw_color_cell!r}")

        if not m or pd.isna(c):
            _dprint(i, f"[DEBUG row={i}] skip: model/cap missing")
            continue
        c = int(c)

        key = (m, c)
        color_to_pn = pn_map.get(key)
        if not color_to_pn:
            _dprint(i, f"[DEBUG row={i}] skip: no pn_map for key={key}")
            continue

        s_color = str(raw_color_cell) if raw_color_cell is not None else ""
        s_price = str(raw_price_cell) if raw_price_cell is not None else ""

        # base price：优先 price 列，其次 color 列（保留你原逻辑）
        base_price = to_int_yen(s_price) or to_int_yen(s_color)

        # 1) 先用 LLM 抽取 abs/delta（核心改动）
        abs_map: Dict[str, int] = {}
        delta_map: Dict[str, int] = {}
        avail_colors_key = tuple(color_to_pn.keys())

        if USE_LLM:
            abs_map, delta_map = _llm_extract_rules_cached(s_price, s_color, avail_colors_key)

        # 2) 若 LLM 没抽到且你希望兜底，则用 regex 回退
        if (not abs_map and not delta_map) and (not USE_LLM or os.getenv("SHOP9_ALLOW_REGEX_FALLBACK", "1") not in {"0","false","False"}):
            abs_list = _extract_abs_prices_regex(s_color) or _extract_abs_prices_regex(s_price)
            deltas   = _extract_deltas_regex(s_color) or _extract_deltas_regex(s_price)

            # 把回退结果映射到 abs_map/delta_map（沿用你原先的宽松匹配逻辑）
            def _match_label_to_colnorm(tok: str) -> Optional[str]:
                if not tok:
                    return None
                tok_norm = _norm(tok)
                for col_norm in color_to_pn.keys():
                    if tok_norm == col_norm:
                        return col_norm
                candidates = set()
                if tok_norm in SYNONYM_LOOKUP:
                    candidates.update(SYNONYM_LOOKUP[tok_norm])
                candidates.add(tok_norm)
                for cand in candidates:
                    candn = _norm(cand)
                    for col_norm in color_to_pn.keys():
                        cn = _norm(col_norm)
                        if candn == cn or candn in cn or cn in candn:
                            return col_norm
                tok_short = re.sub(r"[\s\u3000\-]+", "", tok_norm)
                for col_norm in color_to_pn.keys():
                    cn_short = re.sub(r"[\s\u3000\-]+", "", _norm(col_norm))
                    if tok_short and (tok_short in cn_short or cn_short in tok_short):
                        return col_norm
                return None

            for label_raw, amt in abs_list:
                toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
                for tok in toks:
                    if _is_pure_number_token(tok):
                        continue
                    matched = _match_label_to_colnorm(tok)
                    if matched:
                        abs_map[matched] = int(amt)

            for label_raw, delta in deltas:
                if label_raw == "全色":
                    delta_map["ALL"] = int(delta)
                    continue
                toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
                for tok in toks:
                    if _is_pure_number_token(tok):
                        continue
                    matched = _match_label_to_colnorm(tok)
                    if matched:
                        delta_map[matched] = int(delta)

        _dprint(i, f"[DEBUG row={i}] llm/regex abs_map={abs_map}, delta_map={delta_map}, base_price={base_price}")
        print("/\ /\ /\ /\ /\ /\ /\ /\ /\ ")
        print("|| || || || || || || || ||")

        # ---- 关键新增：用原始 raw_color 文本对 abs_map 做“颜色级别”的覆盖修正 ----
        overrides = _direct_abs_overrides_for_row(
            raw_color_text=s_color,
            color_to_pn=color_to_pn,
            synonym_lookup=SYNONYM_LOOKUP,
        )
        if overrides:
            for col_norm, v in overrides.items():
                abs_map[col_norm] = int(v)
            # _dprint(i, f"[DEBUG row={i}] overrides_from_text={overrides}")
        # =============== 输出生成逻辑（扩展：支持 abs_map['ALL']） ===============
        if "ALL" in delta_map:
            if base_price is None:
                _dprint(i, f"[DEBUG row={i}] ALL delta present but base missing -> skip")
                continue
            final = int(base_price + delta_map["ALL"])
            for col_norm, pn in color_to_pn.items():
                rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(final), "recorded_at": t})
            continue

        if "ALL" in abs_map:
            final = int(abs_map["ALL"])
            for col_norm, pn in color_to_pn.items():
                rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": final, "recorded_at": t})
            continue

        if abs_map:
            for col_norm, pn in color_to_pn.items():
                if col_norm in abs_map:
                    rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(abs_map[col_norm]), "recorded_at": t})
                else:
                    if base_price is not None:
                        rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(base_price), "recorded_at": t})
            continue

        if base_price is None:
            _dprint(i, f"[DEBUG row={i}] no base/abs -> skip")
            continue

        for col_norm, pn in color_to_pn.items():
            delta = int(delta_map.get(col_norm, 0))
            rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(base_price + delta), "recorded_at": t})

    out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
    if not out.empty:
        out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
        out["part_number"] = out["part_number"].astype(str)
        out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
    return out



#
# def clean_shop9(df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
#     import time
#     print("shop9:アキモバ---------->进入清洗器时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#
#     info_df = _load_iphone17_info_df_for_shop2()
#
#     col_model = "機種名"
#     col_price = "買取価格"
#     col_color = "色・詳細等"
#     col_time  = "time-scraped"
#
#     for need in (col_model, col_price, col_color, col_time):
#         if need not in df.columns:
#             raise ValueError(f"shop9 清洗器缺少必要列：{need}")
#
#     # DEBUG: 仅打印“疑似包含颜色/减价/分颜色报价”的行，便于对比原文与抽取结果
#     debug_pos_set = set()
#     if debug:
#         COLOR_PAT = re.compile(
#             r"(ブラック|ホワイト|ブルー|グリーン|ピンク|レッド|イエロー|パープル|オレンジ|"
#             r"シルバー|ゴールド|グラファイト|ミッドナイト|スターライト|ナチュラル|"
#             r"チタニウム|チタン|黒|白|青|緑|赤|黄|紫|橙|銀|金|"
#             r"Black|White|Blue|Green|Pink|Red|Yellow|Purple|Orange|Silver|Gold|Titanium)",
#             re.I,
#         )
#         DISCOUNT_PAT = re.compile(r"(値下げ|値引|割引|円引|OFF|オフ|[+＋\-−–－]\s*[０-９0-9])", re.I)
#         ABS_PRICE_PAT = re.compile(r"(?:¥|￥)?\s*[０-９0-9]{2,3}(?:[０-９0-9,，]{3,})\s*(?:円)?")
#
#         s_color_ser = df[col_color].fillna("").astype(str)
#         s_price_ser = df[col_price].fillna("").astype(str)
#         has_color = s_color_ser.str.contains(COLOR_PAT, na=False) | s_price_ser.str.contains(COLOR_PAT, na=False)
#         has_discount = s_color_ser.str.contains(DISCOUNT_PAT, na=False) | s_price_ser.str.contains(DISCOUNT_PAT, na=False)
#         has_abs_price = s_color_ser.str.contains(ABS_PRICE_PAT, na=False) | s_price_ser.str.contains(ABS_PRICE_PAT, na=False)
#         mask = has_color & (has_discount | has_abs_price)
#
#         hit_cnt = 0
#         for pos, hit in enumerate(mask.to_numpy()):
#             if hit:
#                 debug_pos_set.add(pos)
#                 hit_cnt += 1
#                 if hit_cnt >= int(debug_limit):
#                     break
#         if not debug_pos_set:
#             # 没有命中时，退化为打印前 debug_limit 行
#             debug_pos_set = set(range(min(int(debug_limit), len(df))))
#         print(f"[shop9 debug] total_rows={len(df)}, hit_rows={int(mask.sum())}, print_rows={len(debug_pos_set)}")
#
#     def _dbg_on(row_pos: int) -> bool:
#         return bool(debug) and (row_pos in debug_pos_set)
#
#     def _dprint(row_pos: int, *args, **kwargs):
#         if _dbg_on(row_pos):
#             print(*args, **kwargs)
#
#     # 同义表（可扩充）
#     FAMILY_SYNONYMS_SHOP9 = {
#         "blue": ["ブルー", "青", "ディープブルー", "ディープ ブルー"],
#         "ブルー": ["ブルー", "青", "ディープブルー"],
#         "青": ["ブルー", "青", "ディープブルー"],
#         "ディープブルー": ["ディープブルー", "ブルー", "青"],
#         "silver": ["シルバー", "銀"],
#         "シルバー": ["シルバー", "銀"],
#         "銀": ["シルバー", "銀"],
#         "black": ["ブラック", "黒"],
#         "ブラック": ["ブラック", "黒"],
#         "黒": ["ブラック", "黒"],
#         "orange": ["オレンジ", "橙", "コズミックオレンジ"],
#         "オレンジ": ["オレンジ", "橙"],
#         "橙": ["オレンジ", "橙"],
#         "white": ["ホワイト", "白"],
#         "ホワイト": ["ホワイト", "白"],
#     }
#     SYNONYM_LOOKUP = {}
#     for k, vs in FAMILY_SYNONYMS_SHOP9.items():
#         SYNONYM_LOOKUP[k] = list(dict.fromkeys(vs))
#         for v in vs:
#             SYNONYM_LOOKUP.setdefault(v, [])
#             SYNONYM_LOOKUP[v] = list(dict.fromkeys(SYNONYM_LOOKUP[v] + vs + [k]))
#
#     def _norm(s: str) -> str:
#         if s is None:
#             return ""
#         t = str(s).strip().lower()
#         t = t.replace("　", " ")
#         t = re.sub(r"\s+", " ", t)
#         # 全角数字转半角
#         t = t.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
#         return t
#
#     def _is_pure_number_token(tok: str) -> bool:
#         if not tok:
#             return False
#         t = _norm(tok)
#         t = t.replace(",", "").replace("，", "")
#         return bool(re.fullmatch(r"[0-9]+", t))
#
#     SPLIT_SEPS = r"[/／、，,;；:：\s]+"
#
#     # 绝对价：例如 "青 229,000円" / "ブルー：229000" 等
#     ABS_PRICE_RE = re.compile(
#         # labels 允许包含 '/', ',' 等分隔符；只要不包含数字/货币符号/円
#      r"(?P<labels>[^0-9０-９¥￥円]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
#         re.I,
#     )
#     # 颜色加减：例如 "黒 -2,000円" / "青:+1000" / "全色 -500"
#     DELTA_RE = re.compile(
#         r"(?P<labels>[^0-9０-９¥￥円]+?)\s*[：:\-]?\s*(?P<sign>[+\-−－])\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)\s*(?:円)?",
#         re.I,
#     )
#
#     def _norm_amount_to_int(x: str) -> Optional[int]:
#         if not x:
#             return None
#         s = str(x).strip()
#         s = s.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
#         s = s.replace(",", "")
#         if not s.isdigit():
#             return None
#         return int(s)
#
#     def _extract_abs_prices(text: str) -> List[Tuple[str, int]]:
#         out: List[Tuple[str, int]] = []
#         if not text:
#             return out
#         s = str(text)
#         for m in ABS_PRICE_RE.finditer(s):
#             labels_part = (m.group("labels") or "").strip()
#             amt = _norm_amount_to_int(m.group("amount"))
#             if amt is None:
#                 continue
#             toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
#             for tok in toks:
#                 if _is_pure_number_token(tok):
#                     continue
#                 out.append((tok, int(amt)))
#         # fallback: '青 229,000' 等
#         if not out:
#             m2 = re.finditer(
#                 r"(?P<label>[^\d¥￥円/、，,;；]+?)\s*(?:¥|￥)?\s*(?P<amount>[０-９0-9][０-９0-9,，]*)",
#                 s,
#             )
#             for m in m2:
#                 label = m.group("label").strip()
#                 amt = _norm_amount_to_int(m.group("amount"))
#                 if label and amt is not None and not _is_pure_number_token(label):
#                     out.append((label, int(amt)))
#         return out
#
#     def _extract_deltas(text: str) -> List[Tuple[str, int]]:
#         out: List[Tuple[str, int]] = []
#         if not text:
#             return out
#         s = str(text)
#         for m in DELTA_RE.finditer(s):
#             labels_part = m.group("labels") or ""
#             sign = m.group("sign") or "+"
#             amt_txt = m.group("amount")
#             amt = _norm_amount_to_int(amt_txt)
#             if amt is None:
#                 continue
#             delta = -int(amt) if sign in ("-", "−", "－") else int(amt)
#             toks = [t.strip() for t in re.split(SPLIT_SEPS, labels_part) if t.strip()]
#             for tok in toks:
#                 if _is_pure_number_token(tok):
#                     continue
#                 out.append((tok, delta))
#         # 全色 fallback
#         if not out and "全色" in s:
#             m = re.search(r"全色\s*[：:\-]?\s*([+\-−－])?\s*([０-９0-9][０-９0-9,，]*)", s)
#             if m:
#                 sign = m.group(1) or "+"
#                 amt = _norm_amount_to_int(m.group(2))
#                 if amt is None:
#                     amt = 0
#                 out.append(("全色", -amt if sign in ("-", "−", "－") else amt))
#             else:
#                 out.append(("全色", 0))
#         return out
#
#     # 构建 pn_map: (model_norm, cap_gb) -> {color_norm: part_number}
#     info_df = info_df.copy()
#     info_df["model_name_norm"] = info_df["model_name"].map(_normalize_model_generic)
#     info_df["capacity_gb"] = pd.to_numeric(info_df["capacity_gb"], errors="coerce").astype("Int64")
#     info_df = info_df.dropna(subset=["model_name_norm", "capacity_gb", "part_number", "color"])
#
#     pn_map: Dict[Tuple[str, int], Dict[str, str]] = {}
#     for _, r in info_df.iterrows():
#         m = r["model_name_norm"]
#         cap = r["capacity_gb"]
#         pn = str(r["part_number"]).strip()
#         col = _norm(r["color"])
#         if pd.isna(cap) or not m or not col:
#             continue
#         key = (m, int(cap))
#         pn_map.setdefault(key, {})
#         pn_map[key][col] = pn
#
#     # process rows
#     model_norm = df[col_model].map(_normalize_model_generic)
#     cap_gb     = df[col_model].map(_parse_capacity_gb)
#     recorded_at = df[col_time].map(lambda x: parse_dt_aware(x))
#
#     rows = []
#     for i in range(len(df)):
#         raw_model = df[col_model].iat[i]
#         m = model_norm.iat[i]
#         c = cap_gb.iat[i]
#         t = recorded_at.iat[i]
#         raw_price_cell = df[col_price].iat[i]
#         raw_color_cell = df[col_color].iat[i]
#
#         _dprint(i, f"[DEBUG row={i}] raw_model={raw_model!r} -> norm={m!r}, cap={c!r}, raw_price={raw_price_cell!r}, raw_color={raw_color_cell!r}")
#
#         if not m or pd.isna(c):
#             _dprint(i, f"[DEBUG row={i}] skip: model/cap missing")
#             continue
#         c = int(c)
#
#         key = (m, c)
#         color_to_pn = pn_map.get(key)
#         if not color_to_pn:
#             _dprint(i, f"[DEBUG row={i}] skip: no pn_map for key={key}")
#             continue
#
#         s_color = str(raw_color_cell) if raw_color_cell is not None else ""
#         s_price = str(raw_price_cell) if raw_price_cell is not None else ""
#         # parse from color-col first (优先)
#         abs_list = _extract_abs_prices(s_color)
#         deltas = _extract_deltas(s_color)
#         base_price = to_int_yen(s_price) or to_int_yen(s_color)
#
#         # if not found in color-col, try price-col
#         if not abs_list and not deltas:
#             abs_list = _extract_abs_prices(s_price)
#             deltas = _extract_deltas(s_price)
#             if base_price is None:
#                 base_price = to_int_yen(s_price)
#
#         # final fallback: whole row
#         if not abs_list and not deltas:
#             full_row_parts = []
#             for col in df.columns:
#                 try:
#                     v = df[col].iat[i]
#                 except Exception:
#                     v = df.iloc[i].get(col)
#                 if v is None:
#                     continue
#                 sv = str(v).strip()
#                 if sv and sv.lower() != "nan":
#                     full_row_parts.append(sv)
#             s_full = " ".join(full_row_parts)
#             if s_full and s_full != s_color and s_full != s_price:
#                 _dprint(i, f"[DEBUG row={i}] fallback parsing from full row: {s_full!r}")
#                 abs_list = _extract_abs_prices(s_full)
#                 deltas = _extract_deltas(s_full)
#                 if base_price is None:
#                     base_price = to_int_yen(s_full)
#
#         _dprint(i, f"[DEBUG row={i}] parsed abs_list={abs_list}, deltas={deltas}, base_price={base_price}")
#
#         # label -> col_norm matching（宽松 + 同义表）
#         def _match_label_to_colnorm(tok: str) -> Optional[str]:
#             if not tok:
#                 return None
#             tok_norm = _norm(tok)
#             # direct equal
#             for col_norm in color_to_pn.keys():
#                 if tok_norm == col_norm:
#                     return col_norm
#             # synonyms
#             candidates = set()
#             if tok_norm in SYNONYM_LOOKUP:
#                 candidates.update(SYNONYM_LOOKUP[tok_norm])
#             candidates.add(tok_norm)
#             for cand in candidates:
#                 for col_norm in color_to_pn.keys():
#                     if cand == col_norm or cand in col_norm or col_norm in cand:
#                         return col_norm
#             # fallback substring
#             tok_short = re.sub(r"[\s\u3000\-]+", "", tok_norm)
#             for col_norm in color_to_pn.keys():
#                 if tok_short in col_norm or col_norm in tok_short:
#                     return col_norm
#             return None
#
#         color_abs_map: Dict[str, int] = {}
#         color_delta_map: Dict[str, int] = {}
#
#         for label_raw, amt in abs_list:
#             toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
#             for tok in toks:
#                 if _is_pure_number_token(tok):
#                     _dprint(i, f"[DEBUG row={i}] abs skip numeric token={tok!r}")
#                     continue
#                 matched = _match_label_to_colnorm(tok)
#                 if matched:
#                     color_abs_map[matched] = int(amt)
#                     _dprint(i, f"[DEBUG row={i}] abs match: token={tok!r} -> color_norm={matched!r} price={amt}")
#                 else:
#                     _dprint(i, f"[DEBUG row={i}] abs NO-match token={tok!r}")
#
#         for label_raw, delta in deltas:
#             if label_raw == "全色":
#                 color_delta_map["ALL"] = int(delta)
#                 _dprint(i, f"[DEBUG row={i}] delta ALL -> {delta}")
#                 continue
#             toks = [t.strip() for t in re.split(SPLIT_SEPS, label_raw) if t.strip()]
#             for tok in toks:
#                 if _is_pure_number_token(tok):
#                     _dprint(i, f"[DEBUG row={i}] delta skip numeric token={tok!r}")
#                     continue
#                 matched = _match_label_to_colnorm(tok)
#                 if matched:
#                     color_delta_map[matched] = int(delta)
#                     _dprint(i, f"[DEBUG row={i}] delta match: token={tok!r} -> color_norm={matched!r} delta={delta}")
#                 else:
#                     _dprint(i, f"[DEBUG row={i}] delta NO-match token={tok!r}")
#
#         _dprint(i, f"[DEBUG row={i}] mapped_abs={color_abs_map}, mapped_deltas={color_delta_map}")
#
#         # 输出生成逻辑：ALL -> ABS -> delta/base
#         if "ALL" in color_delta_map:
#             if base_price is None:
#                 _dprint(i, f"[DEBUG row={i}] ALL present but base price missing -> skip")
#                 continue
#             final = int(base_price + color_delta_map["ALL"])
#             for col_norm, pn in color_to_pn.items():
#                 rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(final), "recorded_at": t})
#                 _dprint(i, f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={final} reason=ALL base={base_price} delta={color_delta_map['ALL']}")
#             continue
#
#         if color_abs_map:
#             for col_norm, pn in color_to_pn.items():
#                 if col_norm in color_abs_map:
#                     val = color_abs_map[col_norm]
#                     rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(val), "recorded_at": t})
#                     _dprint(i, f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={val} reason=ABS abs={val}")
#                 else:
#                     if base_price is not None:
#                         rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": int(base_price), "recorded_at": t})
#                         _dprint(i, f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={base_price} reason=BASE-FALLBACK base={base_price}")
#                     else:
#                         _dprint(i, f"[DEBUG row={i}] -> color={col_norm} pn={pn} skipped (no abs, no base)")
#             continue
#
#         if base_price is None:
#             _dprint(i, f"[DEBUG row={i}] no base/abs -> skip")
#             continue
#
#         for col_norm, pn in color_to_pn.items():
#             delta = color_delta_map.get(col_norm, 0)
#             val = int(base_price + delta)
#             rows.append({"part_number": pn, "shop_name": "アキモバ", "price_new": val, "recorded_at": t})
#             _dprint(i, f"[DEBUG row={i}] -> color={col_norm} pn={pn} price={val} reason={'BASE+DELTA' if delta else 'BASE'} base={base_price} delta={delta}")
#
#     out = pd.DataFrame(rows, columns=["part_number","shop_name","price_new","recorded_at"])
#     if not out.empty:
#         out = out.dropna(subset=["part_number","price_new"]).reset_index(drop=True)
#         out["part_number"] = out["part_number"].astype(str)
#         out["price_new"] = pd.to_numeric(out["price_new"], errors="coerce").astype("Int64")
#     return out
