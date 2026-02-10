from __future__ import annotations
from typing import Protocol, Dict, Callable, Optional, List, Tuple, Iterable, Union
from ...external_ingest.helpers import to_int_yen, parse_dt_aware
from ..cleaner_tools import _parse_capacity_gb, _load_iphone17_info_df_from_db
import os
from functools import lru_cache
from pathlib import Path
import re
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
import pytz
import time
import json
import textwrap

# LangExtract + Ollama (本地 LLM) 集成 --------------------------
try:
    import langextract as lx
except Exception:
    lx = None  # 没装或运行失败时，自动退回正则版本

_LANGEXTRACT_MODEL_ID = "gemma3:1b"         # 你在 Ollama 里使用的模型名
_LANGEXTRACT_MODEL_URL = "http://localhost:11434"

# ----------------------------------------------------------------

_YEN_RE = re.compile(r"[^\d]+")

def _parse_yen(val) -> int | None:
    """'¥177,000' / '177,000円' / '177000' -> 177000"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = _YEN_RE.sub("", s)
    if not s:
        return None
    try:
        n = int(s)
        return n
    except Exception:
        return None

def _norm(s: str) -> str:
    return (s or "").strip()

def _norm_model_token(s: str) -> str:
    """
    将 data2-1 的机型片段“宽松”规范化（仅用于和 iphone17_info 里的 model_name 做宽松匹配）
    规则：小写、去空格、去多余符号
    """
    s = (s or "").lower()
    s = re.sub(r"iphone\s*", "iphone ", s)
    s = re.sub(r"[^0-9a-z\s+]", "", s)  # 仅保留 a-z0-9 和空格
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_model_name_loose(model_token: str, iphone17_df: pd.DataFrame) -> str | None:
    """
    宽松匹配：在 iphone17_df['model_name'] 中选与 token 最匹配的项（不严格 Fuzzy，先简单包含匹配）
    """
    token = _norm_model_token(model_token)
    if not token:
        return None
    # 候选（去重）
    candidates = list(
        dict.fromkeys([_norm(x) for x in iphone17_df["model_name"].dropna().tolist()])
    )

    def norm_m(m: str) -> str:
        return _norm_model_token(m)

    # 简单策略：同样规范化后，包含则命中
    hits = [m for m in candidates if token in norm_m(m) or norm_m(m) in token]
    if len(hits) == 1:
        return hits[0]
    # 多命中时偏向更长的 model_name（更具体）
    if hits:
        return sorted(hits, key=lambda m: len(m), reverse=True)[0]
    return None

# ---------------------- 颜色组匹配逻辑（黒 -> ブラック 等） ----------------------

def _match_color_group(group: str, color_name: str) -> tuple[bool, str]:
    """
    返回 (is_match, reason)
    集中管理 data5 里“组名”到实际颜色名的映射规则。
    """
    g = (group or "").strip()
    c = color_name or ""

    # 常见后缀清理：青系 / 橙色 / 黒色 等
    # （避免 LLM 或简单解析把“色/系”也带进 group_label）
    g = re.sub(r"(系|色)$", "", g).strip()

    # Blue 系
    if g in ("青", "ブルー", "ミストブルー", "ディープブルー", "スカイブルー"):
        return ("ブルー" in c), "contains ブルー"

    # Silver 系
    if g in ("銀", "シルバー"):
        return ("シルバー" in c), "contains シルバー"

    # Black 系 —— 黒 也命中 スペースブラック
    if g in ("黒", "ブラック"):
        return (
            ("ブラック" in c) or ("黒" in c) or ("ミッドナイト" in c),
            "contains ブラック/黒/ミッドナイト",
        )

    # White 系（如果你不希望“白”覆盖银色，可以去掉 "シルバー"）
    if g in ("白", "ホワイト"):
        return (
            ("ホワイト" in c) or ("白" in c) or ("シルバー" in c),
            "contains ホワイト/白/シルバー",
        )

    # Gold 系（可选）
    if g in ("金", "ゴールド"):
        return ("ゴールド" in c), "contains ゴールド"

    # ★ Orange 系 —— 修复点：橙 -> オレンジ（コズミックオレンジ 等）
    if g in ("橙", "オレンジ"):
        return (
            ("オレンジ" in c) or ("橙" in c),
            "contains オレンジ/橙",
        )

    # Fallback: 允许 data5 里直接写具体颜色（例如 “コズミックオレンジ-3000”）
    if g:
        return (g in c), "substring match"

    return False, ""

def _apply_adjust_with_trace(color_name: str, rules: dict) -> tuple[int, list[dict]]:
    """
    返回：(adjust_sum, trace)
    trace 项示例：{"group":"青","delta":-1000,"reason":"contains ブルー"}
    """
    c = color_name or ""
    adjust = 0
    trace: list[dict] = []
    for group, delta in (rules or {}).items():
        ok, reason = _match_color_group(group, c)
        if ok:
            adjust += int(delta)
            trace.append(
                {"group": (group or "").strip(), "delta": int(delta), "reason": reason}
            )
    return adjust, trace

# ---------------------- 原来的正则版本（备用 & Fallback） ----------------------

def _parse_adjust_rule_regex(val) -> dict:
    """
    旧版正则解析（fallback）。现在允许 val 为任意类型。
    """
    s = _as_text(val)
    rules = {}
    if not s:
        return rules

    parts = re.split(r"\+{1,3}|[,、，\s]+", s)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"(.+?)-(\d+)", p)
        if not m:
            continue
        group = m.group(1).strip()
        amt = -int(m.group(2))
        rules[group] = amt
    return rules

def _as_text(val) -> str:
    """
    把 data5 这种可能为 NaN/None/数字/字符串 的输入，统一规范成可解析的字符串。
    """
    if val is None:
        return ""

    # pandas 的 NaN / NA
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass

    s = str(val).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

# ---------------------- LangExtract + Ollama 版本的规则解析 ----------------------

if lx is not None:
    # 提示：如何从 data5 文本里抽取 color_rule
    _COLOR_RULE_PROMPT = textwrap.dedent(
        """\
        あなたは中古スマホ買取表の「色ごとの減額条件」を解析するツールです。
        入力は data5 列に入っている短い日本語テキストです。例:
        - "青-1000"
        - "銀-5000+++青-3000"
        - "青-1000円\n※開封品 ¥183,000"
        など、色名と金額（減額/増額）が混在して書かれています。

        タスク:
        - data5 の中から「色グループ」と「基準価格からの差額（円）」をすべて抽出してください。
        - 減額は負の値、増額は正の値とします。
        - 抽出対象は、基準価格(data3)に対する相対額だけです。開封品価格など他の情報は無視してください。

        出力スキーマ:
        - 抽出するエンティティはすべて extraction_class="color_rule" とします。
        - 各 color_rule の attributes には次のキーを入れてください:
          - "group_label": 文字列。元テキスト中の色グループ名（例: "青", "銀", "スペースブラック", "全色"）
          - "delta_yen": 整数。基準価格からの差額（円）。減額は負の値、増額は正の値。

        注意:
        - "青-1000" や "銀-5000" のような書き方は「基準価格から 1000 円/5000 円減額」を意味します。
        - "青+2000" のような表現があれば、それは「基準価格から 2000 円増額」です。
        - テキストの中に色の情報がなく、金額だけの場合は無視してください。
        - 解釈に迷う場合は、その項目を抽出しないでください（安全側）。
        """
    )

    # few-shot 例
    _COLOR_RULE_EXAMPLES: List[lx.data.ExampleData] = [
        lx.data.ExampleData(
            text="青-1000\n※開封品 ¥183,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="青-1000",
                    attributes={"group_label": "青", "delta_yen": -1000},
                )
            ],
        ),
        lx.data.ExampleData(
            text="銀-5000+++青-3000\n※開封品 ¥183,000",
            extractions=[
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="銀-5000",
                    attributes={"group_label": "銀", "delta_yen": -5000},
                ),
                lx.data.Extraction(
                    extraction_class="color_rule",
                    extraction_text="青-3000",
                    attributes={"group_label": "青", "delta_yen": -3000},
                ),
            ],
        ),
    ]
else:
    _COLOR_RULE_PROMPT = ""
    _COLOR_RULE_EXAMPLES = []

@lru_cache(maxsize=1024)
def _parse_adjust_rule_llm(rule_text: str) -> dict:
    s = (rule_text or "").strip()
    if not s:
        return {}

    if lx is None:
        return _parse_adjust_rule_regex(s)

    try:
        result = lx.extract(
            text_or_documents=s,
            prompt_description=_COLOR_RULE_PROMPT,
            examples=_COLOR_RULE_EXAMPLES,
            model_id=_LANGEXTRACT_MODEL_ID,
            model_url=_LANGEXTRACT_MODEL_URL,
            fence_output=False,
            use_schema_constraints=False,
        )

        doc = result.to_dict() if hasattr(result, "to_dict") else json.loads(
            json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o)))
        )

        rules: dict[str, int] = {}

        for ext in doc.get("extractions", []) or []:
            attrs = ext.get("attributes") or {}
            extraction_text = _as_text(ext.get("extraction_text"))

            # 1) 优先从 extraction_text 按行解析（更贴近原文，且可一次吃掉多条）
            if extraction_text:
                for piece in extraction_text.replace("\r", "\n").split("\n"):
                    parsed = _parse_rule_token_simple(piece)
                    if parsed:
                        g, d = parsed
                        rules[g] = d

            # 2) 再用 attributes 兜底（处理 extraction_text 不含金额的情况）
            group_label = _as_text(attrs.get("group_label"))
            delta = _coerce_int(attrs.get("delta_yen"))
            if group_label and (delta is not None):
                rules[group_label] = int(delta)

        # LLM 一条都没解析出来就回退
        if not rules:
            return _parse_adjust_rule_regex(s)

        return rules

    except Exception:
        return _parse_adjust_rule_regex(s)
_INT_RE = re.compile(r"[+-]?\d+")

def _coerce_int(val) -> Optional[int]:
    """把 int/float/str 的数字（含 '円'、'¥'、逗号、全角符号）稳健转成 int。"""
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

_SIGN_MINUS = {"-", "−", "－", "–", "—", "―"}
_SIGN_PLUS = {"+", "＋"}

def _parse_rule_token_simple(token: str) -> Optional[Tuple[str, int]]:
    """
    解析单条规则 token，例如：
      '黒-2000' / '青-2000円' / '銀 +3000' -> ('黒', -2000) / ('青', -2000) / ('銀', 3000)
    """
    s = _as_text(token)
    if not s:
        return None

    # 从末尾找数字串
    i = len(s) - 1
    while i >= 0 and not s[i].isdigit():
        i -= 1
    if i < 0:
        return None

    j = i
    while j >= 0 and s[j].isdigit():
        j -= 1
    num_str = s[j + 1 : i + 1]
    if not num_str:
        return None

    # 数字前找 +/- 符号（允许中间有空格）
    k = j
    while k >= 0 and s[k].isspace():
        k -= 1
    if k < 0:
        return None

    sign_ch = s[k]
    if sign_ch in _SIGN_PLUS:
        sign = 1
    elif sign_ch in _SIGN_MINUS:
        sign = -1
    else:
        return None

    group = s[:k].strip().strip(" :：\t")
    if not group:
        return None

    return group, sign * int(num_str)

def _parse_adjust_rule_simple_all(val) -> dict:
    """
    对原始 data5 做一次“保守补全解析”：
    - 只按分隔符拆开（换行/+++ / + / 逗号等），逐段用 _parse_rule_token_simple 解析
    - 用于补齐 LLM 漏掉的规则；不覆盖 LLM 已解析到的 key
    """
    s = _as_text(val)
    if not s:
        return {}

    t = s
    for rep in ("+++", "++", "+", "＋＋＋", "＋＋", "＋", "\r"):
        t = t.replace(rep, "\n")
    for sep in ("、", "，", ","):
        t = t.replace(sep, "\n")

    rules: dict[str, int] = {}
    for line in t.splitlines():
        parsed = _parse_rule_token_simple(line)
        if parsed:
            g, d = parsed
            rules[g] = d
    return rules

def _parse_adjust_rule(val) -> dict:
    s = _as_text(val)
    if not s:
        return {}

    llm_rules = _parse_adjust_rule_llm(s)
    supplement = _parse_adjust_rule_simple_all(s)

    merged = dict(llm_rules or {})
    for k, v in supplement.items():
        merged.setdefault(k, v)
    return merged

# ---------------------- 机型信息加载 / 容量解析 ----------------------

def clean_shop2(shop2_df: pd.DataFrame, debug: bool = True, debug_limit: int = 30) -> pd.DataFrame:
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"shop2:海峡通信---------->进入清洗器时间: {now}")

    """
    debug=True 时：仅对疑似含“颜色减价规则(data5)”的行输出对照信息，用于核对：
      - data5 的潜在颜色减价
      - 实际抽出的每个颜色 price_new
    LangExtract + Ollama(gemma3:1b) 用于解析 data5 中的颜色减价规则。
    """
    SHOP = "海峡通信"

    # 统一列名（小写）
    df = shop2_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 必要列存在性检查（若缺则补 None，保持兼容）
    need_cols = ["data2-1", "data2-2", "data3", "data5", "time-scraped"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = None

    # 只保留 simfree 未開封
    def _is_target(s: str) -> bool:
        s = (s or "").lower()
        return ("simfree" in s) and ("未開封" in s)

    df = df[df["data2-2"].apply(_is_target)].copy().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(
            columns=["part_number", "shop_name", "price_new", "recorded_at"]
        )

    # iphone17_df 预处理
    info = _load_iphone17_info_df_from_db()
    if "capacity_gb" not in info.columns:
        raise ValueError("iphone17_info.csv 需要包含 capacity_gb 列")
    info["color"] = info["color"].apply(_norm)

    # -------- DEBUG: 选出“疑似颜色减价行”（data5 有规则） --------
    debug_pos_set: set[int] = set()
    if debug:
        # data5 里通常是：青-1000 / 銀-5000+++青-5000 等
        _RULE_PAT = re.compile(r"(青|銀|黒|白|橙|ブルー|シルバー|ブラック|ホワイト|オレンジ).{0,6}-\d+", re.I)
        s5 = df["data5"].fillna("").astype(str)

        mask = s5.str.contains(_RULE_PAT, na=False) | s5.str.contains(
            r"-\d+", na=False
        )
        hit_cnt = 0
        for pos, hit in enumerate(mask.to_numpy()):
            if hit:
                debug_pos_set.add(pos)
                hit_cnt += 1
                if hit_cnt >= int(debug_limit):
                    break

        # 如果没命中任何 “-数字” 规则，则退化为打印前 debug_limit 行
        if not debug_pos_set:
            debug_pos_set = set(range(min(int(debug_limit), len(df))))

        print(
            f"[shop2 debug] total_rows(after_filter)={len(df)}, "
            f"hit_rows={int(mask.sum())}, print_rows={len(debug_pos_set)}"
        )

    def _dbg_on(pos: int) -> bool:
        return bool(debug) and (pos in debug_pos_set)

    def _dprint(pos: int, *args, **kwargs):
        if _dbg_on(pos):
            print(*args, **kwargs)

    out_rows: list[dict] = []

    for pos, row in enumerate(df.to_dict("records")):
        rec_raw = row.get("time-scraped")
        recorded_at = parse_dt_aware(rec_raw)

        raw_modelcap = _norm(row.get("data2-1"))
        raw_targetflag = row.get("data2-2")
        raw_price = row.get("data3")
        raw_rule = row.get("data5")

        # if _dbg_on(pos):
        #     print(f" [shop2 debug] row_pos= {pos}")
        #     # print("\n[shop2 debug] row_pos=", pos)
        #     # print("  data2-2(raw):", repr(raw_targetflag))
        #     print(f" ata2-1(raw)       : {repr(raw_modelcap)}")
        #     # print("  data2-1(raw):", repr(raw_modelcap))
        #     print(f" data3(raw)        : {repr(raw_price)}")
        #     # print("  data3(raw):", repr(raw_price))
        #     print(f" data5(raw)        : {repr(raw_rule)}")
        #     # print("  data5(raw):", repr(raw_rule))
        #     print(f" time-scraped(raw) : {repr(rec_raw)}")
        #     # print("  time-scraped(raw):", repr(rec_raw))

        if not raw_modelcap:
            _dprint(pos, "  SKIP_REASON: data2-1 为空")
            continue

        cap_gb = _parse_capacity_gb(raw_modelcap)
        if not cap_gb:
            _dprint(pos, "  SKIP_REASON: 容量(capacity_gb)解析失败")
            continue

        model_name = _pick_model_name_loose(raw_modelcap, info)
        if not model_name:
            _dprint(pos, "  SKIP_REASON: 机型(model_name)宽松匹配失败")
            continue

        sub = info[
            (info["model_name"] == model_name) & (info["capacity_gb"] == cap_gb)
        ].copy()
        if sub.empty:
            _dprint(
                pos,
                f"  SKIP_REASON: info 中找不到该机型+容量 model={model_name!r}, cap={cap_gb}",
            )
            continue

        base_price = _parse_yen(raw_price)
        if base_price is None:
            _dprint(pos, "  SKIP_REASON: 基础价格(data3)解析失败")
            continue

        # ★★★★★ 这里改成调用 LangExtract 解析颜色减价规则 ★★★★★
        rules = _parse_adjust_rule(raw_rule)

        if _dbg_on(pos):
            print(f" [shop2 debug] row_pos= {pos}")
            print(f" ata2-1(raw)        : {repr(raw_modelcap)}")
            print(f" data3(raw)         : {repr(raw_price)}")
            print(f" data5(raw)         : {repr(raw_rule)}")

            # print("  parsed cap_gb:", cap_gb)
            # print("  matched model_name:", repr(model_name))
            # print("  base_price:", base_price)
            # print("  parsed rules:", rules)
            print(f" parsed rules       : {rules}")
            # print(
            #     "  sub colors:",
            #     sub["color"].dropna().astype(str).unique().tolist(),
            # )

            # 规则命中概览（group -> 命中哪些颜色）
            if rules:
                group_hits = {}
                for g in rules.keys():
                    group_hits[g] = []
                    for c in sub["color"].dropna().astype(str).tolist():
                        adj, tr = _apply_adjust_with_trace(c, {g: rules[g]})
                        if adj != 0:
                            group_hits[g].append(c)
                print(f" rule_hit_overview  : {group_hits}")
                # print("  rule_hit_overview:", group_hits)

        used_groups: set[str] = set()

        for _, it in sub.iterrows():
            part = _norm(it.get("part_number"))
            color = _norm(it.get("color"))
            if not part:
                if _dbg_on(pos):
                    print("  skip item: empty part_number for color=", repr(color))
                continue

            adj, trace = _apply_adjust_with_trace(color, rules)
            for tr in trace:
                used_groups.add(tr["group"])

            price = int(base_price + adj)
            if price <= 0:
                _dprint(
                    pos,
                    f"  SKIP_ITEM_REASON: price<=0 color={color!r} base={base_price} adj={adj}",
                )
                continue

            if _dbg_on(pos):
                print(f" OUT_ITEM--->  color: {color:<10},base: {base_price}, adj: {adj},final: {price},trace: {trace}")
                # print(
                #     "  -> OUT_ITEM:",
                #     {
                #         "color": color,
                #         "part_number": part,
                #         "base": base_price,
                #         "adj": adj,
                #         "final": price,
                #         "trace": trace,
                #     },
                # )

            out_rows.append(
                {
                    "part_number": part,
                    "shop_name": SHOP,
                    "price_new": price,
                    "recorded_at": recorded_at,
                }
            )

        if _dbg_on(pos) and rules:
            unused = [g for g in rules.keys() if g not in used_groups]
            if unused:
                print("  note: rules 未命中任何颜色 ->", unused)

    if not out_rows:
        return pd.DataFrame(
            columns=["part_number", "shop_name", "price_new", "recorded_at"]
        )

    out = pd.DataFrame(
        out_rows, columns=["part_number", "shop_name", "price_new", "recorded_at"]
    )

    # if debug:
    #     print(f"\n[shop2 debug] out_rows={len(out)} head=\n{out.head(10).to_string(index=False)}")

    return out
