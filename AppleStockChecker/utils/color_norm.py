# AppleStockChecker/utils/color_norm.py
from __future__ import annotations
import re
import csv
from typing import Dict, List, Tuple, Iterable, Optional, Any

try:
    import pandas as pd  # 动态加载词表时更方便
except Exception:
    pd = None

# =========================
#  1) “全色/任意” 识别
# =========================
ALL_COLOR_TOKENS = {
    "全色", "全カラー", "全部", "任意", "不限", "任何", "all", "ALL", "any", "any color",
    "カラー問わず", "すべて", "全系", "全機種色", "全機型色"
}

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u3000", " ")).strip()

def is_all_color(text: str | None) -> bool:
    if not text:
        return True
    t = _norm_ws(str(text)).lower()
    return any(tok.lower() in t for tok in ALL_COLOR_TOKENS)

# =======================================
#  2) 内置基础配色（非 Pro 钛金属家族）
# =======================================
# 规范名 → 同义词（全部小写匹配，中/英/日混写尽量覆盖）
BASE_SYNONYMS: Dict[str, List[str]] = {
    "Black":   ["black", "ブラック", "黒", "黑", "黑色", "曜石黑"],
    "White":   ["white", "ホワイト", "白", "白色"],
    "Blue":    ["blue", "ブルー", "青", "蓝", "藍", "蓝色", "藍色", "遠峰藍", "远峰蓝", "天蓝"],
    "Green":   ["green", "グリーン", "緑", "绿", "绿色", "緑色"],
    "Pink":    ["pink", "ピンク", "粉", "粉色", "玫瑰粉"],
    "Yellow":  ["yellow", "イエロー", "黄", "黄色"],
    "Red":     ["red", "レッド", "紅", "红", "红色", "product red", "(product)red", "プロダクトレッド"],
    "Gold":    ["gold", "ゴールド", "金", "金色", "香槟金", "香檳金"],
    "Silver":  ["silver", "シルバー", "銀", "银", "銀色", "银色"],
    "Gray":    ["gray", "grey", "グレー", "グレイ", "灰", "灰色"],
    "Purple":  ["purple", "パープル", "紫", "紫色", "ディープパープル", "深紫"],
    "Graphite": ["graphite", "グラファイト", "石墨", "石墨色"],
    "Space Gray": ["space gray", "スペースグレー", "スペースグレイ", "深空灰", "深空灰色"],
    "Starlight": ["starlight", "スターライト", "星光", "星光色"],
    "Midnight":  ["midnight", "ミッドナイト", "午夜", "午夜色", "暗夜色"],
}

# =======================================
#  3) Pro 钛金属家族：<Color> Titanium
# =======================================
# 颜色词干（基础/扩展）小写 → 规范色名
_TITANIUM_COLOR_CORE: Dict[str, str] = {
    # 已有
    "natural": "Natural Titanium",
    "black":   "Black Titanium",
    "white":   "White Titanium",
    "blue":    "Blue Titanium",
    "desert":  "Desert Titanium",
    # —— iPhone 17（假设/新增常见市场词干；实际以你的清单为准，动态加载会覆盖/补充）——
    "green":   "Green Titanium",
    "purple":  "Purple Titanium",
    "sand":    "Desert Titanium",   # sand 归并到 desert
    "gray":    "Natural Titanium",  # 灰/グレー 常被渠道写作“自然/钛灰”
}

# 将 <任意写法 + チタニウム/Titanium> 规范为 "<Color> Titanium"
_TITANIUM_PAT = re.compile(
    r"(?P<pre>(?:ブラック|ホワイト|ブルー|グリーン|パープル|ナチュラル|サンド|グレー|"
    r"black|white|blue|green|purple|natural|sand|gray|grey))?"
    r"\s*(?:チタニウム|titanium)\s*"
    r"(?P<post>(?:ブラック|ホワイト|ブルー|グリーン|パープル|ナチュラル|サンド|グレー|"
    r"black|white|blue|green|purple|natural|sand|gray|grey))?",
    flags=re.I
)

_JA_TO_CORE = {
    "ブラック": "black", "ホワイト": "white", "ブルー": "blue", "グリーン": "green",
    "パープル": "purple", "ナチュラル": "natural", "サンド": "sand", "グレー": "gray", "グレイ": "gray",
}

# ===============
#  4) 动态词表
# ===============
# 允许把 iPhone 17 清单里的颜色注入到词典
_DYNAMIC_SYNONYMS: Dict[str, List[str]] = {}  # canon -> synonyms (lower)

def register_color(canon: str, *synonyms: str) -> None:
    canon = canon.strip()
    syns = [s.strip().lower() for s in synonyms if s and s.strip()]
    if not syns:
        return
    bucket = _DYNAMIC_SYNONYMS.setdefault(canon, [])
    for s in syns:
        if s not in bucket:
            bucket.append(s)

def load_dynamic_color_synonyms(source: Any, color_columns: Iterable[str] = ("color","颜色","カラー")) -> int:
    """
    从 CSV/Excel/DataFrame 里抽取颜色写法并映射到“可能的” canon：
      - 若值里含 'チタニウム/Titanium'，按钛金属规则规范；
      - 否则尝试映射到 BASE_SYNONYMS 的 canon；
      - 仍无法判断就把原词作为“新 canon”，并把该词加入它自己的同义词（后续可在 DB 层回收/合并）。
    返回注册的条目数（粗略）。
    """
    rows = []  # list[str]
    # 读取
    if pd is not None and isinstance(source, pd.DataFrame):
        df = source
    else:
        # 路径字符串
        path = str(source)
        if path.lower().endswith((".xlsx",".xls",".xlsm",".ods")):
            if pd is None:
                raise RuntimeError("需要 pandas 来解析 Excel")
            df = pd.read_excel(path)
        else:
            # CSV
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                df = pd.read_csv(f)
    # 聚合颜色列
    cols = [c for c in color_columns if c in df.columns]
    if not cols and len(df.columns) > 0:
        # 尝试猜测
        for c in df.columns:
            if re.search(r"(color|颜色|カラー)", str(c), re.I):
                cols.append(c)
        if not cols:
            return 0
    for _, r in df.iterrows():
        for c in cols:
            v = str(r.get(c,"")).strip()
            if v:
                rows.append(v)
    # 归一 & 注册
    count = 0
    for raw in rows:
        _canon, _ = _normalize_titanium_or_base(raw)
        canon = _canon or _guess_base_canon(raw)
        if not canon:
            # 新 canon：直接把自身注册，便于后续可见
            register_color(raw, raw)
            count += 1
        else:
            register_color(canon, raw)
            count += 1
    return count

# ================
#  5) 归一化主过程
# ================
def normalize_color(text: str | None) -> Tuple[str, bool]:
    """
    输入任意描述（中/英/日），返回 (canonical_color, is_all)：
      - 识别 “全色/任意/无颜色” → ("" , True)
      - iPhone Pro 钛金属：规范为 "<Color> Titanium"
      - 其它：映射到基础配色或动态注入配色；若无法识别 → ("", False)
    """
    if not text:
        return ("", True)
    t = _norm_ws(text)
    if is_all_color(t):
        return ("", True)
    # 1) 钛金属家族
    canon, _ok = _normalize_titanium_or_base(t)
    if _ok:
        return (canon, False)
    # 2) 基础配色（含动态注入）
    canon = _guess_base_canon(t)
    if canon:
        return (canon, False)
    # 3) 模糊兜底（尽量避免误判）
    canon = _fuzzy_guess(t)
    return (canon or "", False)

def synonyms_for_query(canon: str) -> List[str]:
    """
    给 DB 检索用的同义词集合（小写；建议用于 icontains/iexact 组合查询）
    """
    canon_l = canon.lower()
    out = {canon_l}
    for dic in (BASE_SYNONYMS, _DYNAMIC_SYNONYMS):
        if canon in dic:
            out.update(s.lower() for s in dic[canon])
    # 钛金属：附带 "<color> titanium" 的英语小写形式，以容错
    if "titanium" in canon_l:
        out.add(canon_l)
        # “<color> チタニウム” 的日文容错
        pre = canon_l.replace(" titanium","")
        out.add(pre + " チタニウム")
    return sorted(out)

# ----------------------------------------
# 内部：钛金属规范化 & 基础色猜测 & 模糊兜底
# ----------------------------------------
def _normalize_titanium_or_base(text: str) -> Tuple[str, bool]:
    """
    若 text 表示“<color> + チタニウム / Titanium”组合，则返回 (canon, True)；
    否则返回 ("", False)。
    """
    m = _TITANIUM_PAT.search(text)
    if not m:
        return ("", False)
    pre = (m.group("pre") or "").lower()
    post = (m.group("post") or "").lower()
    core = ""
    if pre:
        core = _JA_TO_CORE.get(pre, pre)
    elif post:
        core = _JA_TO_CORE.get(post, post)
    if not core:
        return ("", False)
    canon = _TITANIUM_COLOR_CORE.get(core)
    if not canon:
        # 动态注册里是否有 “<Title> Titanium” 的canon？
        candidate = f"{core.title()} Titanium"
        if candidate in _DYNAMIC_SYNONYMS:
            return (candidate, True)
        # 默认把未知 core 归到 Natural Titanium（保守，不易误伤）
        return ("Natural Titanium", True)
    return (canon, True)

def _guess_base_canon(text: str) -> str | None:
    t = text.lower()
    # 先看动态词表（优先最新）
    for canon, syns in _DYNAMIC_SYNONYMS.items():
        if any(s in t for s in syns) or canon.lower() in t:
            return canon
    # 再看内置基础词表
    for canon, syns in BASE_SYNONYMS.items():
        if any(s in t for s in syns) or canon.lower() in t:
            return canon
    return None

def _fuzzy_guess(text: str) -> Optional[str]:
    # 轻量模糊匹配（避免把“jan/数串”当色）
    import difflib
    t = text.lower()
    # 排除明显非颜色的长数字
    if re.search(r"\d{6,}", t):
        return None
    all_canons = list(BASE_SYNONYMS.keys()) + list(_DYNAMIC_SYNONYMS.keys()) + list(_TITANIUM_COLOR_CORE.values())
    best = difflib.get_close_matches(t, all_canons, n=1, cutoff=0.92)  # 比较严格
    return best[0] if best else None
