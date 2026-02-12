# AppleStockChecker/utils/external_ingest/cleaner_tools.py
"""
清洗器通用工具模块
提供数据库访问、数据转换等通用功能
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import logging
import pandas as pd
import re

from .helpers import to_int_yen


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


def _load_iphone17_info_df_from_db() -> pd.DataFrame:
    """
    从数据库中读取 iPhone 机型信息，返回 DataFrame

    输出列：part_number, model_name, capacity_gb, color, jan（如果 jan 字段有值）

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
    """
    df = info_df.copy()
    df["model_name_norm"] = df["model_name"].map(_normalize_model_generic)
    df["capacity_gb"] = pd.to_numeric(df["capacity_gb"], errors="coerce").astype("Int64")
    df["color_norm"] = df["color"].map(lambda x: (str(x) or "").strip())
    cmap: Dict[Tuple[str, int], Dict[str, Tuple[str, str]]] = {}
    for _, r in df.iterrows():
        m = r["model_name_norm"]
        cap = r["capacity_gb"]
        if not m or pd.isna(cap):
            continue
        key = (m, int(cap))
        cmap.setdefault(key, {})
        cmap[key][(str(r["color"]) or "").strip()] = (str(r["part_number"]), str(r["color"]))
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
    """通用 strip 归一化，返回去除首尾空白的字符串"""
    return (s or "").strip()


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
                "source_text_raw_full": source_text_raw_full,
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
        "source_text_raw_full": source_text_raw_full,
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
                "source_text_raw_full": source_text_raw_full,
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
