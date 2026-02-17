from __future__ import annotations

"""
shop16 清洗器 — 携帯空間

  原始文本（買取価格列）
    │ 两阶段流水线：Match → expand_match_tokens → match_tokens_to_specs
    │ SHOP16_ADAPTIVE_SPLIT (环境变量，默认 true)
    │
    ├─ _normalize_price_text_shop16()     ← Step 1: 归一化（换行→/、压缩空白）
    │
    ├─ _extract_base_price_shop16()       ← Step 2: 提取基础价
    │
    ├─ 阶段 1: _match_shop16()            ← NONE_RE / DELTA_RE(分支) / ABS_RE
    │
    ├─ expand_match_tokens()              ← 自适应分割（阶段 1 与 2 之间）
    │
    └─ match_tokens_to_specs()            ← 阶段 2 语义映射 + 边界规则 → (deltas, abs_specs)
    ├─ _label_matches_color_unified()     ← 标签→颜色匹配（cleaner_tools 统一）
    │
    └─ clean_shop16()                     ← 主函数，生成输出行

  自适应分割 (与 shop17 同策略):
    - 环境变量: SHOP16_ADAPTIVE_SPLIT=true/false
    - 默认启用，支持复合标签如 "青/オレンジ -5000"
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ...external_ingest.cleaner_tools import to_int_yen, parse_dt_aware
from ..cleaner_tools import (
    _parse_capacity_gb,
    _normalize_model_generic,
    _norm_strip,
    PriceDecomposition,
    resolve_color_prices,
    _label_matches_color_unified,
    setup_color_cleaner,
    finalize_color_cleaner,
    LABEL_SPLIT_RE_shop16 as SPLIT_TOKENS_RE,
    LABEL_SPLIT_RE_shop16_SIMPLE,
    MatchToken,
    FORMAT_HINT_SIGNED,
    FORMAT_HINT_SEP_MINUS,
    FORMAT_HINT_AFTER_YEN,
    FORMAT_HINT_PLAIN_DIGITS,
    FORMAT_HINT_COLON_PREFIX,
    FORMAT_HINT_NONE,
    expand_match_tokens,
    match_tokens_to_specs,
    EXTRACTION_MODE,
)

# 初始化 logger
logger = logging.getLogger(__name__)

CLEANER_NAME = "shop16"
SHOP_NAME = "携帯空間"

# 自适应分割开关（与 shop17 同策略）
ENABLE_ADAPTIVE_SPLIT_SHOP16 = os.getenv("SHOP16_ADAPTIVE_SPLIT", "true").lower() == "true"

# DEBUG 功能现在由 logging 级别控制（在 settings.py 的 LOGGING 配置中）
# 控制台显示 INFO 级别（简洁），文件记录 DEBUG 级别（详细）

# ----------------------------------------------------------------------
# 配置 (EXTRACTION_MODE 见 cleaner_tools)
# ----------------------------------------------------------------------

MODEL_COL = "iPhone 17 Pro Max"
DESC_COL  = "説明1"
PRICE_COL = "買取価格"

# ----------------------------------------------------------------------
# Step 1-3: 常量与 _norm
# ----------------------------------------------------------------------

_norm = _norm_strip  # 颜色匹配用归一化（去空格 + 转小写）

FIRST_YEN_RE = re.compile(r"(?:￥|\¥)?\s*(\d[\d,]*)\s*円?")
_BASE_ONLY_RE = re.compile(r"^\s*(?:￥|\¥)?\s*\d[\d,]*\s*(?:円)?\s*$")
_TRAILING_AMOUNT_IN_LABEL_RE = re.compile(
    r"(?:[：:])?\s*(?:￥)?\s*[+\-−－]?\s*\d[\d,]*\s*(?:円)?\s*$",
    re.UNICODE,
)

# ----------------------------------------------------------------------
# Step 4: 标签→颜色匹配（2025-02 替换为 cleaner_tools 统一实现）
# ----------------------------------------------------------------------
# 原 shop16 独立实现已迁移至 cleaner_tools._label_matches_color_unified，
# 合并 shop3/4/9/11/12/14/15/16/17 逻辑，供所有清洗器共用。

# ----------------------------------------------------------------------
# Step 5: 正则模式定义（NONE_RE + DELTA_RE + ABS_RE，与 shop17 三正则模式一致）
# ----------------------------------------------------------------------

COLOR_NONE_RE_shop16 = re.compile(
    r"""(?P<label>[^：:\-\s/、／，,\n]+(?:\([^)]*\))?)\s*
        (?:(?P<sep>[：:\-])\s*)?
        (?:減額)?なし
    """,
    re.UNICODE | re.VERBOSE,
)

# label 排除数字，避免 "ブルー229,000円" 中金额被吃进 label
COLOR_DELTA_RE_shop16 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／\n]+(?:\([^)]*\))?)\s*
        (?P<sep>[：:\-])?\s*
        (?P<sign>[+\-−－])?\s*
        (?P<amount>\d[\d,]*)\s*(?:円)?
    """,
    re.UNICODE | re.VERBOSE,
)

COLOR_ABS_RE_shop16 = re.compile(
    r"""(?P<label>[^\d：:\-\s/、／￥円\n]+(?:\([^)]*\))?)\s*￥\s*(?P<amount>\d[\d,]*)\s*(?:円)?""",
    re.UNICODE,
)

_GROUP_SHARED_DELTA_RE = re.compile(
    r"""
    (?P<labels>[^0-9￥円]+?)          # 多颜色标签段（含 /）
    \s*(?P<sign>[+\-−－])\s*         # 显式正负号
    (?P<amount>\d[\d,]*)\s*(?:円)?   # 金额（可无 円）
    """,
    re.UNICODE | re.VERBOSE
)

# 过滤非颜色标签（参考 shop17）
_BAD_LABEL_WORDS_shop16 = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _is_plausible_color_label_shop16(label: str) -> bool:
    """过滤明显非颜色名的 label。"""
    label = _normalize_label_shop16(label)
    if not label or label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_LABEL_WORDS_shop16):
        return False
    return True


# ----------------------------------------------------------------------
# 阶段 1：匹配（输出 MatchToken）
# ----------------------------------------------------------------------

def _match_shop16(text: str) -> List[MatchToken]:
    """
    阶段 1 匹配：从買取価格文本中提取 MatchToken[]。
    使用 NONE_RE / DELTA_RE(分支) / ABS_RE 三正则，支持 pending_labels（标签与金额分处相邻 part）。
    """
    tokens: List[MatchToken] = []
    if not text:
        return tokens

    s = _normalize_price_text_shop16(str(text))
    # 去掉基础价前缀
    m0 = FIRST_YEN_RE.search(s)
    tail = s[m0.end():].strip() if m0 else s
    if not tail:
        return tokens

    parts = [p.strip() for p in SPLIT_TOKENS_RE.split(tail) if p and p.strip()]
    if not parts:
        parts = [tail]

    pending_labels: List[str] = []
    position = 0

    for part in parts:
        m0 = COLOR_NONE_RE_shop16.search(part)
        if m0:
            label_raw = _normalize_label_shop16(m0.group("label"))
            if _is_plausible_color_label_shop16(label_raw):
                tokens.append(MatchToken(
                    label=label_raw,
                    amount_int=0,
                    format_hint=FORMAT_HINT_NONE,
                    position=position,
                ))
                position += 1
            pending_labels = []
            continue

        has_amount_in_part = False
        for m in COLOR_ABS_RE_shop16.finditer(part):
            has_amount_in_part = True
            label_raw = _normalize_label_shop16(m.group("label"))
            if not _is_plausible_color_label_shop16(label_raw):
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
        if has_amount_in_part:
            pending_labels = []
            continue

        has_delta_in_part = False
        for m in COLOR_DELTA_RE_shop16.finditer(part):
            has_delta_in_part = True
            label_raw = _normalize_label_shop16(m.group("label"))
            if not _is_plausible_color_label_shop16(label_raw):
                continue
            sep = m.group("sep")
            sign = m.group("sign")
            amt = to_int_yen(m.group("amount"))
            if amt is None:
                continue
            amt_val = int(amt)
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

            # 当前标签
            tok = MatchToken(label=label_raw, amount_int=amount_int, format_hint=hint, position=position)
            tokens.append(tok)
            position += 1
            # 挂起的标签共用同一金额
            for pl in pending_labels:
                pl_norm = _normalize_label_shop16(pl)
                if pl_norm and _is_plausible_color_label_shop16(pl_norm):
                    tokens.append(MatchToken(
                        label=pl_norm,
                        amount_int=amount_int,
                        format_hint=hint,
                        position=position,
                    ))
                    position += 1
            pending_labels = []
        if has_delta_in_part:
            continue

        # 仅标签无金额：挂起等待下一 part
        for tok in LABEL_SPLIT_RE_shop16_SIMPLE.split(part):
            tok = _normalize_label_shop16(tok)
            if tok:
                pending_labels.append(tok)

    return tokens


# ----------------------------------------------------------------------
# 辅助：归一化与基础价提取
# ----------------------------------------------------------------------

def _normalize_price_text_shop16(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    # 把换行变成分隔（保留"下一行是颜色差价"的结构）
    s = re.sub(r"[\r\n]+", " / ", s)
    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    # 多个分隔合并
    s = re.sub(r"(?:\s*/\s*){2,}", " / ", s).strip()
    return s


def _extract_base_price_shop16(text: str) -> Optional[int]:
    if not text:
        return None
    m = FIRST_YEN_RE.search(str(text))
    if not m:
        return to_int_yen(text)  # 兜底
    return to_int_yen(m.group(1))


def _is_base_only_price_text(price_text_norm: str) -> bool:
    """判断文本是否只包含一个基础价，不含任何颜色差价信息。"""
    return bool(_BASE_ONLY_RE.match(price_text_norm or ""))


def _normalize_label_shop16(lbl: str) -> str:
    s = re.sub(r"[\s\u3000\xa0]+", "", lbl or "")
    s = re.sub(r"(カラー|色)$", "", s)
    # 去掉黏在 label 末尾的金额/符号：-1000 / ￥86100 / :-1,000円 等
    s = _TRAILING_AMOUNT_IN_LABEL_RE.sub("", s)
    return s.strip()


def _split_labels_shop16(lbl: str) -> List[str]:
    # 兼容 "青/オレンジ""黒、白""blue/black" 等
    raw = _normalize_label_shop16(lbl)
    parts = LABEL_SPLIT_RE_shop16_SIMPLE.split(raw)
    return [p for p in (_normalize_label_shop16(x) for x in parts) if p]


def _extract_shared_delta_map_shop16(price_text_norm: str) -> Dict[str, int]:
    """
    从原文中抽取： 'オレンジ/青 -1500' 这种共享差价 -> {オレンジ:-1500, 青:-1500}
    这是"纠错用"的确定性证据，不替代 LLM 抽取的主流程。
    """
    s = price_text_norm or ""
    out: Dict[str, int] = {}
    # 去掉基础价前缀，减少误匹配（基础价一般在最前）
    m0 = FIRST_YEN_RE.search(s)
    tail = s[m0.end():] if m0 else s

    for m in _GROUP_SHARED_DELTA_RE.finditer(tail):
        labels_raw = m.group("labels") or ""
        sign = m.group("sign") or ""
        amt = to_int_yen(m.group("amount"))
        if amt is None:
            continue
        delta = -int(amt) if sign in ("-", "−", "－") else int(amt)

        # 拆分 labels（/、，等）
        for lb in LABEL_SPLIT_RE_shop16_SIMPLE.split(labels_raw):
            lb = _normalize_label_shop16(lb)
            if lb:
                out[lb] = delta
    return out


# ----------------------------------------------------------------------
# 清洗主函数
# ----------------------------------------------------------------------

def clean_shop16(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    ctx, early = setup_color_cleaner(
        df, cleaner_name=CLEANER_NAME, shop_name=SHOP_NAME,
        required_cols=[MODEL_COL, DESC_COL, PRICE_COL, "time-scraped"],
        extraction_mode=EXTRACTION_MODE,
    )
    if ctx is None:
        return early

    rows: List[dict] = []

    for idx, row in df.iterrows():
        model_cell = str(row.get(MODEL_COL) or "").strip()
        desc_cell  = str(row.get(DESC_COL)  or "").strip()
        price_cell = row.get(PRICE_COL)
        rec_at     = parse_dt_aware(row.get("time-scraped"))

        is_unopened = ("未開封" in desc_cell) or ("未開封" in model_cell)
        if not is_unopened:
            continue

        model_text = model_cell.replace("\u3000", " ").replace("\xa0", " ").replace("\n", " ").strip()
        model_norm = _normalize_model_generic(model_text)
        cap_gb = _parse_capacity_gb(model_text)
        if not model_norm or cap_gb is None or pd.isna(cap_gb):
            continue
        cap_gb = int(cap_gb)

        key = (model_norm, cap_gb)
        color_map = ctx.color_map.get(key)
        if not color_map:
            continue

        price_raw = "" if price_cell is None else str(price_cell)
        price_text = _normalize_price_text_shop16(price_raw)

        base_price = _extract_base_price_shop16(price_text)
        tokens = _match_shop16(price_text)
        tokens = expand_match_tokens(
            tokens,
            color_map,
            _label_matches_color_unified,
            enable_adaptive=ENABLE_ADAPTIVE_SPLIT_SHOP16,
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
        )
        deltas, abs_specs = match_tokens_to_specs(
            tokens,
            context={"base_price": base_price, "has_base_price": base_price is not None},
            logger=ctx.logger,
            cleaner_name=CLEANER_NAME,
            shop_name=SHOP_NAME,
            row_index=int(idx),
        )

        decomp = PriceDecomposition(
            base_price=base_price,
            delta_specs=deltas,
            abs_specs=abs_specs,
            extraction_method="regex",
            source_text_raw=price_text,
        )

        if decomp.base_price is None and not decomp.abs_specs:
            continue

        emit_default = decomp.base_price is not None

        new_rows, ctx.log_seq = resolve_color_prices(
            decomp,
            color_map,
            _label_matches_color_unified,
            shop_name=SHOP_NAME,
            cleaner_name=CLEANER_NAME,
            recorded_at=rec_at,
            emit_default_rows=emit_default,
            logger=ctx.logger,
            log_seq_start=ctx.log_seq,
            row_index=int(idx),
            model_text=model_text,
            model_norm=model_norm,
            capacity_gb=cap_gb,
        )
        rows.extend(new_rows)

    return finalize_color_cleaner(ctx, rows)
