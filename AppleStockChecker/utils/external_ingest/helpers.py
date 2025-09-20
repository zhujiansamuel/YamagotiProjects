# AppleStockChecker/utils/external_ingest/helpers.py
from __future__ import annotations
import io, re
from typing import Optional, Tuple
import httpx
import pandas as pd
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date





HTTP_TIMEOUT = 30.0

async def async_http_get_bytes(url: str, *, headers: dict | None = None) -> bytes:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, headers=headers or {})
        r.raise_for_status()
        return r.content

def read_csv_smart(data: bytes, *, encodings=("utf-8-sig", "utf-8", "cp932", "shift_jis"), seps=(",", "\t", ";", "|")) -> pd.DataFrame:
    # 先用 c 引擎快读，不行再用 python 引擎
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, engine="c", on_bad_lines="skip")
                if df.shape[1] == 1 and any(x in str(df.columns[0]) for x in [",", "\t", ";", "|"]):
                    raise ValueError("wrong sep")
                return df
            except Exception:
                pass
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
            except Exception:
                pass
    # 兜底：按 Excel 读（万一远端返回的是 xlsx）
    try:
        return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(f"无法解析为 CSV/Excel: {e}")

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
