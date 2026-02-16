# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import json
from typing import Optional
import httpx
import pandas as pd
from django.conf import settings

TIMEOUT = 60.0


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

async def fetch_webscraper_export(job_id: str, *, format: str = "csv") -> bytes:
    """
    用 WebScraper Cloud API 拉取某个 Job 的导出（默认 CSV 字节流）。
    需要 settings.WEB_SCRAPER_API_TOKEN 与 WEB_SCRAPER_EXPORT_URL_TEMPLATE。
    """
    token = getattr(settings, "WEB_SCRAPER_API_TOKEN", "")
    tpl   = getattr(settings, "WEB_SCRAPER_EXPORT_URL_TEMPLATE", "")
    if not token or not tpl:
        raise RuntimeError("WEB_SCRAPER_API_TOKEN / WEB_SCRAPER_EXPORT_URL_TEMPLATE 未配置")

    url = tpl.format(job_id=job_id)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content

def to_dataframe_from_request(content_type: str, body: bytes) -> pd.DataFrame:
    ct = (content_type or "").lower()
    if "csv" in ct or ct.startswith("text/plain"):
        return read_csv_smart(body)
    if "json" in ct:
        obj = json.loads(body.decode("utf-8"))
        if isinstance(obj, dict) and "rows" in obj:
            return pd.DataFrame(obj["rows"])
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        raise RuntimeError("不支持的 JSON 结构：需要数组或包含 rows 的对象")
    raise RuntimeError(f"不支持的 Content-Type: {content_type}")



def fetch_webscraper_export_sync(job_id: str, *, format: str = "csv") -> bytes:
    token = getattr(settings, "WEB_SCRAPER_API_TOKEN", "")
    tpl   = getattr(settings, "WEB_SCRAPER_EXPORT_URL_TEMPLATE", "")
    if not token or not tpl:
        raise RuntimeError("WEB_SCRAPER_API_TOKEN / WEB_SCRAPER_EXPORT_URL_TEMPLATE 未配置")
    url = tpl.format(job_id=job_id)
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content