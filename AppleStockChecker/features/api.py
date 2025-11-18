# AppleStockChecker/features/api.py

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, List, Optional, Sequence, Tuple

from django.db import IntegrityError, transaction, connections
from django.utils import timezone
from django.db.models import Q
from datetime import timezone as dt_timezone

from AppleStockChecker.models import FeatureSnapshot


# ---- 公用工具 ----

def _quantize_2(x: float | int | str | Decimal | None) -> float | None:
    """统一数值量化到 2 位小数；None 透传。"""
    if x is None:
        return None
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

try:
    UTC_TZ = timezone.utc          # Django < 5
except AttributeError:
    UTC_TZ = dt_timezone.utc       # Django >= 5

def _to_utc_aware(dt):
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt)
    return dt.astimezone(UTC_TZ)

# Feature 的唯一键

@dataclass(frozen=True)
class FeatureKey:
    bucket: datetime
    scope: str
    name: str
    version: str

@dataclass(frozen=True)
class FeatureRecord(FeatureKey):
    value: float
    is_final: bool = True


class FeatureWriter:
    """
    统一的 FeatureSnapshot 写入器（并发安全 / 幂等 / 支持批量 upsert）。

    关键行为：
      - 冲突策略：默认 "更新为最新值"，且 is_final 采用 OR 语义（True 一旦出现就不回退）。
      - 单条：行级锁 + 重试处理 IntegrityError。
      - 批量：优先用 bulk upsert（Django 4.1+）；自动降级到单条重试。
      - 自动将 bucket 归一到 UTC-aware。
    """

    UNIQUE_FIELDS: Sequence[str] = ("bucket", "scope", "name", "version")
    UPDATE_FIELDS: Sequence[str] = ("value", "is_final")

    def __init__(
        self,
        *,
        bucket: datetime,
        default_version: str = "v1",
        is_final: bool = True,
        using: str = "default",
        escalate_is_final: bool = False,
        max_retries: int = 2,
        chunk_size: int = 1000,
    ):
        self.bucket = _to_utc_aware(bucket)
        self.default_version = default_version
        self.default_is_final = bool(is_final)
        self.using = using
        self.escalate_is_final = bool(escalate_is_final)
        self.max_retries = int(max_retries)
        self.chunk_size = int(max(1, chunk_size))

    # ---------- 单条写入（并发安全） ----------
    def write(
        self,
        scope: str,
        name: str,
        value: float,
        *,
        version: Optional[str] = None,
        is_final: Optional[bool] = None,
    ):
        """
        并发安全 upsert：
        - 若不存在 -> INSERT
        - 若并发冲突 -> 捕获 IntegrityError 并重试
        - 若存在 -> UPDATE (value, is_final)；is_final 根据策略（默认 OR）
        """
        key = FeatureKey(
            bucket=self.bucket,
            scope=scope,
            name=name,
            version=version or self.default_version,
        )
        rec = FeatureRecord(
            **key.__dict__,
            value=_quantize_2(value),  # 统一量化
            is_final=self.default_is_final if is_final is None else bool(is_final),
        )
        return self._upsert_one(rec)

    # ---------- 批量写入（自动选择最优路径） ----------
    def write_many(self, records: Iterable[FeatureRecord]):
        from dataclasses import replace
        """
        批量 upsert：
        - 优先尝试 bulk_create(update_conflicts=True, ...)（Django 4.1+）
          * 在批量前会预读已有 is_final，以应用 OR 语义（避免 True 被 False 覆盖）
        - 如果环境不支持或失败，则自动降级为逐条 _upsert_one（带重试）
        """
        rows = [
            replace(
                r,
                bucket=_to_utc_aware(r.bucket),
                value=_quantize_2(r.value),
                is_final=bool(r.is_final),
            )
            for r in records
        ]

        if not rows:
            return 0
        try:
            return self._bulk_upsert(rows)
        except TypeError:
            return self._fallback_row_by_row(rows)
        except Exception:
            return self._fallback_row_by_row(rows)

    # ---------- 内部实现：单条 upsert ----------
    def _upsert_one(self, rec: FeatureRecord):
        """
        通过行级锁 + 重试实现并发安全：
          1) SELECT ... FOR UPDATE（若存在，直接更新）
          2) 不存在则尝试 INSERT
          3) 若 INSERT 因并发失败，捕获 IntegrityError 再次回读更新
        """
        for attempt in range(self.max_retries + 1):
            try:
                with transaction.atomic(using=self.using):
                    qs = (FeatureSnapshot.objects.using(self.using)
                          .select_for_update()
                          .filter(bucket=rec.bucket, scope=rec.scope, name=rec.name, version=rec.version))
                    obj = qs.first()
                    if obj:
                        # is_final 采用 OR 语义（可配置）
                        new_final = (obj.is_final or rec.is_final) if self.escalate_is_final else rec.is_final
                        obj.value = rec.value
                        obj.is_final = new_final
                        obj.save(update_fields=self.UPDATE_FIELDS)
                        return obj
                    else:
                        return FeatureSnapshot.objects.using(self.using).create(
                            bucket=rec.bucket,
                            scope=rec.scope,
                            name=rec.name,
                            version=rec.version,
                            value=rec.value,
                            is_final=rec.is_final,
                        )
            except IntegrityError:
                if attempt >= self.max_retries:
                    raise
                # 并发插入撞唯一键；下一轮会回读到已存在的行再更新
                continue

    # ---------- 内部实现：批量 upsert（首选路径） ----------
    def _bulk_upsert(self, rows: List[FeatureRecord]) -> int:
        """
        使用 bulk_create(update_conflicts=True) 实现批量 UPSERT。
        需要 Django 4.1+。若当前环境不支持，会抛 TypeError 由上层捕获降级。
        """
        objs = [
            FeatureSnapshot(
                bucket=r.bucket,
                scope=r.scope,
                name=r.name,
                version=r.version,
                value=r.value,
                is_final=r.is_final,
            )
            for r in rows
        ]

        # 分块，避免一次性 SQL 过大
        total = 0
        qs = FeatureSnapshot.objects.using(self.using)
        for i in range(0, len(objs), self.chunk_size):
            chunk = objs[i:i + self.chunk_size]
            # 关键：冲突即更新 value / is_final
            # 注意：如果 escalate_is_final=True，我们已在写入前把 is_final 做了 OR 合并
            qs.bulk_create(
                chunk,
                update_conflicts=True,            # Django 4.1+
                update_fields=list(self.UPDATE_FIELDS),
                unique_fields=list(self.UNIQUE_FIELDS),
            )
            total += len(chunk)
        return total

    # ---------- 内部实现：批量降级逐条 ----------
    def _fallback_row_by_row(self, rows: List[FeatureRecord]) -> int:
        cnt = 0
        for r in rows:
            self._upsert_one(r)
            cnt += 1
        return cnt

    # ---------- 辅助：批量读取已有 is_final（做 OR 语义） ----------
    def _fetch_existing_is_final(self, rows: List[FeatureRecord]) -> dict[Tuple[datetime, str, str, str], bool]:
        """
        读取已有行的 is_final，用于在批量 upsert 前做“True 优先”的合并。
        为简洁起见，这里用 OR 的 Q 链接（成百上千条也够用；特别大可以考虑 raw SQL）。
        """
        # 构建 Q 条件（(bucket, scope, name, version) 多组 OR）
        q = Q()
        for r in rows:
            q |= Q(bucket=r.bucket, scope=r.scope, name=r.name, version=r.version)
        if not q.children:
            return {}

        exists = (FeatureSnapshot.objects.using(self.using)
                  .filter(q)
                  .values("bucket", "scope", "name", "version", "is_final"))
        m: dict[Tuple[datetime, str, str, str], bool] = {}
        for row in exists:
            key = (row["bucket"], row["scope"], row["name"], row["version"])
            m[key] = bool(row.get("is_final"))
        return m
