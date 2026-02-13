"""
ClickHouse 读写服务层。
参考: docs/REFACTOR_PLAN_V1.md §11.4, §12
"""
from __future__ import annotations

import logging
from datetime import datetime

from django.conf import settings

logger = logging.getLogger(__name__)


def _get_client():
    """延迟导入 clickhouse_driver 并建立连接。"""
    from clickhouse_driver import Client
    return Client(
        host=getattr(settings, "CLICKHOUSE_HOST", "clickhouse"),
        port=int(getattr(settings, "CLICKHOUSE_PORT", 9000)),
        database=getattr(settings, "CLICKHOUSE_DB", "yamagoti"),
        user=getattr(settings, "CLICKHOUSE_USER", "default"),
        password=getattr(settings, "CLICKHOUSE_PASSWORD", ""),
    )


class ClickHouseService:
    """封装 ClickHouse 的全部读写操作。"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    # ── 写入 ──────────────────────────────────────────────────────────────

    def insert_price_aligned(self, df, run_id: str) -> int:
        """将 DataFrame 批量写入 price_aligned。

        Parameters
        ----------
        df : pd.DataFrame
            columns: bucket, shop_id, iphone_id, price_new, price_a, price_b,
                     alignment_diff_sec, record_time
        run_id : str

        Returns
        -------
        int  写入行数
        """
        if df.empty:
            return 0

        rows = []
        for row in df.itertuples(index=False):
            rows.append({
                "run_id": run_id,
                "bucket": _to_naive(row.bucket),
                "shop_id": int(row.shop_id),
                "iphone_id": int(row.iphone_id),
                "price_new": int(row.price_new),
                "price_a": _nullable_int(row.price_a),
                "price_b": _nullable_int(row.price_b),
                "alignment_diff_sec": int(row.alignment_diff_sec),
                "record_time": _to_naive(row.record_time),
            })

        self.client.execute(
            "INSERT INTO price_aligned "
            "(run_id, bucket, shop_id, iphone_id, price_new, price_a, price_b, "
            " alignment_diff_sec, record_time) VALUES",
            rows,
        )
        logger.info("insert_price_aligned: run_id=%s  rows=%d", run_id, len(rows))
        return len(rows)

    def insert_features(self, df, run_id: str) -> int:
        """将 DataFrame 批量写入 features_wide。

        Parameters
        ----------
        df : pd.DataFrame
            必须包含 bucket, scope 以及各特征列
        run_id : str

        Returns
        -------
        int  写入行数
        """
        if df.empty:
            return 0

        # 动态获取列 (排除 bucket, scope, 这两个必填)
        feature_cols = [c for c in df.columns if c not in ("bucket", "scope")]
        ch_cols = ["run_id", "bucket", "scope"] + feature_cols
        col_list = ", ".join(ch_cols)

        rows = []
        for row in df.itertuples(index=False):
            d = {"run_id": run_id, "bucket": _to_naive(row.bucket), "scope": row.scope}
            for c in feature_cols:
                v = getattr(row, c)
                d[c] = None if _is_nan(v) else float(v)
            rows.append(d)

        self.client.execute(f"INSERT INTO features_wide ({col_list}) VALUES", rows)
        logger.info("insert_features: run_id=%s  rows=%d", run_id, len(rows))
        return len(rows)

    # ── 管理 ──────────────────────────────────────────────────────────────

    def drop_run(self, run_id: str, tables: list[str] | None = None) -> dict[str, int]:
        """按 run_id 删除分区。

        Returns
        -------
        dict  {table_name: dropped_partition_count}
        """
        if tables is None:
            tables = ["price_aligned", "features_wide"]

        result = {}
        for table in tables:
            partitions = self.client.execute(
                f"SELECT partition FROM system.parts "
                f"WHERE database = currentDatabase() AND table = %(tbl)s "
                f"  AND active "
                f"  AND partition LIKE %(prefix)s "
                f"GROUP BY partition",
                {"tbl": table, "prefix": f"('{run_id}',%"},
            )
            for (p,) in partitions:
                self.client.execute(f"ALTER TABLE {table} DROP PARTITION {p}")
            result[table] = len(partitions)
            logger.info("drop_run: %s run_id=%s  partitions_dropped=%d",
                        table, run_id, len(partitions))
        return result

    def list_runs(self) -> list[dict]:
        """列出所有 run_id 及其行数统计。"""
        rows = self.client.execute(
            "SELECT 'price_aligned' AS tbl, run_id, count() AS cnt "
            "FROM price_aligned GROUP BY run_id "
            "UNION ALL "
            "SELECT 'features_wide' AS tbl, run_id, count() AS cnt "
            "FROM features_wide GROUP BY run_id "
            "ORDER BY tbl, run_id"
        )
        return [{"table": r[0], "run_id": r[1], "count": r[2]} for r in rows]

    def promote_run(
        self,
        from_run: str,
        to_run: str,
        *,
        keep_backup: bool = False,
    ) -> dict:
        """将 from_run 的数据提升为 to_run (通常是 'live')。

        Parameters
        ----------
        from_run : str
        to_run : str
        keep_backup : bool
            如果 True, 先将当前 to_run 备份为 backup_YYYYMMDD

        Returns
        -------
        dict  {action: detail, ...}
        """
        info = {}

        # 1. 可选: 备份当前 to_run
        if keep_backup:
            backup_id = f"backup_{datetime.now():%Y%m%d}"
            for table in ("price_aligned", "features_wide"):
                cols = self._get_column_names(table)
                col_expr = ", ".join(
                    f"'{backup_id}' AS run_id" if c == "run_id" else c
                    for c in cols if c != "inserted_at"
                )
                self.client.execute(
                    f"INSERT INTO {table} SELECT {col_expr} "
                    f"FROM {table} WHERE run_id = %(rid)s",
                    {"rid": to_run},
                )
            info["backup"] = backup_id

        # 2. 删旧 to_run
        dropped = self.drop_run(to_run)
        info["dropped"] = dropped

        # 3. INSERT SELECT from_run → to_run
        promoted = {}
        for table in ("price_aligned", "features_wide"):
            cols = self._get_column_names(table)
            col_expr = ", ".join(
                f"'{to_run}' AS run_id" if c == "run_id" else c
                for c in cols if c != "inserted_at"
            )
            self.client.execute(
                f"INSERT INTO {table} SELECT {col_expr} "
                f"FROM {table} WHERE run_id = %(rid)s",
                {"rid": from_run},
            )
            cnt = self.client.execute(
                f"SELECT count() FROM {table} WHERE run_id = %(rid)s",
                {"rid": to_run},
            )[0][0]
            promoted[table] = cnt
        info["promoted"] = promoted
        logger.info("promote_run: %s → %s  %s", from_run, to_run, info)
        return info

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _get_column_names(self, table: str) -> list[str]:
        rows = self.client.execute(
            "SELECT name FROM system.columns "
            "WHERE database = currentDatabase() AND table = %(tbl)s "
            "ORDER BY position",
            {"tbl": table},
        )
        return [r[0] for r in rows]


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _to_naive(dt) -> datetime:
    """去掉 tz 信息，clickhouse-driver DateTime 不接受 aware datetime。"""
    import pandas as pd
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _nullable_int(v):
    """NaN → None, 否则 int。"""
    if _is_nan(v):
        return None
    return int(v)


def _is_nan(v) -> bool:
    if v is None:
        return True
    try:
        import math
        return math.isnan(v)
    except (TypeError, ValueError):
        return False
