"""
Auto Price SQLite 数据库管理模块
管理外部项目商品与本项目 Iphone 实例的映射关系
"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class AutoPriceSQLiteManager:
    """管理 auto_price.sqlite3 数据库"""

    def __init__(self, db_path: str = "auto_price.sqlite3"):
        """
        初始化数据库管理器

        Args:
            db_path: SQLite数据库文件路径,默认为项目根目录的 auto_price.sqlite3
        """
        self.db_path = Path(db_path)
        self._ensure_database()

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 允许通过列名访问
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _ensure_database(self):
        """确保数据库和表结构存在"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 创建映射表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS goods_iphone_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_goods_id INTEGER NOT NULL,
                    external_spec_index INTEGER NOT NULL,
                    iphone_id INTEGER,
                    external_title TEXT NOT NULL,
                    external_spec_name TEXT NOT NULL,
                    external_category_name TEXT,
                    external_category_second_name TEXT,
                    external_category_three_name TEXT,
                    external_price INTEGER,
                    model_name TEXT,
                    capacity_gb INTEGER,
                    color TEXT,
                    confidence_score REAL DEFAULT 0.0,
                    sync_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    last_sync_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(external_goods_id, external_spec_index)
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_goods_status
                ON goods_iphone_mapping(sync_status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_iphone_id
                ON goods_iphone_mapping(iphone_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_external_goods
                ON goods_iphone_mapping(external_goods_id)
            """)

            # 创建同步历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sync_type TEXT NOT NULL,
                    total_items INTEGER DEFAULT 0,
                    matched_items INTEGER DEFAULT 0,
                    unmatched_items INTEGER DEFAULT 0,
                    error_items INTEGER DEFAULT 0,
                    skipped_items INTEGER DEFAULT 0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    error_message TEXT
                )
            """)

            # 迁移：为现有表添加 skipped_items 列（如果不存在）
            cursor.execute("PRAGMA table_info(sync_history)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'skipped_items' not in columns:
                cursor.execute("""
                    ALTER TABLE sync_history
                    ADD COLUMN skipped_items INTEGER DEFAULT 0
                """)
                logger.info("Added skipped_items column to sync_history table")

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def upsert_mapping(self, mapping_data: Dict) -> int:
        """
        插入或更新映射记录

        Args:
            mapping_data: 映射数据字典

        Returns:
            映射记录的ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 准备数据
            now = datetime.now().isoformat()
            mapping_data['updated_at'] = now
            mapping_data['last_sync_at'] = now

            # 使用 INSERT OR REPLACE
            cursor.execute("""
                INSERT INTO goods_iphone_mapping (
                    external_goods_id, external_spec_index, iphone_id,
                    external_title, external_spec_name,
                    external_category_name, external_category_second_name,
                    external_category_three_name, external_price,
                    model_name, capacity_gb, color,
                    confidence_score, sync_status, error_message,
                    last_sync_at, updated_at
                ) VALUES (
                    :external_goods_id, :external_spec_index, :iphone_id,
                    :external_title, :external_spec_name,
                    :external_category_name, :external_category_second_name,
                    :external_category_three_name, :external_price,
                    :model_name, :capacity_gb, :color,
                    :confidence_score, :sync_status, :error_message,
                    :last_sync_at, :updated_at
                )
                ON CONFLICT(external_goods_id, external_spec_index)
                DO UPDATE SET
                    iphone_id = excluded.iphone_id,
                    external_title = excluded.external_title,
                    external_spec_name = excluded.external_spec_name,
                    external_category_name = excluded.external_category_name,
                    external_category_second_name = excluded.external_category_second_name,
                    external_category_three_name = excluded.external_category_three_name,
                    external_price = excluded.external_price,
                    model_name = excluded.model_name,
                    capacity_gb = excluded.capacity_gb,
                    color = excluded.color,
                    confidence_score = excluded.confidence_score,
                    sync_status = excluded.sync_status,
                    error_message = excluded.error_message,
                    last_sync_at = excluded.last_sync_at,
                    updated_at = excluded.updated_at
            """, mapping_data)

            return cursor.lastrowid

    def get_all_mappings(self, status: Optional[str] = None) -> List[Dict]:
        """
        获取所有映射记录

        Args:
            status: 可选的状态过滤 ('matched', 'unmatched', 'pending', 'error')

        Returns:
            映射记录列表
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute("""
                    SELECT * FROM goods_iphone_mapping
                    WHERE sync_status = ?
                    ORDER BY updated_at DESC
                """, (status,))
            else:
                cursor.execute("""
                    SELECT * FROM goods_iphone_mapping
                    ORDER BY updated_at DESC
                """)

            return [dict(row) for row in cursor.fetchall()]

    def get_mapping_by_external_id(
        self,
        goods_id: int,
        spec_index: int
    ) -> Optional[Dict]:
        """
        根据外部商品ID和规格索引查询映射

        Args:
            goods_id: 外部商品ID
            spec_index: 外部规格索引

        Returns:
            映射记录字典或None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM goods_iphone_mapping
                WHERE external_goods_id = ? AND external_spec_index = ?
            """, (goods_id, spec_index))

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_mappings_by_iphone_id(self, iphone_id: int) -> List[Dict]:
        """
        根据本项目的 Iphone ID 查询所有映射

        Args:
            iphone_id: 本项目的 Iphone ID

        Returns:
            映射记录列表
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM goods_iphone_mapping
                WHERE iphone_id = ?
                ORDER BY confidence_score DESC
            """, (iphone_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_mapping_statistics(self) -> Dict:
        """
        获取映射统计信息

        Returns:
            统计信息字典
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN sync_status = 'matched' THEN 1 ELSE 0 END) as matched,
                    SUM(CASE WHEN sync_status = 'unmatched' THEN 1 ELSE 0 END) as unmatched,
                    SUM(CASE WHEN sync_status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN sync_status = 'error' THEN 1 ELSE 0 END) as error,
                    MAX(last_sync_at) as last_sync_at
                FROM goods_iphone_mapping
            """)

            row = cursor.fetchone()
            return dict(row) if row else {}

    def create_sync_record(self, sync_type: str) -> int:
        """
        创建同步历史记录

        Args:
            sync_type: 同步类型

        Returns:
            同步记录ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sync_history (sync_type, started_at, status)
                VALUES (?, ?, 'running')
            """, (sync_type, datetime.now().isoformat()))

            return cursor.lastrowid

    def update_sync_record(
        self,
        sync_id: int,
        status: str = 'completed',
        stats: Optional[Dict] = None,
        error_message: Optional[str] = None
    ):
        """
        更新同步历史记录

        Args:
            sync_id: 同步记录ID
            status: 同步状态
            stats: 统计信息
            error_message: 错误信息
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            update_fields = {
                'completed_at': datetime.now().isoformat(),
                'status': status,
                'error_message': error_message
            }

            if stats:
                update_fields.update(stats)

            set_clause = ', '.join([f"{k} = ?" for k in update_fields.keys()])
            values = list(update_fields.values()) + [sync_id]

            cursor.execute(f"""
                UPDATE sync_history
                SET {set_clause}
                WHERE id = ?
            """, values)

    def clear_all_mappings(self):
        """清空所有映射记录(谨慎使用!)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM goods_iphone_mapping")
            logger.warning("All mappings have been cleared!")
