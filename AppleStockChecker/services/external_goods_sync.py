"""
外部商品同步服务
负责从外部项目获取商品列表并与本项目的 Iphone 实例进行映射
"""
import re
import logging
import requests
from typing import List, Dict, Optional, Tuple
from django.conf import settings
from AppleStockChecker.models import Iphone
from .auto_price_db import AutoPriceSQLiteManager

logger = logging.getLogger(__name__)


class ExternalGoodsClient:
    """外部商品API客户端"""

    def __init__(self, api_url: str, api_token: str):
        """
        初始化外部API客户端

        Args:
            api_url: 外部API的URL
            api_token: 外部API的访问令牌
        """
        self.api_url = api_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'token': f'{self.api_token}',
            'Content-Type': 'application/json',
        })

    def fetch_goods_list(self) -> List[Dict]:
        """
        从外部项目获取商品列表

        Returns:
            商品列表

        Raises:
            requests.RequestException: API请求失败
        """
        try:
            response = self.session.get(
                f'{self.api_url}/api/goodsprice/list',
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # 添加调试日志
            logger.info(f"API Response type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"API Response keys: {list(data.keys())}")

            # 根据实际API响应结构提取商品列表
            goods_list = []

            if isinstance(data, dict):
                # 尝试常见的响应结构
                if 'data' in data:
                    data_field = data['data']
                    # 检查 data 字段是否还是字典（嵌套的 data 结构）
                    if isinstance(data_field, dict) and 'data' in data_field:
                        goods_list = data_field['data']
                        logger.info(f"Found nested data structure")
                    elif isinstance(data_field, list):
                        goods_list = data_field
                    else:
                        # data 字段不是预期的格式
                        logger.warning(f"Unexpected data field type: {type(data_field)}")
                        goods_list = [data_field] if isinstance(data_field, dict) else []
                elif 'list' in data:
                    goods_list = data['list']
                elif 'items' in data:
                    goods_list = data['items']
                elif 'goods' in data:
                    goods_list = data['goods']
                else:
                    # 如果没有找到标准的列表字段，尝试找到第一个值为列表的字段
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            goods_list = value
                            logger.info(f"Found goods list in key: {key}")
                            break

                    # 如果还是没有找到，可能整个dict就是一个商品对象
                    if not goods_list and 'goods_id' in data:
                        goods_list = [data]

            elif isinstance(data, list):
                goods_list = data
            else:
                logger.warning(f"Unexpected API response format: {type(data)}")
                goods_list = [data]

            # 验证 goods_list 是否有效
            if not isinstance(goods_list, list):
                logger.error(f"goods_list is not a list: {type(goods_list)}")
                raise ValueError(f"Invalid goods list format: {type(goods_list)}")

            # 验证列表中的元素是否是字典
            if goods_list:
                first_item = goods_list[0]
                if not isinstance(first_item, dict):
                    logger.error(f"First item in goods_list is not a dict: {type(first_item)}, value: {first_item}")
                    raise ValueError(f"Invalid goods item format: {type(first_item)}")

            logger.info(f"Fetched {len(goods_list)} goods from external API")
            return goods_list

        except requests.RequestException as e:
            logger.error(f"Failed to fetch goods list: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse goods list: {e}", exc_info=True)
            raise

    def update_goods_price(self, goods_id: int, spec_index: int, price: int) -> bool:
        """
        更新外部项目商品价格

        Args:
            goods_id: 商品ID
            spec_index: 规格索引
            price: 新价格

        Returns:
            是否成功
        """
        try:
            response = self.session.post(
                f'{self.api_url}/api/goodsprice/update',
                json={
                    'goods_id': goods_id,
                    'spec_index': spec_index,
                    'price': price
                },
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Updated price for goods {goods_id} spec {spec_index}: {price}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to update price: {e}")
            return False


class IphoneMappingService:
    """iPhone 商品映射服务"""

    # 容量关键字映射 (外部描述 -> GB)
    CAPACITY_PATTERNS = {
        r'(\d+)\s*TB': lambda m: int(m.group(1)) * 1024,  # 1TB -> 1024GB
        r'(\d+)\s*GB': lambda m: int(m.group(1)),         # 256GB -> 256GB
    }

    # 颜色名称映射规则 (如需要可以扩展)
    COLOR_MAPPINGS = {
        # 外部颜色名 -> 本项目颜色名
        # 示例:
        # 'Space Black': 'スペースブラック',
        # 可以根据实际情况添加更多映射
    }

    def __init__(self):
        """初始化映射服务"""
        # 预加载所有 Iphone 实例以提高查询效率
        self.iphones_cache = self._build_iphone_cache()

    def _build_iphone_cache(self) -> Dict[Tuple[str, int, str], Iphone]:
        """
        构建 Iphone 缓存,以 (model_name, capacity_gb, color) 为键

        Returns:
            Iphone实例缓存字典
        """
        cache = {}
        for iphone in Iphone.objects.all():
            key = (iphone.model_name, iphone.capacity_gb, iphone.color)
            cache[key] = iphone

        logger.info(f"Built iPhone cache with {len(cache)} entries")
        return cache

    def parse_capacity_from_title(self, title: str) -> Optional[int]:
        """
        从标题中提取容量

        Args:
            title: 商品标题,如 "iPhone Air 1TB"

        Returns:
            容量(GB)或None
        """
        for pattern, converter in self.CAPACITY_PATTERNS.items():
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return converter(match)
        return None

    def normalize_color(self, external_color: str) -> str:
        """
        规范化颜色名称

        Args:
            external_color: 外部颜色名

        Returns:
            规范化后的颜色名
        """
        # 首先检查是否有映射
        if external_color in self.COLOR_MAPPINGS:
            return self.COLOR_MAPPINGS[external_color]

        # 如果没有映射,直接返回原始值
        return external_color

    def find_matching_iphone(
        self,
        model_name: str,
        capacity_gb: Optional[int],
        color: str
    ) -> Tuple[Optional[Iphone], float]:
        """
        查找匹配的 Iphone 实例

        Args:
            model_name: 机型名称
            capacity_gb: 容量(GB)
            color: 颜色

        Returns:
            (匹配的Iphone实例, 置信度分数) 元组
        """
        # 规范化颜色
        normalized_color = self.normalize_color(color)

        # 精确匹配
        if capacity_gb:
            key = (model_name, capacity_gb, normalized_color)
            if key in self.iphones_cache:
                return self.iphones_cache[key], 1.0

        # 模糊匹配: 尝试不同的匹配策略
        # 策略1: 只匹配机型和容量
        if capacity_gb:
            for (m, c, col), iphone in self.iphones_cache.items():
                if m == model_name and c == capacity_gb:
                    return iphone, 0.7  # 中等置信度

        # 策略2: 只匹配机型和颜色
        for (m, c, col), iphone in self.iphones_cache.items():
            if m == model_name and col == normalized_color:
                return iphone, 0.5  # 较低置信度

        # 策略3: 只匹配机型
        for (m, c, col), iphone in self.iphones_cache.items():
            if m == model_name:
                return iphone, 0.3  # 低置信度

        return None, 0.0

    def map_external_good_to_iphone(
        self,
        external_good: Dict
    ) -> Tuple[Optional[Iphone], float, Dict]:
        """
        将外部商品映射到本项目的 Iphone 实例

        Args:
            external_good: 外部商品数据

        Returns:
            (匹配的Iphone实例, 置信度, 解析的数据) 元组
        """
        # 提取外部商品信息
        title = external_good.get('title', '')
        category_three_name = external_good.get('category_three_name', '')
        spec_name = external_good.get('spec_name', '')

        # 使用 category_three_name 作为机型名
        # 例如: "iPhone Air" -> "iPhone Air"
        model_name = category_three_name

        # 从 title 中提取容量
        capacity_gb = self.parse_capacity_from_title(title)

        # 使用 spec_name 作为颜色
        color = spec_name

        # 查找匹配的 Iphone
        iphone, confidence = self.find_matching_iphone(
            model_name,
            capacity_gb,
            color
        )

        parsed_data = {
            'model_name': model_name,
            'capacity_gb': capacity_gb,
            'color': color,
        }

        return iphone, confidence, parsed_data


class ExternalGoodsSyncService:
    """外部商品同步服务的主类"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        db_path: str = "auto_price.sqlite3",
        category_filter: Optional[str] = None
    ):
        """
        初始化同步服务

        Args:
            api_url: 外部API URL (默认从settings获取)
            api_token: 外部API token (默认从settings获取)
            db_path: SQLite数据库路径
            category_filter: 商品类别过滤 (如 'iPhone')，只同步指定类别的商品
        """
        self.api_url = api_url or getattr(
            settings,
            'EXTERNAL_GOODS_API_URL',
            'http://localhost:8080'
        )
        self.api_token = api_token or getattr(
            settings,
            'EXTERNAL_GOODS_API_TOKEN',
            ''
        )
        self.category_filter = category_filter or getattr(
            settings,
            'EXTERNAL_GOODS_CATEGORY_FILTER',
            None
        )

        self.client = ExternalGoodsClient(self.api_url, self.api_token)
        self.mapper = IphoneMappingService()
        self.db_manager = AutoPriceSQLiteManager(db_path)

    def sync_goods_mappings(self) -> Dict:
        """
        同步外部商品映射

        Returns:
            同步统计信息
        """
        # 创建同步记录
        sync_id = self.db_manager.create_sync_record('full_sync')

        try:
            # 获取外部商品列表
            goods_list = self.client.fetch_goods_list()

            stats = {
                'total_items': len(goods_list),
                'matched_items': 0,
                'unmatched_items': 0,
                'error_items': 0,
                'skipped_items': 0,
            }

            # 处理每个商品
            for idx, good in enumerate(goods_list):
                try:
                    # 验证 good 是字典
                    if not isinstance(good, dict):
                        logger.error(f"Item at index {idx} is not a dict: {type(good)}, value: {good}")
                        stats['error_items'] += 1
                        continue

                    # 类别过滤
                    if self.category_filter:
                        category_name = good.get('category_name', '')
                        if category_name != self.category_filter:
                            stats['skipped_items'] += 1
                            continue

                    self._process_single_good(good, stats)
                except Exception as e:
                    goods_id = good.get('goods_id', 'unknown') if isinstance(good, dict) else f'index-{idx}'
                    logger.error(f"Error processing good {goods_id}: {e}", exc_info=True)
                    stats['error_items'] += 1

            # 更新同步记录
            self.db_manager.update_sync_record(
                sync_id,
                status='completed',
                stats=stats
            )

            logger.info(f"Sync completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.db_manager.update_sync_record(
                sync_id,
                status='failed',
                error_message=str(e)
            )
            raise

    def _process_single_good(self, good: Dict, stats: Dict):
        """
        处理单个商品

        Args:
            good: 商品数据
            stats: 统计信息字典(会被修改)
        """
        # 映射商品到 Iphone
        iphone, confidence, parsed_data = self.mapper.map_external_good_to_iphone(good)

        # 准备映射数据
        mapping_data = {
            'external_goods_id': good.get('goods_id'),
            'external_spec_index': good.get('spec_index', 0),
            'iphone_id': iphone.id if iphone else None,
            'external_title': good.get('title', ''),
            'external_spec_name': good.get('spec_name', ''),
            'external_category_name': good.get('category_name', ''),
            'external_category_second_name': good.get('category_second_name', ''),
            'external_category_three_name': good.get('category_three_name', ''),
            'external_price': good.get('price'),
            'model_name': parsed_data.get('model_name'),
            'capacity_gb': parsed_data.get('capacity_gb'),
            'color': parsed_data.get('color'),
            'confidence_score': confidence,
            'sync_status': 'matched' if iphone else 'unmatched',
            'error_message': None,
        }

        # 保存映射
        self.db_manager.upsert_mapping(mapping_data)

        # 更新统计
        if iphone:
            stats['matched_items'] += 1
        else:
            stats['unmatched_items'] += 1

    def get_mapping_statistics(self) -> Dict:
        """获取映射统计信息"""
        return self.db_manager.get_mapping_statistics()

    def get_all_mappings(self, status: Optional[str] = None) -> List[Dict]:
        """获取所有映射"""
        return self.db_manager.get_all_mappings(status=status)

    def update_external_price(
        self,
        iphone_id: int,
        new_price: int
    ) -> Dict:
        """
        根据 Iphone ID 更新外部项目中的对应商品价格

        Args:
            iphone_id: 本项目的 Iphone ID
            new_price: 新价格

        Returns:
            更新结果统计
        """
        # 查找该 Iphone 的所有映射
        mappings = self.db_manager.get_mappings_by_iphone_id(iphone_id)

        results = {
            'total': len(mappings),
            'success': 0,
            'failed': 0,
            'details': []
        }

        for mapping in mappings:
            goods_id = mapping['external_goods_id']
            spec_index = mapping['external_spec_index']

            success = self.client.update_goods_price(goods_id, spec_index, new_price)

            if success:
                results['success'] += 1
            else:
                results['failed'] += 1

            results['details'].append({
                'goods_id': goods_id,
                'spec_index': spec_index,
                'success': success
            })

        return results
