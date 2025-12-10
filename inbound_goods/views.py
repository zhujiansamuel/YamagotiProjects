# -*- coding: utf-8 -*-
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib.contenttypes.models import ContentType
from django.db.models import Q
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt
import json

from .models import InboundInventory, InventoryStatusHistory
from AppleStockChecker.models import Iphone


def inventory_management(request):
    """
    在库管理主页面
    支持状态过滤、搜索、分页
    """
    # 获取查询参数
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('q', '')
    page_number = request.GET.get('page', 1)

    # 基础查询
    inventories = InboundInventory.objects.select_related(
        'product_content_type'
    ).all()

    # 状态过滤
    if status_filter:
        inventories = inventories.filter(status=status_filter)

    # 搜索
    if search_query:
        inventories = inventories.filter(
            Q(unique_code__icontains=search_query) |
            Q(special_description__icontains=search_query) |
            Q(abnormal_remark__icontains=search_query)
        )

    # 排序
    inventories = inventories.order_by('-created_at')

    # 分页
    paginator = Paginator(inventories, 20)
    page_obj = paginator.get_page(page_number)

    # 获取所有状态选项
    status_choices = InboundInventory.InventoryStatus.choices

    # 统计各状态数量
    status_stats = {}
    for status_code, status_label in status_choices:
        count = InboundInventory.objects.filter(status=status_code).count()
        status_stats[status_code] = {
            'label': status_label,
            'count': count
        }

    context = {
        'page_obj': page_obj,
        'status_choices': status_choices,
        'status_stats': status_stats,
        'current_status': status_filter,
        'search_query': search_query,
        'total_count': inventories.count(),
    }

    return render(request, 'inbound_goods/inventory_management.html', context)


@require_http_methods(["GET"])
def api_get_products(request):
    """
    API: 获取商品列表（目前只有 iPhone）
    用于动态填充商品选择下拉框
    """
    product_type = request.GET.get('type', 'iphone')

    if product_type == 'iphone':
        iphones = Iphone.objects.all().order_by('-release_date', 'model_name')

        products = []
        for iphone in iphones:
            # 格式化容量显示
            capacity = (
                f"{iphone.capacity_gb // 1024}TB"
                if iphone.capacity_gb % 1024 == 0
                else f"{iphone.capacity_gb}GB"
            )

            products.append({
                'id': iphone.id,
                'part_number': iphone.part_number,
                'display_name': f"{iphone.model_name} {capacity} {iphone.color}",
                'model_name': iphone.model_name,
                'capacity': capacity,
                'color': iphone.color,
            })

        return JsonResponse({
            'success': True,
            'products': products,
            'content_type_id': ContentType.objects.get_for_model(Iphone).id,
        })

    return JsonResponse({
        'success': False,
        'error': '不支持的商品类型'
    }, status=400)


@require_POST
@csrf_exempt  # 在生产环境中应该使用 CSRF token
def api_create_inventory(request):
    """
    API: 创建新的库存记录
    """
    try:
        data = json.loads(request.body)

        # 验证必填字段
        required_fields = ['unique_code', 'product_content_type', 'product_object_id', 'status']
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    'success': False,
                    'error': f'缺少必填字段: {field}'
                }, status=400)

        # 检查唯一编码是否已存在
        if InboundInventory.objects.filter(unique_code=data['unique_code']).exists():
            return JsonResponse({
                'success': False,
                'error': '商品编码已存在'
            }, status=400)

        # 获取 ContentType
        content_type = ContentType.objects.get(id=data['product_content_type'])

        # 创建库存记录
        inventory = InboundInventory.objects.create(
            product_content_type=content_type,
            product_object_id=data['product_object_id'],
            unique_code=data['unique_code'],
            status=data['status'],
            special_description=data.get('special_description', ''),
            reserved_arrival_time=data.get('reserved_arrival_time'),
            abnormal_remark=data.get('abnormal_remark', ''),
        )

        return JsonResponse({
            'success': True,
            'message': '库存记录创建成功',
            'inventory_id': inventory.id,
            'unique_code': inventory.unique_code,
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': '无效的 JSON 数据'
        }, status=400)
    except ContentType.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': '无效的商品类型'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_POST
@csrf_exempt  # 在生产环境中应该使用 CSRF token
def api_bulk_update_status(request):
    """
    API: 批量更新库存状态
    """
    try:
        data = json.loads(request.body)

        # 验证必填字段
        if 'inventory_ids' not in data or 'new_status' not in data:
            return JsonResponse({
                'success': False,
                'error': '缺少必填字段: inventory_ids 或 new_status'
            }, status=400)

        inventory_ids = data['inventory_ids']
        new_status = data['new_status']
        change_reason = data.get('change_reason', '批量状态更新')

        # 验证状态值
        valid_statuses = [choice[0] for choice in InboundInventory.InventoryStatus.choices]
        if new_status not in valid_statuses:
            return JsonResponse({
                'success': False,
                'error': '无效的状态值'
            }, status=400)

        # 获取要更新的库存记录
        inventories = InboundInventory.objects.filter(id__in=inventory_ids)

        if not inventories.exists():
            return JsonResponse({
                'success': False,
                'error': '未找到要更新的库存记录'
            }, status=404)

        # 批量更新
        updated_count = 0
        errors = []

        for inventory in inventories:
            old_status = inventory.status

            # 只更新状态不同的记录
            if old_status != new_status:
                try:
                    inventory.status = new_status

                    # 如果是预订类状态，需要设置到货时间
                    if new_status in [
                        'CORPORATE_RESERVED_ARRIVAL',
                        'PERSONAL_RESERVED_ARRIVAL',
                        'PURCHASE_RESERVED_ARRIVAL'
                    ]:
                        if 'reserved_arrival_time' in data:
                            inventory.reserved_arrival_time = data['reserved_arrival_time']

                    # 如果是异常状态，需要设置备注
                    if new_status == 'STATUS_ABNORMAL':
                        if 'abnormal_remark' in data:
                            inventory.abnormal_remark = data['abnormal_remark']

                    inventory.save()

                    # 手动创建历史记录（带自定义原因）
                    InventoryStatusHistory.objects.create(
                        inventory=inventory,
                        old_status=old_status,
                        new_status=new_status,
                        change_reason=change_reason,
                        changed_by=data.get('changed_by', 'system')
                    )

                    updated_count += 1

                except Exception as e:
                    errors.append({
                        'unique_code': inventory.unique_code,
                        'error': str(e)
                    })

        return JsonResponse({
            'success': True,
            'message': f'成功更新 {updated_count} 条记录',
            'updated_count': updated_count,
            'errors': errors if errors else None,
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': '无效的 JSON 数据'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def api_get_inventory_detail(request, inventory_id):
    """
    API: 获取单个库存记录的详细信息
    """
    try:
        inventory = get_object_or_404(
            InboundInventory.objects.select_related('product_content_type'),
            id=inventory_id
        )

        # 获取历史记录
        history = inventory.status_history.all()[:10]

        history_data = []
        for record in history:
            history_data.append({
                'old_status': record.old_status,
                'old_status_display': record.get_old_status_display() if record.old_status else '初始创建',
                'new_status': record.new_status,
                'new_status_display': record.get_new_status_display(),
                'changed_at': record.changed_at.strftime('%Y-%m-%d %H:%M:%S'),
                'change_reason': record.change_reason,
                'changed_by': record.changed_by,
            })

        data = {
            'success': True,
            'inventory': {
                'id': inventory.id,
                'unique_code': inventory.unique_code,
                'product_name': str(inventory.product) if inventory.product else '未关联',
                'status': inventory.status,
                'status_display': inventory.get_status_display(),
                'special_description': inventory.special_description,
                'reserved_arrival_time': inventory.reserved_arrival_time.strftime('%Y-%m-%d %H:%M') if inventory.reserved_arrival_time else None,
                'abnormal_remark': inventory.abnormal_remark,
                'created_at': inventory.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': inventory.updated_at.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'history': history_data,
        }

        return JsonResponse(data)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
