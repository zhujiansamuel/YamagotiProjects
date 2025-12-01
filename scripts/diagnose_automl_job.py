#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoML Job 诊断脚本
用于检查为什么 VAR 模型被跳过（数据不足）
"""
import os
import sys
import django

# Django setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'YamagotiProjects.settings')
django.setup()

from AppleStockChecker.models import (
    AutomlCausalJob,
    AutomlPreprocessedSeries,
    PurchasingShopTimeAnalysis,
)
import pandas as pd


def diagnose_job(job_id):
    """诊断 AutoML Job 的数据情况"""
    try:
        job = AutomlCausalJob.objects.get(pk=job_id)
    except AutomlCausalJob.DoesNotExist:
        print(f"❌ Job ID {job_id} 不存在")
        return

    print("=" * 80)
    print(f"AutoML Job 诊断报告 - Job ID: {job_id}")
    print("=" * 80)
    print()

    # 基本信息
    print("📋 基本信息:")
    print(f"  iPhone: {job.iphone.part_number} ({job.iphone.model_name or 'N/A'})")
    print(f"  时间窗口: {job.window_start} 到 {job.window_end}")
    print(f"  时间桶大小: {job.bucket_freq}")
    print(f"  创建时间: {job.created_at}")
    print()

    # 阶段状态
    print("📊 处理阶段状态:")
    print(f"  预处理 (Preprocessing): {job.preprocessing_status}")
    print(f"  因果检验 (VAR): {job.cause_effect_status}")
    print(f"  影响量化 (Impact): {job.impact_status}")
    if job.last_error:
        print(f"  最后错误: {job.last_error[:200]}...")
    print()

    # 检查原始 PSTA 数据
    print("🔍 原始数据检查 (PurchasingShopTimeAnalysis):")
    psta_count = PurchasingShopTimeAnalysis.objects.filter(
        iphone=job.iphone,
        Timestamp_Time__gte=job.window_start,
        Timestamp_Time__lt=job.window_end,
    ).count()
    print(f"  PSTA 记录数: {psta_count}")

    if psta_count > 0:
        # 检查涉及的店铺数量
        shops = PurchasingShopTimeAnalysis.objects.filter(
            iphone=job.iphone,
            Timestamp_Time__gte=job.window_start,
            Timestamp_Time__lt=job.window_end,
        ).values_list('shop__name', flat=True).distinct()
        print(f"  涉及店铺数: {len(shops)}")
        print(f"  店铺列表: {', '.join(list(shops)[:10])}")
        if len(shops) > 10:
            print(f"  ... 还有 {len(shops) - 10} 个店铺")
    print()

    # 检查预处理数据
    print("🔬 预处理数据检查 (AutomlPreprocessedSeries):")
    series = AutomlPreprocessedSeries.objects.filter(job=job)
    series_count = series.count()
    print(f"  预处理序列数: {series_count}")

    if series_count == 0:
        print("  ❌ 没有预处理数据！请检查预处理阶段是否成功。")
        return

    # 构建 Panel 数据
    df = pd.DataFrame.from_records(
        series.values('shop_id', 'bucket_ts', 'z_dlog_price')
    )

    panel = df.pivot_table(
        index='bucket_ts',
        columns='shop_id',
        values='z_dlog_price',
    ).sort_index()

    print(f"  Panel 形状 (dropna 之前): {panel.shape}")
    print(f"    - 时间点数 (T): {panel.shape[0]}")
    print(f"    - 店铺数 (S): {panel.shape[1]}")
    print()

    # 检查缺失值
    print("  缺失值统计 (每个店铺):")
    missing = panel.isnull().sum()
    for shop_id, count in missing.items():
        pct = (count / len(panel)) * 100
        print(f"    Shop {shop_id}: {count}/{len(panel)} ({pct:.1f}% 缺失)")
    print()

    # dropna 后的数据
    panel_clean = panel.dropna(how='any')
    print(f"  Panel 形状 (dropna 之后): {panel_clean.shape}")
    print(f"    - 时间点数 (T): {panel_clean.shape[0]}")
    print(f"    - 店铺数 (S): {panel_clean.shape[1]}")
    print()

    # VAR 模型要求
    print("✅ VAR 模型要求:")
    print(f"  最少时间点数: 20")
    print(f"  最少店铺数: 2")
    print()

    # 诊断结果
    print("🎯 诊断结果:")
    issues = []

    if panel_clean.shape[0] < 20:
        issues.append(f"时间点不足: {panel_clean.shape[0]} < 20")
        print(f"  ❌ 时间点不足: {panel_clean.shape[0]} < 20 (需要至少 20 个)")
    else:
        print(f"  ✅ 时间点充足: {panel_clean.shape[0]} ≥ 20")

    if panel_clean.shape[1] < 2:
        issues.append(f"店铺数不足: {panel_clean.shape[1]} < 2")
        print(f"  ❌ 店铺数不足: {panel_clean.shape[1]} < 2 (需要至少 2 个)")
    else:
        print(f"  ✅ 店铺数充足: {panel_clean.shape[1]} ≥ 2")
    print()

    # 建议
    if issues:
        print("💡 建议:")
        if panel_clean.shape[0] < 20:
            print("  1. 增加时间窗口（尝试 14 天或 30 天）")
            print("  2. 减小时间桶大小（例如从 10min 改为 5min 或 15min）")
        if panel_clean.shape[1] < 2:
            print("  3. 选择其他有更多店铺数据的机型")
            print("  4. 检查是否所有店铺的数据都有完整的时间序列")

        if panel.shape != panel_clean.shape:
            print("  5. 数据存在大量缺失值，考虑改进数据填充策略")
    else:
        print("  ✅ 数据充足，应该可以成功运行 VAR 模型")
        print("  如果仍然失败，请检查其他错误日志")

    print()
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_automl_job.py <job_id>")
        print("示例: python diagnose_automl_job.py 2")
        sys.exit(1)

    job_id = int(sys.argv[1])
    diagnose_job(job_id)
