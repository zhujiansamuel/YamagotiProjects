# -*- coding: utf-8 -*-
"""
AutoML Celery Tasks
机器学习和深度学习分析任务
"""
from __future__ import annotations
from celery import shared_task
import logging

logger = logging.getLogger(__name__)


@shared_task(name="automl.preprocessing_rapid", queue="automl_preprocessing")
def preprocessing_rapid():
    """
    Preprocessing-Rapid Task
    快速数据预处理任务
    """
    logger.info("Preprocessing-Rapid task started")
    print("Preprocessing-Rapid")
    logger.info("Preprocessing-Rapid task completed")
    return {"status": "success", "task": "Preprocessing-Rapid"}


@shared_task(name="automl.cause_and_effect_testing", queue="automl_cause_effect")
def cause_and_effect_testing():
    """
    Cause-and-Effect-Testing Task
    因果关系测试任务
    """
    logger.info("Cause-and-Effect-Testing task started")
    print("Cause-and-Effect-Testing")
    logger.info("Cause-and-Effect-Testing task completed")
    return {"status": "success", "task": "Cause-and-Effect-Testing"}


@shared_task(name="automl.quantification_of_impact", queue="automl_impact")
def quantification_of_impact():
    """
    Quantification-of-Impact Task
    影响量化任务
    """
    logger.info("Quantification-of-Impact task started")
    print("Quantification-of-Impact")
    logger.info("Quantification-of-Impact task completed")
    return {"status": "success", "task": "Quantification-of-Impact"}
