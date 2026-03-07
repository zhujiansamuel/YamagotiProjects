"""
Celery tasks for LightGBM price prediction: training & inference.
"""
from __future__ import annotations

import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="prediction.train_models", max_retries=1)
def train_price_models_task(self, iphone_ids: list[int], days: int = 14, run_id: str = "live", version: str = "v1"):
    """异步训练 LightGBM 价格预测模型。"""
    from AppleStockChecker.engine.prediction import train_models_for_iphone, save_artifacts
    from AppleStockChecker.services.clickhouse_service import ClickHouseService

    ch = ClickHouseService()
    results = {"ok": 0, "fail": 0, "details": {}}

    for iid in iphone_ids:
        try:
            result = train_models_for_iphone(ch, iid, run_id=run_id, train_days=days)
            save_artifacts(iid, result, version=version)
            results["details"][iid] = {
                "status": "ok",
                "metrics": result["metrics"],
            }
            results["ok"] += 1
        except Exception as e:
            logger.exception("train_price_models_task: failed iphone=%d", iid)
            results["details"][iid] = {"status": "error", "error": str(e)}
            results["fail"] += 1

    logger.info("train_price_models_task done: %d ok, %d fail", results["ok"], results["fail"])
    return results


@shared_task(bind=True, name="prediction.predict_prices", max_retries=1)
def predict_prices_task(self, iphone_ids: list[int], run_id: str = "live", version: str = "v1"):
    """异步推理: 对指定机型做价格预测。"""
    from AppleStockChecker.engine.prediction import predict_for_iphone, save_forecasts
    from AppleStockChecker.services.clickhouse_service import ClickHouseService

    ch = ClickHouseService()
    results = {"ok": 0, "fail": 0, "total_forecasts": 0, "details": {}}

    for iid in iphone_ids:
        try:
            preds = predict_for_iphone(ch, iid, run_id=run_id, version=version)
            n_new = save_forecasts(preds)
            results["details"][iid] = {
                "status": "ok",
                "predictions": len(preds),
                "new": n_new,
            }
            results["total_forecasts"] += len(preds)
            results["ok"] += 1
        except Exception as e:
            logger.exception("predict_prices_task: failed iphone=%d", iid)
            results["details"][iid] = {"status": "error", "error": str(e)}
            results["fail"] += 1

    logger.info("predict_prices_task done: %d ok, %d fail", results["ok"], results["fail"])
    return results
