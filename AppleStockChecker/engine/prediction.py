"""
LightGBM 价格预测模块。
- 训练: 从 ClickHouse features_wide 读取特征, 训练 4 个 horizon 的 LightGBM 回归器
- 推理: 加载最新 ModelArtifact, 预测未来 15/30/45/60 分钟均价
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from AppleStockChecker.engine.config import FEATURE_WINDOWS

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────

MODEL_NAME_PREFIX = "lgbm_price"
MODEL_VERSION = "v1"

HORIZONS = [15, 30, 45, 60]  # 分钟
HORIZON_BUCKETS = [h // 15 for h in HORIZONS]  # 1, 2, 3, 4 buckets

TRAIN_DAYS_DEFAULT = 14
HOLDOUT_RATIO = 0.2  # 最后 20% 用作验证

# 从 features_wide 读取的列 (除 bucket/scope 外)
FEATURE_COLUMNS: list[str] = [
    "mean", "median", "std", "shop_count", "dispersion",
]
for _w in FEATURE_WINDOWS:
    FEATURE_COLUMNS += [f"ema_{_w}", f"sma_{_w}", f"wma_{_w}"]
    FEATURE_COLUMNS += [
        f"boll_mid_{_w}", f"boll_up_{_w}", f"boll_low_{_w}", f"boll_width_{_w}",
    ]
    FEATURE_COLUMNS += [f"logb_{_w}"]
FEATURE_COLUMNS += ["ema_hl_30", "ema_hl_60"]

LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}
LGB_NUM_ROUNDS = 500
LGB_EARLY_STOPPING = 50


# ── 数据获取 ──────────────────────────────────────────────────────────

def _fetch_features_df(
    ch_service,
    iphone_id: int,
    run_id: str,
    bucket_gte: datetime,
    bucket_lte: datetime,
) -> pd.DataFrame:
    """从 ClickHouse features_wide 读取单机型特征, 返回 DataFrame (按 bucket 排序)。"""
    scope = f"iphone:{iphone_id}"
    rows, _ = ch_service.query_features(
        run_id=run_id,
        scope=scope,
        bucket_gte=bucket_gte,
        bucket_lte=bucket_lte,
        columns=FEATURE_COLUMNS,
        limit=0,  # 不限制
        need_total=False,
        ordering="bucket",
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["bucket"] = pd.to_datetime(df["bucket"])
    df = df.sort_values("bucket").reset_index(drop=True)
    return df


# ── 标签构造 ──────────────────────────────────────────────────────────

def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """对 mean 列做负向 shift, 构造未来 1/2/3/4 个 bucket 的目标列。"""
    for hb, hm in zip(HORIZON_BUCKETS, HORIZONS):
        df[f"target_{hm}"] = df["mean"].shift(-hb)
    return df


# ── 训练 ──────────────────────────────────────────────────────────────

def train_models_for_iphone(
    ch_service,
    iphone_id: int,
    *,
    run_id: str = "live",
    train_days: int = TRAIN_DAYS_DEFAULT,
    bucket_lte: datetime | None = None,
) -> dict:
    """训练 4 个 horizon 的 LightGBM 模型。

    Returns
    -------
    dict  包含:
      models:    {horizon_min: lgb.Booster}
      metrics:   {horizon_min: {mae, rmse, n_train, n_val}}
      train_end: datetime
    """
    now = bucket_lte or datetime.utcnow()
    bucket_gte = now - timedelta(days=train_days)

    logger.info(
        "Fetching features for iphone=%d, range=[%s, %s], run_id=%s",
        iphone_id, bucket_gte, now, run_id,
    )
    df = _fetch_features_df(ch_service, iphone_id, run_id, bucket_gte, now)
    if df.empty or len(df) < 20:
        raise ValueError(
            f"Insufficient data for iphone {iphone_id}: {len(df)} rows (need >=20)"
        )

    df = _build_targets(df)
    # 丢弃最后几行 (target 为 NaN)
    df = df.dropna(subset=[f"target_{h}" for h in HORIZONS])

    if len(df) < 20:
        raise ValueError(
            f"Insufficient rows after target shift for iphone {iphone_id}: {len(df)}"
        )

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].astype(float)
    # 用中位数填充 NaN (features_wide 中 std/dispersion 可能为 null)
    X = X.fillna(X.median())

    split_idx = int(len(X) * (1 - HOLDOUT_RATIO))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]

    models: dict[int, lgb.Booster] = {}
    metrics: dict[int, dict] = {}

    for horizon in HORIZONS:
        target_col = f"target_{horizon}"
        y = df[target_col].astype(float)
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        bst = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )

        y_pred = bst.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        logger.info(
            "iphone=%d horizon=%dmin  MAE=%.2f  RMSE=%.2f  (train=%d, val=%d, rounds=%d)",
            iphone_id, horizon, mae, rmse, len(X_train), len(X_val), bst.best_iteration,
        )

        models[horizon] = bst
        metrics[horizon] = {
            "mae": mae,
            "rmse": rmse,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": bst.best_iteration,
        }

    return {
        "models": models,
        "metrics": metrics,
        "train_end": now,
        "feature_columns": feature_cols,
    }


def save_artifacts(
    iphone_id: int,
    train_result: dict,
    version: str = MODEL_VERSION,
) -> list:
    """将训练结果持久化到 ModelArtifact。"""
    from AppleStockChecker.models import ModelArtifact

    artifacts = []
    for horizon, bst in train_result["models"].items():
        model_name = f"{MODEL_NAME_PREFIX}_{iphone_id}_h{horizon}"
        blob = pickle.dumps({
            "booster": bst,
            "feature_columns": train_result["feature_columns"],
        })
        artifact, _created = ModelArtifact.objects.update_or_create(
            model_name=model_name,
            version=version,
            defaults={
                "trained_at": train_result["train_end"],
                "params_blob": blob,
                "metrics_json": {
                    **train_result["metrics"][horizon],
                    "iphone_id": iphone_id,
                    "horizon_min": horizon,
                    "train_days": TRAIN_DAYS_DEFAULT,
                    "lgb_params": LGB_PARAMS,
                },
            },
        )
        artifacts.append(artifact)

    return artifacts


# ── 推理 ──────────────────────────────────────────────────────────────

def load_model(iphone_id: int, horizon: int, version: str = MODEL_VERSION) -> dict:
    """从 ModelArtifact 加载模型。"""
    from AppleStockChecker.models import ModelArtifact

    model_name = f"{MODEL_NAME_PREFIX}_{iphone_id}_h{horizon}"
    artifact = ModelArtifact.objects.filter(
        model_name=model_name, version=version
    ).order_by("-trained_at").first()
    if artifact is None:
        raise FileNotFoundError(f"No artifact found: {model_name} {version}")
    payload = pickle.loads(artifact.params_blob)
    return {
        "booster": payload["booster"],
        "feature_columns": payload["feature_columns"],
        "artifact": artifact,
    }


def predict_for_iphone(
    ch_service,
    iphone_id: int,
    *,
    run_id: str = "live",
    bucket: datetime | None = None,
    version: str = MODEL_VERSION,
) -> list[dict]:
    """对单机型做 4 个 horizon 的推理, 返回预测结果列表。"""
    now = bucket or datetime.utcnow()
    # 需要最近一些 bucket 的特征 (最大窗口 1800min = 30h, 但通常几个小时够了)
    lookback = timedelta(hours=6)
    df = _fetch_features_df(ch_service, iphone_id, run_id, now - lookback, now)

    if df.empty:
        raise ValueError(f"No feature data for iphone {iphone_id} near {now}")

    # 取最新一行
    latest = df.iloc[[-1]]
    latest_bucket = latest["bucket"].iloc[0]

    results = []
    for horizon in HORIZONS:
        loaded = load_model(iphone_id, horizon, version)
        bst = loaded["booster"]
        feat_cols = loaded["feature_columns"]

        X = latest[feat_cols].astype(float).fillna(0)
        yhat = float(bst.predict(X)[0])

        results.append({
            "bucket": latest_bucket,
            "iphone_id": iphone_id,
            "horizon_min": horizon,
            "yhat": yhat,
            "model_name": f"{MODEL_NAME_PREFIX}_{iphone_id}_h{horizon}",
            "version": version,
        })

    return results


def save_forecasts(predictions: list[dict]) -> int:
    """将预测结果写入 ForecastSnapshot。"""
    from AppleStockChecker.models import ForecastSnapshot, Iphone

    created = 0
    for p in predictions:
        iphone = Iphone.objects.get(pk=p["iphone_id"])
        _, was_created = ForecastSnapshot.objects.update_or_create(
            bucket=p["bucket"],
            model_name=p["model_name"],
            version=p["version"],
            horizon_min=p["horizon_min"],
            iphone=iphone,
            defaults={
                "yhat": p["yhat"],
                "yhat_var": None,
                "is_final": False,
            },
        )
        if was_created:
            created += 1
    return created
