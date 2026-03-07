# Price Prediction Models - Implementation Plan

## Overview
Add LightGBM-based price prediction for iPhone cross-shop mean prices.
- **Data Source**: ClickHouse `features_wide` table (scope=`iphone:{id}`)
- **Target**: `mean` column (cross-shop average price)
- **Horizons**: 15, 30, 45, 60 minutes ahead
- **Training window**: 14 days of historical data
- **Algorithm**: LightGBM regression (one model per iPhone × horizon)

## Steps

### Step 1: Add Dependencies
- Add `lightgbm` and `scikit-learn` to `requirements.txt`

### Step 2: Add `iphone` field to ForecastSnapshot
- Current `ForecastSnapshot` has no iPhone/scope identifier
- Add `iphone = ForeignKey(Iphone, null=True)` field
- Update `unique_together` to include `iphone`
- Create Django migration

### Step 3: Create `engine/prediction.py` - Core ML Logic
Training function:
- Query CH `features_wide` for scope=`iphone:{id}`, last 14 days
- Build feature matrix from: all EMA, SMA, WMA, Bollinger, logb columns + `mean`, `median`, `std`, `dispersion`, `shop_count`
- Create targets: `mean` shifted by -1/-2/-3/-4 buckets (15/30/45/60 min)
- Train 4 LightGBM regressors (one per horizon)
- Evaluate with MAE/RMSE on holdout (last 20%)
- Serialize models + metrics → `ModelArtifact`

Inference function:
- Load latest `ModelArtifact` for given iPhone
- Query latest features from CH
- Predict 4 horizons
- Write to `ForecastSnapshot`

### Step 4: Create Management Command `train_price_model.py`
- Args: `--iphone-ids` (comma-sep or "all"), `--days` (default 14), `--run-id` (default "live")
- Calls training function per iPhone
- Logs metrics

### Step 5: Create Management Command `predict_prices.py`
- Args: `--iphone-ids`, `--run-id`, `--bucket` (default: latest)
- Calls inference function per iPhone
- Writes ForecastSnapshot records

### Step 6: Add Celery Tasks
- `train_price_models_task(iphone_ids, days, run_id)`
- `predict_prices_task(iphone_ids, run_id)`
- In `AppleStockChecker/tasks/prediction_tasks.py`

### Step 7: Add API Endpoints
- `ForecastSnapshotViewSet` - list/filter predictions (GET only)
- `ModelArtifactViewSet` - list trained models (GET only)
- `TriggerTrainingView` - POST to trigger async training
- `TriggerPredictionView` - POST to trigger async inference
- Register in `urls.py`

## Feature Columns Used (from `features_wide`)
```
mean, median, std, shop_count, dispersion,
ema_30, ema_60, ema_75, ema_120, ema_900, ema_1800,
sma_30, sma_60, sma_75, sma_120, sma_900, sma_1800,
wma_30, wma_60, wma_75, wma_120, wma_900, wma_1800,
ema_hl_30, ema_hl_60,
boll_mid_30, boll_up_30, boll_low_30, boll_width_30,
boll_mid_60, boll_up_60, boll_low_60, boll_width_60,
boll_mid_75, boll_up_75, boll_low_75, boll_width_75,
boll_mid_120, boll_up_120, boll_low_120, boll_width_120,
boll_mid_900, boll_up_900, boll_low_900, boll_width_900,
boll_mid_1800, boll_up_1800, boll_low_1800, boll_width_1800,
logb_30, logb_60, logb_75, logb_120, logb_900, logb_1800
```

## Files to Create/Modify
| File | Action |
|------|--------|
| `requirements.txt` | Add lightgbm, scikit-learn |
| `AppleStockChecker/models.py` | Add iphone FK to ForecastSnapshot |
| `AppleStockChecker/migrations/XXXX_*.py` | Auto-generated migration |
| `AppleStockChecker/engine/prediction.py` | **NEW** - core training/inference |
| `AppleStockChecker/management/commands/train_price_model.py` | **NEW** |
| `AppleStockChecker/management/commands/predict_prices.py` | **NEW** |
| `AppleStockChecker/tasks/prediction_tasks.py` | **NEW** |
| `AppleStockChecker/views.py` | Add ViewSets |
| `AppleStockChecker/urls.py` | Register routes |
