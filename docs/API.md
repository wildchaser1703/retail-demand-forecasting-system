# API Documentation

## Overview

This document describes the API structure for the Retail Demand Forecasting System. While a full FastAPI implementation is planned for future releases, this document outlines the intended API design and current programmatic interfaces.

---

## Programmatic API

### Data Loading

```python
from src.data_loader import load_sales_data, split_train_test

# Load data
df = load_sales_data("data/raw/sales_data.parquet")

# Split data
train_df, val_df, test_df = split_train_test(
    df, 
    test_weeks=4, 
    validation_weeks=8
)
```

### Feature Engineering

```python
from src.feature_engineering import FeatureEngineer

# Initialize engineer
engineer = FeatureEngineer(df, date_column="date")

# Create all features
featured_df = engineer.create_all_features(
    target_column="sales",
    group_columns=["store_id"],
    include_temporal=True,
    include_holidays=True,
    include_lags=True,
    include_rolling=True
)
```

### Model Training

#### Baseline Models

```python
from src.baselines import BaselineForecaster

# Naive forecast
model = BaselineForecaster(method="naive")
model.fit(train_series)
forecast = model.predict(horizon=42)

# Seasonal naive
model = BaselineForecaster(method="seasonal_naive", season_length=7)
forecast = model.fit_predict(train_series, horizon=42)
```

#### Statistical Models

```python
from src.arima_models import ARIMAForecaster
from src.prophet_model import ProphetForecaster

# ARIMA
arima = ARIMAForecaster(seasonal=True, m=7)
arima.fit(train_series)
forecast = arima.predict(steps=42)

# Prophet
prophet = ProphetForecaster(seasonality_mode="multiplicative")
prophet.fit(train_df, date_col="date", target_col="sales")
forecast_df = prophet.predict(steps=42)
```

#### ML Models

```python
from src.ml_models import MLForecaster

# XGBoost
xgb = MLForecaster(model_type="xgboost", max_depth=6, learning_rate=0.1)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
predictions = xgb.predict(X_test)

# Feature importance
importance = xgb.get_feature_importance(top_n=20)
```

### Evaluation

```python
from src.evaluation import ModelEvaluator, compute_metrics

# Compute metrics
metrics = compute_metrics(y_true, y_pred, y_train)
# Returns: {'mae': 25.3, 'rmse': 35.2, 'mape': 7.5, ...}

# Compare models
evaluator = ModelEvaluator(metrics=['mae', 'rmse', 'mape'])
evaluator.evaluate_model("XGBoost", y_true, y_pred_xgb)
evaluator.evaluate_model("Prophet", y_true, y_pred_prophet)

results_df = evaluator.get_results()
best_model = evaluator.get_best_model(metric='mae')
```

### MLflow Integration

```python
from src.mlflow_utils import MLflowTracker, log_forecast_experiment

# Initialize tracker
tracker = MLflowTracker(experiment_name="retail_forecasting")

# Log experiment
with tracker.start_run(run_name="xgboost_v1"):
    tracker.log_params(model_params)
    tracker.log_metrics(metrics)
    tracker.log_model(model)
```

---

## Future REST API (FastAPI)

### Planned Endpoints

#### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-08T00:00:00Z"
}
```

#### Generate Forecast

```http
POST /forecast
```

**Request Body**:
```json
{
  "store_id": 1,
  "product_id": 5,
  "horizon": 42,
  "model": "xgboost",
  "include_confidence_intervals": true
}
```

**Response**:
```json
{
  "store_id": 1,
  "product_id": 5,
  "forecast": [120, 125, 118, ...],
  "dates": ["2024-01-09", "2024-01-10", ...],
  "confidence_intervals": {
    "lower": [110, 115, 108, ...],
    "upper": [130, 135, 128, ...]
  },
  "model_used": "xgboost",
  "generated_at": "2024-01-08T00:00:00Z"
}
```

#### Batch Forecast

```http
POST /forecast/batch
```

**Request Body**:
```json
{
  "requests": [
    {"store_id": 1, "product_id": 5, "horizon": 42},
    {"store_id": 2, "product_id": 10, "horizon": 42}
  ],
  "model": "ensemble"
}
```

#### Model Information

```http
GET /models
```

**Response**:
```json
{
  "available_models": [
    "naive",
    "seasonal_naive",
    "arima",
    "prophet",
    "xgboost",
    "lightgbm",
    "ensemble"
  ],
  "default_model": "xgboost"
}
```

#### Model Performance

```http
GET /models/{model_name}/performance
```

**Response**:
```json
{
  "model": "xgboost",
  "metrics": {
    "mae": 25.3,
    "rmse": 35.2,
    "mape": 7.5
  },
  "last_trained": "2024-01-07T12:00:00Z",
  "training_samples": 450000
}
```

#### Retrain Model

```http
POST /models/{model_name}/retrain
```

**Request Body**:
```json
{
  "data_path": "data/raw/sales_data.parquet",
  "hyperparameters": {
    "max_depth": 6,
    "learning_rate": 0.1
  }
}
```

**Response**:
```json
{
  "status": "training_started",
  "job_id": "train_xgb_20240108_001",
  "estimated_duration": "15 minutes"
}
```

---

## Command Line Interface

### Data Generation

```bash
python scripts/generate_data.py \
  --num-stores 10 \
  --num-products 50 \
  --start-date 2021-01-01 \
  --end-date 2023-12-31 \
  --seed 42
```

### Training Pipeline

```bash
# Train all models
python scripts/run_pipeline.py

# Train specific models
python scripts/run_pipeline.py --skip-baselines
python scripts/run_pipeline.py --skip-ml

# With MLflow tracking
python scripts/run_pipeline.py --use-mlflow
```

### Using Makefile

```bash
# Generate data
make data

# Train models
make train

# Run tests
make test

# Start MLflow UI
make mlflow
```

---

## Python Package API

After installation (`pip install -e .`):

```python
import retail_forecasting as rf

# Load data
df = rf.load_sales_data("data/raw/sales_data.parquet")

# Create forecaster
forecaster = rf.BaselineForecaster(method="seasonal_naive")

# Evaluate
evaluator = rf.ModelEvaluator()
metrics = rf.compute_metrics(y_true, y_pred)
```

---

## Error Responses

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Model or data not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model training in progress |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "Horizon must be between 1 and 365",
    "details": {
      "parameter": "horizon",
      "provided": 500,
      "allowed_range": [1, 365]
    }
  }
}
```

---

## Rate Limiting (Future)

- **Free tier**: 100 requests/hour
- **Standard tier**: 1000 requests/hour
- **Enterprise**: Unlimited

---

## Authentication (Future)

```http
Authorization: Bearer <api_key>
```

---

## Versioning

API version is included in the URL:

```
/api/v1/forecast
```

---

## WebSocket Support (Future)

For real-time forecast updates:

```javascript
ws://localhost:8000/ws/forecast/{job_id}
```

---

## SDK Examples (Future)

### Python SDK

```python
from retail_forecasting_client import ForecastClient

client = ForecastClient(api_key="your_key")
forecast = client.forecast(store_id=1, product_id=5, horizon=42)
```

### JavaScript SDK

```javascript
const client = new ForecastClient({ apiKey: 'your_key' });
const forecast = await client.forecast({
  storeId: 1,
  productId: 5,
  horizon: 42
});
```

---

## Implementation Roadmap

- ✅ Core forecasting models
- ✅ Programmatic API
- ✅ CLI interface
- ⏳ REST API with FastAPI
- ⏳ Authentication & rate limiting
- ⏳ WebSocket support
- ⏳ Client SDKs
- ⏳ API documentation UI (Swagger/ReDoc)
