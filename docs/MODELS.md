# Model Documentation

## Overview

This document provides detailed information about each forecasting model implemented in the system, including their assumptions, hyperparameters, use cases, and limitations.

---

## Baseline Models

### 1. Naive Forecast

**Description**: The simplest forecasting method that repeats the last observed value for all future periods.

**Formula**: `ŷ(t+h) = y(t)` for all h > 0

**Hyperparameters**: None

**Use Cases**:
- Quick baseline for comparison
- When data has no clear pattern
- Random walk processes

**Limitations**:
- Ignores all historical patterns
- No seasonality or trend capture
- Poor for structured data

**Implementation**: `src/baselines.py::naive_forecast()`

---

### 2. Seasonal Naive Forecast

**Description**: Repeats values from the previous seasonal cycle.

**Formula**: `ŷ(t+h) = y(t+h-m)` where m is the seasonal period

**Hyperparameters**:
- `season_length`: Length of seasonal cycle (default: 7 for weekly)

**Use Cases**:
- Strong seasonal patterns
- Stable seasonality
- Benchmark for seasonal models

**Limitations**:
- Assumes constant seasonality
- No trend adaptation
- Requires at least one full season of data

**Implementation**: `src/baselines.py::seasonal_naive_forecast()`

---

### 3. Moving Average

**Description**: Forecasts using the average of recent observations.

**Formula**: `ŷ(t+h) = (1/k) * Σ y(t-i)` for i=0 to k-1

**Hyperparameters**:
- `window`: Number of recent observations to average (default: 7)

**Use Cases**:
- Smoothing noisy data
- Short-term forecasts
- Stationary series

**Limitations**:
- Lags behind trends
- Equal weight to all observations in window
- Constant forecast for all horizons

**Implementation**: `src/baselines.py::moving_average_forecast()`

---

### 4. Exponential Smoothing

**Description**: Weighted average with exponentially decreasing weights for older observations.

**Formula**: `ŷ(t+h) = α*y(t) + (1-α)*ŷ(t)`

**Hyperparameters**:
- `alpha`: Smoothing parameter (0 < α < 1, default: 0.3)
  - Higher α: More weight on recent observations
  - Lower α: More smoothing

**Use Cases**:
- Noisy data requiring smoothing
- When recent observations are more relevant
- Level-only forecasts

**Limitations**:
- No trend or seasonality
- Single smoothing parameter
- Constant forecast

**Implementation**: `src/baselines.py::exponential_smoothing_forecast()`

---

## Statistical Models

### 5. VAR (Vector AutoRegression)

**Description**: Multivariate time series model capturing dependencies between multiple series.

**Model**: Each variable is regressed on its own lags and lags of all other variables.

**Hyperparameters**:
- `max_lags`: Maximum lag order to consider (default: 14)
- `ic`: Information criterion for lag selection ('aic' or 'bic')
- `trend`: Trend component ('c', 'ct', 'n')

**Use Cases**:
- Multiple related time series
- Cross-series dependencies
- Impulse response analysis

**Assumptions**:
- Stationarity (or differencing applied)
- Linear relationships
- Constant variance

**Limitations**:
- Requires stationarity
- Many parameters with multiple series
- Computational cost increases with series count

**Implementation**: `src/var_model.py::VARForecaster`

**Key Methods**:
- `check_stationarity()`: ADF test
- `select_lag_order()`: Automatic lag selection
- `granger_causality()`: Test for causal relationships

---

### 6. ARIMA (AutoRegressive Integrated Moving Average)

**Description**: Univariate model combining autoregression, differencing, and moving average.

**Model Components**:
- **AR(p)**: Autoregressive terms
- **I(d)**: Differencing for stationarity
- **MA(q)**: Moving average terms

**Hyperparameters**:
- `p`: AR order (default: auto-selected, max 5)
- `d`: Differencing order (default: auto-selected, max 2)
- `q`: MA order (default: auto-selected, max 5)

**Use Cases**:
- Univariate time series
- Non-seasonal or seasonal patterns
- Medium-term forecasts

**Assumptions**:
- Stationarity (after differencing)
- Linear relationships
- Constant variance

**Limitations**:
- Univariate only
- Requires parameter tuning
- Struggles with multiple seasonalities

**Implementation**: `src/arima_models.py::ARIMAForecaster`

**Auto ARIMA**: Automatically selects (p,d,q) using AIC/BIC

---

### 7. SARIMA (Seasonal ARIMA)

**Description**: ARIMA extended with seasonal components.

**Model**: ARIMA(p,d,q)(P,D,Q,s)

**Hyperparameters**:
- Non-seasonal: p, d, q
- Seasonal: P, D, Q, s (seasonal period)

**Use Cases**:
- Strong seasonal patterns
- Seasonal + trend
- Retail, tourism, energy data

**Implementation**: `src/arima_models.py::SARIMAForecaster`

---

### 8. Prophet

**Description**: Facebook's forecasting tool designed for business time series.

**Model Components**:
- Trend: Piecewise linear or logistic
- Seasonality: Fourier series
- Holidays: User-specified events
- Regressors: Additional features

**Hyperparameters**:
- `growth`: 'linear' or 'logistic'
- `seasonality_mode`: 'additive' or 'multiplicative' (default: 'multiplicative')
- `changepoint_prior_scale`: Trend flexibility (default: 0.05)
- `seasonality_prior_scale`: Seasonality strength (default: 10.0)

**Use Cases**:
- Business forecasting
- Strong seasonality with holidays
- Missing data
- Outliers

**Strengths**:
- Handles missing data
- Robust to outliers
- Interpretable components
- Easy to add domain knowledge

**Limitations**:
- Can overfit with default settings
- Less effective for short series
- Assumes additive/multiplicative decomposition

**Implementation**: `src/prophet_model.py::ProphetForecaster`

---

## Machine Learning Models

### 9. XGBoost

**Description**: Gradient boosting with time series features.

**Hyperparameters**:
- `max_depth`: Tree depth (default: 6)
- `learning_rate`: Step size (default: 0.1)
- `n_estimators`: Number of trees (default: 100)
- `subsample`: Row sampling (default: 0.8)
- `colsample_bytree`: Column sampling (default: 0.8)

**Use Cases**:
- Complex non-linear patterns
- Many features
- Feature interactions

**Strengths**:
- Handles non-linearity
- Feature importance
- Robust to outliers
- Fast training

**Limitations**:
- Requires feature engineering
- Can overfit
- Less interpretable
- No native uncertainty

**Implementation**: `src/ml_models.py::MLForecaster(model_type='xgboost')`

---

### 10. LightGBM

**Description**: Microsoft's gradient boosting framework, optimized for speed.

**Hyperparameters**: Similar to XGBoost

**Advantages over XGBoost**:
- Faster training
- Lower memory usage
- Better with categorical features

**Implementation**: `src/ml_models.py::MLForecaster(model_type='lightgbm')`

---

### 11. Random Forest

**Description**: Ensemble of decision trees with bagging.

**Hyperparameters**:
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Tree depth (default: 10)
- `min_samples_split`: Minimum samples to split (default: 5)

**Use Cases**:
- Baseline ML model
- Feature selection
- Non-linear patterns

**Strengths**:
- Less prone to overfitting than single trees
- Feature importance
- Handles non-linearity

**Limitations**:
- Slower than gradient boosting
- Larger model size
- Less accurate than boosting

**Implementation**: `src/ml_models.py::MLForecaster(model_type='random_forest')`

---

## Ensemble Methods

### 12. Simple Ensemble

**Description**: Combines forecasts using averaging.

**Methods**:
- **Mean**: Simple average
- **Median**: Robust to outliers
- **Weighted**: Performance-based weights

**Use Cases**:
- Reduce variance
- Combine diverse models
- Improve robustness

**Implementation**: `src/ensemble.py::SimpleEnsemble`

---

### 13. Stacking Ensemble

**Description**: Uses a meta-learner to combine base model predictions.

**Meta-Learner**: Linear Regression (default)

**Use Cases**:
- Learn optimal combination
- Diverse base models
- Maximize performance

**Implementation**: `src/ensemble.py::StackingEnsemble`

---

### 14. Adaptive Ensemble

**Description**: Dynamically adjusts weights based on recent performance.

**Hyperparameters**:
- `window_size`: Recent performance window (default: 10)
- `decay_factor`: Exponential decay (default: 0.9)

**Use Cases**:
- Non-stationary environments
- Concept drift
- Online learning

**Implementation**: `src/ensemble.py::AdaptiveEnsemble`

---

## Model Selection Guide

| Scenario | Recommended Models |
|----------|-------------------|
| Quick baseline | Naive, Seasonal Naive |
| Strong seasonality | SARIMA, Prophet |
| Multiple series | VAR |
| Complex patterns | XGBoost, LightGBM |
| Interpretability needed | Prophet, ARIMA |
| Uncertainty quantification | ARIMA, Prophet |
| Production deployment | LightGBM, Ensemble |
| Limited data | Seasonal Naive, Prophet |
| Many features | XGBoost, LightGBM |

---

## Performance Comparison

Based on typical retail data:

| Model | Speed | Accuracy | Interpretability | Scalability |
|-------|-------|----------|------------------|-------------|
| Naive | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Seasonal Naive | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ARIMA | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Prophet | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| XGBoost | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| LightGBM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ensemble | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
