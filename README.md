# Retail Demand Forecasting System

## Problem Statement
Retail teams need reliable short-term sales forecasts to plan inventory and promotions.
This project builds a 6-week demand forecasting pipeline for multiple retail stores,
focusing on robustness and interpretability rather than peak accuracy.

## Business Constraints
- Strong weekly seasonality
- Promotions causing non-stationary sales patterns
- High variance across stores
- Forecast stability is more important than marginal accuracy gains

## Data Overview
The dataset contains historical daily sales at the store level along with time-based
and promotion-related signals. Only a small sample is included in this repository for
reproducibility.

## Modeling Approach
The system starts with simple baselines (naive and seasonal naive) to establish a lower
bound. More structured models are then introduced to capture temporal dependencies and
cross-store interactions.

Vector AutoRegression (VAR) is used to model multivariate time series where appropriate,
with careful consideration of its assumptions and limitations.

## Evaluation Strategy
Models are evaluated using rolling forecasts to better reflect real-world deployment
scenarios. Mean Absolute Error (MAE) is used as the primary metric due to its robustness
to outliers.

## Results
The VAR-based approach demonstrated improved stability during promotion-heavy periods
compared to naive baselines, particularly for medium-volume stores.

## Limitations & Next Steps
- VAR assumes linear relationships and stationarity
- Scalability is limited for large numbers of stores

Future work could explore hierarchical or Bayesian time series models to better share
information across stores.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```