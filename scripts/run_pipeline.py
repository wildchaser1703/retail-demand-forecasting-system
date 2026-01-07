"""
Main training pipeline for retail demand forecasting.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.data_loader import load_sales_data, load_metadata, split_train_test
from src.data_validation import DataValidator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.baselines import BaselineForecaster
from src.var_model import VARForecaster
from src.arima_models import ARIMAForecaster
from src.prophet_model import ProphetForecaster
from src.ml_models import MLForecaster
from src.ensemble import SimpleEnsemble
from src.evaluation import ModelEvaluator, compute_metrics
from src.mlflow_utils import MLflowTracker, log_forecast_experiment
from src.visualization import plot_forecast, plot_model_comparison

import warnings
warnings.filterwarnings("ignore")


def load_and_validate_data():
    """Load and validate sales data."""
    print("=" * 60)
    print("Loading and Validating Data")
    print("=" * 60)

    # Load sales data
    data_path = config.paths.raw_data_dir / "sales_data.parquet"
    if not data_path.exists():
        data_path = config.paths.raw_data_dir / "sales_data.csv"

    print(f"Loading data from: {data_path}")
    df = load_sales_data(data_path)

    # Validate data
    print("\nValidating data...")
    validator = DataValidator(df)
    result = validator.validate_all()

    if not result.is_valid:
        print("\nValidation Errors:")
        for error in result.errors:
            print(f"  - {error}")
        raise ValueError("Data validation failed!")

    if result.warnings:
        print("\nValidation Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print("\nData Statistics:")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")

    return df


def prepare_data(df: pd.DataFrame):
    """Prepare data for modeling."""
    print("\n" + "=" * 60)
    print("Preparing Data")
    print("=" * 60)

    # Aggregate to store-level daily sales
    print("Aggregating to store-level daily sales...")
    df_agg = df.groupby(["store_id", "date"])["sales"].sum().reset_index()

    # Add promotion flag (aggregate)
    df_promo = df.groupby(["store_id", "date"])["is_promo"].max().reset_index()
    df_agg = df_agg.merge(df_promo, on=["store_id", "date"], how="left")

    # Split data
    print("Splitting into train/validation/test sets...")
    train_df, val_df, test_df = split_train_test(
        df_agg,
        test_weeks=config.data.test_weeks,
        validation_weeks=config.data.validation_weeks,
    )

    print(f"Train set: {len(train_df)} records ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Validation set: {len(val_df)} records ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test set: {len(test_df)} records ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, val_df, test_df


def engineer_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Engineer features for all datasets."""
    print("\n" + "=" * 60)
    print("Engineering Features")
    print("=" * 60)

    # Combine for feature engineering
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Engineer features
    print("Creating temporal, lag, and rolling features...")
    engineer = FeatureEngineer(combined_df, date_column="date")
    featured_df = engineer.create_all_features(
        target_column="sales",
        group_columns=["store_id"],
    )

    # Split back
    train_featured = featured_df[featured_df["date"] < val_df["date"].min()].copy()
    val_featured = featured_df[
        (featured_df["date"] >= val_df["date"].min()) & (featured_df["date"] < test_df["date"].min())
    ].copy()
    test_featured = featured_df[featured_df["date"] >= test_df["date"].min()].copy()

    # Remove rows with NaN (from lag/rolling features)
    train_featured = train_featured.dropna()
    val_featured = val_featured.dropna()
    test_featured = test_featured.dropna()

    print(f"Features created: {len(featured_df.columns)} columns")
    print(f"Train set after feature engineering: {len(train_featured)} records")
    print(f"Validation set after feature engineering: {len(val_featured)} records")
    print(f"Test set after feature engineering: {len(test_featured)} records")

    return train_featured, val_featured, test_featured


def train_baseline_models(train_df: pd.DataFrame, test_df: pd.DataFrame, evaluator: ModelEvaluator):
    """Train baseline models."""
    print("\n" + "=" * 60)
    print("Training Baseline Models")
    print("=" * 60)

    # Aggregate to total sales per day
    train_series = train_df.groupby("date")["sales"].sum().values
    test_series = test_df.groupby("date")["sales"].sum().values

    forecast_horizon = len(test_series)

    # Naive forecast
    print("\n1. Naive Forecast")
    naive_model = BaselineForecaster(method="naive")
    naive_forecast = naive_model.fit_predict(train_series, forecast_horizon)
    evaluator.evaluate_model("Naive", test_series, naive_forecast, train_series)

    # Seasonal Naive
    print("2. Seasonal Naive Forecast")
    seasonal_naive_model = BaselineForecaster(method="seasonal_naive", season_length=7)
    seasonal_naive_forecast = seasonal_naive_model.fit_predict(train_series, forecast_horizon)
    evaluator.evaluate_model("Seasonal Naive", test_series, seasonal_naive_forecast, train_series)

    # Moving Average
    print("3. Moving Average Forecast")
    ma_model = BaselineForecaster(method="moving_average", window=7)
    ma_forecast = ma_model.fit_predict(train_series, forecast_horizon)
    evaluator.evaluate_model("Moving Average", test_series, ma_forecast, train_series)

    return {
        "Naive": naive_forecast,
        "Seasonal Naive": seasonal_naive_forecast,
        "Moving Average": ma_forecast,
    }


def train_ml_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, evaluator: ModelEvaluator):
    """Train ML models."""
    print("\n" + "=" * 60)
    print("Training Machine Learning Models")
    print("=" * 60)

    # Select features
    feature_cols = [col for col in train_df.columns if col not in ["date", "store_id", "sales"]]
    target_col = "sales"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    forecasts = {}

    # XGBoost
    print("\n1. XGBoost")
    xgb_model = MLForecaster(model_type="xgboost", **config.model.xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    xgb_forecast = xgb_model.predict(X_test)
    evaluator.evaluate_model("XGBoost", y_test.values, xgb_forecast, y_train.values)
    forecasts["XGBoost"] = xgb_forecast

    # LightGBM
    print("2. LightGBM")
    lgb_model = MLForecaster(model_type="lightgbm", **config.model.lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    lgb_forecast = lgb_model.predict(X_test)
    evaluator.evaluate_model("LightGBM", y_test.values, lgb_forecast, y_train.values)
    forecasts["LightGBM"] = lgb_forecast

    # Random Forest
    print("3. Random Forest")
    rf_model = MLForecaster(model_type="random_forest", n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_forecast = rf_model.predict(X_test)
    evaluator.evaluate_model("Random Forest", y_test.values, rf_forecast, y_train.values)
    forecasts["Random Forest"] = rf_forecast

    return forecasts, {"XGBoost": xgb_model, "LightGBM": lgb_model, "Random Forest": rf_model}


def train_ensemble(forecasts: dict, test_series: np.ndarray, evaluator: ModelEvaluator):
    """Train ensemble models."""
    print("\n" + "=" * 60)
    print("Training Ensemble Models")
    print("=" * 60)

    # Simple average ensemble
    print("\n1. Simple Average Ensemble")
    simple_ensemble = SimpleEnsemble(method="mean")
    ensemble_forecast = simple_ensemble.combine(forecasts)
    evaluator.evaluate_model("Ensemble (Mean)", test_series, ensemble_forecast)

    return ensemble_forecast


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train retail demand forecasting models")
    parser.add_argument("--data-path", type=str, help="Path to sales data")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline models")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML models")
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow tracking")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RETAIL DEMAND FORECASTING - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize MLflow if requested
    tracker = None
    if args.use_mlflow:
        print("\nInitializing MLflow...")
        tracker = MLflowTracker(
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
        )

    # Load and validate data
    df = load_and_validate_data()

    # Prepare data
    train_df, val_df, test_df = prepare_data(df)

    # Engineer features
    train_featured, val_featured, test_featured = engineer_features(train_df, val_df, test_df)

    # Initialize evaluator
    evaluator = ModelEvaluator(metrics=config.evaluation.metrics)

    # Train models
    all_forecasts = {}

    if not args.skip_baselines:
        baseline_forecasts = train_baseline_models(train_df, test_df, evaluator)
        all_forecasts.update(baseline_forecasts)

    if not args.skip_ml:
        ml_forecasts, ml_models = train_ml_models(train_featured, val_featured, test_featured, evaluator)
        all_forecasts.update(ml_forecasts)

    # Train ensemble
    if len(all_forecasts) > 1:
        test_series = test_df.groupby("date")["sales"].sum().values
        ensemble_forecast = train_ensemble(all_forecasts, test_series, evaluator)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    results_df = evaluator.get_results()
    print("\n", results_df.to_string())

    # Save results
    results_path = config.paths.models_dir / "evaluation_results.csv"
    results_df.to_csv(results_path)
    print(f"\nResults saved to: {results_path}")

    # Best model
    best_model = evaluator.get_best_model(metric="mae")
    print(f"\nBest model (by MAE): {best_model}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
