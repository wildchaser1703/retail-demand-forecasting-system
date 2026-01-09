"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer  # noqa: E402


@pytest.fixture
def sample_data():
    """Create sample time series data."""
    dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
    data = {
        "date": dates,
        "store_id": [1] * len(dates),
        "product_id": [1] * len(dates),
        "sales": np.random.randint(50, 150, len(dates)),
        "is_promo": np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
    }
    return pd.DataFrame(data)


def test_temporal_features(sample_data):
    """Test temporal feature creation."""
    engineer = FeatureEngineer(sample_data, date_column="date")
    result = engineer.add_temporal_features()

    # Check that temporal features were added
    assert "year" in result.columns
    assert "month" in result.columns
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns

    # Check values
    assert result["year"].iloc[0] == 2023
    assert result["month"].min() >= 1
    assert result["month"].max() <= 12
    assert result["day_of_week"].min() >= 0
    assert result["day_of_week"].max() <= 6


def test_lag_features(sample_data):
    """Test lag feature creation."""
    engineer = FeatureEngineer(sample_data, date_column="date")
    result = engineer.add_lag_features(target_column="sales", lags=[1, 7])

    # Check that lag features were added
    assert "sales_lag_1" in result.columns
    assert "sales_lag_7" in result.columns

    # Check that lag values are correct
    assert pd.isna(result["sales_lag_1"].iloc[0])  # First value should be NaN
    assert result["sales_lag_1"].iloc[1] == result["sales"].iloc[0]


def test_rolling_features(sample_data):
    """Test rolling window feature creation."""
    engineer = FeatureEngineer(sample_data, date_column="date")
    result = engineer.add_rolling_features(
        target_column="sales", windows=[7], stats=["mean", "std"]
    )

    # Check that rolling features were added
    assert "sales_rolling_7_mean" in result.columns
    assert "sales_rolling_7_std" in result.columns

    # Check that rolling mean is calculated correctly
    rolling_mean_7 = result["sales"].rolling(window=7, min_periods=1).mean()
    np.testing.assert_array_almost_equal(
        result["sales_rolling_7_mean"].values, rolling_mean_7.values
    )


def test_holiday_features(sample_data):
    """Test holiday feature creation."""
    engineer = FeatureEngineer(sample_data, date_column="date")
    result = engineer.add_holiday_features()

    # Check that holiday features were added
    assert "is_holiday" in result.columns
    assert "days_to_holiday" in result.columns
    assert "days_from_holiday" in result.columns

    # Check that values are reasonable
    assert result["is_holiday"].isin([0, 1]).all()
    assert (result["days_to_holiday"] >= 0).all()


def test_create_all_features(sample_data):
    """Test creating all features at once."""
    engineer = FeatureEngineer(sample_data, date_column="date")
    result = engineer.create_all_features(
        target_column="sales",
        include_temporal=True,
        include_holidays=True,
        include_lags=True,
        include_rolling=True,
    )

    # Check that multiple feature types were added
    assert "year" in result.columns  # Temporal
    assert "is_holiday" in result.columns  # Holiday
    assert "sales_lag_1" in result.columns  # Lag
    assert "sales_rolling_7_mean" in result.columns  # Rolling

    # Check that original columns are preserved
    assert "date" in result.columns
    assert "sales" in result.columns


def test_grouped_lag_features(sample_data):
    """Test lag features with grouping."""
    # Add another store
    sample_data_2 = sample_data.copy()
    sample_data_2["store_id"] = 2
    combined = pd.concat([sample_data, sample_data_2], ignore_index=True)

    engineer = FeatureEngineer(combined, date_column="date")
    result = engineer.add_lag_features(target_column="sales", lags=[1], group_columns=["store_id"])

    # Check that lags are computed within groups
    store_1_data = result[result["store_id"] == 1]
    assert store_1_data["sales_lag_1"].iloc[1] == store_1_data["sales"].iloc[0]

    store_2_data = result[result["store_id"] == 2]
    assert store_2_data["sales_lag_1"].iloc[1] == store_2_data["sales"].iloc[0]
