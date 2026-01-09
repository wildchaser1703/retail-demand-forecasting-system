"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_time_series():
    """Create a sample time series."""
    np.random.seed(42)
    n = 100
    trend = np.linspace(100, 150, n)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 5, n)
    return trend + seasonality + noise


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "store_id": np.random.choice([1, 2, 3], 100),
        "product_id": np.random.choice([1, 2, 3, 4, 5], 100),
        "sales": np.random.randint(50, 200, 100),
        "is_promo": np.random.choice([0, 1], 100, p=[0.8, 0.2]),
    }
    return pd.DataFrame(data)
