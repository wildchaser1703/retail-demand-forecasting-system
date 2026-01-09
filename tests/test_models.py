"""
Unit tests for baseline models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.baselines import (  # noqa: E402
    naive_forecast,
    seasonal_naive_forecast,
    moving_average_forecast,
    exponential_smoothing_forecast,
    BaselineForecaster,
)


def test_naive_forecast():
    """Test naive forecast."""
    series = [100, 110, 105, 115, 120]
    horizon = 3

    forecast = naive_forecast(series, horizon)

    assert len(forecast) == horizon
    assert all(f == 120 for f in forecast)  # Should repeat last value


def test_naive_forecast_empty_series():
    """Test naive forecast with empty series."""
    with pytest.raises(ValueError, match="Input series is empty"):
        naive_forecast([], 3)


def test_seasonal_naive_forecast():
    """Test seasonal naive forecast."""
    series = [100, 110, 105, 115, 120, 125, 130]  # 7 days
    horizon = 7
    season_length = 7

    forecast = seasonal_naive_forecast(series, horizon, season_length)

    assert len(forecast) == horizon
    # Should repeat the last season
    assert forecast[0] == series[0]
    assert forecast[6] == series[6]


def test_seasonal_naive_forecast_short_series():
    """Test seasonal naive forecast with series shorter than season."""
    series = [100, 110, 105]
    with pytest.raises(ValueError, match="Series length must be"):
        seasonal_naive_forecast(series, 3, season_length=7)


def test_moving_average_forecast():
    """Test moving average forecast."""
    series = [100, 110, 105, 115, 120, 125, 130]
    horizon = 3
    window = 3

    forecast = moving_average_forecast(series, horizon, window)

    assert len(forecast) == horizon
    expected_ma = np.mean(series[-window:])
    assert all(np.isclose(f, expected_ma) for f in forecast)


def test_exponential_smoothing_forecast():
    """Test exponential smoothing forecast."""
    series = [100, 110, 105, 115, 120]
    horizon = 3
    alpha = 0.3

    forecast = exponential_smoothing_forecast(series, horizon, alpha)

    assert len(forecast) == horizon
    # All forecasts should be the same (simple exponential smoothing)
    assert all(f == forecast[0] for f in forecast)


def test_exponential_smoothing_invalid_alpha():
    """Test exponential smoothing with invalid alpha."""
    series = [100, 110, 105]

    with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
        exponential_smoothing_forecast(series, 3, alpha=1.5)


def test_baseline_forecaster_naive():
    """Test BaselineForecaster with naive method."""
    series = [100, 110, 105, 115, 120]
    horizon = 3

    forecaster = BaselineForecaster(method="naive")
    forecaster.fit(series)
    forecast = forecaster.predict(horizon)

    assert len(forecast) == horizon
    assert isinstance(forecast, np.ndarray)


def test_baseline_forecaster_fit_predict():
    """Test BaselineForecaster fit_predict."""
    series = [100, 110, 105, 115, 120]
    horizon = 3

    forecaster = BaselineForecaster(method="seasonal_naive", season_length=3)
    forecast = forecaster.fit_predict(series, horizon)

    assert len(forecast) == horizon


def test_baseline_forecaster_unknown_method():
    """Test BaselineForecaster with unknown method."""
    forecaster = BaselineForecaster(method="unknown")
    forecaster.fit([100, 110, 105])

    with pytest.raises(ValueError, match="Unknown method"):
        forecaster.predict(3)
