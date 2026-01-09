"""
Unit tests for evaluation metrics.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import mae, rmse, mape, smape, mase, compute_metrics  # noqa: E402


def test_mae():
    """Test Mean Absolute Error."""
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])

    result = mae(y_true, y_pred)
    expected = np.mean(np.abs(y_true - y_pred))

    assert np.isclose(result, expected)
    assert result == 10.0


def test_rmse():
    """Test Root Mean Squared Error."""
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])

    result = rmse(y_true, y_pred)
    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))

    assert np.isclose(result, expected)


def test_mape():
    """Test Mean Absolute Percentage Error."""
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])

    result = mape(y_true, y_pred)

    # MAPE should be a percentage
    assert 0 <= result <= 100


def test_mape_zero_values():
    """Test MAPE with zero values."""
    y_true = np.array([0, 150, 200, 250])
    y_pred = np.array([10, 140, 210, 240])

    # Should handle zeros gracefully
    result = mape(y_true, y_pred)
    assert not np.isnan(result)


def test_smape():
    """Test Symmetric Mean Absolute Percentage Error."""
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])

    result = smape(y_true, y_pred)

    # SMAPE should be between 0 and 100
    assert 0 <= result <= 100


def test_mase():
    """Test Mean Absolute Scaled Error."""
    y_train = np.array([100, 110, 105, 115, 120, 125, 130])
    y_true = np.array([135, 140, 145])
    y_pred = np.array([130, 145, 150])

    result = mase(y_true, y_pred, y_train, seasonal_period=1)

    # MASE should be positive
    assert result > 0


def test_compute_metrics():
    """Test computing multiple metrics at once."""
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])
    y_train = np.array([90, 95, 100, 105, 110])

    metrics = compute_metrics(y_true, y_pred, y_train)

    # Check that all metrics are computed
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "smape" in metrics
    assert "r2" in metrics
    assert "mase" in metrics

    # Check that values are reasonable
    for metric_name, value in metrics.items():
        assert not np.isnan(value), f"{metric_name} is NaN"
        assert np.isfinite(value), f"{metric_name} is not finite"


def test_compute_metrics_subset():
    """Test computing a subset of metrics."""
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 210, 240])

    metrics = compute_metrics(y_true, y_pred, metrics=["mae", "rmse"])

    assert len(metrics) == 2
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" not in metrics
