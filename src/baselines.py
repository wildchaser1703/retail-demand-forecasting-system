"""
Baseline forecasting models for retail demand forecasting.
"""
import pandas as pd
import numpy as np
from typing import Sequence, Optional


def naive_forecast(series: Sequence[float], horizon: int) -> list[float]:
    """
    Naive forecast: repeats the last observed value.

    Parameters
    ----------
    series : Sequence[float]
        Historical time series values.
    horizon : int
        Number of future time steps to predict.

    Returns
    -------
    list[float]
        Forecasted values.
    """
    if len(series) == 0:
        raise ValueError("Input series is empty.")

    last_value = series[-1]
    return [last_value] * horizon


def seasonal_naive_forecast(
    series: Sequence[float], horizon: int, season_length: int = 7
) -> list[float]:
    """
    Seasonal naive forecast: repeats values from the previous season.

    Parameters
    ----------
    series : Sequence[float]
        Historical time series values.
    horizon : int
        Number of future time steps to predict.
    season_length : int
        Length of the seasonal cycle (default: 7 for weekly seasonality).

    Returns
    -------
    list[float]
        Forecasted values.
    """
    if len(series) < season_length:
        raise ValueError(f"Series length must be >= season_length ({season_length}).")

    forecasts = []
    for i in range(horizon):
        forecasts.append(series[-season_length + (i % season_length)])
    return forecasts


def moving_average_forecast(
    series: Sequence[float], horizon: int, window: int = 7
) -> list[float]:
    """
    Moving average forecast: uses average of last N values.

    Parameters
    ----------
    series : Sequence[float]
        Historical time series values.
    horizon : int
        Number of future time steps to predict.
    window : int, default=7
        Window size for moving average.

    Returns
    -------
    list[float]
        Forecasted values.
    """
    if len(series) < window:
        raise ValueError(f"Series length must be >= window size ({window}).")

    # Calculate moving average of last window values
    ma_value = np.mean(series[-window:])
    return [ma_value] * horizon


def exponential_smoothing_forecast(
    series: Sequence[float], horizon: int, alpha: float = 0.3
) -> list[float]:
    """
    Simple exponential smoothing forecast.

    Parameters
    ----------
    series : Sequence[float]
        Historical time series values.
    horizon : int
        Number of future time steps to predict.
    alpha : float, default=0.3
        Smoothing parameter (0 < alpha < 1).

    Returns
    -------
    list[float]
        Forecasted values.
    """
    if len(series) == 0:
        raise ValueError("Input series is empty.")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    # Initialize with first value
    smoothed = series[0]

    # Apply exponential smoothing
    for value in series[1:]:
        smoothed = alpha * value + (1 - alpha) * smoothed

    return [smoothed] * horizon


class BaselineForecaster:
    """
    Unified interface for baseline forecasting models.
    """

    def __init__(self, method: str = "seasonal_naive", **kwargs):
        """
        Initialize baseline forecaster.

        Parameters
        ----------
        method : str, default='seasonal_naive'
            Forecasting method: 'naive', 'seasonal_naive', 'moving_average', or 'exponential_smoothing'.
        **kwargs : dict
            Additional parameters for the chosen method.
        """
        self.method = method
        self.params = kwargs
        self.history = None

    def fit(self, series: Sequence[float]) -> "BaselineForecaster":
        """
        Fit the model (store historical data).

        Parameters
        ----------
        series : Sequence[float]
            Historical time series values.

        Returns
        -------
        BaselineForecaster
            Fitted forecaster.
        """
        self.history = list(series)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts.

        Parameters
        ----------
        horizon : int
            Number of future time steps to predict.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if self.history is None:
            raise ValueError("Model must be fitted before prediction.")

        if self.method == "naive":
            forecasts = naive_forecast(self.history, horizon)
        elif self.method == "seasonal_naive":
            season_length = self.params.get("season_length", 7)
            forecasts = seasonal_naive_forecast(self.history, horizon, season_length)
        elif self.method == "moving_average":
            window = self.params.get("window", 7)
            forecasts = moving_average_forecast(self.history, horizon, window)
        elif self.method == "exponential_smoothing":
            alpha = self.params.get("alpha", 0.3)
            forecasts = exponential_smoothing_forecast(self.history, horizon, alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return np.array(forecasts)

    def fit_predict(self, series: Sequence[float], horizon: int) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters
        ----------
        series : Sequence[float]
            Historical time series values.
        horizon : int
            Number of future time steps to predict.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        self.fit(series)
        return self.predict(horizon)

