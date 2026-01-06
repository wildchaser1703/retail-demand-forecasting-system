import pandas as pd
from typing import Sequence


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
        raise ValueError("Series length must be >= saeson_length.")

    forecasts = []
    for i in range(horizon):
        forecasts.append(series[-season_length + (i % season_length)])
    return forecasts
