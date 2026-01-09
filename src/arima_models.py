"""
ARIMA and SARIMA models for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """
    Auto ARIMA forecaster with automatic parameter selection.
    """

    def __init__(
        self,
        seasonal: bool = True,
        m: int = 7,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        information_criterion: str = "aic",
        trace: bool = False,
    ):
        """
        Initialize ARIMA forecaster.

        Parameters
        ----------
        seasonal : bool, default=True
            Whether to fit seasonal ARIMA.
        m : int, default=7
            Seasonal period (7 for weekly data).
        max_p : int, default=5
            Maximum AR order.
        max_d : int, default=2
            Maximum differencing order.
        max_q : int, default=5
            Maximum MA order.
        max_P : int, default=2
            Maximum seasonal AR order.
        max_D : int, default=1
            Maximum seasonal differencing order.
        max_Q : int, default=2
            Maximum seasonal MA order.
        information_criterion : str, default='aic'
            Information criterion for model selection.
        trace : bool, default=False
            Whether to print search progress.
        """
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.information_criterion = information_criterion
        self.trace = trace
        self.model = None
        self.order = None
        self.seasonal_order = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Fit ARIMA model with automatic parameter selection.

        Parameters
        ----------
        series : pd.Series
            Time series to fit.

        Returns
        -------
        ARIMAForecaster
            Fitted forecaster.
        """
        self.model = auto_arima(
            series,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            max_P=self.max_P if self.seasonal else 0,
            max_D=self.max_D if self.seasonal else 0,
            max_Q=self.max_Q if self.seasonal else 0,
            information_criterion=self.information_criterion,
            trace=self.trace,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order if self.seasonal else None

        return self

    def predict(self, steps: int, return_conf_int: bool = False, alpha: float = 0.05) -> np.ndarray:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        return_conf_int : bool, default=False
            Whether to return confidence intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted values, optionally with confidence intervals.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        if return_conf_int:
            forecast, conf_int = self.model.predict(
                n_periods=steps, return_conf_int=True, alpha=alpha
            )
            return forecast, conf_int
        else:
            forecast = self.model.predict(n_periods=steps)
            return forecast

    def get_params(self) -> dict:
        """Get model parameters."""
        if self.model is None:
            raise ValueError("Model must be fitted first.")

        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": self.model.aic(),
            "bic": self.model.bic(),
        }


class SARIMAForecaster:
    """
    Seasonal ARIMA forecaster with manual parameter specification.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        trend: Optional[str] = "c",
    ):
        """
        Initialize SARIMA forecaster.

        Parameters
        ----------
        order : Tuple[int, int, int], default=(1, 1, 1)
            (p, d, q) order of the model.
        seasonal_order : Tuple[int, int, int, int], default=(1, 1, 1, 7)
            (P, D, Q, s) seasonal order of the model.
        trend : str, optional, default='c'
            Trend parameter: 'n', 'c', 't', 'ct'.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.model_fitted = None

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        """
        Fit SARIMA model.

        Parameters
        ----------
        series : pd.Series
            Time series to fit.

        Returns
        -------
        SARIMAForecaster
            Fitted forecaster.
        """
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.model_fitted = self.model.fit(disp=False)

        return self

    def predict(self, steps: int, return_conf_int: bool = False, alpha: float = 0.05) -> np.ndarray:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        return_conf_int : bool, default=False
            Whether to return confidence intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted values, optionally with confidence intervals.
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before prediction.")

        forecast_result = self.model_fitted.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean

        if return_conf_int:
            conf_int = forecast_result.conf_int(alpha=alpha)
            return forecast.values, conf_int.values
        else:
            return forecast.values

    def get_summary(self) -> str:
        """Get model summary."""
        if self.model_fitted is None:
            raise ValueError("Model must be fitted first.")

        return str(self.model_fitted.summary())

    def get_diagnostics(self) -> None:
        """Plot model diagnostics."""
        if self.model_fitted is None:
            raise ValueError("Model must be fitted first.")

        self.model_fitted.plot_diagnostics(figsize=(12, 8))
