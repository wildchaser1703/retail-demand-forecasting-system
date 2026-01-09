"""
Vector AutoRegression (VAR) model for multivariate time series forecasting.
"""

import pandas as pd
from typing import Optional, Tuple
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


class VARForecaster:
    """
    VAR model wrapper for retail demand forecasting.
    """

    def __init__(
        self,
        max_lags: int = 14,
        ic: str = "aic",
        trend: str = "c",
        verbose: bool = False,
    ):
        """
        Initialize VAR forecaster.

        Parameters
        ----------
        max_lags : int, default=14
            Maximum number of lags to consider.
        ic : str, default='aic'
            Information criterion for lag selection ('aic' or 'bic').
        trend : str, default='c'
            Trend parameter: 'c' (constant), 'ct' (constant + trend), 'n' (none).
        verbose : bool, default=False
            Whether to print diagnostic information.
        """
        self.max_lags = max_lags
        self.ic = ic
        self.trend = trend
        self.verbose = verbose
        self.model = None
        self.model_fitted = None
        self.optimal_lag = None
        self.is_stationary = {}

    def check_stationarity(self, series: pd.Series, name: str = "series") -> bool:
        """
        Check if a time series is stationary using ADF test.

        Parameters
        ----------
        series : pd.Series
            Time series to test.
        name : str, default='series'
            Name of the series for reporting.

        Returns
        -------
        bool
            True if series is stationary.
        """
        result = adfuller(series.dropna(), autolag="AIC")
        p_value = result[1]

        is_stationary = p_value < 0.05

        if self.verbose:
            print(f"ADF Test for {name}:")
            print(f"  Test Statistic: {result[0]:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Stationary: {is_stationary}")

        return is_stationary

    def make_stationary(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Make time series stationary through differencing if needed.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with time series columns.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            Stationary dataframe and dictionary of differencing orders.
        """
        df_stationary = df.copy()
        diff_orders = {}

        for col in df.columns:
            is_stationary = self.check_stationarity(df[col], name=col)
            self.is_stationary[col] = is_stationary

            if not is_stationary:
                # Apply first-order differencing
                df_stationary[col] = df[col].diff()
                diff_orders[col] = 1

                # Check again
                is_stationary_after = self.check_stationarity(
                    df_stationary[col].dropna(), name=f"{col} (differenced)"
                )

                if not is_stationary_after and self.verbose:
                    print(f"Warning: {col} may need higher-order differencing")
            else:
                diff_orders[col] = 0

        # Drop NaN values from differencing
        df_stationary = df_stationary.dropna()

        return df_stationary, diff_orders

    def select_lag_order(self, df: pd.DataFrame) -> int:
        """
        Select optimal lag order using information criterion.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with time series columns.

        Returns
        -------
        int
            Optimal lag order.
        """
        model = VAR(df)
        lag_order_results = model.select_order(maxlags=self.max_lags)

        if self.ic == "aic":
            optimal_lag = lag_order_results.aic
        elif self.ic == "bic":
            optimal_lag = lag_order_results.bic
        else:
            raise ValueError(f"Unknown information criterion: {self.ic}")

        if self.verbose:
            print(f"\nLag Order Selection ({self.ic.upper()}):")
            print(f"  Optimal lag: {optimal_lag}")

        return optimal_lag

    def fit(self, df: pd.DataFrame, make_stationary: bool = True) -> "VARForecaster":
        """
        Fit VAR model.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with time series columns (each column is a series).
        make_stationary : bool, default=True
            Whether to make series stationary before fitting.

        Returns
        -------
        VARForecaster
            Fitted forecaster.
        """
        df_model = df.copy()

        # Make stationary if requested
        if make_stationary:
            df_model, self.diff_orders = self.make_stationary(df_model)
        else:
            self.diff_orders = {col: 0 for col in df.columns}

        # Select optimal lag order
        self.optimal_lag = self.select_lag_order(df_model)

        # Fit model
        self.model = VAR(df_model)
        self.model_fitted = self.model.fit(maxlags=self.optimal_lag, ic=self.ic, trend=self.trend)

        if self.verbose:
            print("\nModel Summary:")
            print(self.model_fitted.summary())

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        pd.DataFrame
            Forecasted values.
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before prediction.")

        # Generate forecast
        forecast = self.model_fitted.forecast(
            self.model_fitted.endog[-self.optimal_lag :], steps=steps
        )

        # Convert to dataframe
        forecast_df = pd.DataFrame(forecast, columns=self.model_fitted.names)

        return forecast_df

    def granger_causality(self, max_lag: Optional[int] = None) -> dict:
        """
        Test for Granger causality between variables.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag to test. If None, uses optimal_lag.

        Returns
        -------
        dict
            Dictionary of causality test results.
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before causality testing.")

        if max_lag is None:
            max_lag = self.optimal_lag

        results = {}
        variables = self.model_fitted.names

        for i, var1 in enumerate(variables):
            for var2 in variables[i + 1 :]:
                try:
                    # Test if var1 Granger-causes var2
                    test_result = grangercausalitytests(
                        self.model_fitted.endog[[var2, var1]], maxlag=max_lag, verbose=False
                    )

                    # Extract p-values
                    p_values = [
                        test_result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
                    ]
                    min_p_value = min(p_values)

                    results[f"{var1} -> {var2}"] = {
                        "p_value": min_p_value,
                        "significant": min_p_value < 0.05,
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"Error testing {var1} -> {var2}: {e}")

        return results

    def impulse_response(self, periods: int = 10) -> pd.DataFrame:
        """
        Compute impulse response functions.

        Parameters
        ----------
        periods : int, default=10
            Number of periods for IRF.

        Returns
        -------
        pd.DataFrame
            Impulse response functions.
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before computing IRF.")

        irf = self.model_fitted.irf(periods)
        return irf
