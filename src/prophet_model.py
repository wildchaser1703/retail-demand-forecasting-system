"""
Facebook Prophet model for time series forecasting.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")


class ProphetForecaster:
    """
    Facebook Prophet forecaster wrapper.
    """

    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        interval_width: float = 0.95,
    ):
        """
        Initialize Prophet forecaster.

        Parameters
        ----------
        growth : str, default='linear'
            Growth type: 'linear' or 'logistic'.
        seasonality_mode : str, default='multiplicative'
            Seasonality mode: 'additive' or 'multiplicative'.
        changepoint_prior_scale : float, default=0.05
            Flexibility of trend changes.
        seasonality_prior_scale : float, default=10.0
            Strength of seasonality.
        holidays_prior_scale : float, default=10.0
            Strength of holiday effects.
        daily_seasonality : bool, default=False
            Whether to fit daily seasonality.
        weekly_seasonality : bool, default=True
            Whether to fit weekly seasonality.
        yearly_seasonality : bool, default=True
            Whether to fit yearly seasonality.
        interval_width : float, default=0.95
            Width of uncertainty intervals.
        """
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.interval_width = interval_width

        self.model = None
        self.regressors = []

    def _prepare_data(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """
        Prepare data in Prophet format.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        date_col : str
            Name of date column.
        target_col : str
            Name of target column.

        Returns
        -------
        pd.DataFrame
            Dataframe in Prophet format (ds, y columns).
        """
        prophet_df = pd.DataFrame(
            {"ds": pd.to_datetime(df[date_col]), "y": df[target_col].values}
        )

        # Add regressors if they exist
        for regressor in self.regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor].values

        return prophet_df

    def add_regressor(self, name: str, prior_scale: Optional[float] = None) -> "ProphetForecaster":
        """
        Add a custom regressor.

        Parameters
        ----------
        name : str
            Name of the regressor column.
        prior_scale : float, optional
            Prior scale for the regressor.

        Returns
        -------
        ProphetForecaster
            Self for method chaining.
        """
        if name not in self.regressors:
            self.regressors.append(name)

        return self

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
        holidays: Optional[pd.DataFrame] = None,
    ) -> "ProphetForecaster":
        """
        Fit Prophet model.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe.
        date_col : str, default='date'
            Name of date column.
        target_col : str, default='sales'
            Name of target column.
        holidays : pd.DataFrame, optional
            Dataframe with holiday information.

        Returns
        -------
        ProphetForecaster
            Fitted forecaster.
        """
        # Initialize model
        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            interval_width=self.interval_width,
        )

        # Add holidays if provided
        if holidays is not None:
            self.model.holidays = holidays

        # Add regressors
        for regressor in self.regressors:
            self.model.add_regressor(regressor)

        # Prepare data
        prophet_df = self._prepare_data(df, date_col, target_col)

        # Fit model
        self.model.fit(prophet_df)

        return self

    def predict(
        self,
        steps: int,
        freq: str = "D",
        future_regressors: Optional[pd.DataFrame] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        freq : str, default='D'
            Frequency of predictions.
        future_regressors : pd.DataFrame, optional
            Future values of regressors.
        include_history : bool, default=False
            Whether to include historical fitted values.

        Returns
        -------
        pd.DataFrame
            Forecast dataframe with predictions and uncertainty intervals.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        # Create future dataframe
        if include_history:
            future = self.model.make_future_dataframe(periods=steps, freq=freq)
        else:
            future = self.model.make_future_dataframe(periods=steps, freq=freq, include_history=False)

        # Add future regressor values if provided
        if future_regressors is not None:
            for regressor in self.regressors:
                if regressor in future_regressors.columns:
                    future[regressor] = future_regressors[regressor].values

        # Generate forecast
        forecast = self.model.predict(future)

        return forecast

    def plot_forecast(self, forecast: pd.DataFrame) -> None:
        """
        Plot forecast.

        Parameters
        ----------
        forecast : pd.DataFrame
            Forecast dataframe from predict().
        """
        if self.model is None:
            raise ValueError("Model must be fitted first.")

        self.model.plot(forecast)

    def plot_components(self, forecast: pd.DataFrame) -> None:
        """
        Plot forecast components (trend, seasonality, etc.).

        Parameters
        ----------
        forecast : pd.DataFrame
            Forecast dataframe from predict().
        """
        if self.model is None:
            raise ValueError("Model must be fitted first.")

        self.model.plot_components(forecast)

    def get_changepoints(self) -> pd.DataFrame:
        """
        Get detected changepoints.

        Returns
        -------
        pd.DataFrame
            Dataframe with changepoint dates and trend changes.
        """
        if self.model is None:
            raise ValueError("Model must be fitted first.")

        return pd.DataFrame(
            {
                "changepoint": self.model.changepoints,
                "delta": self.model.params["delta"].mean(axis=0),
            }
        )
