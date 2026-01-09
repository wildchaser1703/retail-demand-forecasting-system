"""
Unified forecasting interface and pipeline.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Any


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    """

    def __init__(self):
        """Initialize forecaster."""
        self.is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseForecaster":
        """
        Fit the forecasting model.

        Returns
        -------
        BaseForecaster
            Fitted forecaster.
        """
        pass

    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        pass

    def fit_predict(self, *args, steps: int, **kwargs) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        self.fit(*args, **kwargs)
        return self.predict(steps)


class ForecastingPipeline:
    """
    Pipeline for end-to-end forecasting workflow.
    """

    def __init__(
        self,
        preprocessor: Optional[Any] = None,
        feature_engineer: Optional[Any] = None,
        model: Optional[BaseForecaster] = None,
    ):
        """
        Initialize forecasting pipeline.

        Parameters
        ----------
        preprocessor : object, optional
            Data preprocessor.
        feature_engineer : object, optional
            Feature engineer.
        model : BaseForecaster, optional
            Forecasting model.
        """
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.model = model
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "sales", **kwargs) -> "ForecastingPipeline":
        """
        Fit the pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe.
        target_col : str, default='sales'
            Name of target column.
        **kwargs : dict
            Additional arguments for model fitting.

        Returns
        -------
        ForecastingPipeline
            Fitted pipeline.
        """
        data = df.copy()

        # Preprocessing
        if self.preprocessor is not None:
            data = self.preprocessor.fit_transform(data)

        # Feature engineering
        if self.feature_engineer is not None:
            data = self.feature_engineer.fit_transform(data)

        # Fit model
        if self.model is not None:
            self.model.fit(data[target_col], **kwargs)

        self.is_fitted = True
        return self

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        **kwargs : dict
            Additional arguments for prediction.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction.")

        if self.model is None:
            raise ValueError("No model specified in pipeline.")

        return self.model.predict(steps, **kwargs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data through preprocessing and feature engineering.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe.
        """
        data = df.copy()

        if self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        if self.feature_engineer is not None:
            data = self.feature_engineer.transform(data)

        return data


class MultiSeriesForecaster:
    """
    Forecaster for multiple time series (e.g., multiple stores/products).
    """

    def __init__(self, base_model: BaseForecaster, group_columns: list):
        """
        Initialize multi-series forecaster.

        Parameters
        ----------
        base_model : BaseForecaster
            Base forecasting model to use for each series.
        group_columns : list
            Columns to group by (e.g., ['store_id', 'product_id']).
        """
        self.base_model = base_model
        self.group_columns = group_columns
        self.models = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "sales", **kwargs) -> "MultiSeriesForecaster":
        """
        Fit a separate model for each series.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe.
        target_col : str, default='sales'
            Name of target column.
        **kwargs : dict
            Additional arguments for model fitting.

        Returns
        -------
        MultiSeriesForecaster
            Fitted forecaster.
        """
        # Group by specified columns
        for group_vals, group_df in df.groupby(self.group_columns):
            # Create a copy of the base model for this group
            model = type(self.base_model)(**self.base_model.__dict__)

            # Fit model
            model.fit(group_df[target_col].values, **kwargs)

            # Store model
            self.models[group_vals] = model

        self.is_fitted = True
        return self

    def predict(self, steps: int, groups: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """
        Generate forecasts for all or specified groups.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        groups : list, optional
            Specific groups to forecast. If None, forecasts all groups.
        **kwargs : dict
            Additional arguments for prediction.

        Returns
        -------
        pd.DataFrame
            Dataframe with forecasts for each group.
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction.")

        forecasts = []

        groups_to_forecast = groups if groups is not None else self.models.keys()

        for group_vals in groups_to_forecast:
            if group_vals not in self.models:
                continue

            # Generate forecast
            forecast = self.models[group_vals].predict(steps, **kwargs)

            # Create dataframe
            forecast_df = pd.DataFrame({"forecast": forecast})

            # Add group columns
            if isinstance(group_vals, tuple):
                for i, col in enumerate(self.group_columns):
                    forecast_df[col] = group_vals[i]
            else:
                forecast_df[self.group_columns[0]] = group_vals

            forecasts.append(forecast_df)

        return pd.concat(forecasts, ignore_index=True)

    def get_model(self, group_vals: tuple) -> BaseForecaster:
        """
        Get the model for a specific group.

        Parameters
        ----------
        group_vals : tuple
            Group values.

        Returns
        -------
        BaseForecaster
            Model for the specified group.
        """
        if group_vals not in self.models:
            raise ValueError(f"No model found for group: {group_vals}")

        return self.models[group_vals]
