"""
Machine Learning models for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.ensemble import RandomForestRegressor
import joblib


class MLForecaster:
    """
    Base class for ML-based time series forecasters.
    """

    def __init__(self, model_type: str = "xgboost", **model_params):
        """
        Initialize ML forecaster.

        Parameters
        ----------
        model_type : str, default='xgboost'
            Type of model: 'xgboost', 'lightgbm', or 'random_forest'.
        **model_params : dict
            Parameters to pass to the model.
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor

            default_params = {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
            default_params.update(self.model_params)
            self.model = XGBRegressor(**default_params)

        elif self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor

            default_params = {
                "objective": "regression",
                "metric": "mae",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
            }
            default_params.update(self.model_params)
            self.model = LGBMRegressor(**default_params)

        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            self.model = RandomForestRegressor(**default_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
    ) -> "MLForecaster":
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        eval_set : list, optional
            Evaluation set for early stopping [(X_val, y_val)].
        early_stopping_rounds : int, optional
            Number of rounds for early stopping.
        verbose : bool, default=False
            Whether to print training progress.

        Returns
        -------
        MLForecaster
            Fitted forecaster.
        """
        self._initialize_model()
        self.feature_names = X.columns.tolist()

        # Fit with early stopping if supported
        if self.model_type in ["xgboost", "lightgbm"] and eval_set is not None:
            fit_params = {
                "eval_set": eval_set,
                "verbose": verbose,
            }
            if early_stopping_rounds is not None:
                if self.model_type == "xgboost":
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
                else:  # lightgbm
                    lgbm = __import__("lightgbm")
                    fit_params["callbacks"] = [
                        lgbm.early_stopping(early_stopping_rounds, verbose=verbose)
                    ]

            self.model.fit(X, y, **fit_params)
        else:
            self.model.fit(X, y)

        # Store feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        return self.model.predict(X)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe.
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first.")

        if top_n is not None:
            return self.feature_importance.head(top_n)
        return self.feature_importance

    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before saving.")

        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
            },
            path,
        )

    def load_model(self, path: str) -> "MLForecaster":
        """
        Load model from disk.

        Parameters
        ----------
        path : str
            Path to load the model from.

        Returns
        -------
        MLForecaster
            Loaded forecaster.
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.feature_names = data["feature_names"]
        self.feature_importance = data["feature_importance"]

        return self


class TimeSeriesMLForecaster:
    """
    ML forecaster with automatic feature engineering for time series.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        **model_params,
    ):
        """
        Initialize time series ML forecaster.

        Parameters
        ----------
        model_type : str, default='xgboost'
            Type of model.
        lags : list, optional
            Lag features to create.
        rolling_windows : list, optional
            Rolling window sizes for features.
        **model_params : dict
            Parameters to pass to the model.
        """
        self.model_type = model_type
        self.lags = lags if lags is not None else [1, 7, 14, 28]
        self.rolling_windows = (
            rolling_windows if rolling_windows is not None else [7, 14, 28]
        )
        if not self.lags or any(lag < 1 for lag in self.lags):
            raise ValueError("Lags must be positive")
        if not self.rolling_windows or any(
            window < 2 for window in self.rolling_windows
        ):
            raise ValueError("Rolling windows must be at least 2")
        self.model_params = model_params
        self.forecaster = MLForecaster(model_type=model_type, **model_params)

    def create_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Create time series features from a series.

        Parameters
        ----------
        series : pd.Series
            Time series data.

        Returns
        -------
        pd.DataFrame
            Feature matrix.
        """
        features = pd.DataFrame(index=series.index)

        # Lag features
        for lag in self.lags:
            features[f"lag_{lag}"] = series.shift(lag)

        # Rolling features
        for window in self.rolling_windows:
            features[f"rolling_mean_{window}"] = (
                series.shift(1).rolling(window=window).mean()
            )
            features[f"rolling_std_{window}"] = (
                series.shift(1).rolling(window=window).std()
            )
            features[f"rolling_min_{window}"] = (
                series.shift(1).rolling(window=window).min()
            )
            features[f"rolling_max_{window}"] = (
                series.shift(1).rolling(window=window).max()
            )

        # Temporal features if index is datetime
        if isinstance(series.index, pd.DatetimeIndex):
            features["day_of_week"] = series.index.dayofweek
            features["day_of_month"] = series.index.day
            features["month"] = series.index.month
            features["quarter"] = series.index.quarter
            features["is_weekend"] = (series.index.dayofweek >= 5).astype(int)

        return features

    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        eval_series: Optional[pd.Series] = None,
        eval_exog: Optional[pd.DataFrame] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> "TimeSeriesMLForecaster":
        """
        Fit the model.

        Parameters
        ----------
        series : pd.Series
            Target time series.
        exog : pd.DataFrame, optional
            Exogenous features.
        eval_series : pd.Series, optional
            Evaluation time series for early stopping.
        eval_exog : pd.DataFrame, optional
            Evaluation exogenous features.
        early_stopping_rounds : int, optional
            Number of rounds for early stopping.

        Returns
        -------
        TimeSeriesMLForecaster
            Fitted forecaster.
        """
        if not series.index.is_unique or not series.index.is_monotonic_increasing:
            raise ValueError("Series index must be unique and increasing")
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError("Series must contain finite values")
        if isinstance(series.index, pd.DatetimeIndex):
            self.frequency = pd.infer_freq(series.index)
            if self.frequency is None:
                raise ValueError("Series must have a regular frequency")
        self.history = series.copy()
        self.exog_columns = list(exog.columns) if exog is not None else []
        if exog is not None and not exog.index.equals(series.index):
            raise ValueError("Exogenous features must align with the series index")
        # Create features
        X = self.create_features(series)

        # Add exogenous features
        if exog is not None:
            X = pd.concat([X, exog], axis=1)

        # Remove rows with NaN (from lag/rolling features)
        y = series.loc[X.index]
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        if X.empty:
            raise ValueError("Insufficient history for requested lags/windows")

        # Prepare evaluation set if provided
        eval_set = None
        if eval_series is not None:
            X_eval = self.create_features(eval_series)
            if eval_exog is not None:
                X_eval = pd.concat([X_eval, eval_exog], axis=1)

            y_eval = eval_series.loc[X_eval.index]
            mask_eval = ~X_eval.isna().any(axis=1)
            X_eval = X_eval[mask_eval]
            y_eval = y_eval[mask_eval]

            eval_set = [(X_eval, y_eval)]

        # Fit model
        self.forecaster.fit(
            X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds
        )

        return self

    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate multi-step forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : pd.DataFrame, optional
            Future exogenous features.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if not hasattr(self, "history") or self.forecaster.model is None:
            raise ValueError("Fit the model before prediction")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.exog_columns:
            if (
                exog is None
                or len(exog) != steps
                or list(exog.columns) != self.exog_columns
            ):
                raise ValueError(
                    "Provide aligned future exogenous features for every step"
                )
        elif exog is not None:
            raise ValueError("Model was fitted without exogenous features")
        history = self.history.copy()
        forecasts = []
        for step in range(steps):
            next_index = (
                history.index[-1] + pd.tseries.frequencies.to_offset(self.frequency)
                if isinstance(history.index, pd.DatetimeIndex)
                else history.index[-1] + 1
            )
            if exog is not None and exog.index[step] != next_index:
                raise ValueError("Future exogenous index must match forecast dates")
            extended = pd.concat([history, pd.Series([np.nan], index=[next_index])])
            features = self.create_features(extended).iloc[[-1]].copy()
            if exog is not None:
                for col in self.exog_columns:
                    features[col] = exog.iloc[step][col]
            prediction = float(self.forecaster.predict(features)[0])
            forecasts.append(prediction)
            extended.iloc[-1] = prediction
            history = extended
        return np.asarray(forecasts)

    def get_feature_importance(self, top_n: Optional[int] = 20) -> pd.DataFrame:
        """Get feature importance."""
        return self.forecaster.get_feature_importance(top_n=top_n)
