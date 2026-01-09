"""
Evaluation metrics and utilities for forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float, default=1e-10
        Small value to avoid division by zero.

    Returns
    -------
    float
        MAPE value (as percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float, default=1e-10
        Small value to avoid division by zero.

    Returns
    -------
    float
        SMAPE value (as percentage).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon

    return np.mean(numerator / denominator) * 100


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    y_train : np.ndarray
        Training data for computing scale.
    seasonal_period : int, default=1
        Seasonal period for naive forecast baseline.

    Returns
    -------
    float
        MASE value.
    """
    # Forecast error
    forecast_error = np.mean(np.abs(y_true - y_pred))

    # Naive forecast error on training data
    naive_error = np.mean(np.abs(np.diff(y_train, n=seasonal_period)))

    # Avoid division by zero
    if naive_error < 1e-10:
        return np.inf

    return forecast_error / naive_error


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    y_train : np.ndarray, optional
        Training data (required for MASE).
    metrics : list, optional
        List of metrics to compute. Default is all metrics.

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape", "smape", "r2"]
        if y_train is not None:
            metrics.append("mase")

    results = {}

    for metric in metrics:
        try:
            if metric == "mae":
                results[metric] = mae(y_true, y_pred)
            elif metric == "rmse":
                results[metric] = rmse(y_true, y_pred)
            elif metric == "mape":
                results[metric] = mape(y_true, y_pred)
            elif metric == "smape":
                results[metric] = smape(y_true, y_pred)
            elif metric == "mase":
                if y_train is not None:
                    results[metric] = mase(y_true, y_pred, y_train)
                else:
                    results[metric] = np.nan
            elif metric == "r2":
                results[metric] = r2_score(y_true, y_pred)
            else:
                warnings.warn(f"Unknown metric: {metric}")
        except Exception as e:
            warnings.warn(f"Error computing {metric}: {e}")
            results[metric] = np.nan

    return results


class TimeSeriesCrossValidator:
    """
    Time series cross-validation with expanding or sliding window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 30,
        strategy: str = "expanding",
        gap: int = 0,
    ):
        """
        Initialize cross-validator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of splits.
        test_size : int, default=30
            Size of test set in each split.
        strategy : str, default='expanding'
            Strategy: 'expanding' (growing train set) or 'sliding' (fixed train size).
        gap : int, default=0
            Gap between train and test sets.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.strategy = strategy
        self.gap = gap

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        list
            List of (train_indices, test_indices) tuples.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate minimum training size
        min_train_size = n_samples - (self.n_splits * self.test_size) - (self.n_splits * self.gap)

        if min_train_size <= 0:
            raise ValueError("Not enough data for the specified number of splits.")

        splits = []

        for i in range(self.n_splits):
            # Calculate test start and end
            test_end = n_samples - (i * self.test_size)
            test_start = test_end - self.test_size

            # Calculate train end (with gap)
            train_end = test_start - self.gap

            if self.strategy == "expanding":
                # Train from beginning to train_end
                train_start = 0
            elif self.strategy == "sliding":
                # Fixed-size training window
                train_start = max(0, train_end - min_train_size)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            if train_start >= train_end:
                continue

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            splits.append((train_indices, test_indices))

        # Reverse to get chronological order
        return list(reversed(splits))


class ModelEvaluator:
    """
    Evaluate and compare multiple forecasting models.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator.

        Parameters
        ----------
        metrics : list, optional
            List of metrics to compute.
        """
        self.metrics = metrics if metrics is not None else ["mae", "rmse", "mape", "smape", "r2"]
        self.results = {}

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single model.

        Parameters
        ----------
        model_name : str
            Name of the model.
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        y_train : np.ndarray, optional
            Training data.

        Returns
        -------
        dict
            Dictionary of metrics.
        """
        metrics = compute_metrics(y_true, y_pred, y_train, self.metrics)
        self.results[model_name] = metrics
        return metrics

    def get_results(self) -> pd.DataFrame:
        """
        Get results as a dataframe.

        Returns
        -------
        pd.DataFrame
            Results dataframe with models as rows and metrics as columns.
        """
        return pd.DataFrame(self.results).T

    def get_best_model(self, metric: str = "mae", lower_is_better: bool = True) -> str:
        """
        Get the best performing model.

        Parameters
        ----------
        metric : str, default='mae'
            Metric to use for comparison.
        lower_is_better : bool, default=True
            Whether lower values are better.

        Returns
        -------
        str
            Name of the best model.
        """
        results_df = self.get_results()

        if metric not in results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        if lower_is_better:
            return results_df[metric].idxmin()
        else:
            return results_df[metric].idxmax()

    def compare_models(self, metric: str = "mae") -> pd.DataFrame:
        """
        Compare models by a specific metric.

        Parameters
        ----------
        metric : str, default='mae'
            Metric to compare.

        Returns
        -------
        pd.DataFrame
            Sorted comparison dataframe.
        """
        results_df = self.get_results()

        if metric not in results_df.columns:
            raise ValueError(f"Metric {metric} not found in results.")

        comparison = results_df[[metric]].sort_values(metric)
        comparison["rank"] = range(1, len(comparison) + 1)

        return comparison


def rolling_forecast_evaluation(
    model,
    df: pd.DataFrame,
    target_col: str,
    forecast_horizon: int,
    n_windows: int = 10,
    retrain: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform rolling forecast evaluation.

    Parameters
    ----------
    model : object
        Model with fit() and predict() methods.
    df : pd.DataFrame
        Time series dataframe.
    target_col : str
        Name of target column.
    forecast_horizon : int
        Forecast horizon.
    n_windows : int, default=10
        Number of rolling windows.
    retrain : bool, default=True
        Whether to retrain model in each window.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        Lists of predictions and actuals for each window.
    """
    predictions_list = []
    actuals_list = []

    n_samples = len(df)
    window_size = (n_samples - forecast_horizon) // n_windows

    for i in range(n_windows):
        # Define train and test sets
        train_end = window_size * (i + 1)
        test_start = train_end
        test_end = test_start + forecast_horizon

        if test_end > n_samples:
            break

        train_data = df.iloc[:train_end]
        test_data = df.iloc[test_start:test_end]

        # Train model
        if retrain or i == 0:
            model.fit(train_data[target_col])

        # Generate forecast
        predictions = model.predict(forecast_horizon)

        # Store results
        predictions_list.append(predictions)
        actuals_list.append(test_data[target_col].values)

    return predictions_list, actuals_list
