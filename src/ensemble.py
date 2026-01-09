"""
Ensemble methods for combining multiple forecasting models.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LinearRegression


class SimpleEnsemble:
    """
    Simple ensemble combining forecasts through averaging or weighted averaging.
    """

    def __init__(self, method: str = "mean", weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble.

        Parameters
        ----------
        method : str, default='mean'
            Ensemble method: 'mean', 'median', or 'weighted'.
        weights : dict, optional
            Weights for each model (required if method='weighted').
        """
        self.method = method
        self.weights = weights

        if method == "weighted" and weights is None:
            raise ValueError("Weights must be provided for weighted ensemble.")

    def combine(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine forecasts from multiple models.

        Parameters
        ----------
        forecasts : dict
            Dictionary mapping model names to forecast arrays.

        Returns
        -------
        np.ndarray
            Combined forecast.
        """
        if len(forecasts) == 0:
            raise ValueError("No forecasts provided.")

        # Stack forecasts
        forecast_matrix = np.column_stack(list(forecasts.values()))

        if self.method == "mean":
            return np.mean(forecast_matrix, axis=1)

        elif self.method == "median":
            return np.median(forecast_matrix, axis=1)

        elif self.method == "weighted":
            # Apply weights
            model_names = list(forecasts.keys())
            weight_array = np.array([self.weights.get(name, 1.0) for name in model_names])

            # Normalize weights
            weight_array = weight_array / weight_array.sum()

            return np.average(forecast_matrix, axis=1, weights=weight_array)

        else:
            raise ValueError(f"Unknown method: {self.method}")


class StackingEnsemble:
    """
    Stacking ensemble using a meta-learner.
    """

    def __init__(self, meta_learner=None):
        """
        Initialize stacking ensemble.

        Parameters
        ----------
        meta_learner : object, optional
            Meta-learner model. Default is LinearRegression.
        """
        self.meta_learner = meta_learner if meta_learner is not None else LinearRegression()
        self.model_names = None

    def fit(self, forecasts: Dict[str, np.ndarray], actuals: np.ndarray) -> "StackingEnsemble":
        """
        Fit the meta-learner.

        Parameters
        ----------
        forecasts : dict
            Dictionary mapping model names to forecast arrays.
        actuals : np.ndarray
            Actual values.

        Returns
        -------
        StackingEnsemble
            Fitted ensemble.
        """
        self.model_names = list(forecasts.keys())

        # Stack forecasts as features
        X = np.column_stack(list(forecasts.values()))

        # Fit meta-learner
        self.meta_learner.fit(X, actuals)

        return self

    def predict(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate ensemble predictions.

        Parameters
        ----------
        forecasts : dict
            Dictionary mapping model names to forecast arrays.

        Returns
        -------
        np.ndarray
            Ensemble predictions.
        """
        if self.model_names is None:
            raise ValueError("Ensemble must be fitted before prediction.")

        # Ensure same model order
        X = np.column_stack([forecasts[name] for name in self.model_names])

        return self.meta_learner.predict(X)


class AdaptiveEnsemble:
    """
    Adaptive ensemble that adjusts weights based on recent performance.
    """

    def __init__(self, window_size: int = 10, decay_factor: float = 0.9):
        """
        Initialize adaptive ensemble.

        Parameters
        ----------
        window_size : int, default=10
            Size of the window for computing recent performance.
        decay_factor : float, default=0.9
            Decay factor for exponential weighting of recent errors.
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.model_names = None
        self.recent_errors = {}
        self.weights = {}

    def update(self, forecasts: Dict[str, float], actual: float) -> None:
        """
        Update model weights based on new observation.

        Parameters
        ----------
        forecasts : dict
            Dictionary mapping model names to single forecast values.
        actual : float
            Actual value.
        """
        if self.model_names is None:
            self.model_names = list(forecasts.keys())
            for name in self.model_names:
                self.recent_errors[name] = []
                self.weights[name] = 1.0 / len(self.model_names)

        # Compute errors
        for name in self.model_names:
            error = abs(forecasts[name] - actual)
            self.recent_errors[name].append(error)

            # Keep only recent errors
            if len(self.recent_errors[name]) > self.window_size:
                self.recent_errors[name].pop(0)

        # Update weights based on recent performance
        self._update_weights()

    def _update_weights(self) -> None:
        """Update weights based on recent errors."""
        # Compute weighted average error for each model
        avg_errors = {}
        for name in self.model_names:
            errors = self.recent_errors[name]
            if len(errors) == 0:
                avg_errors[name] = 1.0
            else:
                # Apply exponential decay
                weights = np.array([self.decay_factor**i for i in range(len(errors) - 1, -1, -1)])
                weights = weights / weights.sum()
                avg_errors[name] = np.average(errors, weights=weights)

        # Convert errors to weights (inverse relationship)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        inv_errors = {name: 1.0 / (error + epsilon) for name, error in avg_errors.items()}

        # Normalize to sum to 1
        total = sum(inv_errors.values())
        self.weights = {name: weight / total for name, weight in inv_errors.items()}

    def predict(self, forecasts: Dict[str, float]) -> float:
        """
        Generate weighted ensemble prediction.

        Parameters
        ----------
        forecasts : dict
            Dictionary mapping model names to single forecast values.

        Returns
        -------
        float
            Ensemble prediction.
        """
        if self.model_names is None:
            # First prediction - use simple average
            return np.mean(list(forecasts.values()))

        # Weighted combination
        prediction = sum(forecasts[name] * self.weights[name] for name in self.model_names)

        return prediction

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
