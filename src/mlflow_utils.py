"""
MLflow integration utilities for experiment tracking and model management.
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.
    """

    def __init__(
        self,
        experiment_name: str = "retail_demand_forecasting",
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Parameters
        ----------
        experiment_name : str, default='retail_demand_forecasting'
            Name of the MLflow experiment.
        tracking_uri : str, optional
            MLflow tracking URI.
        """
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name for the run.
        nested : bool, default=False
            Whether this is a nested run.

        Returns
        -------
        mlflow.ActiveRun
            Active run context.
        """
        return mlflow.start_run(run_name=run_name, nested=nested)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.

        Parameters
        ----------
        params : dict
            Parameters to log.
        """
        # Flatten nested dictionaries
        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"Warning: Could not log param {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Parameters
        ----------
        metrics : dict
            Metrics to log.
        step : int, optional
            Step number for the metrics.
        """
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f"Warning: Could not log metric {key}: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log model to MLflow.

        Parameters
        ----------
        model : object
            Model to log.
        artifact_path : str, default='model'
            Artifact path within the run.
        registered_model_name : str, optional
            Name for model registry.
        """
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        except Exception as e:
            print(f"Warning: Could not log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file.

        Parameters
        ----------
        local_path : str
            Path to local file.
        artifact_path : str, optional
            Artifact path within the run.
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict, filename: str) -> None:
        """
        Log a dictionary as JSON artifact.

        Parameters
        ----------
        dictionary : dict
            Dictionary to log.
        filename : str
            Filename for the artifact.
        """
        with mlflow.start_run(nested=True):
            mlflow.log_dict(dictionary, filename)

    def log_figure(self, figure, filename: str) -> None:
        """
        Log a matplotlib figure.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to log.
        filename : str
            Filename for the artifact.
        """
        mlflow.log_figure(figure, filename)

    def log_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        Log a dataframe as CSV artifact.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to log.
        filename : str
            Filename for the artifact.
        """
        temp_path = f"/tmp/{filename}"
        df.to_csv(temp_path, index=False)
        self.log_artifact(temp_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.

        Parameters
        ----------
        tags : dict
            Tags to set.
        """
        for key, value in tags.items():
            mlflow.set_tag(key, str(value))

    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """
        Flatten a nested dictionary.

        Parameters
        ----------
        d : dict
            Dictionary to flatten.
        parent_key : str, default=''
            Parent key for recursion.
        sep : str, default='.'
            Separator for nested keys.

        Returns
        -------
        dict
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_best_run(
        self, metric: str = "mae", ascending: bool = True
    ) -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric.

        Parameters
        ----------
        metric : str, default='mae'
            Metric to optimize.
        ascending : bool, default=True
            Whether lower is better.

        Returns
        -------
        mlflow.entities.Run or None
            Best run.
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if len(runs) > 0:
            return runs.iloc[0]
        return None

    def compare_runs(self, metric: str = "mae") -> pd.DataFrame:
        """
        Compare all runs by a metric.

        Parameters
        ----------
        metric : str, default='mae'
            Metric to compare.

        Returns
        -------
        pd.DataFrame
            Comparison dataframe.
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])

        if len(runs) == 0:
            return pd.DataFrame()

        # Select relevant columns
        columns = ["run_id", "start_time", "status"]
        columns.extend([col for col in runs.columns if col.startswith("params.")])
        columns.extend([col for col in runs.columns if col.startswith("metrics.")])

        return runs[columns].sort_values(f"metrics.{metric}")


def log_forecast_experiment(
    tracker: MLflowTracker,
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    forecasts: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    feature_importance: Optional[pd.DataFrame] = None,
) -> None:
    """
    Log a complete forecasting experiment.

    Parameters
    ----------
    tracker : MLflowTracker
        MLflow tracker instance.
    model_name : str
        Name of the model.
    model : object
        Trained model.
    params : dict
        Model parameters.
    metrics : dict
        Evaluation metrics.
    forecasts : np.ndarray, optional
        Forecast values.
    actuals : np.ndarray, optional
        Actual values.
    feature_importance : pd.DataFrame, optional
        Feature importance dataframe.
    """
    with tracker.start_run(run_name=model_name):
        # Log parameters
        tracker.log_params(params)

        # Log metrics
        tracker.log_metrics(metrics)

        # Log model
        tracker.log_model(model, artifact_path="model", registered_model_name=model_name)

        # Log forecasts and actuals
        if forecasts is not None and actuals is not None:
            results_df = pd.DataFrame({"actual": actuals, "forecast": forecasts})
            tracker.log_dataframe(results_df, "forecasts.csv")

        # Log feature importance
        if feature_importance is not None:
            tracker.log_dataframe(feature_importance, "feature_importance.csv")

        # Set tags
        tracker.set_tags({"model_type": model_name, "framework": "custom"})
