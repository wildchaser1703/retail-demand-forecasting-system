"""
Data preprocessing utilities for retail sales forecasting.
"""
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """Preprocessor for retail sales data."""

    def __init__(self):
        """Initialize preprocessor."""
        self.scalers = {}
        self.fill_values = {}

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "forward_fill", columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        strategy : str, default='forward_fill'
            Strategy to use: 'forward_fill', 'backward_fill', 'mean', 'median', 'zero'.
        columns : list, optional
            Columns to apply strategy to. If None, applies to all numeric columns.

        Returns
        -------
        pd.DataFrame
            Dataframe with missing values handled.
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if df[col].isna().any():
                if strategy == "forward_fill":
                    df[col] = df[col].fillna(method="ffill")
                elif strategy == "backward_fill":
                    df[col] = df[col].fillna(method="bfill")
                elif strategy == "mean":
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    self.fill_values[col] = fill_value
                elif strategy == "median":
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    self.fill_values[col] = fill_value
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: list,
        method: str = "clip",
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """
        Handle outliers in specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        columns : list
            Columns to process.
        method : str, default='clip'
            Method to use: 'clip' or 'remove'.
        lower_percentile : float, default=0.01
            Lower percentile for clipping.
        upper_percentile : float, default=0.99
            Upper percentile for clipping.

        Returns
        -------
        pd.DataFrame
            Dataframe with outliers handled.
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            lower_bound = df[col].quantile(lower_percentile)
            upper_bound = df[col].quantile(upper_percentile)

            if method == "clip":
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == "remove":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            else:
                raise ValueError(f"Unknown method: {method}")

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: list,
        method: str = "standard",
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale features using standardization or normalization.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        columns : list
            Columns to scale.
        method : str, default='standard'
            Scaling method: 'standard' or 'minmax'.
        fit : bool, default=True
            Whether to fit the scaler. Set to False for test data.

        Returns
        -------
        pd.DataFrame
            Dataframe with scaled features.
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == "standard":
                if fit or col not in self.scalers:
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    df[col] = self.scalers[col].transform(df[[col]])

            elif method == "minmax":
                if fit or col not in self.scalers:
                    scaler = MinMaxScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    df[col] = self.scalers[col].transform(df[[col]])

            else:
                raise ValueError(f"Unknown method: {method}")

        return df

    def resample_time_series(
        self,
        df: pd.DataFrame,
        freq: str = "D",
        agg_func: str = "sum",
        date_column: str = "date",
        group_columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Resample time series data to a different frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        freq : str, default='D'
            Frequency to resample to ('D', 'W', 'M', etc.).
        agg_func : str, default='sum'
            Aggregation function.
        date_column : str, default='date'
            Name of the date column.
        group_columns : list, optional
            Columns to group by before resampling.

        Returns
        -------
        pd.DataFrame
            Resampled dataframe.
        """
        df = df.copy()

        if group_columns:
            # Resample within each group
            resampled_dfs = []
            for group_vals, group_df in df.groupby(group_columns):
                group_df = group_df.set_index(date_column)
                resampled = group_df.resample(freq).agg(agg_func).reset_index()

                # Add back group columns
                if isinstance(group_vals, tuple):
                    for i, col in enumerate(group_columns):
                        resampled[col] = group_vals[i]
                else:
                    resampled[group_columns[0]] = group_vals

                resampled_dfs.append(resampled)

            return pd.concat(resampled_dfs, ignore_index=True)
        else:
            df = df.set_index(date_column)
            return df.resample(freq).agg(agg_func).reset_index()

    def create_sequences(
        self,
        series: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.

        Parameters
        ----------
        series : np.ndarray
            Input time series.
        sequence_length : int
            Length of input sequences.
        forecast_horizon : int, default=1
            Number of steps to forecast.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X (input sequences) and y (target values).
        """
        X, y = [], []

        for i in range(len(series) - sequence_length - forecast_horizon + 1):
            X.append(series[i : i + sequence_length])
            y.append(series[i + sequence_length : i + sequence_length + forecast_horizon])

        return np.array(X), np.array(y)
