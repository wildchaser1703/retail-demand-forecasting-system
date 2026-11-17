"""
Feature engineering for retail demand forecasting.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Feature engineering for time series forecasting."""

    def __init__(self, df: pd.DataFrame, date_column: str = "date"):
        """
        Initialize feature engineer.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        date_column : str, default='date'
            Name of the date column.
        """
        self.df = df.copy()
        self.date_column = date_column

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            self.df[date_column] = pd.to_datetime(self.df[date_column])

    def _ordered(self, group_columns=None):
        keys = list(group_columns or []) + [self.date_column]
        if self.df.duplicated(keys).any():
            raise ValueError("Duplicate time points; provide group_columns for separate series")
        return self.df.sort_values(keys).copy()

    def add_temporal_features(self) -> pd.DataFrame:
        """
        Add temporal features from date column.

        Returns
        -------
        pd.DataFrame
            Dataframe with temporal features added.
        """
        df = self.df.copy()
        date_col = df[self.date_column]

        # Basic temporal features
        df["year"] = date_col.dt.year
        df["month"] = date_col.dt.month
        df["day"] = date_col.dt.day
        df["day_of_week"] = date_col.dt.dayofweek  # Monday=0, Sunday=6
        df["day_of_year"] = date_col.dt.dayofyear
        df["week_of_year"] = date_col.dt.isocalendar().week
        df["quarter"] = date_col.dt.quarter

        # Binary features
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = date_col.dt.is_month_start.astype(int)
        df["is_month_end"] = date_col.dt.is_month_end.astype(int)
        df["is_quarter_start"] = date_col.dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = date_col.dt.is_quarter_end.astype(int)

        # Cyclical encoding for periodic features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        self.df = df
        return df

    def add_holiday_features(self, country: str = "US") -> pd.DataFrame:
        """
        Add holiday features.

        Parameters
        ----------
        country : str, default='US'
            Country for holiday calendar.

        Returns
        -------
        pd.DataFrame
            Dataframe with holiday features added.
        """
        df = self.df.copy()

        if country != "US":
            raise ValueError("Only the US federal holiday calendar is supported")
        from pandas.tseries.holiday import USFederalHolidayCalendar

        dates = df[self.date_column]
        holidays = USFederalHolidayCalendar().holidays(
            start=dates.min() - pd.Timedelta(days=370), end=dates.max() + pd.Timedelta(days=370)
        )
        day_values = dates.to_numpy(dtype="datetime64[D]")
        holiday_values = holidays.to_numpy(dtype="datetime64[D]")
        next_idx = np.searchsorted(holiday_values, day_values, side="left")
        prev_idx = np.searchsorted(holiday_values, day_values, side="right") - 1
        df["is_holiday"] = dates.isin(holidays).astype(int)
        df["days_to_holiday"] = (holiday_values[next_idx] - day_values).astype(int)
        df["days_from_holiday"] = (day_values - holiday_values[prev_idx]).astype(int)

        self.df = df
        return df

    def add_lag_features(
        self,
        target_column: str = "sales",
        lags: Optional[List[int]] = None,
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add lag features.

        Parameters
        ----------
        target_column : str, default='sales'
            Column to create lags for.
        lags : list, optional
            List of lag periods. Default is [1, 7, 14, 28].
        group_columns : list, optional
            Columns to group by before creating lags.

        Returns
        -------
        pd.DataFrame
            Dataframe with lag features added.
        """
        if lags is None:
            lags = [1, 7, 14, 28]

        df = self._ordered(group_columns)

        if not lags or any(lag < 1 for lag in lags):
            raise ValueError("Lags must be positive")
        if group_columns:
            for lag in lags:
                df[f"{target_column}_lag_{lag}"] = df.groupby(group_columns)[target_column].shift(
                    lag
                )
        else:
            for lag in lags:
                df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

        self.df = df
        return df

    def add_rolling_features(
        self,
        target_column: str = "sales",
        windows: Optional[List[int]] = None,
        stats: Optional[List[str]] = None,
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add rolling window features.

        Parameters
        ----------
        target_column : str, default='sales'
            Column to create rolling features for.
        windows : list, optional
            List of window sizes. Default is [7, 14, 28].
        stats : list, optional
            List of statistics to compute. Default is ['mean', 'std', 'min', 'max'].
        group_columns : list, optional
            Columns to group by before computing rolling features.

        Returns
        -------
        pd.DataFrame
            Dataframe with rolling features added.
        """
        if windows is None:
            windows = [7, 14, 28]
        if stats is None:
            stats = ["mean", "std", "min", "max"]

        df = self._ordered(group_columns)

        if not windows or any(window < 2 for window in windows):
            raise ValueError("Rolling windows must be at least 2")
        for window in windows:
            for stat in stats:
                if stat not in {"mean", "std", "min", "max"}:
                    raise ValueError(f"Unsupported rolling statistic: {stat}")

                def compute(values):
                    return values.shift(1).rolling(window, min_periods=1).agg(stat)

                df[f"{target_column}_rolling_{window}_{stat}"] = (
                    df.groupby(group_columns)[target_column].transform(compute)
                    if group_columns
                    else compute(df[target_column])
                )

        self.df = df
        return df

    def add_diff_features(
        self,
        target_column: str = "sales",
        periods: Optional[List[int]] = None,
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add difference features.

        Parameters
        ----------
        target_column : str, default='sales'
            Column to create differences for.
        periods : list, optional
            List of periods for differencing. Default is [1, 7].
        group_columns : list, optional
            Columns to group by before differencing.

        Returns
        -------
        pd.DataFrame
            Dataframe with difference features added.
        """
        if periods is None:
            periods = [1, 7]

        df = self._ordered(group_columns)

        if not periods or any(period < 1 for period in periods):
            raise ValueError("Difference periods must be positive")
        for period in periods:
            col_name = f"{target_column}_diff_{period}"

            if group_columns:
                df[col_name] = df.groupby(group_columns)[target_column].transform(
                    lambda x: x.shift(1).diff(period)
                )
            else:
                df[col_name] = df[target_column].shift(1).diff(period)

        self.df = df
        return df

    def add_promotion_features(
        self, promo_column: str = "is_promo", group_columns=None
    ) -> pd.DataFrame:
        """
        Add promotion-related features.

        Parameters
        ----------
        promo_column : str, default='is_promo'
            Name of the promotion indicator column.

        Returns
        -------
        pd.DataFrame
            Dataframe with promotion features added.
        """
        if promo_column not in self.df.columns:
            return self.df

        df = self._ordered(group_columns)
        df["days_since_promo"] = 999
        df["days_to_promo"] = 999
        groups = df.groupby(group_columns) if group_columns else [(None, df)]
        for _, group in groups:
            dates = group[self.date_column].where(group[promo_column].eq(1))
            df.loc[group.index, "days_since_promo"] = (
                group[self.date_column] - dates.ffill()
            ).dt.days.fillna(999)
            df.loc[group.index, "days_to_promo"] = (
                dates.bfill() - group[self.date_column]
            ).dt.days.fillna(999)

        self.df = df
        return df

    def create_all_features(
        self,
        include_temporal: bool = True,
        include_holidays: bool = True,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_diff: bool = True,
        include_promo: bool = False,
        target_column: str = "sales",
        group_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create all features at once.

        Parameters
        ----------
        include_temporal : bool, default=True
            Whether to include temporal features.
        include_holidays : bool, default=True
            Whether to include holiday features.
        include_lags : bool, default=True
            Whether to include lag features.
        include_rolling : bool, default=True
            Whether to include rolling features.
        include_diff : bool, default=True
            Whether to include difference features.
        include_promo : bool, default=True
            Whether to include promotion features.
        target_column : str, default='sales'
            Target column for lag/rolling features.
        group_columns : list, optional
            Columns to group by for lag/rolling features.

        Returns
        -------
        pd.DataFrame
            Dataframe with all requested features.
        """
        if include_temporal:
            self.add_temporal_features()

        if include_holidays:
            self.add_holiday_features()

        if include_lags:
            self.add_lag_features(target_column=target_column, group_columns=group_columns)

        if include_rolling:
            self.add_rolling_features(target_column=target_column, group_columns=group_columns)

        if include_diff:
            self.add_diff_features(target_column=target_column, group_columns=group_columns)

        if include_promo and "is_promo" in self.df.columns:
            self.add_promotion_features(group_columns=group_columns)

        return self.df
