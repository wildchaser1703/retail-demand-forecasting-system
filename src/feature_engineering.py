"""
Feature engineering for retail demand forecasting.
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


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

        # Define major US retail holidays
        holidays = {
            "new_year": (1, 1),
            "valentine": (2, 14),
            "easter": (4, 15),  # Approximate
            "memorial_day": (5, 27),  # Last Monday of May (approximate)
            "independence_day": (7, 4),
            "labor_day": (9, 2),  # First Monday of September (approximate)
            "halloween": (10, 31),
            "thanksgiving": (11, 28),  # Fourth Thursday of November (approximate)
            "black_friday": (11, 29),  # Day after Thanksgiving (approximate)
            "christmas": (12, 25),
        }

        # Initialize holiday columns
        df["is_holiday"] = 0
        df["days_to_holiday"] = 365
        df["days_from_holiday"] = 365

        for holiday_name, (month, day) in holidays.items():
            # Mark holiday
            is_holiday = (df[self.date_column].dt.month == month) & (
                df[self.date_column].dt.day == day
            )
            df.loc[is_holiday, "is_holiday"] = 1

            # Calculate days to/from holiday for each year
            for year in df[self.date_column].dt.year.unique():
                try:
                    holiday_date = pd.Timestamp(year=year, month=month, day=day)
                    year_mask = df[self.date_column].dt.year == year

                    days_diff = (df.loc[year_mask, self.date_column] - holiday_date).dt.days

                    # Days to holiday (negative if before, 0 on holiday, positive if after)
                    df.loc[year_mask, "days_to_holiday"] = np.minimum(
                        df.loc[year_mask, "days_to_holiday"], np.abs(days_diff)
                    )

                    # Days from last holiday
                    df.loc[year_mask, "days_from_holiday"] = np.minimum(
                        df.loc[year_mask, "days_from_holiday"], np.abs(days_diff)
                    )
                except ValueError:
                    # Skip invalid dates (e.g., Feb 30)
                    continue

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

        df = self.df.copy()

        if group_columns:
            for lag in lags:
                df[f"{target_column}_lag_{lag}"] = df.groupby(group_columns)[
                    target_column
                ].shift(lag)
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

        df = self.df.copy()

        for window in windows:
            for stat in stats:
                col_name = f"{target_column}_rolling_{window}_{stat}"

                if group_columns:
                    if stat == "mean":
                        df[col_name] = (
                            df.groupby(group_columns)[target_column]
                            .rolling(window=window, min_periods=1)
                            .mean()
                            .reset_index(level=list(range(len(group_columns))), drop=True)
                        )
                    elif stat == "std":
                        df[col_name] = (
                            df.groupby(group_columns)[target_column]
                            .rolling(window=window, min_periods=1)
                            .std()
                            .reset_index(level=list(range(len(group_columns))), drop=True)
                        )
                    elif stat == "min":
                        df[col_name] = (
                            df.groupby(group_columns)[target_column]
                            .rolling(window=window, min_periods=1)
                            .min()
                            .reset_index(level=list(range(len(group_columns))), drop=True)
                        )
                    elif stat == "max":
                        df[col_name] = (
                            df.groupby(group_columns)[target_column]
                            .rolling(window=window, min_periods=1)
                            .max()
                            .reset_index(level=list(range(len(group_columns))), drop=True)
                        )
                else:
                    if stat == "mean":
                        df[col_name] = df[target_column].rolling(window=window, min_periods=1).mean()
                    elif stat == "std":
                        df[col_name] = df[target_column].rolling(window=window, min_periods=1).std()
                    elif stat == "min":
                        df[col_name] = df[target_column].rolling(window=window, min_periods=1).min()
                    elif stat == "max":
                        df[col_name] = df[target_column].rolling(window=window, min_periods=1).max()

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

        df = self.df.copy()

        for period in periods:
            col_name = f"{target_column}_diff_{period}"

            if group_columns:
                df[col_name] = df.groupby(group_columns)[target_column].diff(period)
            else:
                df[col_name] = df[target_column].diff(period)

        self.df = df
        return df

    def add_promotion_features(self, promo_column: str = "is_promo") -> pd.DataFrame:
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

        df = self.df.copy()

        # Days since last promotion
        df["days_since_promo"] = 0
        promo_dates = df[df[promo_column] == 1].index

        for idx in df.index:
            prev_promos = promo_dates[promo_dates < idx]
            if len(prev_promos) > 0:
                df.loc[idx, "days_since_promo"] = idx - prev_promos[-1]
            else:
                df.loc[idx, "days_since_promo"] = 999  # No previous promo

        # Days to next promotion
        df["days_to_promo"] = 0
        for idx in df.index:
            next_promos = promo_dates[promo_dates > idx]
            if len(next_promos) > 0:
                df.loc[idx, "days_to_promo"] = next_promos[0] - idx
            else:
                df.loc[idx, "days_to_promo"] = 999  # No upcoming promo

        self.df = df
        return df

    def create_all_features(
        self,
        include_temporal: bool = True,
        include_holidays: bool = True,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_diff: bool = True,
        include_promo: bool = True,
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
            self.add_promotion_features()

        return self.df
