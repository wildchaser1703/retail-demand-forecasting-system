"""
Data validation and quality checks for retail sales data.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation results."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict


class DataValidator:
    """Validator for retail sales data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate.
        """
        self.df = df
        self.errors = []
        self.warnings = []
        self.stats = {}

    def validate_schema(self, required_columns: Optional[List[str]] = None) -> bool:
        """Validate that required columns exist."""
        if required_columns is None:
            required_columns = ["date", "store_id", "product_id", "sales"]

        missing_cols = set(required_columns) - set(self.df.columns)
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return False
        return True

    def validate_data_types(self) -> bool:
        """Validate data types of columns."""
        valid = True

        # Check date column
        if "date" in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
                self.errors.append("'date' column must be datetime type")
                valid = False

        # Check numeric columns
        numeric_cols = ["store_id", "product_id", "sales"]
        for col in numeric_cols:
            if col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    self.errors.append(f"'{col}' column must be numeric type")
                    valid = False

        return valid

    def validate_missing_values(self, max_missing_pct: float = 0.05) -> bool:
        """Check for missing values."""
        valid = True

        for col in self.df.columns:
            missing_pct = self.df[col].isna().mean()
            if missing_pct > 0:
                if missing_pct > max_missing_pct:
                    self.errors.append(
                        f"Column '{col}' has {missing_pct:.1%} missing values "
                        f"(threshold: {max_missing_pct:.1%})"
                    )
                    valid = False
                else:
                    self.warnings.append(f"Column '{col}' has {missing_pct:.1%} missing values")

        return valid

    def validate_value_ranges(self) -> bool:
        """Validate that values are within expected ranges."""
        valid = True

        # Sales should be non-negative
        if "sales" in self.df.columns:
            if (self.df["sales"] < 0).any():
                self.errors.append("Sales column contains negative values")
                valid = False

            # Check for unrealistic values (e.g., too high)
            q99 = self.df["sales"].quantile(0.99)
            extreme_values = (self.df["sales"] > q99 * 10).sum()
            if extreme_values > 0:
                self.warnings.append(
                    f"Found {extreme_values} sales values > 10x the 99th percentile"
                )

        return valid

    def validate_temporal_consistency(self) -> bool:
        """Check for temporal consistency."""
        valid = True

        if "date" not in self.df.columns:
            return valid

        # Check for duplicate dates within store-product combinations
        if all(col in self.df.columns for col in ["date", "store_id", "product_id"]):
            duplicates = self.df.duplicated(subset=["date", "store_id", "product_id"]).sum()
            if duplicates > 0:
                self.errors.append(f"Found {duplicates} duplicate date-store-product combinations")
                valid = False

        # Check for gaps in time series
        date_range = pd.date_range(start=self.df["date"].min(), end=self.df["date"].max(), freq="D")
        expected_days = len(date_range)
        actual_days = self.df["date"].nunique()

        if actual_days < expected_days:
            missing_days = expected_days - actual_days
            self.warnings.append(f"Time series has {missing_days} missing days")

        return valid

    def compute_statistics(self) -> Dict:
        """Compute summary statistics."""
        stats = {
            "num_records": len(self.df),
            "num_columns": len(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
        }

        if "date" in self.df.columns:
            stats["date_range"] = {
                "start": str(self.df["date"].min()),
                "end": str(self.df["date"].max()),
                "num_days": (self.df["date"].max() - self.df["date"].min()).days + 1,
            }

        if "store_id" in self.df.columns:
            stats["num_stores"] = self.df["store_id"].nunique()

        if "product_id" in self.df.columns:
            stats["num_products"] = self.df["product_id"].nunique()

        if "sales" in self.df.columns:
            stats["sales"] = {
                "total": float(self.df["sales"].sum()),
                "mean": float(self.df["sales"].mean()),
                "median": float(self.df["sales"].median()),
                "std": float(self.df["sales"].std()),
                "min": float(self.df["sales"].min()),
                "max": float(self.df["sales"].max()),
            }

        self.stats = stats
        return stats

    def validate_all(self) -> ValidationResult:
        """Run all validation checks."""
        self.errors = []
        self.warnings = []

        self.validate_schema()
        self.validate_data_types()
        self.validate_missing_values()
        self.validate_value_ranges()
        self.validate_temporal_consistency()
        self.compute_statistics()

        is_valid = len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid, errors=self.errors, warnings=self.warnings, stats=self.stats
        )


def detect_outliers(series: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a series.

    Parameters
    ----------
    series : pd.Series
        Input series.
    method : str, default='iqr'
        Method to use: 'iqr' (interquartile range) or 'zscore'.
    threshold : float, default=3.0
        Threshold for outlier detection.

    Returns
    -------
    pd.Series
        Boolean series indicating outliers.
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")
