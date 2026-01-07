"""
Data loading utilities for retail sales data.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def load_sales_data(
    path: str | Path, file_format: Optional[str] = None, validate: bool = True
) -> pd.DataFrame:
    """
    Load and validate sales data from CSV or Parquet.

    Parameters
    ----------
    path : str or Path
        Path to file containing sales data.
    file_format : str, optional
        File format ('csv' or 'parquet'). If None, inferred from extension.
    validate : bool, default=True
        Whether to validate required columns.

    Returns
    -------
    pd.DataFrame
        Validated sales dataframe with parsed dates.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")

    # Infer format from extension if not provided
    if file_format is None:
        file_format = path.suffix.lower().replace(".", "")

    # Load data based on format
    if file_format == "csv":
        df = pd.read_csv(path)
    elif file_format in ["parquet", "pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # Validate required columns
    if validate:
        required_columns = {"date", "store_id", "product_id", "sales"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    # Parse dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="raise")

    # Sort by store, product, and date
    sort_cols = [col for col in ["store_id", "product_id", "date"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def load_metadata(data_dir: str | Path, metadata_type: str = "stores") -> pd.DataFrame:
    """
    Load store or product metadata.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing metadata files.
    metadata_type : str
        Type of metadata to load ('stores' or 'products').

    Returns
    -------
    pd.DataFrame
        Metadata dataframe.
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"{metadata_type}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    return pd.read_csv(file_path)


def split_train_test(
    df: pd.DataFrame,
    test_weeks: int = 4,
    validation_weeks: int = 8,
    date_column: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data.
    test_weeks : int, default=4
        Number of weeks for test set.
    validation_weeks : int, default=8
        Number of weeks for validation set.
    date_column : str, default='date'
        Name of the date column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test dataframes.
    """
    df = df.sort_values(date_column).reset_index(drop=True)

    # Calculate split dates
    max_date = df[date_column].max()
    test_start = max_date - pd.Timedelta(weeks=test_weeks)
    val_start = test_start - pd.Timedelta(weeks=validation_weeks)

    # Split data
    train_df = df[df[date_column] < val_start].copy()
    val_df = df[(df[date_column] >= val_start) & (df[date_column] < test_start)].copy()
    test_df = df[df[date_column] >= test_start].copy()

    return train_df, val_df, test_df


def aggregate_sales(
    df: pd.DataFrame,
    group_by: list,
    agg_column: str = "sales",
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate sales data by specified grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Sales dataframe.
    group_by : list
        Columns to group by.
    agg_column : str, default='sales'
        Column to aggregate.
    agg_func : str, default='sum'
        Aggregation function.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe.
    """
    return df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
