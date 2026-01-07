"""
Unit tests for data loading module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_sales_data, split_train_test, aggregate_sales


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data."""
    data = {
        "date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "store_id": [1] * 100,
        "product_id": [1] * 100,
        "sales": np.random.randint(50, 150, 100),
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    Path(f.name).unlink()


def test_load_sales_data_csv(sample_csv_file):
    """Test loading sales data from CSV."""
    df = load_sales_data(sample_csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "date" in df.columns
    assert "store_id" in df.columns
    assert "sales" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_sales_data_missing_file():
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_sales_data("nonexistent_file.csv")


def test_load_sales_data_missing_columns(tmp_path):
    """Test loading data with missing required columns."""
    # Create CSV with missing columns
    data = {"date": ["2023-01-01"], "sales": [100]}
    df = pd.DataFrame(data)

    file_path = tmp_path / "incomplete.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_sales_data(file_path)


def test_split_train_test():
    """Test train/test splitting."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "store_id": [1] * 100,
        "sales": np.random.randint(50, 150, 100),
    }
    df = pd.DataFrame(data)

    train_df, val_df, test_df = split_train_test(df, test_weeks=1, validation_weeks=2)

    # Check that splits are non-overlapping
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()

    # Check sizes
    assert len(test_df) == 7  # 1 week
    assert len(val_df) == 14  # 2 weeks
    assert len(train_df) == 79  # Remaining


def test_aggregate_sales():
    """Test sales aggregation."""
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "store_id": [1, 1, 1],
        "product_id": [1, 2, 1],
        "sales": [100, 150, 120],
    }
    df = pd.DataFrame(data)

    agg_df = aggregate_sales(df, group_by=["date", "store_id"], agg_column="sales")

    assert len(agg_df) == 2  # 2 unique date-store combinations
    assert agg_df[agg_df["date"] == "2023-01-01"]["sales"].values[0] == 250  # 100 + 150
