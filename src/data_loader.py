from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"store", "date", "sales"}


def load_sales_data(path: str | Path) -> pd.DataFrame:
    """
    Load and validate raw sales data.

    Parameters
    ----------
    path : str or Path
        Path to CSV file containing sales data.

    Returns
    -------
    pd.DataFrame
        Validated sales dataframe with parsed dates.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at {path}")

    df = pd.read_csv(path)

    missing_coumns = REQUIRED_COLUMNS - set(df.columns)

    if missing_coumns:
        raise ValueError(f"Missing columns: {missing_coumns}")

    df = df.copy()

    df.date = pd.to_datetime(df["date"], errors="raise")

    return df.sort_values(["store", "date"]).reset_index(drop=True)
