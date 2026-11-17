"""Compare fixed-origin forecasts on identical store/day observations."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import BaselineForecaster
from src.ml_models import TimeSeriesMLForecaster
from src.evaluation import compute_metrics


def evaluate(df, horizon=14):
    required = {"date", "store_id", "sales"}
    if not required <= set(df):
        raise ValueError(f"Required columns: {sorted(required)}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df[list(required)].isna().any().any() or not np.isfinite(df.sales).all():
        raise ValueError("Input contains missing or nonfinite observations")
    daily = df.groupby(["store_id", "date"], as_index=False).sales.sum()
    cutoff = daily.date.max() - pd.Timedelta(days=horizon - 1)
    records = []
    for store, rows in daily.groupby("store_id"):
        series = rows.set_index("date").sales.sort_index()
        if not series.index.equals(pd.date_range(series.index.min(), series.index.max(), freq="D")):
            raise ValueError(
                f"Store {store} has missing dates; fill these explicitly before evaluation"
            )
        train, test = series[series.index < cutoff], series[series.index >= cutoff]
        if len(train) < 60 or len(test) != horizon:
            raise ValueError(
                f"Store {store} needs 60 training days and the complete common test horizon"
            )
        predictions = {}
        for method in ["naive", "seasonal_naive", "moving_average"]:
            predictions[method] = BaselineForecaster(
                method=method, season_length=7, window=7
            ).fit_predict(train.values, horizon)
        model = TimeSeriesMLForecaster(
            model_type="random_forest", n_estimators=40, max_depth=8, n_jobs=1
        )
        predictions["random_forest"] = model.fit(train).predict(horizon)
        for name, values in predictions.items():
            for date, actual, predicted in zip(test.index, test.values, values):
                records.append(
                    {
                        "model": name,
                        "store_id": store,
                        "date": date,
                        "actual": float(actual),
                        "predicted": float(predicted),
                    }
                )
    if not records:
        raise ValueError("No observations to evaluate")
    predictions = pd.DataFrame(records)
    metrics = {
        name: compute_metrics(group.actual.values, group.predicted.values, metrics=["mae", "rmse"])
        for name, group in predictions.groupby("model")
    }
    return predictions, {
        "protocol": "fixed-origin recursive forecast; store/day grain; no test targets used as inputs",
        "cutoff": str(cutoff.date()),
        "horizon": horizon,
        "metrics": metrics,
    }


def demo_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "store_id": store,
                    "sales": 100
                    + store * 30
                    + 15 * np.sin(np.arange(180) * 2 * np.pi / 7)
                    + rng.normal(0, 3, 180),
                }
            )
            for store in [1, 2]
        ],
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-path",
        type=Path,
        help="CSV with date, store_id, sales; product rows are summed per store/day",
    )
    source.add_argument(
        "--demo", action="store_true", help="Seeded synthetic data; smoke evidence only"
    )
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--output", type=Path, default=Path("reports/latest"))
    args = parser.parse_args()
    predictions, report = evaluate(
        demo_data() if args.demo else pd.read_csv(args.data_path), args.horizon
    )
    report["data"] = "synthetic smoke fixture" if args.demo else str(args.data_path)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
