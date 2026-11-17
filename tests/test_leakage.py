import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer
from src.ml_models import TimeSeriesMLForecaster
from scripts.run_pipeline import evaluate, demo_data


def features(frame):
    engineer = FeatureEngineer(frame)
    engineer.add_lag_features(lags=[1], group_columns=["store_id"])
    engineer.add_rolling_features(windows=[2], group_columns=["store_id"])
    return engineer.add_diff_features(periods=[1], group_columns=["store_id"])


def test_current_and_future_targets_do_not_change_features():
    frame = demo_data().iloc[:180].copy()
    before = features(frame)
    frame.loc[frame.index >= 100, "sales"] = 100000
    after = features(frame)
    cols = [c for c in before if c.startswith("sales_")]
    pd.testing.assert_frame_equal(before.loc[:100, cols], after.loc[:100, cols])


def test_group_isolation_and_sorting():
    frame = demo_data().sample(frac=1, random_state=4)
    before = features(frame)
    frame.loc[frame.store_id.eq(2), "sales"] = 99999
    after = features(frame)
    pd.testing.assert_frame_equal(
        before[before.store_id.eq(1)], after[after.store_id.eq(1)]
    )


def test_recursive_predictions_ignore_test_targets_and_are_repeatable():
    frame = demo_data()
    first, report = evaluate(frame, 7)
    frame.loc[frame.date >= frame.date.max() - pd.Timedelta(days=6), "sales"] = 99999
    changed, _ = evaluate(frame, 7)
    np.testing.assert_allclose(first.predicted, changed.predicted)
    keys = first.groupby("model").apply(
        lambda g: list(zip(g.store_id, g.date)), include_groups=False
    )
    assert all(v == keys.iloc[0] for v in keys)
    assert report["horizon"] == 7


def test_requires_fit_and_enough_history():
    model = TimeSeriesMLForecaster(model_type="random_forest")
    with pytest.raises(ValueError, match="Fit"):
        model.predict(2)
    with pytest.raises(ValueError):
        model.fit(pd.Series([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "kwargs", [{"lags": [0]}, {"lags": [-1]}, {"rolling_windows": [1]}]
)
def test_invalid_history_windows(kwargs):
    with pytest.raises(ValueError):
        TimeSeriesMLForecaster(**kwargs)
