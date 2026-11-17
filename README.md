# Retail Demand Forecasting

A forecasting workbench comparing naive, seasonal-naive, moving-average and random-forest forecasts on the **same store/day observations and fixed forecast origin**.

## Run (Python 3.11 or 3.12)

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e .
python -m pytest -q
python scripts/run_pipeline.py --demo
python scripts/run_pipeline.py --data-path data/raw/sales_data.csv --horizon 14
```

The CSV must contain `date`, `store_id`, and `sales`; product-level rows are summed per store/day. Each store needs a complete daily series, at least 60 training days and the common held-out horizon. Missing dates are rejected, not silently filled. Outputs are `reports/latest/predictions.csv` and `metrics.json`, including the cutoff and evaluation protocol.

## What is evaluated

Training ends before a common cutoff. Every model forecasts the entire held-out horizon without reading any held-out targets. The random forest recursively feeds its own predictions back into lag and rolling features. MAE/RMSE compare exactly the same store/date keys for every model. Changing any test target must not change predictions; a regression test enforces this. Current/future targets also cannot change the current row's lag, rolling or difference features.

The seeded `--demo` data is **synthetic smoke evidence**, not retail accuracy evidence. Existing historical results are not a verified benchmark for the corrected pipeline. Re-evaluate real data before claiming performance.

## Deliberately small scope

The maintained entry point uses NumPy, pandas and scikit-learn. Removed the mismatched daily-total/store-level ensemble comparison, unused tracking flag, and incompatible generic pipeline adapter. No MLflow, Prophet, API server, dashboard, GPU, or distributed scheduler is needed for the core workflow. Older statistical/boosting/visualization modules and notebooks remain as experimental references; they are outside the supported quickstart and need their own optional libraries. Do not interpret their historical configuration or deployment documents as supported production infrastructure.

The shared feature helper uses an actual US federal holiday calendar. Promotion features are opt-in and require a genuinely known future promotion schedule. Do not pass realized future promotions into a forecasting experiment.

## CI and tested dependencies

`requirements.lock.txt` records the isolated Linux/Python 3.12 test environment. CI installs that lock, runs `pip check`, and executes the offline suite on pushes and pull requests; it does not deploy. For exact replay, use the lock in place of `requirements.txt` (Depth still needs the CPU Torch install first), then `pip install --no-deps -e .` for packaged repos. No credentials are required by CI.
