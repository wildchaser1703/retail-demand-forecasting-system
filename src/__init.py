"""
Retail Demand Forecasting System

An industry-standard ML system for forecasting retail demand using multiple
time series models including baselines, statistical models, and ML models.
"""

__version__ = "1.0.0"
__author__ = "Toshali Mohapatra"

from src.config import config
from src.data_loader import load_sales_data, load_metadata, split_train_test
from src.baselines import BaselineForecaster
from src.evaluation import ModelEvaluator, compute_metrics

__all__ = [
    "config",
    "load_sales_data",
    "load_metadata",
    "split_train_test",
    "BaselineForecaster",
    "ModelEvaluator",
    "compute_metrics",
]
