"""
Configuration management for the retail demand forecasting system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


@dataclass
class PathConfig:
    """File and directory paths."""

    # Base directories
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "./models")))
    logs_dir: Path = field(default_factory=lambda: Path("./logs"))

    # Data paths
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    sample_data_dir: Path = field(init=False)

    def __post_init__(self):
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.sample_data_dir = self.data_dir / "sample"

        # Create directories if they don't exist
        for dir_path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data generation and processing configuration."""

    # Data generation parameters
    num_stores: int = 10
    num_products: int = 50
    num_categories: int = 5
    start_date: str = "2021-01-01"
    end_date: str = "2023-12-31"

    # Time series parameters
    weekly_seasonality: bool = True
    monthly_seasonality: bool = True
    yearly_seasonality: bool = True
    trend_strength: float = 0.1
    noise_level: float = 0.15

    # Promotion parameters
    promo_probability: float = 0.2
    promo_impact_range: tuple = (1.1, 1.5)

    # Validation split
    validation_weeks: int = int(os.getenv("VALIDATION_WEEKS", "8"))
    test_weeks: int = int(os.getenv("TEST_WEEKS", "4"))

    # Random seed
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Lag features
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 28])

    # Rolling window features
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])

    # Temporal features
    include_temporal: bool = True
    include_holidays: bool = True

    # Store and product features
    include_store_features: bool = True
    include_product_features: bool = True


@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration."""

    # Forecast horizon
    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "42"))  # 6 weeks

    # Baseline models
    seasonal_period: int = 7  # Weekly seasonality

    # VAR model
    var_max_lags: int = 14
    var_ic: str = "aic"  # Information criterion for lag selection

    # ARIMA model
    arima_seasonal_period: int = 7
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    arima_max_P: int = 2
    arima_max_D: int = 1
    arima_max_Q: int = 2

    # Prophet model
    prophet_seasonality_mode: str = "multiplicative"
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0

    # XGBoost model
    xgb_params: Dict = field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
    )

    # LightGBM model
    lgb_params: Dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "mae",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Ensemble
    ensemble_weights: Optional[Dict[str, float]] = None


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""

    # Metrics to compute
    metrics: List[str] = field(
        default_factory=lambda: ["mae", "rmse", "mape", "smape", "mase", "r2"]
    )

    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "expanding"  # 'expanding' or 'sliding'

    # Prediction intervals
    confidence_level: float = 0.95


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""

    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "retail_demand_forecasting")
    artifact_location: Optional[str] = None
    log_models: bool = True


@dataclass
class APIConfig:
    """API configuration."""

    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    workers: int = 1
    log_level: str = os.getenv("LOG_LEVEL", "info")


@dataclass
class Config:
    """Main configuration object."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global config instance
config = Config()
