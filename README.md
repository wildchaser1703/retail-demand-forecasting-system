# Retail Demand Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An industry-standard machine learning system for retail demand forecasting, featuring multiple time series models, comprehensive MLOps integration, and production-ready deployment infrastructure.

## 🎯 Project Overview

This project demonstrates best practices in building production-ready ML systems for time series forecasting. It addresses the business challenge of predicting retail sales across multiple stores and products, enabling better inventory management and promotional planning.

### Business Context

Retail teams need reliable short-term sales forecasts to:
- Optimize inventory levels and reduce stockouts
- Plan promotions and marketing campaigns
- Allocate resources efficiently across stores
- Improve supply chain operations

### Key Features

- **📊 Realistic Data Generation**: Synthetic retail data with seasonality, trends, promotions, and holiday effects
- **🤖 Multiple Models**: 6 different forecasting approaches from simple baselines to advanced ML
- **📈 Comprehensive Evaluation**: Multiple metrics (MAE, RMSE, MAPE, SMAPE, MASE, R²) with time series cross-validation
- **🔄 MLOps Integration**: MLflow for experiment tracking, model versioning, and registry
- **🧪 Production-Ready**: Complete test suite, CI/CD, Docker containerization
- **📉 Rich Visualizations**: Interactive dashboards and detailed forecast analysis
- **⚙️ Configurable**: Centralized configuration management with environment variables

## 🏗️ Architecture

```
├── data/                  # Data storage
│   ├── raw/              # Raw generated data
│   ├── processed/        # Processed features
│   └── sample/           # Sample data
├── src/                   # Source code
│   ├── config.py         # Configuration management
│   ├── data_loader.py    # Data loading utilities
│   ├── data_validation.py # Data quality checks
│   ├── preprocessing.py  # Data preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── baselines.py      # Baseline models
│   ├── var_model.py      # VAR model
│   ├── arima_models.py   # ARIMA/SARIMA
│   ├── prophet_model.py  # Facebook Prophet
│   ├── ml_models.py      # XGBoost, LightGBM, RF
│   ├── ensemble.py       # Ensemble methods
│   ├── evaluation.py     # Metrics and evaluation
│   ├── forecasting.py    # Unified forecasting interface
│   ├── visualization.py  # Plotting utilities
│   └── mlflow_utils.py   # MLflow integration
├── scripts/              # Executable scripts
│   ├── generate_data.py  # Data generation
│   └── run_pipeline.py   # Main training pipeline
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for EDA
├── models/               # Saved models
└── mlruns/               # MLflow tracking data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/wildchaser1703/retail-demand-forecasting-system.git
cd retail-demand-forecasting-system-1
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
make install
# or
pip install -r requirements.txt
pip install -e .
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Generate Data

```bash
make data
# or
python scripts/generate_data.py
```

This generates:
- 10 stores with different characteristics
- 50 products across 5 categories
- 3 years of daily sales data (2021-2023)
- Realistic seasonality, trends, and promotions

### Train Models

```bash
make train
# or
python scripts/run_pipeline.py
```

Options:
```bash
# Train only baseline models
python scripts/run_pipeline.py --skip-ml

# Train only ML models
python scripts/run_pipeline.py --skip-baselines

# Use MLflow tracking
python scripts/run_pipeline.py --use-mlflow
```

### View Results

```bash
# Start MLflow UI
make mlflow
# Visit http://localhost:5000
```

## 📊 Models

### 1. Baseline Models
- **Naive**: Repeats last observed value
- **Seasonal Naive**: Repeats values from previous season
- **Moving Average**: Average of recent values
- **Exponential Smoothing**: Weighted average with exponential decay

### 2. Statistical Models
- **VAR (Vector AutoRegression)**: Multivariate time series with automatic lag selection
- **ARIMA/SARIMA**: Auto ARIMA with seasonal components
- **Prophet**: Facebook's forecasting tool with holiday effects

### 3. Machine Learning Models
- **XGBoost**: Gradient boosting with time series features
- **LightGBM**: Fast gradient boosting
- **Random Forest**: Ensemble of decision trees

### 4. Ensemble Methods
- **Simple Average**: Mean of all model predictions
- **Weighted Average**: Weighted combination based on validation performance
- **Stacking**: Meta-learner combining base models

## 📈 Results

Typical performance on test set (6 weeks ahead):

| Model | MAE | RMSE | MAPE | Training Time |
|-------|-----|------|------|---------------|
| Naive | 45.2 | 62.3 | 12.5% | < 1s |
| Seasonal Naive | 38.7 | 54.1 | 10.8% | < 1s |
| Prophet | 32.4 | 45.6 | 8.9% | ~30s |
| XGBoost | 28.1 | 39.2 | 7.6% | ~15s |
| LightGBM | 27.5 | 38.7 | 7.4% | ~10s |
| Ensemble | 26.8 | 37.9 | 7.2% | N/A |

*Note: Results vary based on data characteristics and hyperparameters*

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run tests without coverage (faster)
make test-fast

# Run specific test file
pytest tests/test_features.py -v
```

## 🎨 Code Quality

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint
```

## 📦 Deployment

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Using docker-compose
make docker-compose-up
```

### Production Considerations

1. **Data Pipeline**: Set up automated data ingestion
2. **Model Retraining**: Schedule periodic retraining
3. **Monitoring**: Track prediction accuracy and data drift
4. **Scaling**: Use distributed computing for large datasets
5. **API**: Deploy FastAPI service for real-time predictions

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Model Documentation](docs/MODELS.md) - Detailed model descriptions
- [API Documentation](docs/API.md) - API endpoints and usage

## 🔬 Notebooks

Explore the `notebooks/` directory for:
- `01_eda.ipynb`: Exploratory data analysis
- `02_feature_engineering.ipynb`: Feature creation and analysis
- `03_model_training.ipynb`: Model training and comparison
- `04_results_analysis.ipynb`: Results visualization and insights

## 🛠️ Development

### Project Structure

- **src/**: Core library code
- **scripts/**: Executable scripts
- **tests/**: Unit and integration tests
- **notebooks/**: Jupyter notebooks
- **data/**: Data storage
- **models/**: Saved models
- **mlruns/**: MLflow experiments

### Adding New Models

1. Create model class in appropriate module
2. Implement `fit()` and `predict()` methods
3. Add to `run_pipeline.py`
4. Write unit tests
5. Update documentation

## 📊 MLOps Features

- **Experiment Tracking**: All runs logged to MLflow
- **Model Registry**: Version control for models
- **Hyperparameter Tuning**: Optuna integration (optional)
- **Feature Store**: Centralized feature management
- **Model Monitoring**: Performance tracking over time

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Toshali Mohapatra**
- GitHub: [@wildchaser1703](https://github.com/wildchaser1703)

## 🙏 Acknowledgments

- Facebook Prophet for time series forecasting
- MLflow for experiment tracking
- The open-source community for amazing tools

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review notebooks for examples

---

**Note**: This is a portfolio project demonstrating industry best practices in ML engineering. The data is synthetically generated for demonstration purposes.