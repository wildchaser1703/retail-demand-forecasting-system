.PHONY: help install install-dev clean test lint format data train api docker

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

clean:  ## Clean generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage htmlcov
	rm -rf mlruns

test:  ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term -v

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting checks
	flake8 src/ tests/ scripts/
	mypy src/

format:  ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format-check:  ## Check code formatting
	black --check src/ tests/ scripts/
	isort --check src/ tests/ scripts/

data:  ## Generate synthetic data
	python scripts/generate_data.py

train:  ## Train all models
	python scripts/run_pipeline.py

train-baselines:  ## Train only baseline models
	python scripts/run_pipeline.py --skip-ml

train-ml:  ## Train only ML models
	python scripts/run_pipeline.py --skip-baselines

mlflow:  ## Start MLflow UI
	mlflow ui --backend-store-uri ./mlruns

docker-build:  ## Build Docker image
	docker build -t retail-forecasting:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 retail-forecasting:latest

docker-compose-up:  ## Start services with docker-compose
	docker-compose up -d

docker-compose-down:  ## Stop services
	docker-compose down

all: clean install data train test  ## Run complete pipeline
