# Fin-Volatility

A machine learning project for predicting financial volatility using XGBoost and GARCH models.

## Overview

This project trains and evaluates volatility prediction models for financial tickers using:
- **XGBoost**: Machine learning approach with feature engineering
- **GARCH**: Traditional econometric models (GARCH, EGARCH, GJR-GARCH, APARCH)

## Features

- Historical data fetching via `yfinance`
- Feature engineering with lagged returns and rolling statistics
- Model training with hyperparameter optimization
- MLflow integration for experiment tracking
- GPU support for XGBoost models
- Portfolio optimization capabilities

## Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Start MLflow server** (optional):
   ```bash
   ./run_mlflow.sh
   ```

## Usage

### Training Models

```python
from src.models.xgboost_model import train_xgboost_volatility_model
from src.models.garch_model import train_validate_garch_with_plots

# XGBoost model
model, test_set, predictions, metrics = train_xgboost_volatility_model(
    ticker="AAPL",
    n_ahead=5,
    test_size=0.2
)

# GARCH model
best_model, results, mse, mae, plot_path, resid_path = train_validate_garch_with_plots(
    data=df,
    ticker="AAPL",
    p_max=3,
    q_max=3,
    means=['Zero', 'AR'],
    distributions=['normal', 't'],
    vols=['GARCH', 'EGARCH']
)
```

### Data Loading

```python
from src.data.loader import fetch_ticker_data

df = fetch_ticker_data(
    ticker="AAPL",
    start="2020-01-01",
    vix=True,  # Include VIX data
    earnings=True  # Include earnings data
)
```

## Project Structure

```
├── src/
│   ├── models/          # Model implementations
│   │   ├── xgboost_model.py
│   │   ├── garch_model.py
│   │   └── xgboost_volatility_infer.py
│   └── data/
│       └── loader.py    # Data fetching utilities
├── notebooks/           # Jupyter notebooks for testing
├── tests/               # Unit tests
└── mlruns/              # MLflow experiment tracking
```

## Dependencies

- See pyproject.toml file