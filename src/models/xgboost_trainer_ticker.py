import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import uuid
import logging
from datetime import datetime
from itertools import product
from xgboost_model import train_validate_xgboost_with_mlflow

# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", "-t", type=str, required=True, help="Ticker to train on")

# ------------------------------------------------------------------------------

args = parser.parse_args()
ticker = args.ticker

# Static configurations
save_plot = True
earnings = False
objective = ['reg:squarederror', 'reg:pseudohubererror']
early_stopping_rounds = [None, 10, 20]
end = datetime.now().strftime("%Y-%m-%d")
start_dates = ["2000-01-01", "2010-01-01", "2016-01-01", "2021-01-01"]
vix_features = [True, False]
n_ahead_max = 20
n_ahead_step = 5
n_ahead_values = [1] + list(range(n_ahead_step, n_ahead_max + 1, n_ahead_step))

# ------------------------------------------------------------------------------

# Parameter combinations
param_combinations = product(n_ahead_values, start_dates, vix_features, objective, early_stopping_rounds)

for n_ahead, start, vix, objective, early_stopping_rounds in param_combinations:
    model_id = datetime.now().strftime("%Y%m%d") + "_" + str(uuid.uuid4())[:8]
    mlflow_experiment = f"{ticker}_{n_ahead}"

    logger.info(
        f"Training XGBoost Ticker: {ticker} | N-Ahead: {n_ahead} | VIX: {vix} | "
        f"Save plot: {save_plot} | MLflow experiment: {mlflow_experiment} | Start: {start} | End: {end}"
    )

    model, (X_test, y_test, y_pred), metrics = train_validate_xgboost_with_mlflow(
        ticker=ticker,
        n_ahead=n_ahead,
        vix=vix,
        save_plot=save_plot,
        mlflow_experiment=mlflow_experiment,
        start=start,
        end=end,
        earnings=earnings,
        model_id=model_id,
        objective=objective,
        early_stopping_rounds=early_stopping_rounds,
    )
    logger.info(f"Metrics: {metrics}")