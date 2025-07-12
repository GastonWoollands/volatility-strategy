import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import uuid
import logging
from datetime import datetime
from xgboost_model import train_validate_xgboost_with_mlflow

# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

model_id = datetime.now().strftime("%Y%m%d") + "_" + str(uuid.uuid4())[:8]

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", "-t", type=str, required=True, help="Ticker to train on")
parser.add_argument("--n_ahead", "-f", type=int, required=True, help="Number of periods ahead to forecast")
parser.add_argument("--vix", "-v", action="store_true", help="Whether to include VIX feature")
parser.add_argument("--save-plot", action="store_true", help="Whether to save plots")
parser.add_argument("--mlflow_experiment", "-mflx", type=str, required=False, help="MLflow experiment name")
parser.add_argument("--start", "-s", type=str, required=False, default="2000-01-01", help="Start date")
parser.add_argument("--end", "-e", type=str, required=False, default=datetime.now().strftime("%Y-%m-%d"), help="End date")
parser.add_argument("--earnings", "-earn", required=False, default=False, action="store_true", help="Whether to include earnings feature")
parser.add_argument("--info", "-i", type=str, required=False, default=None, help="Note to add to the model name")
parser.add_argument("--model_id", "-mid", type=str, required=False, default=model_id, help="Model ID")

# ------------------------------------------------------------------------------

args = parser.parse_args()

ticker = args.ticker
n_ahead = args.n_ahead
vix = True if args.vix else False
save_plot = args.save_plot
note = f"_{args.info}" if args.info else ""
mlflow_experiment = args.mlflow_experiment if args.mlflow_experiment else args.ticker + "_" + str(n_ahead) + note
start = args.start
end = args.end
earnings = args.earnings

# ------------------------------------------------------------------------------

logger.info(f"Training XGBoost Ticker: {ticker} | N-Ahead: {n_ahead} | VIX: {vix} | Save plot: {save_plot} | MLflow experiment: {mlflow_experiment}")

# ------------------------------------------------------------------------------

model, (X_test, y_test, y_pred), metrics = train_validate_xgboost_with_mlflow(
    ticker=ticker,
    n_ahead=n_ahead,
    vix=vix,
    save_plot=save_plot,
    mlflow_experiment=mlflow_experiment,
    start=start,
    end=end,
    earnings=earnings,
    model_id=model_id
)
logger.info(f"Metrics: {metrics}")
