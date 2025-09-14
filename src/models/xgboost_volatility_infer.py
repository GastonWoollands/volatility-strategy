import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.xgboost_model import make_features, make_target
from data.loader import fetch_ticker_data


def infer_with_mlflow_registry(
    ticker,
    n_ahead=5,
    start_date="2000-01-01",
    stage="Staging",
    plot=False,
    interval="1d",
    auto_adjust=True,
    log_returns=True,
    earnings=False,
    output_csv=None
):
    end_date = datetime.today().strftime("%Y-%m-%d")

    # Load model from MLflow Model Registry
    model_name = f"{ticker}_{n_ahead}"
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    input_schema = model.metadata.get_input_schema()
    feature_names = [col.name for col in input_schema.inputs]

    # Detect if VIX is needed
    needs_vix = any('vix' in name for name in feature_names)
    # Detect which lags are needed
    lag_numbers = set()
    for name in feature_names:
        if 'lag_' in name:
            try:
                lag_num = int(name.split('lag_')[-1].split('_')[0])
                lag_numbers.add(lag_num)
            except Exception:
                pass
    lags = sorted(lag_numbers) if lag_numbers else [1, 2, 3, 5, 10, 21]

    # Fetch and prepare data
    df = fetch_ticker_data(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        log_returns=log_returns,
        earnings=earnings,
        vix=needs_vix
    )
    df = make_features(df, lags=lags, vix=needs_vix)
    df = make_target(df, n_ahead=n_ahead)
    df = df.dropna()

    # Use only the features required by the model
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in dataframe: {missing_features}")
    X = df[feature_names].iloc[-1]
    y_true = df.get(f'target_vol_{n_ahead}', None)

    # Predict
    y_pred = model.predict(X)
    df_result = df.copy()
    df_result['predicted_vol'] = y_pred

    # Output
    if output_csv:
        df_result[['predicted_vol']].to_csv(output_csv)
        print(f"Predictions saved to {output_csv}")
    else:
        print(df_result[['predicted_vol']].tail(10))

    # Plot
    if plot:
        plt.figure(figsize=(12, 6))
        if y_true is not None:
            plt.plot(df_result.index, y_true, label='Actual Volatility', color='gray')
        plt.plot(df_result.index, y_pred, label='Predicted Volatility', color='red')
        plt.title(f"Predicted Volatility for {ticker} ({n_ahead}-period ahead)")
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return df_result


def main():
    parser = argparse.ArgumentParser(description="Infer future volatility using an MLflow-registered XGBoost model.")
    parser.add_argument('--ticker', '-t', type=str, required=True, help='Ticker symbol')
    parser.add_argument('--n_ahead', '-f', type=int, default=5, help='Periods ahead to forecast')
    parser.add_argument('--start', '-s', type=str, default="2000-01-01", help='Start date')
    parser.add_argument('--end', '-e', type=str, default=datetime.today().strftime("%Y-%m-%d"), help='End date')
    parser.add_argument('--stage', type=str, default="Staging", help='MLflow model stage (e.g., Staging, Production)')
    parser.add_argument('--plot', action='store_true', help='Plot predictions')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV file for predictions')
    parser.add_argument('--earnings', action='store_true', help='Include earnings feature')
    args = parser.parse_args()

    infer_with_mlflow_registry(
        ticker=args.ticker,
        n_ahead=args.n_ahead,
        start_date=args.start,
        end_date=args.end,
        stage=args.stage,
        plot=args.plot,
        earnings=args.earnings,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main() 