import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import mlflow
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.loader import fetch_ticker_data
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import uuid
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# ------------------------------------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

def realized_volatility(series, window):
    """
    Compute realized volatility (standard deviation of log returns) over a rolling window.
    """
    return series.rolling(window).std()

# ------------------------------------------------------------------------------

def make_features(df, lags=[1, 2, 3, 5, 10, 21], vix: bool = False):
    """
    Create lagged features and rolling statistics for XGBoost.
    """
    for lag in lags:
        if lag == 1:
            df[f'log_return_lag_{lag}'] = df['log_returns'].shift(lag)
            df[f'mean_lag_{lag}'] = df['log_returns'].rolling(lag).mean().shift(1)
        else:
            df[f'log_return_lag_{lag}'] = df['log_returns'].shift(lag)
            df[f'vol_lag_{lag}'] = df['log_returns'].rolling(lag).std().shift(1)
            df[f'mean_lag_{lag}'] = df['log_returns'].rolling(lag).mean().shift(1)
    
    if vix:
        df['vix_lag_1'] = df['vix'].shift(1)

    return df

# ------------------------------------------------------------------------------

def make_target(df, n_ahead=5):
    """
    Target: realized volatility over the next n periods (future rolling std).
    """
    if n_ahead == 1:
        df[f'target_vol_{n_ahead}'] = df['log_returns'].shift(-1).abs()
    else:
        df[f'target_vol_{n_ahead}'] = df['log_returns'].rolling(n_ahead).std().shift(-n_ahead+1)
    return df

# ------------------------------------------------------------------------------

def plot_actual_vs_predicted(y_test, y_pred, ticker, n_ahead, mse=None, mae=None, save_path_prefix=None, save_plot=True):
    """
    Plot actual vs predicted volatility for the test set.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(y_test.index), y_test.values, label='Actual Volatility', color='darkgray')
    plt.plot(np.array(y_test.index), y_pred, label='Predicted Volatility', color='darkred')
    title = f'Actual vs Predicted Volatility for {ticker} ({n_ahead}-period ahead)'
    if mse is not None and mae is not None:
        title += f"\nMSE: {mse:.4f} | MAE: {mae:.4f}"
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plot_path = None
    if save_plot:
        if save_path_prefix is None:
            save_path_prefix = f'xgb_{ticker}_{n_ahead}'
        plot_path = f'{save_path_prefix}_actual_vs_predicted.png'
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()
    return plot_path

# ------------------------------------------------------------------------------

def train_xgboost_volatility_model(
    ticker: str,
    n_ahead: int = 5,
    lags: list = [1, 2, 3, 5, 10, 21],
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 20,
    cv_splits: int = 3,
    param_distributions: dict = None,
    start: str = "2000-01-01",
    end: str = None,
    vix: bool = False,
    interval: str = "1d",
    auto_adjust: bool = True,
    log_returns: bool = True,
    earnings: bool = False,
    save_mlflow: bool = False,
    mlflow_experiment: str = "XGBoost-Volatility",
    save_plot: bool = False,
    model_id: str = None,
    objective: str = 'reg:squarederror',
    # delta: float = 0.0,
    early_stopping_rounds: int = None,
    **kwargs
):
    """
    Train XGBoost model to predict n-period ahead volatility for a given ticker.
    Returns: trained model, test set, predictions, metrics
    """
    # 1. Load data
    df = fetch_ticker_data(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        log_returns=log_returns,
        earnings=earnings,
        vix=vix,
        **kwargs
    )
    # 2. Feature engineering
    df = make_features(df, lags=lags, vix=vix)
    df = make_target(df, n_ahead=n_ahead)
    df = df.dropna()

    # 3. Feature selection
    if vix:
        feature_cols = [col for col in df.columns if col.startswith('log_return_lag_') or col.startswith('vol_lag_') or col.startswith('mean_lag_') or col.startswith('vix_lag_')]
    else:
        feature_cols = [col for col in df.columns if col.startswith('log_return_lag_') or col.startswith('vol_lag_') or col.startswith('mean_lag_')]

    X = df[feature_cols]
    y = df[f'target_vol_{n_ahead}']

    # 3. Train/test split (chronological)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. Hyperparameter search space
    if param_distributions is None:
        param_distributions = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 1],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0.1, 1, 10],
            'min_child_weight': [1, 2, 3, 4, 5],
        }
        if early_stopping_rounds:
            param_distributions['early_stopping_rounds'] = [early_stopping_rounds]

    # 5. Model & hyperparameter optimization
    xgb = XGBRegressor(objective=objective, random_state=random_state)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )
    fit_params = {}
    fit_params = {
        'eval_set': [(X_test, y_test)]
    }
    search.fit(X_train, y_train, **fit_params)

    best_model = search.best_estimator_

    # 6. Validation
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # 7. Visualization

    try:
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        workspace_root = os.getcwd()

    model_dir = os.path.join(workspace_root, "models", ticker, model_id)
    os.makedirs(model_dir, exist_ok=True)
    plot_path = plot_actual_vs_predicted(
        y_test, y_pred, ticker, n_ahead,
        mse=mse, mae=mae,
        save_path_prefix=os.path.join(model_dir, f'xgb_{ticker}_{n_ahead}'),
        save_plot=save_plot
    )

    # 8. MLflow logging
    run_id = None
    if save_mlflow:
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name=f"XGB_{ticker}_{n_ahead}_{model_id}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("n_ahead", n_ahead)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_param("vix", vix)
            mlflow.log_param("start", start)
            mlflow.log_param("end", end)
            mlflow.log_param("interval", interval)
            mlflow.log_param("auto_adjust", auto_adjust)
            mlflow.log_param("log_returns", log_returns)
            mlflow.log_param("earnings", earnings)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_iter", n_iter)
            mlflow.log_param("cv_splits", cv_splits)
            mlflow.log_param("save_mlflow", save_mlflow)
            mlflow.log_param("mlflow_experiment", mlflow_experiment)
            mlflow.log_param("save_plot", save_plot)
            mlflow.log_param("lags", lags)
            mlflow.log_param("param_distributions", param_distributions)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
            mlflow.log_param("objective", objective)

            # Save model in models/ticker/ with unique model id
            model_path = os.path.join(model_dir, f"xgb_model_{ticker}_{n_ahead}_{model_id}.pkl")
            joblib.dump(best_model, model_path)

            signature = infer_signature(X_train, best_model.predict(X_train))
            input_example = X_train.iloc[:5]
            mlflow.xgboost.log_model(
                best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )
            
            results_df = pd.DataFrame(search.cv_results_)
            cv_results_path = os.path.join(model_dir, f"xgb_cv_results_{model_id}.csv")
            results_df.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(cv_results_path)

            if plot_path:
                mlflow.log_artifact(plot_path)
    
            run_id = mlflow.active_run().info.run_id

    return best_model, (X_test, y_test, y_pred), {
        'mse': mse,
        'mae': mae,
        'plot_path': plot_path,
        'model_id': model_id,
        'run_id': run_id
    }

# ------------------------------------------------------------------------------

def validate_xgboost_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mse, mae


# ------------------------------------------------------------------------------

def train_validate_xgboost_with_mlflow(
    ticker: str,
    n_ahead: int = 5,
    lags: list = [1, 2, 3, 5, 10, 21],
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 20,
    cv_splits: int = 3,
    param_distributions: dict = None,
    start: str = "2000-01-01",
    end: str = datetime.now().strftime("%Y-%m-%d"),
    interval: str = "1d",
    auto_adjust: bool = True,
    log_returns: bool = True,
    earnings: bool = False,
    save_mlflow: bool = True,
    mlflow_experiment: str = "XGBoost-Volatility",
    save_plot: bool = True,
    model_id: str = None,
    **kwargs
):
    model, (X_test, y_test, y_pred), metrics = train_xgboost_volatility_model(
        ticker=ticker,
        n_ahead=n_ahead,
        lags=lags,
        test_size=test_size,
        random_state=random_state,
        n_iter=n_iter,
        cv_splits=cv_splits,
        param_distributions=param_distributions,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        log_returns=log_returns,
        earnings=earnings,
        save_mlflow=save_mlflow,
        mlflow_experiment=mlflow_experiment,
        save_plot=save_plot,
        model_id=model_id,
        **kwargs
    )
    return model, (X_test, y_test, y_pred), metrics
