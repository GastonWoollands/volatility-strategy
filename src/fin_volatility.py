import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from .fin_data import atm_option
from .fin_options import implied_volatility, get_greeks

#------------------------------------------------------------------------------------------------

def calculate_hist_vol(df: pd.DataFrame, periods: int = 252):
    """Calculates daily and annualized historical volatility."""
    historical_vol_daily = df.iloc[-60:]["log_return"].std()
    historical_vol_yearly = historical_vol_daily * np.sqrt(periods)
    return round(float(historical_vol_daily), 4), round(float(historical_vol_yearly), 4)

#------------------------------------------------------------------------------------------------

def calculate_volatility_expiration(
    stock, expiration: str, S: float, historical_vol_daily: float, 
    predicted_volatility: np.array, predicted_volatility_gru: np.array, 
    op_type: str = "P", r: float = 0.0256, periods: int = 252
):
    """Calculates implied volatility and predicted volatility for a single expiration date."""
    today = datetime.now().date()
    days_to_expiration = (pd.to_datetime(expiration).date() - today).days

    if days_to_expiration <= 0 or days_to_expiration > len(predicted_volatility):
        return None

    try:
        options_chain = stock.option_chain(expiration)
        calls, puts = options_chain.calls, options_chain.puts
        _T = days_to_expiration / periods

        atm_call, atm_put = atm_option(calls, puts, S)

        if op_type == "C":
            K = atm_call.strike.item()
            market_price = atm_call.lastPrice.item()
        else:
            K = atm_put.strike.item()
            market_price = atm_put.lastPrice.item()

        imp_vol = implied_volatility(S, K, _T, r, market_price, op_type=op_type, periods=periods) * 100
        pred_vol = predicted_volatility[days_to_expiration - 1].item()
        pred_vol_gru = predicted_volatility_gru[days_to_expiration - 1].item()

        expiration_date = (datetime.today() + pd.Timedelta(days=days_to_expiration)).date()

        return {
            "date": expiration_date,
            "imp_vol": imp_vol,
            "pred_vol": pred_vol,
            "pred_vol_gru": pred_vol_gru,
            "historical_vol": historical_vol_daily
        }
    except Exception as e:
        print(f"Error processing expiration {expiration}: {e}")
        return None

#------------------------------------------------------------------------------------------------

def calculate_volatilities(
    stock, df: pd.DataFrame, predicted_volatility: np.array, 
    predicted_volatility_gru: np.array, op_type: str = "P", r: float = 0.0256, periods: int = 252
):
    """Calculates volatilities for multiple expiration dates."""
    S = df.Close.iloc[-1].item()
    historical_vol_daily, _ = calculate_hist_vol(df, periods)
    expirations = list(stock.options)[:30]

    results = []
    for expiration in expirations:
        result = calculate_volatility_expiration(
            stock, expiration, S, historical_vol_daily, predicted_volatility, 
            predicted_volatility_gru, op_type=op_type, r=r, periods=periods
        )
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df

#------------------------------------------------------------------------------------------------

def calculate_implied_vol_expiration_byma(
    expiration: Union[str, pd.Timestamp], S: float, init_date: Union[str, pd.Timestamp], periods: int, 
    op_type: str, r: float, predicted_volatility: np.array, predicted_volatility_gru: np.array,
    data_options: pd.DataFrame ):

    """Calculates implied and predicted volatilities for a single expiration."""
    init_date = pd.to_datetime(init_date)
    expiration = pd.to_datetime(expiration)
    days_to_expiration = (expiration - init_date).days
    if days_to_expiration <= 0:
        return None

    calls = get_date_options_byma(data_options, expiration, "C")
    puts = get_date_options_byma(data_options, expiration, "P")
    _T = days_to_expiration / periods

    atm_call, atm_put = atm_option(calls, puts, S)

    if op_type == "C" and atm_call is not None:
        K = atm_call.strike.item()
        market_price = atm_call.lastPrice.item()
        
    elif op_type == "P" and atm_put is not None:
        K = atm_put.strike.item()
        market_price = atm_put.lastPrice.item()
    else:
        print(f"No ATM option found for {expiration}")
        return None

    try:
        imp_vol = implied_volatility(S, K, _T, r, market_price, op_type=op_type, periods=periods) * 100
        pred_vol = predicted_volatility[days_to_expiration - 1].item()
        pred_vol_gru = predicted_volatility_gru[days_to_expiration - 1].item()

        delta, gamma, vega, theta, rho = get_greeks(op_type, S, K, _T, r, imp_vol / 100)

        return {
            "date": expiration.strftime('%Y-%m-%d'),
            "market_price": S,
            "option_type": op_type,
            "strike": K,
            "option_price": market_price,
            "imp_vol": round(float(imp_vol), 4),
            "pred_vol": round(float(pred_vol), 4),
            "pred_vol_gru": round(float(pred_vol_gru), 4),
            "delta": round(float(delta), 6) ,
            "gamma": round(float(gamma), 6),
            "vega": round(float(vega), 6),
            "theta": round(float(theta), 6),
            "rho": round(float(rho), 6)
        }
    
    except Exception as e:
        print(f"Error processing expiration {expiration}: {e}")
        return None

#------------------------------------------------------------------------------------------------

def calculate_vol_results_byma(
    data_op: pd.DataFrame, df: pd.DataFrame, predicted_volatility: np.array, 
    predicted_volatility_gru: np.array, op_type: str = "P", r: float = 0.0256, periods: int = 252
):
    """Calculates volatility results for all expiration dates."""
    today = pd.Timestamp.today()
    expirations = sorted(list(set(data_op.expiration)))

    S = df.Close.iloc[-1].item()
    vol_daily_hist, _ = calculate_hist_vol(df, periods)

    results = []
    for expiration in expirations:
        result = calculate_implied_vol_expiration_byma(
            expiration, S, today, periods, op_type, r, predicted_volatility, 
            predicted_volatility_gru, data_op
        )
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df["historical_vol"] = vol_daily_hist
    return results_df

#------------------------------------------------------------------------------------------------

def get_date_options_byma(data_options: pd.DataFrame, expiration, option_type: str):
    """Filters options by expiration date and type."""
    if isinstance(expiration, str):
        expiration = pd.to_datetime(expiration)
    data_options['expiration'] = pd.to_datetime(data_options['expiration'])
    filtered_options = data_options[(data_options['expiration'] == expiration) & (data_options['option_type'] == option_type)].reset_index(drop=True)
    
    return filtered_options
