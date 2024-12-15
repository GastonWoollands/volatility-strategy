import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
import matplotlib.pyplot as plt
from .fin_data import atm_option
from .fin_options import implied_volatility, get_greeks
import logging

#------------------------------------------------------------------------------------------------

def get_hist_vol(df: pd.DataFrame, days: int, periods: int = 252):
    """Calculates daily and annualized historical volatility."""
    historical_vol_daily = df.iloc[-days:]["log_return"].std()
    historical_vol_yearly = historical_vol_daily * np.sqrt(periods)
    return round(float(historical_vol_daily), 4), round(float(historical_vol_yearly), 4)

#------------------------------------------------------------------------------------------------

def get_option_metrics(
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

def get_option_metrics_expirations(
    stock, df: pd.DataFrame, predicted_volatility: np.array, 
    predicted_volatility_gru: np.array, op_type: str = "P", r: float = 0.0256, periods: int = 252
):
    """Calculates volatilities for multiple expiration dates."""
    S = df.Close.iloc[-1].item()
    historical_vol_daily, _ = get_hist_vol(df, periods)
    expirations = list(stock.options)[:30]

    results = []
    for expiration in expirations:
        result = get_option_metrics(
            stock, expiration, S, historical_vol_daily, predicted_volatility, 
            predicted_volatility_gru, op_type=op_type, r=r, periods=periods
        )
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df

#------------------------------------------------------------------------------------------------

def get_option_metrics_byma(
    expiration: Union[str, pd.Timestamp], S: float, init_date: Union[str, pd.Timestamp], periods: int, 
    op_type: str, r: float, predicted_volatility: np.array, predicted_volatility_gru: np.array,
    data_options: pd.DataFrame ):

    """Calculates implied and predicted volatilities for a single expiration."""

    init_date, expiration = pd.to_datetime(init_date), pd.to_datetime(expiration)

    days_to_expiration = (expiration - init_date).days
    if days_to_expiration <= 0:
        return None
    
    _T = days_to_expiration / periods

    options = filter_options_byma(data_options, expiration, op_type)

    atm_call, atm_put = atm_option(calls=options if op_type == "C" else None, 
                                   puts=options if op_type == "P" else None, 
                                   S=S)

    atm_option_data = atm_call if op_type == "C" else atm_put
    
    if atm_option_data is None:
        print(f"No ATM option found for {expiration}")
        return None

    K, market_price = atm_option_data.strike.item(), atm_option_data.close.item() if atm_option_data.close.item() !=0 else atm_option_data.lastPrice.item()

    try:
        imp_vol = implied_volatility(S, K, _T, r, market_price, op_type=op_type, periods=periods) * 100

        index = days_to_expiration - 1

        pred_vol = predicted_volatility[index].item() if index < len(predicted_volatility) else np.nan
        pred_vol_gru = predicted_volatility_gru[index].item() if index < len(predicted_volatility_gru) else np.nan
        delta, gamma, vega, theta, rho = get_greeks(op_type, S, K, _T, r, imp_vol / 100)

        # Compile metrics
        return {
            "days_to_expiration": days_to_expiration,
            "asset_price": S,
            "option_type": op_type,
            "strike": K,
            "option_price": market_price,
            "imp_vol": float(round(imp_vol, 4)),
            "pred_vol": float(round(pred_vol, 4)),
            "pred_vol_gru": float(round(pred_vol_gru, 4)),
            "delta": float(round(delta, 6)),
            "gamma": float(round(gamma, 6)),
            "vega": float(round(vega, 6)),
            "theta": float(round(theta, 6)),
            "rho": float(round(rho, 6)),
        }

    except Exception as e:
        print(f"Error processing expiration {expiration}: {e}")
        return None

#------------------------------------------------------------------------------------------------

def get_option_metrics_expirations_byma(
    data_op: pd.DataFrame, df: pd.DataFrame, predicted_volatility: np.array, 
    predicted_volatility_gru: np.array, op_type: str = "P", r: float = 0.0256, periods: int = 252):

    """Calculates volatility results for all expiration dates."""
    today = pd.Timestamp.today()

    expirations = sorted(list(set(data_op.expiration)))

    S = df.Close.iloc[-1].item()
    historical_vols = {10: None, 20: None, 30: None, 60: None}
    for days in historical_vols.keys():
        historical_vols[days], _ = get_hist_vol(df, days, periods)

    results = {}
    for expiration in expirations:
        result = get_option_metrics_byma(
            expiration, S, today, periods, op_type, r, predicted_volatility, 
            predicted_volatility_gru, data_op
        )
        if result:
            for days, vol in historical_vols.items():
                result[f'historical_vol_{days}'] = vol
            results[expiration] = result

    return results

#------------------------------------------------------------------------------------------------

def filter_options_byma(data_options: pd.DataFrame, expiration, option_type: str):
    
    """Filters options by expiration date and type."""

    if isinstance(expiration, str):
        expiration = pd.to_datetime(expiration)

    data_options['expiration'] = pd.to_datetime(data_options['expiration'])
    filtered_options = data_options[(data_options['expiration'] == expiration) & (data_options['option_type'] == option_type)].reset_index(drop=True)
    
    return filtered_options

#------------------------------------------------------------------------------------------------

class OptionStrategy:
    def __init__(self, data):
        self.data = data
        self.positions = {}
        self.expirations = list(data.keys())

    def generate_signals(self):
        for date, details in self.data.items():
            imp_vol = details.get('imp_vol')
            hist_vol_30 = details.get('historical_vol_30') 

            if None in (imp_vol, hist_vol_30):
                continue

            self.positions[date] = {
                "action": "SELL" if imp_vol > hist_vol_30 else "BUY",
                "option_type": details["option_type"],
                "strike": details["strike"],
                "asset_price": details["asset_price"], 
                "option_price": details["option_price"],
                "imp_vol": imp_vol,
                "hist_vol_30": hist_vol_30
            }

    def plot_payoff(self, expiration_date, mix_strategy=False):
        if expiration_date not in self.positions:
            print(f"No data for {expiration_date}.")
            return

        pos = self.positions[expiration_date]
        strike = pos["strike"]
        # asset_price = pos["asset_price"]
        option_price = pos["option_price"]
        option_type = pos["option_type"]
        action = pos["action"]
        imp_vol = pos["imp_vol"]
        hist_vol_30 = pos["hist_vol_30"]

        # Price range
        asset_prices = np.linspace(0.5 * strike, 1.5 * strike, 500)

        # Initial payoff
        if option_type == "C":  # Call
            intrinsic_value = np.maximum(asset_prices - strike, 0)
        else:  # Put
            intrinsic_value = np.maximum(strike - asset_prices, 0)
        
        payoff_option = intrinsic_value - option_price

        if action == "SELL":
            payoff_option = -payoff_option

        # Combined payoff for mixed strategies
        payoff_combined = payoff_option.copy()
        decision = "Estrategia Simple"

        if mix_strategy:
            if imp_vol < hist_vol_30:  # VI < VH
                # Strategy: Sell PUT + Buy CALL
                # Sell PUT
                put_intrinsic = np.maximum(strike - asset_prices, 0)
                payoff_put = -put_intrinsic + option_price  # Sell PUT (SELL P)

                # Buy CALL
                call_intrinsic = np.maximum(asset_prices - strike, 0)
                payoff_call = call_intrinsic - option_price  # Buy CALL (BUY C)

                payoff_combined = payoff_put + payoff_call
                decision = "Venta de PUT + Compra de CALL"
            else:  # VI > VH
                # Strategy: Sell CALL + Buy PUT
                # Sell CALL
                call_intrinsic = np.maximum(asset_prices - strike, 0)
                payoff_call = -call_intrinsic + option_price  # Sell CALL (SELL C)

                # Buy PUT
                put_intrinsic = np.maximum(strike - asset_prices, 0)
                payoff_put = put_intrinsic - option_price  # Buy PUT (BUY P)

                payoff_combined = payoff_call + payoff_put
                decision = "Venta de CALL + Compra de PUT"

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(asset_prices, payoff_combined, label="Payoff", color="black")

        ax.fill_between(asset_prices, payoff_combined, 0, where=(payoff_combined > 0), color="lightgreen", alpha=0.5, label="Ganancia")
        ax.fill_between(asset_prices, payoff_combined, 0, where=(payoff_combined < 0), color="lightcoral", alpha=0.5, label="Pérdida")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Payoff - Fecha Expiración: {expiration_date}\n{decision}", fontsize=14)
        ax.set_xlabel("Precio del Activo", fontsize=12)
        ax.set_ylabel("Payoff", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.show()
