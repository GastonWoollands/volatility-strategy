import datetime
import numpy as np
import pandas as pd
from .fin_options import implied_volatility
from .fin_data import atm_option, get_next_expiration

#------------------------------------------------------------------------------------------------

def calculate_volatilities(stock, df: pd.DataFrame, today: datetime.date, predicted_volatility: np.array, op_type: str="P", r: float = 0.0256, year_trading_days: int=252):
    """Calculate implied and predicted volatilities for a given period
    Args:
        - stock: stock object
        - df: DataFrame with the historical data
        - today: today's date
        - predicted_volatility: predicted volatility
        - op_type: option type
        - r: interest rate
        - year_trading_days (int): days in the year       

    Returns:
        - results: DataFrame with the results: cols date, imp_vol, pred_vol, historical_vol
    """
    results = []
    S = df.Close.iloc[-1].item()

    expirations = list(stock.options)[:30]

    historical_vol_daily  = df['log_return'].std()
    # historical_vol_yearly = historical_vol_daily * np.sqrt(year_trading_days)

    for expiration in expirations:
        days_to_expiration = (pd.to_datetime(expiration) - pd.to_datetime(today)).days

        if days_to_expiration <= 0:
            continue
        
        if days_to_expiration > len(predicted_volatility):
            break

        options_chain = stock.option_chain(expiration)

        T = days_to_expiration / year_trading_days

        atm_call, atm_put = atm_option(options_chain.calls, options_chain.puts, S)

        if op_type == "C" and atm_call:
            K = atm_call.strike.item()
            market_price = atm_call.lastPrice.item()

        elif op_type == "P" and atm_put:
            K = atm_put.strike.item()
            market_price = atm_put.lastPrice.item()
        else:
            continue

        imp_vol = implied_volatility(S, K, T, r, market_price, op_type=op_type)
        imp_vol *= 100 # Log returns are in percentage
        
        pred_vol = predicted_volatility[days_to_expiration - 1].item()

        current_date = today + pd.Timedelta(days=days_to_expiration)

        results.append([current_date, imp_vol, pred_vol, historical_vol_daily])

    results_df = pd.DataFrame(results, columns=['date', 'imp_vol', 'pred_vol', 'historical_vol'])

    return results_df