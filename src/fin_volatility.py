import datetime
import numpy as np
import pandas as pd
from fin_options import implied_volatility
from fin_data import atm_option, get_next_expiration

#------------------------------------------------------------------------------------------------

def calculate_volatilities(stock, df: pd.DataFrame, today: datetime.date, periods: int, expirations: list, predicted_volatility: np.array, op_type: str="P", r: float = 0.0256, year_trading_days: int=252):
    """Calculate implied and predicted volatilities for a given period
    Args:
        - stock: stock object
        - df: DataFrame with the historical data
        - today: today's date
        - periods: number of periods to forecast
        - expirations: list of available expiration dates
        - predicted_volatility: predicted volatility
        - op_type: option type
    Returns:
        - results: DataFrame with the results: cols date, imp_vol, pred_vol, historical_vol
    """
    results = []
    S = df.Close.iloc[-1].item()
    
    historical_vol = df['log_return'].std() * np.sqrt(year_trading_days)

    for period in range(1, periods):
        expiration = get_next_expiration(today, period, expirations)
        options_chain = stock.option_chain(expiration)

        T = period / year_trading_days

        atm_call, atm_put = atm_option(options_chain.calls, options_chain.puts, S)

        if op_type == "C":
            K = atm_call.strike.item()
            market_price = atm_call.lastPrice.item()
        else:
            K = atm_put.strike.item()
            market_price = atm_put.lastPrice.item()

        imp_vol = implied_volatility(S, K, T, r, market_price, op_type=op_type)
        pred_vol = predicted_volatility[period - 1].item()

        current_date = today + pd.Timedelta(days=period)

        results.append([current_date, imp_vol, pred_vol, historical_vol])

    results_df = pd.DataFrame(results, columns=['date', 'imp_vol', 'pred_vol', 'historical_vol'])

    return results_df