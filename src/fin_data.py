import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

#------------------------------------------------------------------------------------------------

def get_data(ticker, start_date: str, end_date: str, include_earnings: bool = False):
    """Get historical data from Yahoo Finance
    Args:
        - ticker: ticker of the stock
        - start_date: start date of the historical data YYYY-MM-DD
        - end_date: end date of the historical data YYYY-MM-DD
        - include_earnings: whether to include earnings data
    Returns:
        - df: DataFrame with the historical data
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    df = pd.DataFrame(hist, columns=['Close'])

    df['log_return'] = np.log(df['Close']).diff() * 100

    df.dropna(inplace=True)

    df.index = pd.to_datetime(df.index).date

    if include_earnings:
        earnings_dates = stock.get_earnings_dates(limit=100)
        earnings_dates.index = pd.to_datetime(earnings_dates.index).date

        earnings_dates = earnings_dates.groupby(earnings_dates.index).agg({
            'EPS Estimate': 'max',
            'Surprise(%)': 'max'
        }).rename(columns={
            'EPS Estimate': 'estimate',
            'Surprise(%)': 'surprise'
        })

        df['earnings'] = df.index.isin(earnings_dates.index).astype(int)
        df = df.merge(earnings_dates, left_index=True, right_index=True, how='left')
        df.fillna(0, inplace=True)
    return df

#------------------------------------------------------------------------------------------------

def get_next_expiration(date, n_days, expirations):
    """
    Fetch the expiration date N days ahead from a particular date.
    
    Args:
        date (datetime.date): The starting date.
        n_days (int): Number of days to look ahead.
        expirations (list): List of available expiration dates as strings (YYYY-MM-DD).
    
    Returns:
        str: The closest expiration date after n_days, in YYYY-MM-DD format.
    """
    target_date = date + timedelta(days=n_days)

    expiration_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
    
    future_dates = [d for d in expiration_dates if d >= target_date]
    
    if not future_dates:
        return None
    
    closest_date = min(future_dates, key=lambda d: abs(d - target_date))
    
    return closest_date.strftime('%Y-%m-%d')

#------------------------------------------------

def atm_option(calls, puts, S):
    """Get the at-the-money option based on the closest strike to the spot price S."""
    atm_call = calls.iloc[(calls['strike'] - S).abs().idxmin()] if calls is not None and not calls.empty else None
    atm_put = puts.iloc[(puts['strike'] - S).abs().idxmin()] if puts is not None and not puts.empty else None

    return atm_call, atm_put