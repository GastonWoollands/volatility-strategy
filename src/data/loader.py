import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web # ['DGS1MO', 'DGS5', 'DGS10', 'DGS30', 'CPIAUCSL']
import logging
from datetime import datetime

#------------------------------------------------------------------------------

def _fetch_ticker_data(ticker: str, start: str, end: str, interval: str, auto_adjust: bool, **kwargs):
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        **kwargs
    )

    df.index = pd.to_datetime(df.index).tz_localize(None)

    df.columns = df.columns.str.lower()

    return df, ticker_obj

#------------------------------------------------------------------------------

def fetch_ticker_data(
    ticker: str,
    start: str = "2000-01-01",
    end: str = datetime.now().strftime('%Y-%m-%d'),
    interval: str = "1d",
    auto_adjust: bool = True,
    log_returns: bool = True,
    earnings: bool = False,
    vix: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Fetch historical market data for a single ticker using yfinance.

    Args:
        ticker (str): The ticker symbol to fetch (e.g., 'AAPL').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format. Defaults to today.
        interval (str): Data interval ('1d', '1wk', '1mo', etc.).
        auto_adjust (bool): Adjust prices for splits/dividends.
        progress (bool): Show download progress bar.
        **kwargs: Additional arguments for yfinance.Ticker.history.

    Returns:
        pd.DataFrame: DataFrame indexed by DatetimeIndex with OHLCV data.
    """
    df, ticker_obj = _fetch_ticker_data(ticker, start, end, interval, auto_adjust, **kwargs)

    if log_returns:
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df.dropna(inplace=True)

    if earnings:
        earnings_dates = ticker_obj.get_earnings_dates(limit=1000)
        earnings_dates.index = pd.to_datetime(earnings_dates.index).tz_localize(None)

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
    
    if vix:
        vix_data, _ = _fetch_ticker_data("^VIX", start, end, "1d", True)
        vix_data = vix_data.close
        vix_data.name = 'vix'        
        df = df.merge(vix_data, left_index=True, right_index=True, how='left')
        df['vix'] = df['vix'].ffill().bfill().astype(float)

    return df

#------------------------------------------------------------------------------

def fetch_merge_rates(df: pd.DataFrame, rate_cols: list = None, start: str = "2000-01-01", end: str = None, **kwargs):
    """Include rate columns to dataframe"""
        
    for rate in rate_cols:
        try:
            rate_df = web.DataReader(rate, 'fred', start, end, **kwargs)
            df = df.merge(rate_df, left_index=True, right_index=True, how='left')
            df[rate] = df[rate].ffill().bfill()
        except Exception as e:
            logging.error(f"Failed to fetch rate {rate}: {e}")
    return df