"""
Options Data Loader Module

This module provides functionality to fetch and analyze options data from Yahoo Finance.
It includes classes and functions for retrieving options chains, calculating implied volatility,
and analyzing options data for volatility modeling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
import logging
from dataclasses import dataclass
import warnings
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.black_scholes import (
    implied_volatility_call, 
    implied_volatility_put, 
    calculate_greeks,
    get_risk_free_rate
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class OptionsData:
    """Data class to hold options data with metadata."""
    ticker: str
    expiration_date: datetime
    calls: pd.DataFrame
    puts: pd.DataFrame
    underlying_price: float
    fetch_timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.calls.empty and self.puts.empty:
            raise ValueError("Both calls and puts dataframes cannot be empty")
        
        if self.underlying_price <= 0:
            raise ValueError("Underlying price must be positive")


class OptionsDataFetcher:
    """
    A class to fetch and manage options data from Yahoo Finance.
    
    This class provides methods to retrieve options chains, calculate implied volatility,
    and analyze options data for volatility modeling purposes.
    """
    
    def __init__(self, ticker: str):
        """
        Initialize the OptionsDataFetcher.
        
        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL')
        """
        self.ticker = ticker.upper()
        self.ticker_obj = yf.Ticker(self.ticker)
        self._validate_ticker()
    
    def _validate_ticker(self) -> None:
        """Validate that the ticker exists and has options data."""
        try:
            info = self.ticker_obj.info
            if not info or 'regularMarketPrice' not in info:
                raise ValueError(f"Invalid ticker: {self.ticker}")
            
            # Check if options are available
            options = self.ticker_obj.options
            if not options:
                logger.warning(f"No options data available for {self.ticker}")
                
        except Exception as e:
            raise ValueError(f"Error validating ticker {self.ticker}: {str(e)}")
    
    def get_expiration_dates(self) -> List[datetime]:
        """
        Get all available expiration dates for the ticker.
        
        Returns:
            List[datetime]: List of available expiration dates
        """
        try:
            dates = self.ticker_obj.options
            return [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        except Exception as e:
            logger.error(f"Error fetching expiration dates for {self.ticker}: {str(e)}")
            return []
    
    def get_options_chain(self, expiration_date: Union[str, datetime]) -> OptionsData:
        """
        Fetch options chain for a specific expiration date.
        
        Args:
            expiration_date (Union[str, datetime]): Expiration date in 'YYYY-MM-DD' format or datetime object
            
        Returns:
            OptionsData: Object containing calls, puts, and metadata
            
        Raises:
            ValueError: If expiration date is invalid or no data available
        """
        try:
            # Convert to string format if datetime
            if isinstance(expiration_date, datetime):
                expiration_str = expiration_date.strftime('%Y-%m-%d')
            else:
                expiration_str = expiration_date
            
            # Fetch options chain
            options_chain = self.ticker_obj.option_chain(expiration_str)
            
            # Get underlying price
            underlying_price = self.ticker_obj.info.get('regularMarketPrice', 0)
            
            # Create OptionsData object
            options_data = OptionsData(
                ticker=self.ticker,
                expiration_date=datetime.strptime(expiration_str, '%Y-%m-%d'),
                calls=options_chain.calls,
                puts=options_chain.puts,
                underlying_price=underlying_price,
                fetch_timestamp=datetime.now()
            )
            
            logger.info(f"Successfully fetched options data for {self.ticker} expiring {expiration_str}")
            return options_data
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {self.ticker} expiring {expiration_date}: {str(e)}")
            raise ValueError(f"Failed to fetch options data: {str(e)}")
    
    def get_multiple_expirations(self, expiration_dates: List[Union[str, datetime]]) -> Dict[datetime, OptionsData]:
        """
        Fetch options chains for multiple expiration dates.
        
        Args:
            expiration_dates (List[Union[str, datetime]]): List of expiration dates
            
        Returns:
            Dict[datetime, OptionsData]: Dictionary mapping expiration dates to options data
        """
        results = {}
        
        for date in expiration_dates:
            try:
                options_data = self.get_options_chain(date)
                results[options_data.expiration_date] = options_data
            except Exception as e:
                logger.warning(f"Skipping {date}: {str(e)}")
                continue
        
        return results
    
    def get_nearest_expirations(self, n: int = 3) -> Dict[datetime, OptionsData]:
        """
        Fetch options chains for the nearest n expiration dates.
        
        Args:
            n (int): Number of nearest expirations to fetch
            
        Returns:
            Dict[datetime, OptionsData]: Dictionary mapping expiration dates to options data
        """
        available_dates = self.get_expiration_dates()
        if not available_dates:
            raise ValueError(f"No expiration dates available for {self.ticker}")
        
        # Sort by date and take nearest n
        nearest_dates = sorted(available_dates)[:n]
        return self.get_multiple_expirations(nearest_dates)


class OptionsAnalyzer:
    """
    A class to analyze options data for volatility modeling.
    
    This class provides methods to calculate implied volatility, analyze options skew,
    and extract volatility-related information from options data.
    """
    
    def __init__(self, options_data: OptionsData):
        """
        Initialize the OptionsAnalyzer.
        
        Args:
            options_data (OptionsData): Options data to analyze
        """
        self.options_data = options_data
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that the options data is suitable for analysis."""
        if self.options_data.calls.empty and self.options_data.puts.empty:
            raise ValueError("No options data available for analysis")
    
    def calculate_implied_volatility(self, option_type: str = 'both') -> pd.DataFrame:
        """
        Calculate implied volatility for options.
        
        Note: This is a simplified calculation. For accurate IV, you would need
        a proper options pricing model like Black-Scholes.
        
        Args:
            option_type (str): 'calls', 'puts', or 'both'
            
        Returns:
            pd.DataFrame: DataFrame with implied volatility calculations
        """
        results = []
        
        if option_type in ['calls', 'both'] and not self.options_data.calls.empty:
            calls_iv = self._calculate_iv_for_options(self.options_data.calls, 'call')
            results.append(calls_iv)
        
        if option_type in ['puts', 'both'] and not self.options_data.puts.empty:
            puts_iv = self._calculate_iv_for_options(self.options_data.puts, 'put')
            results.append(puts_iv)
        
        if not results:
            raise ValueError("No options data available for IV calculation")
        
        return pd.concat(results, ignore_index=True)
    
    def _calculate_iv_for_options(self, options_df: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """
        Calculate implied volatility for a specific option type using Black-Scholes.
        
        Args:
            options_df (pd.DataFrame): Options dataframe
            option_type (str): 'call' or 'put'
            
        Returns:
            pd.DataFrame: DataFrame with IV calculations
        """
        df = options_df.copy()
        
        # Add option type
        df['option_type'] = option_type
        
        # Calculate moneyness (S/K ratio)
        df['moneyness'] = self.options_data.underlying_price / df['strike']
        
        # Calculate time to expiration (in years)
        days_to_expiry = (self.options_data.expiration_date - self.options_data.fetch_timestamp).days
        df['time_to_expiry'] = days_to_expiry / 365.25
        
        # Get risk-free rate
        r = get_risk_free_rate()
        
        # Calculate implied volatility using Black-Scholes
        df['implied_volatility'] = None
        df['delta'] = None
        df['gamma'] = None
        df['theta'] = None
        df['vega'] = None
        df['rho'] = None
        
        for idx, row in df.iterrows():
            S = self.options_data.underlying_price
            K = row['strike']
            T = row['time_to_expiry']
            
            # Use mid-price for IV calculation
            if pd.notna(row['bid']) and pd.notna(row['ask']):
                price = (row['bid'] + row['ask']) / 2
            elif pd.notna(row['lastPrice']):
                price = row['lastPrice']
            else:
                continue
            
            if T <= 0 or price <= 0:
                continue
            
            # Calculate implied volatility
            if option_type == 'call':
                iv = implied_volatility_call(price, S, K, T, r)
            else:  # put
                iv = implied_volatility_put(price, S, K, T, r)
            
            if iv is not None:
                df.at[idx, 'implied_volatility'] = iv
                
                # Calculate Greeks
                greeks = calculate_greeks(S, K, T, r, iv, option_type)
                df.at[idx, 'delta'] = greeks['delta']
                df.at[idx, 'gamma'] = greeks['gamma']
                df.at[idx, 'theta'] = greeks['theta']
                df.at[idx, 'vega'] = greeks['vega']
                df.at[idx, 'rho'] = greeks['rho']
        
        # Add useful columns
        df['expiration_date'] = self.options_data.expiration_date
        df['underlying_price'] = self.options_data.underlying_price
        
        return df
    

    
    def get_volatility_skew(self, option_type: str = 'both') -> pd.DataFrame:
        """
        Calculate volatility skew (IV vs moneyness).
        
        Args:
            option_type (str): 'calls', 'puts', or 'both'
            
        Returns:
            pd.DataFrame: DataFrame with volatility skew data
        """
        iv_data = self.calculate_implied_volatility(option_type)
        
        # Filter for reasonable data
        skew_data = iv_data[
            (iv_data['implied_volatility'] > 0) & 
            (iv_data['implied_volatility'] < 2) &
            (iv_data['moneyness'] > 0.5) &
            (iv_data['moneyness'] < 2.0)
        ].copy()
        
        # Add skew metrics
        skew_data['log_moneyness'] = np.log(skew_data['moneyness'])
        
        return skew_data
    
    def get_atm_implied_volatility(self, tolerance: float = 0.05) -> Dict[str, float]:
        """
        Get at-the-money implied volatility using Black-Scholes calculations.
        
        Args:
            tolerance (float): Tolerance for ATM definition (default 5%)
            
        Returns:
            Dict[str, float]: Dictionary with ATM IV for calls and puts
        """
        results = {}
        
        # Define ATM range
        atm_min = 1 - tolerance
        atm_max = 1 + tolerance
        
        iv_data = self.calculate_implied_volatility('both')
        
        for option_type in ['call', 'put']:
            atm_options = iv_data[
                (iv_data['option_type'] == option_type) &
                (iv_data['moneyness'] >= atm_min) &
                (iv_data['moneyness'] <= atm_max) &
                (iv_data['implied_volatility'] > 0)
            ]
            
            if not atm_options.empty:
                # Use volume-weighted average if available
                if 'volume' in atm_options.columns and atm_options['volume'].sum() > 0:
                    atm_iv = np.average(
                        atm_options['implied_volatility'],
                        weights=atm_options['volume']
                    )
                else:
                    atm_iv = atm_options['implied_volatility'].mean()
                
                results[option_type] = atm_iv
            else:
                results[option_type] = None
        
        return results
    
    def get_options_summary(self) -> Dict:
        """
        Get a summary of options data.
        
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'ticker': self.options_data.ticker,
            'expiration_date': self.options_data.expiration_date,
            'underlying_price': self.options_data.underlying_price,
            'fetch_timestamp': self.options_data.fetch_timestamp,
            'num_calls': len(self.options_data.calls),
            'num_puts': len(self.options_data.puts),
            'total_options': len(self.options_data.calls) + len(self.options_data.puts)
        }
        
        # Add strike range information
        if not self.options_data.calls.empty:
            summary['calls_strike_range'] = (
                self.options_data.calls['strike'].min(),
                self.options_data.calls['strike'].max()
            )
        
        if not self.options_data.puts.empty:
            summary['puts_strike_range'] = (
                self.options_data.puts['strike'].min(),
                self.options_data.puts['strike'].max()
            )
        
        # Add ATM IV if available
        try:
            atm_iv = self.get_atm_implied_volatility()
            summary.update(atm_iv)
        except Exception as e:
            logger.warning(f"Could not calculate ATM IV: {str(e)}")
        
        # Add IV statistics if available
        try:
            iv_data = self.calculate_implied_volatility('both')
            if not iv_data.empty and 'implied_volatility' in iv_data.columns:
                valid_iv = iv_data['implied_volatility'].dropna()
                if len(valid_iv) > 0:
                    summary['iv_range'] = float(valid_iv.max() - valid_iv.min())
                    summary['iv_std'] = float(valid_iv.std())
                    
                    # Calculate put-call skew if both types available
                    call_iv = iv_data[iv_data['option_type'] == 'call']['implied_volatility'].dropna()
                    put_iv = iv_data[iv_data['option_type'] == 'put']['implied_volatility'].dropna()
                    
                    if len(call_iv) > 0 and len(put_iv) > 0:
                        summary['put_call_skew'] = float(put_iv.mean() - call_iv.mean())
        except Exception as e:
            logger.warning(f"Could not calculate IV statistics: {str(e)}")
        
        return summary

    def get_atm_options_data(self, tolerance: float = 0.05) -> pd.DataFrame:
        """
        Get at-the-money options data including Greeks for the current expiration.
        
        Args:
            tolerance (float): Tolerance for ATM definition (default 5%)
            
        Returns:
            pd.DataFrame: DataFrame with ATM options data including Greeks
        """
        # Define ATM range
        atm_min = 1 - tolerance
        atm_max = 1 + tolerance
        
        iv_data = self.calculate_implied_volatility('both')
        
        # Filter for ATM options
        atm_options = iv_data[
            (iv_data['moneyness'] >= atm_min) &
            (iv_data['moneyness'] <= atm_max) &
            (iv_data['implied_volatility'] > 0)
        ].copy()
        
        if atm_options.empty:
            return pd.DataFrame()
        
        # Add additional useful columns
        atm_options['days_to_expiry'] = (self.options_data.expiration_date - self.options_data.fetch_timestamp).days
        atm_options['expiration_date'] = self.options_data.expiration_date
        atm_options['underlying_price'] = self.options_data.underlying_price
        atm_options['ticker'] = self.options_data.ticker
        
        # Reorder columns for better readability
        column_order = [
            'ticker', 'expiration_date', 'option_type', 'strike', 'moneyness',
            'days_to_expiry', 'time_to_expiry', 'underlying_price',
            'bid', 'ask', 'lastPrice', 'volume', 'openInterest',
            'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in column_order if col in atm_options.columns]
        atm_options = atm_options[available_columns]
        
        return atm_options

    @classmethod
    def get_atm_options_across_expirations(
        cls, 
        options_data_dict: Dict[datetime, OptionsData], 
        tolerance: float = 0.10
    ) -> pd.DataFrame:
        """
        Get ATM options data including Greeks across all available expirations.
        
        Args:
            options_data_dict (Dict[datetime, OptionsData]): Dictionary of options data by expiration
            tolerance (float): Tolerance for ATM definition (default 5%)
            
        Returns:
            pd.DataFrame: DataFrame with ATM options data for all expirations
        """
        all_atm_data = []
        
        for expiration_date, options_data in options_data_dict.items():
            analyzer = cls(options_data)
            atm_data = analyzer.get_atm_options_data(tolerance)
            
            if not atm_data.empty:
                all_atm_data.append(atm_data)
        
        if not all_atm_data:
            return pd.DataFrame()
        
        # Combine all ATM data
        combined_data = pd.concat(all_atm_data, ignore_index=True)
        
        # Sort by expiration date and option type
        combined_data = combined_data.sort_values(['expiration_date', 'option_type', 'strike'])
        
        return combined_data

    def get_atm_greeks_summary(self, tolerance: float = 0.05) -> Dict:
        """
        Get a summary of ATM Greeks for the current expiration.
        
        Args:
            tolerance (float): Tolerance for ATM definition (default 5%)
            
        Returns:
            Dict: Summary of ATM Greeks
        """
        atm_data = self.get_atm_options_data(tolerance)
        
        if atm_data.empty:
            return {}
        
        summary = {
            'ticker': self.options_data.ticker,
            'expiration_date': self.options_data.expiration_date,
            'underlying_price': self.options_data.underlying_price,
            'days_to_expiry': (self.options_data.expiration_date - self.options_data.fetch_timestamp).days
        }
        
        # Calculate Greeks summary for calls and puts
        for option_type in ['call', 'put']:
            type_data = atm_data[atm_data['option_type'] == option_type]
            
            if not type_data.empty:
                summary[f'{option_type}_count'] = len(type_data)
                summary[f'{option_type}_avg_iv'] = float(type_data['implied_volatility'].mean())
                summary[f'{option_type}_avg_delta'] = float(type_data['delta'].mean())
                summary[f'{option_type}_avg_gamma'] = float(type_data['gamma'].mean())
                summary[f'{option_type}_avg_theta'] = float(type_data['theta'].mean())
                summary[f'{option_type}_avg_vega'] = float(type_data['vega'].mean())
                summary[f'{option_type}_avg_rho'] = float(type_data['rho'].mean())
                
                # Add strike range for this option type
                summary[f'{option_type}_strike_range'] = (
                    float(type_data['strike'].min()),
                    float(type_data['strike'].max())
                )
        
        return summary

    @classmethod
    def get_atm_greeks_across_expirations(
        cls, 
        options_data_dict: Dict[datetime, OptionsData], 
        tolerance: float = 0.05
    ) -> Dict[datetime, Dict]:
        """
        Get ATM Greeks summary across all available expirations.
        
        Args:
            options_data_dict (Dict[datetime, OptionsData]): Dictionary of options data by expiration
            tolerance (float): Tolerance for ATM definition (default 5%)
            
        Returns:
            Dict[datetime, Dict]: Dictionary mapping expiration dates to Greeks summaries
        """
        results = {}
        
        for expiration_date, options_data in options_data_dict.items():
            analyzer = cls(options_data)
            summary = analyzer.get_atm_greeks_summary(tolerance)
            if summary:
                results[expiration_date] = summary
        
        return results


def fetch_options_data(
    ticker: str,
    expiration_date: Optional[Union[str, datetime]] = None,
    nearest_expirations: Optional[int] = None
) -> Union[OptionsData, Dict[datetime, OptionsData]]:
    """
    Convenience function to fetch options data.
    
    Args:
        ticker (str): Stock ticker symbol
        expiration_date (Optional[Union[str, datetime]]): Specific expiration date
        nearest_expirations (Optional[int]): Number of nearest expirations to fetch
        
    Returns:
        Union[OptionsData, Dict[datetime, OptionsData]]: Options data
        
    Raises:
        ValueError: If parameters are invalid
    """
    fetcher = OptionsDataFetcher(ticker)
    
    if expiration_date:
        return fetcher.get_options_chain(expiration_date)
    elif nearest_expirations:
        return fetcher.get_nearest_expirations(nearest_expirations)
    else:
        # Return nearest 3 expirations by default
        return fetcher.get_nearest_expirations(3)


def analyze_options_volatility(
    options_data: Union[OptionsData, Dict[datetime, OptionsData]]
) -> Dict:
    """
    Analyze options data for volatility insights.
    
    Args:
        options_data (Union[OptionsData, Dict[datetime, OptionsData]]): Options data to analyze
        
    Returns:
        Dict: Analysis results
    """
    if isinstance(options_data, OptionsData):
        analyzer = OptionsAnalyzer(options_data)
        return analyzer.get_options_summary()
    else:
        # Multiple expiration dates
        results = {}
        for exp_date, data in options_data.items():
            analyzer = OptionsAnalyzer(data)
            results[exp_date] = analyzer.get_options_summary()
        return results 