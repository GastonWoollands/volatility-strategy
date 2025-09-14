#!/usr/bin/env python3
"""
Black-Scholes Option Pricing Comparison Tool

This script compares Black-Scholes theoretical prices with actual market prices
to identify overvalued, undervalued, or fairly priced options.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.black_scholes import (
    black_scholes_call, 
    black_scholes_put, 
    get_risk_free_rate,
    calculate_greeks
)
from data.loader import fetch_ticker_data

def calculate_historical_volatility(ticker: str, days: int = 30) -> float:
    """
    Calculate historical volatility for the given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to look back for volatility calculation
        
    Returns:
        float: Annualized historical volatility
    """
    try:
        # Fetch historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')
        
        df = fetch_ticker_data(ticker, start=start_date, end=end_date, log_returns=True)
        
        if df.empty or 'log_returns' not in df.columns:
            return 0.2  # Default volatility if data unavailable
        
        # Calculate rolling volatility and take the most recent
        vol = df['log_returns'].rolling(min(days, len(df))).std().iloc[-1]
        annualized_vol = vol * np.sqrt(252)  # Annualize
        
        return annualized_vol if not np.isnan(annualized_vol) else 0.2
        
    except Exception as e:
        print(f"Warning: Could not calculate historical volatility for {ticker}: {e}")
        return 0.2  # Default volatility

def parse_option_ticker(option_ticker: str) -> tuple:
    """
    Parse option ticker to extract underlying stock and option details.
    
    Args:
        option_ticker (str): Option ticker (e.g., 'AAPL240119C00185000')
        
    Returns:
        tuple: (underlying_ticker, option_details)
    """
    import re
    from datetime import datetime
    
    # Pattern for option ticker: SYMBOL + YYMMDD + C/P + STRIKE
    # Example: AAPL240119C00185000 (AAPL + 240119 + C + 00185000)
    pattern = r'^([A-Z]+)(\d{6})([CP])(\d{8})$'
    match = re.match(pattern, option_ticker.upper())
    
    if not match:
        raise ValueError(f"Invalid option ticker format: {option_ticker}")
    
    underlying_ticker, date_str, option_type, strike_str = match.groups()
    
    # Parse date (YYMMDD format)
    year = 2000 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    expiration_date = datetime(year, month, day)
    
    # Parse strike price (remove leading zeros and convert to float)
    strike = float(strike_str) / 1000  # Assuming 3 decimal places
    
    return underlying_ticker, {
        'type': 'call' if option_type == 'C' else 'put',
        'strike': strike,
        'expiration': expiration_date
    }

def analyze_option_pricing(option_ticker: str, 
                          use_historical_vol: bool = True,
                          custom_vol: float = None) -> pd.DataFrame:
    """
    Analyze option pricing for a specific option ticker from Yahoo Finance.
    
    Args:
        option_ticker (str): Option ticker symbol (e.g., 'AAPL240119C00185000')
        use_historical_vol (bool): Whether to use historical volatility
        custom_vol (float): Custom volatility to use (overrides historical)
        
    Returns:
        pd.DataFrame: Analysis results
    """
    print(f"🔍 Analyzing option pricing for {option_ticker}")
    
    # Parse option ticker to extract underlying stock and option details
    try:
        underlying_ticker, option_details = parse_option_ticker(option_ticker)
    except Exception as e:
        print(f"❌ Error parsing option ticker: {e}")
        return pd.DataFrame()
    
    # Get option data from Yahoo Finance
    try:
        import yfinance as yf
        option_obj = yf.Ticker(option_ticker)
        option_info = option_obj.info
        
        if not option_info or 'regularMarketPrice' not in option_info:
            print(f"❌ Could not fetch option data for {option_ticker}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching option data: {e}")
        return pd.DataFrame()
    
    # Extract option parameters
    S = option_info.get('underlyingPrice', 0)
    K = option_details['strike']
    option_type = option_details['type']
    expiration_date = option_details['expiration']
    
    # Calculate time to expiration
    days_to_expiry = (expiration_date - datetime.now()).days
    T = days_to_expiry / 365.25
    
    if T <= 0:
        print(f"❌ Option has expired or expires today")
        return pd.DataFrame()
    
    # Get risk-free rate
    r = get_risk_free_rate()
    
    # Determine volatility to use
    if custom_vol:
        sigma = custom_vol
        vol_source = "custom"
    elif use_historical_vol:
        sigma = calculate_historical_volatility(underlying_ticker, days=min(30, days_to_expiry))
        vol_source = "historical"
    else:
        # Use implied volatility from market prices
        sigma = 0.2  # Default
        vol_source = "default"
    
    print(f"📊 Using {vol_source} volatility: {sigma:.1%}")
    print(f"💰 Underlying price: ${S:.2f}")
    print(f"⏰ Time to expiration: {days_to_expiry} days ({T:.3f} years)")
    print(f"📈 Risk-free rate: {r:.1%}")
    
    # Get market price
    market_price = option_info.get('regularMarketPrice', 0)
    bid = option_info.get('bid', 0)
    ask = option_info.get('ask', 0)
    volume = option_info.get('volume', 0)
    open_interest = option_info.get('openInterest', 0)
    
    if market_price <= 0:
        print(f"❌ Invalid market price: ${market_price}")
        return pd.DataFrame()
    
    # Calculate Black-Scholes price
    if option_type.upper() == 'CALL':
        bs_price = black_scholes_call(S, K, T, r, sigma)
    else:  # PUT
        bs_price = black_scholes_put(S, K, T, r, sigma)
    
    # Calculate Greeks
    greeks = calculate_greeks(S, K, T, r, sigma, option_type.lower())
    
    # Calculate pricing metrics
    price_diff = market_price - bs_price
    price_diff_pct = (price_diff / bs_price) * 100 if bs_price > 0 else 0
    
    # Determine valuation
    if abs(price_diff_pct) < 5:
        valuation = "Fair"
    elif price_diff_pct > 5:
        valuation = "Overvalued"
    else:
        valuation = "Undervalued"
    
    # Create results
    results = [{
        'option_ticker': option_ticker,
        'underlying_ticker': underlying_ticker,
        'option_type': option_type.upper(),
        'strike': K,
        'expiration_date': expiration_date.strftime('%Y-%m-%d'),
        'days_to_expiry': days_to_expiry,
        'underlying_price': S,
        'market_price': market_price,
        'bs_price': bs_price,
        'price_difference': price_diff,
        'price_difference_pct': price_diff_pct,
        'valuation': valuation,
        'volatility_used': sigma,
        'volatility_source': vol_source,
        'delta': greeks['delta'],
        'gamma': greeks['gamma'],
        'theta': greeks['theta'],
        'vega': greeks['vega'],
        'rho': greeks['rho'],
        'bid': bid,
        'ask': ask,
        'volume': volume,
        'open_interest': open_interest
    }]
    
    return pd.DataFrame(results)

def print_analysis_summary(df: pd.DataFrame):
    """Print a summary of the analysis results."""
    if df.empty:
        print("❌ No options data to analyze")
        return
    
    print(f"\n{'='*80}")
    print(f"OPTION PRICING ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Get the single option data
    opt = df.iloc[0]
    
    print(f"📊 Option Details:")
    print(f"   Ticker: {opt['option_ticker']}")
    print(f"   Underlying: {opt['underlying_ticker']}")
    print(f"   Type: {opt['option_type']}")
    print(f"   Strike: ${opt['strike']:.2f}")
    print(f"   Expiration: {opt['expiration_date']} ({opt['days_to_expiry']} days)")
    print(f"   Underlying Price: ${opt['underlying_price']:.2f}")
    
    print(f"\n💰 Pricing Analysis:")
    print(f"   Market Price: ${opt['market_price']:.2f}")
    print(f"   Black-Scholes Price: ${opt['bs_price']:.2f}")
    print(f"   Difference: ${opt['price_difference']:.2f} ({opt['price_difference_pct']:.1f}%)")
    print(f"   Valuation: {opt['valuation']}")
    
    print(f"\n📈 Greeks:")
    print(f"   Delta: {opt['delta']:.4f}")
    print(f"   Gamma: {opt['gamma']:.4f}")
    print(f"   Theta: {opt['theta']:.4f}")
    print(f"   Vega: {opt['vega']:.4f}")
    print(f"   Rho: {opt['rho']:.4f}")
    
    print(f"\n📊 Market Data:")
    print(f"   Bid: ${opt['bid']:.2f}")
    print(f"   Ask: ${opt['ask']:.2f}")
    print(f"   Volume: {opt['volume']:,}")
    print(f"   Open Interest: {opt['open_interest']:,}")
    
    print(f"\n🔍 Volatility Analysis:")
    print(f"   Source: {opt['volatility_source']}")
    print(f"   Value: {opt['volatility_used']:.1%}")

def save_results(df: pd.DataFrame, ticker: str, output_dir: str = "output/option_pricing"):
    """Save analysis results to CSV file."""
    if df.empty:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{ticker}_option_pricing_{timestamp}.csv"
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    print(f"\n💾 Results saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(
        description="Black-Scholes Option Pricing Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific option ticker
  python src/cli/option_pricing.py AAPL240119C00185000
  
  # Analyze with custom volatility
  python src/cli/option_pricing.py TSLA240119P00200000 --volatility 0.25
  
  # Use default volatility instead of historical
  python src/cli/option_pricing.py NVDA240119C00400000 --no-historical-vol
  
  # Save results to CSV
  python src/cli/option_pricing.py AAPL240119C00185000 --save

Option Ticker Format:
  SYMBOL + YYMMDD + C/P + STRIKE
  Example: AAPL240119C00185000
  - AAPL: Underlying stock
  - 240119: Expiration date (Jan 19, 2024)
  - C: Call option (P for Put)
  - 00185000: Strike price ($185.000)
        """
    )
    
    parser.add_argument('option_ticker', help='Option ticker symbol (e.g., AAPL240119C00185000)')
    parser.add_argument('--volatility', type=float, help='Custom volatility to use (e.g., 0.25 for 25%)')
    parser.add_argument('--no-historical-vol', action='store_true',
                       help='Use default volatility instead of historical')
    parser.add_argument('--save', action='store_true', help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Determine volatility settings
    use_historical_vol = not args.no_historical_vol
    custom_vol = args.volatility
    
    print(f"🚀 Starting Option Pricing Analysis")
    print(f" Option Ticker: {args.option_ticker}")
    print(f" Volatility: {'Custom' if custom_vol else 'Historical' if use_historical_vol else 'Default'}")
    if custom_vol:
        print(f"   Custom Value: {custom_vol:.1%}")
    
    print("\n" + "="*60)
    
    try:
        # Run analysis
        results_df = analyze_option_pricing(
            option_ticker=args.option_ticker,
            use_historical_vol=use_historical_vol,
            custom_vol=custom_vol
        )
        
        if not results_df.empty:
            # Print summary
            print_analysis_summary(results_df)
            
            # Save results if requested
            if args.save:
                save_results(results_df, args.option_ticker)
            
            print(f"\n✅ Analysis completed successfully!")
        else:
            print(f"\n❌ No option data found for {args.option_ticker}")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()