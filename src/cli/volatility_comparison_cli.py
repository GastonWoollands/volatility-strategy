#!/usr/bin/env python3
"""
Command Line Interface for Volatility Comparison Analysis

Usage:
    python src/cli/volatility_comparison_cli.py --ticker AAPL --start-date 2023-01-01
    python src/cli/volatility_comparison_cli.py --ticker AAPL --multi-tickers AAPL,MSFT,GOOGL
    python src/cli/volatility_comparison_cli.py --ticker AAPL --expirations 2024-01-19,2024-02-16
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from src.analysis.volatility_comparison import run_volatility_comparison, VolatilityComparisonAnalyzer


def parse_date(date_str: str) -> str:
    """Parse and validate date string."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_expirations(expirations_str: str) -> List[str]:
    """Parse comma-separated expiration dates."""
    if not expirations_str:
        return None
    
    expirations = [date.strip() for date in expirations_str.split(',')]
    
    # Validate each date
    for date in expirations:
        parse_date(date)
    
    return expirations


def parse_tickers(tickers_str: str) -> List[str]:
    """Parse comma-separated ticker symbols."""
    if not tickers_str:
        return []
    
    return [ticker.strip().upper() for ticker in tickers_str.split(',')]


def run_single_analysis(args):
    """Run analysis for a single ticker."""
    print(f"Running volatility comparison analysis for {args.ticker}")
    print("=" * 60)
    
    try:
        results = run_volatility_comparison(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            expiration_dates=args.expirations,
            n_ahead=args.n_ahead,
            mlflow_stage=args.mlflow_stage,
            output_format=args.output_format
        )
        
        # Print summary
        summary = results['summary']
        print(f"\nAnalysis Summary for {args.ticker}")
        print("-" * 40)
        print(f"Analysis Date: {summary['analysis_date']}")
        print(f"Data Period: {summary['data_period']['start']} to {summary['data_period']['end']}")
        print(f"Total Days: {summary['data_period']['total_days']}")
        
        # Print current volatility metrics
        print(f"\nCurrent Volatility Metrics:")
        for vol_type, metrics in summary['volatility_comparison'].items():
            if metrics['current']:
                print(f"  {vol_type}: {metrics['current']:.4f}")
        
        # Print options analysis
        if summary['options_analysis']:
            print(f"\nOptions Analysis:")
            print(f"  Total Options Dates: {summary['options_analysis']['total_options_dates']}")
            if summary['options_analysis']['avg_atm_iv']['current']:
                print(f"  Current ATM IV: {summary['options_analysis']['avg_atm_iv']['current']:.4f}")
        
        # Print model performance
        if summary['model_performance']:
            print(f"\nModel Performance:")
            print(f"  MSE: {summary['model_performance']['mse']:.6f}")
            print(f"  MAE: {summary['model_performance']['mae']:.6f}")
            print(f"  RMSE: {summary['model_performance']['rmse']:.6f}")
            if summary['model_performance']['current_prediction']:
                print(f"  Current Prediction: {summary['model_performance']['current_prediction']:.4f}")
        
        print(f"\nResults saved to output/volatility_comparison/")
        return results
        
    except Exception as e:
        print(f"Error running analysis for {args.ticker}: {e}")
        return None


def run_multi_analysis(args):
    """Run analysis for multiple tickers."""
    tickers = parse_tickers(args.multi_tickers)
    
    if not tickers:
        print("No valid tickers provided")
        return
    
    print(f"Running volatility comparison analysis for {len(tickers)} tickers: {', '.join(tickers)}")
    print("=" * 60)
    
    all_results = {}
    comparison_data = []
    
    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            
            results = run_volatility_comparison(
                ticker=ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                expiration_dates=args.expirations,
                n_ahead=args.n_ahead,
                mlflow_stage=args.mlflow_stage,
                output_format="json"  # Only JSON for multiple tickers
            )
            
            all_results[ticker] = results['summary']
            
            # Create comparison row
            row = {'ticker': ticker}
            
            # Add realized volatility
            for vol_type in ['realized_volatility_5d', 'realized_volatility_21d', 'realized_volatility_63d']:
                if vol_type in results['summary']['volatility_comparison']:
                    row[f'{vol_type}_current'] = results['summary']['volatility_comparison'][vol_type].get('current')
            
            # Add options IV
            if results['summary']['options_analysis']:
                row['atm_iv_current'] = results['summary']['options_analysis']['avg_atm_iv'].get('current')
            
            # Add model prediction
            if results['summary']['model_performance']:
                row['model_prediction'] = results['summary']['model_performance'].get('current_prediction')
                row['model_rmse'] = results['summary']['model_performance'].get('rmse')
            
            comparison_data.append(row)
            
            # Print quick summary
            if results['summary']['volatility_comparison']:
                current_vol = results['summary']['volatility_comparison'].get('realized_volatility_21d', {}).get('current')
                if current_vol:
                    print(f"  Current 21d Volatility: {current_vol:.4f}")
            
            if results['summary']['options_analysis']:
                current_iv = results['summary']['options_analysis'].get('avg_atm_iv', {}).get('current')
                if current_iv:
                    print(f"  Current ATM IV: {current_iv:.4f}")
                    
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")
            continue
    
    # Create and display comparison table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n=== Multi-Ticker Comparison Summary ===")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"output/volatility_comparison/multi_ticker_comparison_{timestamp}.csv"
        comparison_df.to_csv(filename, index=False)
        print(f"\nMulti-ticker comparison saved to {filename}")
    
    return all_results


def run_quick_analysis(args):
    """Run quick analysis with minimal output."""
    print(f"Running quick volatility analysis for {args.ticker}")
    
    try:
        # Create analyzer
        analyzer = VolatilityComparisonAnalyzer(args.ticker)
        
        # Fetch historical data
        historical_data = analyzer.fetch_historical_data(args.start_date, args.end_date or datetime.now().strftime('%Y-%m-%d'))
        
        # Get current metrics
        current_date = historical_data.index[-1]
        current_price = historical_data.loc[current_date, 'close']
        current_vol_21d = historical_data.loc[current_date, 'realized_volatility_21d']
        
        print(f"\nQuick Analysis for {args.ticker} ({current_date.strftime('%Y-%m-%d')})")
        print("-" * 40)
        print(f"Current Price: ${current_price:.2f}")
        print(f"21-Day Realized Volatility: {current_vol_21d:.4f}")
        
        # Get options data
        try:
            options_data = analyzer.fetch_options_data()
            if options_data:
                # Get nearest expiration
                nearest_exp = min(options_data.keys())
                options_iv = analyzer.analyze_options_iv(options_data)
                
                if nearest_exp in options_iv:
                    iv_data = options_iv[nearest_exp]
                    print(f"Nearest Expiration: {iv_data['expiration_date']} ({iv_data['days_to_expiry']} days)")
                    print(f"ATM Implied Volatility: {iv_data['avg_atm_iv']:.4f}" if iv_data['avg_atm_iv'] else "ATM IV: N/A")
        except Exception as e:
            print(f"Options data: Not available ({e})")
        
        # Get model prediction
        try:
            model_predictions = analyzer.get_mlflow_predictions(historical_data, args.n_ahead, args.mlflow_stage)
            if not model_predictions.empty:
                current_prediction = model_predictions['predicted_vol'].iloc[-1]
                print(f"Model Prediction ({args.n_ahead}-day ahead): {current_prediction:.4f}")
        except Exception as e:
            print(f"Model prediction: Not available ({e})")
        
        return True
        
    except Exception as e:
        print(f"Error in quick analysis: {e}")
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Volatility Comparison Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single ticker analysis
  python volatility_comparison_cli.py --ticker AAPL --start-date 2023-01-01
  
  # Multiple tickers
  python volatility_comparison_cli.py --multi-tickers AAPL,MSFT,GOOGL --start-date 2023-01-01
  
  # Quick analysis
  python volatility_comparison_cli.py --ticker AAPL --quick
  
  # Specific expirations
  python volatility_comparison_cli.py --ticker AAPL --expirations 2024-01-19,2024-02-16
  
  # Different output format
  python volatility_comparison_cli.py --ticker AAPL --output-format json
        """
    )
    
    # Analysis type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ticker', type=str, help='Single ticker symbol')
    group.add_argument('--multi-tickers', type=str, help='Comma-separated list of ticker symbols')
    group.add_argument('--quick', action='store_true', help='Quick analysis mode')
    
    # Date parameters
    parser.add_argument('--start-date', type=parse_date, default='2023-01-01',
                       help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_date, 
                       help='End date for historical data (YYYY-MM-DD, defaults to today)')
    
    # Options parameters
    parser.add_argument('--expirations', type=parse_expirations,
                       help='Comma-separated list of options expiration dates (YYYY-MM-DD)')
    
    # Model parameters
    parser.add_argument('--n-ahead', type=int, default=5,
                       help='Number of periods ahead for model predictions (default: 5)')
    parser.add_argument('--mlflow-stage', type=str, default='Staging',
                       choices=['Staging', 'Production', 'Archived'],
                       help='MLflow model stage (default: Staging)')
    
    # Output parameters
    parser.add_argument('--output-format', type=str, default='both',
                       choices=['table', 'json', 'both'],
                       help='Output format (default: both)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate analysis
    if args.quick:
        run_quick_analysis(args)
    elif args.multi_tickers:
        run_multi_analysis(args)
    else:
        run_single_analysis(args)


if __name__ == "__main__":
    main() 