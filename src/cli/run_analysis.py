#!/usr/bin/env python3
"""
Simple script to run volatility opportunity analysis from terminal
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.volatility_comparison import run_enhanced_volatility_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Run Volatility Opportunity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis for TSLA
  python src/cli/run_analysis.py TSLA
  
  # Analysis with specific expiration dates
  python src/cli/run_analysis.py AAPL --expirations 2024-01-19 2024-02-16
  
  # Analysis with custom date range
  python src/cli/run_analysis.py NVDA --start-date 2023-06-01 --end-date 2024-01-01
  
  # Analysis without model predictions
  python src/cli/run_analysis.py TSLA --no-predictions
        """
    )
    
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., TSLA)')
    parser.add_argument('--start-date', default='2023-06-01', 
                       help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None,
                       help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--expirations', nargs='+', default=None,
                       help='Specific option expiration dates (YYYY-MM-DD)')
    parser.add_argument('--n-ahead', type=int, default=5,
                       help='Prediction horizon in days')
    parser.add_argument('--mlflow-stage', default='Staging',
                       help='MLflow model stage to use')
    parser.add_argument('--no-predictions', action='store_true',
                       help='Exclude model predictions from analysis')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Volatility Analysis for {args.ticker}")
    print(f"📅 Date Range: {args.start_date} to {args.end_date or 'today'}")
    print(f" Prediction Horizon: {args.n_ahead} days")
    print(f"📊 Include Predictions: {not args.no_predictions}")
    
    if args.expirations:
        print(f"📋 Specific Expirations: {', '.join(args.expirations)}")
    
    print("\n" + "="*60)
    
    try:
        # Run the enhanced analysis
        results = run_enhanced_volatility_analysis(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            expiration_dates=args.expirations,
            n_ahead=args.n_ahead,
            mlflow_stage=args.mlflow_stage,
            include_predictions=not args.no_predictions,
            print_summary=True
        )
        
        print(f"\n✅ Analysis completed successfully!")
        
        # Show quick summary of results
        opportunities_df = results['opportunity_analysis']['opportunities_df']
        if not opportunities_df.empty:
            high_conf = opportunities_df[opportunities_df['confidence_level'] == 'high']
            print(f"📈 High confidence opportunities: {len(high_conf)}")
            print(f"🎯 Total opportunities analyzed: {len(opportunities_df)}")
            
            if not opportunities_df.empty:
                best_opp = opportunities_df.loc[opportunities_df['opportunity_score'].idxmax()]
                print(f"🏆 Best opportunity: {best_opp['option_type'].upper()} expiring {best_opp['expiration_date']}")
                print(f"   Score: {best_opp['opportunity_score']:.1f} | Confidence: {best_opp['confidence_level'].upper()}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 