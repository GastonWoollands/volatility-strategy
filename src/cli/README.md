# CLI Tools for Financial Volatility Analysis

This directory contains command-line interface tools for analyzing financial volatility, option pricing, and identifying trading opportunities.

## 📁 Available Tools

### 1. `volatility_comparison_cli.py` - Comprehensive Volatility Analysis
**Purpose**: Compare historical volatility, implied volatility, and model predictions for stocks and options.

**Features**:
- Single ticker analysis
- Multi-ticker comparison
- Quick analysis mode
- Options data integration
- MLflow model predictions

#### Usage Examples:
```bash
# Single ticker analysis
python src/cli/volatility_comparison_cli.py --ticker AAPL --start-date 2023-01-01

# Multiple tickers comparison
python src/cli/volatility_comparison_cli.py --multi-tickers AAPL,MSFT,GOOGL --start-date 2023-01-01

# Quick analysis (minimal output)
python src/cli/volatility_comparison_cli.py --ticker AAPL --quick

# Specific option expirations
python src/cli/volatility_comparison_cli.py --ticker AAPL --expirations 2024-01-19,2024-02-16

# Different output format
python src/cli/volatility_comparison_cli.py --ticker AAPL --output-format json
```

#### Parameters:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--ticker` | string | No* | - | Single ticker symbol (e.g., AAPL) |
| `--multi-tickers` | string | No* | - | Comma-separated list of tickers |
| `--quick` | flag | No* | - | Quick analysis mode |
| `--start-date` | date | No | 2023-01-01 | Start date (YYYY-MM-DD) |
| `--end-date` | date | No | today | End date (YYYY-MM-DD) |
| `--expirations` | string | No | - | Comma-separated option expiration dates |
| `--n-ahead` | int | No | 5 | Prediction horizon in days |
| `--mlflow-stage` | string | No | Staging | MLflow model stage (Staging/Production/Archived) |
| `--output-format` | string | No | both | Output format (table/json/both) |

*One of `--ticker`, `--multi-tickers`, or `--quick` is required.

---

### 2. `run_analysis.py` - Enhanced Opportunity Analysis
**Purpose**: Run enhanced volatility analysis with opportunity identification for trading decisions.

**Features**:
- Historical vs Implied volatility comparison
- Trading opportunity identification
- Model predictions integration
- User-friendly output with emojis

#### Usage Examples:
```bash
# Quick analysis for TSLA
python src/cli/run_analysis.py TSLA

# Analysis with specific expiration dates
python src/cli/run_analysis.py AAPL --expirations 2024-01-19 2024-02-16

# Analysis with custom date range
python src/cli/run_analysis.py NVDA --start-date 2023-06-01 --end-date 2024-01-01

# Analysis without model predictions
python src/cli/run_analysis.py TSLA --no-predictions
```

#### Parameters:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ticker` | string | Yes | - | Stock ticker symbol (e.g., TSLA) |
| `--start-date` | string | No | 2023-06-01 | Start date (YYYY-MM-DD) |
| `--end-date` | string | No | today | End date (YYYY-MM-DD) |
| `--expirations` | list | No | - | Specific option expiration dates |
| `--n-ahead` | int | No | 5 | Prediction horizon in days |
| `--mlflow-stage` | string | No | Staging | MLflow model stage |
| `--no-predictions` | flag | No | - | Exclude model predictions |

---

### 3. `option_pricing.py` - Black-Scholes Option Pricing
**Purpose**: Compare Black-Scholes theoretical prices with actual market prices for specific options.

**Features**:
- Direct option ticker analysis
- Black-Scholes price calculation
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Valuation assessment (Overvalued/Undervalued/Fair)
- Historical volatility integration

#### Usage Examples:
```bash
# Analyze specific option ticker
python src/cli/option_pricing.py AAPL240119C00185000

# Analyze with custom volatility
python src/cli/option_pricing.py TSLA240119P00200000 --volatility 0.25

# Use default volatility instead of historical
python src/cli/option_pricing.py NVDA240119C00400000 --no-historical-vol

# Save results to CSV
python src/cli/option_pricing.py AAPL240119C00185000 --save
```

#### Parameters:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `option_ticker` | string | Yes | - | Option ticker symbol (e.g., AAPL240119C00185000) |
| `--volatility` | float | No | - | Custom volatility (e.g., 0.25 for 25%) |
| `--no-historical-vol` | flag | No | - | Use default volatility instead of historical |
| `--save` | flag | No | - | Save results to CSV file |

#### Option Ticker Format:
```
SYMBOL + YYMMDD + C/P + STRIKE
Example: AAPL240119C00185000
- AAPL: Underlying stock
- 240119: Expiration date (Jan 19, 2024)
- C: Call option (P for Put)
- 00185000: Strike price ($185.000)
```

---

## 🚀 Quick Start Guide

### 1. Basic Volatility Analysis
```bash
# Analyze a single stock
python src/cli/volatility_comparison_cli.py --ticker AAPL

# Quick analysis
python src/cli/volatility_comparison_cli.py --ticker AAPL --quick
```

### 2. Trading Opportunities
```bash
# Find trading opportunities
python src/cli/run_analysis.py TSLA

# With specific options
python src/cli/run_analysis.py AAPL --expirations 2024-01-19 2024-02-16
```

### 3. Option Pricing Analysis
```bash
# Analyze a specific option
python src/cli/option_pricing.py AAPL240119C00185000

# With custom volatility
python src/cli/option_pricing.py TSLA240119P00200000 --volatility 0.30
```

## 📊 Output Examples

### Volatility Comparison Output:
```
Analysis Summary for AAPL
----------------------------------------
Analysis Date: 2024-01-15 14:30:25
Data Period: 2023-01-01 to 2024-01-15
Total Days: 379

Current Volatility Metrics:
  realized_volatility_5d: 0.0234
  realized_volatility_21d: 0.0187
  realized_volatility_63d: 0.0212

Options Analysis:
  Total Options Dates: 3
  Current ATM IV: 0.0245

Model Performance:
  MSE: 0.000123
  MAE: 0.008765
  RMSE: 0.011090
  Current Prediction: 0.0198
```

### Opportunity Analysis Output:
```
================================================================================
OPPORTUNITY ANALYSIS - TSLA
================================================================================
Analysis Date: 2024-01-15 14:30:25
Total Opportunities: 8

🏆 TOP 5 OPPORTUNITIES:
------------------------------------------------------------
1. CALL expiring 2024-01-19
   Strike: $245.00 | Underlying: $245.50
   IV: 35.2% | HV: 28.1%
   Spread: 7.1% (25.3%)
   Score: 50.6 | Confidence: MEDIUM

💡 TRADING RECOMMENDATIONS:
------------------------------------------------------------
• Strong SELL signal: call expiring 2024-01-19 (IV 35.2% vs HV 28.1%)
• Model confirms 3 opportunities with strong statistical backing
```

### Option Pricing Output:
```
================================================================================
OPTION PRICING ANALYSIS SUMMARY
================================================================================
📊 Option Details:
   Ticker: AAPL240119C00185000
   Underlying: AAPL
   Type: CALL
   Strike: $185.00
   Expiration: 2024-01-19 (15 days)
   Underlying Price: $185.50

💰 Pricing Analysis:
   Market Price: $2.45
   Black-Scholes Price: $1.89
   Difference: $0.56 (29.6%)
   Valuation: Overvalued

📈 Greeks:
   Delta: 0.5234
   Gamma: 0.0123
   Theta: -0.0456
   Vega: 0.1234
   Rho: 0.0234
```

## 📁 Output Files

All tools save results to the `output/` directory:

- **Volatility Comparison**: `output/volatility_comparison/`
- **Opportunity Analysis**: `output/opportunity_analysis/`
- **Option Pricing**: `output/option_pricing/`

## 🔧 Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, yfinance, scipy, scikit-learn
- MLflow (for model predictions)
- Internet connection (for data fetching)

## 📝 Notes

- All date formats should be YYYY-MM-DD
- Option tickers must follow the specified format
- Historical data is fetched from Yahoo Finance
- Model predictions require trained MLflow models
- Results are automatically saved with timestamps

## 🆘 Troubleshooting

### Common Issues:
1. **Invalid option ticker format**: Ensure format is SYMBOL+YYMMDD+C/P+STRIKE
2. **No data available**: Check if ticker exists and has options data
3. **Model prediction errors**: Ensure MLflow is running and models are available
4. **Date format errors**: Use YYYY-MM-DD format for all dates

### Getting Help:
- Use `--help` flag with any tool for detailed parameter information
- Check output directory for saved results
- Verify internet connection for data fetching
