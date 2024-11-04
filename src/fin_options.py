import numpy as np
from scipy import stats, optimize
from scipy.stats import norm

#------------------------------------------------------------------------------------------------

def option_europ_bs(op_type: str, S: float, K: float, T: float, r: float, sigma: float, div: float = 0.0):
    """Option price with Black-Scholes model
    Args:
        - op_type: option type
        - S: spot price
        - K: strike price
        - T: time to maturity
        - r: risk-free rate
        - sigma: volatility
        - div: dividend rate
    Returns:
        - option price
    """
    d1 = (np.log(S / K) + (r - div + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if op_type.upper() == "C":  # Call
        precio_BS = S * np.exp(-div * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    elif op_type.upper() == "P":  # Put
        precio_BS = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-div * T) * stats.norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'C' for Call or 'P' for Put")
    
    return precio_BS

#------------------------------------------------------------------------------------------------

def implied_volatility(S, K, T, r, market_price, op_type:str="C", periods:int=252):
    """Implied volatility
    Args:
        - S: spot price
        - K: strike price
        - T: time to maturity
        - r: risk-free rate
        - market_price: market price of the option
        - op_type: option type
    Returns:
        - implied volatility
    """
    def error_function(sigma):
        return option_europ_bs(op_type, S, K, T, r, sigma) - market_price

    result = optimize.root_scalar(error_function, bracket=[0.01, 5.0], method='bisect')
    return result.root / np.sqrt(periods) if result.converged else np.nan

#------------------------------------------------------------------------------------------------

def get_greeks(op_type: str, S: float, K: float, T: float, r: float, sigma: float, div: float = 0.0):
    """Calculate the Greeks of an option.
    
    Args:
        op_type (str): Option type ("C" for Call, "P" for Put).
        S (float): Spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate (as a decimal).
        sigma (float): Volatility of the underlying asset (as a decimal).
        div (float, optional): Dividend yield (default is 0.0).

    Returns:
        tuple: (delta, gamma, vega, theta, rho)
    """

    if S <= 0 or K <= 0 or T <= 0 or sigma < 0:
        raise ValueError("S, K, T must be positive and sigma must be non-negative.")

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    common_gamma = norm.pdf(d1) / (S * sigma * sqrt_T)  # Common gamma calculation
    common_vega = S * norm.pdf(d1) * sqrt_T / 100  # Common vega calculation

    if op_type.upper() == "C":  # Call option
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        
    elif op_type.upper() == "P":  # Put option
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")
    
    return delta, common_gamma, common_vega, theta, rho

#------------------------------------------------------------------------------------------------