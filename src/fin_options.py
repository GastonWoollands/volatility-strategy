import numpy as np
from scipy import stats, optimize

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
