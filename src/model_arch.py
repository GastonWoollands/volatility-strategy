import numpy as np
import pandas as pd
from tqdm import tqdm
from arch import arch_model

#------------------------------------------------------------------------------------------------

def train_garch_iter(data, p_values: list, q_values: list, means: list, distributions: list, vols: list, rescale: bool=False):
    results_list = []
    total_iterations = len(p_values) * len(q_values) * len(means) * len(distributions) * len(vols)
    
    with tqdm(total=total_iterations, desc='Fitting GARCH Models') as pbar:
        for p in p_values:
            for q in q_values:
                for mean in means:
                    for vol in vols:
                        for dist in distributions:
                            model = arch_model(data, mean=mean, vol=vol, p=p, q=q, dist=dist, rescale=rescale)
                            res = model.fit(disp='off')
                            
                            results_list.append([
                                f'GARCH({p},{q})', p, q, mean, dist, vol, res.aic, res.bic])
                            
                            pbar.update(1)
    
    return pd.DataFrame(results_list, columns=['model', 'p', 'q', 'm', 'd', 'v', 'aic', 'bic'])

#------------------------------------------------------------------------------------------------

def get_best_model(results):
    best_model_aic = results.loc[results['aic'].idxmin()]
    best_model_bic = results.loc[results['bic'].idxmin()]
    return best_model_aic, best_model_bic

#------------------------------------------------------------------------------------------------

def get_best_model_params(best_model):
    p = best_model.p.item()
    q = best_model.q.item()
    m = best_model.m
    d = best_model.d
    v = best_model.v
    return p, q, m, d, v

#------------------------------------------------------------------------------------------------

def train_garch(data: pd.DataFrame, p:int=1, q:int=1, mean:str="Zero", dist:str="t", vol:str="Garch"):
    """Train GARCH model
    Args:
        - data: data to train model
    Returns:
        - res_garch: GARCH model results
    """
    data.dropna(inplace=True)

    garch_mod = arch_model(data, mean=mean, vol=vol, p=p, q=q, dist=dist, rescale=False)
    res_garch = garch_mod.fit(disp="off")
    
    return res_garch

#------------------------------------------------------------------------------------------------

def garch_forecast(res_garch, horizon=30):
    """Forecast of future volatility with GARCH
    Args:
        - res_garch: GARCH model results
        - horizon: number of days to forecast
    Returns:
        - forecast: GARCH forecast results
    """
    forecast = res_garch.forecast(horizon=horizon)
    return forecast

#------------------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import chi2

def mcleod_li_test(residuals, k):
    """
    Calculates the McLeod-Li test statistic for a time series with k lags.
    Returns the test statistic and its p-value.
    
    H0: No significant autocorrelation in the squared residuals --> Homocedasticity.
    Args:
        - residuals: residuals serie
        - k (int): lags
    Return:
        - stat
        - p-value
    """
    n = len(residuals)
    residuals_sq = residuals ** 2

    x_sum = np.sum(residuals_sq)
    x_lag_sum = np.sum(residuals_sq[k:])  # sum from index k to n

    test_stat = (n * (n + 2) * x_lag_sum) / (x_sum ** 2)
    
    df = k
    
    p_value = 1 - chi2.cdf(test_stat, df)
    
    return test_stat, p_value
