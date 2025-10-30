from fastapi import APIRouter, Query
import pandas as pd
import numpy as np
import os
from pypfopt import EfficientFrontier, expected_returns, risk_models, objective_functions
from typing import List
from ..variables import END_REALDATA_DATE, INDUSTRIES_FILENAME, STOCKS_FILENAME, START_TRAINING_DATE, END_TRAINING_DATE, \
                        RISK_FREE_RATE
from ..utils import get_filtered_stocks, mega_clean_weights

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDUSTRIES_PATH = os.path.join(DATA_DIR, INDUSTRIES_FILENAME)
STOCKS_PATH = os.path.join(DATA_DIR, STOCKS_FILENAME)

industries_df = pd.read_excel(INDUSTRIES_PATH)
stocks_df = pd.read_csv(STOCKS_PATH, index_col=[0,1], parse_dates=[1])
stocks_df.index.names = ["symbol", "date"]

@router.get('/portfolio/max-sharpe')
def recommend_portfolio_max_sharpe(industries: List[str] = Query(...)):
    filtered = industries_df[industries_df['sector'].isin(industries)]
    symbols = filtered['Symbol'].dropna().unique().tolist()
    df_filtered = stocks_df.loc[stocks_df.index.get_level_values('symbol').isin(symbols)]
    training_data = df_filtered[(df_filtered.index.get_level_values('date') >= START_TRAINING_DATE) &
                               (df_filtered.index.get_level_values('date') <= END_TRAINING_DATE)]
    # Obtener datos filtrados con alrededor de 30 stocks
    df_final, final_symbols = get_filtered_stocks(training_data, industries, filtered)    
    prices = df_final.reset_index().pivot(index="date", columns="symbol", values="close")
    min_valid = len(prices) * 0.5
    prices_clean = prices.dropna(axis=1, thresh=min_valid)
    prices_clean = prices_clean.fillna(method='ffill').dropna()
    mu = expected_returns.mean_historical_return(prices_clean)
    S = risk_models.CovarianceShrinkage(prices_clean).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1), solver='ECOS')
    #ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance()
    mega_cleaned_weights = mega_clean_weights(weights)
    return {
        "weights": mega_cleaned_weights,
        "expected_annual_return": perf[0],
        "annual_volatility": perf[1],
        "sharpe_ratio": perf[2]
    }
