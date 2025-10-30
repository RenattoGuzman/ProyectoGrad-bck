from fastapi import APIRouter, Query
import pandas as pd
import numpy as np
import os
from pypfopt import EfficientFrontier, expected_returns, risk_models
from typing import List
from ..variables import END_REALDATA_DATE, INDUSTRIES_FILENAME, STOCKS_FILENAME, START_TRAINING_DATE, END_TRAINING_DATE, RISK_FREE_RATE
from ..utils import get_filtered_stocks

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDUSTRIES_PATH = os.path.join(DATA_DIR, INDUSTRIES_FILENAME)
STOCKS_PATH = os.path.join(DATA_DIR, STOCKS_FILENAME)

industries_df = pd.read_excel(INDUSTRIES_PATH)
stocks_df = pd.read_csv(STOCKS_PATH, index_col=[0,1], parse_dates=[1])
stocks_df.index.names = ["symbol", "date"]

@router.get('/portfolio/risk-ranges')
def portfolio_risk_ranges(industries: List[str] = Query(...)):
    filtered = industries_df[industries_df['sector'].isin(industries)]
    symbols = filtered['Symbol'].dropna().unique().tolist()

    df_filtered = stocks_df.loc[stocks_df.index.get_level_values('symbol').isin(symbols)]
    training_data = df_filtered[
        (df_filtered.index.get_level_values('date') >= START_TRAINING_DATE) &
        (df_filtered.index.get_level_values('date') <= END_TRAINING_DATE)
    ]

    df_final, final_symbols = get_filtered_stocks(training_data, industries, filtered)

    prices = df_final.reset_index().pivot(index="date", columns="symbol", values="close")
    min_valid = len(prices) * 0.5
    prices_clean = prices.dropna(axis=1, thresh=min_valid)
    prices_clean = prices_clean.fillna(method='ffill').dropna()

    mu = expected_returns.mean_historical_return(prices_clean)
    S = risk_models.CovarianceShrinkage(prices_clean).ledoit_wolf()

    # ---- Minimum volatility portfolio ----
    ef_min = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_min.min_volatility()
    _, min_volatility, _ = ef_min.portfolio_performance()


    # ---- Maximum Volatility portfolio ----
    ef_max = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_max.efficient_risk(target_volatility=50.0)  # Arbitrary high risk to get max volatility
    _, max_volatility, _ = ef_max.portfolio_performance()

    # ---- Sharpe ratio volatility ----
    ef_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_sharpe.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    _, sharpe_volatility, _ = ef_sharpe.portfolio_performance()

    return {
      "min_volatility": min_volatility,
      "max_volatility": max_volatility,
      "sharpe_ratio_volatility": sharpe_volatility,
      "last_available_date": prices_clean.index[-1].strftime("%Y-%m-%d")
    }
