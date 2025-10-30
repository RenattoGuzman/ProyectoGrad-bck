from fastapi import APIRouter, Query, Body
import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.stats import gaussian_kde
from ..variables import START_MONTECARLO_DATE, END_TRAINING_DATE, INDUSTRIES_FILENAME, STOCKS_FILENAME
from ..utils import get_filtered_stocks
import os

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
STOCKS_PATH = os.path.join(DATA_DIR, STOCKS_FILENAME)

stocks_df = pd.read_csv(STOCKS_PATH, index_col=[0,1], parse_dates=[1])
stocks_df.index.names = ["symbol", "date"]

@router.post("/simulation")
def mc_simulation(
    portfolio_weights: Dict[str, float] = Body(..., example={"AAPL":0.3,"MSFT":0.4,"GOOGL":0.3}),
):

    _stocks = stocks_df.copy()
    # print("Filtered symbols DONE")
    # ---- Filter training range ----
    training_data = _stocks[
        (_stocks.index.get_level_values('date') >= START_MONTECARLO_DATE) &
        (_stocks.index.get_level_values('date') <= END_TRAINING_DATE)
    ]

    # ---- Pivot to price matrix ----
    prices_full = training_data.reset_index().pivot(index="date", columns="symbol", values="close")
    # print(f"==>> prices_full:\n {prices_full.columns.tolist()}")
    
    portfolio_symbols = list(portfolio_weights.keys())
    prices = prices_full[portfolio_symbols].dropna()

    # print("PRICES DONE")

    # ---- Calculate returns ----
    daily_returns = prices.pct_change().dropna()
    mu = daily_returns.mean().values
    cov = daily_returns.cov().values
    w = np.array([portfolio_weights[s] for s in portfolio_symbols])

    # print("MU COV DONE")
    # ---- Monte Carlo Simulation ----
    num_simulations = 10000
    trading_days = 252
    results = np.zeros(num_simulations)
    sim_lines = []

    for i in range(100):  # Only store 100 lines for UI
        simulated_daily = np.random.multivariate_normal(mu, cov, trading_days)
        portfolio_daily = simulated_daily.dot(w)
        growth = np.cumprod(1 + portfolio_daily)
        sim_lines.append([{ "day": d+1, "value": float(v) } for d, v in enumerate(growth)])

    for i in range(num_simulations):
        simulated_daily = np.random.multivariate_normal(mu, cov, trading_days)
        results[i] = np.prod(1 + simulated_daily.dot(w)) - 1

    # print("SIMULATION DONE")

    # ---- Calculate mean and mode ----
    mean_return = float(results.mean())
    kde = gaussian_kde(results)
    x_vals = np.linspace(min(results), max(results), 500)
    mode_return = float(x_vals[np.argmax(kde(x_vals))])

    # ---- Construct mean and mode lines ----
    mean_growth = (1 + mean_return) ** (np.linspace(0, 1, trading_days))
    mode_growth = (1 + mode_return) ** (np.linspace(0, 1, trading_days))

    mean_line = [{"day": i+1, "value": float(v)} for i,v in enumerate(mean_growth)]
    mode_line = [{"day": i+1, "value": float(v)} for i,v in enumerate(mode_growth)]

    return {
        "simulation_lines": sim_lines,
        "mean_return_line": mean_line,
        "mode_return_line": mode_line,
        "mean_return": mean_return,
        "mode_return": mode_return
    }


@router.post('/backtesting')
def backtesting(portfolio_weights: Dict[str, float] = Body(..., example={"AAPL":0.3,"MSFT":0.4,"GOOGL":0.3}), start_date: str = Body(..., example="2024-01-01"), end_date: str = Body(..., example="2025-01-01")):

    # ---- Filter real data for backtesting ----
    _stocks = stocks_df.copy()
    real_data = _stocks[
        (_stocks.index.get_level_values('date') >= start_date) &
        (_stocks.index.get_level_values('date') <= end_date)
    ]

    # --- Extract tickers & weights from portfolio_weights dict ---
    tickers = list(portfolio_weights.keys())
    weights = np.array(list(portfolio_weights.values()))

    # --- Filter real data to only include selected tickers ---
    real_data = real_data.reset_index()
    real_data = real_data[real_data['symbol'].isin(tickers)]
    if real_data.empty:
        return {"error": "No price data found for the provided portfolio symbols."}

    # --- Pivot prices to Date x Ticker table ---
    price_df = real_data.pivot(index="date", columns="symbol", values="close").sort_index()

    # --- Clean missing data with forward/back fill ---
    price_df = price_df.ffill().bfill()

    # --- Calculate daily returns ---
    returns_df = price_df.pct_change().dropna()
    if returns_df.empty:
        return {"error": "Not enough price history to compute returns."}

    # --- Ensure weights align with available price data ---
    valid_tickers = [t for t in tickers if t in price_df.columns]
    if not valid_tickers:
        return {"error": "None of the tickers in weights match price data."}

    weights = np.array([portfolio_weights[t] for t in valid_tickers])
    weights = weights / weights.sum()

    # --- Portfolio returns ---
    portfolio_returns = returns_df[valid_tickers].dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # --- Build response JSON ---
    results = [
        {"date": str(date.date()), "value": float(value)}
        for date, value in cumulative_returns.items()
    ]

    return {
        "portfolio_growth": results,
        "final_return": float(cumulative_returns.iloc[-1]),
        "start_date": str(cumulative_returns.index.min().date()),
        "end_date": str(cumulative_returns.index.max().date())
    }