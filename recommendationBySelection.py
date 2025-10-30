# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, expected_returns, risk_models, objective_functions

# %%
# Load your Alpaca dataset
df = pd.read_csv("nasdaq_data_2020-2024.csv", index_col=[0,1], parse_dates=[1])
df.index.names = ["symbol", "date"]

industries = pd.read_excel("industries.xlsx")

# %%
selected_industries = ['Financial Services', 'Consumer Defensive', 'Industrials',
        'Technology', 'Healthcare', 'Consumer Cyclical', 'Energy',
        'Basic Materials', 'Real Estate', 'Communication Services',
        'Utilities']

selected_industries = ['Technology', 'Healthcare']

# %%
# Filter stocks based on selected industries

industries_filtered = industries[industries['sector'].isin(selected_industries)]
df_filtered = df.loc[df.index.get_level_values('symbol').isin(industries_filtered['Symbol'].values)]

print(len(df.index.get_level_values('symbol').unique()))
print(len(df_filtered.index.get_level_values('symbol').unique()))

# %%

START_TRAINING_DATE = '2020-01-01'
END_TRAINING_DATE = '2023-12-31'

# Filter the DataFrame by the timestamp index level using date strings
training_data = df_filtered[(df_filtered.index.get_level_values('date') >= START_TRAINING_DATE) &
				   (df_filtered.index.get_level_values('date') <= END_TRAINING_DATE)]

real_data = df_filtered[(df_filtered.index.get_level_values('date') > END_TRAINING_DATE)]

# Pivot so symbols are columns, rows are dates, values = close price
prices = training_data.reset_index().pivot(index="date", columns="symbol", values="close")


# %%
print("Available stocks in the dataset:")
print(f"Total number of stocks: {len(prices.columns)}")
print("\nAll stock symbols:")
for i, symbol in enumerate(prices.columns, 1):
  print(f"{i:4d}: {symbol}")

# %%
# Keep only the selected stocks
selected_symbols = "all"
#selected_symbols = ["AAPL","MSFT", "LAZR", "PEP" ]

# %%
prices

# %%
# Drop rows with missing values across all symbols (optional: you could ffill instead)

if selected_symbols != "all":
    # Allow a single symbol string or a list of symbols
    if isinstance(selected_symbols, str):
        selected_symbols = [selected_symbols]

    # Compute intersection between requested symbols and available columns to avoid KeyError
    available_cols = list(prices.columns)
    requested = list(selected_symbols)
    common = [s for s in requested if s in available_cols]
    missing = [s for s in requested if s not in available_cols]

    if len(common) == 0:
        raise KeyError(f"None of the requested symbols were found in prices.columns. Available columns (first 20): {available_cols[:20]}")

    if missing:
        print("Warning: The following requested symbols were not found and will be ignored:", missing)

    prices = prices[common].dropna()
    print("Use these symbols:", prices.columns.tolist())
else:
    print("Use all available symbols")
    prices = prices.dropna(how="all")
    print(prices.head())
    print(len(prices))

# %%
# # Compute daily returns
# stocks_lr = np.log(1 + prices.pct_change()).dropna()

# print("Stocks log returns shape:", stocks_lr.shape)
# # print("Any NaNs in stocks_lr?", stocks_lr.isna().any().any())
# # print("Any infinite values?", np.isinf(stocks_lr.values).any())
# # print("Min / Max / Mean / Std of log returns:")
# # print(stocks_lr.describe().T[['min', 'max', 'mean', 'std']])

# # Clip extreme returns first
# stocks_lr_safe = stocks_lr.clip(-0.2, 0.2)

# # Compute Ledoit-Wolf shrinkage
# cov = risk_models.CovarianceShrinkage(stocks_lr_safe).ledoit_wolf()

# # Force small eigenvalues to minimum threshold
# eigvals, eigvecs = np.linalg.eigh(cov)
# eigvals_clipped = np.clip(eigvals, 1e-6, None)
# mu = expected_returns.mean_historical_return(stocks_lr_safe)
# cov = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

# print("Stable condition number:", np.linalg.cond(cov))


# # cov = risk_models.CovarianceShrinkage(stocks_lr_safe).ledoit_wolf()
# # print("Condition number:", np.linalg.cond(cov))


# %%
def optimize_max_sharpe(prices, rf=0.02):
    """
    Optimize the maximum Sharpe portfolio given a prices DataFrame.
   
    Parameters
    ----------
    prices : pd.DataFrame
        Rows = dates, Columns = symbols, values = close prices.
    rf : float
        Risk-free rate for Sharpe ratio.
    
    Returns
    -------
    max_sharpe_w : np.ndarray
        Portfolio weights array aligned with prices columns.
    ef : EfficientFrontier object
        PyPortfolioOpt EfficientFrontier object.
    """

    max_assets = 1000

    if len(prices.columns) > max_assets:
        data_completeness = prices.notna().mean()
        top_assets = data_completeness.nlargest(max_assets).index
        _prices = prices[top_assets]
        print(f"Pre-filtered from {len(data_completeness)} to {max_assets} stocks")
        print("Selected stocks based on data completeness:")
        print(top_assets)
    else:
        _prices = prices.copy()

    # Drop columns with too many NaN values (keep if >50% valid data)
    min_valid = len(_prices) * 0.5
    prices_clean = _prices.dropna(axis=1, thresh=min_valid)
    
    # Forward fill then drop any remaining NaN rows
    prices_clean = prices_clean.fillna(method='ffill').dropna()
    
    # Ensure we have enough data points
    if len(prices_clean) < 200:
        raise ValueError("Insufficient data after cleaning")
    
    # Calculate expected returns
    mu = expected_returns.mean_historical_return(prices_clean)
    
    # Calculate covariance matrix with Ledoit-Wolf shrinkage for stability
    S = risk_models.CovarianceShrinkage(prices_clean).ledoit_wolf()
    
    # Initialize the Efficient Frontier with relaxed weight bounds
    ef = EfficientFrontier(mu, S, weight_bounds=(0.05, 1), solver='ECOS')
    
    # Add regularization to improve solver stability
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    try:
        # Optimize for maximum Sharpe ratio
        ef.max_sharpe(risk_free_rate=rf)
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Attempting with different solver settings...")
        
        # Try again with ECOS solver and higher iterations
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1), solver='ECOS')
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=rf)
    
    
    # Get the cleaned weights
    cleaned_weights = ef.clean_weights()
    
    # Convert to numpy array aligned with original prices columns
    max_sharpe_w = np.zeros(len(prices.columns))
    for i, symbol in enumerate(prices.columns):
        if symbol in cleaned_weights:
            max_sharpe_w[i] = cleaned_weights[symbol]
    
    return max_sharpe_w, ef

# %%


# Drop columns with too many NaN values (keep if >50% valid data)
min_valid = len(prices) * 0.5
prices_clean = prices.dropna(axis=1, thresh=min_valid)

# Forward fill then drop any remaining NaN rows
prices_clean = prices_clean.fillna(method='ffill').dropna()





mu = expected_returns.mean_historical_return(prices_clean)
S = risk_models.CovarianceShrinkage(prices_clean).ledoit_wolf()

# MINIMUM variance
ef_min = EfficientFrontier(mu, S)
ef_min.min_volatility()
min_return, min_vol, _ = ef_min.portfolio_performance()

print(f"Min Variance Portfolio:")
print(f"  Return: {min_return:.2%}")
print(f"  Volatility: {min_vol:.2%}")
print(f"  Variance: {min_vol**2:.4f}")

# MAXIMUM return (100% in highest expected return asset)
max_return_idx = mu.idxmax()
max_return = mu.max()
max_vol = np.sqrt(S.loc[max_return_idx, max_return_idx])

print(f"\nMax Return Portfolio (100% in {max_return_idx}):")
print(f"  Return: {max_return:.2%}")
print(f"  Volatility: {max_vol:.2%}")
print(f"  Variance: {max_vol**2:.4f}")

# %%
weights, ef_object = optimize_max_sharpe(prices, rf=0.02)
print("Portfolio weights:", weights)
print("Expected annual return:", ef_object.portfolio_performance()[0])
print("Annual volatility:", ef_object.portfolio_performance()[1])
print("Sharpe ratio:", ef_object.portfolio_performance()[2])

# %%
type(weights)

# %%
print("Weights MAX SHARPE (>1%):")
weights_series_sharpe = pd.Series(weights, index=prices.columns.to_list())

print(f"Total number of stocks in max sharpe portfolio: {len(weights_series_sharpe[weights_series_sharpe > 0.000001])}")
print(weights_series_sharpe[weights_series_sharpe > 0.000001].sort_values(ascending=False))

# print("Weights MIN VAR (>1%):")
# weights_series_minvar = pd.Series(special_ports["min_variance"][2], index=stocks_lr.columns)
# print(weights_series_minvar[weights_series_minvar > 0.01].sort_values(ascending=False))


# %%
import matplotlib.dates as mdates


def simulate_portfolio_performance(real_data, tickers, weights):
    # Debug: Check the date range of your input data using timestamp
    print("Input data date range (from timestamp):")
    if 'timestamp' in real_data.columns:
        timestamps = pd.to_datetime(real_data['timestamp'])
        print(f"Start: {timestamps.min()}")
        print(f"End: {timestamps.max()}")
        print(f"Total trading days: {len(timestamps.unique())}")
        
        # Set timestamp as index for proper date handling
        real_data_copy = real_data.copy()
        real_data_copy['timestamp'] = pd.to_datetime(real_data_copy['timestamp'])
        real_data_copy = real_data_copy.set_index(['timestamp', 'symbol'])
    else:
        print("No timestamp column found, using existing index")
        real_data_copy = real_data
        timestamps = real_data.index.get_level_values('date') if 'date' in real_data.index.names else real_data.index.get_level_values(0)
        print(f"Start: {timestamps.min()}")
        print(f"End: {timestamps.max()}")
    
    # 1. Pivot price data: Date x Ticker format
    price_df = real_data_copy['close'].unstack(level='symbol')
    
    # Ensure the index is datetime
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    
    # Debug: Check price_df date range
    print(f"\nPrice DataFrame date range:")
    print(f"Start: {price_df.index.min()}")
    print(f"End: {price_df.index.max()}")
    print(f"Shape: {price_df.shape}")
    
    # Show actual trading days by month
    monthly_counts = price_df.groupby(price_df.index.to_period('M')).size()
    print(f"\nActual trading days by month:")
    for period, count in monthly_counts.items():
        print(f"{period}: {count} days")
    
    # 2. Keep only tickers with sufficient data (e.g., at least 70% complete)
    missing_pct = price_df[tickers].isna().mean()
    valid_tickers = missing_pct < 0.3
    filtered_tickers = [t for t, keep in zip(tickers, valid_tickers) if keep]
    
    print(f"\nTicker data completeness:")
    for ticker in tickers[:10]:  # Show first 10 tickers
        if ticker in missing_pct:
            print(f"{ticker}: {(1-missing_pct[ticker])*100:.1f}% complete")
    
    price_df = price_df[filtered_tickers]
    
    print(f"\nFiltered tickers: {len(filtered_tickers)} out of {len(tickers)}")
    
    # 3. Forward-fill + back-fill missing values to retain full date coverage
    price_df = price_df.ffill().bfill()
    
    # Debug: Check for any remaining NaN values
    nan_count = price_df.isna().sum().sum()
    print(f"Remaining NaN values after ffill/bfill: {nan_count}")
    
    # 4. Calculate daily returns using actual trading days
    returns_df = price_df.pct_change().dropna()
    
    # Debug: Check returns_df date range
    print(f"\nReturns DataFrame date range:")
    print(f"Start: {returns_df.index.min()}")
    print(f"End: {returns_df.index.max()}")
    print(f"Shape: {returns_df.shape}")
    
    # Show returns by month
    monthly_returns = returns_df.groupby(returns_df.index.to_period('M')).size()
    print(f"\nReturns data by month:")
    for period, count in monthly_returns.items():
        print(f"{period}: {count} days")
    
    # 5. Filter weights to match valid tickers
    weights = np.array(weights)
    valid_indices = [i for i, t in enumerate(tickers) if t in filtered_tickers]
    weights = weights[valid_indices]
    weights /= weights.sum()  # re-normalize to sum to 1
    
    print(f"\nWeights shape: {weights.shape}")
    print(f"Weights sum: {weights.sum():.6f}")
    
    # 6. Compute portfolio returns using actual trading dates
    portfolio_returns = returns_df[filtered_tickers].dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Debug: Check final results date range
    print(f"\nFinal results date range:")
    print(f"Portfolio returns start: {portfolio_returns.index.min()}")
    print(f"Portfolio returns end: {portfolio_returns.index.max()}")
    print(f"Total portfolio return days: {len(portfolio_returns)}")
    
    # 7. Plot with proper date handling for trading days only
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio Value', linewidth=2)
    plt.title('Portfolio Performance on Real Data (Trading Days Only)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Improved x-axis formatting for trading days
    ax = plt.gca()
    
    # Set major ticks to months
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # Set minor ticks to weeks for better granularity
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    
    # Ensure all data is visible - use actual data range
    ax.set_xlim(cumulative_returns.index.min(), cumulative_returns.index.max())
    
    # Add some padding
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 0.98, y_max * 1.02)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Additional info about the final date range
    print(f"\nPlot date range:")
    print(f"X-axis start: {cumulative_returns.index.min()}")
    print(f"X-axis end: {cumulative_returns.index.max()}")
    print(f"Final cumulative return: {cumulative_returns.iloc[-1]:.4f}")
    
    return portfolio_returns, cumulative_returns


# %%
print("Portfolio returns and cumulative returns calculated successfully.")
print("--"*50)
print("Portfolio returns:")


portfolio_returns, cumulative_returns = simulate_portfolio_performance(
    real_data=real_data,
    tickers=prices.columns.to_list(),
    weights=weights
)



