import pandas as pd
import numpy as np

from .variables import RISK_FREE_RATE


def get_filtered_stocks(training_data, selected_industries, industries_filtered):
  # --- Calculate daily returns ---
  returns = training_data['close'].unstack('symbol').pct_change()

  # --- Compute Sharpe Ratio (using daily data, annualized with 252 trading days) ---
  sharpe = (returns.mean()) / returns.std() * np.sqrt(252)

  # --- Compute 6-month momentum (126 trading days) ---
  momentum = training_data['close'].unstack('symbol').pct_change(126).iloc[-1]

  # --- Combine into score ---
  score_df = pd.DataFrame({
      "SharpeRatio": sharpe,
      "Momentum6m": momentum
  })
  score_df["Score"] = 0.6 * score_df["SharpeRatio"] + 0.4 * score_df["Momentum6m"]
  score_df = score_df.dropna()

  # --- Select top N per sector ---
  num_sectors = len(selected_industries)
  if num_sectors == 1:
      top_per_sector = 30
  else:
      top_per_sector = max(1, 30 // num_sectors)  # split 30 equally if multiple

  final_symbols = (
      industries_filtered
      .merge(score_df[["Score"]], left_on="Symbol", right_index=True)
      .sort_values("Score", ascending=False)
      .groupby("sector")
      .head(top_per_sector)["Symbol"]
      .tolist()
  )

  # --- Filter final price dataset ---
  df_final = training_data.loc[training_data.index.get_level_values("symbol").isin(final_symbols)]

  return df_final, final_symbols

def mega_clean_weights(weights):
  """Cleans portfolio weights by removing negligible allocations, renormalizing,
  and ordering by descending weight.

  Args:
    weights (dict): Dictionary of asset weights.

  Returns:
    dict: Cleaned, renormalized, and sorted (largest first) weights.
  """
  # Remove negligible weights
  cleaned = {k: v for k, v in weights.items() if abs(v) > 1e-5}
  total = sum(cleaned.values())
  # Renormalize if possible
  if total > 0:
    cleaned = {k: v / total for k, v in cleaned.items()}
  # Order by bigger weights first (descending)
  cleaned = dict(sorted(cleaned.items(), key=lambda item: item[1], reverse=True))
  return cleaned