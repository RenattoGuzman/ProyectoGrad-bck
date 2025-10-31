from fastapi import APIRouter, Query
import pandas as pd
import os
from typing import List
from ..variables import INDUSTRIES_FILENAME, STOCKS_FILENAME

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDUSTRIES_PATH = os.path.join(DATA_DIR, INDUSTRIES_FILENAME)
STOCKS_PATH = os.path.join(DATA_DIR, STOCKS_FILENAME)


industries_df = pd.read_excel(INDUSTRIES_PATH)
stocks_df = pd.read_csv(STOCKS_PATH, index_col=[0,1], parse_dates=[1])
stocks_df.index.names = ["symbol", "date"]

@router.get('/stocks-available')
def stocks_available(industries: List[str] = Query(...)):
    
    filtered = industries_df[industries_df['sector'].isin(industries)]
    symbols = filtered['Symbol'].dropna().unique().tolist()
    df_filtered = stocks_df.loc[stocks_df.index.get_level_values('symbol').isin(symbols)]
    stocks = df_filtered.index.get_level_values('symbol').unique().tolist()
    
    return {"stocks": stocks}
