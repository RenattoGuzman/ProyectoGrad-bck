from fastapi import APIRouter, Query
import pandas as pd
import os
from typing import List
from ..variables import INDUSTRIES_FILENAME, STOCKS_FILENAME

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDUSTRIES_PATH = os.path.join(DATA_DIR, INDUSTRIES_FILENAME)
STOCKS_PATH = os.path.join(DATA_DIR, STOCKS_FILENAME)

@router.get('/stocks-available')
def stocks_available(industries: List[str] = Query(...)):
    industries_df = pd.read_excel(INDUSTRIES_PATH)
    filtered = industries_df[industries_df['sector'].isin(industries)]
    stocks = filtered['Symbol'].dropna().unique().tolist()
    return {"stocks": stocks}
