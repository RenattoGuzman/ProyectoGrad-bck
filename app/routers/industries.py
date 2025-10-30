from fastapi import APIRouter
import pandas as pd
import os

router = APIRouter()
from ..variables import INDUSTRIES_FILENAME, STOCKS_FILENAME

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDUSTRIES_PATH = os.path.join(DATA_DIR, INDUSTRIES_FILENAME)

@router.get('/industries-available')
def industries_available():
    industries = pd.read_excel(INDUSTRIES_PATH)
    sectors = industries['sector'].dropna().unique().tolist()
    return {"industries": sectors}
