"""
Macro Regime Tracker — Data Fetcher
Fetches macro data from FRED, yfinance, US Treasury, and SNB.
Writes macro_data.json for the dashboard.

Usage:
  export FRED_API_KEY="your_key_here"
  python fetch_macro_data.py
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_PATH          = Path(__file__).parent / "macro_data.json"
FRED_API_KEY         = os.environ.get("FRED_API_KEY", "")
LOOKBACK_YEARS       = 5
ROC_WINDOW_MONTHS    = 3
ACCEL_THRESHOLD      = 0.005   # > +0.5% relative change → accelerating
DECEL_THRESHOLD      = -0.005  # < -0.5% relative change → decelerating
SPARKLINE_POINTS     = 6       # monthly data points for sparkline

# FRED series IDs
FRED_CPI      = "CPIAUCSL"
FRED_PCE      = "PCEPI"
FRED_M2       = "M2SL"
FRED_FED_BS   = "WALCL"
FRED_REAL_YLD = "DFII10"
FRED_HY_OAS   = "BAMLH0A0HYM2"
FRED_IG_OAS   = "BAMLC0A0CM"
FRED_2S10S    = "T10Y2Y"
FRED_US10Y    = "DGS10"
FRED_BUND     = "IRLTLT01DEM156N"
FRED_FED_RATE = "DFEDTARU"

# yfinance tickers
YF_GOLD    = "GC=F"
YF_DXY     = "DX-Y.NYB"
YF_SPX     = "^GSPC"
YF_VIX     = "^VIX"
YF_USDCHF  = "USDCHF=X"
YF_GBP10Y  = "^GUKG10"

# Treasury XML namespaces
TREASURY_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
    'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
}
TREASURY_MATURITIES = {
    '1M': 'BC_1MONTH', '3M': 'BC_3MONTH', '6M': 'BC_6MONTH',
    '1Y': 'BC_1YEAR',  '2Y': 'BC_2YEAR',  '3Y': 'BC_3YEAR',
    '5Y': 'BC_5YEAR',  '7Y': 'BC_7YEAR',  '10Y': 'BC_10YEAR',
    '20Y': 'BC_20YEAR','30Y': 'BC_30YEAR',
}

# Deployment radar signal thresholds
RADAR_THRESHOLDS = {
    "hy_spread":     {"low": 300, "high": 500},   # bp
    "ig_spread":     {"low": 80,  "high": 150},   # bp
    "vix":           {"low": 20,  "high": 30},
    "spx_vs_200ma":  {"low": -5,  "high": 0},     # % below 200MA → concern
}

# ─── Shared Utilities ────────────────────────────────────────────────────────

def sanitize(obj):
    """Recursively replace NaN/Infinity with None for valid JSON."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def safe_download(ticker, **kwargs):
    """yfinance download with 3 retries and MultiIndex column flattening."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, progress=False, auto_adjust=True, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            if attempt == 2:
                return pd.DataFrame()
    return pd.DataFrame()
