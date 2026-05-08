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


# ─── Regime Indicators ───────────────────────────────────────────────────────

def compute_direction(series: list, window_months: int = None,
                      accel: float = None, decel: float = None) -> str:
    """Rate-of-change threshold direction badge for a monthly time series."""
    w = window_months or ROC_WINDOW_MONTHS
    a = accel if accel is not None else ACCEL_THRESHOLD
    d = decel if decel is not None else DECEL_THRESHOLD
    clean = [v for v in series if v is not None]
    if len(clean) < w + 1:
        return "stable"
    latest = clean[-1]
    prior  = clean[-(w + 1)]
    if prior == 0:
        return "stable"
    roc = (latest - prior) / abs(prior)
    if roc > a:
        return "accelerating"
    if roc < d:
        return "decelerating"
    return "stable"


def build_sparkline(values: list, n_points: int = None) -> list:
    """Return last n non-None values for sparkline rendering."""
    n = n_points or SPARKLINE_POINTS
    clean = [v for v in values if v is not None]
    return clean[-n:] if len(clean) >= n else clean


def assign_radar_signal(metric: str, value) -> str:
    """Map a metric value to low / neutral / elevated signal string."""
    if value is None or metric not in RADAR_THRESHOLDS:
        return "neutral"
    t = RADAR_THRESHOLDS[metric]
    if value < t["low"]:
        return "low"
    if value > t["high"]:
        return "elevated"
    return "neutral"


# ─── FRED Fetcher ────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, lookback_days: int = 365) -> list:
    """Fetch FRED observations. Returns list of {date, value} dicts, desc order."""
    if not FRED_API_KEY:
        log.warning(f"No FRED_API_KEY — skipping {series_id}")
        return []
    try:
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}"
            f"&observation_start={start}&observation_end={end}"
            f"&file_type=json&sort_order=desc&limit=2000"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        return [{"date": o["date"], "value": o["value"]}
                for o in obs if o["value"] != "."]
    except Exception as e:
        log.error(f"FRED fetch failed for {series_id}: {e}")
        return []


def fetch_fred_latest(series_id: str) -> float | None:
    """Return the most recent numeric value from a FRED series."""
    obs = fetch_fred_series(series_id, lookback_days=90)
    if obs:
        try:
            return round(float(obs[0]["value"]), 4)
        except (ValueError, IndexError):
            return None
    return None


def fetch_fred_history(series_id: str, lookback_days: int) -> list[float]:
    """Return chronological list of floats (ascending date) for time series use."""
    obs = fetch_fred_series(series_id, lookback_days=lookback_days)
    values = []
    for o in reversed(obs):
        try:
            values.append(float(o["value"]))
        except ValueError:
            pass
    return values


def fred_yoy_series(series_id: str, lookback_days: int = 365 * 6) -> list[float]:
    """
    Compute YoY % change series for a monthly FRED index (e.g. CPI, PCE, M2).
    Returns chronological list of YoY values (length = raw_length - 12).
    """
    obs = fetch_fred_series(series_id, lookback_days=lookback_days)
    if len(obs) < 13:
        return []
    # obs is desc; reverse to ascending
    asc = list(reversed(obs))
    yoy = []
    for i in range(12, len(asc)):
        try:
            current = float(asc[i]["value"])
            prior   = float(asc[i - 12]["value"])
            if prior != 0:
                yoy.append(round((current / prior - 1) * 100, 4))
        except (ValueError, ZeroDivisionError):
            pass
    return yoy


def build_macro_regime() -> dict:
    """Fetch and compute the Macro Regime panel data."""
    lookback = LOOKBACK_YEARS * 365

    # CPI YoY
    cpi_yoy = fred_yoy_series(FRED_CPI, lookback_days=lookback + 365)
    cpi_yoy_latest = cpi_yoy[-1] if cpi_yoy else None

    # PCE YoY
    pce_yoy = fred_yoy_series(FRED_PCE, lookback_days=lookback + 365)
    pce_yoy_latest = pce_yoy[-1] if pce_yoy else None

    # M2 YoY
    m2_yoy = fred_yoy_series(FRED_M2, lookback_days=lookback + 365)
    m2_yoy_latest = m2_yoy[-1] if m2_yoy else None

    # Fed Balance Sheet (WALCL, in millions → report in $T)
    fed_bs_raw = fetch_fred_history(FRED_FED_BS, lookback_days=lookback)
    fed_bs_t   = [round(v / 1_000_000, 3) for v in fed_bs_raw]
    fed_bs_latest = fed_bs_t[-1] if fed_bs_t else None

    # Real Yield (DFII10, already a %)
    real_yld_raw = fetch_fred_history(FRED_REAL_YLD, lookback_days=lookback)
    real_yld_latest = real_yld_raw[-1] if real_yld_raw else None

    return {
        "inflation": {
            "value":     cpi_yoy_latest,
            "direction": compute_direction(cpi_yoy),
            "sparkline": build_sparkline(cpi_yoy),
            "series":    "CPI YoY %",
        },
        "pce": {
            "value":     pce_yoy_latest,
            "direction": compute_direction(pce_yoy),
            "sparkline": build_sparkline(pce_yoy),
            "series":    "PCE YoY %",
        },
        "liquidity": {
            "value":     m2_yoy_latest,
            "direction": compute_direction(m2_yoy),
            "sparkline": build_sparkline(m2_yoy),
            "series":    "M2 YoY %",
        },
        "fed_bs": {
            "value":     fed_bs_latest,
            "direction": compute_direction(fed_bs_t),
            "sparkline": build_sparkline(fed_bs_t),
            "series":    "Fed BS $T",
        },
        "real_yield": {
            "value":     real_yld_latest,
            "direction": compute_direction(real_yld_raw),
            "sparkline": build_sparkline(real_yld_raw),
            "series":    "TIPS 10Y %",
        },
    }
