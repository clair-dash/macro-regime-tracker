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
# FRED series for full US Treasury yield curve (replaces Treasury XML endpoint)
FRED_YIELD_CURVE = {
    '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO',
    '1Y': 'DGS1',   '2Y': 'DGS2',   '3Y': 'DGS3',
    '5Y': 'DGS5',   '7Y': 'DGS7',   '10Y': 'DGS10',
    '20Y': 'DGS20', '30Y': 'DGS30',
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


# ─── yfinance Helpers ────────────────────────────────────────────────────────

def yf_latest_price(ticker: str) -> float | None:
    """Get the most recent closing price for a ticker."""
    import math
    try:
        df = safe_download(ticker, period="5d", interval="1d")
        if df.empty:
            return None
        price = float(df["Close"].dropna().iloc[-1])
        return None if (math.isnan(price) or math.isinf(price)) else round(price, 4)
    except Exception as e:
        log.error(f"Price fetch failed for {ticker}: {e}")
        return None


def yf_returns(ticker: str, current_price: float) -> dict:
    """Compute 1W / 1M / YTD / 1Y returns. Returns dict with None on failure."""
    import math
    empty = {"w1": None, "m1": None, "ytd": None, "y1": None}
    if current_price is None:
        return empty
    try:
        df = safe_download(ticker, period="13mo", interval="1d")
        if df.empty or len(df) < 5:
            return empty
        closes = df["Close"].dropna()

        def pct(past):
            if past is None or past == 0:
                return None
            v = (current_price - past) / past * 100
            return round(v, 2) if not math.isnan(v) else None

        def price_n_ago(n):
            idx = max(0, len(closes) - 1 - n)
            v = float(closes.iloc[idx])
            return None if math.isnan(v) else v

        # YTD
        ytd = None
        try:
            yr_start = pd.Timestamp(datetime(datetime.now().year, 1, 1))
            if closes.index.tz:
                yr_start = yr_start.tz_localize(closes.index.tz)
            ytd_slice = closes[closes.index >= yr_start]
            if len(ytd_slice):
                s = float(ytd_slice.iloc[0])
                ytd = pct(s) if not math.isnan(s) else None
        except Exception:
            pass

        return {
            "w1":  pct(price_n_ago(5)),
            "m1":  pct(price_n_ago(21)),
            "ytd": ytd,
            "y1":  pct(price_n_ago(252)) if len(closes) > 252 else pct(float(closes.iloc[0])),
        }
    except Exception as e:
        log.error(f"Returns failed for {ticker}: {e}")
        return empty


def yf_history_series(ticker: str, years: int = 5) -> list[float]:
    """Return chronological monthly close prices for a ticker."""
    try:
        df = safe_download(ticker, period=f"{years * 12}mo", interval="1mo")
        if df.empty:
            return []
        return [round(float(v), 4) for v in df["Close"].dropna()]
    except Exception as e:
        log.error(f"History failed for {ticker}: {e}")
        return []


def build_gold_data() -> dict:
    """Fetch gold price, returns, DXY, and real yield for Gold Positioning panel."""
    import math

    gold_usd = yf_latest_price(YF_GOLD)
    usdchf   = yf_latest_price(YF_USDCHF)
    gold_chf = round(gold_usd * usdchf, 2) if (gold_usd and usdchf) else None

    returns = yf_returns(YF_GOLD, gold_usd)

    # Gold CHF returns (cross-currency)
    chf_returns = {"w1": None, "m1": None, "ytd": None, "y1": None}
    try:
        g_hist  = safe_download(YF_GOLD,   period="13mo", interval="1d")
        fx_hist = safe_download(YF_USDCHF, period="13mo", interval="1d")
        if not g_hist.empty and not fx_hist.empty:
            gc  = g_hist["Close"].dropna()
            uc  = fx_hist["Close"].dropna()
            common = gc.index.intersection(uc.index)
            if len(common) > 5:
                gold_chf_series = (gc.loc[common] * uc.loc[common])
                now_chf = float(gold_chf_series.iloc[-1])

                def chf_pct(n):
                    idx  = max(0, len(gold_chf_series) - 1 - n)
                    past = float(gold_chf_series.iloc[idx])
                    if past == 0 or math.isnan(past):
                        return None
                    v = (now_chf - past) / past * 100
                    return round(v, 2) if not math.isnan(v) else None

                chf_returns["w1"]  = chf_pct(5)
                chf_returns["m1"]  = chf_pct(21)
                chf_returns["y1"]  = chf_pct(252) if len(gold_chf_series) > 252 else chf_pct(len(gold_chf_series) - 1)

                yr_start = pd.Timestamp(datetime(datetime.now().year, 1, 1))
                if gold_chf_series.index.tz:
                    yr_start = yr_start.tz_localize(gold_chf_series.index.tz)
                ytd_s = gold_chf_series[gold_chf_series.index >= yr_start]
                if len(ytd_s):
                    s = float(ytd_s.iloc[0])
                    if s and not math.isnan(s):
                        chf_returns["ytd"] = round((now_chf - s) / s * 100, 2)
    except Exception as e:
        log.warning(f"Gold CHF returns failed: {e}")

    # Real yield history for overlay
    real_yld_hist = fetch_fred_history(FRED_REAL_YLD, lookback_days=LOOKBACK_YEARS * 365)
    gold_hist     = yf_history_series(YF_GOLD, years=LOOKBACK_YEARS)

    # DXY
    dxy_price = yf_latest_price(YF_DXY)
    if dxy_price is None:
        dxy_signal = "unknown"
    elif dxy_price < 100:
        dxy_signal = "supportive"
    elif dxy_price > 105:
        dxy_signal = "headwind"
    else:
        dxy_signal = "neutral"

    return {
        "price_usd":          gold_usd,
        "price_chf":          gold_chf,
        "returns":            returns,
        "returns_chf":        chf_returns,
        "real_yield_history": real_yld_hist,
        "gold_price_history": gold_hist,
        "dxy": {
            "value":   dxy_price,
            "signal":  dxy_signal,
            "returns": yf_returns(YF_DXY, dxy_price),
        },
    }


# ─── US Treasury Yield Curve ─────────────────────────────────────────────────

def _fred_curve_snapshot(target_date: datetime, obs_by_series: dict) -> dict:
    """Extract yield curve snapshot for a target date from pre-fetched FRED observations."""
    target_str = target_date.strftime("%Y-%m-%d")
    snapshot = {}
    for label, obs in obs_by_series.items():
        for o in obs:
            if o["date"] <= target_str:
                try:
                    snapshot[label] = round(float(o["value"]), 2)
                except (ValueError, KeyError):
                    pass
                break
    return snapshot


def build_yield_curve_data() -> dict:
    """Fetch US yield curves (today/1Y/2Y ago) and international benchmarks from FRED."""
    now       = datetime.now()
    one_y_ago = now - timedelta(days=365)
    two_y_ago = now - timedelta(days=730)

    log.info("Fetching US yield curve from FRED...")
    obs_by_series = {}
    for label, series_id in FRED_YIELD_CURVE.items():
        obs = fetch_fred_series(series_id, lookback_days=365 * 3)
        if obs:
            obs_by_series[label] = obs  # already desc order

    today_curve  = _fred_curve_snapshot(now,       obs_by_series)
    one_yr_curve = _fred_curve_snapshot(one_y_ago, obs_by_series)
    two_yr_curve = _fred_curve_snapshot(two_y_ago, obs_by_series)

    # International 10Y benchmarks
    eur_10y = fetch_fred_latest(FRED_BUND)
    gbp_10y = fetch_fred_latest("IRLTLT01GBM156N")
    chf_10y = _fetch_swiss_10y()

    return {
        "today":    today_curve,
        "1y_ago":   one_yr_curve,
        "2y_ago":   two_yr_curve,
        "benchmarks": {
            "EUR": eur_10y,
            "GBP": round(gbp_10y, 2) if gbp_10y is not None else None,
            "CHF": chf_10y,
        },
    }


def _fetch_swiss_10y() -> float | None:
    """Fetch Swiss 10Y yield from SNB API."""
    try:
        url  = "https://data.snb.ch/api/cube/rendoblim/data/csv/en"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            for line in reversed(resp.text.strip().split("\n")):
                parts = line.split(";")
                if len(parts) >= 2:
                    try:
                        return round(float(parts[-1].strip().strip('"')), 2)
                    except ValueError:
                        continue
        return None
    except Exception as e:
        log.warning(f"SNB fetch failed: {e}")
        return None


# ─── Deployment Radar ────────────────────────────────────────────────────────

def build_deployment_radar() -> dict:
    """Fetch credit spreads, VIX, and SPX vs 200MA for the Deployment Radar panel."""
    import math

    hy  = fetch_fred_latest(FRED_HY_OAS)
    ig  = fetch_fred_latest(FRED_IG_OAS)
    vix = yf_latest_price(YF_VIX)

    # SPX vs 200-day MA
    spx_vs_200 = None
    try:
        df = safe_download(YF_SPX, period="14mo", interval="1d")
        if not df.empty and len(df) >= 200:
            closes = df["Close"].dropna()
            current = float(closes.iloc[-1])
            ma200   = float(closes.iloc[-200:].mean())
            if ma200 > 0 and not math.isnan(ma200):
                spx_vs_200 = round((current / ma200 - 1) * 100, 2)
    except Exception as e:
        log.warning(f"SPX 200MA failed: {e}")

    # Sparklines for spreads (last 6 monthly values)
    hy_hist = fetch_fred_history(FRED_HY_OAS, lookback_days=LOOKBACK_YEARS * 365)
    ig_hist = fetch_fred_history(FRED_IG_OAS, lookback_days=LOOKBACK_YEARS * 365)

    return {
        "hy_spread": {
            "value":     round(hy, 0) if hy else None,
            "signal":    assign_radar_signal("hy_spread", hy),
            "sparkline": build_sparkline(hy_hist),
            "unit":      "bp",
        },
        "ig_spread": {
            "value":     round(ig, 0) if ig else None,
            "signal":    assign_radar_signal("ig_spread", ig),
            "sparkline": build_sparkline(ig_hist),
            "unit":      "bp",
        },
        "vix": {
            "value":  round(vix, 1) if vix else None,
            "signal": assign_radar_signal("vix", vix),
            "unit":   "",
        },
        "spx_vs_200ma": {
            "value":  spx_vs_200,
            "signal": assign_radar_signal("spx_vs_200ma", spx_vs_200),
            "unit":   "%",
        },
    }


# ─── Assembly ────────────────────────────────────────────────────────────────

def build_data() -> dict:
    """Fetch all sources and assemble macro_data.json payload."""
    log.info("Building macro regime data...")
    macro_regime   = build_macro_regime()

    log.info("Building gold positioning data...")
    gold           = build_gold_data()

    log.info("Building deployment radar data...")
    radar          = build_deployment_radar()

    log.info("Building yield curve data...")
    yield_curve    = build_yield_curve_data()

    return {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "macro_regime":     macro_regime,
        "gold":             gold,
        "deployment_radar": radar,
        "yield_curve":      yield_curve,
    }


def main():
    try:
        data = sanitize(build_data())
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"Written to {OUTPUT_PATH}")
        log.info(f"Timestamp: {data['timestamp']}")
        # Log direction summaries
        for k, v in data["macro_regime"].items():
            log.info(f"  {k}: {v.get('value')} → {v.get('direction')}")
    except Exception as e:
        log.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
