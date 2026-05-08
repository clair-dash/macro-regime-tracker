"""Unit tests for pure functions in fetch_macro_data.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fetch_macro_data import sanitize

def test_sanitize_replaces_nan():
    import math
    result = sanitize(float('nan'))
    assert result is None

def test_sanitize_replaces_inf():
    import math
    result = sanitize(float('inf'))
    assert result is None

def test_sanitize_leaves_valid_float():
    assert sanitize(3.14) == 3.14

def test_sanitize_recurses_dict():
    import math
    result = sanitize({"a": float('nan'), "b": 1.0})
    assert result == {"a": None, "b": 1.0}

def test_sanitize_recurses_list():
    import math
    result = sanitize([float('nan'), 2.0, float('inf')])
    assert result == [None, 2.0, None]

# ─── Regime Indicator Tests ───────────────────────────────────────────────────

from fetch_macro_data import compute_direction, build_sparkline, assign_radar_signal

# compute_direction tests
def test_direction_accelerating():
    # Latest = 4.0, 3 months ago = 3.0 → ROC = 0.333 > 0.005
    series = [3.0, 3.2, 3.5, 4.0]
    assert compute_direction(series) == "accelerating"

def test_direction_decelerating():
    # Latest = 3.0, 3 months ago = 4.0 → ROC = -0.25 < -0.005
    series = [4.0, 3.8, 3.5, 3.0]
    assert compute_direction(series) == "decelerating"

def test_direction_stable():
    # Latest = 3.001, 3 months ago = 3.0 → ROC = 0.00033 between thresholds
    series = [3.0, 3.0, 3.0, 3.001]
    assert compute_direction(series) == "stable"

def test_direction_too_short_returns_stable():
    assert compute_direction([3.0]) == "stable"

def test_direction_zero_prior_returns_stable():
    assert compute_direction([0.0, 0.0, 0.0, 1.0]) == "stable"

# build_sparkline tests
def test_sparkline_returns_last_n_points():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert build_sparkline(values, n_points=6) == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def test_sparkline_filters_none():
    values = [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0]
    result = build_sparkline(values, n_points=6)
    assert None not in result

def test_sparkline_fewer_than_n_points():
    values = [1.0, 2.0]
    assert build_sparkline(values, n_points=6) == [1.0, 2.0]

# assign_radar_signal tests
def test_radar_signal_hy_low():
    assert assign_radar_signal("hy_spread", 250) == "low"

def test_radar_signal_hy_neutral():
    assert assign_radar_signal("hy_spread", 400) == "neutral"

def test_radar_signal_hy_elevated():
    assert assign_radar_signal("hy_spread", 600) == "elevated"

def test_radar_signal_vix_low():
    assert assign_radar_signal("vix", 15) == "low"

def test_radar_signal_vix_stress():
    assert assign_radar_signal("vix", 35) == "elevated"

def test_radar_signal_unknown_key():
    assert assign_radar_signal("unknown", 100) == "neutral"

# ─── FRED Fetcher Tests ───────────────────────────────────────────────────────

from unittest.mock import patch, MagicMock
from fetch_macro_data import fetch_fred_series, fetch_fred_latest

def test_fetch_fred_series_returns_list_on_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2026-05-01", "value": "3.3"},
            {"date": "2026-04-01", "value": "3.2"},
            {"date": "2026-03-01", "value": "."},   # missing — should be excluded
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("fetch_macro_data.requests.get", return_value=mock_resp):
        with patch.dict("os.environ", {"FRED_API_KEY": "testkey"}):
            import fetch_macro_data
            fetch_macro_data.FRED_API_KEY = "testkey"
            result = fetch_fred_series("CPIAUCSL", lookback_days=90)
    assert len(result) == 2
    assert result[0]["value"] == "3.3"

def test_fetch_fred_series_returns_empty_without_key():
    import fetch_macro_data
    original = fetch_macro_data.FRED_API_KEY
    fetch_macro_data.FRED_API_KEY = ""
    result = fetch_fred_series("CPIAUCSL")
    fetch_macro_data.FRED_API_KEY = original
    assert result == []

def test_fetch_fred_latest_returns_float():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "observations": [{"date": "2026-05-01", "value": "1.94"}]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("fetch_macro_data.requests.get", return_value=mock_resp):
        import fetch_macro_data
        fetch_macro_data.FRED_API_KEY = "testkey"
        result = fetch_fred_latest("DFII10")
    assert result == 1.94
