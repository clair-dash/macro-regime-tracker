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
