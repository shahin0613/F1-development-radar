"""Changing FUTURE rows must not change any forecast made from the past."""
import numpy as np
import pandas as pd
import pytest

from devradar.backtest import load
from devradar.causal_model import DESCRIPTORS, forecast_race

DATA = "data/"


@pytest.fixture(scope="module")
def data():
    return load(DATA + "gaps_quali_2026.csv", DATA + "circuit_descriptors.csv")


@pytest.mark.parametrize("r", [6, 8, 10, 13])
@pytest.mark.parametrize("oracle", [False, True])
def test_future_does_not_leak(data, r, oracle):
    Y, S, D = data
    base, sig = forecast_race(Y, D, r, oracle=oracle)

    Y2, D2 = Y.copy(), D.copy()
    # wreck every gap from the target race onwards (the target's own gaps are the answer, never an input)
    Y2.iloc[r:] = np.random.default_rng(1).normal(size=Y2.iloc[r:].shape) * 50
    # wreck descriptors AFTER the target race; the target race's own descriptors stay legitimate
    # oracle input, and must stay untouched
    D2.loc[D2.index[r + 1:], DESCRIPTORS + ["best_lap"]] = 1e6
    base2, sig2 = forecast_race(Y2, D2, r, oracle=oracle)

    pd.testing.assert_series_equal(base, base2)
    pd.testing.assert_series_equal(sig, sig2)


def test_honest_forecast_ignores_target_descriptors(data):
    """oracle=False must not read the target race's own (weekend-measured) descriptors."""
    Y, S, D = data
    r = 10
    base, _ = forecast_race(Y, D, r, oracle=False)
    D2 = D.copy()
    D2.iloc[r, D2.columns.get_indexer(DESCRIPTORS)] = 1e6
    base2, _ = forecast_race(Y, D2, r, oracle=False)
    pd.testing.assert_series_equal(base, base2)


def test_honest_forecast_is_rolling_median(data):
    Y, S, D = data
    r = 9
    base, _ = forecast_race(Y, D, r, k=3, oracle=False)
    pd.testing.assert_series_equal(base, Y.iloc[:r].iloc[-3:].median(), check_names=False)
