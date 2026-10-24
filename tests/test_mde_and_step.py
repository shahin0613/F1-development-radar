"""Sanity checks that the statistical machinery does what the README says."""
import numpy as np
import pandas as pd

from devradar.mde_simulation import calibrate, simulate
from devradar.trend import step_effect


def test_false_positive_rate_is_controlled():
    # zero true gain: one-sided 5% test should reject about 5% of the time
    power, _, _ = simulate(0.0, 17, 9, race_sd=0.1, team_sd=0.4, n_sims=1500, rng=np.random.default_rng(3))
    assert power < 0.09


def test_power_rises_with_effect_size():
    lo, _, _ = simulate(0.2, 17, 9, 0.1, 0.4, n_sims=800, rng=np.random.default_rng(4))
    hi, _, _ = simulate(1.2, 17, 9, 0.1, 0.4, n_sims=800, rng=np.random.default_rng(4))
    assert hi > 0.9 and hi > lo


def test_step_effect_recovers_planted_step():
    rng = np.random.default_rng(0)
    rows = []
    for race in range(1, 13):
        for team in ["A", "B"]:
            gap = 2.0 + rng.normal(0, 0.1) - (1.0 if team == "A" and race >= 7 else 0.0)
            rows.append(dict(RaceIdx=race, Team=team, gap_pct=gap, se_pct=0.1))
    g = pd.DataFrame(rows)
    a, b = step_effect(g, "A", 7), step_effect(g, "B", 7)
    assert abs(a["gain_pct"] - 1.0) < 0.3 and a["ci90"][0] > 0
    assert b["ci90"][0] < 0 < b["ci90"][1]
