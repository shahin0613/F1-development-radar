"""Corner-class analysis on synthetic laps with known corners and a planted team effect."""
import numpy as np
import pandas as pd

from devradar.corners import CLASSES, find_zones, fit_class_gaps, lap_class_features

STEP = 5.0
GRID = np.arange(0.0, 4000.0, STEP)
APEXES = {500: 90.0, 1500: 150.0, 2500: 230.0, 3300: 100.0}      # metres -> km/h (Low, Medium, High, Low)


def synthetic_profile():
    """Flat-out at 300 km/h with a dip to the apex speed at each corner; brake before, throttle after."""
    speed = np.full_like(GRID, 300.0)
    brake = np.zeros_like(GRID)
    throttle = np.full_like(GRID, 100.0)
    for pos, v in APEXES.items():
        d = GRID - pos
        dip = np.where(d < 0, np.clip(1 + d / 150.0, 0, 1), np.clip(1 - d / 100.0, 0, 1))
        speed = np.minimum(speed, 300.0 - (300.0 - v) * dip)
        brake[(d > -150) & (d < -10)] = 1.0
        throttle[(d > -20) & (d < 70)] = 20.0
    return speed, brake, throttle


def lap_time(speed):
    v = np.maximum(speed, 20.0) / 3.6
    return np.concatenate([[0.0], np.cumsum(STEP / v[:-1])])


def test_find_zones_recovers_corners_and_classes():
    speed, brake, thr = synthetic_profile()
    zones = find_zones(speed, brake, thr, GRID)
    assert len(zones) == 4
    assert [z["cls"] for z in zones] == ["Low", "Medium", "High", "Low"]
    for z, pos in zip(zones, APEXES):
        assert abs(GRID[z["apex"]] - pos) <= 10
        assert z["start"] < z["apex"] < z["end"]
    assert all(a["end"] <= b["start"] for a, b in zip(zones[:-1], zones[1:]))    # zones do not overlap


def test_class_times_add_up_to_lap_time():
    speed, brake, thr = synthetic_profile()
    zones = find_zones(speed, brake, thr, GRID)
    res = dict(speed=speed, time=lap_time(speed), grid=GRID)
    total = float(res["time"][-1])
    f = lap_class_features(res, zones, total)
    assert set(f["Class"]) == set(CLASSES)
    assert abs(f["time_s"].sum() - total) < 1e-9
    assert (f["time_s"] > 0).all()
    assert f.set_index("Class").loc["Low", "n_zones"] == 2


def test_fit_class_gaps_recovers_planted_low_speed_deficit():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(6)]
    rows = []
    for ti, team in enumerate(teams):
        for k in (0, 1):
            drv = f"{team}d{k}"
            for qs, evo in (("Q1", 0.0), ("Q2", -0.01), ("Q3", -0.02)):
                for cls, base in (("Low", 25.0), ("Medium", 12.0), ("Straight", 40.0)):
                    plant = 0.015 if (team == "T0" and cls == "Low") else 0.0
                    t = base * (1 + plant + evo + rng.normal(0, 0.002))
                    rows.append(dict(Driver=drv, Team=team, qs=qs, Class=cls, time_s=t, apex_kmh=100.0))
    long_df = pd.DataFrame(rows)
    g = fit_class_gaps(long_df)
    low = g[(g.Metric == "time") & (g.Class == "Low")].set_index("Team")["gap_pct"]
    assert 1.2 < low["T0"] < 1.8                                    # planted +1.5%
    assert low.drop("T0").abs().max() < 0.4                         # other teams unaffected
    med = g[(g.Metric == "time") & (g.Class == "Medium")].set_index("Team")["gap_pct"]
    assert med.abs().max() < 0.4                                    # no spill-over into other classes
