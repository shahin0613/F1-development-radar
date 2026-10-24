"""
Race-pace extraction.

Raw lap times -> clean-air, fuel/tyre-corrected team pace with standard errors.

Per race we fit one regression on clean laps:

    LapTime ~ driver + compound + compound:TyreLife + lap_number

  driver effect          = that driver's underlying pace (what we want)
  compound:TyreLife      = tyre degradation, separate slope per compound
  lap_number             = fuel burn + track evolution (combined, linear)

Team pace = mean of its drivers' effects. Gaps are expressed as % of the
field (trimmed-mean) pace so races at different circuits are comparable.
Convention: gap_pct > 0 means SLOWER than the field median.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import trim_mean

SLICKS = ["SOFT", "MEDIUM", "HARD"]


def load_race_laps(year: int, rnd: int, cache_dir: str = "cache"):
    """Pull one race from FastF1. Returns (laps_df, event_name)."""
    import fastf1

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    s = fastf1.get_session(year, rnd, "R")
    s.load(telemetry=False, weather=False, messages=False)
    return s.laps.copy(), str(s.event["EventName"])


def add_gap_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """Gap (s) to the car directly ahead at the end of each lap."""
    d = laps.copy()
    key = d[["LapNumber", "Position", "Time"]].dropna()
    ahead = key.assign(Position=key["Position"] + 1).rename(columns={"Time": "TimeAhead"})
    d = d.merge(ahead, on=["LapNumber", "Position"], how="left")
    d["GapAheadS"] = (d["Time"] - d["TimeAhead"]).dt.total_seconds()
    return d


def clean_laps(laps: pd.DataFrame, clean_air_s: float = 1.5, outlier_pct: float = 5.0) -> pd.DataFrame:
    """Keep only representative laps: green flag, slicks, no pit, clean air, no outliers."""
    d = add_gap_ahead(laps)
    d["LapTimeS"] = d["LapTime"].dt.total_seconds()
    ok = (
        d["LapTimeS"].notna()
        & d["PitInTime"].isna()
        & d["PitOutTime"].isna()
        & (d["LapNumber"] > 1)
        & (d["TrackStatus"].astype(str) == "1")  # green for the whole lap
        & d["Compound"].isin(SLICKS)
        & d["TyreLife"].notna()
        & (d["GapAheadS"].isna() | (d["GapAheadS"] > clean_air_s))  # leader has no car ahead
    )
    if "IsAccurate" in d:
        ok &= d["IsAccurate"].fillna(False).astype(bool)
    if "Deleted" in d:
        ok &= ~d["Deleted"].fillna(False).astype(bool)
    d = d[ok].copy()
    med = d.groupby("Driver")["LapTimeS"].transform("median")
    return d[d["LapTimeS"] <= med * (1 + outlier_pct / 100)]


def fit_race_pace(clean: pd.DataFrame, min_laps: int = 8):
    """Fit the correction model. Returns (team_df, fitted_ols)."""
    d = clean.groupby("Driver").filter(lambda g: len(g) >= min_laps).copy()
    if d["Driver"].nunique() < 6:
        raise ValueError("too few drivers with enough clean laps (wet/chaotic race?)")
    d["lap_c"] = d["LapNumber"] - d["LapNumber"].mean()
    used = [c for c in SLICKS if c in set(d["Compound"])]
    d["Compound"] = pd.Categorical(d["Compound"], categories=used)

    res = smf.ols(
        "LapTimeS ~ 0 + C(Driver) + C(Compound) + C(Compound):TyreLife + lap_c", data=d
    ).fit()

    team_of = d.groupby("Driver")["Team"].agg(lambda s: s.mode().iat[0])
    rows = [
        (team, drv, res.params[f"C(Driver)[{drv}]"], res.bse[f"C(Driver)[{drv}]"])
        for drv, team in team_of.items()
    ]
    drv = pd.DataFrame(rows, columns=["Team", "Driver", "pace_s", "se_s"])
    drv["var"] = drv["se_s"] ** 2
    g = drv.groupby("Team").agg(pace_s=("pace_s", "mean"), var=("var", "sum"), n=("pace_s", "size"))
    # NB: ignores covariance between teammates' estimates (small; fine for v1)
    g["se_s"] = np.sqrt(g["var"]) / g["n"]
    return g.reset_index()[["Team", "pace_s", "se_s"]], res


def to_gap_table(team_df: pd.DataFrame) -> pd.DataFrame:
    """Seconds -> % gap to field median (positive = slower)."""
    # trimmed mean of the field: robust to the slowest/fastest teams, but unlike the median of an
    # odd number of teams it is not literally one team's own estimate (which pinned it to exactly 0)
    ref = float(trim_mean(team_df["pace_s"].to_numpy(), 0.2))
    out = team_df.copy()
    out["gap_pct"] = (out["pace_s"] - ref) / ref * 100
    out["se_pct"] = out["se_s"] / ref * 100
    return out[["Team", "gap_pct", "se_pct"]]


def race_gaps(laps: pd.DataFrame, **kw) -> pd.DataFrame:
    team_df, _ = fit_race_pace(clean_laps(laps, **kw))
    return to_gap_table(team_df)


def build_season(year: int, rounds, cache_dir: str = "cache") -> pd.DataFrame:
    """Run the pipeline over a list of rounds. One row per team per race."""
    frames = []
    for i, rnd in enumerate(rounds, start=1):
        try:
            laps, event = load_race_laps(year, rnd, cache_dir)
            g = race_gaps(laps)
            g.insert(0, "Event", event)
            g.insert(0, "Round", rnd)
            g.insert(0, "Year", year)
            g["RaceIdx"] = i  # sequential index used as the GP's x-axis
            frames.append(g)
            print(f"[ok]   R{rnd:02d} {event}")
        except Exception as e:  # keep going; wet races etc. will fail
            print(f"[skip] R{rnd:02d}: {e}")
    if not frames:
        raise RuntimeError("No races loaded. FastF1 could not download data (network blocked or source down). Try --source tracing.")
    return pd.concat(frames, ignore_index=True)
