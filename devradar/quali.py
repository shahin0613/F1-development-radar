"""
Qualifying pace: the cleanest read of raw car speed (low fuel, fresh tyres, one flying lap).

Per race:
  1. each driver's best VALID lap in each segment they ran (official personal-best flag => deleted laps excluded)
  2. fit   100*ln(lap time) ~ driver + segment    (segment effects = track evolution / push level, learned from
     drivers who ran several segments, so Q1 exits and Q3 finalists end up on one common scale)
  3. team = mean of its drivers' effects; gap_pct vs a trimmed-mean field reference (positive = slower)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import trim_mean

from .pace import SLICKS
from .source_tracing import chronological_events, load_raw_session


def best_laps(d: dict, max_pct_off: float = 7.0) -> pd.DataFrame:
    df = pd.DataFrame({
        "Driver": d["drv"], "Team": d["team"], "qs": d["qs"],
        "t": pd.to_numeric(pd.Series(d["time"]), errors="coerce"),
        "pb": d["pb"], "Compound": d["compound"],
    })
    df = df[(df["pb"] == True) & df["t"].notna() & df["qs"].isin(["Q1", "Q2", "Q3"])]  # noqa: E712
    b = df.sort_values("t").groupby(["Driver", "qs"], as_index=False).first()
    # drop crashed / damaged / aborted laps: > max_pct_off slower than the segment's fastest
    return b[b["t"] <= b.groupby("qs")["t"].transform("min") * (1 + max_pct_off / 100)].copy()


def fit_quali_pace(b: pd.DataFrame, min_slick_share: float = 0.75) -> pd.DataFrame:
    if b["Driver"].nunique() < 8:
        raise ValueError("too few drivers with a valid lap")
    if b["Compound"].isin(SLICKS).mean() < min_slick_share:
        raise ValueError("wet / mixed-condition qualifying")
    b = b.copy()
    b["y"] = np.log(b["t"]) * 100
    res = smf.ols("y ~ 0 + C(Driver) + C(qs)", data=b).fit()
    team_of = b.groupby("Driver")["Team"].agg(lambda s: s.mode().iat[0])
    drv = pd.DataFrame(
        [(team_of[k], res.params[f"C(Driver)[{k}]"], res.bse[f"C(Driver)[{k}]"]) for k in team_of.index],
        columns=["Team", "pace", "se"],
    )
    drv["var"] = drv["se"] ** 2
    g = drv.groupby("Team").agg(pace=("pace", "mean"), var=("var", "sum"), n=("pace", "size"))
    g["se_pct"] = np.sqrt(g["var"]) / g["n"]
    ref = float(trim_mean(g["pace"].to_numpy(), 0.2))
    g["gap_pct"] = g["pace"] - ref
    return g.reset_index()[["Team", "gap_pct", "se_pct"]]


def build_quali_season(year: int, cache_dir: str = "cache", first: int | None = None, last: int | None = None):
    order = chronological_events(year, cache_dir)
    frames = []
    for i, ev in enumerate(order, start=1):
        if (first and i < first) or (last and i > last):
            continue
        try:
            g = fit_quali_pace(best_laps(load_raw_session(year, ev, "Qualifying", cache_dir)))
            g.insert(0, "Event", ev)
            g.insert(0, "Year", year)
            g["RaceIdx"] = i
            frames.append(g)
            print(f"[ok]   #{i:02d} {ev} (qualifying)")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] #{i:02d} {ev}: {e}")
    if not frames:
        raise RuntimeError("No qualifying sessions produced estimates.")
    return pd.concat(frames, ignore_index=True)
