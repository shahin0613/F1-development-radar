"""
Corner-speed-class analysis from qualifying telemetry: WHERE on the lap does a car gain or lose time?

The pace pipeline (quali.py) says how much slower or faster a team is overall. This module splits the
same flying laps into corner zones by speed class and asks the vehicle-dynamics question: is the car
slow in low-speed corners (mechanical grip, traction), in high-speed corners (aero load), or on the
straights (drag, power unit)?

Method, per qualifying session
  1. Take each driver's best valid lap in each segment (Q1/Q2/Q3), exactly as quali.py does.
  2. Resample every lap onto a common 5 m grid of normalised lap distance (laps differ by ~1% in
     measured length).
  3. Build a FIELD reference from the median speed / brake / throttle profile over all laps. Apexes are
     local speed minima of that median profile (prominence >= 20 km/h, >= 80 m apart), so zone
     boundaries and corner classes do not depend on any single team's car.
  4. Corner class by reference apex speed: Low < 120 km/h <= Medium < 200 km/h <= High.
     (A convention, not a law: change LOW_KMH / HIGH_KMH to test sensitivity.)
  5. Zone = from the start of the braking block before the apex (fallback: 100 m before) to where the
     median throttle first reaches 95% after it (at least 40 m, at most 250 m after). Neighbouring zones
     are cut at their midpoint. Everything outside zones is "Straight".
  6. Per lap and class: time spent in the class, and mean apex (minimum) speed.
  7. Per class: log-time ~ driver + segment (segment effects absorb track evolution, as in quali.py);
     team = mean of its drivers; gap = % versus a trimmed-mean field reference. Positive = slower.

Because time spent in the four classes adds up to lap time, the class results are a decomposition of
the overall qualifying gap.

Limits: ~7 Hz telemetry (about 12 m between samples at 300 km/h) so apex speed is a coarse estimate;
position-derived distance is integrated per lap; one lap per driver per segment; the 2026 cars'
energy management changes where drivers lift and deploy, so "straight" time also reflects power-unit
strategy, not only drag.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import find_peaks
from scipy.stats import trim_mean

from devradar.source_tracing import load_raw_session

RAW_TEL = "https://raw.githubusercontent.com/TracingInsights/{year}/main/{event}/Qualifying/{drv}/{lap}_tel.json"
CLASSES = ["Low", "Medium", "High", "Straight"]
STEP_M = 5.0
LOW_KMH, HIGH_KMH = 120.0, 200.0
PROMINENCE_KMH = 20.0
MIN_APEX_SEP_M = 80.0
BRAKE_BLOCK = 0.5          # fraction of laps braking for a point to count as a braking zone
THROTTLE_FULL = 95.0
FALLBACK_ENTRY_M = 100.0
MIN_EXIT_M, MAX_ZONE_SIDE_M = 40.0, 250.0
MAX_LENGTH_DEVIATION = 0.015  # laps whose integrated length is >1.5% off the median have telemetry dropouts: dropped
TEL_KEYS = ("rel_distance", "distance", "speed", "time", "throttle", "brake")


# --------------------------------------------------------------------------- data access
def best_laps_with_numbers(d: dict, max_pct_off: float = 7.0) -> pd.DataFrame:
    """Best valid lap per driver per segment (like quali.best_laps, but keeps the lap number)."""
    n = len(d["lap"])
    df = pd.DataFrame({
        "Driver": d["drv"], "Team": d["team"], "lap": pd.to_numeric(pd.Series(d["lap"]), errors="coerce"),
        "qs": d["qs"], "t": pd.to_numeric(pd.Series(d["time"]), errors="coerce"), "pb": d["pb"],
        "Compound": d["compound"],
    })
    df = df[(df["pb"] == True) & df["t"].notna() & df["lap"].notna() & df["qs"].isin(["Q1", "Q2", "Q3"])]  # noqa: E712
    b = df.sort_values("t").groupby(["Driver", "qs"], as_index=False).first()
    b = b[b["t"] <= b.groupby("qs")["t"].transform("min") * (1 + max_pct_off / 100)].copy()
    b["lap"] = b["lap"].astype(int)
    return b.reset_index(drop=True)


def load_tel(year: int, event: str, drv: str, lap: int, cache_dir: str = "cache") -> dict | None:
    """One lap of telemetry, reduced to the channels we need and cached locally."""
    path = os.path.join(cache_dir, "tracing", str(year), "tel", f"{event}__{drv}_{lap}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    url = RAW_TEL.format(year=year, event=urllib.parse.quote(event), drv=drv, lap=lap)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "devradar/0.1"})
        with urllib.request.urlopen(req, timeout=90) as r:
            tel = json.loads(r.read().decode("utf-8"))["tel"]
    except Exception:  # noqa: BLE001  (missing file, network error)
        return None
    small = {k: [round(float(x), 3) for x in tel[k]] for k in TEL_KEYS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(small, f)
    return small


# --------------------------------------------------------------------------- geometry
def resample(tel: dict, length_m: float, step: float = STEP_M) -> dict:
    """Interpolate a lap onto a common grid of normalised distance (grid in metres of the median lap)."""
    grid = np.arange(0.0, length_m, step)
    x = np.maximum.accumulate(np.asarray(tel["rel_distance"], float)) * length_m
    keep = np.concatenate([[True], np.diff(x) > 0])  # strictly increasing for interp
    out = {"grid": grid}
    for k in ("speed", "time", "throttle", "brake"):
        out[k] = np.interp(grid, x[keep], np.asarray(tel[k], float)[keep])
    return out


def find_zones(ref_speed, brake_frac, ref_throttle, grid, low=LOW_KMH, high=HIGH_KMH):
    """Corner zones on the field-median profile. Returns a list of dicts (index ranges on the grid)."""
    step = float(grid[1] - grid[0])
    smooth = np.convolve(ref_speed, np.ones(3) / 3, mode="same")
    smooth[0], smooth[-1] = ref_speed[0], ref_speed[-1]
    apexes, _ = find_peaks(-smooth, prominence=PROMINENCE_KMH, distance=max(int(MIN_APEX_SEP_M / step), 1))
    n = len(grid)
    zones = []
    for i in apexes:
        # entry: start of the last braking block that ends at/just after the apex
        lo = max(i - int(MAX_ZONE_SIDE_M / step), 0)
        on = brake_frac[lo:i + 1] >= BRAKE_BLOCK
        start = max(i - int(FALLBACK_ENTRY_M / step), 0)
        if on.any():
            last_on = np.flatnonzero(on)[-1] + lo
            j = last_on
            while j > lo and brake_frac[j - 1] >= BRAKE_BLOCK:
                j -= 1
            start = j
        # exit: first point where median throttle is (nearly) full, bounded
        hi = min(i + int(MAX_ZONE_SIDE_M / step), n - 1)
        end = hi
        for k in range(i + int(MIN_EXIT_M / step), hi + 1):
            if ref_throttle[k] >= THROTTLE_FULL:
                end = k
                break
        end = max(end, min(i + int(MIN_EXIT_M / step), n - 1))
        v = float(smooth[i])
        zones.append(dict(apex=int(i), start=int(start), end=int(end), ref_kmh=v,
                          cls="Low" if v < low else "Medium" if v < high else "High"))
    # cut overlapping neighbours at the midpoint between the earlier zone's end and the later zone's start
    for a, b in zip(zones[:-1], zones[1:]):
        if b["start"] < a["end"]:
            mid = (a["apex"] + b["apex"]) // 2
            a["end"], b["start"] = min(a["end"], mid), max(b["start"], mid)
    return zones


def lap_class_features(res: dict, zones: list, lap_time: float) -> pd.DataFrame:
    """Time spent and mean apex speed per class for one resampled lap."""
    t, v = res["time"], res["speed"]
    rows = {c: dict(time_s=0.0, apex=[]) for c in CLASSES}
    for z in zones:
        rows[z["cls"]]["time_s"] += float(t[z["end"]] - t[z["start"]])
        rows[z["cls"]]["apex"].append(float(v[z["start"]:z["end"] + 1].min()))
    rows["Straight"]["time_s"] = lap_time - sum(rows[c]["time_s"] for c in CLASSES if c != "Straight")
    return pd.DataFrame([dict(Class=c, time_s=r["time_s"], apex_kmh=float(np.mean(r["apex"])) if r["apex"] else np.nan,
                              n_zones=len(r["apex"])) for c, r in rows.items()])


# --------------------------------------------------------------------------- per-session pipeline
def session_features(year: int, event: str, cache_dir: str = "cache", workers: int = 8):
    """Class features for every best lap of one qualifying session. Returns (long_df, zones)."""
    laps = best_laps_with_numbers(load_raw_session(year, event, "Qualifying", cache_dir))
    with ThreadPoolExecutor(workers) as ex:
        tels = list(ex.map(lambda r: load_tel(year, event, r.Driver, r.lap, cache_dir), laps.itertuples()))
    ok = [i for i, t in enumerate(tels) if t is not None and len(t["speed"]) > 50]
    if len(ok) < 12:
        raise ValueError(f"only {len(ok)} laps with telemetry")
    laps, tels = laps.iloc[ok].reset_index(drop=True), [tels[i] for i in ok]
    lengths = np.array([t["distance"][-1] for t in tels])
    keep = np.abs(lengths / np.median(lengths) - 1) < MAX_LENGTH_DEVIATION   # drop truncated / odd laps
    laps, tels = laps[keep].reset_index(drop=True), [t for t, k in zip(tels, keep) if k]
    length_m = float(np.median(lengths[keep]))
    res = [resample(t, length_m) for t in tels]
    S = np.array([r["speed"] for r in res])
    ref_speed = np.median(S, axis=0)
    brake_frac = np.mean([r["brake"] for r in res], axis=0)
    ref_thr = np.median([r["throttle"] for r in res], axis=0)
    grid = res[0]["grid"]
    zones = find_zones(ref_speed, brake_frac, ref_thr, grid)
    frames = []
    for row, r in zip(laps.itertuples(), res):
        f = lap_class_features(r, zones, row.t)
        f["Driver"], f["Team"], f["qs"] = row.Driver, row.Team, row.qs
        frames.append(f)
    return pd.concat(frames, ignore_index=True), zones


def fit_class_gaps(long_df: pd.DataFrame, min_class_s: float = 1.0) -> pd.DataFrame:
    """Per class: team gap in % (time) and % (apex speed), positive = slower. Mirrors quali.fit_quali_pace."""
    out = []
    lap_t = float(long_df.groupby(["Driver", "qs"])["time_s"].sum().median())
    for cls, d in long_df.groupby("Class"):
        d = d[d["time_s"] > 0]
        if d.empty or d["time_s"].median() < min_class_s or d["Driver"].nunique() < 8:
            continue
        for metric, col, sign in (("time", "time_s", 1.0), ("apex_speed", "apex_kmh", -1.0)):
            if metric == "apex_speed" and (cls == "Straight" or d["apex_kmh"].isna().all()):
                continue
            b = d.dropna(subset=[col]).copy()
            b["y"] = sign * np.log(b[col]) * 100
            res = smf.ols("y ~ 0 + C(Driver) + C(qs)", data=b).fit()
            team_of = b.groupby("Driver")["Team"].agg(lambda s: s.mode().iat[0])
            drv = pd.DataFrame(
                [(team_of[k], res.params[f"C(Driver)[{k}]"], res.bse[f"C(Driver)[{k}]"]) for k in team_of.index],
                columns=["Team", "pace", "se"])
            drv["var"] = drv["se"] ** 2
            g = drv.groupby("Team").agg(pace=("pace", "mean"), var=("var", "sum"), n=("pace", "size"))
            g["se_pct"] = np.sqrt(g["var"]) / g["n"]
            g["gap_pct"] = g["pace"] - float(trim_mean(g["pace"].to_numpy(), 0.2))
            g = g.reset_index()[["Team", "gap_pct", "se_pct"]]
            g.insert(1, "Class", cls)
            g.insert(2, "Metric", metric)
            g["share"] = float(d["time_s"].median()) / lap_t   # field share of the lap spent in this class
            out.append(g)
    if not out:
        raise ValueError("no class had enough data")
    return pd.concat(out, ignore_index=True)


def build_corner_season(year: int, order: pd.DataFrame, cache_dir: str = "cache") -> pd.DataFrame:
    """order: DataFrame with RaceIdx, Event. One row per team x class x metric x race."""
    frames, zone_tab = [], []
    for row in order.itertuples():
        try:
            feats, zones = session_features(year, row.Event, cache_dir)
            g = fit_class_gaps(feats)
            g.insert(0, "Event", row.Event)
            g.insert(0, "Year", year)
            g["RaceIdx"] = row.RaceIdx
            frames.append(g)
            counts = {c: sum(z["cls"] == c for z in zones) for c in CLASSES[:3]}
            print(f"[ok]   #{row.RaceIdx:02d} {row.Event}: {len(zones)} corners {counts}")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] #{row.RaceIdx:02d} {row.Event}: {e}")
    if not frames:
        raise RuntimeError("no sessions produced corner estimates")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import sys

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    order = pd.read_csv("data/race_order_2026.csv")
    cache = os.environ.get("DEVRADAR_CACHE", "cache")
    df = build_corner_season(year, order, cache)
    df.to_csv(f"data/corner_gaps_quali_{year}.csv", index=False)
    print("saved", f"data/corner_gaps_quali_{year}.csv", len(df), "rows")
