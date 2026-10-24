"""
Alternative lap source: TracingInsights public GitHub cache (FastF1-derived, 2018-current).
Use when FastF1's own download from F1's servers fails. Needs only plain HTTPS to github.

Each race has one file:  <Event>/Race/session_laptimes.json  (column-oriented: lists per field).
We reshape it into the same columns FastF1's `session.laps` has, so pace.py works unchanged.

Missing here vs FastF1: IsAccurate / Deleted flags (not in the session-level file).
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/TracingInsights/{year}/main/{event}/{session}/session_laptimes.json"
API = "https://api.github.com/repos/TracingInsights/{year}/contents"
NOT_EVENTS = ("Pre-Season", "cache", "images", "schemas", "sections", ".github")


def _get_json(url: str, retries: int = 3):
    last = None
    for k in range(retries):
        try:
            headers = {"User-Agent": "devradar/0.1"}
            token = os.environ.get("GITHUB_TOKEN")  # optional: lifts the 60 requests/hour API limit
            if token and "api.github.com" in url:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def list_events(year: int, cache_dir: str = "cache") -> list[str]:
    """Event folders for a season. Cached locally; falls back to the cache if GitHub's API rate-limits us."""
    path = os.path.join(cache_dir, "tracing", str(year), "_events.json")
    try:
        items = _get_json(API.format(year=year))
        names = [x["name"] for x in items if x["type"] == "dir" and not x["name"].startswith(NOT_EVENTS)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(names, f)
        return names
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            print(f"[warn] GitHub API unavailable ({str(e)[:50]}); using cached event list")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        raise


def _td(series_like) -> pd.Series:
    return pd.to_timedelta(pd.to_numeric(pd.Series(series_like), errors="coerce"), unit="s")


def to_fastf1_laps(d: dict) -> pd.DataFrame:
    n = len(d["lap"])
    return pd.DataFrame({
        "Driver": d["drv"],
        "Team": d["team"],
        "LapNumber": pd.to_numeric(pd.Series(d["lap"]), errors="coerce"),
        "LapTime": _td(d["time"]),
        "Time": _td(d["sesT"]),
        "Position": pd.to_numeric(pd.Series(d["pos"]), errors="coerce"),
        "PitInTime": _td(d["pin"]),
        "PitOutTime": _td(d["pout"]),
        "TrackStatus": pd.Series(d["status"]).astype(str),
        "Compound": d["compound"],
        "TyreLife": pd.to_numeric(pd.Series(d["life"]), errors="coerce"),
        "_lSD": d.get("lSD", [None] * n),
        "IsAccurate": pd.Series(d["iacc"]).astype(bool) if "iacc" in d else True,
        "Deleted": pd.Series(d["del"]).astype(bool) if "del" in d else False,
        "RainLapShare": float(pd.Series(d["wR"]).astype(bool).mean()) if "wR" in d else 0.0,
    })


def load_raw_session(year: int, event: str, session: str = "Race", cache_dir: str = "cache") -> dict:
    """Raw column-oriented session_laptimes.json for one session ("Race" or "Qualifying"), cached locally."""
    fname = f"{event}.json" if session == "Race" else f"{event}__{session}.json"  # keep old race cache valid
    path = os.path.join(cache_dir, "tracing", str(year), fname)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    d = _get_json(RAW.format(year=year, event=urllib.parse.quote(event), session=session))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return d


def load_race_laps(year: int, event: str, cache_dir: str = "cache") -> pd.DataFrame:
    return to_fastf1_laps(load_raw_session(year, event, "Race", cache_dir))


def chronological_events(year: int, cache_dir: str = "cache") -> list[str]:
    """Events that have a race file, in calendar order (same numbering as build_season_tracing)."""
    loaded = []
    for ev in list_events(year, cache_dir):
        try:
            loaded.append((ev, load_race_laps(year, ev, cache_dir)))
        except Exception:  # noqa: BLE001
            pass
    loaded.sort(key=lambda t: (pd.isna(_event_date(t[1])), _event_date(t[1])))
    return [ev for ev, _ in loaded]


def _event_date(laps: pd.DataFrame):
    return pd.to_datetime(laps["_lSD"], errors="coerce", utc=True).min()


def build_season_tracing(year: int, cache_dir: str = "cache", first: int | None = None, last: int | None = None):
    """Pull every available race of `year`, in calendar order. Returns (gaps_df, event_order_df)."""
    from .pace import race_gaps

    loaded = []
    for ev in list_events(year, cache_dir):
        try:
            loaded.append((ev, load_race_laps(year, ev, cache_dir)))
        except Exception as e:  # noqa: BLE001  (future races have no file yet)
            print(f"[none] {ev}: not available yet ({str(e)[:60]})")
    loaded.sort(key=lambda t: (pd.isna(_event_date(t[1])), _event_date(t[1])))

    frames, order = [], []
    for i, (ev, laps) in enumerate(loaded, start=1):
        order.append({"RaceIdx": i, "Event": ev})
        if (first and i < first) or (last and i > last):
            continue
        try:
            g = race_gaps(laps)
            g.insert(0, "Event", ev)
            g.insert(0, "Year", year)
            g["RaceIdx"] = i  # chronological position among ALL races with data (skips keep their slot)
            frames.append(g)
            print(f"[ok]   #{i:02d} {ev}")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] #{i:02d} {ev}: {e}")
    if not frames:
        raise RuntimeError("No races produced pace estimates.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(order)
