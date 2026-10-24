"""Synthetic races in FastF1's lap-table shape, with a KNOWN planted upgrade.
Used to prove the pipeline recovers truth before you trust it on real data."""
import numpy as np
import pandas as pd

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet"]


def make_race(rng, team_gap_pct: dict, n_laps: int = 56, base: float = 90.0) -> pd.DataFrame:
    rows = []
    for t in team_gap_pct:
        for k in (1, 2):
            drv = f"{t[:2].upper()}{k}"
            off = rng.normal(0, 0.15)  # driver-level offset
            pit = int(rng.integers(20, 32))
            t_cum = rng.uniform(0, 4)  # staggered grid
            for lap in range(1, n_laps + 1):
                comp = "MEDIUM" if lap <= pit else "HARD"
                life = lap if lap <= pit else lap - pit
                lt = (
                    base * (1 + team_gap_pct[t] / 100) + off
                    + (-0.4 if comp == "MEDIUM" else 0.0)
                    + (0.06 if comp == "MEDIUM" else 0.03) * life   # tyre deg
                    - 0.035 * lap                                    # fuel burn
                    + rng.normal(0, 0.25)
                )
                pit_in = lap == pit
                pit_out = lap == pit + 1
                lt += 2.0 if pit_in else (20.0 if pit_out else 0.0)
                t_cum += lt
                rows.append(dict(
                    Driver=drv, Team=t, LapNumber=lap, Compound=comp, TyreLife=life,
                    LapTime=pd.to_timedelta(lt, unit="s"), Time=pd.to_timedelta(t_cum, unit="s"),
                    PitInTime=pd.to_timedelta(t_cum, unit="s") if pit_in else pd.NaT,
                    PitOutTime=pd.to_timedelta(t_cum, unit="s") if pit_out else pd.NaT,
                    TrackStatus="1", IsAccurate=True,
                ))
    df = pd.DataFrame(rows)
    df["Position"] = df.groupby("LapNumber")["Time"].rank(method="first")
    return df


def make_season(n_races=10, upgrade_team="Alpha", upgrade_at=6, upgrade_gain_pct=0.5, seed=1):
    """Returns (list_of_lap_tables, truth_df). Alpha gets faster by `upgrade_gain_pct` from race `upgrade_at`."""
    rng = np.random.default_rng(seed)
    base_gap = {t: 0.25 * i for i, t in enumerate(TEAMS)}
    races, truth = [], []
    for r in range(1, n_races + 1):
        gaps = dict(base_gap)
        if r >= upgrade_at:
            gaps[upgrade_team] -= upgrade_gain_pct
        med = np.median(list(gaps.values()))
        for t, v in gaps.items():
            truth.append(dict(RaceIdx=r, Team=t, true_gap_pct=v - med))
        races.append(make_race(rng, gaps))
    return races, pd.DataFrame(truth)
