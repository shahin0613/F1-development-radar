"""
Rolling-origin backtest of one-race-ahead pace forecasts. Leakage-free.

Methods (all scored on the RAW gap, so they are judged on the same quantity):
  naive   : last observed race
  median  : median of the last k races  <- the real pre-race benchmark
  oracle  : median + track factor predicted from the TARGET race's own descriptors
            (weekend-measured, so an UPPER BOUND on a track factor, not a usable forecast)

Reports:
  * MAE per method
  * paired bootstrap CI of the MAE difference, resampled by RACE (teams in a race are correlated)
  * 90% interval coverage for the median forecast, with the textbook +/-1.645*sigma interval and with a
    causally recalibrated multiplier (empirical 90% quantile of past |error|/sigma, past races only)
"""
import sys
import numpy as np
import pandas as pd

from devradar.causal_model import forecast_race

Z90 = 1.645


def load(gaps_csv, desc_csv):
    g = pd.read_csv(gaps_csv)
    Y = g.pivot(index="RaceIdx", columns="Team", values="gap_pct").sort_index()
    S = g.pivot(index="RaceIdx", columns="Team", values="se_pct").sort_index()
    D = pd.read_csv(desc_csv).set_index("RaceIdx").sort_index().reindex(Y.index)
    return Y, S, D


def run(Y, D, start=6, k=3):
    rows = []
    for r in range(start, len(Y)):
        actual = Y.iloc[r]
        naive = Y.iloc[:r].ffill().iloc[-1]                 # last observed (causal)
        med, sig = forecast_race(Y, D, r, k, oracle=False)
        oracle, _ = forecast_race(Y, D, r, k, oracle=True)
        for t in Y.columns:
            if np.isnan(actual[t]):
                continue
            rows.append(dict(race=r + 1, team=t, actual=actual[t], naive=naive[t], median=med[t],
                             oracle=oracle[t], sigma=sig[t]))
    return pd.DataFrame(rows)


def calibrated_multiplier(df, min_races=3, level=0.90):
    """Per-race multiplier m_r = empirical `level` quantile of |err|/sigma over races BEFORE r."""
    z = ((df.actual - df["median"]).abs() / df.sigma)
    mult = {}
    for r in sorted(df.race.unique()):
        past = z[df.race < r].dropna()
        n_past = df[df.race < r].race.nunique()
        mult[r] = np.quantile(past, level) if n_past >= min_races and len(past) else Z90
    return df.race.map(mult)


def report(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    methods = ["naive", "median", "oracle"]
    err = {m: (df[m] - df.actual).abs() for m in methods}
    print(f"{len(df)} team-race forecasts over races {df.race.min()}-{df.race.max()}\n")
    for m in methods:
        print(f"  MAE {m:7s}: {err[m].mean():.3f}")
    races = df.race.unique()
    print("\nPaired bootstrap (resampling races) of MAE difference vs median; negative = better than median")
    for m in ["naive", "oracle"]:
        d = (err[m] - err["median"]).groupby(df.race).mean()
        boots = [d.loc[rng.choice(races, len(races))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [5, 95])
        print(f"  {m:7s}: {d.mean():+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]")
    inside = ((df.actual - df["median"]).abs() <= Z90 * df.sigma)
    print(f"\n90% interval coverage, median +/- 1.645*sigma         : {inside.mean():.1%}   (target 90%)")
    m = calibrated_multiplier(df)
    inside_c = ((df.actual - df["median"]).abs() <= m * df.sigma)
    print(f"90% interval coverage, causally recalibrated multiplier: {inside_c.mean():.1%}   (target 90%)")
    e = df.assign(e=df.actual - df["median"]).pivot(index="race", columns="team", values="e")
    common = e.mean(axis=1)
    print(f"\nShare of median-forecast error variance common across teams in a race: {common.var() / e.stack().var():.1%}")
    return dict(mae={m_: float(err[m_].mean()) for m_ in methods}, coverage=float(inside.mean()),
                coverage_recalibrated=float(inside_c.mean()))


if __name__ == "__main__":
    gaps = sys.argv[1] if len(sys.argv) > 1 else "data/gaps_quali_2026.csv"
    desc = sys.argv[2] if len(sys.argv) > 2 else "data/circuit_descriptors.csv"
    Y, S, D = load(gaps, desc)
    report(run(Y, D))
