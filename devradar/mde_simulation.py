"""
Power curve and minimum detectable effect (MDE) for a mid-season step change.

Noise is calibrated from the real gap matrix (not invented):
  * race effect  : sd of the cross-team mean in each race (weather/SC/track evolution hit everyone)
  * team scatter : sd of a team's race-to-race residual around its own median
Estimator: step = mean(post) - mean(pre) of (team gap - median of other teams), which cancels
the race effect. One-sided Welch test at 5% (equivalent to a 90% two-sided CI).
Sign convention: positive gap = slower, so a successful upgrade is a NEGATIVE shift; effects are
reported as "gain" (positive = faster).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats


def calibrate(Y):
    race_sd = Y.mean(axis=1).std()
    resid = Y - Y.median()
    team_sd = resid.sub(resid.mean(axis=1), axis=0).stack().std()
    return race_sd, team_sd


def simulate(gain, n_races, upgrade_at, race_sd, team_sd, n_other=10, n_sims=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    hits = 0
    est = []
    for _ in range(n_sims):
        race_eff = rng.normal(0, race_sd, n_races)
        target = race_eff + rng.normal(0, team_sd, n_races)
        target[upgrade_at - 1:] -= gain                      # faster = smaller gap
        others = race_eff[:, None] + rng.normal(0, team_sd, (n_races, n_other))
        d = target - np.median(others, axis=1)
        pre, post = d[: upgrade_at - 1], d[upgrade_at - 1:]
        t, p = stats.ttest_ind(pre, post, equal_var=False, alternative="greater")  # pre slower than post
        est.append(pre.mean() - post.mean())
        hits += p < 0.05
    return hits / n_sims, float(np.mean(est)), float(np.std(est))


def main(gaps_csv="data/gaps_quali_2026.csv", n_races=None):
    g = pd.read_csv(gaps_csv)
    Y = g.pivot(index="RaceIdx", columns="Team", values="gap_pct")
    n_races = n_races or len(Y)
    race_sd, team_sd = calibrate(Y)
    print(f"calibrated: race-effect sd = {race_sd:.3f}%, team scatter sd = {team_sd:.3f}%\n")
    for upgrade_at in (5, 9, 13):
        print(f"Upgrade first raced at race {upgrade_at} of {n_races}  (pre={upgrade_at-1}, post={n_races-upgrade_at+1} races)")
        mde = None
        for gain in np.arange(0.0, 1.55, 0.1):
            power, mean_est, sd_est = simulate(gain, n_races, upgrade_at, race_sd, team_sd)
            tag = "false-positive rate" if gain == 0 else "power"
            print(f"  gain {gain:4.2f}%  {tag:19s} {power:5.1%}   est {mean_est:+.3f} +/- {sd_est:.3f}")
            if mde is None and gain > 0 and power >= 0.8:
                mde = gain
        mde_txt = f"{mde:.1f}" if mde is not None else ">1.5"
        print(f"  -> MDE at 80% power ~ {mde_txt}%\n")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))