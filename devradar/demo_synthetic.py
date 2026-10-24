"""End-to-end check on synthetic data: does the pipeline recover a planted upgrade?"""
import pandas as pd
from devradar.pace import race_gaps
from devradar.synthetic import make_season

races, truth = make_season(n_races=10, upgrade_team="Alpha", upgrade_at=6, upgrade_gain_pct=0.5)
frames = []
for i, laps in enumerate(races, start=1):
    g = race_gaps(laps)
    g["RaceIdx"] = i
    frames.append(g)
gaps = pd.concat(frames, ignore_index=True).merge(truth, on=["RaceIdx", "Team"])
gaps["err"] = gaps["gap_pct"] - gaps["true_gap_pct"]
print("Pace-extraction error vs truth (gap %):")
print(f"  mean err {gaps.err.mean():+.3f} | RMSE {(gaps.err**2).mean()**.5:.3f} | typical se {gaps.se_pct.mean():.3f}")

from devradar.trend import step_effect
print("\nPlanted: Alpha +0.50% faster from race 6 (Delta unchanged)")
for who in ("Alpha", "Delta"):
    print(who, step_effect(gaps, who, 6))
gaps.to_csv("synthetic_gaps.csv", index=False)

try:  # optional: GP trend plot needs requirements-gp.txt
    from devradar.trend import fit_all, plot_trends
    plot_trends(gaps, fit_all(gaps), "synthetic_trends.png")
except ImportError:
    print("(GP stack not installed; skipped trend plot. pip install -r requirements-gp.txt)")
