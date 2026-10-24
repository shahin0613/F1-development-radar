"""
Aston Martin B-spec step test, pre-specified from public reporting.

Upgrade timing is NOT read off the data: Aston Martin's own site and the trade press (F1Technical,
PlanetF1, Crash.net) report the B-spec AMR26 (16 new parts, new chassis-level aero) debuting at the
Hungarian GP, which is RaceIdx 11 in data/race_order_2026.csv. A second step (Honda power-unit
improvements) was reported for the Dutch GP (RaceIdx 12) onwards, so the post-upgrade window mixes
two changes and the step is NOT a clean aero-only estimate.

Outputs:
  * a table: step estimate for Aston at RaceIdx 11 (race + qualifying), window sensitivity,
    drop-one-race sensitivity, and a placebo (the same test on every other team)
  * figures/aston_step.png

Usage:  python -m devradar.aston_report
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from devradar.trend import step_effect  # noqa: E402

UPGRADE_RACE = 11          # Hungarian GP, from public reporting (see docstring)
TEAM = "Aston Martin"
DATA = {"race": "data/gaps_2026.csv", "qualifying": "data/gaps_quali_2026.csv"}


def _fmt(r):
    lo, hi = r["ci90"]
    return f"{r['gain_pct']:+.2f}%  90% CI [{lo:+.2f}, {hi:+.2f}]"


def tables():
    out = {}
    for name, path in DATA.items():
        g = pd.read_csv(path)
        print(f"\n=== {name} pace, {TEAM}, upgrade at RaceIdx {UPGRADE_RACE} ===")
        main = step_effect(g, TEAM, UPGRADE_RACE, window=3)
        print(f"main (3 races each side): {_fmt(main)}")
        print("window sensitivity:")
        for w in (2, 3, 4, 5):
            print(f"  w={w}: {_fmt(step_effect(g, TEAM, UPGRADE_RACE, window=w))}")
        print("drop one post-upgrade race (w=3):")
        for drop in (11, 12, 13):
            gg = g[~((g.Team == TEAM) & (g.RaceIdx == drop))]
            print(f"  drop R{drop}: {_fmt(step_effect(gg, TEAM, UPGRADE_RACE, window=3))}")
        print("placebo, same test on every team:")
        for t in sorted(g.Team.unique()):
            print(f"  {t:16s} {_fmt(step_effect(g, t, UPGRADE_RACE, window=3))}")
        out[name] = main
    return out


def figure(path="figures/aston_step.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (name, csv) in zip(axes, DATA.items()):
        g = pd.read_csv(csv)
        M = g.pivot(index="RaceIdx", columns="Team", values="gap_pct")
        for t in M.columns:
            if t != TEAM:
                ax.plot(M.index, M[t], color="0.8", lw=1, zorder=1)
        ax.plot(M.index, M[TEAM], "o-", color="#00665E", lw=2, ms=4, label=TEAM, zorder=3)
        ax.axvline(UPGRADE_RACE - 0.5, color="crimson", ls=":", lw=1.5, label="B-spec debut (Hungarian GP)")
        r = step_effect(g, TEAM, UPGRADE_RACE, window=3)
        ax.set_title(f"{name.capitalize()} pace: step {r['gain_pct']:+.2f}% "
                     f"(90% CI {r['ci90'][0]:+.2f} to {r['ci90'][1]:+.2f})", fontsize=10)
        ax.set_xlabel("Race index (chronological)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Gap to field, % of lap time (up = faster)")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Aston Martin vs the field, 2026 (grey = other teams). Upgrade timing from public reporting.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tables()
    print("\nsaved", figure())
