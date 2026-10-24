"""
Where did Aston Martin's B-spec gain time? Corner-speed-class step test (qualifying telemetry).

Reads data/corner_gaps_quali_2026.csv (built by `python -m devradar.corners`) and, for Aston Martin at
the Hungarian GP (RaceIdx 11, timing from public reporting, see aston_report.py), reports per class
(Low / Medium / High corners, Straight):
  * step in class TIME (positive = got faster) with 90% CI, using the same step test as the lap-time gap
  * step in apex (minimum) speed for the corner classes
  * contribution to the overall lap-time step = class step x field share of the lap
  * placebo: the same test on every other team

Usage:  python -m devradar.corner_report
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from devradar.trend import step_effect  # noqa: E402

TEAM, UPGRADE_RACE, WINDOW = "Aston Martin", 11, 3
CLASSES = ["Low", "Medium", "High", "Straight"]
CSV = "data/corner_gaps_quali_2026.csv"


def _step(c, team, metric, cls, window=WINDOW):
    g = c[(c.Metric == metric) & (c.Class == cls)][["Team", "RaceIdx", "gap_pct", "se_pct"]]
    try:
        return step_effect(g, team, UPGRADE_RACE, window=window)
    except Exception:  # noqa: BLE001  (too few races on one side)
        return None


def _fmt(r):
    if r is None:
        return "n/a (too few races with this class on one side)"
    lo, hi = r["ci90"]
    return f"{r['gain_pct']:+.2f}%  90% CI [{lo:+.2f}, {hi:+.2f}]  (n_pre={r['n_pre']}, n_post={r['n_post']})"


def tables(c):
    share = c[(c.Metric == "time")].groupby("Class")["share"].mean()
    print(f"=== {TEAM}: qualifying step at RaceIdx {UPGRADE_RACE}, {WINDOW} races each side (positive = faster) ===")
    print("\nTime in class:")
    contrib = {}
    for cls in CLASSES:
        r = _step(c, TEAM, "time", cls)
        print(f"  {cls:9s} {_fmt(r)}")
        if r is not None:
            contrib[cls] = r["gain_pct"] * share[cls]
    print("\nApex (minimum) speed, 'faster' = higher:")
    for cls in CLASSES[:3]:
        r = _step(c, TEAM, "apex_speed", cls)
        print(f"  {cls:9s} {_fmt(r)}")
    tot = sum(contrib.values())
    print("\nContribution to the overall lap-time step (class step x average field share of the lap):")
    for cls, v in contrib.items():
        print(f"  {cls:9s} {v:+.2f} pct points of lap time   (share of lap {share[cls]:.0%})")
    print(f"  {'sum':9s} {tot:+.2f}   (compare the lap-time step from aston_report.py)")
    print("\nWindow sensitivity (class time step):")
    for cls in ("Low", "Straight"):
        for w in (2, 3, 4, 5):
            print(f"  {cls:9s} w={w}: {_fmt(_step(c, TEAM, 'time', cls, w))}")
    print("\nPlacebo (same test, class time, every team):")
    for cls in ("Low", "Straight"):
        print(f"  -- {cls}")
        for t in sorted(c.Team.unique()):
            print(f"    {t:16s} {_fmt(_step(c, t, 'time', cls))}")


def figure(c, path="figures/aston_corner_classes.png"):
    t = c[(c.Metric == "time") & (c.Team == TEAM)]
    wide = t.pivot(index="RaceIdx", columns="Class", values="gap_pct")
    sh = c[c.Metric == "time"].groupby("Class")["share"].mean()
    pre = range(UPGRADE_RACE - WINDOW, UPGRADE_RACE)
    post = range(UPGRADE_RACE, UPGRADE_RACE + WINDOW)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    x = np.arange(len(CLASSES))
    pre_v = [wide.loc[wide.index.isin(pre), k].mean() for k in CLASSES]
    post_v = [wide.loc[wide.index.isin(post), k].mean() for k in CLASSES]
    ax.bar(x - 0.2, pre_v, 0.4, label=f"before (races {min(pre)}-{max(pre)})", color="0.65")
    ax.bar(x + 0.2, post_v, 0.4, label=f"after (races {min(post)}-{max(post)})", color="#00665E")
    ax.set_xticks(x, [f"{k}\n({sh[k]:.0%} of lap)" for k in CLASSES])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Time in class vs field, % (positive = slower)")
    ax.set_title("Where Aston Martin loses time in qualifying", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    for k, col in (("Straight", "#C0392B"), ("Low", "#00665E"), ("Medium", "#7F8C8D")):
        ax.plot(wide[k].dropna().index, wide[k].dropna(), "o-", ms=3, lw=1.6, color=col, label=k)
    ax.axvline(UPGRADE_RACE - 0.5, color="crimson", ls=":", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Race index (chronological)")
    ax.set_title("Aston Martin time gap by class (dotted = B-spec debut)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data = pd.read_csv(CSV)
    tables(data)
    print("\nsaved", figure(data))
