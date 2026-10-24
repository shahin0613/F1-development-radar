"""
Usage (default source = TracingInsights GitHub cache; works when FastF1's own download is blocked):
    python -m devradar.run --year 2026
    python -m devradar.run --year 2026 --rounds 1-10
    python -m devradar.run --year 2026 --session qualifying       # qualifying pace (cleaner read of raw car speed)
    python -m devradar.run --year 2026 --upgrade "Aston Martin:9" --upgrade "Ferrari:7"
    python -m devradar.run --year 2025 --source fastf1 --rounds 1-10      # original FastF1 download path

--upgrade "Team:N" means: the upgrade first raced at race N (see race_order_<year>.csv in --out for the numbering).
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from devradar.pace import build_season
from devradar.quali import build_quali_season
from devradar.source_tracing import build_season_tracing
from devradar.trend import fit_all, plot_trends, step_effect


def parse_rounds(s):
    a, b = s.split("-")
    return int(a), int(b)


ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, required=True)
ap.add_argument("--source", choices=["tracing", "fastf1"], default="tracing")
ap.add_argument("--session", choices=["race", "qualifying"], default="race", help="tracing source only")
ap.add_argument("--rounds", default=None, help='e.g. "1-10" in calendar order; default = all available')
ap.add_argument("--upgrade", action="append", default=[], help='"Team:race_number" (repeatable)')
ap.add_argument("--out", default="out")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

first, last = parse_rounds(args.rounds) if args.rounds else (None, None)
tag = f"{args.year}" if args.session == "race" else f"quali_{args.year}"
if args.source == "tracing" and args.session == "qualifying":
    gaps = build_quali_season(args.year, first=first, last=last)
elif args.source == "tracing":
    gaps, order = build_season_tracing(args.year, first=first, last=last)
    order.to_csv(f"{args.out}/race_order_{args.year}.csv", index=False)
    print("\nRace numbering:\n" + order.to_string(index=False) + "\n")
else:
    gaps = build_season(args.year, list(range(first or 1, (last or 10) + 1)))
gaps.to_csv(f"{args.out}/gaps_{tag}.csv", index=False)

try:  # GP trend plots are optional (pip install -r requirements-gp.txt)
    models = fit_all(gaps)
    print("saved", plot_trends(gaps, models, f"{args.out}/trends_{tag}.png"))
except ImportError:
    print("[info] GP stack not installed, skipping trend plots (pip install -r requirements-gp.txt)")

for spec in args.upgrade:
    team, idx = spec.rsplit(":", 1)
    try:
        r = step_effect(gaps, team, int(idx))
    except Exception as e:  # noqa: BLE001
        print(f"{team} @ race {idx}: could not test ({e})")
        continue
    lo, hi = r["ci90"]
    print(f"{team} upgrade @ race {idx}: {r['gain_pct']:+.2f}% pace gain, 90% CI [{lo:+.2f}, {hi:+.2f}], P(faster)={r['p_faster']:.2f}")

