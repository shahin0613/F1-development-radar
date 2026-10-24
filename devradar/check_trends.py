import pandas as pd
from devradar.robustness import trend_with_circuit_control

for f in ["data/gaps_2026.csv", "data/gaps_quali_2026.csv"]:
    g = pd.read_csv(f)
    print(f)
    for team in ["Mercedes", "Aston Martin", "Ferrari", "Red Bull Racing", "McLaren"]:
        r = trend_with_circuit_control(g, team, k=2)
        lo, hi = r["ci90"]
        print(f"  {team:16s} {r['slope_pct_per_race']:+.3f}%/race  90% CI [{lo:+.3f}, {hi:+.3f}]")