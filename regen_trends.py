import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import pandas as pd
from devradar.trend import fit_all, plot_trends

for csv, png in [
    ("data/gaps_2026.csv", "figures/trends_2026.png"),
    ("data/gaps_quali_2026.csv", "figures/trends_quali_2026.png"),
]:
    gaps = pd.read_csv(csv)
    models = fit_all(gaps)
    print("saved", plot_trends(gaps, models, png))