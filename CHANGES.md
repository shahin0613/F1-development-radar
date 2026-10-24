# What changed from the previous repo (you can delete this file)

**Structure**
- Flat files restored into a `devradar/` package (the code imported `devradar.*`, so it could not run flat). Data in `data/`, figures in `figures/`, tests in `tests/`.
- File names normalised (`Causal model.py` -> `devradar/causal_model.py`, `Mde simulation.py` -> `devradar/mde_simulation.py`, `Robustness.py` -> `devradar/robustness.py`, `Test no leakage.py` -> `tests/test_no_leakage.py`). README commands now match the files.
- Windows line endings converted to LF. `requirements.txt` was missing `statsmodels` (and the GP stack); split into core / GP / dev.

**Removed**
- `joint_model_sklearn.py`, `analyze_performance.py`, `true_car_performance.png`: the old plot used `ffill().bfill()` and PCA over all races (future data leaked into the "track-adjusted" curve) and drew the upgrade line at race 11 by hand, so the step was partly manufactured. Replaced by `devradar/aston_report.py` + `figures/aston_step.png`, which uses raw field-relative gaps and a step test with timing from public reporting.

**Fixed**
- The "honest" track-factor forecast was actually median + each team's mean past residual (a momentum term). The README's "no gain from a track factor" claim was therefore not testing a track factor. Now: honest forecast = rolling median, track factor only as an oracle upper bound, with the reasoning in the README.
- `Test no leakage.py` imported a module that did not exist (`backtest_v2`) and only covered one race and one mode. Now parametrised over races and both modes, plus a test that honest mode ignores the target race's descriptors.
- `mde_simulation` used a nested f-string that is a syntax error before Python 3.12.
- Interval coverage now reported raw and causally recalibrated.
- `trend.py` no longer needs TensorFlow just to import; the GP part is optional so `step_effect`, tests and backtests run anywhere.
- Optional `GITHUB_TOKEN` for the data download (the GitHub API rate-limits unauthenticated calls).

**Added in the second round**
- Y-axis label on `aston_step.png` now says "up = faster" (the axis is inverted). README notes that race 15 has no race-pace estimate.
- `validate_synthetic` now runs 200 simulated seasons instead of 30, so its false-positive rate means something.
- New corner-speed-class analysis from qualifying telemetry: `devradar/corners.py`, `devradar/corner_report.py`, `data/corner_gaps_quali_2026.csv`, `figures/aston_corner_classes.png`, `tests/test_corners.py`, and a README section. A first run with a looser telemetry-length filter let corrupted Aston laps into Singapore (a 9% straight-line "gap"); the strict filter is restored and the README says which team-races lack estimates.

**Not changed / not verified here**
- Raw-lap download and the GP plots were not re-run end to end in my environment (GitHub API rate limit; TensorFlow not installed). `data/*.csv` are your original files; every number in the README comes from re-running the code on them.
- `figures/trends_2026.png` and `figures/trends_quali_2026.png` are your original GP plots, not regenerated.
- License is MIT under your name: change it if you prefer something else.
