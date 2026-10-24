"""
Validation pipeline for synthetic F1 telemetry data.
Evaluates the accuracy of the upgrade detection model and false-positive rates.
"""
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from devradar.pace import race_gaps
from devradar.synthetic import make_season
from devradar.trend import step_effect

def run_synthetic_validation(n_simulations: int = 30, n_races: int = 10, upgrade_race: int = 6):
    """
    Runs Monte Carlo simulations to validate the pace-extraction model against ground truth.
    """
    coverage_count = 0
    detected_count = 0
    false_positive_count = 0
    alpha_gains = []

    for seed_offset in range(n_simulations):
        # 1. Generate Synthetic Season
        races, _ = make_season(n_races=n_races, seed=500 + seed_offset)
        
        # 2. Extract Pace Gaps
        frames = []
        for idx, laps in enumerate(races, start=1):
            gap_df = race_gaps(laps)
            gap_df["RaceIdx"] = idx
            frames.append(gap_df)
        
        season_gaps = pd.concat(frames, ignore_index=True)
        
        # 3. Estimate Upgrade Effects (Interrupted Time Series)
        alpha_effect = step_effect(season_gaps, "Alpha", upgrade_race)
        delta_effect = step_effect(season_gaps, "Delta", upgrade_race)
        
        alpha_ci_low, alpha_ci_high = alpha_effect["ci90"]
        delta_ci_low, delta_ci_high = delta_effect["ci90"]
        
        # 4. Record Metrics
        if alpha_ci_low <= 0.50 <= alpha_ci_high:
            coverage_count += 1
        
        if alpha_ci_low > 0:
            detected_count += 1
            
        if delta_ci_low > 0 or delta_ci_high < 0:
            false_positive_count += 1
            
        alpha_gains.append(alpha_effect["gain_pct"])

    # 5. Output Final Audit Report
    print("\n" + "="*50)
    print("SYNTHETIC VALIDATION PIPELINE RESULTS")
    print("="*50)
    print(f"Alpha Team (Planted True Effect: +0.50%)")
    print(f"  -> Mean Estimated Gain : {np.mean(alpha_gains):.3f}%")
    print(f"  -> 90% CI Coverage     : {coverage_count / n_simulations:.2%}")
    print(f"  -> Detection Power     : {detected_count / n_simulations:.2%}")
    print("-" * 50)
    print(f"Delta Team (Planted True Effect: 0.00%)")
    print(f"  -> False-Positive Rate : {false_positive_count / n_simulations:.2%} (Target ~10.0%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_synthetic_validation(n_simulations=200)