"""Drop into devradar/. Answers: does a team's trend survive removing shared circuit effects?

Circuit effects are estimated from the OTHER teams (leave-one-team-out SVD of the races x teams gap matrix),
then the team's gap is regressed on race index + k circuit factors. Uncertainty is a leave-one-race-out
jackknife, which is honest at n~17 (HAC/asymptotic p-values are not).
Caveat: the factors also contain the other teams' own development, so this is a robustness check, not proof.
"""
import numpy as np
import pandas as pd


def _slope(y, x, F):
    X = np.column_stack([np.ones_like(x), x] + ([F] if F.size else []))
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def trend_with_circuit_control(gaps: pd.DataFrame, team: str, k: int = 1) -> dict:
    """slope_pct_per_race > 0 means the team got FASTER relative to the field per race."""
    M = gaps.pivot(index="RaceIdx", columns="Team", values="gap_pct")
    others = M.drop(columns=team)
    R = (others.fillna(others.mean()) - others.mean()).to_numpy()
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    F = U[:, :k] * S[:k] if k else np.empty((len(M), 0))
    ok = M[team].notna().to_numpy()
    y, x, F = M[team].to_numpy()[ok], M.index.to_numpy(float)[ok], F[ok]
    b = -_slope(y, x, F)  # gap shrinking = faster
    n = len(y)
    jk = np.array([-_slope(np.delete(y, i), np.delete(x, i), np.delete(F, i, axis=0)) for i in range(n)])
    se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    return {"slope_pct_per_race": float(b), "jackknife_se": float(se),
            "ci90": (float(b - 1.645 * se), float(b + 1.645 * se)), "n": n, "k_factors": k}


def check_race_order(order_dates: pd.Series):
    """Call in chronological_events(): NaT sorts LAST, so a race with missing dates silently lands at the end."""
    if order_dates.isna().any():
        raise ValueError("some events have no lap-start dates; calendar order would be wrong")