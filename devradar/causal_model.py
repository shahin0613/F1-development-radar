"""
Causal (no-lookahead) pace forecasts and a track-factor "oracle" ceiling.

For a target race r every quantity uses ONLY rows 0..r-1 of the gap matrix.

  base forecast : median of each team's last k observed races (no ffill/bfill).
  sigma         : std of each team's prior one-step residuals (gap - rolling median of the k before it).
  track factor  : PCA on row-centred, standardised residuals; scores are regressed (ridge) on circuit
                  descriptors. The factor's contribution to the target race is only computed when
                  oracle=True.

Why there is no "honest" track-factor forecast here
---------------------------------------------------
The descriptors in circuit_descriptors.csv (top_speed, mean_trap, track_temp) are all measured DURING
the race weekend, so they are unknown when a pre-race forecast is made. Without them the factor has
nothing to predict from, so the only honest pre-race forecast is the rolling median (an earlier version
added each team's mean past residual instead; that is a momentum term, not a track effect, and it
only added noise).

  oracle=False : returns the rolling median (the real pre-race forecast).
  oracle=True  : adds the track factor predicted from the TARGET race's own descriptors. This is an
                 upper bound on what a track factor could contribute, not a usable forecast.

A genuine pre-race track factor needs descriptors known in advance (circuit geometry, or the same
circuit's values from a previous season). That is future work, see README.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

DESCRIPTORS = ["top_speed", "mean_trap", "track_temp"]


def _center_rows(Y):
    """Gaps are relative to the field, so remove each race's cross-team mean."""
    return Y.sub(Y.mean(axis=1), axis=0)


def base_pace(history: pd.DataFrame, k: int = 3) -> pd.Series:
    return history.iloc[-k:].median(skipna=True)


def prior_residuals(history: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """resid_t = Y_t - median(Y_{t-k..t-1}); row t never sees itself or the future."""
    prior = history.rolling(k, min_periods=1).median().shift(1)
    return (history - prior).iloc[1:]


def forecast_race(Y: pd.DataFrame, D: pd.DataFrame, r: int, k: int = 3,
                  n_components: int = 1, oracle: bool = False, ridge_alpha: float = 10.0):
    """Return (point forecast Series, sigma Series) for race row r using rows < r only
    (plus, if oracle=True, the target race's own descriptors)."""
    history = Y.iloc[:r]
    base = base_pace(history, k)
    resid = prior_residuals(history, k)
    sigma = resid.std(skipna=True).fillna(resid.stack().std())

    if not oracle or D is None or len(resid) < 6:
        return base, sigma

    resid_c = _center_rows(resid)
    scale = resid_c.std(skipna=True).replace(0, 1)
    Z = (resid_c / scale).fillna(0.0)
    pca = PCA(n_components=n_components).fit(Z.values)
    scores = pca.transform(Z.values)                      # (n_hist-1, n_comp)

    X_hist = D.iloc[1:r][DESCRIPTORS]                     # rows aligned with resid
    mu, sd = X_hist.mean(), X_hist.std().replace(0, 1)
    reg = Ridge(alpha=ridge_alpha).fit(((X_hist - mu) / sd).values, scores)

    x_new = ((D.iloc[[r]][DESCRIPTORS] - mu) / sd).values
    pred_score = np.asarray(reg.predict(x_new)).reshape(1, -1)
    # loading * predicted score only (no PCA mean term): a pure track-shape adjustment
    track = (pred_score @ pca.components_)[0] * scale.values
    return base + pd.Series(track, index=Y.columns), sigma
