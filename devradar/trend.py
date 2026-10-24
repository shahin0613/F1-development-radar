"""
Development trends: one Gaussian process per team over the season.

x = race index, y = gap_pct (lower = faster). The GP gives a smooth trend plus
uncertainty bands, and lets us ask "did this upgrade actually work?" as a
probability instead of eyeballing a line.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # tfp compatibility on newer TF

import numpy as np
import pandas as pd

# The Gaussian-process functions need the optional GP stack (requirements-gp.txt: gpflow +
# tensorflow-probability). `step_effect` below needs only numpy/scipy, so the core pipeline,
# tests and backtests run without TensorFlow.


def fit_team_gp(race_idx, gap_pct, se_pct, fix_noise: bool = False, noise_inflate: float = 1.0,
                ls_bounds=(3.0, 15.0)):
    import tensorflow_probability as tfp
    import gpflow
    from gpflow.utilities import to_default_float

    X = np.asarray(race_idx, float).reshape(-1, 1)
    Y = np.asarray(gap_pct, float).reshape(-1, 1)

    kernel = gpflow.kernels.Matern32(variance=0.1)
    # keep the lengthscale in a sane range (in races): avoids interpolating noise
    kernel.lengthscales = gpflow.Parameter(
        max(4.0, ls_bounds[0] + 0.5), transform=tfp.bijectors.Sigmoid(to_default_float(ls_bounds[0]), to_default_float(ls_bounds[1]))
    )
    model = gpflow.models.GPR(
        (X, Y),
        kernel=kernel,
        mean_function=gpflow.mean_functions.Constant(float(Y.mean())),
        # shared noise, initialised from the measured standard errors.
        # v2 idea: per-point noise from se_pct (heteroscedastic likelihood).
        noise_variance=float(max(np.mean(np.asarray(se_pct) ** 2) * noise_inflate, 1e-3)),
    )
    floor = float(max(np.mean(np.asarray(se_pct) ** 2), 1e-4))
    if fix_noise:  # trust the measured standard errors exactly (fine on synthetic data)
        gpflow.set_trainable(model.likelihood.variance, False)
    else:
        # real races carry extra scatter the standard errors don't capture (circuit-suits-car effects).
        # Let the noise grow, but never below the measured standard errors.
        model.likelihood.variance = gpflow.Parameter(
            max(2 * floor, 1e-3), transform=gpflow.utilities.positive(lower=floor)
        )
    gpflow.optimizers.Scipy().minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=300)
    )
    return model


def fit_all(gaps: pd.DataFrame, min_races: int = 5) -> dict:
    models = {}
    for team, g in gaps.groupby("Team"):
        if len(g) >= min_races:
            models[team] = fit_team_gp(g["RaceIdx"], g["gap_pct"], g["se_pct"])
    return models


def upgrade_effect(model, race_before: float, race_after: float, n: int = 4000, seed: int = 0) -> dict:
    """
    Posterior of the pace GAIN between two points in the season.
    gain > 0 means the car got faster (gap % shrank).
    """
    import tensorflow as tf

    tf.random.set_seed(seed)
    xs = np.array([[race_before], [race_after]], float)
    s = model.predict_f_samples(xs, n)[:, :, 0].numpy()  # (n, 2)
    gain = s[:, 0] - s[:, 1]
    lo, hi = np.percentile(gain, [5, 95])
    return {
        "gain_pct_mean": float(gain.mean()),
        "ci90": (float(lo), float(hi)),
        "p_faster": float((gain > 0).mean()),
    }


def plot_trends(gaps: pd.DataFrame, models: dict, out_png: str = "trends.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teams = list(models)
    cols = 3
    rows = int(np.ceil(len(teams) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    xmax = gaps["RaceIdx"].max()
    grid = np.linspace(1, xmax, 120).reshape(-1, 1)
    for ax, team in zip(axes, teams):
        g = gaps[gaps["Team"] == team]
        mu, var = models[team].predict_f(grid)
        mu, sd = mu.numpy().ravel(), np.sqrt(var.numpy().ravel())
        ax.fill_between(grid.ravel(), mu - 1.96 * sd, mu + 1.96 * sd, alpha=0.25)
        ax.plot(grid, mu)
        ax.errorbar(g["RaceIdx"], g["gap_pct"], yerr=g["se_pct"], fmt="o", ms=3, capsize=2, color="k")
        ax.set_title(team, fontsize=10)
        ax.invert_yaxis()  # up = faster
    for ax in axes[len(teams):]:
        ax.axis("off")
    fig.supxlabel("Race #")
    fig.supylabel("Gap to field (trimmed mean), % (up = faster)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return out_png


def step_effect(gaps: pd.DataFrame, team: str, upgrade_race: int, window: int = 3, z: float = 1.645) -> dict:
    """
    Explicit before/after test for an upgrade that landed at a KNOWN race.

    Inverse-variance-weighted mean gap over `window` races before vs after.
    Use this for "did upgrade X work?"; use the GP for the overall development
    trend. (A GP is a smoother and blurs sharp steps, so it under-reads them.)
    gain_pct > 0 means the car got faster.
    """
    from scipy.stats import norm

    g = gaps[gaps["Team"] == team]
    pre = g[(g["RaceIdx"] < upgrade_race) & (g["RaceIdx"] >= upgrade_race - window)]
    post = g[(g["RaceIdx"] >= upgrade_race) & (g["RaceIdx"] < upgrade_race + window)]
    if len(pre) < 2 or len(post) < 2:
        raise ValueError("need >=2 races on each side of the upgrade")

    # Real races carry scatter beyond the per-race standard errors (circuit-suits-car effects etc.).
    # Estimate it from the team's whole season and add it to every race's variance. Deliberately
    # conservative: any genuine development also counts as "scatter", which only widens the interval.
    extra = float(max(np.var(g["gap_pct"], ddof=1) - np.mean(g["se_pct"] ** 2), 0.0)) if len(g) >= 4 else 0.0

    def wmean(d):
        w = 1.0 / (d["se_pct"] ** 2 + extra)
        return (w * d["gap_pct"]).sum() / w.sum(), (1.0 / w.sum()) ** 0.5

    (a, sa), (b, sb) = wmean(pre), wmean(post)
    gain, se = a - b, float(np.hypot(sa, sb))
    return {
        "gain_pct": float(gain),
        "ci90": (float(gain - z * se), float(gain + z * se)),
        "p_faster": float(norm.cdf(gain / se)),
        "extra_scatter_sd": float(extra ** 0.5),
        "n_pre": len(pre),
        "n_post": len(post),
    }
