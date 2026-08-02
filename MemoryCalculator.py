"""Hurst-exponent estimation from a price series.

INPUT CONTRACT (read this before calling anything here)
-------------------------------------------------------
Every estimator in this module measures how a fluctuation statistic scales
with window size, and every one of them must see the *cumulative profile*
(the fBm-like path), not the increments (the fGn-like returns).

Textbook DFA/MFDFA/R-S descriptions bundle that integration step inside the
estimator and are documented as taking increments; the implementations here
historically did not integrate at all, so callers passing log-returns were
measuring the roughness of the noise rather than the memory of the process.
On synthetic fGn with true H = 0.75 that produced H_hat ~ 0.03-0.13.

The fix is an explicit contract rather than a convention.  Each estimator
takes `input_type`:

    input_type="increments"  (default) -- you are passing returns / fGn;
                                          the estimator integrates for you.
    input_type="profile"                -- you are passing a price level or
                                          log-price path / fBm; used as-is.

Pass whichever you actually have and the estimator does the right thing.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression


INCREMENTS = "increments"
PROFILE = "profile"


def _as_profile(series, input_type):
    """Coerce an input series to the cumulative profile the estimators need.

    Increments are demeaned before integration so that a nonzero drift does
    not add a deterministic linear trend to the profile.  (DFA detrends each
    window linearly and so is largely immune, but the variogram and Higuchi
    estimators are not.)
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if input_type == PROFILE:
        return x
    if input_type == INCREMENTS:
        return np.cumsum(x - x.mean())
    raise ValueError(
        f"input_type must be {INCREMENTS!r} or {PROFILE!r}, got {input_type!r}"
    )


def _windows(profile, lag):
    """Non-overlapping complete windows of length `lag`.

    Only complete windows are used.  The previous version kept the ragged
    tail chunk and admitted chunks of length 2, which a linear fit detrends
    to exactly zero residual -- injecting spurious zeros into the fluctuation
    average and biasing the log-log slope.
    """
    n_windows = len(profile) // lag
    if n_windows == 0:
        return np.empty((0, lag))
    return profile[: n_windows * lag].reshape(n_windows, lag)


def _detrended_variance(window_matrix):
    """Per-window mean squared residual after removing a linear trend."""
    lag = window_matrix.shape[1]
    x = np.arange(lag, dtype=float)
    # Least-squares line per row, vectorised.
    coeffs = np.polyfit(x, window_matrix.T, 1)
    trend = np.vstack([np.polyval(c, x) for c in coeffs.T])
    residual = window_matrix - trend
    return np.mean(residual**2, axis=1)


# ---------------------------------------------------------------------------
# Variogram estimator (previously, and inaccurately, named hurst_rs)
# ---------------------------------------------------------------------------
def hurst_variogram(series, max_lag=50, input_type=INCREMENTS):
    """Estimate H from the scaling of lagged increments of the profile.

    Despite its former name this is *not* rescaled-range analysis: there is
    no range, no rescaling by the running standard deviation, and no R/S
    statistic anywhere in it.  It is a variogram (structure-function)
    estimator, which for a self-similar profile obeys

        std( X[t+k] - X[t] )  ~  k^H

    so the slope of log(std) against log(k) estimates H directly.
    """
    profile = _as_profile(series, input_type)
    lags = np.arange(2, max_lag)
    tau = np.array(
        [np.std(profile[lag:] - profile[:-lag]) for lag in lags]
    )
    ok = tau > 0
    poly = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)
    return poly[0]


def hurst_rs(series, max_lag=50, input_type=INCREMENTS):
    """Deprecated alias for `hurst_variogram` (this was never R/S)."""
    import warnings

    warnings.warn(
        "hurst_rs is a misnomer -- this estimator is a variogram, not "
        "rescaled range. Use hurst_variogram instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return hurst_variogram(series, max_lag=max_lag, input_type=input_type)


# ---------------------------------------------------------------------------
# Detrended Fluctuation Analysis
# ---------------------------------------------------------------------------
def hurst_dfa(series, max_lag=50, input_type=INCREMENTS):
    """Estimate H by detrended fluctuation analysis.

    F(s) = sqrt( mean over windows of the mean squared linear-detrended
    residual ) scales as s^H for a self-similar profile.

    Note the fluctuation function is a root-mean-square *across* windows;
    the previous version averaged the per-window RMS values, which is a
    different (and biased) statistic.
    """
    profile = _as_profile(series, input_type)
    lags = np.arange(10, max_lag)
    F, used = [], []
    for lag in lags:
        windows = _windows(profile, lag)
        if len(windows) < 2:
            continue
        f = np.sqrt(np.mean(_detrended_variance(windows)))
        if f > 0:
            F.append(f)
            used.append(lag)
    poly = np.polyfit(np.log(used), np.log(F), 1)
    return poly[0]


# ---------------------------------------------------------------------------
# Higuchi fractal dimension
# ---------------------------------------------------------------------------
def higuchi_fd(series, k_max=10, input_type=INCREMENTS):
    """Higuchi fractal dimension of the profile curve.

    Higuchi's method measures the length of a *curve*, so it too needs the
    profile.  For fBm the relation to the Hurst exponent is FD = 2 - H, so a
    persistent series (H = 0.75) should give FD ~ 1.25.

    Two corrections relative to the original implementation:

      * the profile, not the increments, is the curve whose length is measured;
      * Higuchi's normalisation carries a factor 1/k that was missing, which
        shifted the fitted exponent by exactly 1 (the estimator returned
        1 - H instead of 2 - H).  This was verified on synthetic fGn: at
        H = 0.30/0.50/0.75/0.90 the old code returned FD = 0.70/0.50/0.26/0.14.

    The interval count is `len(x) - 1` (the number of increments actually
    summed), not `len(x)`.
    """
    profile = _as_profile(series, input_type)
    n = len(profile)
    L = []
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(k):
            x = profile[m::k]
            if len(x) < 2:
                continue
            n_intervals = len(x) - 1
            Lkm = np.sum(np.abs(np.diff(x))) * (n - 1) / (n_intervals * k * k)
            Lk += Lkm
        L.append(np.log(Lk / k))
    x = np.log(np.arange(1, k_max + 1))
    y = np.array(L)
    fd = -np.polyfit(x, y, 1)[0]
    return fd


# ---------------------------------------------------------------------------
# Multifractal DFA
# ---------------------------------------------------------------------------
def mfdfa(series, q_list=range(-5, 6), max_lag=50, input_type=INCREMENTS):
    """Generalised Hurst exponents h(q) by multifractal DFA.

    h(2) coincides with the DFA exponent.  A flat h(q) across q indicates a
    monofractal process; curvature indicates multifractality.
    """
    profile = _as_profile(series, input_type)
    q_list = list(q_list)
    F_q = {q: [] for q in q_list}
    used = []

    for lag in range(10, max_lag):
        windows = _windows(profile, lag)
        if len(windows) < 2:
            continue
        var = _detrended_variance(windows)
        var = var[var > 0]
        if len(var) < 2:
            continue
        used.append(lag)
        for q in q_list:
            if q == 0:
                F_q[q].append(np.exp(0.5 * np.mean(np.log(var))))
            else:
                F_q[q].append(np.mean(np.power(var, q / 2)) ** (1 / q))

    H_q = {}
    log_lags = np.log(used)
    for q in q_list:
        if len(F_q[q]) > 1:
            H_q[q] = np.polyfit(log_lags, np.log(F_q[q]), 1)[0]
    return H_q


def log_returns(prices):
    """Log returns, with an explicit check that the series is a price level.

    `np.log` of a non-positive price silently yields NaN, which propagates
    into every estimator and produces a plausible-looking but meaningless H.
    Fail loudly instead.
    """
    p = pd.Series(np.asarray(prices, dtype=float))
    if (p <= 0).any():
        raise ValueError(
            "prices must be strictly positive to take log returns; "
            f"found {(p <= 0).sum()} non-positive value(s)"
        )
    return np.log(p).diff().dropna()


def predict_direction(prices, window=100, future_steps=5):
    """Predict price direction from the locally estimated Hurst exponent."""
    returns = log_returns(prices)
    recent_data = returns[-window:].values

    # Increments in, profile handled internally by mfdfa.
    H_q = mfdfa(recent_data, q_list=[2], input_type=INCREMENTS)
    H = H_q[2]

    if H > 0.55:
        # Persistent trend: extrapolate last trend
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data
        model = LinearRegression()
        model.fit(X, y)
        future_trend = model.predict([[len(recent_data) + future_steps]])[0]
        direction = "UP" if future_trend > 0 else "DOWN"
    elif H < 0.45:
        # Mean-reverting: predict reversal
        direction = "DOWN" if recent_data[-1] > 0 else "UP"
    else:
        direction = "NEUTRAL (Random)"

    return H, direction


def report(prices):
    """Print all estimates for a price level series."""
    returns = log_returns(prices).values

    print("Variogram (structure function)")
    print(f"  Hurst: {hurst_variogram(returns):.4f}")
    print("Detrended Fluctuation Analysis (DFA)")
    print(f"  Hurst: {hurst_dfa(returns):.4f}")
    fd = higuchi_fd(returns)
    print(f"Fractal Dimension (Higuchi): {fd:.4f}  -> implied H = {2 - fd:.4f}")


def load_prices(path):
    """Load the cotton price series used by the original script."""
    df = pd.read_csv(path)
    return df["Cotton Prices - 45 Year Historical Chart"]


if __name__ == "__main__":
    # The CSV read lives here, not at module level: importing this file to
    # reuse the estimators must not require a data file to exist.
    path = sys.argv[1] if len(sys.argv) > 1 else "chart.csv"
    try:
        prices = load_prices(path)
        print(f"Loaded {len(prices)} prices from {path}\n")
    except FileNotFoundError:
        print(f"{path} not found -- falling back to a synthetic random walk.\n")
        # Geometric (multiplicative) walk: stays strictly positive, so the
        # log returns are well defined.  The additive walk used previously
        # crossed zero within ~1000 steps and produced NaNs.
        # True H = 0.5 by construction, which is the demo's sanity check.
        steps = np.random.default_rng(0).standard_normal(4000) * 0.01
        prices = pd.Series(100.0 * np.exp(np.cumsum(steps)))

    report(prices)

    H, direction = predict_direction(prices, window=200, future_steps=5)
    print(f"\nHurst (q=2): {H:.3f} -> Predicted Direction: {direction}")
