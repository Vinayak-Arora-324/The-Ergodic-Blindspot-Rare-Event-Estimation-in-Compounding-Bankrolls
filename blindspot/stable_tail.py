"""
Corrected reference for the exploratory α-stable option script (`Calculation.py`),
which was retired in the rebuild along with its audit note and test file. This
file is the surviving half: it exists to record the diagnosis, not to be used.
Nothing in the project imports it, and the T/B labels below refer to that retired
test file, not to `test_regressions.py`.

The central change is not a bug fix: for alpha < 2 the exponential-stable model
has E[S_T] = infinity unless beta = -1, so `martingale_condition` is asking for
a number that does not exist. The corrected version refuses instead of
returning one.
"""
import numpy as np
from scipy import integrate, interpolate
from scipy.stats import levy_stable


# --------------------------------------------------------------------------
# FIX 1 (T2, B6): martingale correction.
#
# For alpha in (1,2), E[e^X] is finite ONLY at beta = -1 (Carr-Wu finite-moment
# log-stable). There the correct correction is
#       mu = (r-q)T + gamma^alpha * sec(pi*alpha/2) * T
# The original computed
#       mu = (r-q)T - gamma^alpha * sec(pi*alpha/2) * tan(pi*alpha/2) * T
# i.e. a spurious tan(pi*alpha/2) factor AND a flipped sign. Verified against
# numerical log E[e^(sZ)] to 6 digits for alpha in {1.9, 1.7, 1.5, 1.3}.
# --------------------------------------------------------------------------
def martingale_condition(alpha, beta, gamma, T, r, q, strict=True):
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    base_drift = (r - q) * T

    if np.isclose(alpha, 2.0):
        sigma = gamma * np.sqrt(2)
        return base_drift - 0.5 * sigma**2 * T

    if alpha <= 1.0:
        raise ValueError(
            f"alpha={alpha}: E[S_T] is infinite for alpha <= 1 at any beta. "
            "No martingale measure of this form exists.")

    if not np.isclose(beta, -1.0):
        if strict:
            raise ValueError(
                f"alpha={alpha}, beta={beta}: E[S_T] = infinity. The "
                "exponential alpha-stable model admits a martingale drift only "
                "at beta = -1. Any finite answer here is an artifact of "
                "truncating a divergent integral.")
        return np.nan

    return base_drift + gamma**alpha / np.cos(np.pi * alpha / 2) * T


def option_itm_probability(S0, K, T, r, q, alpha, beta, gamma,
                           option_type='call', strict=True):
    mu = martingale_condition(alpha, beta, gamma, T, r, q, strict=strict)
    scale = gamma * (T ** (1 / alpha))
    log_threshold = np.log(K / S0)
    if option_type == 'call':
        return levy_stable.sf(log_threshold, alpha, beta, loc=mu, scale=scale)
    return levy_stable.cdf(log_threshold, alpha, beta, loc=mu, scale=scale)


# --------------------------------------------------------------------------
# FIX 2 (B1, T3, B5): the call payoff is (S0*e^x - K), not S0*e^x/(S0*e^x - K).
# FIX 3 (B2): integrate to convergence with a split range instead of a
#             hard-coded +10, and report the truncation residual so the caller
#             can see when the integral is not converging.
# --------------------------------------------------------------------------
def expected_itm_payoff(S0, K, T, r, q, alpha, beta, gamma, option_type,
                        strict=True, return_diagnostics=False):
    mu = martingale_condition(alpha, beta, gamma, T, r, q, strict=strict)
    scale = gamma * (T ** (1 / alpha))
    lo = np.log(K / S0)

    if option_type == 'call':
        def integrand(x):
            return (S0 * np.exp(x) - K) * levy_stable.pdf(
                x, alpha, beta, loc=mu, scale=scale)

        def upto(U):
            pts = [lo] + [lo + s for s in (0.25, 1, 3, 8, 20, 50) if s < U] + [lo + U]
            return sum(integrate.quad(integrand, a, b, limit=600)[0]
                       for a, b in zip(pts[:-1], pts[1:]))

        v_ref, v_far = upto(30), upto(60)
    else:
        def integrand(x):
            return (K - S0 * np.exp(x)) * levy_stable.pdf(
                x, alpha, beta, loc=mu, scale=scale)

        def upto(U):
            pts = [lo - U] + [lo - s for s in (50, 20, 8, 3, 1, 0.25) if s < U] + [lo]
            return sum(integrate.quad(integrand, a, b, limit=600)[0]
                       for a, b in zip(pts[:-1], pts[1:]))

        v_ref, v_far = upto(30), upto(60)

    drift = abs(v_far - v_ref) / max(abs(v_ref), 1e-300)
    val = max(0.0, v_ref)
    if return_diagnostics:
        return val, {"truncation_drift": drift, "converged": drift < 1e-6}
    if drift > 1e-6:
        raise RuntimeError(
            f"integral has not converged: moving the cut from +30 to +60 "
            f"changes it by {drift:.3g} (relative). The reported value is a "
            f"property of the cutoff, not of the model.")
    return val


# --------------------------------------------------------------------------
# FIX 4 (B3, T5): McCulloch (1986) quantile estimator.
#
# The original hard-coded two linear fits with the branch test inverted:
# nu_alpha = (x95-x05)/(x75-x25) DECREASES with alpha (2.4386 at alpha=2 and
# rising as alpha falls), so "nu_alpha <= 2.439" means alpha >= 2, but the code
# mapped it to alpha ~ 0.1. Combined with np.clip(.., 1.1, 2.0) this produced a
# discontinuous flip between 1.1 and 2.0 on Gaussian data.
#
# Replaced with an exact lookup: nu_alpha and nu_beta are computed from
# levy_stable.ppf on a grid and inverted by interpolation. No tables to mistype.
# --------------------------------------------------------------------------
_GRID = None


def _build_grid(n_a=25, n_b=13):
    alphas = np.linspace(1.05, 2.0, n_a)
    betas = np.linspace(-1.0, 1.0, n_b)
    NU_A = np.zeros((n_a, n_b))
    NU_B = np.zeros((n_a, n_b))
    NU_G = np.zeros((n_a, n_b))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            q05, q25, q50, q75, q95 = levy_stable.ppf(
                [0.05, 0.25, 0.50, 0.75, 0.95], a, b)
            NU_A[i, j] = (q95 - q05) / (q75 - q25)
            NU_B[i, j] = (q95 + q05 - 2 * q50) / (q95 - q05)
            NU_G[i, j] = q75 - q25          # IQR of the standardized law
    return alphas, betas, NU_A, NU_B, NU_G


def quantile_estimation(returns):
    global _GRID
    if _GRID is None:
        _GRID = _build_grid()
    alphas, betas, NU_A, NU_B, NU_G = _GRID

    x = np.asarray(returns, dtype=float)
    x05, x25, x50, x75, x95 = np.percentile(x, [5, 25, 50, 75, 95])
    phi_1 = (x95 - x05) / (x75 - x25)
    phi_2 = (x95 + x05 - 2 * x50) / (x95 - x05)

    # invert the (alpha, beta) -> (nu_alpha, nu_beta) map by least squares on
    # the grid, then refine by local bilinear interpolation
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    # scale the two residuals to comparable size
    cost = ((NU_A - phi_1) / max(phi_1, 1e-9))**2 + (NU_B - phi_2)**2
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    alpha_est, beta_est = A[i, j], B[i, j]

    # local refinement
    sub_a = alphas[max(0, i-1):i+2]
    sub_b = betas[max(0, j-1):j+2]
    if len(sub_a) > 1 and len(sub_b) > 1:
        fa = interpolate.RegularGridInterpolator(
            (sub_a, sub_b), NU_A[max(0, i-1):i+2, max(0, j-1):j+2])
        fb = interpolate.RegularGridInterpolator(
            (sub_a, sub_b), NU_B[max(0, i-1):i+2, max(0, j-1):j+2])
        aa = np.linspace(sub_a[0], sub_a[-1], 60)
        bb = np.linspace(sub_b[0], sub_b[-1], 60)
        AA, BB = np.meshgrid(aa, bb, indexing='ij')
        pts = np.stack([AA.ravel(), BB.ravel()], axis=-1)
        c = ((fa(pts) - phi_1) / max(phi_1, 1e-9))**2 + (fb(pts) - phi_2)**2
        k = np.argmin(c)
        alpha_est, beta_est = pts[k]

    alpha_est = float(np.clip(alpha_est, 1.05, 2.0))
    beta_est = float(np.clip(beta_est, -1.0, 1.0))

    # gamma from the IQR ratio against the standardized law at (alpha, beta)
    q25s, q75s = levy_stable.ppf([0.25, 0.75], alpha_est, beta_est)
    gamma_est = (x75 - x25) / (q75s - q25s)

    # delta: match the median of the fitted law to the sample median
    q50s = levy_stable.ppf(0.50, alpha_est, beta_est)
    delta_est = x50 - gamma_est * q50s

    return {'alpha': alpha_est, 'beta': beta_est,
            'gamma': gamma_est, 'delta': delta_est}
