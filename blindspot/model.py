"""
model.py -- the single canonical model for the project, plus the three estimators.

The project is a *methods* project: the object of study is the estimator, and the
tail-hedge strategy is only the test case.  Everything downstream (experiments,
regression tests) imports the model from here and nowhere else.  The exploratory
phase duplicated the model across five scripts with divergent parameterizations,
which is exactly how the `exponnorm` bug (B3) survived undetected; one module is
the fix.

Contents
--------
  1. Model parameters and the return density         (jump-diffusion, closed form)
  2. Sampler                                          (must agree with the density)
  3. Hedge: rolling deep-OTM put, actuarially fair    (markup knob for the sweep)
  4. Quadrature truth                                 (g_hedged, edge, p_itm)
  5. Estimators: crude, cv, cv_is                     (all unbiased, all target g_h)
  6. Fractional Gaussian noise                        (driver for experiment E5)
  7. The fBm drawdown family                          (estimand for E6 and E7)
  8. Fat tails + transient volatility memory          (unified E8 model)

Standing caveat that must accompany every reported result: the default is
`markup = 1.0`, i.e. the put is priced actuarially fair.  There is no variance
risk premium at that setting, so every "the hedge wins" number below is
conditional on an options market that does not exist.  `markup > 1` is the
opposite-side sensitivity and is the first knob to sweep.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import integrate, stats
from scipy.special import erfc

# ---------------------------------------------------------------------------
# 1. Model parameters
# ---------------------------------------------------------------------------
# Monthly log return: a light-tailed bulk plus a rare negative jump.
#
#     r = m + S*Z            with probability 1 - LAM
#     r = m + S*Z - E        with probability LAM,   E ~ Exp(mean BETA), Z ~ N(0,1)
#
# Left-skewed rather than symmetric, which is the honest shape for equity
# crashes, and the natural one-sided restriction of CGMY.  It replaced an
# earlier symmetric Student-t model, which was discarded (see the rebuild spec,
# section 5 -- the t-model's cross-nu comparisons were invalidated by bug B1).

LAM = 1e-3      # jump probability per month
BETA = 1.5      # mean jump size, in log points
S = 0.05        # bulk monthly volatility
MU = 0.005      # TARGET E[r]; pinned, see below

# The bulk mean is *solved for*, not chosen: E[r] = m - LAM*BETA, so setting
# m = MU + LAM*BETA pins E[r] = MU exactly.  This is load-bearing.  It makes
# g_unhedged = MU known in closed form, so all of the estimation difficulty
# sits on the hedged side and none of it is contaminated by baseline noise.
M_BULK = MU + LAM * BETA        # = 0.0065

# Proposal measure for importance sampling: the same model with the jump
# probability inflated.  Only LAM changes, so the likelihood ratio f/q is a
# ratio of two mixtures over identical components and is therefore *bounded*
# (verified range [0.0033, 1.4228]) -- finite IS variance by construction, not
# by hope.
LAM_Q = 0.30

# Quadrature window.  See `truth()` for why the lower limit is far out at -40.
LO, HI = -40.0, 2.0
QUAD_LIMIT = 1500

_KP = BETA / S          # exponnorm's shape parameter K; = 30 at these params


# ---------------------------------------------------------------------------
# 2. Density and sampler
# ---------------------------------------------------------------------------
def bulk_pdf(r):
    """Density of the no-jump component, N(M_BULK, S)."""
    r = np.asarray(r, dtype=float)
    return np.exp(-0.5 * ((r - M_BULK) / S) ** 2) / (S * np.sqrt(2 * np.pi))


def jump_pdf(r):
    """
    Density of the jump component, N(M_BULK, S) - Exp(BETA).

    That is an exponentially modified Gaussian, reached by a sign flip:
    -r = Exp(BETA) + N(-M_BULK, S), so in scipy terms the density is
    `exponnorm.pdf(-r, K=BETA/S, loc=-M_BULK, scale=S)`.

    This closed-form numpy version exists purely for speed: E3 performs 1e7+
    density evaluations and `scipy.stats.*.pdf` dominates the runtime there.
    `check_density()` asserts it agrees with scipy to rtol=1e-10 -- that
    assertion *is* regression test B3, because this parameterization (evaluate
    at -r, K = BETA/S, loc = -M_BULK, scale = S) is very easy to get wrong and
    a wrong version fails silently.
    """
    r = np.asarray(r, dtype=float)
    y = (-r + M_BULK) / S
    return (
        (1.0 / (2.0 * _KP))
        * np.exp(1.0 / (2.0 * _KP**2) - y / _KP)
        * erfc((1.0 / _KP - y) / np.sqrt(2.0))
        / S
    )


def pdf(r, lam=LAM):
    """Mixture density of the log return with jump probability `lam`.

    `pdf(r)` is the true measure f; `pdf(r, LAM_Q)` is the IS proposal q.
    """
    return (1.0 - lam) * bulk_pdf(r) + lam * jump_pdf(r)


def check_density(rtol=1e-10):
    """Regression B3: closed form must match scipy across bulk and deep tail.

    The grid deliberately spans the far-left tail (-8), the crash region (-3),
    the strike neighbourhood, the bulk, and the right tail.  There is no
    overflow risk at these parameters (_KP = 30).
    """
    grid = np.array([-8.0, -3.0, -0.5, 0.0, 0.05, 0.4])
    ref = (1.0 - LAM) * stats.norm.pdf(grid, M_BULK, S) + LAM * stats.exponnorm.pdf(
        -grid, _KP, loc=-M_BULK, scale=S
    )
    ours = pdf(grid)
    assert np.allclose(ours, ref, rtol=rtol), (
        f"closed-form density disagrees with scipy: {ours} vs {ref}"
    )
    return grid, ours, ref


def sample_returns(rng, n, lam=LAM, return_jumps=False):
    """Draw `n` log returns with jump probability `lam`.

    The sampler and the density are written independently and are cross-checked
    against each other by test T1 (empirical vs quadrature P(r < log(K/S0))).
    A silent mismatch between the two is the single most dangerous failure mode
    in this codebase: every estimator here mixes sampled draws with analytic
    densities, so a mismatch would corrupt the IS weights without raising
    anything.
    """
    jump = rng.random(n) < lam
    r = M_BULK + S * rng.standard_normal(n)
    r[jump] -= rng.exponential(BETA, jump.sum())
    return (r, jump) if return_jumps else r


# ---------------------------------------------------------------------------
# 3. Hedge
# ---------------------------------------------------------------------------
# Roll a deep out-of-the-money put every month.  Each month a wealth fraction C
# buys puts and 1 - C stays in the index.
S0 = 100.0      # index level at the start of each month
K = 50.0        # strike: 50% OTM, so only a genuine crash pays
C = 0.0005      # premium spend as a fraction of wealth

R_STAR = np.log(K / S0)     # return below which the put finishes in the money


def put_payoff(r):
    """Payoff of one put at expiry, as a function of the realized log return."""
    return np.maximum(K - S0 * np.exp(np.asarray(r, dtype=float)), 0.0)


@lru_cache(maxsize=8)
def put_price(markup=1.0):
    """Price paid per put = `markup` x the actuarially fair price E_f[payoff].

    markup = 1.0 (the default) means no variance risk premium: the hedger buys
    insurance at cost.  That is the most favourable possible assumption for the
    hedge and it is not a market that exists.  Results must be reported as
    conditional on it.
    """
    fair, _ = integrate.quad(
        lambda r: put_payoff(r) * pdf(r), LO, HI, limit=QUAD_LIMIT
    )
    return markup * fair


def log_growth(r, markup=1.0, c=C):
    """One-month log growth factor of the hedged portfolio.

    log( (1-c)*exp(r) + c*payoff(r)/price )

    Note the shape of this function, because it is the reason the project's
    original founding sentence was wrong.  In the bulk the put expires
    worthless and this is just log(1-c) + r: a constant bleed.  In a crash the
    payoff term *cancels* the index loss, so the result is mild and bounded
    below.  Hedging destroys the rare-event structure of the hedged quantity.
    The rare-event difficulty lives in the unhedged baseline and in the
    hedged-minus-unhedged difference, not here.
    """
    r = np.asarray(r, dtype=float)
    return np.log((1.0 - c) * np.exp(r) + c * put_payoff(r) / put_price(markup))


def control_variate(r, markup=1.0, c=C):
    """D(r) = log_growth(r) - r, the hedge's *effect* on log growth.

    Used as a control variate: E[r] = MU is known exactly (by the pinning in
    section 1), so only E[D] needs estimating and the estimator is
    mean(D) + MU.  D is constant (= log(1-c)) throughout the bulk and nonzero
    only in crashes -- which is what makes it the right target for importance
    sampling, and simultaneously what makes it *harder* than log_growth to
    estimate by crude sampling.  See `estimator_note` below.
    """
    return log_growth(r, markup=markup, c=c) - np.asarray(r, dtype=float)


# ---------------------------------------------------------------------------
# 4. Quadrature truth
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def truth(markup=1.0, lo=LO):
    """Ground truth by 1-D quadrature. Returns a dict.

    Under iid returns the ergodic growth rate of a bankroll *is* this
    single-period expectation (LLN), so g collapses to a 1-D integral and no
    bankroll loop is needed -- a loop would only add Monte Carlo noise to a
    quantity that was never path-dependent.  This is why the rare-event
    machinery in section 5 is, for this estimand, unnecessary: quadrature beats
    every Monte Carlo method here.  Experiment E4 is where that stops being
    true.

    Caveat, deliberately kept in a comment rather than asserted as a claim:
    adaptive `quad` over [-40, 2] resolves the width-0.05 bulk only because the
    jump density spreads mass across the interval and guides the subdivision.
    That is fragile -- for a pure-normal integrand the same call can silently
    return near zero.  Do not call this output "exact".  It is accurate to
    ~machine precision *at these parameters*, and that is verified by tests T1
    and B4 (stability when the lower limit is widened to -60), not assumed.
    """
    g_h, g_h_err = integrate.quad(
        lambda r: log_growth(r, markup) * pdf(r), lo, HI, limit=QUAD_LIMIT
    )
    p_itm, _ = integrate.quad(pdf, lo, R_STAR, limit=QUAD_LIMIT)
    return {
        "g_hedged": g_h,
        "g_hedged_quaderr": g_h_err,
        "g_unhedged": MU,           # exact, by construction (E[r] pinned to MU)
        "edge": g_h - MU,
        "p_itm": p_itm,             # P(put finishes in the money) in one month
        "p_jump": LAM,              # P(a jump occurs) -- a DIFFERENT number; see B5
        "price": put_price(markup),
        "markup": markup,
    }


def edge_and_p(lam, c, beta=BETA, s=S, k=K, markup=1.0, lo=-60.0):
    """Recompute (edge, p_itm) from scratch at an arbitrary jump rate and premium.

    Everything is rebuilt locally -- the bulk mean is re-solved so that
    E[r] = MU still holds, the put is re-priced under the new density -- so this
    is a standalone sensitivity, not a perturbation of the module globals.

    This exists to guard bug B2 and the rule that came out of it. An
    exploratory sweep compared jump intensities across a FIXED premium grid
    floored at c = 2e-4. At small lam the option is far cheaper than that
    floor, so every rare-lam configuration was overpaying by construction and
    "bled" -- which was then written up as "tail hedging only helps when
    p ~ 1e-3". That claim is false. Under fair pricing the edge stays positive
    at any rarity once c is sized to p; it merely shrinks in proportion to p.

    Rule: any premium sweep must scale c with p. Never use a fixed grid.
    """
    m = MU + lam * beta
    kp = beta / s

    def dens(r):
        r = np.asarray(r, dtype=float)
        bulk = np.exp(-0.5 * ((r - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        y = (-r + m) / s
        psi = (
            (1.0 / (2.0 * kp))
            * np.exp(1.0 / (2.0 * kp**2) - y / kp)
            * erfc((1.0 / kp - y) / np.sqrt(2.0))
            / s
        )
        return (1.0 - lam) * bulk + lam * psi

    pay = lambda r: np.maximum(k - S0 * np.exp(r), 0.0)
    price, _ = integrate.quad(lambda r: pay(r) * dens(r), lo, HI, limit=2000)
    price *= markup
    g, _ = integrate.quad(
        lambda r: np.log((1 - c) * np.exp(r) + c * pay(r) / price) * dens(r),
        lo, HI, limit=2000,
    )
    p, _ = integrate.quad(dens, lo, np.log(k / S0), limit=2000)
    return g - MU, p


# ---------------------------------------------------------------------------
# 5. Estimators
# ---------------------------------------------------------------------------
# All three are unbiased for g_hedged.  They differ only in variance.
#
#   crude : mean( log_growth(x) ),                 x ~ f
#   cv    : mean( D(x) ) + MU,                     x ~ f
#   cv_is : mean( D(y) * f(y)/q(y) ) + MU,         y ~ q
#
# The ordering of their variances is counter-intuitive and *verified*; code
# downstream must not try to "fix" it:
#
#   * cv alone INCREASES variance (~0.6x, i.e. worse than crude).  Per-sample
#     sd is 0.0510 for log_growth but 0.0679 for D.  The control variate
#     removes the bulk noise, but what is left -- the hedge's effect -- is
#     purely crash-driven and violently skewed, and so is harder to estimate
#     than the level was.
#   * IS alone also FAILS (~0.8x): it spends its budget oversampling a region
#     that carries only ~2% of Var(log_growth), while inflating the bulk
#     variance through the ~1.43 weight.
#   * cv_is WORKS (~190x).  The composition is the whole point: CV reshapes the
#     estimand into the crash-driven object that IS is built for.  Neither half
#     works alone.
#
# And the deflationary reading, which must be reported alongside the 190x:
# crude MC needs only ~14.7k samples here (milliseconds) and quadrature needs
# none at all.  The speedup is a real capability aimed at a problem this iid
# model does not have.

def est_crude(rng, n, markup=1.0):
    """Crude Monte Carlo: sample from f, average the log growth."""
    return log_growth(sample_returns(rng, n, LAM), markup).mean()


def est_cv(rng, n, markup=1.0):
    """Control variate only: average D under f, add back the known E[r] = MU."""
    return control_variate(sample_returns(rng, n, LAM), markup).mean() + MU


def est_cv_is(rng, n, markup=1.0):
    """Control variate + importance sampling: average w*D under q, add MU."""
    y = sample_returns(rng, n, LAM_Q)
    w = pdf(y, LAM) / pdf(y, LAM_Q)
    return (control_variate(y, markup) * w).mean() + MU


def est_is(rng, n, markup=1.0):
    """IS without the control variate -- included only to show that it fails."""
    y = sample_returns(rng, n, LAM_Q)
    w = pdf(y, LAM) / pdf(y, LAM_Q)
    return (log_growth(y, markup) * w).mean()


#: The three estimators of record, with the colours they carry in every figure.
#: Blue / orange / violet: checked, not eyeballed -- each clears 3:1 against a
#: white ground and every pair stays separable under protanopia, deuteranopia
#: and tritanopia. The earlier red/orange/green failed both, and green-vs-orange
#: in particular is invisible to a protan reader, which is the most common form.
ESTIMATORS = (
    ("crude MC", est_crude, "#2a78d6"),
    ("+ control variate", est_cv, "#eb6834"),
    ("+ CV + importance sampling", est_cv_is, "#4a3aa7"),
)


def run_trials(estimator, n, trials, rng, markup=1.0):
    """Repeat `estimator` at sample size `n` for `trials` independent runs.

    Returns the array of estimates, i.e. a draw from the estimator's *sampling
    distribution* -- which is the actual object of study in E1 and E3.
    """
    return np.array([estimator(rng, n, markup) for _ in range(trials)])


def is_weight_range(rng, n=200_000):
    """Empirical range of the IS likelihood ratio f/q (regression test T5).

    Bounded weights are the reason IS is trustworthy here rather than a source
    of silent infinite variance.  Bulk draws carry weight ~1.43; the
    oversampled crash draws carry ~1/300.
    """
    y = sample_returns(rng, n, LAM_Q)
    w = pdf(y, LAM) / pdf(y, LAM_Q)
    return float(w.min()), float(w.max())


# ---------------------------------------------------------------------------
# 6. Fractional Gaussian noise (driver for E5)
# ---------------------------------------------------------------------------
def fgn_cholesky(hurst, t_max):
    """Cholesky factor of the fGn covariance matrix, for exact fGn simulation.

    Autocovariance of fractional Gaussian noise at lag k, unit variance:
        gamma(k) = 0.5 * ( |k+1|^2H - 2|k|^2H + |k-1|^2H )

    Exact simulation (as opposed to a spectral approximation) matters here
    because the whole experiment is about whether a *reported* error bar is
    honest; an approximate long-memory driver would put the artefact under
    suspicion.  Cholesky also has a convenient property: a prefix of a
    simulated path is itself a valid shorter path, so nested values of T can
    reuse one set of draws instead of being separately noisy.

    Cost is O(t_max^2) memory and O(t_max^3) time -- fine at t_max = 4096, not
    beyond.  The small ridge keeps the factorization numerically safe.
    """
    k = np.arange(t_max, dtype=float)
    gamma = 0.5 * (
        (k + 1.0) ** (2 * hurst) - 2.0 * k ** (2 * hurst) + np.abs(k - 1.0) ** (2 * hurst)
    )
    idx = np.arange(t_max)
    cov = gamma[np.abs(np.subtract.outer(idx, idx))]
    return np.linalg.cholesky(cov + 1e-10 * np.eye(t_max))


# E5's volatility model.  Returns stay serially UNcorrelated (matching the
# stylized fact); long memory enters the *growth* estimand only through the
# variance drag sigma_t^2 / 2.
#
#     r_t     = MU_A - sigma_t^2/2 + sigma_t * Z_t,   Z iid N(0,1)
#     log sig = log(SBAR) + XI * G_t,                 G = fGn with Hurst H
#
# Since Var(G_t) = 1 by construction, E[sigma_t^2] = SBAR^2 * exp(2*XI^2) and
# the true growth rate is available in closed form.
MU_A = 0.01
SBAR = 0.15
XI = 0.9


def lrd_g_true(mu_a=MU_A, sbar=SBAR, xi=XI):
    """Closed-form growth rate of the fGn-vol model: mu_a - 0.5*sbar^2*exp(2*xi^2)."""
    return mu_a - 0.5 * sbar**2 * np.exp(2 * xi**2)


# ---------------------------------------------------------------------------
# 7. The fBm drawdown family (E6, E7)
# ---------------------------------------------------------------------------
# E6 and E7 need a family of bankrolls indexed by ONE parameter, where the
# estimand is a path functional and the parameter controls how the path is
# rough or persistent.  Log wealth is fractional Brownian motion:
#
#     X(t) = MU_T * (t/T) + sigma_H * B_H(t),     Var B_H(t) = t^(2H)
#
# and sigma_H = DD_SIGMA_T / T^H, so Var X(T) = DD_SIGMA_T^2 for EVERY H.
# Pinning the terminal variance is what makes the family comparable: without
# it, sweeping H would mostly sweep the scale of the process and the drawdown
# curves in E6 would be telling you about volatility, not about roughness.
#
# WHERE THIS DEPARTS FROM THE REST OF THE PROJECT, stated plainly because the
# figure travels without its caption: `docs/memory_evidence.md` gives H ~ 0.5 for the
# DIRECTION of S&P returns and H ~ 0.84-0.95 for their VOLATILITY.  Putting the
# memory in the price path itself, as here, is therefore NOT the calibrated
# model -- it is a test bed.  It is the right test bed anyway, because the
# estimator pathologies E7 exhibits are driven by the roughness and the
# dependence of whatever series the practitioner actually samples, and H is the
# one knob that moves both.  Read E6/E7 as a stress test of two estimator
# designs across a family, not as a claim about equity prices.

DD_HORIZON = 12                       # months in the drawdown horizon
DD_SUB = 21                           # sub-steps per month: the DAILY grid
DD_SIGMA_T = S * np.sqrt(DD_HORIZON)  # terminal sd of log wealth, fixed across H
DD_MU_T = MU * DD_HORIZON             # total drift over the horizon
DD_LEVEL = 99.9                       # the 1-in-1000 read


def fgn_davies_harte(hurst, n_steps, n_paths, rng, psd_tol=1e-3):
    """Exact fractional Gaussian noise by circulant embedding.

    Same law as `fgn_cholesky` and checked against it in T8, but O(n log n)
    instead of O(n^3), which is the difference between E7 being runnable and
    not: its single-record design needs paths of several million steps.

    Davies-Harte embeds the Toeplitz fGn covariance in a 2m circulant whose
    eigenvalues are the FFT of the first row.  For fGn the embedding is
    non-negative definite for every H in (0, 1), so no eigenvalue clipping is
    needed and the output is exact rather than approximate -- which matters for
    the same reason it mattered in E5: the experiment is about whether a
    reported error bar is honest, and an approximate driver would put the
    finding under suspicion.

    The circulant row is real and even, so its spectrum is real and the whole
    filter runs through rfft/irfft rather than a complex transform: identical
    output, half the memory.

    Output variance is 1.0 exactly, not approximately, and that is load-bearing
    rather than cosmetic: E5's closed-form truth `lrd_g_true` assumes
    Var(G_t) = 1, and E6/E7's terminal-variance pinning assumes it again.  The
    reason it holds is Parseval.  The filter applied to iid unit normals is
    h = ifft(sqrt(lam)), so Var(y_t) = sum_j h_j^2 = (1/2m) sum_k lam_k, and
    sum_k lam_k = 2m * row[0] = 2m * gamma(0) = 2m for any H.  Checked
    deterministically in T11.

    Numerical caveat, and it is a real one rather than boilerplate.  The
    embedding is psd in exact arithmetic, but the eigenvalues come out of an
    FFT of a length-2m row, and for H near 1 the smallest true eigenvalues are
    O(0.1) while the largest grow like m^(2H-1).  Past a few million steps the
    roundoff floor overtakes the small end and a slice of the spectrum lands
    just below zero.  Those modes are discarded, and the same Parseval identity
    says exactly what that costs: clipping REMOVES negative mass, so it raises
    the output variance to 1 + `lost`, and every autocovariance is perturbed on
    the same scale.  The mass is therefore measured rather than assumed: below
    `psd_tol` it is negligible and the run continues; above it the simulation
    is not trustworthy and this raises instead of quietly returning a process
    with the wrong long-range structure.  At H = 0.9 and 4M steps the discarded
    mass is 3.8e-4; at H = 0.95 it is 5.4e-2, and that one should fail.

    `lost` is a fraction of the FULL 2m spectrum, not of the half-spectrum
    `lam` that rfft returns -- the interior bins k = 1..m-1 each stand for two
    eigenvalues.  Weighting them accordingly is what makes `lost` equal the
    variance error exactly (verified to 13 digits in T11) instead of
    understating it by ~20%, which is the direction that matters for a gate.
    """
    m = 1 << int(np.ceil(np.log2(max(n_steps, 2))))
    k = np.arange(m + 1, dtype=float)
    gamma = 0.5 * ((k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst)
                   + np.abs(k - 1) ** (2 * hurst))
    row = np.concatenate([gamma, gamma[-2:0:-1]])        # length 2m, even
    lam = np.fft.rfft(row).real
    neg_mask = lam < 0.0
    if neg_mask.any():
        # Multiplicity of each rfft bin in the full length-2m spectrum: the DC
        # and Nyquist bins appear once, every interior bin twice.  The full
        # spectrum sums to 2m * gamma(0) = 2m, which is the denominator.
        mult = np.full(lam.size, 2.0)
        mult[0] = mult[-1] = 1.0
        lost = -(mult * lam)[neg_mask].sum() / (2 * m)
        if lost > psd_tol:
            raise ValueError(
                f"Davies-Harte embedding lost {lost:.2%} of the spectrum at "
                f"H = {hurst}, n = {n_steps:,}: too much to clip. Use a shorter "
                f"record, or fgn_cholesky if the length allows it.")
    root = np.sqrt(np.clip(lam, 0.0, None))
    z = rng.standard_normal((n_paths, 2 * m))
    y = np.fft.irfft(np.fft.rfft(z, axis=1) * root, n=2 * m, axis=1)
    return y[:, :n_steps]


def _max_drawdown(x):
    """Max log drawdown of each row of a cumulative log-wealth array."""
    z = np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1)
    return (np.maximum.accumulate(z, axis=1) - z).max(axis=1)


def _fbm_wealth(hurst, inc, sub):
    """Cumulative log wealth from unit-step fGn increments on a `sub` grid."""
    n = inc.shape[1]
    scale = DD_SIGMA_T / DD_HORIZON ** hurst * sub ** (-hurst)
    return scale * np.cumsum(inc, axis=1) + DD_MU_T * np.arange(1, n + 1) / n


def drawdown_replicates(hurst, n_paths, rng, sub=1, chunk=20_000):
    """Max log drawdown of `n_paths` INDEPENDENT bankrolls.

    `sub` is the grid the simulator chose: sub = 1 is monthly, sub = DD_SUB is
    the daily grid the estimand is defined on.  A coarse grid cannot see the
    drawdown that happens between its own sample points, and for fBm that gap
    scales like (steps)^-H -- negligible for a smooth persistent path, ruinous
    for a rough one.  That is E7's left-hand failure, and it is a property of
    the grid, not of the sample size, so no amount of extra paths removes it.
    """
    out, n = [], DD_HORIZON * sub
    for i in range(0, n_paths, chunk):
        k = min(chunk, n_paths - i)
        inc = fgn_davies_harte(hurst, n, k, rng)
        out.append(_max_drawdown(_fbm_wealth(hurst, inc, sub)))
    return np.concatenate(out)


def drawdown_windows(hurst, n_windows, rng, sub=DD_SUB):
    """Max log drawdown of consecutive windows cut from ONE long record.

    This is the design forced on anyone working from history rather than from a
    simulator: there is one realized path, and the sample size comes from
    walking a window along it.  The windows here are NON-overlapping, which is
    the most charitable version available -- at H = 1/2 they are exactly iid,
    so any failure at other H is dependence in the process and not
    double-counting by the analyst.
    """
    n = DD_HORIZON * sub
    # One record, then cut. Reshaping the INCREMENTS and accumulating per row
    # restarts each window at its own zero, which is what a drawdown measured
    # from the window's own high-water mark means; the dependence between
    # windows survives untouched, because it lives in the increments.
    inc = fgn_davies_harte(hurst, n_windows * n, 1, rng)[0].reshape(n_windows, n)
    return _max_drawdown(_fbm_wealth(hurst, inc, sub))


def bootstrap_halfwidth(sample, rng, level=DD_LEVEL, boot=400):
    """The 95% half-width an iid bootstrap reports for a percentile.

    The bootstrap is the right bar to indict rather than a parametric formula:
    it assumes no distributional shape at all, it is what a careful
    practitioner actually quotes, and it is still wrong in E7 -- because
    resampling a sample cannot recover structure the sample does not know it
    has.  Verified calibrated (0.9-1.1x) against the true sd on genuinely iid
    draws from this family, at both grids, in T9.
    """
    idx = rng.integers(0, sample.size, size=(boot, sample.size))
    return 1.96 * np.percentile(sample[idx], level, axis=1).std(ddof=1)


def wipeout(log_drawdown):
    """Log drawdown as a percentage of wealth destroyed."""
    return 100.0 * (1.0 - np.exp(-np.asarray(log_drawdown)))


# ---------------------------------------------------------------------------
# 8. Unified fat-tail + transient-memory market (E8)
# ---------------------------------------------------------------------------
# E1-E7 deliberately isolate mechanisms.  E8 is the model the original project
# question actually asks for: fat left tails and locally rough/persistent
# volatility in the SAME market, with crude and variance-reduced Monte Carlo
# pricing the SAME rolling tail hedge.
#
# A constant-H fGn cannot be locally H != 1/2 and asymptotically H = 1/2.  The
# volatility driver below therefore resets after MEM_CUTOFF months.  Inside a
# regime it is exact fGn of `local_h`; distinct regimes are independent.  For a
# partial sum of n observations,
#
#   Var(sum_1^n G_t) = q * cutoff^(2H) + remainder^(2H).
#
# Thus the local scaling is n^(2H), while at horizons much longer than the
# cutoff it is proportional to n and H_eff -> 1/2.  Passing a tuple such as
# (0.1, 0.8) alternates rough and persistent regimes in one simulated history.

MEM_CUTOFF = 64
REGIME_H = (0.1, 0.5, 0.8)

RS_SBAR = 0.05
RS_XI = 0.65
RS_RF = 0.002             # monthly continuously-compounded risk-free rate
RS_MU_P = 0.006           # target monthly arithmetic growth under P
RS_LAM_P = 0.001          # physical crash probability
RS_LAM_Q = 0.003          # risk-neutral crash probability (tail-risk premium)
RS_LAM_IS = 0.20          # proposal crash probability
RS_BETA = 0.60            # mean downward jump in log points
RS_K = 0.70               # one-month put strike, spot normalised to 1
RS_C = 0.001               # fraction of bankroll spent on the put each month


def _jump_gross_moment(lam, beta=RS_BETA):
    """E[exp(-I*E)] for I~Bernoulli(lam), E~Exponential(mean beta)."""
    return (1.0 - lam) + lam / (1.0 + beta)


def regime_sum_variance(local_h, n, cutoff=MEM_CUTOFF):
    """Exact variance of a reset-fGn partial sum.

    `local_h` may be one H or a cycle of regime exponents.  This deterministic
    identity is the audit trail for the claim that local anomalous scaling
    crosses to ordinary H=1/2 scaling at long horizons.
    """
    if n < 0 or cutoff < 1:
        raise ValueError("n must be non-negative and cutoff must be positive")
    hs = np.atleast_1d(np.asarray(local_h, dtype=float))
    if np.any((hs <= 0) | (hs >= 1)):
        raise ValueError("every local H must lie in (0, 1)")
    full, rem = divmod(int(n), int(cutoff))
    blocks = np.arange(full) % len(hs)
    var = float(np.sum(cutoff ** (2.0 * hs[blocks]))) if full else 0.0
    if rem:
        var += float(rem ** (2.0 * hs[full % len(hs)]))
    return var


def regime_fgn(rng, n_paths, horizon, local_h=0.8, cutoff=MEM_CUTOFF):
    """Unit-variance fGn in independent finite regimes.

    A scalar H gives repeated regimes of the same local roughness.  A sequence
    cycles through regimes, e.g. `(0.1, 0.8)` for alternating rough/persistent
    markets.  Resetting is intentional, not a simulation shortcut: it is the
    finite memory that restores H=1/2 at long horizons.
    """
    hs = np.atleast_1d(np.asarray(local_h, dtype=float))
    if np.any((hs <= 0) | (hs >= 1)):
        raise ValueError("every local H must lie in (0, 1)")
    out = np.empty((int(n_paths), int(horizon)))
    for block_no, start in enumerate(range(0, int(horizon), int(cutoff))):
        width = min(int(cutoff), int(horizon) - start)
        h = float(hs[block_no % len(hs)])
        out[:, start:start + width] = fgn_davies_harte(
            h, width, int(n_paths), rng
        )
    return out


def _market_parameters(measure):
    key = str(measure).upper()
    if key == "Q":
        return RS_RF, RS_LAM_Q
    if key == "P":
        return RS_MU_P, RS_LAM_P
    raise ValueError("measure must be 'P' or 'Q'")


def regime_market_drift(sigma, measure="Q"):
    """Conditional log drift that pins E[exp(r)|sigma] under P or Q."""
    gross_rate, lam = _market_parameters(measure)
    return (gross_rate - 0.5 * np.asarray(sigma) ** 2
            - np.log(_jump_gross_moment(lam)))


def _left_emg_cdf(x, mean, sigma, beta):
    """CDF of Normal(mean, sigma) - Exponential(mean beta)."""
    return stats.exponnorm.sf(
        -np.asarray(x), beta / np.asarray(sigma),
        loc=-np.asarray(mean), scale=np.asarray(sigma)
    )


def _conditional_tail_put_components(sigma):
    """Discounted conditional put values given no jump and given a jump."""
    sigma = np.asarray(sigma, dtype=float)
    m = regime_market_drift(sigma, "Q")
    k = np.log(RS_K)

    a = (k - m) / sigma
    p_bulk = stats.norm.cdf(a)
    eg_bulk = np.exp(m + 0.5 * sigma**2) * stats.norm.cdf(a - sigma)
    put_bulk = RS_K * p_bulk - eg_bulk

    p_jump = _left_emg_cdf(k, m, sigma, RS_BETA)
    tilted_beta = RS_BETA / (1.0 + RS_BETA)
    eg_jump = (
        np.exp(m + 0.5 * sigma**2) / (1.0 + RS_BETA)
        * _left_emg_cdf(k, m + sigma**2, sigma, tilted_beta)
    )
    put_jump = RS_K * p_jump - eg_jump
    discount = np.exp(-RS_RF)
    return discount * np.maximum(put_bulk, 0.0), discount * np.maximum(put_jump, 0.0)


def conditional_tail_put_price(sigma, markup=1.0):
    """Risk-neutral one-month price of the rolling tail put, conditional on sigma.

    The formula is analytic for both mixture components.  The exponential tilt
    in the truncated first moment turns Exp(mean beta) into
    Exp(mean beta/(1+beta)) and contributes the factor 1/(1+beta).
    """
    put_bulk, put_jump = _conditional_tail_put_components(sigma)
    expected = (1.0 - RS_LAM_Q) * put_bulk + RS_LAM_Q * put_jump
    return markup * expected


@lru_cache(maxsize=4)
def tail_put_truth_and_cv(order=36):
    """Deterministic quadrature truth and the crash-indicator control coefficient.

    Returns the discounted Q price and theta for
    `put - theta*(I_jump - lambda_Q)`.  Conditional put moments are analytic;
    Hermite quadrature only averages them over the lognormal volatility state.
    This is more stable than sending a generic quadrature rule hunting for a
    0.3%-weight deep-tail component.
    """
    from numpy.polynomial.hermite import hermgauss

    hx, hw = hermgauss(order)
    normal_x, normal_w = np.sqrt(2.0) * hx, hw / np.sqrt(np.pi)
    sigma = RS_SBAR * np.exp(RS_XI * normal_x)
    bulk, jump = _conditional_tail_put_components(sigma)
    mean_bulk = float(normal_w @ bulk)
    mean_jump = float(normal_w @ jump)
    price = (1.0 - RS_LAM_Q) * mean_bulk + RS_LAM_Q * mean_jump
    return {
        "price": float(price),
        "underlying": 1.0,       # discounted Q expectation, pinned by drift
        "theta": float(mean_jump - mean_bulk),
        "mean_no_jump": mean_bulk,
        "mean_jump": mean_jump,
    }


def simulate_regime_market(rng, n_paths, horizon, local_h=0.8, measure="Q",
                           cutoff=MEM_CUTOFF, proposal_lam=None):
    """Simulate fat-tailed stochastic-volatility returns and likelihood weights."""
    _, target_lam = _market_parameters(measure)
    sim_lam = target_lam if proposal_lam is None else float(proposal_lam)
    if not 0 < sim_lam < 1:
        raise ValueError("proposal_lam must lie in (0, 1)")

    g = regime_fgn(rng, n_paths, horizon, local_h, cutoff)
    sigma = RS_SBAR * np.exp(RS_XI * g)
    jump = rng.random(sigma.shape) < sim_lam
    r = regime_market_drift(sigma, measure) + sigma * rng.standard_normal(sigma.shape)
    r -= jump * rng.exponential(RS_BETA, sigma.shape)
    weights = np.where(jump, target_lam / sim_lam,
                       (1.0 - target_lam) / (1.0 - sim_lam))
    return {"r": r, "sigma": sigma, "jump": jump, "weights": weights}


def price_estimators_once(rng, budget=16_384, horizon=256, local_h=0.8,
                          cutoff=MEM_CUTOFF):
    """One equal-budget draw from crude, CV, IS, and CV+IS put-price estimators."""
    n_paths = max(1, int(budget) // int(horizon))
    n_obs = n_paths * int(horizon)
    theta = tail_put_truth_and_cv()["theta"]
    discount = np.exp(-RS_RF)

    target = simulate_regime_market(
        rng, n_paths, horizon, local_h, "Q", cutoff, proposal_lam=RS_LAM_Q
    )
    proposal = simulate_regime_market(
        rng, n_paths, horizon, local_h, "Q", cutoff, proposal_lam=RS_LAM_IS
    )

    def discounted_values(draw):
        gross = discount * np.exp(draw["r"])
        put = discount * np.maximum(RS_K - np.exp(draw["r"]), 0.0)
        return put, gross

    p, _ = discounted_values(target)
    pp, _ = discounted_values(proposal)
    w = proposal["weights"]
    return {
        "crude": float(p.mean()),
        "cv": float((p - theta * (target["jump"] - RS_LAM_Q)).mean()),
        "is": float((w * pp).mean()),
        "cv_is": float((w * (pp - theta * proposal["jump"])).mean()
                       + theta * RS_LAM_Q),
        "n_obs": n_obs,
    }


def run_price_trials(rng, trials=200, budget=16_384, horizon=256,
                     local_h=0.8, cutoff=MEM_CUTOFF):
    """Sampling distributions of all four E8 pricing estimators."""
    names = ("crude", "cv", "is", "cv_is")
    out = {name: np.empty(int(trials)) for name in names}
    for i in range(int(trials)):
        row = price_estimators_once(rng, budget, horizon, local_h, cutoff)
        for name in names:
            out[name][i] = row[name]
    return out


def tail_hedge_strategy_paths(rng, n_paths=50_000, horizon=120, local_h=0.8,
                              cutoff=MEM_CUTOFF, markup=1.0):
    """Physical-measure bankroll paths for the rolling, Q-priced tail hedge."""
    draw = simulate_regime_market(
        rng, n_paths, horizon, local_h, "P", cutoff, proposal_lam=RS_LAM_P
    )
    r, sigma = draw["r"], draw["sigma"]
    payoff = np.maximum(RS_K - np.exp(r), 0.0)
    premium = conditional_tail_put_price(sigma, markup=markup)
    gross_h = (1.0 - RS_C) * np.exp(r) + RS_C * payoff / premium
    log_h = np.log(gross_h)
    x_u, x_h = np.cumsum(r, axis=1), np.cumsum(log_h, axis=1)
    return {
        "terminal_unhedged": x_u[:, -1],
        "terminal_hedged": x_h[:, -1],
        "difference": x_h[:, -1] - x_u[:, -1],
        "drawdown_unhedged": _max_drawdown(x_u),
        "drawdown_hedged": _max_drawdown(x_h),
    }


if __name__ == "__main__":
    # Smoke test: the model's own consistency checks and the headline truths.
    check_density()
    t = truth()
    print("closed-form density vs scipy .............. OK (rtol 1e-10)")
    print(f"fair put price ........................... {t['price']:.6f}")
    print(f"g_unhedged (exact, pinned) ............... {t['g_unhedged']:+.6f}")
    print(f"g_hedged   (quadrature) .................. {t['g_hedged']:+.6f}")
    print(f"edge ..................................... {t['edge']:+.6f}")
    print(f"p_itm  P(put ITM in one month) ........... {t['p_itm']:.3e}")
    print(f"p_jump P(a jump occurs in one month) ..... {t['p_jump']:.3e}  (different number)")
    print(f"markup ................................... {t['markup']:.2f}"
          "  <- fair pricing; no variance risk premium")
