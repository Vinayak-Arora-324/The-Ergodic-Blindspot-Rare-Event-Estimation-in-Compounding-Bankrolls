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


#: The three estimators of record, with the colours used in the E3 figure.
ESTIMATORS = (
    ("crude MC", est_crude, "#c0392b"),
    ("+ control variate", est_cv, "#e67e22"),
    ("+ CV + importance sampling", est_cv_is, "#1f6f4a"),
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
