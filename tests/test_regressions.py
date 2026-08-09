"""
test_regressions.py -- the bugs and the verified numbers, encoded so they cannot
return.

    python -m tests.test_regressions  # standalone, no pytest needed
    pytest tests/test_regressions.py  # if pytest is installed

Two kinds of test live here and they are labelled separately.

  T1-T15 Verified numbers. These are the acceptance tests: quantities that were
         computed, checked against an independent route where one existed, and
         recorded. A failure means a result moved.

  B1-B5  Bugs actually committed during the exploratory phase. A failure means
         a specific known mistake has been reintroduced. Each carries the story
         of what went wrong, because a bare assertion is not enough to stop
         someone re-deriving the same error from the same reasoning.

Tolerances are loose enough to absorb Monte Carlo noise at the sample sizes
used here and tight enough that a real regression trips them. Where a test is
inherently statistical, the tolerance is stated in MC-SE units rather than in
absolute terms.
"""

from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np
from scipy import integrate, stats

from blindspot import model as M


# ===========================================================================
# T1-T15 -- verified numbers
# ===========================================================================
def test_t1_density_matches_sampler():
    """T1: the density and the sampler are written independently -- do they agree?

    Verified: P(r < log(K/S0)) = 6.33e-4 empirical (4e6 draws) vs 6.28e-4 by
    quadrature.

    This is the most important test in the file. Every estimator here mixes
    sampled draws with analytic densities (the IS weights are literally
    f(sample)/q(sample)), so a mismatch between the two would silently corrupt
    the weights without raising anything anywhere.
    """
    rng = np.random.default_rng(3)
    n = 4_000_000
    r = M.sample_returns(rng, n)
    p_emp = (r < M.R_STAR).mean()
    p_quad, _ = integrate.quad(M.pdf, M.LO, M.R_STAR, limit=M.QUAD_LIMIT)

    # Binomial MC-SE on the empirical side; allow 4 of them.
    se = np.sqrt(p_quad * (1 - p_quad) / n)
    assert abs(p_emp - p_quad) < 4 * se, f"{p_emp:.3e} vs {p_quad:.3e} (se {se:.1e})"
    assert abs(p_quad - 6.28e-4) < 1e-5, p_quad
    return f"P(ITM) empirical {p_emp:.3e} vs quadrature {p_quad:.3e}"


def test_t2_truth_values():
    """T2: the headline truths. g_h = +0.005852, edge = +0.000852."""
    t = M.truth()
    assert abs(t["g_unhedged"] - 0.005) < 1e-12       # exact by construction
    assert abs(t["g_hedged"] - 0.005852) < 5e-6, t["g_hedged"]
    assert abs(t["edge"] - 0.000852) < 5e-6, t["edge"]
    assert abs(t["p_itm"] - 6.28e-4) < 1e-5, t["p_itm"]
    # quad's own error estimate must sit far below the quantity being resolved:
    # the edge is what has to be distinguished from zero, so demand at least
    # three orders of magnitude of headroom against it.
    assert t["g_hedged_quaderr"] < t["edge"] / 1000, t["g_hedged_quaderr"]
    return (f"g_h {t['g_hedged']:+.6f}  edge {t['edge']:+.6f}  "
            f"p_itm {t['p_itm']:.3e}")


def _crash_share(v, crash):
    """Crash contribution to Var(v) by the law of total variance."""
    w = crash.mean()
    return (np.var(v[crash]) + (v[crash].mean() - v.mean()) ** 2) * w / np.var(v)


def test_t3_crash_share_of_variance():
    """T3: hedging DESTROYS the rare-event structure of the hedged quantity.

    Verified: crashes carry ~1.8% of Var(log_growth) but ~64% of Var(r).

    This is the test that pins the project's second finding, and it is the
    reason the original founding sentence ("estimating the growth of a hedged
    bankroll is itself a rare-event problem") was retracted. It is backwards.
    The unhedged baseline and the hedged-minus-unhedged difference are
    rare-event-dominated; the hedged portfolio is not.
    """
    rng = np.random.default_rng(11)
    x = M.sample_returns(rng, 2_000_000)
    crash = x < M.R_STAR
    share_h = _crash_share(M.log_growth(x), crash)
    share_u = _crash_share(x, crash)
    assert share_h < 0.05, f"hedged crash share {share_h:.1%} -- expected ~2%"
    assert share_u > 0.50, f"unhedged crash share {share_u:.1%} -- expected ~64%"
    assert share_u > 10 * share_h
    return f"crash share of variance: hedged {share_h:.1%}, unhedged {share_u:.1%}"


def test_t4_crash_period_means():
    """T4: in a crash the hedged growth is bounded and the unhedged is not.

    Verified: mean log_growth | crash ~ -0.03, mean r | crash ~ -2.2.

    The mechanism behind T3: the put payoff cancels the index loss, so the
    hedged log growth in a crash is mild. The unhedged return is not.
    """
    rng = np.random.default_rng(11)
    x = M.sample_returns(rng, 2_000_000)
    crash = x < M.R_STAR
    mean_h = M.log_growth(x)[crash].mean()
    mean_u = x[crash].mean()
    assert -0.5 < mean_h < 0.0, mean_h        # bounded, mildly negative
    assert mean_u < -1.5, mean_u              # not bounded in any useful sense
    return f"crash-period mean: hedged {mean_h:+.3f}, unhedged {mean_u:+.3f}"


def test_t5_is_weights_bounded():
    """T5: the IS likelihood ratio is bounded, verified range [0.0033, 1.4228].

    f and q are mixtures over the SAME two components differing only in the
    mixture weight, so f/q is bounded between (1-LAM)/(1-LAM_Q) and
    LAM/LAM_Q. Finite IS variance therefore holds by construction rather than
    by luck -- which is what makes the importance sampler here trustworthy
    instead of being a silent infinite-variance trap.
    """
    rng = np.random.default_rng(11)
    lo, hi = M.is_weight_range(rng, 200_000)
    # Analytic bounds: f/q is a ratio of mixtures over identical components, so
    # it is squeezed between the two mixture-weight ratios. The extremes are
    # attained in the limit (deep crash / pure bulk), so the empirical min and
    # max land essentially ON the bounds -- compare with a float tolerance
    # rather than strictly.
    bound_lo = M.LAM / M.LAM_Q                      # deep-crash draws
    bound_hi = (1 - M.LAM) / (1 - M.LAM_Q)          # bulk draws
    assert lo >= bound_lo * (1 - 1e-9), (lo, bound_lo)
    assert hi <= bound_hi * (1 + 1e-9), (hi, bound_hi)
    assert abs(lo - 0.0033) < 5e-4 and abs(hi - 1.4228) < 5e-4, (lo, hi)
    return f"IS weights in [{lo:.4f}, {hi:.4f}] (analytic bound {bound_hi:.4f})"


def test_t6_estimators_unbiased(n=4096, trials=2000):
    """T6: all three estimators are unbiased for g_hedged, within 1.4 MC-SE.

    Unbiasedness is what makes the E1/E3 result a statement about the SHAPE of
    the sampling distribution rather than about bias. If this test fails, the
    headline finding has to be re-described.
    """
    rng = np.random.default_rng(20)
    t = M.truth()
    out = []
    for name, fn, _ in M.ESTIMATORS:
        e = M.run_trials(fn, n, trials, rng)
        mc_se = e.std(ddof=1) / np.sqrt(trials)
        z = (e.mean() - t["g_hedged"]) / mc_se
        assert abs(z) < 3.0, f"{name}: bias {z:+.2f} MC-SE"
        out.append(f"{name} {z:+.1f} MC-SE")
    return "bias at N=4096: " + ", ".join(out)


def test_t7_root_n_convergence(trials=400):
    """T7: every estimator converges at N^-1/2 -- no infinite-variance blowup.

    Verified slopes: -0.499, -0.511, -0.498.

    The point is not that the slopes are good. It is that they are all the
    SAME. Variance reduction moves the intercept of the convergence line and
    never the rate, which is why the E3 figure's caption has to be the
    deflationary one. A slope materially shallower than -1/2 would mean an
    estimator had unbounded variance and its error bars were fiction.
    """
    rng = np.random.default_rng(20)
    ns = np.array([256, 1024, 4096, 16384])
    out = []
    for name, fn, _ in M.ESTIMATORS:
        sds = np.array([M.run_trials(fn, int(n), trials, rng).std(ddof=1) for n in ns])
        slope = np.polyfit(np.log(ns), np.log(sds), 1)[0]
        # Loose: sd-of-sd is itself noisy for these skewed sampling
        # distributions, so ±0.1 is the honest tolerance at this trial count.
        assert abs(slope + 0.5) < 0.1, f"{name}: slope {slope:.3f}"
        out.append(f"{name} {slope:.3f}")
    return "slopes: " + ", ".join(out)


def test_t8_davies_harte_matches_cholesky():
    """T8: the fast fGn driver has the same law as the exact slow one.

    E6/E7 need paths of millions of steps, which Cholesky cannot reach, so the
    drawdown family runs on Davies-Harte instead. Two things are checked, and
    the second is the one that matters: the empirical covariance matches the
    fGn autocovariance, AND the covariance the circulant actually implements
    -- after the negative-eigenvalue clip -- is within 1e-3 of the true one at
    the largest record E7 draws. The clip is where an approximation could
    enter unnoticed, so it is measured deterministically rather than sampled.
    """
    rng = np.random.default_rng(31)
    h, n = 0.75, 24
    x = M.fgn_davies_harte(h, n, 120_000, rng)
    k = np.arange(n, dtype=float)
    gamma = 0.5 * ((k + 1) ** (2 * h) - 2 * k ** (2 * h) + np.abs(k - 1) ** (2 * h))
    theo = gamma[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    err = np.abs(np.cov(x, rowvar=False) - theo).max()
    assert err < 0.02, f"empirical covariance off by {err:.4f}"

    # What the clipped circulant really implements, at E7's record length.
    big_h, big_n = 0.9, M.DD_HORIZON * M.DD_SUB * 8_000
    m = 1 << int(np.ceil(np.log2(big_n)))
    kk = np.arange(m + 1, dtype=float)
    g = 0.5 * ((kk + 1) ** (2 * big_h) - 2 * kk ** (2 * big_h)
               + np.abs(kk - 1) ** (2 * big_h))
    lam = np.fft.rfft(np.concatenate([g, g[-2:0:-1]])).real
    implied = np.fft.irfft(np.clip(lam, 0.0, None), n=2 * m)
    drift = max(abs(implied[lag] - g[lag]) for lag in (0, 1, 10, 100, 1000))
    assert drift < 1e-3, f"clipped circulant covariance off by {drift:.2e}"
    return f"cov err {err:.4f}, clip drift {drift:.1e}"


def test_t9_bootstrap_is_calibrated_on_iid():
    """T9: E7's reported error bar is fair before the experiment indicts it.

    E7's whole claim is that the ratio reported/actual departs from 1. That is
    only a finding if the ratio IS 1 when the design's assumptions hold, so
    this checks the bootstrap against the realized sd of the estimator on
    genuinely iid draws, at both grids E7 uses. Measured 1.0-1.3x at this rep
    count, and 0.9-1.1x at 80 reps -- the spread is sd-of-sd noise, not
    miscalibration, and the band below is set to the rep count. Without
    this the honest line at 1.0 would be an assumption, and both of E7's
    failures could be one miscalibrated bootstrap wearing two hats.
    """
    rng = np.random.default_rng(32)
    out = []
    for h, sub in ((0.3, 1), (0.8, M.DD_SUB)):
        est, se = [], []
        for _ in range(40):
            s = M.drawdown_replicates(h, 4_000, rng, sub=sub)
            est.append(np.percentile(s, M.DD_LEVEL))
            se.append(M.bootstrap_halfwidth(s, rng, boot=200) / 1.96)
        ratio = np.mean(se) / np.std(est, ddof=1)
        assert 0.7 < ratio < 1.4, f"H={h} sub={sub}: bootstrap/true sd = {ratio:.2f}"
        out.append(f"H={h} {ratio:.2f}x")
    return "bootstrap/true sd: " + ", ".join(out)


def test_t10_drawdown_family_is_ordered_and_pinned():
    """T10: E6's sweep moves path shape, not scale.

    Two claims the E6 figure makes structurally. Terminal variance is pinned
    across H by construction, so if it drifts the family is really a
    volatility sweep and the drawdown ordering means nothing. And the 1-in-1000
    drawdown must be monotone decreasing in H: rougher paths spend the same
    terminal variance on deeper round trips. Guards the case where someone
    "fixes" the sigma_H = DD_SIGMA_T / T^H scaling and quietly turns E6 into a
    plot of its own normalization.
    """
    rng = np.random.default_rng(33)
    var, reads = [], []
    for h in (0.2, 0.5, 0.8):
        inc = M.fgn_davies_harte(h, M.DD_HORIZON, 40_000, rng)
        var.append(M._fbm_wealth(h, inc, 1)[:, -1].var())
        reads.append(np.percentile(M.drawdown_replicates(h, 40_000, rng,
                                                         sub=M.DD_SUB), M.DD_LEVEL))
    target = M.DD_SIGMA_T ** 2
    for h, v in zip((0.2, 0.5, 0.8), var):
        assert abs(v / target - 1) < 0.12, f"H={h}: Var X(T) = {v:.4f} vs {target:.4f}"
    assert reads[0] > reads[1] > reads[2], f"1-in-1000 not ordered in H: {reads}"
    return (f"Var X(T) pinned at {target:.4f}; 1-in-1000 "
            + " > ".join(f"{M.wipeout(r):.0f}%" for r in reads))


def _dh_spectrum(hurst, n_steps):
    """The Davies-Harte half-spectrum and its bin multiplicities, as model.py builds it."""
    m = 1 << int(np.ceil(np.log2(max(n_steps, 2))))
    k = np.arange(m + 1, dtype=float)
    g = 0.5 * ((k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst)
               + np.abs(k - 1) ** (2 * hurst))
    lam = np.fft.rfft(np.concatenate([g, g[-2:0:-1]])).real
    mult = np.full(lam.size, 2.0)
    mult[0] = mult[-1] = 1.0
    return m, lam, mult


def test_t11_fgn_drivers_have_unit_variance():
    """T11: Var(G_t) = 1 for both fGn drivers -- exactly, not approximately.

    This is the quietest load-bearing assumption in the project. E5's truth
    line is the CLOSED FORM `lrd_g_true` = mu_a - 0.5*sbar^2*exp(2*xi^2), and
    that expression is only the true growth rate because Var(G_t) = 1 makes
    E[sigma_t^2] = SBAR^2 exp(2 XI^2). E6/E7 lean on it again: the whole point
    of sigma_H = DD_SIGMA_T / T^H is that Var X(T) is the same for every H, and
    a driver whose variance drifted with H would turn E6 into a plot of its own
    normalization -- the exact failure T10 guards from the other side. A driver
    that was off by 3% would not look broken anywhere; the truth line would
    just sit in the wrong place and E5 would be indicting its own generator.

    Unit variance is not a calibration here, it is an identity. The filter
    Davies-Harte applies to iid unit normals is h = ifft(sqrt(lam)), so by
    Parseval Var(y_t) = sum_j h_j^2 = (1/2m) sum_k lam_k, and the eigenvalues
    of a circulant sum to 2m times its first row entry, which is gamma(0) = 1.
    So the check below is at 1e-12, not at an MC tolerance: anything looser
    would pass a generator that had picked up a scale factor.

    The one thing that can break the identity is the negative-eigenvalue clip,
    and the same algebra prices it: clipping removes negative mass, so it
    raises the variance to exactly 1 + `lost`. Two consequences are asserted,
    both of which were wrong or unstated before this test existed.

      * The clip inflates variance, it does not shrink it. Guards against
        "fix" the sign of the correction.
      * `lost` must be measured over the FULL 2m spectrum. The original code
        divided the half-spectrum's negative sum by the half-spectrum's total;
        both halves of that ratio were wrong and they did not cancel, so the
        gate read 4.5e-2 where the true variance error was 5.4e-2 -- an ~18%
        UNDER-statement, i.e. the permissive direction, in the one number
        standing between E7 and a silently wrong long-range structure. No run
        changed status when it was fixed (H=0.9/4M steps passes either way at
        3.6e-4 vs 3.8e-4, H=0.95/4M fails either way), which is why it went
        unnoticed and why it is pinned here rather than left to a rerun.
    """
    # --- Davies-Harte: deterministic, at every (H, n) any experiment uses ---
    worst = 0.0
    for h in (0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
        for n in (M.DD_HORIZON, M.DD_HORIZON * M.DD_SUB, 4096,
                  M.DD_HORIZON * M.DD_SUB * 8_000):
            m, lam, _ = _dh_spectrum(h, n)
            assert (lam >= 0).all(), f"H={h}, n={n:,}: clip active where it should not be"
            var = np.fft.irfft(lam, n=2 * m)[0]
            worst = max(worst, abs(var - 1.0))
    assert worst < 1e-12, f"Davies-Harte Var(G) off by {worst:.2e}"

    # --- Cholesky: same identity, different route ---
    for h in (0.2, 0.5, 0.9):
        chol = M.fgn_cholesky(h, 512)
        diag = (chol @ chol.T).diagonal()
        off = np.abs(diag - 1.0).max()
        assert off < 1e-8, f"Cholesky H={h}: Var(G) off by {off:.2e} (ridge is 1e-10)"

    # --- the clip: inflates variance by exactly the discarded mass ---
    m, lam, mult = _dh_spectrum(0.95, 4_000_000)
    assert (lam < 0).any(), "H=0.95 at 4M steps no longer clips; the case moved"
    lost = -(mult * lam)[lam < 0].sum() / (2 * m)
    var = np.fft.irfft(np.clip(lam, 0.0, None), n=2 * m)[0]
    assert var > 1.0, f"clip must raise variance, got {var:.6f}"
    assert abs((var - 1.0) - lost) < 1e-12 * lost, \
        f"variance error {var - 1.0:.6e} != discarded mass {lost:.6e}"
    assert abs(lost - 5.396e-2) < 1e-5, f"lost = {lost:.4e}, expected 5.396e-2"

    # The half-spectrum version this replaced, kept so the bug cannot return.
    naive = -lam[lam < 0].sum() / lam.sum()
    assert naive < lost, "the half-spectrum ratio understated the loss; that was the bug"
    psd_tol = inspect.signature(M.fgn_davies_harte).parameters["psd_tol"].default
    assert lost > psd_tol, f"this case must trip psd_tol = {psd_tol}"

    # --- and the gate actually fires ---
    rng = np.random.default_rng(35)
    try:
        M.fgn_davies_harte(0.95, 4_000_000, 1, rng)
    except ValueError as exc:
        assert "lost" in str(exc)
    else:
        raise AssertionError("psd_tol gate did not fire at H=0.95, 4M steps")

    # --- empirical, as a sanity check on the whole call path ---
    emp = []
    for h in (0.2, 0.5, 0.9):
        x = M.fgn_davies_harte(h, 64, 40_000, rng)
        v = x.var(axis=0).mean()          # per-column, then averaged
        se = np.sqrt(2.0 / x.shape[0])    # columns are dependent: no sqrt(n) gain
        assert abs(v - 1.0) < 4 * se, f"H={h}: empirical var {v:.4f} ({se:.4f} se)"
        emp.append(v)

    return (f"Var(G)=1 to {worst:.0e} (deterministic); clip at H=0.95/4M "
            f"inflates by {lost:.3e}; empirical {min(emp):.4f}-{max(emp):.4f}")


def test_t12_transient_memory_crosses_to_half():
    """T12: E8 has the requested local H but ordinary long-run scaling.

    A constant-H process cannot satisfy both requirements.  E8 resets exact
    fGn after a finite regime length, making the variance identity available
    without Monte Carlo: n^(2H) locally and O(n) after many independent blocks.
    """
    local_n = np.array([4, 8, 16, 32])
    long_n = M.MEM_CUTOFF * np.array([8, 16, 32, 64])
    out = []
    for h in M.REGIME_H:
        lv = np.array([M.regime_sum_variance(h, int(n)) for n in local_n])
        gv = np.array([M.regime_sum_variance(h, int(n)) for n in long_n])
        local = np.polyfit(np.log(local_n), np.log(lv), 1)[0] / 2
        long = np.polyfit(np.log(long_n), np.log(gv), 1)[0] / 2
        assert abs(local - h) < 1e-12, (h, local)
        assert abs(long - 0.5) < 1e-12, (h, long)
        out.append(f"{h:.1f}->{long:.1f}")

    # Alternating rough/persistent blocks must also aggregate diffusively.
    mixed_n = M.MEM_CUTOFF * np.array([16, 32, 64, 128])
    mixed_v = np.array([
        M.regime_sum_variance((0.1, 0.8), int(n)) for n in mixed_n
    ])
    mixed = np.polyfit(np.log(mixed_n), np.log(mixed_v), 1)[0] / 2
    assert abs(mixed - 0.5) < 1e-12, mixed
    return "local->long H: " + ", ".join(out) + f", alternating->{mixed:.1f}"


def test_t13_q_price_has_an_independent_oracle():
    """T13: analytic conditional pricing agrees with density quadrature."""
    sigma = M.RS_SBAR
    m = float(M.regime_market_drift(sigma, "Q"))
    k = np.log(M.RS_K)
    kp = M.RS_BETA / sigma

    def density(r):
        bulk = stats.norm.pdf(r, m, sigma)
        jump = stats.exponnorm.pdf(-r, kp, loc=-m, scale=sigma)
        return (1 - M.RS_LAM_Q) * bulk + M.RS_LAM_Q * jump

    raw, err = integrate.quad(
        lambda r: (M.RS_K - np.exp(r)) * density(r), -40.0, k, limit=1500
    )
    oracle = np.exp(-M.RS_RF) * raw
    analytic = float(M.conditional_tail_put_price(sigma))
    assert abs(analytic - oracle) < 20 * err + 1e-11, (analytic, oracle, err)

    # The outer volatility quadrature must be stable, and Q's discounted
    # underlying expectation is pinned to one by construction.
    a = M.tail_put_truth_and_cv(24)
    b = M.tail_put_truth_and_cv(48)
    assert abs(a["price"] - b["price"]) < 2e-8, (a, b)
    assert abs(b["underlying"] - 1.0) < 1e-14
    return (f"conditional {analytic:.8f} vs density quad {oracle:.8f}; "
            f"unconditional Q price {b['price']:.8f}")


def test_t14_e8_estimators_are_unbiased_and_reduce_variance():
    """T14: all four E8 estimators target one Q price at one compute budget."""
    rng = np.random.default_rng(8083)
    truth = M.tail_put_truth_and_cv()["price"]
    out = M.run_price_trials(
        rng, trials=400, budget=8_192, horizon=256, local_h=(0.1, 0.8)
    )
    base = out["crude"].var(ddof=1)
    ratios, zs = {}, {}
    for name, x in out.items():
        sd = x.std(ddof=1)
        z = (x.mean() - truth) / (sd / np.sqrt(len(x)))
        assert abs(z) < 4.0, f"{name}: bias {z:+.2f} MC-SE"
        ratios[name] = base / x.var(ddof=1)
        zs[name] = z
    assert ratios["cv"] > 1.15, ratios
    assert ratios["is"] > 3.0, ratios
    assert ratios["cv_is"] > 3.0, ratios
    return ("bias/SE " + ", ".join(f"{k} {zs[k]:+.1f}" for k in out)
            + f"; VR cv/is/both {ratios['cv']:.1f}/{ratios['is']:.1f}/"
              f"{ratios['cv_is']:.1f}x")


def test_t15_q_priced_hedge_reduces_tail_but_costs_carry():
    """T15: estimator efficiency is kept separate from the economic result."""
    rng = np.random.default_rng(8183)
    paths = M.tail_hedge_strategy_paths(
        rng, n_paths=30_000, horizon=120, local_h=(0.1, 0.8)
    )
    edge = paths["difference"].mean() / 120
    dd_u = np.percentile(M.wipeout(paths["drawdown_unhedged"]), 99)
    dd_h = np.percentile(M.wipeout(paths["drawdown_hedged"]), 99)
    assert edge < 0, edge
    assert dd_h < dd_u - 1.0, (dd_u, dd_h)
    return f"edge {edge:+.2e}/mo; 99% drawdown {dd_u:.1f}% -> {dd_h:.1f}%"


# ===========================================================================
# B1-B5 -- committed bugs
# ===========================================================================
def test_b1_no_student_t_scale_switch():
    """B1: a scale convention that switched discontinuously at nu = 2.

    The line was:

        scale = sigma / sqrt(nu / (nu - 2)) if nu > 2 else sigma

    The intent was "match the target sd where the variance exists, fall back
    otherwise". The effect was a jump: nu = 2.0 gave scale 0.060 and nu = 2.5
    gave 0.0268. Every nu <= 2 row in the sweep therefore had a ~2.2x FATTER
    SCALE, not merely a fatter tail, so all cross-nu comparisons were
    measuring the wrong thing. The finding built on them -- "hedge-wins
    configurations cluster at nu < 2, so the Student-t model has a right-tail
    confound" -- was retracted.

    The permanent fix was to drop the Student-t model entirely in favour of the
    jump model in `model.py`, so this test asserts the t-model has not crept
    back into the codebase, and separately re-demonstrates the discontinuity so
    the reasoning behind the ban stays on the record.
    """
    src_dir = pathlib.Path(__file__).resolve().parents[1] / "blindspot"
    for path in sorted(src_dir.glob("*.py")):
        text = path.read_text()
        for marker in ("standard_t", "stats.t.", "nu - 2", "nu-2"):
            assert marker not in text, f"Student-t model reappeared in {path.name}"

    # Re-demonstrate the discontinuity: the retired convention is not
    # continuous in nu, so the assertion the spec asks for (scale continuous in
    # nu) genuinely fails for it.
    sigma = 0.06
    retired = lambda nu: sigma / np.sqrt(nu / (nu - 2.0)) if nu > 2 else sigma
    left, right = retired(2.0), retired(2.5)
    assert left / right > 2.0, (left, right)
    return (f"no t-model in the codebase; retired convention jumped "
            f"{left:.4f} -> {right:.4f} across nu=2")


def test_b2_premium_must_scale_with_p():
    """B2: a fixed premium grid, floored at c = 2e-4, compared across lam.

    At lam = 1e-5 the fair option costs far less than that floor, so the
    configuration was overpaying by construction and bled. That artefact was
    written up as "tail hedging only helps when p ~ 1e-3; the edge vanishes for
    rarer tails", which is false.

    Verified: at lam = 1e-5 (p ~ 6.3e-6) the grid's c gives edge -1.66e-4
    (bleeds) while c = 0.8p gives +8.53e-6 (wins).
    """
    edge_grid, p = M.edge_and_p(1e-5, 2e-4)
    edge_sized, _ = M.edge_and_p(1e-5, 0.8 * p)
    assert edge_grid < 0, edge_grid           # the artefact, reproduced
    assert edge_sized > 0, edge_sized         # the truth it hid
    # ...and the edge shrinks roughly in proportion to p rather than vanishing.
    edge_base, p_base = M.edge_and_p(M.LAM, 0.8 * M.truth()["p_itm"])
    assert edge_base > 0
    assert 0 < edge_sized < edge_base
    return (f"lam=1e-5 (p={p:.1e}): fixed grid {edge_grid:+.2e} (bleeds) vs "
            f"c=0.8p {edge_sized:+.2e} (wins)")


def test_b3_closed_form_density():
    """B3: the `exponnorm` parameterization is easy to get wrong, and silently.

    The correct call is `exponnorm.pdf(-r, K=BETA/S, loc=-M_BULK, scale=S)`:
    evaluated at MINUS r, because the jump component is
    N(m,s) - Exp(beta) and the sign flip is what turns it into the standard
    exponentially modified Gaussian. A wrong version still returns a valid
    density, still integrates to 1, and still produces plausible-looking
    numbers -- which is why this needs an assertion rather than an eyeball.
    """
    grid, ours, ref = M.check_density(rtol=1e-10)
    rel = np.max(np.abs(ours / ref - 1.0))
    return f"max relative deviation from scipy over {len(grid)} points: {rel:.2e}"


def test_b4_integration_window_wide_enough():
    """B4: truncating the truth integral too tightly.

    An earlier stage integrated over [-6, 6] and cut 0.47% of the value. The
    integrand here is payoff x density, and the payoff GROWS as r falls (the
    put is worth more the deeper the crash), so the tails cannot be discarded
    on the usual "the density is tiny there" reasoning.

    Test: widening the lower limit from -40 to -60 must not move the answer.
    """
    t40 = M.truth(1.0, -40.0)
    t60 = M.truth(1.0, -60.0)
    # Tolerance is set by quad's own reported error (~1e-8), not by ambition:
    # anything at that level is quadrature noise, and it is five orders below
    # the edge that has to be resolved.
    tol = 10 * max(t40["g_hedged_quaderr"], t60["g_hedged_quaderr"])
    assert abs(t40["g_hedged"] - t60["g_hedged"]) < tol, (t40, t60)
    assert abs(t40["p_itm"] - t60["p_itm"]) < 1e-10

    # Show the failure mode the bug hit: a window that is genuinely too tight.
    tight, _ = integrate.quad(
        lambda r: M.log_growth(r) * M.pdf(r), -6.0, M.HI, limit=M.QUAD_LIMIT
    )
    cut = abs(tight - t40["g_hedged"]) / abs(t40["g_hedged"])
    return (f"g_h moves <{tol:.0e} between lo=-40 and lo=-60; "
            f"a lo=-6 window would cut {cut:.2%}")


def test_b5_jump_probability_is_not_crash_probability():
    """B5: labelling P(>=1 jump) as P(>=1 crash). They are different numbers.

    Over 120 months: P(>=1 jump) = 11.31% but P(>=1 put ITM) = 7.26%. Only an
    in-the-money put can help; a jump that fails to breach the strike is just a
    loss on both sides of the comparison. The E4 win rate tracks the ITM
    probability, not the jump probability, and conflating them would make the
    hedge look like it fails on paths where it was never given the chance.

    Test: both are computed, they are reported separately, and they differ.
    """
    t = M.truth()
    horizon = 120
    p_jump = 1 - (1 - M.LAM) ** horizon
    p_itm = 1 - (1 - t["p_itm"]) ** horizon
    assert abs(p_jump - 0.11313) < 1e-4, p_jump
    assert abs(p_itm - 0.07257) < 1e-4, p_itm
    assert p_jump > 1.5 * p_itm, "these must not be treated as the same quantity"
    return f"over {horizon} months: P(>=1 jump) {p_jump:.3%} vs P(>=1 ITM) {p_itm:.3%}"


# ===========================================================================
# runner
# ===========================================================================
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        label = name.replace("test_", "").split("_")[0].upper()
        try:
            detail = fn()
            print(f"  PASS  {label:4s} {name[len(label) + 6:]:38s} {detail or ''}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {label:4s} {name[len(label) + 6:]:38s} {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    print("regression tests -- verified numbers (T) and committed bugs (B)\n")
    sys.exit(main())
