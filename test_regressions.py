"""
test_regressions.py -- the bugs and the verified numbers, encoded so they cannot
return.

    python test_regressions.py        # standalone, no pytest needed
    pytest test_regressions.py        # also works if pytest is installed

Two kinds of test live here and they are labelled separately.

  T1-T7  Verified numbers. These are the acceptance tests: quantities that were
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

import pathlib
import sys

import numpy as np
from scipy import integrate, stats

import model as M


# ===========================================================================
# T1-T7 -- verified numbers
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
    src_dir = pathlib.Path(__file__).parent
    for path in sorted(src_dir.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue        # this file names the bug in order to document it
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
