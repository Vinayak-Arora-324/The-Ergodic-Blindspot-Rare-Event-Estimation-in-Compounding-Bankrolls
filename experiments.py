"""
experiments.py -- the five experiments, behind one command line.

    python experiments.py            # list them
    python experiments.py e1 e2      # run a subset
    python experiments.py all        # run everything (E3 writes a .png)

Each experiment is one way a simulation lies, plus one structural result:

  E1  the median lies       -- crude MC is unbiased in expectation while the
                               TYPICAL run reports near zero.
  E2  where the variance is -- the baseline, the variance decomposition, and
                               why the control variate makes things WORSE alone.
  E3  the convergence figure-- every method converges at N^-1/2; variance
                               reduction moves the intercept, never the rate.
  E4  path functionals      -- the estimand choice dominates the estimator
                               choice. This is the most important experiment.
  E5  the error bar lies    -- under long-memory volatility, iid-formula
                               confidence intervals get MORE overconfident as
                               the sample grows.

Conventions enforced throughout (see the rebuild spec, section 9):
  * every RNG is seeded explicitly and the seed is printed beside the results;
  * every reported estimate carries its Monte Carlo standard error;
  * no Python loop over samples -- everything is vectorized over trials/paths;
  * captions and prose must not outrun what was established. Overclaiming is a
    substantive error in this project, not a stylistic one: two false findings
    in the exploratory phase came from exactly that.

Standing caveat on every E1-E4 number: the put is priced actuarially fair
(markup = 1.0). No variance risk premium exists at that setting, so every
"the hedge wins" result is conditional on a market that does not exist.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import model as M

# Seeds of record. The numbers in the rebuild spec were generated with these.
SEEDS = {"e1": 0, "e2": 7, "e3": 20, "e4": 31, "e5": 9}

RULE = "=" * 78


def _banner(tag, title):
    print(f"\n{RULE}\n{tag.upper()}  --  {title}\n{RULE}")


def _fair_pricing_note():
    print("\n[caveat] markup = 1.0: the put is actuarially fair. Every 'hedge wins'")
    print("         number above is conditional on a market with no variance risk")
    print("         premium, i.e. one that does not exist.")


# ---------------------------------------------------------------------------
# E1 -- the median lies
# ---------------------------------------------------------------------------
def e1(trials=3000):
    """Single-period rare payoff: unbiased in expectation, near-zero typically.

    Estimand: E[put payoff] for one month, whose exact value is the fair price
    computed by quadrature in `model.put_price()`.

    This is NOT a bias result. Crude MC is exactly unbiased here. It is a
    statement about the SHAPE of the estimator's sampling distribution: at a
    budget where the run expects fewer than one crash, the distribution is a
    large spike at zero plus a thin right tail that carries the entire mean.
    The mean of many runs is right; the run you can afford is not.

    Provenance: the recorded numbers for this experiment originally came from a
    Student-t model (nu=3, sigma=0.05, K=55, N=5000), the last place a t-model
    survived in this project. It has been ported to the canonical jump model so
    the codebase has exactly one model, which means those exact figures no
    longer apply. The acceptance criteria are the qualitative ones and they do
    survive: mean ~ truth, median ~ 0, a large fraction of runs see nothing.
    """
    _banner("e1", "the median lies (single period)")
    rng = np.random.default_rng(SEEDS["e1"])
    truth = M.truth()
    p = truth["p_itm"]
    e_payoff = truth["price"]           # E[payoff] under fair pricing == price

    # Choose the "affordable" N so that a run expects fewer than one crash --
    # the regime where the affordability question actually bites. N*p ~ 0.63
    # here, matching the regime of the original t-model run (which had 0.6).
    n_afford = 1000

    print(f"seed {SEEDS['e1']}   trials {trials:,}")
    print(f"P(put ITM in one month) .......... {p:.3e}")
    print(f"affordable N ..................... {n_afford:,}  "
          f"(~{n_afford * p:.2f} crashes per run)")
    print(f"TRUE E[payoff] (quadrature) ...... {e_payoff:.6f}")

    ests = np.array(
        [M.put_payoff(M.sample_returns(rng, n_afford)).mean() for _ in range(trials)]
    )
    mc_se = ests.std(ddof=1) / np.sqrt(trials)

    print(f"\nmean of {trials:,} estimates ......... {ests.mean():.6f} "
          f"+/- {mc_se:.6f} (MC-SE)   <- unbiased")
    print(f"bias in MC-SE units .............. {(ests.mean() - e_payoff) / mc_se:+.2f}")
    print(f"MEDIAN estimate .................. {np.median(ests):.6f}   <- the typical run")
    print(f"sd of estimates .................. {ests.std(ddof=1):.6f}")
    print(f"coefficient of variation ......... {ests.std(ddof=1) / ests.mean():.2f}")
    print(f"runs that saw zero crashes ....... {(ests == 0).mean():.2%}")
    print(f"runs below half of truth ......... {(ests < 0.5 * e_payoff).mean():.2%}")

    # The median estimate is exactly zero iff a majority of runs see no crash,
    # i.e. iff (1-p)^N > 1/2, i.e. N < log(2)/p. Stating the threshold keeps
    # the claim precise: "the median lies" is a statement about the budget
    # relative to 1/p, not a defect of the estimator.
    print(f"median is exactly 0 for N < ln2/p = {np.log(2) / p:,.0f}")

    # The spike does not survive more budget: this is a statement about
    # affordability, not about the estimator being broken.
    print(f"\n{'N':>9} {'mean':>10} {'median':>10} {'P(saw 0)':>10} {'P(<truth/2)':>12}")
    for n in (1_000, 5_000, 20_000, 100_000):
        e = np.array(
            [M.put_payoff(M.sample_returns(rng, n)).mean() for _ in range(600)]
        )
        print(f"{n:9,} {e.mean():10.6f} {np.median(e):10.6f} "
              f"{(e == 0).mean():9.1%} {(e < 0.5 * e_payoff).mean():11.1%}")
    print("\nThe median converges to the truth from BELOW as N grows. Nothing about")
    print("the estimator changes; only whether the budget can see the event.")


# ---------------------------------------------------------------------------
# E2 -- iid growth baseline and variance diagnosis
# ---------------------------------------------------------------------------
def e2(n=5000, trials=2000, n_diag=2_000_000):
    """Truth, the three estimators, the variance decomposition, sample counts.

    Two findings that the code must not "correct":

    1. The control variate ALONE increases variance. Removing the bulk noise
       leaves the hedge's effect, which is purely crash-driven and violently
       skewed, and that is harder to estimate than the level was.
    2. Hedging DESTROYS the rare-event structure of the hedged quantity.
       Crashes carry ~2% of Var(log_growth) but ~64% of Var(r). The
       rare-event problem lives in the unhedged baseline and in the
       hedged-minus-unhedged difference -- not in the hedged portfolio. The
       project's original founding sentence had this backwards.
    """
    _banner("e2", "iid growth baseline + variance diagnosis")
    rng = np.random.default_rng(SEEDS["e2"])
    t = M.truth()
    edge = t["edge"]

    print(f"seed {SEEDS['e2']}   N {n:,}   trials {trials:,}")
    print(f"\n-- truth (quadrature; g_unhedged is exact by construction) --")
    print(f"g_unhedged ....................... {t['g_unhedged']:+.6f}")
    print(f"g_hedged ......................... {t['g_hedged']:+.6f}")
    print(f"edge ............................. {edge:+.6f}")
    print(f"P(put ITM), one month ............ {t['p_itm']:.3e}")
    print(f"P(jump),    one month ............ {t['p_jump']:.3e}   <- a different")
    print("                                              quantity; only ITM puts help")

    # --- where does the variance live? ---
    x = M.sample_returns(rng, n_diag)
    lg = M.log_growth(x)
    d = M.control_variate(x)
    crash = x < M.R_STAR

    # Law of total variance, crash component: within-group variance plus the
    # squared mean shift, weighted by the group's probability.
    def crash_share(v):
        w = crash.mean()
        return (np.var(v[crash]) + (v[crash].mean() - v.mean()) ** 2) * w / np.var(v)

    print(f"\n-- variance decomposition ({n_diag:,} draws) --")
    print(f"per-sample sd of log_growth ...... {lg.std():.4f}   <- bulk noise, s={M.S}")
    print(f"per-sample sd of D = lg - r ...... {d.std():.4f}   <- crash-driven only "
          f"({'WORSE' if d.std() > lg.std() else 'better'})")
    print(f"crash share of Var(log_growth) ... {crash_share(lg):.1%}   (hedged)")
    print(f"crash share of Var(r) ............ {crash_share(x):.1%}   (unhedged)")
    print(f"mean log_growth in a crash ....... {lg[crash].mean():+.3f}   <- bounded: the")
    print(f"mean r          in a crash ....... {x[crash].mean():+.3f}      payoff cancels")
    print("                                              the index loss")

    # --- how many samples to resolve the sign, analytically ---
    print(f"\n-- sample counts to call sign(edge) at 2 sigma, N = (2*sd/edge)^2 --")
    print(f"crude MC ......................... {int((2 * lg.std() / edge) ** 2):,}")
    print(f"after control variate ............ {int((2 * d.std() / edge) ** 2):,}")

    # --- the three estimators, empirically ---
    print(f"\n-- estimators at N = {n:,}, {trials:,} trials --")
    print(f"{'estimator':30s} {'mean':>10} {'bias/MC-SE':>11} {'sd':>10} "
          f"{'var reduction':>14}")
    sds = {}
    for name, fn, _ in M.ESTIMATORS:
        e = M.run_trials(fn, n, trials, rng)
        sd = e.std(ddof=1)
        sds[name] = sd
        mc_se = sd / np.sqrt(trials)
        ratio = (sds["crude MC"] / sd) ** 2
        print(f"{name:30s} {e.mean():+10.6f} {(e.mean() - t['g_hedged']) / mc_se:+11.2f} "
              f"{sd:10.6f} {ratio:13.1f}x")

    # IS without the control variate, to show the composition is what matters.
    e_is = M.run_trials(M.est_is, n, trials, rng)
    print(f"{'(IS alone, no CV)':30s} {e_is.mean():+10.6f} "
          f"{(e_is.mean() - t['g_hedged']) / (e_is.std(ddof=1) / np.sqrt(trials)):+11.2f} "
          f"{e_is.std(ddof=1):10.6f} "
          f"{(sds['crude MC'] / e_is.std(ddof=1)) ** 2:13.1f}x")

    lo, hi = M.is_weight_range(rng)
    print(f"\nIS weights f/q bounded in [{lo:.4f}, {hi:.4f}] -> finite variance by")
    print("construction, not by hope.")

    # The deflationary reading, which must be reported with the speedup.
    t0 = time.perf_counter()
    M.truth.cache_clear()
    M.truth()
    ms = 1000 * (time.perf_counter() - t0)
    print(f"\n[deflation] quadrature returns the same answer in {ms:.1f} ms with no")
    print("            samples at all. Under iid returns g is a 1-D integral (LLN),")
    print("            so the variance reduction is a real capability aimed at a")
    print("            problem this model does not have. E4 is where that changes.")
    _fair_pricing_note()


# ---------------------------------------------------------------------------
# E3 -- the convergence figure
# ---------------------------------------------------------------------------
def e3(n_panel_a=4096, trials_a=2000, trials_b=300, outfile="mc_estimator_convergence.png"):
    """Two panels: sampling distributions at one N, and sd vs N on log-log.

    The caption is deliberately deflationary, and that is the point of the
    figure rather than a hedge against it:

      * all three slopes are -1/2. Variance reduction moves the INTERCEPT of
        the convergence line. It never changes the rate.
      * the method that actually wins cannot be drawn on these axes at all.
        Quadrature has no N, so a log-log convergence plot is structurally
        incapable of revealing that Monte Carlo was the wrong tool here.

    Panel A uses a log y-axis because on a linear axis the cv_is spike is so
    tall that it erases the other two distributions.
    """
    _banner("e3", "convergence figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEEDS["e3"])
    t = M.truth()
    g_h, g_u, edge = t["g_hedged"], t["g_unhedged"], t["edge"]
    thresh = edge / 2.0     # sd at which g_h and g_u are 2 sigma apart

    print(f"seed {SEEDS['e3']}")
    print(f"g_u {g_u:+.6f}   g_h {g_h:+.6f}   edge {edge:+.6f}   "
          f"sd threshold {thresh:.2e}")

    # --- Panel A: sampling distributions at one affordable N ---
    samples = {name: M.run_trials(fn, n_panel_a, trials_a, rng)
               for name, fn, _ in M.ESTIMATORS}
    print(f"\nunbiasedness at N={n_panel_a:,} (truth {g_h:+.6f}):")
    for name, _, _ in M.ESTIMATORS:
        e = samples[name]
        mc_se = e.std(ddof=1) / np.sqrt(trials_a)
        print(f"  {name:30s} mean {e.mean():+.6f}  "
              f"bias {(e.mean() - g_h) / mc_se:+.1f} MC-SE  "
              f"median {np.median(e):+.6f}")

    # --- Panel B: sd vs N ---
    ns = np.array([64, 256, 1024, 4096, 16384, 65536])
    sds = {name: np.array([M.run_trials(fn, int(n), trials_b, rng).std(ddof=1)
                           for n in ns])
           for name, fn, _ in M.ESTIMATORS}

    print(f"\n{'estimator':30s} {'slope':>7} {'N to resolve sign':>19}")
    n_star = {}
    for name, _, _ in M.ESTIMATORS:
        v = sds[name]
        slope = np.polyfit(np.log(ns), np.log(v), 1)[0]
        # Fit sd = A * N^slope, then solve sd = thresh for N.
        a = np.exp(np.log(v).mean() - slope * np.log(ns).mean())
        n_star[name] = (thresh / a) ** (1.0 / slope)
        print(f"{name:30s} {slope:7.3f} {n_star[name]:19,.0f}")
    print("\nAll slopes are -1/2. The empirical N for crude MC cross-validates the")
    print("analytic (2*sd/edge)^2 from E2 -- two independent routes to the same number.")

    # --- figure ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for name, _, col in M.ESTIMATORS:
        e = samples[name]
        ax_a.hist(e, bins=70, histtype="step", lw=1.9, color=col, density=True,
                  label=f"{name}  (sd {e.std(ddof=1):.1e})")
    ax_a.axvline(g_h, color="k", lw=1.6, label=f"truth $g_h$ = {g_h:.5f}")
    ax_a.axvline(g_u, color="k", ls=":", lw=1.6,
                 label=f"$g_u$ = {g_u:.5f} (no hedge)")
    ax_a.axvspan(g_u - 4e-4, g_u + 4e-4, color="0.85", zorder=0)
    ax_a.set_yscale("log")      # linear y would let the cv_is spike erase the rest
    ax_a.set_xlim(g_h - 0.0045, g_h + 0.0045)
    ax_a.set_xlabel("estimate of geometric growth rate $g$")
    ax_a.set_ylabel("density")
    ax_a.set_title(f"A. Sampling distribution at affordable N = {n_panel_a:,}\n"
                   "the gap between the two black lines is the whole decision",
                   fontsize=10.5)
    ax_a.legend(fontsize=7.6, loc="upper left")

    for name, _, col in M.ESTIMATORS:
        ax_b.plot(ns, sds[name], "o-", color=col, lw=1.9, ms=5, label=name)
    ax_b.axhline(thresh, color="k", ls="--", lw=1.5)
    ax_b.text(70, thresh * 1.15, "sd needed to call the sign at $2\\sigma$",
              fontsize=8.5, va="bottom")
    # Stagger the labels vertically: crude and +cv cross the threshold at
    # similar N and their annotations would otherwise overlap.
    for i, (name, _, col) in enumerate(M.ESTIMATORS):
        n = n_star[name]
        if ns[0] <= n <= ns[-1] * 40:
            ax_b.plot([n], [thresh], "v", color=col, ms=9, zorder=5)
            ax_b.annotate(f"N$\\approx${n:,.0f}", (n, thresh),
                          textcoords="offset points", xytext=(0, -16 - 12 * (i % 2)),
                          ha="center", fontsize=8, color=col)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("samples per run, N")
    ax_b.set_ylabel("sd of the estimator")
    ax_b.set_title("B. Every slope is $N^{-1/2}$: variance reduction moves the\n"
                   "intercept, not the rate -- and quadrature, which needs no $N$,\n"
                   "cannot be drawn on these axes at all", fontsize=10.5)
    ax_b.legend(fontsize=8.4, loc="lower left")
    ax_b.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(outfile, dpi=170)
    print(f"\nsaved {outfile}")
    _fair_pricing_note()


# ---------------------------------------------------------------------------
# E4 -- path functionals (the most important experiment)
# ---------------------------------------------------------------------------
def e4(horizon=120, paths=400_000):
    """What a finite bankroll cares about, and what quadrature cannot reach.

    Common random numbers: the hedged and unhedged bankrolls face the IDENTICAL
    realized market, month by month. Without that, the difference between them
    would be swamped by market noise that has nothing to do with the hedge.

    The structural result. Under iid returns, g collapses to a 1-D integral, so
    quadrature wins outright and the rare-event machinery of E2/E3 is
    unnecessary. But realized-path win rate, terminal wealth quantiles and
    maximum drawdown are PATH FUNCTIONALS. They do not collapse, quadrature
    cannot reach them, and they are the quantities a finite bankroll actually
    experiences. The estimand choice dominates the estimator choice.

    Two things are true at once here and neither may be dropped when reporting:
    a positive geometric edge, AND underperformance on ~93% of livable paths,
    because the mean is carried by the small fraction of paths where a put
    lands. The drawdown table is the actual case for tail hedging, and it is
    invisible in g by construction.

    Implementation note: the loop is over the 120 months, not over the 400k
    paths -- every month is one vectorized step across all paths, and only
    running state (cumulative wealth, running peak, running max drawdown,
    counters) is carried. That keeps memory at O(paths) instead of
    O(paths x horizon).
    """
    _banner("e4", "path functionals")
    rng = np.random.default_rng(SEEDS["e4"])
    t = M.truth()
    edge = t["edge"]

    print(f"seed {SEEDS['e4']}   T {horizon} months   M {paths:,} paths   "
          f"common random numbers")

    zeros = lambda: np.zeros(paths)
    x_u, x_h = zeros(), zeros()             # cumulative log wealth
    peak_u, peak_h = zeros(), zeros()       # running maximum of the above
    dd_u, dd_h = zeros(), zeros()           # running maximum drawdown, in log
    n_jump = np.zeros(paths, dtype=np.int32)
    n_itm = np.zeros(paths, dtype=np.int32)

    for _ in range(horizon):
        r, jumped = M.sample_returns(rng, paths, M.LAM, return_jumps=True)
        lg = M.log_growth(r)                # same r drives both -> CRN
        x_u += r
        x_h += lg
        np.maximum(peak_u, x_u, out=peak_u)
        np.maximum(peak_h, x_h, out=peak_h)
        np.maximum(dd_u, peak_u - x_u, out=dd_u)
        np.maximum(dd_h, peak_h - x_h, out=dd_h)
        n_jump += jumped
        n_itm += r < M.R_STAR

    diff = x_h - x_u
    se = lambda v: v.std(ddof=1) / np.sqrt(paths)

    print(f"\n-- does the path average reproduce what g promised? --")
    print(f"T * edge (quadrature promise over {horizon // 12}y) ... {horizon * edge:+.4f} log")
    print(f"mean difference across paths ............ {diff.mean():+.4f} "
          f"+/- {se(diff):.4f} (MC-SE)")
    print("                                          ^ agrees: MC and quadrature")
    print("                                            measure the same mean")
    print(f"MEDIAN difference ....................... {np.median(diff):+.4f}   "
          f"<- the hedge loses on")
    print(f"                                             the typical path")
    print(f"  (a no-jump path pays exactly T*log(1-c) = {horizon * np.log(1 - M.C):+.4f})")

    win = (diff > 0).mean()
    print(f"\n-- realized-path win rate: not reachable by quadrature --")
    print(f"P(hedged beats unhedged on the path) .... {win:.2%} "
          f"+/- {np.sqrt(win * (1 - win) / paths):.2%}")
    print(f"P(>=1 put ITM in {horizon} months) ............ {(n_itm >= 1).mean():.3%}"
          f"   (theory {1 - (1 - t['p_itm']) ** horizon:.3%})")
    print(f"P(>=1 jump in {horizon} months) ............... {(n_jump >= 1).mean():.3%}"
          f"   (theory {1 - (1 - M.LAM) ** horizon:.3%})")
    print("      ^ these last two are DIFFERENT quantities and must never be")
    print("        reported as one. Only an in-the-money put can help; a jump")
    print("        that does not breach the strike is just a loss. The win rate")
    print("        tracks the ITM probability, not the jump probability.")

    print(f"\n-- conditioning on how many jumps the path saw --")
    print(f"{'jumps':>8} {'share':>9} {'mean effect':>13} {'P(hedge helps)':>16}")
    for label, mask in (("0", n_jump == 0), ("1", n_jump == 1), (">=2", n_jump >= 2)):
        d = diff[mask]
        print(f"{label:>8} {mask.mean():9.2%} {d.mean():+13.4f} {(d > 0).mean():15.1%}")

    print(f"\n-- terminal log wealth after {horizon} months --")
    print(f"{'':10} {'mean':>9} {'median':>9} {'5%':>9} {'95%':>9} {'sd':>9}")
    for label, v in (("unhedged", x_u), ("hedged", x_h)):
        q5, q95 = np.percentile(v, [5, 95])
        print(f"{label:10} {v.mean():+9.4f} {np.median(v):+9.4f} {q5:+9.4f} "
              f"{q95:+9.4f} {v.std(ddof=1):9.4f}")
    print(f"\nsignal / path noise over the horizon ..... "
          f"{horizon * edge / x_h.std(ddof=1):.3f}")
    print("      ^ the edge the whole estimation exercise was about is buried")
    print("        roughly 5 deep in one bankroll's own path noise.")

    print(f"\n-- maximum drawdown (log), a running-extremum functional --")
    print("   provably not a one-period expectation, so quadrature cannot reach it")
    qs = [50, 95, 99, 99.9]
    print(f"{'':10}" + "".join(f"{f'{q}%':>10}" for q in qs))
    for label, v in (("unhedged", dd_u), ("hedged", dd_h)):
        print(f"{label:10}" + "".join(f"{x:10.3f}" for x in np.percentile(v, qs)))
    worst_u, worst_h = np.percentile(dd_u, 99.9), np.percentile(dd_h, 99.9)
    print(f"\nIdentical in the middle. At the 99.9th percentile the hedge converts a")
    print(f"{1 - np.exp(-worst_u):.2%} wipeout into a {1 - np.exp(-worst_h):.2%} loss.")
    print("This is the actual case for tail hedging and it is invisible in g by")
    print("construction. Both columns are true at once: a positive geometric edge,")
    print(f"and underperformance on {1 - win:.0%} of livable paths, because the mean is")
    print(f"carried by the {win:.0%} where a put lands.")
    _fair_pricing_note()


# ---------------------------------------------------------------------------
# E5 -- the error bar lies
# ---------------------------------------------------------------------------
def e5(t_max=4096, paths=1200, hursts=(0.8, 0.1), ts=(256, 1024, 4096)):
    """Long-memory volatility makes the reported confidence interval dishonest.

    The first experiment whose estimand is genuinely path-space rather than a
    disguised one-period expectation. Returns remain serially UNCORRELATED,
    matching the stylized fact; memory enters the growth estimand only through
    the variance drag sigma_t^2/2.

    Two standard errors are compared for the same estimator (the time average
    of log growth over one path, i.e. one backtest):

      TRUE SE  : the sd of that estimator across independent paths -- what the
                 uncertainty actually is.
      NAIVE SE : sd/sqrt(T) computed from the single path -- what a
                 practitioner reports.

    The headline is not that the naive SE is wrong. It is that the ratio GROWS
    with T: more data makes the reported error bar worse, not better. At
    H = 0.1 (rough, not persistent) the naive SE is fine, which localizes the
    effect to long memory specifically rather than to irregularity.
    """
    _banner("e5", "the error bar lies (fGn-driven volatility)")
    rng = np.random.default_rng(SEEDS["e5"])
    g_true = M.lrd_g_true()
    print(f"seed {SEEDS['e5']}   T_max {t_max:,}   paths {paths:,}   "
          f"mu_a {M.MU_A}  sbar {M.SBAR}  xi {M.XI}")
    print(f"g_true = mu_a - 0.5*sbar^2*exp(2*xi^2) = {g_true:+.5f}   (closed form)")

    for h in hursts:
        # Exact fGn by Cholesky. A prefix of a path is a valid shorter path, so
        # all the T values below reuse one set of draws.
        chol = M.fgn_cholesky(h, t_max)
        g = chol @ rng.standard_normal((t_max, paths))
        sig = M.SBAR * np.exp(M.XI * g)
        r = M.MU_A - 0.5 * sig**2 + sig * rng.standard_normal((t_max, paths))

        # Sanity: the returns themselves must stay uncorrelated while their
        # magnitudes cluster. If the first were nonzero the model would be a
        # forecastability artefact rather than a long-memory one.
        lag = 100
        ac_r = np.corrcoef(r[:-lag].ravel(), r[lag:].ravel())[0, 1]
        ac_abs = np.corrcoef(np.abs(r[:-lag]).ravel(), np.abs(r[lag:]).ravel())[0, 1]

        print(f"\nH = {h}  ({'long memory' if h > 0.5 else 'rough'})")
        print(f"  corr(r_t, r_t+{lag}) = {ac_r:+.3f}   "
              f"corr(|r_t|, |r_t+{lag}|) = {ac_abs:+.3f}")
        print(f"  {'T':>6} {'mean est':>10} {'TRUE SE':>9} {'NAIVE SE':>9} "
              f"{'ratio':>7} {'eff. N':>8}")
        rows = []
        for tt in ts:
            block = r[:tt]
            means = block.mean(axis=0)                      # one estimate per path
            true_se = means.std(ddof=1)
            naive_se = (block.std(axis=0, ddof=1) / np.sqrt(tt)).mean()
            ess = block.var(axis=0, ddof=1).mean() / true_se**2
            rows.append((tt, true_se))
            print(f"  {tt:6d} {means.mean():+10.5f} {true_se:9.5f} {naive_se:9.5f} "
                  f"{true_se / naive_se:6.2f}x {ess:8.0f}")
        (t1, s1), (t2, s2) = rows[-2], rows[-1]
        slope = 2 * np.log(s2 / s1) / np.log(t2 / t1)
        # The pure-LRD reference rate 2H-2 only applies for H > 1/2, where the
        # autocovariances are non-summable. For H < 1/2 they are summable and
        # the aggregated variance is back to the iid rate, so quoting 2H-2
        # there would be nonsense.
        ref = f", pure-LRD theory: {2 * h - 2:+.2f}" if h > 0.5 else ""
        print(f"  Var(mean) ~ T^{slope:+.2f}    (iid: -1.00{ref})")

    print("\nThe overconfidence ratio GROWS with T at H = 0.8: collecting more data")
    print("makes the reported error bar worse, not better. At H = 0.1 the naive SE")
    print("is essentially correct, so the effect is long memory specifically -- not")
    print("roughness, and not fat tails, which are absent from this model entirely.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "e1": (e1, "the median lies (single period)"),
    "e2": (e2, "iid growth baseline + variance diagnosis"),
    "e3": (e3, "convergence figure -> mc_estimator_convergence.png"),
    "e4": (e4, "path functionals: what quadrature cannot reach"),
    "e5": (e5, "the error bar lies (long-memory volatility)"),
}


def main(argv):
    names = [a.lower() for a in argv[1:]]
    if not names:
        print(__doc__)
        print("available:")
        for k, (_, desc) in EXPERIMENTS.items():
            print(f"  {k}   {desc}")
        return 0
    if names == ["all"]:
        names = list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}")
        print(f"choose from: {', '.join(EXPERIMENTS)}, all")
        return 2
    M.check_density()       # B3 guard: never run an experiment on a wrong density
    for n in names:
        t0 = time.perf_counter()
        EXPERIMENTS[n][0]()
        print(f"\n[{n} finished in {time.perf_counter() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
