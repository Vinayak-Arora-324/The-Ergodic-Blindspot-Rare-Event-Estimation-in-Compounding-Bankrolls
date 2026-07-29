"""
experiments.py -- the five experiments, behind one command line.

    python experiments.py            # list them
    python experiments.py e1 e2      # run a subset
    python experiments.py all        # run everything (each writes a .png)
    python experiments.py all --no-plots    # numbers only, no figures

Each experiment is one way a simulation lies, plus one structural result:

  E1  the median lies       -- crude MC is unbiased in expectation while the
                               TYPICAL run reports near zero.
  E2  where the variance is -- the baseline, the variance decomposition, and
                               why the control variate makes things WORSE alone.
  E3  the convergence figure-- every method converges at N^-1/2; variance
                               reduction moves the intercept, never the rate.
                               The figure is the deliverable here.
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
    in the exploratory phase came from exactly that. A figure is prose: an axis
    that flatters is the same error as a sentence that flatters;
  * every figure carries the fair-pricing caveat in its footer, for the same
    reason every table of numbers carries it in its trailer.

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

#: Set False by `--no-plots`; the printed numbers are unaffected either way.
PLOTS = True


def _banner(tag, title):
    print(f"\n{RULE}\n{tag.upper()}  --  {title}\n{RULE}")


def _fair_pricing_note():
    print("\n[caveat] markup = 1.0: the put is actuarially fair. Every 'hedge wins'")
    print("         number above is conditional on a market with no variance risk")
    print("         premium, i.e. one that does not exist.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Each experiment writes one PNG. The figures are results, not decoration: E1's
# claim is about the SHAPE of a sampling distribution, E4's estimands live in
# path space, and E5's is a divergence between two curves. None of those three
# survives being flattened into a row of numbers, which is exactly how they were
# missed during the exploratory phase.
#
# One palette across the whole set, checked rather than eyeballed -- every hue
# clears 3:1 against white, and every pair that shares a frame stays separable
# under protanopia, deuteranopia and tritanopia:
#
#   blue / orange / violet    the three estimators of record (model.ESTIMATORS)
#   blue / orange             unhedged vs hedged (E4), and H = 0.8 vs H = 0.1 (E5)
#   greys                     truth lines, reference levels, annotation
#
# No single figure uses both blue/orange roles, so a colour never means two
# things in one frame. Series are legended or directly labelled as well, so
# identity never rides on colour alone.

BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
UNHEDGED, HEDGED = BLUE, ORANGE
INK, INK_2, MUTED, GRID = "#1a1a19", "#52514e", "#8a8984", "#d9d8d2"

#: Colours for E5's Hurst exponents, in the order they are passed.
HURST_COLOURS = (BLUE, ORANGE, VIOLET)

_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 9.0,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.2,
    "legend.labelspacing": 0.45,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "font.size": 9.5,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
}


def _mpl():
    """Matplotlib configured for headless PNG output, in the house style.

    Imported lazily and per-call: `--no-plots` must not pay for the import, and
    `test_regressions.py` imports this module without ever drawing anything.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RC)
    return plt


def _grid(ax, axis="y", which="major"):
    """Recessive grid, behind the data."""
    ax.grid(True, axis=axis, which=which, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _title(ax, head, sub, pad=17):
    """Panel title: one bold claim, then what the reader is looking at.

    The sub-line sits between the title and the axes, so the title needs pad
    enough to clear it -- 17pt for one 8.4pt line plus its leading, ~28 for
    two. A two-line sub is worth the space when the alternative is parking a
    text block on top of the data.
    """
    ax.set_title(head, fontsize=10.5, fontweight="semibold", pad=pad)
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8.4,
            color=INK_2, va="bottom", ha="left")


def _note(fig, text):
    """Footer note, set in the same grey as the axis furniture."""
    fig.text(0.006, 0.006, text, fontsize=7.4, color=MUTED, ha="left", va="bottom")


#: Travels on every E1-E4 figure, for the same reason it travels on every table.
FAIR_PRICING_FOOTER = (
    "markup = 1.0: the put is priced actuarially fair, so every 'the hedge wins' "
    "reading here is conditional on a market with no variance risk premium."
)


def _save(fig, outfile, top=0.92):
    """Write the PNG and say so, in the same voice as the rest of the output."""
    plt = _mpl()
    fig.tight_layout(rect=(0, 0.028, 1, top))
    fig.savefig(outfile, dpi=170)
    plt.close(fig)
    print(f"\nsaved {outfile}")


def _suptitle(fig, text, sub):
    """Figure title plus one line of standfirst, both flush left.

    The standfirst is offset from the title in POINTS converted to figure
    fraction, not in a fixed fraction: these figures range from 4.9 to 10.4
    inches tall and a fixed fraction collides on the short ones.
    """
    gap = 22.0 / (fig.get_figheight() * 72.0)
    fig.text(0.006, 0.988, text, ha="left", va="top", fontsize=13.5,
             fontweight="semibold", color=INK)
    fig.text(0.006, 0.988 - gap, sub, ha="left", va="top", fontsize=9.0, color=INK_2)


# ---------------------------------------------------------------------------
# E1 -- the median lies
# ---------------------------------------------------------------------------
def e1(trials=3000, outfile="e1_median_lies.png"):
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
    # affordability, not about the estimator being broken. The grid straddles
    # ln2/p (~1.1k) so the median's departure from zero is visible on both
    # sides of it, and trials are cut at large N where each run costs more --
    # the sweep is a picture of the sampling distribution, not a precision
    # estimate of any one point on it.
    ns = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000])
    sweep = {k: [] for k in ("mean", "se", "median", "p_zero", "p_half")}
    print(f"\n{'N':>9} {'trials':>7} {'mean':>10} {'median':>10} {'P(saw 0)':>10} "
          f"{'P(<truth/2)':>12}")
    for n in ns:
        tr = 600 if n <= 10_000 else 300
        e = np.array(
            [M.put_payoff(M.sample_returns(rng, int(n))).mean() for _ in range(tr)]
        )
        sweep["mean"].append(e.mean())
        sweep["se"].append(e.std(ddof=1) / np.sqrt(tr))
        sweep["median"].append(np.median(e))
        sweep["p_zero"].append((e == 0).mean())
        sweep["p_half"].append((e < 0.5 * e_payoff).mean())
        print(f"{n:9,} {tr:7,} {e.mean():10.6f} {np.median(e):10.6f} "
              f"{(e == 0).mean():9.1%} {(e < 0.5 * e_payoff).mean():11.1%}")
    sweep = {k: np.array(v) for k, v in sweep.items()}
    print("\nThe median converges to the truth from BELOW as N grows. Nothing about")
    print("the estimator changes; only whether the budget can see the event.")

    if PLOTS:
        _e1_figure(ests, e_payoff, p, n_afford, ns, sweep, trials, outfile)


def _e1_figure(ests, e_payoff, p, n_afford, ns, sweep, trials, outfile):
    """Panel A is the finding; B and C are the same finding as a function of budget.

    Panel A plots the estimator's sampling distribution on a log count axis.
    That is not a cosmetic choice and it is the only honest one available: on a
    linear axis the spike at zero is ~53% of the mass and the right tail that
    carries the entire mean is a handful of runs one pixel tall, so a linear
    axis would draw the estimator as if it were a point mass at zero -- the
    exact misreading the experiment exists to correct.
    """
    plt = _mpl()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.4, 4.9))
    n_thresh = np.log(2) / p

    # --- A. the sampling distribution at the affordable budget ---
    share = np.full(ests.size, 100.0 / ests.size)
    ax_a.hist(ests, bins=np.linspace(0, ests.max(), 90), weights=share,
              color=BLUE, edgecolor="white", lw=0.4)
    ax_a.set_yscale("log")
    ax_a.set_ylim(top=ax_a.get_ylim()[1] * 6)   # headroom for the annotation layer
    ax_a.axvline(e_payoff, color=INK, ls="--", lw=1.6)
    ax_a.annotate(f"truth {e_payoff:.4f}", (e_payoff, 0.60), xycoords=("data", "axes fraction"),
                  textcoords="offset points", xytext=(7, 0), fontsize=8.4, color=INK)
    ax_a.annotate(
        f"{(ests == 0).mean():.0%} of runs report exactly zero.\n"
        f"Median = 0.  Mean = {ests.mean():.4f}: unbiased.",
        (0, (ests == 0).mean() * 100), xytext=(0.20, 0.93), textcoords="axes fraction",
        fontsize=8.4, color=INK_2, va="top",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                        connectionstyle="angle3,angleA=0,angleB=75"))
    ax_a.annotate("and the entire mean is carried out here,\n"
                  "by the runs that happened to see a crash",
                  (ests.max() * 0.72, 0.05), xytext=(0.97, 0.42),
                  textcoords="axes fraction", ha="right", va="top",
                  fontsize=8.4, color=INK_2,
                  arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                  connectionstyle="angle3,angleA=0,angleB=80"))
    ax_a.set_xlabel("estimate of E[put payoff] from one run")
    ax_a.set_ylabel("share of runs (%, log scale)")
    _grid(ax_a)
    _title(ax_a, f"A. One affordable run, {trials:,} times over",
           f"N = {n_afford:,} draws per run, ~{n_afford * p:.2f} crashes expected")

    # --- B. mean and median against budget ---
    ax_b.axhline(e_payoff, color=INK, ls="--", lw=1.6)
    ax_b.text(ns[0], e_payoff * 1.06, "truth", fontsize=8.2, color=INK)
    ax_b.fill_between(ns, sweep["mean"] - 2 * sweep["se"], sweep["mean"] + 2 * sweep["se"],
                      color=BLUE, alpha=0.16, lw=0)
    ax_b.plot(ns, sweep["mean"], "o-", color=BLUE, ms=5, label="mean of runs  (±2 MC-SE)")
    ax_b.plot(ns, sweep["median"], "o-", color=ORANGE, ms=5, label="median run")
    ax_b.axvline(n_thresh, color=MUTED, ls=":", lw=1.4)
    ax_b.text(0.315, 0.34, f"N = ln2/p = {n_thresh:,.0f}\nmedian is exactly 0\nto the left of here",
              transform=ax_b.transAxes, ha="right", va="center",
              multialignment="right", fontsize=8.0, color=MUTED)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("samples per run, N")
    ax_b.set_ylabel("estimate of E[put payoff]")
    ax_b.set_ylim(-0.1 * e_payoff, 1.9 * e_payoff)
    ax_b.legend(loc="upper left")
    _grid(ax_b)
    _title(ax_b, "B. The mean is right at every budget",
           "the median climbs to it from below, and only from below")

    # --- C. the same thing as a failure rate ---
    ax_c.plot(ns, 100 * sweep["p_zero"], "o-", color=BLUE, ms=5,
              label="runs that saw no crash at all")
    ax_c.plot(ns, 100 * sweep["p_half"], "o-", color=ORANGE, ms=5,
              label="runs reporting less than half the truth")
    ax_c.axvline(n_thresh, color=MUTED, ls=":", lw=1.4)
    ax_c.axhline(50, color=INK, ls="--", lw=1.2)
    ax_c.text(ns[-1], 52, "half of runs", fontsize=8.0, color=INK, ha="right")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("samples per run, N")
    ax_c.set_ylabel("share of runs (%)")
    ax_c.set_ylim(-3, 103)
    ax_c.legend(loc="upper right")
    _grid(ax_c)
    _title(ax_c, "C. It is a budget problem, not a bias",
           "nothing about the estimator changes along this axis")

    _suptitle(fig, "E1  The median lies",
              "Crude MC for a rare payoff is exactly unbiased in expectation, and "
              "the run you can afford still reports zero.")
    _note(fig, "Estimand: E[put payoff] over one month, truth by quadrature. "
               "Unbiasedness is a property of the average over runs; you get one run.")
    _save(fig, outfile, top=0.87)


# ---------------------------------------------------------------------------
# E2 -- iid growth baseline and variance diagnosis
# ---------------------------------------------------------------------------
def e2(n=5000, trials=2000, n_diag=2_000_000, outfile="e2_variance_diagnosis.png"):
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
    sds["IS alone, no CV"] = e_is.std(ddof=1)
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

    if PLOTS:
        _e2_figure(lg, d, x, crash_share, sds, n, trials, ms, outfile)


def _e2_figure(lg, d, x, crash_share, sds, n, trials, ms, outfile):
    """Where the variance is (A, B), and what removing it is worth (C).

    Panel C is a bar chart on a log axis with the reference at 1x drawn, not
    implied. Two of the four bars sit BELOW that line and the figure has to say
    so plainly -- these are the two results a reader is most likely to assume
    are bugs, so the axis has to make "worse than crude" as legible as "190x
    better". A bar chart of variance-reduction factors with no 1x line is the
    standard way that gets hidden.
    """
    plt = _mpl()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.4, 4.9))

    # --- A. where the variance lives, hedged vs unhedged ---
    shares = [100 * crash_share(x), 100 * crash_share(lg)]
    labels = ["unhedged\nreturn $r$", "hedged\nlog growth"]
    bars = ax_a.bar(labels, shares, width=0.5, color=[MUTED, MUTED],
                    edgecolor="white", lw=1.5)
    bars[0].set_color(INK_2)
    for b, s in zip(bars, shares):
        ax_a.text(b.get_x() + b.get_width() / 2, s + 2.2, f"{s:.0f}%",
                  ha="center", fontsize=11.5, color=INK, fontweight="semibold")
    ax_a.set_ylabel("share of variance carried by crashes (%)")
    ax_a.set_ylim(0, 100)
    ax_a.annotate("hedging DESTROYS the rare-event\nstructure it was meant to capture:\n"
                  "in a crash the payoff cancels the\nindex loss, so what is left is mild",
                  (1, shares[1]), textcoords="offset points", xytext=(-8, 34),
                  ha="center", fontsize=8.2, color=INK_2)
    _grid(ax_a)
    # Panel B needs a two-line sub, so all three carry one: titles that sit at
    # three different heights across one figure read as three figures.
    _title(ax_a, "A. The rare event is not in the hedged quantity",
           f"law of total variance over {len(x):,} draws,\ncrash = $r < \\log(K/S_0)$",
           pad=28)

    # --- B. the two per-sample distributions the estimators actually average ---
    ax_b.hist(lg, bins=140, histtype="step", lw=2.0, color=BLUE, density=True,
              label=f"$\\log$ growth       sd {lg.std():.4f}")
    ax_b.hist(d, bins=140, histtype="step", lw=2.0, color=ORANGE, density=True,
              label=f"$D = \\log$ growth $- \\,r$   sd {d.std():.4f}")
    ax_b.set_yscale("log")
    ax_b.set_xlim(-0.35, 0.25)
    ax_b.set_xlabel("value of one sampled draw")
    ax_b.set_ylabel("density (log scale)")
    # Both curves peak in the middle of the frame, so the legend goes to the
    # floor of the panel -- the one region with no data in it. An opaque legend
    # box parked over the peaks would have been the alternative, and hiding the
    # data to make room for its own key is not a trade a figure gets to make.
    ax_b.legend(loc="lower left")
    _grid(ax_b)
    _title(ax_b, "B. The control variate trades bulk noise for skew",
           "$D$ is a spike at $\\log(1-c)$ plus a pure crash tail: narrower in\n"
           "the bulk, but skewed enough that its sd is LARGER than the level's",
           pad=28)

    # --- C. what each method is worth, against the 1x line ---
    order = [(name, col) for name, _, col in M.ESTIMATORS]
    order.append(("IS alone, no CV", MUTED))
    ratios = [(sds["crude MC"] / sds[name]) ** 2 for name, _ in order]
    names = [name.replace("+ ", "+\n") for name, _ in order]
    bars = ax_c.bar(names, ratios, width=0.55,
                    color=[c for _, c in order], edgecolor="white", lw=1.5)
    bars[-1].set_hatch("//")
    for b, r in zip(bars, ratios):
        # The two sub-1x bars sit within a hair of the reference line, so their
        # labels get a white ground rather than being nudged somewhere they no
        # longer point at their own bar.
        ax_c.text(b.get_x() + b.get_width() / 2, r * 1.14, f"{r:.1f}x", ha="center",
                  fontsize=10, color=INK, fontweight="semibold",
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
    ax_c.axhline(1.0, color=INK, ls="--", lw=1.5)
    ax_c.text(-0.45, 1.62, "1x: no better than crude MC", fontsize=8.2, color=INK,
              ha="left")
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.3, max(ratios) * 4)
    ax_c.set_ylabel("variance reduction vs crude MC (log scale)")
    ax_c.text(0.02, 0.46, "each half alone LOSES;\nonly composed do they win",
              transform=ax_c.transAxes, va="top", fontsize=8.4, color=INK_2)
    _grid(ax_c)
    _title(ax_c, "C. Neither half works alone",
           f"N = {n:,} per run, {trials:,} runs;\nCV reshapes the estimand IS was built for",
           pad=28)

    _suptitle(fig, "E2  The variance is in the baseline, not the hedge",
              "Diagnose where a rare event carries the variance before reaching "
              "for rare-event machinery.")
    _note(fig, f"{FAIR_PRICING_FOOTER}  Deflation: quadrature returns this same "
               f"answer in {ms:.0f} ms with no samples at all -- see E4 for where "
               f"that stops being true.")
    _save(fig, outfile, top=0.88)


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

    if PLOTS:
        _e3_figure(samples, ns, sds, n_star, thresh, g_h, g_u, n_panel_a, outfile)
    _fair_pricing_note()


def _e3_figure(samples, ns, sds, n_star, thresh, g_h, g_u, n_panel_a, outfile):
    """The two panels described in `e3`.

    Panel A's y-axis is log for the same reason E1's is: the cv_is spike is
    ~200x taller than the crude distribution, and on a linear axis it erases
    the two curves the panel exists to compare.
    """
    plt = _mpl()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # --- A. three sampling distributions, one budget ---
    ax_a.axvspan(g_u - 4e-4, g_u + 4e-4, color=GRID, alpha=0.5, zorder=0)
    for name, _, col in M.ESTIMATORS:
        e = samples[name]
        ax_a.hist(e, bins=70, histtype="step", lw=2.0, color=col, density=True,
                  label=f"{name}  (sd {e.std(ddof=1):.1e})")
    ax_a.axvline(g_h, color=INK, lw=1.6, label=f"truth $g_h$ = {g_h:.5f}")
    ax_a.axvline(g_u, color=INK, ls=":", lw=1.6,
                 label=f"$g_u$ = {g_u:.5f} (no hedge)")
    ax_a.set_yscale("log")      # linear y would let the cv_is spike erase the rest
    ax_a.set_xlim(g_h - 0.0045, g_h + 0.0045)
    ax_a.set_xlabel("estimate of geometric growth rate $g$")
    ax_a.set_ylabel("density (log scale)")
    ax_a.legend(loc="upper left", fontsize=7.8)
    _grid(ax_a)
    _title(ax_a, f"A. Sampling distribution at an affordable N = {n_panel_a:,}",
           "the gap between the two black lines is the entire decision")

    # --- B. sd against N, all three slopes ---
    for name, _, col in M.ESTIMATORS:
        ax_b.plot(ns, sds[name], "o-", color=col, ms=5, label=name)
    ax_b.axhline(thresh, color=INK, ls="--", lw=1.5)
    ax_b.text(ns[0] * 1.06, thresh * 1.18, "sd needed to call the sign at $2\\sigma$",
              fontsize=8.2, color=INK, va="bottom")
    # Stagger the labels vertically: crude and +cv cross the threshold at
    # similar N and their annotations would otherwise overlap.
    for i, (name, _, col) in enumerate(M.ESTIMATORS):
        n = n_star[name]
        if ns[0] <= n <= ns[-1] * 40:
            ax_b.plot([n], [thresh], "v", color=col, ms=9, zorder=5)
            ax_b.annotate(f"N$\\approx${n:,.0f}", (n, thresh),
                          textcoords="offset points", xytext=(0, -18 - 13 * (i % 2)),
                          ha="center", fontsize=8.2, color=col)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("samples per run, N")
    ax_b.set_ylabel("sd of the estimator (log scale)")
    ax_b.legend(loc="lower left")
    _grid(ax_b, axis="both", which="both")
    _title(ax_b, "B. Every slope is $N^{-1/2}$",
           "variance reduction moves the intercept, never the rate")
    ax_b.annotate("quadrature needs no $N$ at all, so the method that\n"
                  "actually wins here cannot be drawn on these axes",
                  (0.97, 0.93), xycoords="axes fraction", ha="right", va="top",
                  fontsize=8.4, color=INK_2)

    _suptitle(fig, "E3  Variance reduction moves the intercept, not the rate",
              "Three unbiased estimators of the same number, and the log-log plot "
              "that cannot see its own blind spot.")
    _note(fig, FAIR_PRICING_FOOTER)
    _save(fig, outfile, top=0.88)


# ---------------------------------------------------------------------------
# E4 -- path functionals (the most important experiment)
# ---------------------------------------------------------------------------
def e4(horizon=120, paths=400_000, n_track=5_000, outfile="e4_path_functionals.png"):
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
    O(paths x horizon). The one exception is `n_track` paths whose full history
    is kept for the fan chart, because a picture of path space is the one thing
    the running-state trick cannot give back; 5k paths is ample for a 5-95%
    band and costs ~5 MB, while every NUMBER below still uses all `paths`.
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

    n_track = min(n_track, paths)
    traj_u = np.zeros((horizon + 1, n_track))   # kept for the fan chart only
    traj_h = np.zeros((horizon + 1, n_track))

    for month in range(horizon):
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
        traj_u[month + 1] = x_u[:n_track]
        traj_h[month + 1] = x_h[:n_track]

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

    if PLOTS:
        _e4_figure(traj_u, traj_h, x_u, x_h, diff, dd_u, dd_h, edge, win,
                   horizon, paths, outfile)


def _e4_figure(traj_u, traj_h, x_u, x_h, diff, dd_u, dd_h, edge, win,
               horizon, paths, outfile):
    """Four panels, and the second and fourth are the ones that matter.

    A is the conventional Monte Carlo picture -- median path plus a 5-95% band
    -- and it is included precisely because it shows almost nothing: the two
    bankrolls are visually indistinguishable at the scale of their own path
    noise. That is the honest starting point, and B, C, D are what has to be
    plotted instead to see anything at all.

    D is a survival curve (log y) rather than a histogram because the entire
    claim lives beyond the 99th percentile, where a histogram has no resolution
    and no reader can count bars. The x-axis is stated as a wipeout percentage
    rather than in log points because -6.9 log is not a quantity anyone has
    intuitions about, and 99.9% of wealth gone is.
    """
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    months = np.arange(traj_u.shape[0])

    # --- A. the fan: both bankrolls through path space ---
    for traj, col, label in ((traj_u, UNHEDGED, "unhedged"), (traj_h, HEDGED, "hedged")):
        q5, q50, q95 = np.percentile(traj, [5, 50, 95], axis=1)
        ax_a.fill_between(months, q5, q95, color=col, alpha=0.13, lw=0)
        ax_a.plot(months, q5, color=col, lw=1.0, ls="--")
        ax_a.plot(months, q95, color=col, lw=1.0, ls="--")
        ax_a.plot(months, q50, color=col, label=f"{label}  (median, 5% and 95%)")
    ax_a.plot(months, months * (M.MU + edge), color=INK, ls="--", lw=1.4,
              label="what $g_h$ promises: $T \\times g_h$")
    ax_a.set_xlim(0, horizon)
    ax_a.set_xlabel("months")
    ax_a.set_ylabel("cumulative log wealth")
    ax_a.legend(loc="upper left")
    ax_a.text(0.03, 0.125, "The median paths are indistinguishable and $T\\times g_h$ runs "
              "through both.\nEverything the hedge does is in the 5% floor -- and B, C, D "
              "are what\nit takes to see even that much.",
              transform=ax_a.transAxes, va="top", fontsize=8.2, color=INK_2)
    _grid(ax_a)
    _title(ax_a, "A. The two bankrolls, month by month",
           f"{traj_u.shape[1]:,} paths under common random numbers, {horizon} months")

    # --- B. the difference across paths, mean vs median ---
    med = np.median(diff)
    ax_b.hist(diff, bins=np.linspace(np.percentile(diff, 0.05), np.percentile(diff, 99.95), 160),
              color=HEDGED, alpha=0.85, lw=0)
    ax_b.set_yscale("log")
    ax_b.axvline(diff.mean(), color=INK, ls="--", lw=1.6)
    ax_b.axvline(med, color=INK, ls=":", lw=1.6)
    ax_b.set_xlim(left=med - 0.35)
    # The mean and the median differ by 0.16 on an axis nine log points wide, so
    # pointing at each line separately would put two arrowheads in one place.
    # One block, one arrow, and the numbers said out loud instead.
    ax_b.annotate(f"mean   {diff.mean():+.4f}   (dashed) $= T\\times$edge, as promised\n"
                  f"median {med:+.4f}   (dotted) -- the typical path pays the\n"
                  f"premium and gets nothing back\n\n"
                  f"Both lines sit here, {diff.mean() - med:.2f} apart. The gap between them\n"
                  f"is manufactured entirely by the tail running off to the right.",
                  (med, 0.62), xycoords=("data", "axes fraction"),
                  xytext=(0.30, 0.96), textcoords="axes fraction", va="top",
                  fontsize=8.4, color=INK,
                  arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                  connectionstyle="angle3,angleA=0,angleB=75"))
    ax_b.set_xlabel("hedged $-$ unhedged terminal log wealth, per path")
    ax_b.set_ylabel("paths (log scale)")
    _grid(ax_b)
    _title(ax_b, "B. Both of these are true at once",
           f"a positive mean edge, and a loss on {1 - win:.0%} of paths")

    # --- C. terminal wealth, and its left tail ---
    lo = min(np.percentile(x_u, 0.05), np.percentile(x_h, 0.05))
    hi = max(np.percentile(x_u, 99.95), np.percentile(x_h, 99.95))
    bins = np.linspace(lo, hi, 140)
    for v, col, label in ((x_u, UNHEDGED, "unhedged"), (x_h, HEDGED, "hedged")):
        ax_c.hist(v, bins=bins, histtype="step", lw=2.0, color=col, density=True,
                  label=f"{label}   median {np.median(v):+.3f}")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(f"terminal log wealth after {horizon} months")
    ax_c.set_ylabel("density (log scale)")
    ax_c.legend(loc="upper left")
    ax_c.text(0.94, 0.30, "identical in the bulk,\nwhere a bankroll spends\n"
              "its life. Everything the\nhedge does is in the left tail.",
              transform=ax_c.transAxes, ha="right", va="top",
              fontsize=8.2, color=INK_2)
    _grid(ax_c)
    _title(ax_c, "C. The distributions the two bankrolls actually face",
           f"signal / path noise over the horizon: {horizon * edge / x_h.std(ddof=1):.3f}")

    # --- D. the drawdown tail: the actual case for the hedge ---
    p_exceed = np.logspace(np.log10(0.5), np.log10(2e-4), 140)
    for dd, col, label in ((dd_u, UNHEDGED, "unhedged"), (dd_h, HEDGED, "hedged")):
        wipe = 100 * (1 - np.exp(-np.quantile(dd, 1 - p_exceed)))
        ax_d.plot(wipe, 100 * p_exceed, color=col, label=label)
        q999 = 100 * (1 - np.exp(-np.percentile(dd, 99.9)))
        ax_d.plot([q999], [0.1], "o", color=col, ms=8, zorder=5,
                  markeredgecolor="white", markeredgewidth=1.5)
        ax_d.annotate(f"{q999:.2f}%", (q999, 0.1), textcoords="offset points",
                      xytext=(0, -17), ha="center", fontsize=9, color=col,
                      fontweight="semibold")
    ax_d.axhline(0.1, color=INK, ls=":", lw=1.3)
    ax_d.text(2, 0.115, "1 path in 1,000", fontsize=8.2, color=INK, va="bottom")
    ax_d.set_yscale("log")
    ax_d.set_xlim(0, 100)
    ax_d.set_xlabel("maximum drawdown over the horizon (% of wealth)")
    ax_d.set_ylabel("share of paths that exceed it (%, log scale)")
    ax_d.legend(loc="lower left")
    _grid(ax_d)
    _title(ax_d, "D. What the hedge is actually for",
           "a running-extremum functional: no one-period integral can reach it")

    _suptitle(fig, "E4  The estimand choice dominates the estimator choice",
              "Under iid returns quadrature beats every estimator in E2-E3 -- and "
              "cannot reach a single quantity on this page.")
    _note(fig, f"{FAIR_PRICING_FOOTER}  {paths:,} paths, common random numbers: "
               f"both bankrolls face the identical realized market, month by month.")
    _save(fig, outfile, top=0.93)


# ---------------------------------------------------------------------------
# E5 -- the error bar lies
# ---------------------------------------------------------------------------
def e5(t_max=4096, paths=1200, hursts=(0.8, 0.1), ts=(256, 1024, 4096),
       outfile="e5_error_bar_lies.png"):
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

    # The printed table stays on `ts` -- those are the numbers of record. The
    # figure walks a denser grid of the same prefixes, which costs nothing: the
    # expensive part (one Cholesky factorization and one matmul per H) is
    # already paid, and every extra T is another partial mean over draws that
    # already exist.
    ts_fig = np.array([t for t in (32, 64, 128, 256, 512, 1024, 2048, 4096)
                       if t <= t_max])
    curves = {}

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
        true_c, naive_c = [], []
        for tt in ts_fig:
            block = r[:tt]
            true_c.append(block.mean(axis=0).std(ddof=1))
            naive_c.append((block.std(axis=0, ddof=1) / np.sqrt(tt)).mean())
        curves[h] = {"t": ts_fig, "true": np.array(true_c),
                     "naive": np.array(naive_c)}

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

    if PLOTS:
        _e5_figure(curves, paths, outfile)


def _e5_figure(curves, paths, outfile):
    """Two curves that should coincide, and the gap between them as a ratio.

    Colour carries the Hurst exponent; line style carries which standard error
    is being drawn (solid = true, dashed = what the practitioner reports). That
    split is deliberate -- the comparison the figure is making is WITHIN a
    colour, so the two SEs must not be two colours or the eye pairs the wrong
    curves.

    Both panels are log-x. Panel A is log-y as well, because on those axes an
    honest iid error bar is a straight line of slope -1/2 and the H = 0.8 true
    SE visibly is not one; a linear axis turns that structural statement into a
    pair of curves drifting apart for no visible reason.
    """
    plt = _mpl()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.6, 5.6))

    for i, (h, c) in enumerate(curves.items()):
        col = HURST_COLOURS[i % len(HURST_COLOURS)]
        kind = "long memory" if h > 0.5 else "rough"
        ax_a.plot(c["t"], c["true"], "o-", color=col, ms=5,
                  label=f"H = {h} ({kind})   TRUE SE, across paths")
        ax_a.plot(c["t"], c["naive"], "o--", color=col, ms=4, alpha=0.75,
                  label=f"H = {h}   NAIVE SE, sd/$\\sqrt{{T}}$ from one path")
        ax_b.plot(c["t"], c["true"] / c["naive"], "o-", color=col, ms=5,
                  label=f"H = {h} ({kind})")
        ax_b.annotate(f"{c['true'][-1] / c['naive'][-1]:.1f}x",
                      (c["t"][-1], c["true"][-1] / c["naive"][-1]),
                      textcoords="offset points", xytext=(9, 4), fontsize=9.5,
                      color=col, fontweight="semibold")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("length of the backtest, T months")
    ax_a.set_ylabel("standard error of the growth-rate estimate (log scale)")
    ax_a.legend(loc="lower left", fontsize=7.8)
    _grid(ax_a, axis="both", which="both")
    _title(ax_a, "A. The reported error bar and the real one",
           "they coincide at H = 0.1 and separate at H = 0.8 -- and keep separating")

    ax_b.axhline(1.0, color=INK, ls="--", lw=1.5)
    ax_b.set_xscale("log")      # before any axes-fraction text: log rescales x
    ax_b.set_xlabel("length of the backtest, T months")
    ax_b.set_ylabel("true SE / reported SE  (overconfidence factor)")
    ax_b.set_ylim(0, None)
    ax_b.legend(loc="upper left")
    _grid(ax_b)
    ax_b.text(0.015, 0.90, "1x: an honest error bar", transform=ax_b.get_yaxis_transform(),
              va="top", fontsize=8.2, color=INK)
    ax_b.text(0.04, 0.44, "the curve slopes UP: every extra month of data\n"
              "makes the reported confidence interval worse.\n"
              "Not a small-sample problem -- the opposite of one.",
              transform=ax_b.transAxes, va="top", fontsize=8.4, color=INK_2)
    _title(ax_b, "B. More data makes it worse",
           "the same overconfidence as a single factor, against an honest 1x")

    _suptitle(fig, "E5  The error bar lies, and lies harder with more data",
              "Same estimator, same model, two ways of reporting its uncertainty.")
    _note(fig, f"{paths:,} independent paths per H, exact fGn by Cholesky. Returns "
               f"stay serially uncorrelated: memory enters only through the "
               f"variance drag $\\sigma_t^2/2$. No fat tails in this model at all.")
    _save(fig, outfile, top=0.88)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "e1": (e1, "the median lies -> e1_median_lies.png"),
    "e2": (e2, "variance diagnosis -> e2_variance_diagnosis.png"),
    "e3": (e3, "convergence figure -> mc_estimator_convergence.png"),
    "e4": (e4, "path functionals -> e4_path_functionals.png"),
    "e5": (e5, "the error bar lies -> e5_error_bar_lies.png"),
}


def main(argv):
    global PLOTS
    args = [a.lower() for a in argv[1:]]
    PLOTS = "--no-plots" not in args
    names = [a for a in args if not a.startswith("--")]
    if not names:
        print(__doc__)
        print("available:")
        for k, (_, desc) in EXPERIMENTS.items():
            print(f"  {k}   {desc}")
        return 0
    bad_flags = [a for a in args if a.startswith("--") and a != "--no-plots"]
    if bad_flags:
        print(f"unknown flag(s): {', '.join(bad_flags)}   (only --no-plots)")
        return 2
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
