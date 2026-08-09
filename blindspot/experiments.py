"""
experiments.py -- eight experiments, behind one command line.

    python -m blindspot.experiments             # list them
    python -m blindspot.experiments e1 e2       # run a subset
    python -m blindspot.experiments all         # run all and write figures
    python -m blindspot.experiments all --no-plots

E1-E7 isolate individual failure modes; E8 recombines the pieces into the
original fat-tail, transient-memory hedge-pricing question.

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
  E6  drawdown family       -- one path estimand across rough/persistent H.
  E7  honesty panel         -- grid bias and dependence defeat different
                               estimator designs in different H regimes.
  E8  unified pricing       -- fat tails, finite memory, separate P/Q measures,
                               and crude/CV/IS/CV+IS at one compute budget.

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
from pathlib import Path

import numpy as np

from . import model as M

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT_ROOT / "figures"

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


def _title(ax, head, sub=None, pad=9):
    """Panel title: one short claim, and nothing else.

    The claim goes in the title so it does not have to be parked on top of the
    data. A sub-line is available but is deliberately rare -- if a panel needs
    two lines of explanation to be read, the panel is doing too much and the
    fix is to plot less, not to caption more. When a sub IS passed the title
    needs pad enough to clear it: ~17pt for one 8.4pt line plus its leading.
    """
    ax.set_title(head, fontsize=10.5, fontweight="semibold", pad=pad)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8.4,
                color=INK_2, va="bottom", ha="left")


def _note(fig, text):
    """Footer note, set in the same grey as the axis furniture.

    Records its own height on the figure so `_save` can reserve room for it:
    the notes run to two lines on some figures and one on others, and a fixed
    reserve either clips the long ones or floats the short ones.
    """
    fig.text(0.006, 0.006, text, fontsize=7.4, color=MUTED, ha="left", va="bottom")
    lines = text.count("\n") + 1
    fig._note_bottom = 0.010 + lines * 11.0 / (fig.get_figheight() * 72.0)


#: Travels on every E1-E4 figure, for the same reason it travels on every table.
FAIR_PRICING_FOOTER = (
    "markup = 1.0: the put is priced actuarially fair, so every 'the hedge wins' "
    "reading here is conditional on a market with no variance risk premium."
)


def _save(fig, outfile, top=0.92):
    """Write the PNG and say so, in the same voice as the rest of the output."""
    plt = _mpl()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, getattr(fig, "_note_bottom", 0.028), 1, top))
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
def e1(trials=3000, outfile=FIGURES / "rare_payoff.png"):
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
    ax_a.set_ylim(top=ax_a.get_ylim()[1] * 2.5)   # a little headroom over the spike
    ax_a.axvline(e_payoff, color=INK, ls="--", lw=1.6)
    ax_a.annotate(f"truth {e_payoff:.4f}", (e_payoff, 0.66), xycoords=("data", "axes fraction"),
                  textcoords="offset points", xytext=(7, 0), fontsize=8.4, color=INK)
    # White ground: this label sits at the spike, and the truth line runs
    # through the space the text needs.
    ax_a.annotate(f"{(ests == 0).mean():.0%} report zero", (0, 0.88),
                  xycoords=("data", "axes fraction"), textcoords="offset points",
                  xytext=(7, 0), fontsize=8.4, color=INK_2,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
    ax_a.set_xlabel("estimate of E[put payoff] from one run")
    ax_a.set_ylabel("share of runs (%, log scale)")
    _grid(ax_a)
    _title(ax_a, f"A. Median 0, mean {ests.mean():.4f}: unbiased and useless")

    # --- B. mean and median against budget ---
    ax_b.axhline(e_payoff, color=INK, ls="--", lw=1.6)
    ax_b.text(ns[0], e_payoff * 1.06, "truth", fontsize=8.2, color=INK)
    ax_b.fill_between(ns, sweep["mean"] - 2 * sweep["se"], sweep["mean"] + 2 * sweep["se"],
                      color=BLUE, alpha=0.16, lw=0)
    ax_b.plot(ns, sweep["mean"], "o-", color=BLUE, ms=5, label="mean of runs  (±2 MC-SE)")
    ax_b.plot(ns, sweep["median"], "o-", color=ORANGE, ms=5, label="median run")
    ax_b.axvline(n_thresh, color=MUTED, ls=":", lw=1.4)
    ax_b.annotate(f"N = ln2/p = {n_thresh:,.0f}", (n_thresh, 0.80),
                  xycoords=("data", "axes fraction"), textcoords="offset points",
                  xytext=(-7, 0), ha="right", fontsize=8.0, color=MUTED)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("samples per run, N")
    ax_b.set_ylabel("estimate of E[put payoff]")
    ax_b.set_ylim(-0.1 * e_payoff, 1.9 * e_payoff)
    ax_b.legend(loc="upper left")
    _grid(ax_b)
    _title(ax_b, "B. The mean is right at every budget")

    # --- C. the same thing as a failure rate ---
    ax_c.plot(ns, 100 * sweep["p_zero"], "o-", color=BLUE, ms=5,
              label="saw no crash at all")
    ax_c.plot(ns, 100 * sweep["p_half"], "o-", color=ORANGE, ms=5,
              label="reported less than half the truth")
    ax_c.axvline(n_thresh, color=MUTED, ls=":", lw=1.4)
    ax_c.axhline(50, color=INK, ls="--", lw=1.2)
    ax_c.text(ns[-1], 52, "half of runs", fontsize=8.0, color=INK, ha="right")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("samples per run, N")
    ax_c.set_ylabel("share of runs (%)")
    ax_c.set_ylim(-3, 103)
    ax_c.legend(loc="upper right")
    _grid(ax_c)
    _title(ax_c, "C. A budget problem, not a bias")

    _suptitle(fig, "E1  The median lies",
              "Crude MC for a rare payoff is unbiased in expectation, and the run "
              "you can afford still reports zero.")
    _note(fig, "Estimand: E[put payoff] over one month, truth by quadrature. "
               "Unbiasedness is a property of the average over runs; you get one run.")
    _save(fig, outfile, top=0.88)


# ---------------------------------------------------------------------------
# E2 -- iid growth baseline and variance diagnosis
# ---------------------------------------------------------------------------
def e2(n=5000, trials=2000, n_diag=2_000_000,
       outfile=FIGURES / "variance_methods.png"):
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
    _grid(ax_a)
    _title(ax_a, "A. Hedging destroys the rare event it was built for")

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
    _title(ax_b, "B. The control variate trades bulk noise for skew")

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
    # Above the line and hard right: every bar descends from its own top to the
    # floor, so the only clear space in the frame is over the short last bar.
    ax_c.text(len(ratios) - 0.55, 1.55, "1x: no better than crude MC", fontsize=8.2,
              color=INK, ha="right", va="bottom")
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.3, max(ratios) * 4)
    ax_c.set_ylabel("variance reduction vs crude MC (log scale)")
    _grid(ax_c)
    _title(ax_c, "C. Neither half works alone")

    _suptitle(fig, "E2  The variance is in the baseline, not the hedge",
              "Diagnose where a rare event carries the variance before reaching "
              "for rare-event machinery.")
    _note(fig, f"A: law of total variance over {len(x):,} draws, crash = "
               f"$r < \\log(K/S_0)$.  C: N = {n:,} per run, {trials:,} runs.  "
               f"{FAIR_PRICING_FOOTER}\nQuadrature returns this same answer in "
               f"{ms:.0f} ms with no samples at all -- see E4 for where that stops "
               f"being true.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E3 -- the convergence figure
# ---------------------------------------------------------------------------
def e3(n_panel_a=4096, trials_a=2000, trials_b=300,
       outfile=FIGURES / "convergence.png"):
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
    _title(ax_a, f"A. One affordable budget, N = {n_panel_a:,}")

    # --- B. sd against N, all three slopes ---
    for name, _, col in M.ESTIMATORS:
        ax_b.plot(ns, sds[name], "o-", color=col, ms=5, label=name)
    ax_b.axhline(thresh, color=INK, ls="--", lw=1.5)
    ax_b.text(ns[0] * 1.06, thresh * 1.18, "sd needed to call the sign",
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
    _title(ax_b, "B. Every slope is $N^{-1/2}$")

    _suptitle(fig, "E3  Variance reduction moves the intercept, not the rate",
              "Three unbiased estimators of the same number, and the log-log plot "
              "that cannot see its own blind spot.")
    _note(fig, f"{FAIR_PRICING_FOOTER}\nQuadrature needs no $N$ at all, so the "
               f"method that actually wins here cannot be drawn on these axes.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E4 -- path functionals (the most important experiment)
# ---------------------------------------------------------------------------
def e4(horizon=120, paths=400_000, n_track=5_000,
       outfile=FIGURES / "bankroll_paths.png"):
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
    _grid(ax_a)
    _title(ax_a, "A. The two bankrolls are indistinguishable here")

    # --- B. the difference across paths, mean vs median ---
    med = np.median(diff)
    ax_b.hist(diff, bins=np.linspace(np.percentile(diff, 0.05), np.percentile(diff, 99.95), 160),
              color=HEDGED, alpha=0.85, lw=0)
    ax_b.set_yscale("log")
    ax_b.axvline(diff.mean(), color=INK, ls="--", lw=1.6)
    ax_b.axvline(med, color=INK, ls=":", lw=1.6)
    ax_b.set_xlim(left=med - 0.35)
    # The mean and the median differ by 0.16 on an axis nine log points wide, so
    # both labels are stacked at the same place and staggered in height rather
    # than pointed at individually.
    ax_b.annotate(f"mean {diff.mean():+.3f}", (diff.mean(), 0.93),
                  xycoords=("data", "axes fraction"), textcoords="offset points",
                  xytext=(9, 0), fontsize=8.4, color=INK)
    ax_b.annotate(f"median {med:+.3f}", (med, 0.80),
                  xycoords=("data", "axes fraction"), textcoords="offset points",
                  xytext=(9, 0), fontsize=8.4, color=INK)
    ax_b.set_xlabel("hedged $-$ unhedged terminal log wealth, per path")
    ax_b.set_ylabel("paths (log scale)")
    _grid(ax_b)
    _title(ax_b, f"B. A positive mean edge, and a loss on {1 - win:.0%} of paths")

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
    _grid(ax_c)
    _title(ax_c, "C. Identical in the bulk; the hedge lives in the left tail")

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
    _title(ax_d, "D. What the hedge is actually for: the drawdown tail")

    _suptitle(fig, "E4  The estimand choice dominates the estimator choice",
              "Under iid returns quadrature beats every estimator in E2-E3 -- and "
              "cannot reach a single quantity on this page.")
    _note(fig, f"{FAIR_PRICING_FOOTER}\n{paths:,} paths over {horizon} months, "
               f"common random numbers: both bankrolls face the identical realized "
               f"market, month by month. Signal / path noise over the horizon: "
               f"{horizon * edge / x_h.std(ddof=1):.3f}.\nD is a running-extremum "
               f"functional, which no one-period integral can reach.")
    _save(fig, outfile, top=0.94)


# ---------------------------------------------------------------------------
# E5 -- the error bar lies
# ---------------------------------------------------------------------------
def e5(t_max=4096, paths=1200, hursts=(0.8, 0.1), ts=(256, 1024, 4096),
       outfile=FIGURES / "memory_error_bars.png"):
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
                  label=f"H = {h} ({kind})   true SE")
        ax_a.plot(c["t"], c["naive"], "o--", color=col, ms=4, alpha=0.75,
                  label=f"H = {h}   reported SE")
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
    _title(ax_a, "A. The reported error bar and the real one")

    ax_b.axhline(1.0, color=INK, ls="--", lw=1.5)
    ax_b.set_xscale("log")      # before any axes-fraction text: log rescales x
    ax_b.set_xlabel("length of the backtest, T months")
    ax_b.set_ylabel("true SE / reported SE  (overconfidence factor)")
    ax_b.set_ylim(0, None)
    ax_b.legend(loc="upper left")
    _grid(ax_b)
    ax_b.text(0.015, 0.92, "1x: an honest error bar", transform=ax_b.get_yaxis_transform(),
              va="top", fontsize=8.2, color=INK)
    _title(ax_b, "B. More data makes it worse")

    _suptitle(fig, "E5  The error bar lies, and lies harder with more data",
              "Same estimator, same model, two ways of reporting its uncertainty.")
    _note(fig, f"{paths:,} independent paths per H, exact fGn by Cholesky. Returns "
               f"stay serially uncorrelated: memory enters only through the "
               f"variance drag $\\sigma_t^2/2$. No fat tails in this model at all."
               f"\nB slopes UP: every extra month of data makes the reported interval "
               f"worse. Not a small-sample problem -- the opposite of one.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E6 -- the estimand: a drawdown family indexed by H
# ---------------------------------------------------------------------------
def e6(paths=200_000, hursts=(0.1, 0.3, 0.5, 0.7, 0.9),
       outfile=FIGURES / "drawdown_by_hurst.png"):
    """Max-drawdown exceedance curves for the fBm bankroll family, one per H.

    One estimand, one family parameter, read at one level.  Everything E7 says
    about error bars is about the number this figure reads off: the drawdown
    exceeded by 1 path in 1,000 over a 12-month horizon, measured on the daily
    grid.

    The exceedance curve rather than a histogram, and log y, for E4's reason:
    the whole claim lives past the 99th percentile, where a histogram has no
    resolution and no reader can count bars.
    """
    _banner("E6", "the estimand -- drawdown against the family parameter H")
    print("Log wealth is fBm with Hurst H; terminal variance is pinned across H,")
    print("so H moves the SHAPE of the path and not its scale. Daily grid.\n")
    print(f"  {'H':>5} {'median':>9} {'99th':>9} {'99.9th':>9} {'ratio':>8}")

    rng = np.random.default_rng(606)
    curves, reads = {}, {}
    for h in hursts:
        dd = M.drawdown_replicates(h, paths, rng, sub=M.DD_SUB)
        curves[h] = np.sort(M.wipeout(dd))
        q50, q99, q999 = np.percentile(curves[h], [50, 99, 99.9])
        reads[h] = q999
        print(f"  {h:5.1f} {q50:8.2f}% {q99:8.2f}% {q999:8.2f}% {q999 / q50:7.1f}x")

    lo, hi = reads[max(hursts)], reads[min(hursts)]
    print(f"\nThe 1-in-1000 drawdown runs from {lo:.1f}% at H = {max(hursts)} to "
          f"{hi:.1f}% at H = {min(hursts)}:")
    print("rough paths spend the same terminal variance on far deeper round trips.")
    print("No one-period integral reaches any of these numbers -- E4's point, now")
    print("as a family. E7 asks whether either way of estimating one is honest.")

    if PLOTS:
        _e6_figure(curves, reads, paths, outfile)
    return curves


def _e6_figure(curves, reads, paths, outfile):
    """One panel. Five curves, one read line, and nothing else.

    Colour runs light-to-dark with H so the family reads as an ordered sweep
    rather than five unrelated series; the legend is therefore redundant with
    the direct labels and is dropped.
    """
    plt = _mpl()
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 6.2))
    hs = sorted(curves)
    # Viridis truncated at both ends: the pale yellow tail of the full map does
    # not clear 3:1 on white, and the family has to stay legible as an ordered
    # ramp, so the sweep runs dark violet to mid teal-green and stops there.
    cmap = plt.get_cmap("viridis")
    level = 100 - M.DD_LEVEL                      # 0.1% of paths
    top = max(reads.values())

    for i, h in enumerate(hs):
        v = curves[h]
        col = cmap(0.06 + 0.62 * (1 - i / max(len(hs) - 1, 1)))
        # Exceedance: share of paths whose drawdown is worse than x.
        share = 100.0 * (1.0 - np.arange(v.size) / v.size)
        keep = share >= 0.6 * level
        ax.plot(v[keep], share[keep], color=col, lw=2.0,
                label=f"H = {h}      {reads[h]:4.1f}%")
        ax.plot([reads[h]], [level], "o", color=col, ms=8, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)

    ax.axhline(level, color=INK, ls=":", lw=1.3)
    ax.text(1.5, level * 1.13, "1 path in 1,000", fontsize=8.4, color=INK, va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.18 * top)
    ax.set_ylim(0.6 * level, 130)
    ax.set_xlabel(f"maximum drawdown over {M.DD_HORIZON} months (% of wealth)")
    ax.set_ylabel("share of paths that exceed it (%, log scale)")
    # Five near-parallel curves converge in the tail, so direct labels would
    # collide exactly where the reader is looking. The key carries the read-off
    # instead, and sits in the one empty quadrant.
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95),
                    title="                1 in 1,000", alignment="left")
    leg.get_title().set_fontsize(8.2)
    leg.get_title().set_color(INK_2)
    _grid(ax)
    _title(ax, "Same terminal variance, and a 1-in-1000 drawdown that is not the same")

    _suptitle(fig, "E6  One estimand, one family parameter",
              "Log wealth is fBm of Hurst H, terminal variance pinned across H. "
              "This is the number E7 tries to put an error bar on.")
    _note(fig, f"{paths:,} independent paths per H on the daily grid "
               f"({M.DD_SUB} steps per month). Pinning Var X(T) across H means the "
               f"sweep moves path shape, not scale.\nH is swept as a family "
               f"parameter here, not calibrated: memory evidence gives H ~ 0.5 for "
               f"S&P direction and 0.84-0.95 for its volatility.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E7 -- the honesty panel (the headline)
# ---------------------------------------------------------------------------
def e7_point(h, m=8_000, reps=24, ref_paths=200_000, seed=707):
    """One H of E7: truth, then both designs, each over `reps` full runs.

    Seeded per-H rather than off one stream for the whole sweep, so a single
    point can be re-run or added without moving any other point -- which is
    what makes the robustness sweep in `e7`'s docstring a comparison rather
    than a reshuffle.
    """
    rng = np.random.default_rng(seed + int(round(h * 100)))
    truth = float(np.percentile(
        M.drawdown_replicates(h, ref_paths, rng, sub=M.DD_SUB), M.DD_LEVEL))
    row = {"truth": truth}
    for kind in ("replicate", "window"):
        est, rep = [], []
        for _ in range(reps):
            s = (M.drawdown_replicates(h, m, rng, sub=1) if kind == "replicate"
                 else M.drawdown_windows(h, m, rng))
            est.append(np.percentile(s, M.DD_LEVEL))
            rep.append(M.bootstrap_halfwidth(s, rng))
        est = np.asarray(est)
        rmse = float(np.sqrt(((est - truth) ** 2).mean()))
        row[kind] = {"bias": float(est.mean() - truth), "sd": float(est.std(ddof=1)),
                     "rmse": rmse, "reported": float(np.mean(rep)),
                     "ratio": float(np.mean(rep) / (1.96 * rmse))}
    return row


def e7(hursts=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), m=8_000, reps=24,
       ref_paths=200_000, seed=707,
       outfile=FIGURES / "interval_honesty.png"):
    """Two ways to estimate E6's number, and the error bar each one reports.

    Both designs get the SAME sample size `m` and the same estimand, so the
    comparison is about structure and not about budget:

      independent replicates -- `m` fresh simulated bankrolls, on the MONTHLY
        grid, because that is the grid a simulator picks.  Unbiased sampling,
        biased grid.

      single-path windows -- one realized record at DAILY resolution, cut into
        `m` consecutive windows.  Right grid, dependent sample.

    Both report an iid bootstrap 95% half-width.  Against that we put the
    half-width that would ACTUALLY have covered: 1.96 x the rmse of the
    estimator around the truth, over `reps` independent runs of the whole
    design.  Ratio 1 is an honest error bar; below 1 is overconfidence.

    Robustness, because a crossing is the kind of result that is easy to tune
    into existence. Sweeping m over 4k / 8k / 16k moves both curves DOWN --
    at H = 0.5 the window design runs 1.07, 1.04, 0.74 -- but the crossing
    stays put between H = 0.6 and 0.7 throughout. So the location is a
    property of the family and the level is a property of the budget: more
    data does not buy honesty here, it spends it, which is E5's finding
    arriving again through a completely different door.

    Cost note: this is the expensive experiment in the file (~4 min), because
    the window design needs a fresh multi-million-step record per rep.
    """
    _banner("E7", "the honesty panel -- reported half-width over actual error")
    one_in = round(100.0 / (100.0 - M.DD_LEVEL))
    print(f"Estimand: the 1-in-{one_in:,} drawdown of E6, "
          f"{M.DD_HORIZON}-month horizon.")
    print(f"Both designs get m = {m:,} observations; {reps} reps each; "
          f"truth from {ref_paths:,} daily-grid paths.\n")
    print(f"  {'H':>5} {'truth':>8} | {'repl bias':>10} {'rmse':>8} {'reported':>9} "
          f"{'RATIO':>6} | {'win bias':>9} {'rmse':>8} {'reported':>9} {'RATIO':>6}")

    res = {"h": [], "replicate": [], "window": [], "bias": {}, "truth": []}
    for h in hursts:
        row = e7_point(h, m=m, reps=reps, ref_paths=ref_paths, seed=seed)
        for kind in ("replicate", "window"):
            res[kind].append(row[kind]["ratio"])
        res["h"].append(h)
        res["truth"].append(row["truth"])
        res["bias"][h] = row
        truth = row["truth"]
        r, w = row["replicate"], row["window"]
        print(f"  {h:5.1f} {truth:8.4f} | {r['bias']:+10.4f} {r['rmse']:8.4f} "
              f"{r['reported']:9.4f} {r['ratio']:6.2f} | {w['bias']:+9.4f} "
              f"{w['rmse']:8.4f} {w['reported']:9.4f} {w['ratio']:6.2f}")

    print("\nThe replicate design fails on the LEFT: its error bar measures sampling")
    print("noise, and its error is discretization bias, which no extra path removes.")
    print("The window design fails on the RIGHT: its grid is right and its sample is")
    print("dependent, so the bootstrap divides by an m it does not really have.")
    print("Each is honest only where the other is not, and they cross at H ~ 0.65.")
    print("Sweeping m over 4k/8k/16k moves both curves down and leaves the crossing")
    print("where it is: more data does not buy honesty here, it spends it.")

    if PLOTS:
        _e7_figure(res, m, reps, outfile)
    return res


def _e7_figure(res, m, reps, outfile):
    """One panel, two curves, one honest line at 1.

    Log y: the failures are multiplicative and run from ~0.1x to ~1x, so a
    linear axis would compress the entire left-hand failure into the bottom
    tenth of the frame and draw "8x overconfident" as indistinguishable from
    "3x overconfident".
    """
    plt = _mpl()
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 6.2))
    h = np.asarray(res["h"])
    rep, win = np.asarray(res["replicate"]), np.asarray(res["window"])

    ax.set_yscale("log")
    ax.set_xlim(h[0] - 0.035, h[-1] + 0.035)
    ax.set_ylim(0.05, 2.4)
    # Shade below the honest line rather than annotating "overconfident" twice:
    # once the reader knows which side is bad, every crossing reads itself.
    ax.axhspan(0.05, 1.0, color=GRID, alpha=0.35, lw=0)
    ax.axhline(1.0, color=INK, ls="--", lw=1.5)

    # The measured exponent for S&P direction, from docs/memory_evidence.md.
    # it is where a reader actually stands, and both designs are wrong there.
    ax.axvline(0.49, color=MUTED, ls=":", lw=1.4)
    ax.annotate("S&P direction,\nmeasured", (0.49, 0.058),
                xycoords=("data", "data"), textcoords="offset points",
                xytext=(-7, 0), ha="right", va="bottom", fontsize=8.2,
                color=MUTED, multialignment="right")

    ax.plot(h, rep, "o-", color=BLUE, ms=6)
    ax.plot(h, win, "o-", color=ORANGE, ms=6)
    ax.annotate("single-path windows\n(daily grid)", (h[0], win[0]),
                textcoords="offset points", xytext=(9, 5), va="bottom",
                fontsize=9, color=ORANGE, fontweight="semibold")
    ax.annotate("independent replicates\n(monthly grid)", (h[0], rep[0]),
                textcoords="offset points", xytext=(9, -3), va="top",
                fontsize=9, color=BLUE, fontweight="semibold")
    ax.text(h[-1] + 0.02, 1.06, "honest", fontsize=8.4, color=INK,
            ha="right", va="bottom")

    ax.set_xlabel("Hurst exponent $H$ of the bankroll")
    ax.set_ylabel("reported half-width / the one that would cover (log scale)")
    _grid(ax, axis="both")
    _title(ax, "Each design is honest only where the other is not")

    _suptitle(fig, "E7  Neither error bar knows which half of the family it is in",
              "The same estimand and the same sample size, estimated two ways. "
              "Below the line is overconfidence.")
    _note(fig, f"m = {m:,} observations per estimate for BOTH designs, {reps} "
               f"independent runs per point; reported bar is an iid bootstrap 95% "
               f"half-width, verified calibrated on iid draws (T9).\nActual = 1.96 x "
               f"rmse around the truth, so bias counts. Windows are non-overlapping, "
               f"which is the charitable case: at H = 1/2 they are exactly iid.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E8 -- the original question, unified
# ---------------------------------------------------------------------------
def e8(trials=250, budget=16_384, horizon=256, strategy_paths=40_000,
       strategy_horizon=120, outfile=FIGURES / "unified_comparison.png"):
    """Price and run one tail hedge under fat tails and transient memory."""
    _banner("E8", "fat tails + transient memory + tail-hedge pricing")
    truth = M.tail_put_truth_and_cv()
    print(f"Q price of one {M.RS_K:.0%}-strike monthly put ... {truth['price']:.8f}")
    print(f"physical jump probability ................. {M.RS_LAM_P:.3%}")
    print(f"risk-neutral jump probability ............. {M.RS_LAM_Q:.3%}")
    print(f"IS proposal jump probability .............. {M.RS_LAM_IS:.1%}")
    print(f"finite memory cutoff ...................... {M.MEM_CUTOFF} months")
    print(f"pricing budget ............................ {budget:,} month-observations/run")
    print(f"pricing horizon ........................... {horizon:,} months/path\n")

    regimes = (
        (0.1, "rough H=.1"),
        (0.5, "Brownian H=.5"),
        (0.8, "persistent H=.8"),
        ((0.1, 0.8), "alternating .1/.8"),
    )
    methods = ("crude", "cv", "is", "cv_is")
    labels = {"crude": "crude MC", "cv": "+ crash control",
              "is": "+ importance sampling", "cv_is": "+ control + IS"}
    pricing, strategy = {}, {}

    print("PRICING UNDER Q -- same truth and same compute budget")
    print(f"  {'regime':>18} {'method':>23} {'mean':>11} {'bias/SE':>9} "
          f"{'sd':>11} {'variance reduction':>19}")
    for i, (h, regime_label) in enumerate(regimes):
        rng = np.random.default_rng(8080 + i)
        draws = M.run_price_trials(rng, trials, budget, horizon, h)
        base_var = draws["crude"].var(ddof=1)
        pricing[regime_label] = {}
        for name in methods:
            x = draws[name]
            sd = x.std(ddof=1)
            z = (x.mean() - truth["price"]) / (sd / np.sqrt(trials))
            vr = base_var / x.var(ddof=1)
            pricing[regime_label][name] = {
                "mean": float(x.mean()), "sd": float(sd), "z": float(z),
                "vr": float(vr),
            }
            print(f"  {regime_label:>18} {labels[name]:>23} {x.mean():11.8f} "
                  f"{z:+9.2f} {sd:11.8f} {vr:18.2f}x")

    print("\nMEMORY CROSSOVER -- exact from the block-reset covariance")
    for h in M.REGIME_H:
        local_n = np.array([4, 8, 16, 32])
        long_n = M.MEM_CUTOFF * np.array([8, 16, 32, 64])
        local_v = np.array([M.regime_sum_variance(h, int(n)) for n in local_n])
        long_v = np.array([M.regime_sum_variance(h, int(n)) for n in long_n])
        h_local = np.polyfit(np.log(local_n), np.log(local_v), 1)[0] / 2
        h_long = np.polyfit(np.log(long_n), np.log(long_v), 1)[0] / 2
        print(f"  local H={h:.1f}: measured locally {h_local:.3f}, "
              f"measured beyond cutoff {h_long:.3f}")

    print("\nROLLING HEDGE UNDER P -- premiums priced conditionally under Q")
    print(f"  {'regime':>18} {'mean edge/mo':>14} {'P(hedge wins)':>15} "
          f"{'99% DD unhedged':>18} {'99% DD hedged':>15}")
    for i, (h, regime_label) in enumerate(regimes):
        rng = np.random.default_rng(8180 + i)
        paths = M.tail_hedge_strategy_paths(
            rng, strategy_paths, strategy_horizon, h
        )
        edge = paths["difference"].mean() / strategy_horizon
        win = (paths["difference"] > 0).mean()
        dd_u = float(np.percentile(M.wipeout(paths["drawdown_unhedged"]), 99))
        dd_h = float(np.percentile(M.wipeout(paths["drawdown_hedged"]), 99))
        strategy[regime_label] = {
            "edge": float(edge), "win": float(win),
            "dd_u": dd_u, "dd_h": dd_h,
        }
        print(f"  {regime_label:>18} {edge:+14.6f} {win:14.2%} "
              f"{dd_u:17.2f}% {dd_h:14.2f}%")

    print("\nVariance reduction makes the Q price cheaper to estimate; it does not make")
    print("the hedge profitable under P. Here Q assigns more crash mass than P, so the")
    print("rolling hedge buys a tail-risk premium: it can reduce extreme drawdown while")
    print("retaining negative mean carry.")

    result = {"truth": truth, "pricing": pricing, "strategy": strategy}
    if PLOTS:
        _e8_figure(result, regimes, budget, horizon, outfile)
    return result


def _e8_figure(result, regimes, budget, horizon, outfile):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax1, ax2, ax3 = axes

    ns = np.unique(np.logspace(0, np.log10(M.MEM_CUTOFF * 64), 90).astype(int))
    for h, color in zip(M.REGIME_H, (ORANGE, INK, BLUE)):
        var = np.array([M.regime_sum_variance(h, int(n)) for n in ns])
        ax1.loglog(ns, var, color=color, lw=2.2, label=f"local H = {h:.1f}")
    ax1.axvline(M.MEM_CUTOFF, color=MUTED, ls=":", lw=1.5)
    ax1.text(M.MEM_CUTOFF * 1.08, ax1.get_ylim()[0] * 1.7, "memory cutoff",
             color=MUTED, fontsize=9)
    ax1.set_xlabel("aggregation horizon, months")
    ax1.set_ylabel("Var(sum of volatility driver)")
    _title(ax1, "A. Local scaling, long-run H = 1/2")
    ax1.legend(frameon=False, fontsize=9)
    _grid(ax1, axis="both", which="both")

    regime_labels = [label for _, label in regimes]
    x = np.arange(len(regime_labels))
    names = ("crude", "cv", "is", "cv_is")
    labs = ("crude MC", "+ crash control", "+ IS", "+ control + IS")
    colors = (MUTED, ORANGE, BLUE, VIOLET)
    width = 0.19
    for j, (name, lab, color) in enumerate(zip(names, labs, colors)):
        y = [result["pricing"][r][name]["vr"] for r in regime_labels]
        ax2.bar(x + (j - 1.5) * width, y, width, color=color, label=lab)
    ax2.axhline(1.0, color=INK, ls="--", lw=1.2)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(("H=.1", "H=.5", "H=.8", ".1/.8"))
    ax2.set_ylabel("variance reduction vs crude MC")
    ax2.set_xlabel("local volatility regime")
    _title(ax2, "B. Same Q price, four estimators")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    _grid(ax2)

    dd_u = [result["strategy"][r]["dd_u"] for r in regime_labels]
    dd_h = [result["strategy"][r]["dd_h"] for r in regime_labels]
    ax3.bar(x - 0.18, dd_u, 0.36, color=INK, label="unhedged")
    ax3.bar(x + 0.18, dd_h, 0.36, color=ORANGE, label="rolling hedge")
    for i, r in enumerate(regime_labels):
        win = result["strategy"][r]["win"]
        edge = result["strategy"][r]["edge"]
        ax3.text(i, max(dd_u[i], dd_h[i]) + 1.5,
                 f"win {win:.0%}\nedge {edge:+.1e}/mo", ha="center", fontsize=8)
    ax3.set_ylim(0, max(dd_u) + 13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(("H=.1", "H=.5", "H=.8", ".1/.8"))
    ax3.set_ylabel("99th-percentile max drawdown (% wealth)")
    ax3.set_xlabel("local volatility regime")
    _title(ax3, "C. Q price versus P bankroll outcome")
    ax3.legend(frameon=False, fontsize=9)
    _grid(ax3)

    _suptitle(fig, "E8  The unified experiment",
              "Fat tails, transient market memory, a Q-priced rolling tail hedge, and equal-budget Monte Carlo.")
    _note(fig, f"Pricing truth {result['truth']['price']:.8f}; {budget:,} month-observations per run, "
               f"{horizon}-month paths. Memory is in volatility and resets after {M.MEM_CUTOFF} months, "
               "so local H can differ from 1/2 without asserting permanent anomalous scaling. "
               f"Q crash probability {M.RS_LAM_Q:.3%} vs P {M.RS_LAM_P:.3%}.")
    _save(fig, outfile, top=0.86)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "e1": (e1, "rare payoff -> figures/rare_payoff.png"),
    "e2": (e2, "variance methods -> figures/variance_methods.png"),
    "e3": (e3, "convergence -> figures/convergence.png"),
    "e4": (e4, "bankroll paths -> figures/bankroll_paths.png"),
    "e5": (e5, "memory error bars -> figures/memory_error_bars.png"),
    "e6": (e6, "drawdown by H -> figures/drawdown_by_hurst.png"),
    "e7": (e7, "interval honesty -> figures/interval_honesty.png  (~4 min)"),
    "e8": (e8, "unified comparison -> figures/unified_comparison.png"),
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
