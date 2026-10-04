"""Run the project's eight simulation experiments from one command line.

    python -m blindspot.experiments             # list them
    python -m blindspot.experiments e1 e2       # run a subset
    python -m blindspot.experiments all         # run all and write figures
    python -m blindspot.experiments all --no-plots

E1-E7 each explain one way a simulation or its error bar can mislead us. E8
combines them in one crash-insurance example.

  E1  rare payoffs          -- most small runs miss the event, even though the
                               average across many runs is correct.
  E2  sources of noise      -- shows where simulation noise comes from and why
                               one noise-reduction method hurts when used alone.
  E3  convergence           -- compares how quickly the methods become precise.
  E4  bankroll paths        -- compares averages with the outcomes experienced
                               along individual multi-month paths.
  E5  misleading error bars -- shows how volatility memory breaks an error
                               formula that assumes independent months.
  E6  drawdowns             -- compares extreme losses across Hurst settings.
  E7  interval accuracy     -- checks whether two methods report realistic
                               uncertainty for those drawdowns.
  E8  full comparison       -- combines rare crashes, finite volatility memory,
                               option pricing, and bankroll outcomes.

Every random-number generator has a fixed, printed seed, and every estimate is
reported with its Monte Carlo standard error. Charts use the same cautions as
the printed results.

Important assumption for E1-E4: the put is sold at its average expected payoff
(`markup = 1.0`). Real options normally include an extra charge for risk, so
these experiments show a deliberately favorable case for the hedge.
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
    print("\n[note] markup = 1.0: the put costs only its average expected payoff.")
    print("       Real options normally include an extra charge for risk, so these")
    print("       results are a deliberately favorable case for the hedge.")


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
    "markup = 1.0: the put costs only its average expected payoff. Real options "
    "normally include an extra charge for risk, so this favors the hedge."
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
    """Show why a correct average can hide a useless typical estimate.

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
    _banner("e1", "most small runs miss a rare payoff")
    rng = np.random.default_rng(SEEDS["e1"])
    truth = M.truth()
    p = truth["p_itm"]
    e_payoff = truth["price"]           # E[payoff] under fair pricing == price

    # Choose the "affordable" N so that a run expects fewer than one crash --
    # the regime where the affordability question actually bites. N*p ~ 0.63
    # here, matching the regime of the original t-model run (which had 0.6).
    n_afford = 1000

    print(f"seed {SEEDS['e1']}   trials {trials:,}")
    print(f"chance the put pays this month ... {p:.3e}")
    print(f"samples in one small run ......... {n_afford:,}  "
          f"(~{n_afford * p:.2f} crashes per run)")
    print(f"benchmark average payoff ......... {e_payoff:.6f}")

    ests = np.array(
        [M.put_payoff(M.sample_returns(rng, n_afford)).mean() for _ in range(trials)]
    )
    mc_se = ests.std(ddof=1) / np.sqrt(trials)

    print(f"\nmean of {trials:,} estimates ......... {ests.mean():.6f} "
          f"+/- {mc_se:.6f} (error in the mean)")
    print(f"difference / error in mean ....... {(ests.mean() - e_payoff) / mc_se:+.2f}")
    print(f"typical (median) estimate ........ {np.median(ests):.6f}")
    print(f"spread of estimates (sd) ......... {ests.std(ddof=1):.6f}")
    print(f"runs that saw zero crashes ....... {(ests == 0).mean():.2%}")
    print(f"runs below half the benchmark .... {(ests < 0.5 * e_payoff).mean():.2%}")

    # The median estimate is exactly zero iff a majority of runs see no crash,
    # i.e. iff (1-p)^N > 1/2, i.e. N < log(2)/p. Stating the threshold keeps
    # the claim precise: "the median lies" is a statement about the budget
    # relative to 1/p, not a defect of the estimator.
    print(f"the typical result is zero below N = ln(2)/p = {np.log(2) / p:,.0f}")

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
    print("\nAs the sample budget grows, the typical result moves toward the benchmark.")
    print("The method was always correct on average; small runs simply miss the event.")

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
    ax_a.set_xlabel("estimated average put payoff from one run")
    ax_a.set_ylabel("share of runs (%, log scale)")
    _grid(ax_a)
    _title(ax_a, f"A. The average is {ests.mean():.4f}, but most runs report zero")

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
    ax_b.set_ylabel("estimated average put payoff")
    ax_b.set_ylim(-0.1 * e_payoff, 1.9 * e_payoff)
    ax_b.legend(loc="upper left")
    _grid(ax_b)
    _title(ax_b, "B. The average across runs is right at every budget")

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
    _title(ax_c, "C. Small budgets miss rare crashes")

    _suptitle(fig, "E1  Most small runs miss a rare payoff",
              "Ordinary Monte Carlo is correct on average, but an affordable run "
              "will often see no crash and report zero.")
    _note(fig, "Target: the average one-month put payoff; benchmark calculated by "
               "numerical integration. Correctness describes the average across many "
               "runs, while a user normally gets only one run.")
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
    _banner("e2", "where the simulation noise comes from")
    rng = np.random.default_rng(SEEDS["e2"])
    t = M.truth()
    edge = t["edge"]

    print(f"seed {SEEDS['e2']}   N {n:,}   trials {trials:,}")
    print("\n-- benchmark values --")
    print(f"average growth without hedge ..... {t['g_unhedged']:+.6f}")
    print(f"average growth with hedge ........ {t['g_hedged']:+.6f}")
    print(f"difference caused by hedge ....... {edge:+.6f}")
    print(f"chance the put pays this month ... {t['p_itm']:.3e}")
    print(f"chance any jump occurs ........... {t['p_jump']:.3e}")
    print("Only jumps large enough to cross the strike make the put pay.")

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

    print(f"\n-- sources of noise ({n_diag:,} draws) --")
    print(f"spread of hedged growth .......... {lg.std():.4f}")
    print(f"spread of the hedge's effect ..... {d.std():.4f}   "
          f"({'larger' if d.std() > lg.std() else 'smaller'} than before)")
    print(f"crashes' share of hedged variance  {crash_share(lg):.1%}")
    print(f"crashes' share of unhedged variance {crash_share(x):.1%}")
    print(f"average hedged growth in a crash . {lg[crash].mean():+.3f}")
    print(f"average unhedged return in a crash {x[crash].mean():+.3f}")
    print("The put offsets most of the index loss during a crash.")

    # --- how many samples to resolve the sign, analytically ---
    print("\n-- samples needed to tell whether the hedge helps --")
    print(f"ordinary Monte Carlo ............. {int((2 * lg.std() / edge) ** 2):,}")
    print(f"crash control alone .............. {int((2 * d.std() / edge) ** 2):,}")

    # --- the three estimators, empirically ---
    print(f"\n-- methods at N = {n:,}, across {trials:,} runs --")
    print(f"{'method':30s} {'mean':>10} {'difference/SE':>13} {'spread':>10} "
          f"{'noise improvement':>17}")
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
    print(f"\nReweighting factors stay between {lo:.4f} and {hi:.4f}, so a few")
    print("samples cannot produce unbounded noise.")

    # The deflationary reading, which must be reported with the speedup.
    t0 = time.perf_counter()
    M.truth.cache_clear()
    M.truth()
    ms = 1000 * (time.perf_counter() - t0)
    print(f"\n[note] Numerical integration gets the same answer in {ms:.1f} ms without")
    print("       simulation. For independent monthly returns, average growth is only")
    print("       a one-dimensional integral. E4 introduces results that depend on a")
    print("       whole multi-month path and cannot be calculated this way.")
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
    _title(ax_a, "A. The put removes most crash noise from hedged growth")

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
    _title(ax_b, "B. Removing ordinary noise leaves an uneven crash-only result")

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
    _title(ax_c, "C. The two noise-reduction steps work best together")

    _suptitle(fig, "E2  Most of the noise comes from the unhedged return",
              "The crash control and crash oversampling hurt when used alone, but "
              "reduce noise sharply when combined.")
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
    _banner("e3", "how precision improves with more samples")
    rng = np.random.default_rng(SEEDS["e3"])
    t = M.truth()
    g_h, g_u, edge = t["g_hedged"], t["g_unhedged"], t["edge"]
    thresh = edge / 2.0     # sd at which g_h and g_u are 2 sigma apart

    print(f"seed {SEEDS['e3']}")
    print(f"unhedged growth {g_u:+.6f}   hedged growth {g_h:+.6f}")
    print(f"hedge difference {edge:+.6f}   target spread {thresh:.2e}")

    # --- Panel A: sampling distributions at one affordable N ---
    samples = {name: M.run_trials(fn, n_panel_a, trials_a, rng)
               for name, fn, _ in M.ESTIMATORS}
    print(f"\nresults at N={n_panel_a:,} (benchmark {g_h:+.6f}):")
    for name, _, _ in M.ESTIMATORS:
        e = samples[name]
        mc_se = e.std(ddof=1) / np.sqrt(trials_a)
        print(f"  {name:30s} mean {e.mean():+.6f}  "
              f"difference/SE {(e.mean() - g_h) / mc_se:+.1f}  "
              f"median {np.median(e):+.6f}")

    # --- Panel B: sd vs N ---
    ns = np.array([64, 256, 1024, 4096, 16384, 65536])
    sds = {name: np.array([M.run_trials(fn, int(n), trials_b, rng).std(ddof=1)
                           for n in ns])
           for name, fn, _ in M.ESTIMATORS}

    print(f"\n{'method':30s} {'rate':>7} {'N needed to see effect':>24}")
    n_star = {}
    for name, _, _ in M.ESTIMATORS:
        v = sds[name]
        slope = np.polyfit(np.log(ns), np.log(v), 1)[0]
        # Fit sd = A * N^slope, then solve sd = thresh for N.
        a = np.exp(np.log(v).mean() - slope * np.log(ns).mean())
        n_star[name] = (thresh / a) ** (1.0 / slope)
        print(f"{name:30s} {slope:7.3f} {n_star[name]:19,.0f}")
    print("\nEvery method's spread shrinks at the usual rate: doubling precision needs")
    print("about four times as many samples. The methods differ in how much noise they")
    print("start with, not in how quickly additional samples help.")

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
    ax_a.set_xlabel("estimated long-run growth rate $g$")
    ax_a.set_ylabel("density (log scale)")
    ax_a.legend(loc="upper left", fontsize=7.8)
    _grid(ax_a)
    _title(ax_a, f"A. Results from one sample budget, N = {n_panel_a:,}")

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
    ax_b.set_ylabel("spread of estimates (log scale)")
    ax_b.legend(loc="lower left")
    _grid(ax_b, axis="both", which="both")
    _title(ax_b, "B. Every method improves at the same rate")

    _suptitle(fig, "E3  Less noise, but the same rate of improvement",
              "All three methods estimate the same number. Better methods start "
              "with less noise, but extra samples help each one at the same rate.")
    _note(fig, f"{FAIR_PRICING_FOOTER}\nNumerical integration needs no simulated "
               f"samples, so it does not appear on this sample-size chart.")
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
    _banner("e4", "what happens along individual bankroll paths")
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

    print("\n-- average result across all paths --")
    print(f"benchmark over {horizon // 12} years ................. {horizon * edge:+.4f} log")
    print(f"mean difference across paths ............ {diff.mean():+.4f} "
          f"+/- {se(diff):.4f}")
    print("The simulated average agrees with the numerical benchmark.")
    print(f"typical (median) difference .............. {np.median(diff):+.4f}")
    print(f"With no jump, the hedge loses exactly {horizon * np.log(1 - M.C):+.4f} "
          "from premiums.")

    win = (diff > 0).mean()
    print("\n-- results experienced on individual paths --")
    print(f"chance the hedge beats no hedge ......... {win:.2%} "
          f"+/- {np.sqrt(win * (1 - win) / paths):.2%}")
    print(f"chance the put pays at least once ............. {(n_itm >= 1).mean():.3%}"
          f"   (theory {1 - (1 - t['p_itm']) ** horizon:.3%})")
    print(f"chance any jump occurs ........................ {(n_jump >= 1).mean():.3%}"
          f"   (theory {1 - (1 - M.LAM) ** horizon:.3%})")
    print("A jump does not always push the market below the put's strike. Only a jump")
    print("large enough to make the put pay can help the hedge.")

    print("\n-- results grouped by the number of jumps --")
    print(f"{'jumps':>8} {'paths':>9} {'average effect':>15} {'hedge wins':>16}")
    for label, mask in (("0", n_jump == 0), ("1", n_jump == 1), (">=2", n_jump >= 2)):
        d = diff[mask]
        print(f"{label:>8} {mask.mean():9.2%} {d.mean():+13.4f} {(d > 0).mean():15.1%}")

    print(f"\n-- terminal log wealth after {horizon} months --")
    print(f"{'':10} {'mean':>9} {'median':>9} {'5%':>9} {'95%':>9} {'sd':>9}")
    for label, v in (("unhedged", x_u), ("hedged", x_h)):
        q5, q95 = np.percentile(v, [5, 95])
        print(f"{label:10} {v.mean():+9.4f} {np.median(v):+9.4f} {q5:+9.4f} "
              f"{q95:+9.4f} {v.std(ddof=1):9.4f}")
    print(f"\naverage hedge effect / normal path spread  "
          f"{horizon * edge / x_h.std(ddof=1):.3f}")
    print("The average benefit is small compared with the normal variation between")
    print("individual bankroll paths.")

    print("\n-- worst peak-to-trough loss on each path (in log units) --")
    print("This depends on the order of monthly returns, so a one-month integral cannot calculate it.")
    qs = [50, 95, 99, 99.9]
    print(f"{'':10}" + "".join(f"{f'{q}%':>10}" for q in qs))
    for label, v in (("unhedged", dd_u), ("hedged", dd_h)):
        print(f"{label:10}" + "".join(f"{x:10.3f}" for x in np.percentile(v, qs)))
    worst_u, worst_h = np.percentile(dd_u, 99.9), np.percentile(dd_h, 99.9)
    print("\nTypical drawdowns are nearly identical. Among the worst 0.1% of paths,")
    print(f"the hedge changes a {1 - np.exp(-worst_u):.2%} wipeout into a "
          f"{1 - np.exp(-worst_h):.2%} loss.")
    print(f"Both results are true: the average effect is positive, yet the hedge loses")
    print(f"on {1 - win:.0%} of paths because its benefit comes from the {win:.0%} where")
    print("a put pays.")
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
              label="average-growth benchmark")
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
    _title(ax_b, f"B. The average improves, but {1 - win:.0%} of paths lose")

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
    _title(ax_c, "C. Typical outcomes are alike; the difference is in bad crashes")

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
    _title(ax_d, "D. The hedge helps most in the worst drawdowns")

    _suptitle(fig, "E4  A good average can hide what most investors experience",
              "One-month calculations can find average growth, but they cannot find "
              "win rates or worst losses along a multi-month path.")
    _note(fig, f"{FAIR_PRICING_FOOTER}\n{paths:,} paths over {horizon} months, "
               f"common random numbers: both bankrolls face the identical realized "
               f"market, month by month. Average hedge effect / path spread: "
               f"{horizon * edge / x_h.std(ddof=1):.3f}.\nMaximum drawdown depends "
               f"on the order of returns and cannot be found from one-month averages.")
    _save(fig, outfile, top=0.94)


# ---------------------------------------------------------------------------
# E5 -- the error bar lies
# ---------------------------------------------------------------------------
def e5(t_max=4096, paths=1200, hursts=(0.8, 0.1), ts=(256, 1024, 4096),
       outfile=FIGURES / "memory_error_bars.png"):
    """Long-memory volatility makes the reported confidence interval dishonest.

    The first experiment whose estimand is genuinely path-space rather than a
    disguised one-period expectation. Simple returns have a constant conditional
    mean, but log returns inherit memory through variance drag sigma_t^2/2.

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
    _banner("e5", "volatility memory makes ordinary error bars too small")
    rng = np.random.default_rng(SEEDS["e5"])
    g_true = M.lrd_g_true()
    print(f"seed {SEEDS['e5']}   T_max {t_max:,}   paths {paths:,}   "
          f"mu_a {M.MU_A}  sbar {M.SBAR}  xi {M.XI}")
    print(f"benchmark average growth = {g_true:+.5f}   (calculated exactly)")

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

        print(f"\nH = {h}  ({'persistent volatility' if h > 0.5 else 'quickly reversing volatility'})")
        # Short lags reveal the log-return dependence that a lag-100 check misses.
        for lag in (1, 100):
            if lag >= t_max:
                continue
            ac_r = np.corrcoef(r[:-lag].ravel(), r[lag:].ravel())[0, 1]
            ac_abs = np.corrcoef(np.abs(r[:-lag]).ravel(), np.abs(r[lag:]).ravel())[0, 1]
            theory = M.lrd_log_return_correlation(h, lag)
            print(f"  log-return correlation, lag {lag:3d} ..... {ac_r:+.3f} "
                  f"(theory {theory:+.3f})")
            print(f"  size correlation, lag {lag:3d} ........... {ac_abs:+.3f}")
        print(f"  {'months':>6} {'mean':>10} {'real SE':>9} {'usual SE':>9} "
              f"{'gap':>7} {'effective N':>11}")
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
        print(f"  variance of the mean changes as T^{slope:+.2f} "
              f"(independent months: -1.00{ref})")

    print("\nWith persistent volatility (H = 0.8), the usual error bar becomes more")
    print("overconfident as the record grows. With quickly reversing volatility")
    print("(H = 0.1), it remains about right. The problem comes from persistence;")
    print("this experiment does not include rare jumps or heavy tails.")

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
    ax_a.set_ylabel("uncertainty in estimated growth (log scale)")
    ax_a.legend(loc="lower left", fontsize=7.8)
    _grid(ax_a, axis="both", which="both")
    _title(ax_a, "A. Usual and actual uncertainty move apart")

    ax_b.axhline(1.0, color=INK, ls="--", lw=1.5)
    ax_b.set_xscale("log")      # before any axes-fraction text: log rescales x
    ax_b.set_xlabel("length of the backtest, T months")
    ax_b.set_ylabel("actual uncertainty / reported uncertainty")
    ax_b.set_ylim(0, None)
    ax_b.legend(loc="upper left")
    _grid(ax_b)
    ax_b.text(0.015, 0.92, "1x: the error bar is accurate", transform=ax_b.get_yaxis_transform(),
              va="top", fontsize=8.2, color=INK)
    _title(ax_b, "B. With persistent volatility, the gap grows with more data")

    _suptitle(fig, "E5  Volatility memory makes ordinary error bars too small",
              "The usual formula assumes independent months. Persistent volatility "
              "breaks that assumption.")
    _note(fig, f"{paths:,} independent paths per H. Log returns inherit volatility "
               f"memory through variance drag; normal shocks are independent. No rare "
               f"jumps.\nAn upward line in B means the reported error bar falls "
               f"farther behind the actual uncertainty as more months are added.")
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
    _banner("E6", "how volatility memory changes extreme drawdowns")
    print("Every Hurst setting has the same final variance. Changing H therefore")
    print("changes the route each path takes, not its overall scale. Daily steps.\n")
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
    print(f"\nThe drawdown exceeded by 1 path in 1,000 ranges from {lo:.1f}% at "
          f"H = {max(hursts)} to "
          f"{hi:.1f}% at H = {min(hursts)}:")
    print("Quickly reversing paths make deeper peak-to-trough moves even though their")
    print("final variance is the same. E7 checks whether two estimation methods put")
    print("realistic error bars around these results.")

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
    _title(ax, "Same final variance, very different worst drawdowns")

    _suptitle(fig, "E6  Volatility memory changes the path, not just the endpoint",
              "All settings have the same final variance. The chart compares the "
              "drawdown exceeded by one path in 1,000.")
    _note(fig, f"{paths:,} independent paths per H on the daily grid "
               f"({M.DD_SUB} steps per month). Every H setting has the same final "
               f"variance, so the comparison isolates path shape.\nThese H values "
               f"are scenarios, not fitted values. The S&P evidence gives H ~ 0.5 "
               f"for return direction and 0.84-0.95 for volatility.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E7 -- the honesty panel (the headline)
# ---------------------------------------------------------------------------
def _interval_diagnostics(estimates, halfwidths, benchmark):
    """Measure coverage of each reported interval; keep RMSE as a diagnostic."""
    est = np.asarray(estimates, dtype=float)
    rep = np.asarray(halfwidths, dtype=float)
    if est.ndim != 1 or est.size < 2 or rep.shape != est.shape:
        raise ValueError("at least two estimates and matching half-widths are required")
    lower, upper = est - rep, est + rep
    covered = (lower <= benchmark) & (benchmark <= upper)
    coverage = float(covered.mean())
    # Wilson 95% bounds show sampling uncertainty even at 0% or 100% coverage.
    n, z = est.size, 1.96
    denom = 1.0 + z**2 / n
    center = (coverage + z**2 / (2 * n)) / denom
    radius = z * np.sqrt(coverage * (1 - coverage) / n + z**2 / (4 * n**2)) / denom
    rmse = float(np.sqrt(np.mean((est - benchmark) ** 2)))
    return {
        "estimates": est, "halfwidths": rep,
        "interval_lower": lower, "interval_upper": upper, "covered": covered,
        "coverage": coverage,
        # At 0%/100%, roundoff can otherwise put an endpoint just inside the
        # point estimate and turn a plotted error length negative.
        "coverage_lo": float(min(coverage, max(0.0, center - radius))),
        "coverage_hi": float(max(coverage, min(1.0, center + radius))),
        "bias": float(est.mean() - benchmark), "sd": float(est.std(ddof=1)),
        "rmse": rmse, "reported": float(rep.mean()),
        "rmse_ratio": float(rep.mean() / (1.96 * rmse)) if rmse else float("nan"),
    }


def e7_point(h, m=8_000, reps=24, ref_paths=200_000, seed=707):
    """One H of E7: truth, then both designs, each over `reps` full runs.

    Seeded per-H rather than off one stream for the whole sweep, so a single
    point can be re-run or added without moving any other point -- which is
    what makes the robustness sweep in `e7`'s docstring a comparison rather
    than a reshuffle.
    """
    if reps < 2:
        raise ValueError("reps must be at least two to measure interval uncertainty")
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
        row[kind] = _interval_diagnostics(est, rep, truth)
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

    Each interval is estimate +/- 1.96 times its iid-bootstrap standard error.
    We count the intervals containing the daily-grid Monte Carlo benchmark
    and give Wilson bounds for this coverage rate across independent runs.
    The benchmark itself has simulation error, which these bounds exclude.

    Mean reported half-width / (1.96 x RMSE) remains a separate scale
    diagnostic. It does not establish 95% coverage for biased or skewed errors.
    The returned per-run estimates and interval bounds allow further checks.

    Cost note: this is the expensive experiment in the file (~4 min), because
    the window design needs a fresh multi-million-step record per rep.
    """
    _banner("E7", "do the reported error bars match the actual errors?")
    one_in = round(100.0 / (100.0 - M.DD_LEVEL))
    print(f"Target: the drawdown exceeded by 1 path in {one_in:,}, over a "
          f"{M.DD_HORIZON}-month horizon.")
    print(f"Both methods get {m:,} observations and are repeated {reps} times. The "
          f"benchmark uses {ref_paths:,} paths with daily steps.\n")
    print("Intervals: estimate +/- 1.96 x bootstrap SE; nominal coverage 95%.")
    print("Coverage bounds measure uncertainty across runs, excluding benchmark error.")
    print(f"  {'H':>5} {'method':>10} {'bias':>9} {'RMSE':>9} {'half-width':>10} "
          f"{'coverage':>9} {'95% coverage bounds':>21} {'width/1.96RMSE':>15}")

    res = {"h": [], "replicate": [], "window": [], "bias": {}, "truth": [],
           "coverage": {"replicate": [], "window": []}}
    for h in hursts:
        row = e7_point(h, m=m, reps=reps, ref_paths=ref_paths, seed=seed)
        for kind in ("replicate", "window"):
            res[kind].append(row[kind]["rmse_ratio"])
            res["coverage"][kind].append(row[kind]["coverage"])
        res["h"].append(h)
        res["truth"].append(row["truth"])
        res["bias"][h] = row
        print(f"  H={h:.1f} benchmark={row['truth']:.4f}")
        for kind in ("replicate", "window"):
            r = row[kind]
            bounds = f"[{r['coverage_lo']:.1%}, {r['coverage_hi']:.1%}]"
            print(f"  {h:5.1f} {kind:>10} {r['bias']:+9.4f} {r['rmse']:9.4f} "
                  f"{r['reported']:10.4f} {r['coverage']:9.1%} {bounds:>21} "
                  f"{r['rmse_ratio']:15.2f}")

    print("\nThe independent-path method misses losses that happen between its monthly")
    print("observations. More paths reduce random noise but cannot fix that coarse grid.")
    print("The window method uses daily observations, but neighboring windows come from")
    print("one dependent history. Its usual bootstrap acts as if they were independent.")
    print("Measured coverage checks those intervals directly. The width/RMSE ratio")
    print("compares scales only; its crossings do not establish interval calibration.")

    if PLOTS:
        _e7_figure(res, m, reps, outfile)
    return res


def _e7_figure(res, m, reps, outfile):
    """Measured coverage and the separate width/RMSE scale diagnostic."""
    plt = _mpl()
    fig, (ax_cov, ax_ratio) = plt.subplots(1, 2, figsize=(13.6, 5.6))
    h = np.asarray(res["h"])
    for kind, col, label, offset in (
        ("replicate", BLUE, "independent paths (monthly)", -0.008),
        ("window", ORANGE, "single-path windows (daily)", 0.008),
    ):
        rows = [res["bias"][hh][kind] for hh in h]
        coverage = 100 * np.array([r["coverage"] for r in rows])
        lo = 100 * np.array([r["coverage_lo"] for r in rows])
        hi = 100 * np.array([r["coverage_hi"] for r in rows])
        ax_cov.errorbar(h + offset, coverage, yerr=[coverage - lo, hi - coverage],
                        fmt="o-", color=col, ms=5, capsize=3, label=label)
        ax_ratio.plot(h, res[kind], "o-", color=col, ms=5, label=label)

    ax_cov.axhline(95, color=INK, ls="--", lw=1.5, label="nominal 95% coverage")
    ax_cov.set_ylim(-3, 108)
    ax_cov.set_ylabel("intervals containing the benchmark (%)")
    ax_cov.legend(loc="center left", bbox_to_anchor=(0.01, 0.45), fontsize=8)
    _title(ax_cov, "A. Measured coverage, with 95% Wilson bounds")

    ax_ratio.axhline(1.0, color=INK, ls="--", lw=1.5, label="equal scales")
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("mean half-width / (1.96 × RMSE), log scale")
    ax_ratio.legend(loc="lower right", fontsize=8)
    _title(ax_ratio, "B. Width versus RMSE: a scale diagnostic")
    for ax in (ax_cov, ax_ratio):
        ax.set_xlim(h[0] - 0.035, h[-1] + 0.035)
        ax.set_xlabel("Hurst exponent $H$ of the bankroll")
        _grid(ax, axis="both")

    _suptitle(fig, "E7  Checking drawdown intervals against repeated runs",
              "Coverage counts the reported intervals that contain the benchmark. "
              "The width/RMSE ratio measures a separate property.")
    _note(fig, f"m = {m:,} observations per estimate for BOTH designs, {reps} "
               f"independent runs per point. Intervals use estimate ± 1.96 × "
               f"iid-bootstrap SE; windows are non-overlapping.\nCoverage is against "
               f"a simulated daily-grid benchmark; its uncertainty is excluded. "
               f"A width/RMSE ratio of 1 does not imply 95% coverage.")
    _save(fig, outfile, top=0.90)


# ---------------------------------------------------------------------------
# E8 -- the original question, unified
# ---------------------------------------------------------------------------
def e8(trials=250, budget=16_384, horizon=256, strategy_paths=40_000,
       strategy_horizon=120, outfile=FIGURES / "unified_comparison.png"):
    """Price crash insurance, then test it on multi-month bankroll paths."""
    _banner("E8", "full crash-insurance comparison")
    truth = M.tail_put_truth_and_cv()
    print(f"benchmark price of one {M.RS_K:.0%}-strike monthly put  {truth['price']:.8f}")
    print(f"monthly crash chance in bankroll paths (P) .... {M.RS_LAM_P:.3%}")
    print(f"monthly crash chance used for pricing (Q) ..... {M.RS_LAM_Q:.3%}")
    print(f"crash chance used while oversampling .......... {M.RS_LAM_IS:.1%}")
    print(f"volatility memory resets after ................ {M.MEM_CUTOFF} months")
    print(f"simulation budget per pricing run ............. {budget:,} months")
    print(f"length of each pricing path ................... {horizon:,} months\n")

    regimes = (
        (0.1, "rough H=.1"),
        (0.5, "memoryless H=.5"),
        (0.8, "persistent H=.8"),
        ((0.1, 0.8), "alternating .1/.8"),
    )
    methods = ("crude", "cv", "is", "cv_is")
    labels = {"crude": "ordinary Monte Carlo", "cv": "+ crash control",
              "is": "+ crash oversampling", "cv_is": "+ both methods"}
    pricing, strategy = {}, {}

    print("OPTION PRICING -- same benchmark and same simulation budget")
    print(f"  {'regime':>18} {'method':>23} {'mean':>11} {'difference/SE':>13} "
          f"{'spread':>11} {'noise improvement':>19}")
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

    print("\nVOLATILITY MEMORY -- local behavior before and after each reset")
    for h in M.REGIME_H:
        local_n = np.array([4, 8, 16, 32])
        long_n = M.MEM_CUTOFF * np.array([8, 16, 32, 64])
        local_v = np.array([M.regime_sum_variance(h, int(n)) for n in local_n])
        long_v = np.array([M.regime_sum_variance(h, int(n)) for n in long_n])
        h_local = np.polyfit(np.log(local_n), np.log(local_v), 1)[0] / 2
        h_long = np.polyfit(np.log(long_n), np.log(long_v), 1)[0] / 2
        print(f"  chosen local H={h:.1f}: measured before reset {h_local:.3f}, "
              f"measured over many regimes {h_long:.3f}")

    print("\nBANKROLL RESULTS -- puts priced with Q, outcomes generated with P")
    print(f"  {'regime':>18} {'avg effect/month':>16} {'hedge wins':>15} "
          f"{'99% drawdown, no hedge':>23} {'with hedge':>15}")
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

    print("\nThe improved methods estimate the put price with less noise; they do not")
    print("change whether the hedge makes money. The pricing model Q assumes more")
    print("crashes than occur in the bankroll model P, so protection is expensive.")
    print("The hedge loses money on average but reduces the most severe drawdowns.")

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
    ax1.set_xlabel("time span, months")
    ax1.set_ylabel("variance of accumulated volatility signal")
    _title(ax1, "A. Local memory fades after repeated resets")
    ax1.legend(frameon=False, fontsize=9)
    _grid(ax1, axis="both", which="both")

    regime_labels = [label for _, label in regimes]
    x = np.arange(len(regime_labels))
    names = ("crude", "cv", "is", "cv_is")
    labs = ("ordinary MC", "+ crash control", "+ oversampling", "+ both")
    colors = (MUTED, ORANGE, BLUE, VIOLET)
    width = 0.19
    for j, (name, lab, color) in enumerate(zip(names, labs, colors)):
        y = [result["pricing"][r][name]["vr"] for r in regime_labels]
        ax2.bar(x + (j - 1.5) * width, y, width, color=color, label=lab)
    ax2.axhline(1.0, color=INK, ls="--", lw=1.2)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(("H=.1", "H=.5", "H=.8", ".1/.8"))
    ax2.set_ylabel("noise improvement over ordinary Monte Carlo")
    ax2.set_xlabel("local volatility regime")
    _title(ax2, "B. Four methods estimate the same put price")
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
    ax3.set_ylabel("99th-percentile worst drawdown (% of wealth)")
    ax3.set_xlabel("local volatility regime")
    _title(ax3, "C. The hedge reduces severe drawdowns but usually loses")
    ax3.legend(frameon=False, fontsize=9)
    _grid(ax3)

    _suptitle(fig, "E8  Pricing crash insurance and testing the result",
              "Four equal-budget pricing methods, finite volatility memory, and "
              "multi-year bankroll outcomes.")
    _note(fig, f"Pricing truth {result['truth']['price']:.8f}; {budget:,} month-observations per run, "
               f"{horizon}-month paths. Volatility memory resets after "
               f"{M.MEM_CUTOFF} months, so short-run persistence need not last "
               f"forever. Pricing crash chance (Q): {M.RS_LAM_Q:.3%}; bankroll "
               f"crash chance (P): {M.RS_LAM_P:.3%}.")
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
