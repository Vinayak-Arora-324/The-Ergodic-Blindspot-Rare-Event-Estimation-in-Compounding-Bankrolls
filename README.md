# The Ergodic Blindspot — Rare-Event Estimation in Compounding Bankrolls

A **methods** project. The object of study is the estimator; the tail-hedge
strategy is the test case.

> Simulate the growth of a finite bankroll running a tail-hedge rule under
> fat-tailed dynamics; show which quantities are rare-event-dominated; and show
> how much the answer moves with the estimator and the model. The deliverable is
> a map of where the answer *can't* be known, not a verdict that tail hedging
> works.

## Layout

| file | contents |
|---|---|
| `model.py` | the one canonical model — density, sampler, hedge, quadrature truth, the three estimators, the fGn driver |
| `experiments.py` | E1–E5 behind a CLI, and the figure for each |
| `test_regressions.py` | T1–T7 (verified numbers) and B1–B5 (committed bugs) |
| `ReBuild-Spec.md` | the spec these were built from; the recorded numbers are the acceptance tests |
| `.old-files/` | the exploratory scripts these replaced, kept for provenance. Several contain bugs B1–B5 and support the retracted claims in spec §5 — read them as history, not as reference |

```bash
pip install -r requirements.txt
python model.py                    # smoke test: density check + headline truths
python experiments.py all          # ~15s; each experiment writes one .png
python experiments.py all --no-plots   # the numbers only
python test_regressions.py         # ~3s
```

## Figures

One per experiment, written to the repository root. They are results, not
illustrations: three of the four findings below are statements about the *shape*
of a distribution or about path space, and a table of numbers is the wrong
instrument for both.

| figure | what it shows |
|---|---|
| `e1_median_lies.png` | the sampling distribution at an affordable budget (52% of runs report exactly zero), then mean and median against budget |
| `e2_variance_diagnosis.png` | where crashes carry the variance (63% unhedged vs 2% hedged), the CV's bad trade, and each method against the 1× line |
| `mc_estimator_convergence.png` | E3: three sampling distributions at one *N*, and three `N^-1/2` slopes |
| `e4_path_functionals.png` | the path fan, the mean-vs-median split, terminal wealth, and the drawdown tail |
| `e5_error_bar_lies.png` | true vs reported SE against *T*, and the overconfidence factor that grows with *T* |

Every E1–E4 figure carries the fair-pricing caveat in its footer, because a
figure travels without its caption.

## Findings

Three ways a simulation lies, and one structural result.

**1. The median lies** (E1). For a rare payoff, crude MC is exactly unbiased in
expectation while the *typical* run reports near zero. Not bias — the extreme
right-skew of the estimator's sampling distribution. The median estimate is
exactly zero whenever the budget satisfies `N < ln2/p`.

**2. The hard object is the edge, not the level** (E2). Hedging *destroys* the
rare-event structure of the hedged quantity: in a crash the put payoff cancels
the index loss, so hedged log-growth is mild and bounded. Crashes carry ~2% of
its variance against ~63% for the unhedged return. Rare-event machinery is
needed for the unhedged baseline and for the hedged-minus-unhedged *difference*,
not for the hedged portfolio.

**3. The error bar lies** (E5). Under long-memory volatility (`H = 0.8`),
iid-formula confidence intervals are overconfident by 2.0× at `T = 256` and
3.5× at `T = 4096` — the overconfidence *grows* with sample size. At `H = 0.1`
the naive SE is fine, so the effect is long memory specifically, not roughness.

**4. The estimand choice dominates the estimator choice** (E4). Under iid, the
ergodic growth rate `g` collapses to a 1-D integral (LLN), so quadrature beats
every Monte Carlo method and the rare-event machinery of E2/E3 is unnecessary.
But path functionals — realized-path win rate, max drawdown, ruin — do not
collapse, are unreachable by quadrature, and are what a finite bankroll actually
cares about.

## Headline numbers

Truth by quadrature: `g_unhedged = +0.005000` (exact by construction),
`g_hedged = +0.005852`, `edge = +0.000852`, `p_itm = 6.28e-4`.

Variance reduction (E2/E3): the control variate **alone makes things worse**
(0.6×) and importance sampling **alone also fails** (0.8×); composed, they give
~190×. Neither half works alone. All three estimators nonetheless converge at
`N^-1/2` — variance reduction moves the intercept of the convergence line, never
the rate.

Over 120 months (E4), the same configuration gives both of these at once:

| | |
|---|---|
| mean difference across paths | `+0.1021` — matches `T·edge = +0.1023` |
| median difference | `−0.0600` — the hedge loses on the typical path |
| P(hedged beats unhedged on the realized path) | `7.13%` |
| max drawdown, 99.9th pct, unhedged → hedged | `99.96%` → `75%` wipeout |

The mean is carried entirely by the ~7% of paths where a put lands. The drawdown
column is the actual case for tail hedging and it is invisible in `g` by
construction.

## Two caveats that travel with every number

1. **Fair pricing.** `markup = 1.0` throughout: the put costs its actuarial
   value. There is no variance risk premium at that setting, so every "the hedge
   wins" result is conditional on a market that does not exist. `markup > 1` is
   the first knob to sweep.
2. **Quadrature is accurate here, not exact anywhere.** Adaptive `quad` over
   `[-40, 2]` resolves the width-0.05 bulk only because the jump density spreads
   mass across the interval and guides the subdivision. For a pure-normal
   integrand the same call can silently return near zero. Verified by T1 and B4,
   not assumed.

Claims retracted during the exploratory phase are listed in `ReBuild-Spec.md` §5
and must not be restated. `test_regressions.py` encodes the bugs that produced
them.
