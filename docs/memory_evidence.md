# Why H is a real parameter, not a modelling convenience

**Figure:** `figures/market_memory.png` · **Code:** `blindspot/sp500_memory.py` · **Data:** ^GSPC daily,
1990-01-02 to 2026-07-31, 9,211 log returns (Yahoo Finance, split/dividend adjusted).

The model in `model.py` drives volatility with fractional Gaussian noise,
`log sigma_t = log SBAR + XI * G_t` with `G` fGn of Hurst exponent `H`. That
choice needs an empirical warrant, otherwise `H` is just a knob that makes the
drawdown tail move.

Measured on the S&P 500, the warrant holds, but only for the volatility:

| series | variogram | DFA | Higuchi | sign null |
|---|---|---|---|---|
| log returns (direction) | 0.491 | 0.449 | 0.467 | ≈ 0.50 ± 0.03 |
| \|log returns\| (volatility) | 0.842 | 0.951 | 0.868 | ≈ 0.50 ± 0.04 |

Direction is indistinguishable from memoryless (p = 0.92, 0.16, 0.007). Volatility
is not remotely so: z = 7.6, 11.7, 21.5. The left panel of the figure shows why no
significance test is really needed — the two fluctuation curves have visibly
different slopes over 1.4 decades of scale.

So the model is calibrated to a real stylized fact. Long memory lives exactly where
`model.py` puts it, in `sigma_t`, and not in the returns themselves.

## The methodological point, which is the actual reason this is here

The obvious null — shuffle the returns — is wrong, and wrong in the direction that
manufactures a result. Shuffling destroys volatility clustering along with
directional memory, so on any equity series it rejects almost by construction.
Under that null the returns look significantly anti-persistent (p ≤ 0.005 on all
three estimators). Under the sign-randomised null, `r* = ±|r_t|`, which preserves
the `|r|` sequence exactly and destroys only memory in direction, the effect
largely evaporates.

The nulls also move the reference point, not just the spread. Higuchi's null mean
sits at 0.560 under shuffling and 0.507 under sign randomisation: 0.06 of apparent
Hurst exponent that was pure heteroskedasticity artifact. In 5-year windows the
shuffle null wanders between 0.516 and 0.601, tracking the volatility regime and
peaking in the 2008-09 windows. A 2004-2009 estimate of H = 0.416 reads as strong
anti-persistence against a fixed 0.5 line, and as an estimator being dragged by fat
tails against its own null.

That is this project's thesis in miniature: a reported number whose error bar does
not know it is wrong. Here the estimator is a Hurst estimator rather than a Monte
Carlo one, but the failure mode is identical.

## Caveats

- **The volatility exponent is scale-dependent.** DFA on `|r|` gives H = 0.82 out
  to 50 days, 0.95 out to 250-1000 days. The table uses 250 days (~1 trading year).
  The qualitative conclusion is stable; the specific number is not.
- **H ≈ 0.95 is near the stationarity boundary.** The DFA bootstrap CI is
  [0.864, 1.060] and crosses 1.0, where the fBm picture strains. Part of this is
  plausibly slow drift in the volatility level rather than self-similar long memory.
  Read the volatility exponent as "0.84-0.95, strongly persistent" and not as a
  point estimate.
- **Nothing here survives at 5-year horizons.** Rolling 5-year windows put the null
  spread at ±0.04-0.09 (vs ±0.015 on the full sample), and essentially every window
  estimate falls inside its own null band. At that sample size these estimators
  cannot distinguish the index from a memoryless process.
- **Estimator bias is not negligible.** On synthetic fGn of known H, DFA runs ~0.02
  high at H = 0.3 and the variogram ~0.08 low at H = 0.9. Estimates near the extremes
  should not be read to three decimal places. The harness that measured this was
  retired with the rebuild; the numbers are recorded here rather than reproducible
  from the current tree, and should be re-measured before being leaned on.
