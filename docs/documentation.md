# Documentation

This file holds the full description of the project. The [README](../README.md)
gives the main finding only.

- [1. The question](#1-the-question)
- [2. Main experiment: E4](#2-main-experiment-e4)
- [3. Side experiments](#3-side-experiments)
- [4. Setup and commands](#4-setup-and-commands)
- [5. Files](#5-files)
- [6. What the results do not prove](#6-what-the-results-do-not-prove)

---

## 1. The question

A bankroll compounds. Every month it follows one fixed, mechanical rule: buy
the same small amount of cheap, far out-of-the-money crash insurance and hold
it to expiry. What does that do to the outcome:

- on average across many possible histories,
- along the single history an investor actually lives through,
- in the worst drawdowns?

These answers can differ, and that difference is the "ergodic blindspot." An
average across paths can improve while most individual paths get worse.
Quantities that depend on the order of returns, such as maximum drawdown,
cannot be computed from a one-month average at all.

E4 answers the question directly and is the project's main result. The side
experiments test how far simulated estimates of these quantities, and their
error bars, can be trusted.

### A naive hedge, on purpose

The hedge is deliberately as simple as possible. It is a toy for separating
averages from individual paths and for testing simulation methods, not a
trading strategy:

- **No timing or market view.** It buys the same option every month whatever
  the market is doing.
- **No sizing rule.** The spend is a fixed fraction of wealth, not optimised
  for growth or risk.
- **No active management.** Options are held to expiry. Nothing is sold early,
  rolled on a spike, or monetised after a selloff.
- **No volatility spikes.** E1–E4 use constant volatility. E5 and E8 let
  volatility vary smoothly with memory, but it never jumps suddenly and
  option prices never spike. Real tail-hedging programmes earn much of their return by
  selling options when implied volatility jumps, which this model cannot
  represent.
- **Simple pricing.** The put costs its fair expected payoff (E1–E4) or a
  fixed multiple of a crash rate (E8). There is no implied-volatility surface
  or skew.

The results therefore describe this rule in this model only. They say nothing
about how a real, actively managed tail hedge would perform.

---

## 2. Main experiment: E4

### Model

Monthly log returns are independent. Most months are a normal move, and rare
months add a large downward jump:

```text
r = m + S Z              with probability 1 − λ
r = m + S Z − E          with probability λ,      Z ~ N(0,1), E ~ Exponential(mean β)
```

| parameter | value | meaning |
|---|---|---|
| `λ` (`LAM`) | 0.001 | crash chance per month |
| `β` (`BETA`) | 1.5 | mean crash size, in log points (a 78% fall) |
| `S` | 0.05 | volatility of ordinary months |
| `μ` (`MU`) | 0.005 | expected monthly log return; `m = μ + λβ` keeps it exact |

### Hedge (deliberately naive)

Each month, with no conditions, the rule puts a fraction `C = 0.0005` (0.05%) of wealth into a
one-month put struck 50% below the current level (`K = 50`, `S0 = 100`). The
rest stays in the index. The hedged log growth for a month with return `r` is

```text
log( (1 − C) e^r + C · payoff(r) / price )
```

The put is priced at its fair expected payoff (`markup = 1.0`), which is
0.018828 per unit of index. Because returns are independent, the average
monthly growth of both bankrolls is a one-dimensional integral. This gives an
exact benchmark:

| | average monthly log growth |
|---|---:|
| unhedged | +0.005000 |
| hedged | +0.005852 |
| difference | **+0.000852** |

### Simulation

The run uses 400,000 paths of 120 months each with seed 31. Both bankrolls use
**common random numbers**: they face the same realized market month by month,
so every difference between them comes from the hedge. Running state
(cumulative wealth, running peak, maximum drawdown) is updated one month at a
time, which keeps memory proportional to the number of paths. The full
histories of 5,000 paths are kept for the fan chart.

### Results

**Average versus individual paths**

| quantity | value |
|---|---:|
| benchmark gain over 10 years (120 × 0.000852) | +0.1023 |
| simulated mean gain | **+0.1021 ± 0.0012** |
| median gain | −0.0600 |
| share of paths where the hedge ends ahead | **7.13% ± 0.04%** |
| share of paths where a put pays at least once | 7.24% (theory 7.26%) |
| share of paths with at least one crash | 11.30% (theory 11.31%) |

When no put pays, the hedge loses exactly 120 × log(1 − C) = −0.0600. That is
why the median gain is −0.0600. A crash only helps when it is deep enough to
put the option in the money.

| crashes on the path | share of paths | mean gain | hedge ends ahead |
|---|---:|---:|---:|
| 0 | 88.70% | −0.0600 | 0.0% |
| 1 | 10.65% | +1.2951 | 61.7% |
| 2 or more | 0.65% | +2.6676 | 86.0% |

**Terminal log wealth after 120 months**

| | mean | median | 5% | 95% | sd |
|---|---:|---:|---:|---:|---:|
| unhedged | +0.5999 | +0.7143 | −0.7021 | +1.6526 | 0.9162 |
| hedged | **+0.7020** | +0.7039 | **−0.2151** | +1.6160 | **0.5572** |

**Maximum drawdown over 120 months**

| percentile | unhedged | hedged |
|---|---:|---:|
| 50% | 31.7% | 31.7% |
| 95% | 78.7% | 54.6% |
| 99% | 98.3% | 65.0% |
| 99.9% | **99.96%** | **75.5%** |

(The script reports drawdowns in log units; the table converts them as
`1 − e^(−d)`.)

### Interpretation

All three statements hold at the same time, and none should be dropped:

1. **Worst cases improve sharply.** The hedge turns a 1-in-1,000 wipeout into
   a 75% loss and roughly halves the spread of terminal wealth. Typical
   drawdowns are unchanged.
2. **The average improves.** Long-run geometric growth rises by about 0.085
   percentage points per month. The gain comes from the ~7% of paths where a
   put pays.
3. **The premium is paid everywhere.** The hedged bankroll trails on about 93%
   of paths, by exactly the cumulative premium when no crash occurs.

The average gain (+0.10) is small next to the normal spread between paths
(sd 0.56), with a ratio of 0.18. One ten-year history cannot reveal whether the
hedge "worked." Even for this naive rule, the main effect is in the drawdown
table, which a one-month integral cannot compute because drawdown depends on
the order of returns. None of this is a claim about real tail-hedging
strategies (see [A naive hedge, on purpose](#a-naive-hedge-on-purpose)).

### Sensitivity to price

E4's positive average depends on the put's price. Using `model.truth(markup)`:

| put price / fair value | average monthly gain |
|---:|---:|
| 1× | +0.00085 |
| 2× | +0.00055 |
| 4× | +0.00030 |
| 10× | +0.00003 |
| ~11.1× | 0 (break-even) |
| 20× | −0.00013 |

The average gain is very robust in E4 because the crashes are severe (mean
fall 78%), and log growth punishes near-wipeouts heavily. With milder crashes,
as in E8, a much smaller premium removes it.

Reproduce:

```bash
.venv/bin/python -m blindspot.experiments e4
.venv/bin/python -c "from blindspot import model as M; print(M.truth(markup=3.0)['edge'])"
```

![E4](../figures/bankroll_paths.png)

---

## 3. Side experiments

These experiments do not change the main finding. They test how reliable the
simulated numbers and their error bars are. E1–E3 use the same market as E4.
E5–E7 add volatility memory. E8 combines everything with a priced risk
premium.

### E1 – Rare payoffs: the median run reports zero

*Figure: `figures/rare_payoff.png`*

Ordinary Monte Carlo for the put payoff is unbiased at every budget. But
below about `N = ln2 / p ≈ 1,100` samples, most runs see no crash at all, so
the typical run reports zero. At small budgets, 52% of runs report exactly
zero. This is a budget problem, not a bias.

### E2 – Where the noise comes from

*Figure: `figures/variance_methods.png`*

Crashes carry 63% of the variance of the unhedged return but only 2% of the
variance of hedged log growth: the hedge removes the event it was built for.
As a result, the noise-reduction methods fail when used alone:

| method | variance reduction |
|---|---:|
| control variate alone | 0.6× (worse) |
| importance sampling alone | 0.8× (worse) |
| both together | **163×** |

### E3 – How many samples are needed?

*Figure: `figures/convergence.png`*

To estimate the hedge's growth effect precisely enough to tell its sign:

| method | samples needed |
|---|---:|
| plain Monte Carlo | ~14,000 |
| control variate | ~24,700 |
| control variate + importance sampling | **~73** |

Every method still converges at the `N^(−1/2)` rate. Variance reduction lowers
the starting point, not the slope.

### E5 – Volatility memory makes error bars too small

*Figure: `figures/memory_error_bars.png`*

With persistent volatility (H = 0.8), the usual standard-error formula, which
assumes independent months, understates the real uncertainty in estimated
growth. The gap is about 2× for short backtests and 3.5× at 4,096 months, so
it grows as more data are added. For rough volatility (H = 0.1) the formula
stays accurate. E5 keeps the conditional mean of simple returns constant, so
log returns inherit volatility memory through variance drag. Its diagnostics
report log-return correlations at lags 1 and 100 next to their theoretical
values.

### E6 – Same variance, different extreme drawdowns

*Figure: `figures/drawdown_by_hurst.png`*

Log wealth is modelled as fractional Brownian motion with the 12-month
variance fixed across H, on a daily grid. The 1-in-1,000 twelve-month
drawdown still depends strongly on H:

| H | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| 1-in-1,000 drawdown | 55.3% | 47.3% | 41.6% | 38.5% | 37.9% |

### E7 – Are drawdown intervals honest?

*Figure: `figures/interval_honesty.png`*

This puts ±1.96 × bootstrap-SE intervals on E6's 1-in-1,000 drawdown, using
two designs with 8,000 observations each and 24 repetitions per H:

- **Independent monthly paths.** Coverage is 0% for H ≤ 0.3 and rises to
  about 95% at H = 0.9. At low H the interval half-width is under 0.1× the
  actual error. This points to bias rather than noise, most likely because
  monthly sampling misses drops within the month on rough paths.
- **Windows of one daily path.** Coverage is close to nominal for mid-range H
  but falls to 29% at H = 0.9, where neighbouring windows are strongly
  dependent.

E7 keeps every run's estimate and interval and measures how often the
intervals contain the simulated daily-grid benchmark. The chart shows 95%
Wilson bounds for that coverage rate, and benchmark uncertainty is excluded.
The separate ratio half-width/(1.96 × RMSE) compares scales; it does not
establish 95% coverage.

### E8 – Naive rolling hedge with a risk premium and volatility memory

*Figure: `figures/unified_comparison.png`*

E8 combines the earlier ingredients:

- Normal monthly moves are mixed with occasional crashes (mean size 0.60 log
  points, a 45% fall).
- Volatility follows `σ_t = 0.05 exp(0.65 G_t)`, where `G_t` is fractional
  Gaussian noise with local Hurst exponent H. The noise resets every 64
  months.
- There are four regimes: rough (H = 0.1), memoryless (0.5), persistent (0.8),
  and alternating 0.1/0.8.
- The pricing model `Q` assumes a 0.3% monthly crash chance. The bankroll
  paths use `P` with 0.1%. The put therefore costs 3× its fair value under
  `P`.
- Each month the same naive rule spends 0.1% of wealth on a one-month put
  that pays below −30% and holds it to expiry.
- Volatility varies smoothly with memory but has no sudden spikes, and the
  hedge never gains from a rise in option prices before expiry.

```text
r_t = drift(σ_t, measure) + σ_t Z_t − I_t E_t,   E_t ~ Exponential(mean 0.60)
```

**Pricing.** Each run gets the same budget of 16,384 simulated months and must
recover the benchmark Q price of 0.00063126. All four methods do so within
simulation error.

| method | variance reduction vs plain Monte Carlo |
|---|---:|
| crash control | 1.4–1.6× |
| crash oversampling (importance sampling) | 4.3–5.5× |
| both | 4.4–5.6× |

**Bankroll outcomes.** These use 40,000 ten-year paths per regime, with seeds
8180–8183.

| regime | mean gain / month | hedge ends ahead | 99% drawdown, unhedged → hedged | 99.9% drawdown, unhedged → hedged |
|---|---:|---:|---:|---:|
| H = 0.1 | −0.00040 | 6.8% | 91.5% → 87.2% | 97.6% → 93.3% |
| H = 0.5 | −0.00043 | 6.7% | 91.3% → 87.4% | 97.8% → 93.8% |
| H = 0.8 | −0.00039 | 7.1% | 93.8% → 91.5% | 98.4% → 96.9% |
| 0.1 / 0.8 | −0.00041 | 6.8% | 92.6% → 89.6% | 98.0% → 95.5% |

The 99.9% column was produced by re-running E8's bankroll stage with the same
seeds; `e8` itself prints only the 99th percentile. With a 3× premium and
milder crashes, the average gain turns negative, and even the 1-in-1,000
drawdown improves by only 2–5 points. Better pricing methods reduce
estimation noise but do not change the economics of this naive hedge.

**Why the long-run H returns to 0.5.** A process with one fixed Hurst exponent
cannot have H ≠ 0.5 over short periods and H = 0.5 over long ones. E8 handles
this by joining independent regimes of length `B = 64`. For a span
`n = qB + r`:

```text
Var(Σ G_t) = q B^(2H) + r^(2H)
```

Within one regime, variance grows like `n^(2H)`. Across many regimes it grows
linearly in `n`, which is the H = 0.5 behaviour. Test T12 checks this.

### Supporting evidence: memory in S&P 500 volatility

*Figure: `figures/market_memory.png`. Details: [memory_evidence.md](memory_evidence.md)*

In daily S&P 500 returns (1990–2026), return **direction** has H ≈ 0.45–0.49,
consistent with no memory against a sign-randomized baseline. Return **size**
has H ≈ 0.84–0.95, meaning strong persistence. This is why E5 and E8 put
memory in volatility, not in direction.

---

## 4. Setup and commands

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Check the core model and its benchmark values.
.venv/bin/python -m blindspot.model

# Main experiment.
.venv/bin/python -m blindspot.experiments e4

# List experiments, run a subset, or run all.
.venv/bin/python -m blindspot.experiments
.venv/bin/python -m blindspot.experiments e1 e2
.venv/bin/python -m blindspot.experiments all --no-plots

# Numerical and regression checks.
.venv/bin/python -m tests.test_regressions

# Volatility memory in S&P 500 data (downloads ^GSPC on first run).
.venv/bin/python -m blindspot.sp500_memory

# Memory estimators on synthetic data.
.venv/bin/python -m blindspot.memory
```

Every experiment uses fixed, printed seeds, so results reproduce exactly. E4
uses seed 31. E8 uses 8080–8083 for pricing and 8180–8183 for bankroll paths.
The test suite contains 17 numerical checks (T1–T17) and eight guards against
bugs found during development (B1–B8). E7 takes about four minutes; the
others take seconds.

---

## 5. Files

| path | contents |
|---|---|
| `blindspot/model.py` | market models, benchmark prices, simulators and estimators |
| `blindspot/experiments.py` | command-line runner for E1–E8 and their charts |
| `blindspot/memory.py` | Hurst-exponent estimators |
| `blindspot/sp500_memory.py` | memory estimates from S&P 500 data |
| `blindspot/stable_tail.py` | corrected reference code from an earlier model |
| `tests/test_regressions.py` | numerical checks and bug guards |
| `docs/documentation.md` | this file |
| `docs/memory_evidence.md` | evidence for putting memory in volatility |
| `figures/` | generated charts |
| `data/` | downloaded market data (re-creatable cache) |

---

## 6. What the results do not prove

- **The hedge is naive.** It is a fixed monthly rule with no timing, sizing,
  early exit or monetisation of volatility spikes. The results do not describe
  real tail-hedging strategies, which are actively managed.
- **There are no volatility spikes.** Volatility is constant (E1–E4) or varies
  smoothly (E5, E8). Sudden jumps in volatility and in option prices, which
  are common in real markets and drive much of the value of real tail hedges,
  are not modelled.
- **E4 prices the put fairly.** Real crash insurance carries a risk premium.
  E4's average gain survives up to ~11× fair value in its own market, but E8
  shows that with milder crashes a 3× premium is enough to make it negative.
  The drawdown benefit holds in both, but its size depends on crash severity
  and price.
- **These are scenarios, not calibrations.** Crash probabilities, crash sizes,
  strikes, monthly spend and the 64-month reset are chosen, not fitted to
  option prices. They should be varied before drawing financial conclusions.
- **E4 assumes independent months.** Volatility memory (E5–E8) widens the
  real uncertainty around any estimate from one history.
- The crash model produces strongly negative outcomes, but it does not claim
  that real crashes follow a specific power law.
- In E5 and E8, memory enters through volatility, and log returns inherit it
  through volatility-dependent drift. E6–E7 use a separate fBm bankroll
  family to stress-test drawdown estimators.
- Real market regimes do not end on a fixed 64-month schedule; the reset is a
  simple way to model finite memory.
- Extreme-drawdown estimates need their own uncertainty ranges (see E7) before
  being used as risk limits.
- The script that measured the Hurst estimators' bias on synthetic data is no
  longer in the repository; see [memory_evidence.md](memory_evidence.md).
