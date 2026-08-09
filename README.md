# The Ergodic Blindspot


The question for this project is whether, in a market regime resembling real conditions with a longer or shorter Hurst exponent over shorter periods but a diffusive one (0.5) around longer terms, how reliably can crude Monte Carlo and variance-reduction methods can price the same hedge and how they differ from realized outcomes

The "ergodic blindspot" of the title is the gap between ensemble averages and
the one compounding path a bankroll actually lives. E1 and E4 exhibit it
directly: an unbiased estimator whose typical run reports zero, and path
functionals -- win rate, maximum drawdown -- that no one-period expectation can
reach. E5 and E7 show long-range dependence widening the gap by making reported
error bars dishonest, and E8's finite-memory regimes are the construction that
restores trustworthy long-run time averages.


## At a glance

```text
blindspot/   model, estimators, experiments, and memory tools
tests/       numerical and regression checks
figures/     generated results with descriptive names
docs/        research motivation and interpretation
data/        regenerable market-data cache
```

Run the central comparison with:

```bash
.venv/bin/python -m blindspot.experiments e8
```

## Central experiment: E8

E8 combines the ingredients that E1–E7 study separately:

- Frequent crashes or turbulent behaviour create a very fat left tail
- lognormal stochastic volatility is driven locally by exact fGn;
- fGn resets after 64 months, so a regime has local `H = 0.1`, `0.5`, or `0.8`
  while aggregated variance eventually grows linearly and effective `H -> 0.5`;
- `(0.1, 0.8)` alternates rough and persistent regimes in one history;
- `Q` prices the monthly 30%-OTM put and assigns crashes probability 0.3%;
- `P` generates bankroll outcomes and assigns crashes probability 0.1%;
- the rolling strategy spends 0.1% of wealth on the conditionally Q-priced put
  each month.

The return model is

```text
r_t = drift(sigma_t, measure) + sigma_t Z_t - I_t E_t,
sigma_t = 0.05 exp(0.65 G_t),
E_t ~ Exponential(mean 0.60).
```

The drift pins the conditional expected gross return under each measure. This
keeps option pricing under `Q` distinct from strategy evaluation under `P`.

### Equal-budget estimators

Every E8 pricing run receives 16,384 simulated month-observations and targets
the same deterministic quadrature price:

```text
Q put price = 0.00063126 per unit spot
```

The four estimators are:

1. crude Monte Carlo;
2. crash-indicator control variate, using known `E_Q[I] = lambda_Q`;
3. importance sampling, increasing simulated crash probability from 0.3% to
   20% and applying the exact likelihood ratio;
4. the control variate and importance sampling together.

Across rough, Brownian, persistent, and alternating regimes, the recorded
variance reductions are approximately:

| method | variance reduction vs crude |
|---|---:|
| crash control | 1.4–1.6x |
| importance sampling | 4.3–5.5x |
| control + importance sampling | 4.4–5.6x |

All four estimates remain unbiased within Monte Carlo error. Variance reduction
makes the price cheaper to estimate; it does not make the strategy profitable.
With the specified `Q` tail premium, the rolling hedge has roughly `-4e-4` mean
monthly log carry and wins on about 7% of 10-year paths, while reducing the
99th-percentile maximum drawdown by roughly 2–4 percentage points depending on
the local volatility regime.

![Unified experiment](figures/unified_comparison.png)

## Why long-run H becomes 0.5

A single constant-H fBm/fGn process cannot be both locally `H != 0.5` and
asymptotically `H = 0.5`. E8 therefore uses independent finite regimes. For
regime length `B`, horizon `n = qB + r`, and local exponent `H`,

```text
Var(sum G_t) = q B^(2H) + r^(2H).
```

Inside one regime this scales as `n^(2H)`. Across many regimes it is
proportional to `n`, giving effective `H = 0.5`. Test T12 verifies this identity
for rough, Brownian, persistent, and alternating regimes.


## Run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m blindspot.model
.venv/bin/python -m blindspot.experiments e8
.venv/bin/python -m blindspot.experiments all --no-plots
.venv/bin/python -m tests.test_regressions

# Empirical Hurst analysis; downloads ^GSPC on the first run.
.venv/bin/python -m blindspot.sp500_memory

# Standalone estimator demonstration (uses a synthetic walk without chart.csv).
.venv/bin/python -m blindspot.memory
```

`experiments.py` uses fixed seeds. E8 uses seeds 8080–8083 for pricing and
8180–8183 for strategy paths. The current regression suite contains 15 verified
numerical/structural tests and five committed-bug guards.

## Project layout

| path | purpose |
|---|---|
| `blindspot/model.py` | canonical models, pricing formulas, simulators, and estimators |
| `blindspot/experiments.py` | E1–E8 command line and figure generation |
| `blindspot/memory.py` | variogram, DFA, Higuchi, and MFDFA estimators |
| `blindspot/sp500_memory.py` | empirical S&P 500 memory analysis |
| `blindspot/stable_tail.py` | retained corrected alpha-stable reference |
| `tests/test_regressions.py` | T1–T15 and B1–B5 |
| `docs/memory_evidence.md` | empirical motivation and limits of H estimation |
| `figures/` | named result figures |
| `data/` | regenerable market-data cache |

## Interpretation limits

- E8 is a controlled methods experiment, not a calibrated trading strategy.
- Its Gaussian-plus-negative-exponential mixture is strongly left-skewed and
  much heavier-tailed than the Gaussian bulk, but it is not a regularly varying
  Mandelbrot/Pareto tail; that tail-law choice remains a replaceable model layer.
- The P/Q crash probabilities, memory cutoff, jump scale, strike, and spend are
  explicit scenario parameters and should be swept before economic conclusions.
- The memory is placed in volatility, not in return direction.
- Reset regimes are a transparent finite-memory construction, not a claim that
  real regimes terminate exactly every 64 months.
- Tail drawdown estimates should carry quantile uncertainty before being treated
  as production risk numbers.
