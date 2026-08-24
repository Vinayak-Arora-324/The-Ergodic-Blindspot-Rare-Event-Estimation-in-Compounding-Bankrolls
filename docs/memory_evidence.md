# Why the model gives volatility a memory setting

**Figure:** `figures/market_memory.png` · **Code:** `blindspot/sp500_memory.py` ·
**Data:** adjusted daily S&P 500 prices from 1990-01-02 to 2026-07-31 (9,211
log returns, downloaded from Yahoo Finance).

The model lets volatility depend on its recent past. It describes the strength
of that dependence with the Hurst exponent `H`:

- `H ≈ 0.5` means no lasting pattern.
- `H > 0.5` means high or low values tend to continue.
- `H < 0.5` means values tend to reverse direction quickly.

This setting needs evidence. Otherwise, changing `H` would be an arbitrary way
to move the simulated drawdown results.

The S&P 500 data support memory in the **size** of returns, but not in their
**direction**:

| series | variogram | DFA | Higuchi | result after randomizing signs |
|---|---:|---:|---:|---:|
| log returns (direction) | 0.491 | 0.449 | 0.467 | ≈ 0.50 ± 0.03 |
| absolute log returns (volatility) | 0.842 | 0.951 | 0.868 | ≈ 0.50 ± 0.04 |

The three methods disagree slightly on the exact number, but they tell the same
story. Return direction is close to memoryless. Large moves, however, tend to
cluster with other large moves, and quiet periods tend to remain quiet. That is
why `model.py` puts memory in volatility rather than using it to predict whether
the market will rise or fall next.

## Why the comparison method matters

A memory estimate needs a fair “no directional memory” baseline. Simply
shuffling the returns is not fair because it destroys both directional patterns
and volatility clustering. The shuffled data therefore look very different
from real market data even when return direction itself has no memory.

This analysis uses a more focused comparison. It keeps every return magnitude
in its original position and randomly changes only its sign:

```text
r* = randomly chosen + or - sign × |r_t|
```

The resulting series keeps calm and turbulent periods intact while removing
any ability to predict direction. Against this baseline, most of the apparent
directional pattern disappears. The strong pattern in absolute returns remains.

The baseline also changes what counts as a neutral Hurst estimate. For the
Higuchi method, shuffled returns average `H = 0.560`, while sign-randomized
returns average `H = 0.507`. That difference of about 0.05 comes from changing
the volatility pattern, not from directional memory.

Short samples make the problem worse. In rolling five-year windows, the
baseline estimate changes with the market's volatility and reaches its highest
levels around 2008–09. A 2004–09 estimate of `H = 0.416` looks strongly
mean-reverting when compared with a fixed `0.5`. Compared with a baseline built
from the same volatility pattern, it mostly looks like an estimator being
distorted by extreme returns.

This is a smaller version of the project's main point: an error bar can look
precise while leaving out the most important source of uncertainty.

## Limits of this evidence

- **The answer depends on the time scale.** DFA estimates `H ≈ 0.82` for
  absolute returns using windows up to 50 days and about `0.95` using windows
  up to 250–1,000 days. The table uses 250 days, or roughly one trading year.
  The conclusion that volatility is persistent is stable; the exact estimate
  is not.
- **An estimate near 1 needs caution.** The DFA 95% interval is
  `[0.864, 1.060]`. Values around 1 strain the assumptions behind this model.
  A slow change in the general level of volatility may explain part of the
  result. Read it as “strong persistence, roughly 0.84–0.95,” not as a precise
  estimate of `0.951`.
- **Five years of data are not enough here.** In five-year windows, almost every
  estimate falls inside its own no-memory range. These methods cannot reliably
  distinguish the index from a memoryless process with so little data.
- **The methods have some built-in bias.** On synthetic data with a known `H`,
  DFA runs about 0.02 high at `H = 0.3`, and the variogram runs about 0.08 low
  at `H = 0.9`. The script that measured those biases is no longer in the
  repository, so they should be measured again before being used in a formal
  analysis.
