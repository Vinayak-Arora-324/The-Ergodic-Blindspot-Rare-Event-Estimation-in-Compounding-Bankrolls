# `Calculation.py` — audit

Tested against analytic oracles, not against its own recorded output. Files:
`test_calculation.py` (21 checks), `calculation_fixed.py` (corrected reference),
`calculation_audit.png` (the two failure modes).

```bash
python test_calculation.py     # 9 pass, 14 fail
python verify_fixed.py         # all oracles green on the corrected version
```

## What it does when you run it

The script executes without error and prints an expected call payoff of
**$3.66 × 10⁸⁶** on a 4500/4600 three-month call, with a **95.9%** probability
that an out-of-the-money call finishes in the money. Neither number is flagged.
That is the finding: the failure is silent and confidently quantified.

## Findings

### 1. The model is undefined at the parameters it ships with — `martingale_condition`

For an exponential α-stable model with α ∈ (1,2), `E[S_T] = ∞` unless β = −1
(the finite-moment log-stable case). The script's defaults are α = 1.1,
β = −0.2, where the mean does not exist, so there is no drift that makes the
discounted price a martingale. `martingale_condition` returns one anyway.

Numerically, `E[e^X]` under the script's own density does not converge:

| truncation of the standardized variable | 50 | 200 | 1000 | 5000 | 20000 |
|---|---|---|---|---|---|
| `E[e^X]`, α=1.5, β=0 | 1.011 | 1.342 | 1.0×10¹⁵ | 1.3×10⁹⁷ | `inf` |

Even in the one well-posed case (β = −1) the formula is wrong. Solving for the
true correction numerically and comparing:

| α | true correction | code | ratio |
|---|---|---|---|
| 1.9 | −8.538×10⁻⁴ | +1.352×10⁻⁴ | −0.158 |
| 1.7 | −1.723×10⁻³ | +8.780×10⁻⁴ | −0.510 |
| 1.5 | −3.953×10⁻³ | +3.953×10⁻³ | −1.000 |
| 1.3 | −1.121×10⁻² | +2.200×10⁻² | −1.963 |

The ratio is exactly `−tan(πα/2)` at every α. The correct expression is

```
mu = (r-q)*T + gamma**alpha / cos(pi*alpha/2) * T          # sec, no tan
```

and the code carries a spurious `tan(πα/2)` factor plus a flipped sign. Worth
noting: **at α = 1.5, `tan(3π/4) = −1` exactly**, so the magnitude is right and
only the sign is wrong. A single spot-check at α = 1.5 against a magnitude
tolerance would have passed.

At the shipped defaults the bad correction implies a **+72.5% expected move
over three months** where the risk-free leg is +0.875%. That alone explains the
95.9% ITM probability.

### 2. The call integrand computes the reciprocal of the payoff — `expected_itm_payoff`

```python
log_payoff = np.log(S0) + x - np.log(S0 * np.exp(x) - K + 1e-100)
```

exponentiates to `S0·eˣ / (S0·eˣ − K)`, not `S0·eˣ − K`. Two consequences:

- At a point 0.5 above the strike it returns **2.54** where the payoff is
  **71.36** — off by 28×, and the direction of the error flips with moneyness.
- The `+ 1e-100` guard puts a **pole at `x = log(K/S0)`**, which is exactly the
  lower limit of integration. Integrand values approaching it: 10.5 → 10³ →
  10⁵ → 10⁷. This is the source of the `IntegrationWarning` and the 10⁸⁶.

The put branch has the correct payoff and reproduces Black–Scholes to 1×10⁻⁴ at
α = 2 (**T4 passes**), which isolates the bug to the call branch. Put–call
parity on the code's own outputs: `C − P = 4.37×10⁸⁶` against a true
`S₀e^{−qT} − Ke^{−rT} = 0.7435`.

### 3. The `+10` cutoff is load-bearing — panel A

`quad(integrand, log_threshold, log_threshold + 10)` is the only thing making
the answer finite. With the payoff corrected, at α = 1.5, β = 0:

| upper limit | +5 | +10 | +20 | +40 |
|---|---|---|---|---|
| expected payoff | 0.817 | 9.32 | 2.85×10⁴ | 2.27×10¹² |

At β = −1 the same sweep is flat to 12 digits. So the cutoff is harmless
exactly when the model is well-posed and load-bearing exactly when it isn't —
the divergence is invisible from the output.

### 4. The α estimator is saturated, not noisy — `quantile_estimation`, panel B

`ν_α = (x₉₅−x₀₅)/(x₇₅−x₂₅)` **decreases** in α: it is 2.4386 for a Gaussian and
rises as tails fatten. The branch test reads it backwards — `phi_1 <= 2.4390`
maps to `alpha_est ≈ 0.10`, when that condition means α ≥ 2. Measured transfer
function on 300k stable draws:

| true α | 1.1 | 1.3 | 1.5 | 1.7 | 1.9 | 2.0 |
|---|---|---|---|---|---|---|
| estimated | 1.10 | 1.10 | 1.10 | 1.41 | 1.86 | **1.10** |

Every true α ≤ 1.6 returns the `np.clip` floor of 1.1, and **true α = 2 returns
1.10** because ν_α straddles 2.4386 by sampling noise. On twelve independent
Gaussian samples of 400k the estimate flipped 1.10 / 2.00 / 1.10 / 1.10 / 2.00 /
… — a discontinuity sitting precisely on the Gaussian boundary, which is the one
place this project needs the estimator to be reliable.

β is recovered no better: true β = 0.3 → estimated 0.021; true β = 0.0 →
estimated −0.369. γ carries a 12–16% low bias throughout. The `delta` line also
has the sign of McCulloch's ζ→δ conversion backwards, though β being this
badly estimated makes that the smaller problem.

## Corrected reference

`calculation_fixed.py` — after the fixes, against the same oracles:

| check | result |
|---|---|
| call price vs Black–Scholes at α=2 | rel. err 4.2×10⁻¹⁵ |
| put–call parity | exact to 8 decimals |
| `E[S_T]/S₀ = e^{(r−q)T}`, β=−1, α ∈ {1.9,…,1.3} | rel. err 2.6×10⁻¹⁰ → 1.5×10⁻⁶ |
| β > −1 | raises `ValueError` instead of returning a truncation artifact |
| α round-trip, α ∈ [1.3, 1.9] | \|Δα\| ≤ 0.009, \|Δβ\| ≤ 0.025, γ within 0.4% |
| Gaussian data, 12 seeds | 2.00 every time |

The α estimator was replaced with a grid inversion of ν_α, ν_β computed from
`levy_stable.ppf` — no hard-coded table to mistype, and self-verifying by
round-trip.

The `expected_itm_payoff` rewrite integrates to +30 and +60 and **raises if the
two disagree**, rather than reporting the +30 value. That is the design point:
the divergence in finding 3 is undetectable from a single evaluation and
obvious from two.

## Why this belongs in front of the project

The README's thesis is that a simulation can be unbiased, converge at the right
rate, and still report a number that is wrong in a way its own error bar cannot
see. `Calculation.py` is the same failure without the Monte Carlo:

- **A finite answer to a question whose answer is infinite** (findings 1, 3).
  The integral diverges; the cutoff hides it; the output has no diagnostic.
  This is E5's claim — reported uncertainty not knowing it is wrong — in a
  deterministic quadrature setting, which is a cleaner statement of it than
  anything currently in the repo.
- **A parameter estimate pinned to a clip boundary** (finding 4) is the same
  shape as E1's median-lies result: a statistic that looks like a measurement,
  is stable across reruns, and is not responding to the data. Panel B is that
  claim in one line.
- The α = 1.5 coincidence in finding 1 — where two errors cancel to give the
  right magnitude — is a concrete instance of the project's own methodological
  point about checking against analytic ground truth at more than one setting.

There is also a direct connection to the current fBm direction. The Hurst
exponent and the stability index α are both tail/memory parameters recovered
from data by quantile-type estimators, and finding 4 shows what a
badly-conditioned one looks like: not scatter around the truth, but saturation
against a bound with a discontinuity at the null. Worth checking whether the
`hurst_rs` and `hurst_dfa` estimators in `MemoryCalculator.py` have the same
property near H = ½ before any H-parameterized drawdown curve rests on them.

## Caveat on this audit

Two of my own tests initially failed for the wrong reason. Integrating
`e^x · f(x)` over `[−60, 40]` with `quad` returned ~10⁻¹⁷ at α = 2, which I
first read as divergence in the code; it was `quad` missing a width-0.05 bulk in
a 100-wide window. Rewriting the tests in standardized coordinates with a split
range fixed it. This is caveat 2 in the README, encountered live — the numbers
above are from the corrected tests.
