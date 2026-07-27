# Tail-Hedge Estimator Project — Rebuild Spec

**Purpose of this document.** Exploratory work produced ~11 scratch scripts, several
of which contain bugs or support retracted claims. This is the consolidated spec for
a clean rebuild. Everything here has been numerically verified; the recorded numbers
are acceptance tests. Build only what is listed in §2.

---

## 1. The research question

A methods project, not a finance project. The object of study is **the estimator**;
the tail-hedge strategy is the test case.

> Simulate the growth of a finite bankroll running a tail-hedge rule under
> fat-tailed dynamics; show which quantities are rare-event-dominated; and show how
> much the answer moves with the estimator and the model. The deliverable is a map
> of where the answer *can't* be known, not a verdict that tail hedging works.

Three findings organize the code. Each is a distinct way a simulation lies:

1. **The median lies.** For a rare payoff, crude MC is unbiased in expectation, but
   the *typical* run reports near-zero. Not bias — extreme right-skew of the
   estimator's sampling distribution. (E1)
2. **The hard object is the edge, not the level.** Hedging *destroys* the
   rare-event structure of the hedged quantity: in a crash the put payoff cancels
   the index loss, so hedged log-growth is mild and bounded. Crashes carry ~2% of
   its variance vs ~64% for the unhedged return. So rare-event machinery is needed
   for the unhedged baseline and the hedged-minus-unhedged *difference*, not for the
   hedged portfolio. (E2)
3. **The error bar lies.** Under long-memory volatility, iid-formula confidence
   intervals are systematically overconfident, and the overconfidence *grows* with
   sample size. (E5)

And one structural result that reframes the whole thing:

4. **The estimand choice dominates the estimator choice.** Under iid, the ergodic
   growth rate `g` collapses to a 1-D integral (LLN), so quadrature beats every
   Monte Carlo method and the rare-event machinery is unnecessary. But path
   functionals — realized-path win rate, max drawdown, ruin — do *not* collapse, are
   unreachable by quadrature, and are what a finite bankroll actually cares about.
   (E4) This is the honest bridge to the next stage (§8).

---

## 2. What to build

Nine scratch files collapse to eight real ones. **One** model module — the model was
previously duplicated across five scripts with divergent parameterizations, which is
how bug B1 survived.

```
tailhedge/
  model.py                    # canonical model + hedge + quadrature truth
  estimators.py               # crude, cv, cv_is
  experiments/
    e1_single_period.py       # the median lies
    e2_iid_growth.py          # baseline + variance diagnosis (merge of 2 old scripts)
    e3_convergence_fig.py     # the two-panel figure
    e4_path_functionals.py    # what quadrature cannot reach
    e5_lrd_errorbars.py       # fGn-driven vol; the error bar lies
  tests/test_regressions.py   # encodes §6 so the bugs cannot return
```

Do **not** build: parameter sweeps over the Student-t model (bugs B1+B2, and they
supported retracted claims — see §5); any bankroll `for`-loop to estimate `g` (the
LLN already did it; a loop only adds MC noise to a 1-D integral).

Dependencies: `numpy`, `scipy`, `matplotlib`. No network.

---

## 3. Canonical model (`model.py`)

Monthly log return: light-tailed bulk plus a rare negative jump. This is the honest
shape for equity crashes (left-skewed, not symmetric) and is the natural restriction
of CGMY. It replaced a symmetric Student-t model, which was discarded — see §5.

```
r = m + s·Z          with prob 1−λ
r = m + s·Z − E      with prob λ,   E ~ Exp(mean β),  Z ~ N(0,1)
```

| param | value | meaning |
|---|---|---|
| `LAM`  | `1e-3`  | jump probability per month |
| `BETA` | `1.5`   | mean jump size (log points) |
| `S`    | `0.05`  | bulk monthly vol |
| `MU`   | `0.005` | **target** `E[r]`, pinned |
| `m`    | `MU + LAM*BETA` = `0.0065` | bulk mean, set so `E[r] = MU` exactly |

Pinning `E[r] = MU` is load-bearing: it makes `g_unhedged = MU` known in closed
form, so all estimation difficulty sits on the hedged side.

### Density

`f(r) = (1−λ)·N(r; m, s) + λ·ψ(r)` where `ψ` is the density of `N(m,s) − Exp(β)`.

`ψ` is an exponentially modified Gaussian, reached by sign-flip:
`−r = Exp(β) + N(−m, s)`. In scipy terms `ψ(r) = exponnorm.pdf(−r, K=β/s, loc=−m, scale=s)`.

Provide a closed-form numpy version for speed (needed in E3 — scipy pdf calls
dominate runtime at 1e7+ evaluations):

```python
y = (-r + m) / S;  Kp = BETA / S
psi = (1/(2*Kp)) * exp(1/(2*Kp**2) - y/Kp) * erfc((1/Kp - y)/sqrt(2)) / S
```

**Required assertion** (this is bug B3): the closed form must match
`scipy.stats.exponnorm.pdf` to `rtol=1e-10` on a grid spanning bulk and deep tail,
e.g. `[-8, -3, -0.5, 0, 0.05, 0.4]`. No overflow risk at these params (`Kp = 30`).

The sampler and the density must be independently verified against each other —
see test T1.

### Hedge

Roll a deep-OTM put monthly. Each month, wealth fraction `c` buys puts, `1−c` holds
the index.

| param | value |
|---|---|
| `S0` | `100.0` |
| `K`  | `50.0` (strike) |
| `c`  | `0.0005` (premium as fraction of wealth) |

```
payoff(r)     = max(K − S0·exp(r), 0)
price         = ∫ payoff(r)·f(r) dr          # actuarially fair, markup = 1
log_growth(r) = log( (1−c)·exp(r) + c·payoff(r)/price )
```

Keep a `markup` parameter (default `1.0`). At `markup=1` there is no variance risk
premium, so every "hedge wins" result is conditional on a put market that does not
exist. This is the project's standing pro-Spitznagel tilt and must be stated
wherever results are reported. `markup > 1` is the AQR-side sensitivity and is the
first knob to sweep in §8.

### Quadrature truth

Integrate on `[-40, 2]` with `limit=1500`.

```
g_hedged   = ∫ log_growth(r)·f(r) dr
g_unhedged = MU                          # exact, by construction
edge       = g_hedged − g_unhedged
p_itm      = ∫_{-40}^{log(K/S0)} f(r) dr # P(put finishes in the money)
```

| quantity | verified value |
|---|---|
| `g_unhedged` | `+0.005000` |
| `g_hedged`   | `+0.005852` |
| `edge`       | `+0.000852` |
| `p_itm`      | `6.28e-4` |

Caveat to carry in a comment, not a docstring claim: adaptive `quad` over `[-40, 2]`
resolves the width-0.05 bulk only because the jump density fills the interval and
guides subdivision. This is fragile — with a pure-normal model the same call can
silently return near-zero. Do not describe quadrature output as "exact"; it is
accurate to ~machine precision *here*, verified by T1.

---

## 4. Estimators (`estimators.py`)

All three are unbiased and target `g_hedged`.

| name | definition |
|---|---|
| `crude`  | `mean(log_growth(x))`, `x ~ f` |
| `cv`     | `mean(D(x)) + MU`, `x ~ f`, where `D(r) = log_growth(r) − r` |
| `cv_is`  | `mean(D(y)·f(y)/q(y)) + MU`, `y ~ q` |

`D` is a **control variate**: `E[r] = MU` is known exactly, so only `E[D]` needs
estimating. `D` is constant (`log(1−c)`) in the bulk and nonzero only in crashes.

`q` is the same model with jump probability inflated `LAM → LAM_Q = 0.30`. Weights
`f/q` are **bounded** — verified range `[0.0033, 1.4228]` — hence finite variance by
construction. Bulk samples get weight ≈1.43, oversampled crash samples ≈1/300.

Counter-intuitive but verified, and the code must not "fix" it:

- `cv` alone **increases** variance (0.6× vs crude). Per-sample sd: `log_growth`
  0.0510, `D` **0.0679**. `D` is the hedge's effect — purely crash-driven and
  violently skewed — so it is *harder* to estimate than the level.
- `is` alone (no CV) also **fails** (0.8×): it oversamples a region carrying ~2% of
  the variance while inflating bulk variance via the 1.43 weight.
- `cv_is` **works** (190×): CV reshapes the estimand into the crash-driven thing IS
  is built for. The composition is the point; neither half works alone.

---

## 5. Retracted claims — do not restate

These appeared in exploratory work and are **false**. They must not reappear in
comments, docstrings, or write-ups.

- ❌ *"Tail hedging only helps when `p ~ 1e-3`; the edge vanishes for rarer tails."*
  Artifact of a premium grid floored at `c = 2e-4`. Verified: at `λ=1e-5`
  (`p≈6.3e-6`), the grid's `c` gives edge `−1.66e-4` (bleeds) but `c ≈ 0.8p` gives
  `+8.53e-6` (wins). Under fair pricing the edge stays positive at any rarity when
  `c` is sized to `p`; it shrinks ∝ `p`. **Any premium sweep must scale `c` with
  `p`, never use a fixed grid.**
- ❌ *"Hedge-wins configurations cluster at `ν < 2` (infinite variance), so the
  Student-t model has a right-tail confound."* Contaminated by bug B1. The
  underlying sweep also showed wins at `ν = 3` and `ν = 4`.
- ❌ *"Estimating the growth of a hedged bankroll is itself a rare-event problem."*
  This was the project's original founding sentence and it is **backwards** — see
  finding 2 in §1. The unhedged baseline and the edge are rare-event-dominated; the
  hedged portfolio is not.
- ❌ *"A 190× speedup matters here."* It is real machinery aimed at a problem the
  iid model does not have: crude MC needs ~14.7k samples (milliseconds) and
  quadrature needs none. Report it as a verified capability plus a negative result.

---

## 6. Known bugs — do not reintroduce (`tests/test_regressions.py`)

| id | bug | test |
|---|---|---|
| **B1** | Scale-convention switch on tail index: `scale = σ/sqrt(ν/(ν−2)) if ν>2 else σ`. Discontinuous — `ν=2.0` → `0.060`, `ν=2.5` → `0.0268`, so `ν≤2` rows silently had ~2.2× fatter scale, not just fatter tails. Invalidated all cross-`ν` comparisons. | Never write this line. If any t-model appears, assert scale is continuous in `ν`. |
| **B2** | Fixed premium grid, floor `c=2e-4`, compared across jump intensities → every rare-`λ` config bled by construction. | Assert `edge(λ=1e-5, c=0.8·p) > 0`. |
| **B3** | `exponnorm` parameterization is easy to get wrong (`K = β/s`, `loc = −m`, `scale = s`, evaluated at `−r`). | Assert closed form matches scipy, `rtol=1e-10`. |
| **B4** | Truncating the truth integral too tightly. Stage-1's `[-6,6]` cut 0.47% of the value. | Assert truth is stable when the lower limit is widened (`-40` vs `-60`). |
| **B5** | Labelling `P(≥1 jump)` as `P(≥1 crash)`. Different numbers: over 120 months, `11.31%` vs `7.26%`. Only in-the-money puts can help. | Assert both are computed and reported separately. |

### Required regression tests

| id | test | verified value |
|---|---|---|
| T1 | density vs sampler: `P(r < log(K/S0))` empirical (4e6 draws) vs quadrature | `6.33e-4` vs `6.28e-4` |
| T2 | truth values | `g_h = +0.005852`, `edge = +0.000852` |
| T3 | crash share of variance: hedged `log_growth` vs unhedged `r` | `1.8%` vs `64.0%` |
| T4 | crash-period means: hedged `log_growth` bounded, unhedged not | `−0.034` vs `−2.231` |
| T5 | IS weights bounded | `[0.0033, 1.4228]` |
| T6 | all three estimators unbiased | within `1.4` MC-SE of `g_h` at `N=4096`, `2000` trials |
| T7 | all three converge at `N^{-1/2}` (no infinite-variance blowup) | slopes `−0.499`, `−0.511`, `−0.498` |

---

## 7. Experiments and acceptance numbers

### E1 — the median lies (single period)

**Note:** the verified numbers below come from a Student-t model
(`ν=3, σ=0.05, μ=0, S0=100, K=55, N=5000`), the only place a t-model survives.
Preferred: **port E1 to the canonical jump model** so the codebase has one model,
and regenerate. If ported, these exact numbers will not hold; the qualitative
acceptance criteria are what must survive (median ≈ 0, large fraction of runs see
zero payoff, mean ≈ truth). If kept as-is, isolate the t-model inside E1 and
**do not** write B1's scale line.

| quantity | verified value |
|---|---|
| `p` (put ITM) | `1.23e-4` (≈0.6 crashes per run of N=5000) |
| true `E[payoff]` (quad) | `0.00141` |
| mean of 3000 estimates | `0.00135` (≈unbiased) |
| **median estimate** | **`0.00000`** |
| runs that saw zero crashes | `55.43%` |
| runs below half of truth | `65.40%` |
| coefficient of variation | `1.83` |

### E2 — iid growth baseline + variance diagnosis

Merge of the old `stage2_final.py` and `diagnose.py`. Reports truth, the three
estimators at `N=5000`, the variance decomposition, and the sample counts needed.

| quantity | verified value |
|---|---|
| per-sample sd of `log_growth` | `0.0510` |
| per-sample sd of `D` | `0.0679` (CV hurts) |
| `N` for crude to resolve sign at 2σ (analytic `(2σ/edge)²`) | `14,321` |
| `N` after CV | `25,401` |
| variance reduction: `cv` | `0.6×` |
| variance reduction: `cv_is` | `189.5×` |
| crude sd at `N=5000` | `0.000746` |

### E3 — convergence figure

Two panels. Panel A: sampling distributions at `N=4096` (log y-axis — on a linear
axis the `cv_is` spike erases the other two), with `g_h` and `g_u` marked; the gap
between them is the whole decision. Panel B: sd vs `N` log-log over
`N ∈ {64, 256, 1024, 4096, 16384, 65536}`, 300 trials each, with a dashed threshold
at `edge/2` (the sd needed to call the sign at 2σ) and markers where each method
crosses.

| estimator | slope | `N` to resolve sign |
|---|---|---|
| crude | `−0.499` | `14,745` |
| `+cv` | `−0.511` | `27,332` |
| `+cv_is` | `−0.498` | `82` |

Empirical `14,745` vs analytic `14,321` (§E2) agree to ~3% — cross-validates both.

**The figure's caption must be the deflationary one.** All slopes are `−1/2`:
variance reduction moves the *intercept*, never the rate. And the winning method
cannot be drawn on these axes — quadrature has no `N`, so a log-log convergence plot
structurally cannot reveal that Monte Carlo was the wrong tool. Caption accordingly.

### E4 — path functionals (most important experiment)

`T = 120` months, `M = 400,000` paths, **common random numbers** so hedged and
unhedged face the identical realized market. This is what quadrature cannot reach.

| quantity | verified value |
|---|---|
| `T·edge` (what `g` promises over 10y) | `+0.1023` log |
| mean difference across paths | `+0.1031` ✓ matches — MC agrees with quadrature |
| **median difference** | **`−0.0600`** (hedge loses on the typical path) |
| **P(hedged beats unhedged on the realized path)** | **`7.13%`** |
| `P(≥1 put ITM in 120 mo)` | `7.257%` — matches the win rate |
| `P(≥1 jump in 120 mo)` | `11.249%` (theory `11.313%`) — different quantity, see B5 |
| paths with no jump at all | `88.75%`, mean effect `−0.0600`, P(help) `0.0%` |
| paths with exactly 1 jump | `10.58%`, mean effect `+1.3039`, P(help) `61.9%` |
| paths with 2+ jumps | `0.67%`, mean effect `+2.7501` |
| sd of 120-month log wealth | `0.5584` |
| signal ÷ path noise over horizon | `0.183` (edge buried ~5 deep) |

Terminal log wealth: unhedged mean `+0.6002`, median `+0.7166`, 5% `−0.7148`,
95% `+1.6511`. Hedged mean `+0.7033`, median `+0.7060`, 5% `−0.2204`, 95% `+1.6130`.

Max drawdown (log) — a running-extremum functional, provably not a one-period
expectation:

| | median | 95% | 99% | 99.9% |
|---|---|---|---|---|
| unhedged | `0.379` | `1.544` | `4.078` | `7.753` |
| hedged | `0.379` | `0.787` | `1.049` | `1.400` |

Identical in the middle; at the 99.9th percentile the hedge converts a ~99.96%
wipeout into a ~75% loss. **This is the actual case for tail hedging and it is
invisible in `g` by construction.** Both columns of E4 are true at once: positive
geometric edge, and underperformance on 93% of livable paths, because the mean is
carried by the 7% where a put lands.

### E5 — the error bar lies (fGn-driven volatility)

First experiment where the estimand is genuinely path-space. Returns stay serially
**uncorrelated** (matching the stylized fact); memory enters the growth estimand
through the variance drag `σ_t²/2`.

```
r_t      = μ_a − σ_t²/2 + σ_t·Z_t,      Z iid N(0,1)
log σ_t  = log(σ̄) + ξ·G_t,              G = fractional Gaussian noise, Hurst H
g_true   = μ_a − ½·σ̄²·exp(2ξ²)          # closed form, since Var(G_t)=1
```

Params: `μ_a=0.01`, `σ̄=0.15`, `ξ=0.9`, `T_MAX=4096`, `M=1200`.
Generate exact fGn by Cholesky of the covariance (a path prefix is then a valid
shorter path, so nested `T` values reuse one draw). Estimate `g` by the time average
over one path (i.e. one backtest); compare the **true SE** (sd across independent
paths) to the **naive SE** (`sd/√T` from one path, what a practitioner reports).

`g_true = −0.04685`. Sanity: `corr(r_t, r_{t+100}) ≈ 0`, `corr(|r_t|, |r_{t+100}|) > 0`.

| `H` | `T=256` | `T=1024` | `T=4096` | `Var(mean) ~ T^?` |
|---|---|---|---|---|
| `0.8` (long memory) | `2.04×` | `2.56×` | **`3.47×`** | `−0.52` (iid `−1.00`, pure-LRD `−0.40`) |
| `0.1` (rough) | `1.09×` | `1.01×` | `1.01×` | `−0.95` |

The overconfidence ratio **grows with `T`** — more data makes the reported error bar
worse, not better. At `H=0.1` the naive SE is fine, so the effect is specifically
long memory, not roughness. (At `σ̄=0.06` the effect is present but weaker: ratios
`1.34/1.53/1.90`, slope `−0.66`.)

---

## 8. Next stage — designed, not built

Where the project stops being a demonstration. Two independent reasons the iid work
is too easy: `g` is a 1-D integral (E2/E3), and the interesting quantities are path
functionals (E4). Both are fixed by the same move.

**Put fBm in the volatility, never in the price.** Rationale to preserve:

- fBm is exactly self-similar but **Gaussian** — it supplies Mandelbrot's *Joseph*
  effect (long memory) and would *delete* the *Noah* effect (fat tails). Keep the
  jump process for Noah; add fGn to log-vol for Joseph.
- Geometric-fBm prices are **not semimartingales** and admit arbitrage (Rogers).
  Returns become forecastable, so any strategy comparison inside the model answers
  the wrong question. Price stays Brownian + jumps.
- This is the rough-volatility program (Gatheral–Jaisson–Rosenbaum). The result is
  **non-Markovian**: the state is the whole vol history, which kills quadrature,
  PDE, and transition-density methods — leaving path-space Monte Carlo natively.
  First point where sequential methods aren't decoration.

Ports of existing machinery: IS becomes a measure change on the Gaussian *driver*
(Volterra / Molchan–Golosov representation), not per-draw reweighting. The CV
generalizes by using the `H = 1/2` model as the control for `H ≠ 1/2`, differencing
the bulk exactly as `D = log_growth − r` does. Practical schemes approximate the
power-law kernel by a sum of exponentials (finite bank of mean-reverting factors).

`H` is the new sweep axis for the ignorance map, alongside `markup` and jump
intensity. Two contested battlefields to **name and route around, not adjudicate**:
Spitznagel vs AQR (Ilmanen) on whether tail hedging is cost-efficient; and rough
(`H≈0.1`) vs long-memory (`H>1/2`) volatility, where Hurst estimation is notorious
for spurious long memory from regime switches. Sweep the disputed range and report
how much the verdict moves.

Honest limitation to state in any write-up: one `H` cannot produce both small-scale
roughness and large-scale persistence. Real markets are arguably multifractal
(Mandelbrot's later program); monofractal fBm-in-vol is a tractable slice, chosen
knowingly.

---

## 9. Conventions

- Seed every RNG explicitly (`np.random.default_rng(seed)`); record seeds beside
  results. Recorded numbers used seeds `0, 1, 7, 9, 11, 20, 31`.
- Vectorize over trials; no Python loops over samples. E3 needs ~1e7+ density
  evaluations, so use the closed-form numpy density, not `scipy.stats.*.pdf`.
- Every reported estimate needs its Monte Carlo standard error alongside it.
- Any claim of the form "method X wins" must state the estimand and the alternative
  it beat, including non-sampling alternatives.
- **Write captions and docstrings that do not outrun what was established.** The
  project's thesis is claim-discipline (an honest map of what can't be known), so
  overclaiming in a figure caption is a substantive error, not a stylistic one. Two
  false findings in the exploratory phase (§5) came from exactly this.
