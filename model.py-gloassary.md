# `model.py` — variable inventory

Reconstructed by reading the file, not from a spec. Anything marked **[?]** is an
inference I could not verify from the code and you should confirm.

The single most important fact: **this file contains two unrelated models.**
They share no parameters and no call paths.

- **Model A** (sections 1–5): iid jump-diffusion + rolling put hedge. Estimand `g_hedged`.
- **Model B** (section 6): stochastic vol, fGn-driven log-volatility. Estimand `lrd_g_true`.

---

## Name collisions — read this first

These are the reason the file is hard to hold in your head.

| symbol | meaning A | meaning B | meaning C |
|---|---|---|---|
| `K` | strike price = 50.0 (§3) | exponnorm shape = 30, called "K" in the `jump_pdf` docstring, stored as `_KP` (§2) | lag index in `fgn_cholesky` (local `k`, §6) |
| `S` | bulk monthly volatility = 0.05 (§1) | — but `S0` = index level = 100.0 (§3), a *price*, not a vol | |
| `MU` | target `E[r]` = 0.005 (§1) | `M_BULK` = bulk mean = 0.0065 (§1) | `MU_A` = Model B drift = 0.01 (§6) |
| `C` / `c` | premium spend fraction = 0.0005 | passed as `c=` through `log_growth`, `control_variate`, `edge_and_p` | |

Recommendation: `K`→`STRIKE`, `S`→`SIG_BULK`, `S0`→`SPOT`, `_KP`→`EMG_SHAPE`.

---

## Model A — sections 1–5

### Primitive parameters (the true measure *f*)

| symbol | value | units | meaning |
|---|---|---|---|
| `LAM` | 1e-3 | per month | P(a jump occurs in a given month) |
| `BETA` | 1.5 | log points | mean jump size; jump is `Exp(mean=BETA)` subtracted |
| `S` | 0.05 | log points/month | volatility of the no-jump (bulk) component |
| `MU` | 0.005 | log points/month | **target** `E[r]`; pinned, not free |
| `M_BULK` | 0.0065 | log points/month | **derived**: `MU + LAM*BETA`, so `E[r] = MU` exactly |
| `_KP` | 30.0 | dimensionless | `BETA/S`; scipy `exponnorm` shape parameter |

Return model: `r = M_BULK + S*Z` w.p. `1-LAM`, `r = M_BULK + S*Z - E` w.p. `LAM`,
with `Z ~ N(0,1)`, `E ~ Exp(mean BETA)`. `r` is a **log** return, so the growth
rate is `E[r]` directly.

### Importance-sampling proposal (*q*)

| symbol | value | meaning |
|---|---|---|
| `LAM_Q` | 0.30 | jump probability under `q`; **only** parameter that differs from `f` |

Because `f` and `q` are mixtures over *identical* components, the likelihood
ratio is bounded analytically. With `u = jump_pdf/bulk_pdf ∈ [0,∞)`:

```
f/q = [(1−λ) + λu] / [(1−λ_q) + λ_q u]
```

Möbius in `u`, hence monotone, hence extremes at the endpoints:

- `sup = (1−LAM)/(1−LAM_Q) = 1.4271428…`  (at `u→0`, i.e. the bulk)
- `inf = LAM/LAM_Q       = 0.0033333…`   (at `u→∞`, i.e. deep crash)

**The docstring's `[0.0033, 1.4228]` is a sample maximum, not a bound** — it sits
0.3% below the true supremum. Replace with the closed forms.

### Hedge parameters

| symbol | value | units | meaning |
|---|---|---|---|
| `S0` | 100.0 | price | index level at the start of each month |
| `K` | 50.0 | price | put strike (50% OTM) |
| `C` | 0.0005 | fraction of wealth | monthly premium spend |
| `R_STAR` | −0.693147 | log points | `log(K/S0)`; return below which the put finishes ITM |
| `markup` | 1.0 | multiple | price paid ÷ actuarially fair price. **1.0 = no variance risk premium** |

### Quadrature controls

| symbol | value | meaning |
|---|---|---|
| `LO`, `HI` | −40.0, 2.0 | integration limits in `r`-space |
| `QUAD_LIMIT` | 1500 | scipy subdivision cap |

Caveat from the source, worth preserving: adaptive `quad` over this window
resolves the width-0.05 bulk only because the jump density guides subdivision.
Fragile at other parameters. Verified by tests T1 and B4, not assumed.

### `truth()` output keys

| key | value at defaults | status |
|---|---|---|
| `g_unhedged` | +0.005000 | **exact** (pinned by construction) |
| `g_hedged` | +0.005852 | quadrature |
| `g_hedged_quaderr` | — | scipy's own error estimate |
| `edge` | +0.000852 | `g_hedged − MU` |
| `p_itm` | 6.276e-4 | P(put finishes ITM in one month) |
| `p_jump` | 1.000e-3 | `= LAM`. **A different number from `p_itm`** (bug B5) |
| `price` | 0.018828 | fair put price × markup |

### Estimators (§5) — all unbiased for `g_hedged`

| function | formula | measured ratio vs crude |
|---|---|---|
| `est_crude` | `mean(log_growth(x))`, `x ~ f` | 1.00× |
| `est_cv` | `mean(D(x)) + MU`, `x ~ f` | **0.50×** (docstring says ~0.6×) |
| `est_is` | `mean(log_growth(y)·w) `, `y ~ q` | **0.67×** (docstring says ~0.8×) |
| `est_cv_is` | `mean(D(y)·w) + MU`, `y ~ q` | **151×** (docstring says ~190×; a figure says 163.1×) |

`D(r) = control_variate(r) = log_growth(r) − r`, the hedge's *effect* on growth.
`w = pdf(y, LAM)/pdf(y, LAM_Q)`, the likelihood ratio.

Measured at `n=5000`, `400` trials, seed 11. The chi-square relative standard
error on a variance ratio from 400 trials is ≈7%, which reconciles 151 with 163
but **not** with 190. Something beyond trial noise differs between those runs —
find it before quoting any of the three.

---

## Model B — section 6 (fGn)

Shares **nothing** with Model A. Different drift, different vol, unstated time unit.

| symbol | value | meaning |
|---|---|---|
| `MU_A` | 0.01 | drift. **[?]** time unit unstated — monthly? annual? |
| `SBAR` | 0.15 | baseline volatility level |
| `XI` | 0.9 | log-vol amplitude (vol-of-vol) |
| `hurst` | argument | Hurst exponent H |
| `t_max` | argument | path length |

Dynamics: `r_t = MU_A − σ_t²/2 + σ_t·Z_t`, `log σ_t = log(SBAR) + XI·G_t`,
`G` = fGn with `Var(G_t)=1`. Returns stay serially uncorrelated; long memory
enters growth only through the variance drag.

**Flag at defaults:** `lrd_g_true() = −0.0469`. `E[σ²] = 0.1137`, so `E[σ] ≈ 0.34`.
`XI = 0.9` is a very large vol-of-vol, and the model's growth rate is strongly
negative. Confirm this is intended and confirm the time unit.

### `fgn_cholesky(hurst, t_max)`

Returns the Cholesky factor `L` of the fGn covariance. `L @ z` gives increments;
`cumsum` for the path.

Autocovariance at lag `k`, unit variance:
`γ(k) = 0.5·(|k+1|^2H − 2|k|^2H + |k−1|^2H)`, with `γ(0) = 1` ✓, `γ(1) = 0` at H=½ ✓.

Useful property, worth keeping: because `L` is lower-triangular, a **prefix of a
path is a valid shorter path**, so nested `T` values reuse one set of draws.
That is ideal for checking the scaling identity `MDD[0,aT] =ᵈ a^H · MDD[0,T]` —
but note the nested estimates are then *correlated*, so any error bar on the
ratio must account for it rather than treating the two T's as independent.

**Two problems:**
1. **No `lru_cache`**, unlike `put_price` and `truth`. Measured cost: 0.06 s at
   t_max=512, 0.26 s at 1024, 1.03 s at 2048 (O(t³), O(t²) memory: ~67 MB at 2048).
   If this is ever called inside a path loop it is the entire runtime.
2. **The `+ 1e-10·eye` ridge silently perturbs the covariance** instead of
   checking positive-definiteness. Same species of fix as the `1.4228` bound.
   This bites hardest at extreme H — exactly the regime of interest. Either
   assert on the minimum eigenvalue or move to circulant embedding.

---

## What survives a pivot to drawdown-vs-H

`fgn_cholesky` — 23 lines. That is genuinely all. Model A is a different
estimand; Model B has fGn driving *volatility* with a growth-rate estimand,
whereas drawdown-vs-H needs fBm driving the *path* plus a max-drawdown
functional. Neither is a foundation.

## Results tiering — what can be frozen as a standard

**Exact, freeze permanently.** `E[r] = MU = 0.005`; `M_BULK = 0.0065`; the
exponnorm parameterization (verified against direct simulation); `f/q ∈
[0.0033333, 1.4271428]`; `γ(0) = 1`; the `lrd_g_true` closed form.

**Deterministic, freeze as reproduce-to-tolerance.** `price = 0.018828`,
`g_hedged = 0.005852`, `edge = 0.000852`, `p_itm = 6.276e-4`. Reran; stable.

**Do not freeze.** Every variance-reduction ratio. None currently carries a
standard error, the three quoted values disagree by more than trial noise
explains, and freezing one as "standard" would be freezing a number that cannot
presently be reproduced.
