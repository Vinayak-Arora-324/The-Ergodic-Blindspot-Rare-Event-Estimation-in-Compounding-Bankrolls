# Code review — `experiments.py`

Reviewed against `model.py` and `test_regressions.py`. Findings verified by running the
code (numpy 2.2.6 / scipy 1.15.3) unless marked otherwise.

The file is in good shape. The numerics are careful, the vectorization is right, and the
docstrings do real work — several of them argue for a design choice rather than describe
it, which is rarer than it should be. Most of what follows is either a latent bug that
only fires off the default path, or a place where the code quietly violates a convention
the module's own docstring commits to.

---

## 1. Correctness and latent bugs

### 1.1 `e5`: `ts` is never clipped to `t_max` — silent mislabeling

`ts_fig` is filtered (`if t <= t_max`), but the printed table walks `ts` unfiltered:

```python
for tt in ts:
    block = r[:tt]          # r has t_max rows
```

Call `e5(t_max=1024)` and the loop still prints a `T = 4096` row, computed from 1024
observations. NumPy slices past the end without complaining. Verified:

```
r[:4096].shape when t_max=256 -> (256, 5)   # printed as T = 4096
```

This is the worst kind of bug for this project: it produces a plausible number under a
false label. Fix at the top of `e5`:

```python
ts = tuple(t for t in ts if t <= t_max)
if not ts:
    raise ValueError(f"no T in ts is <= t_max={t_max}")
```

### 1.2 `e2`: the deflation timing understates the cold quadrature cost by ~2x

```python
M.truth.cache_clear()
M.truth()
```

`truth()` calls `log_growth()`, which calls `put_price()` — and `put_price` carries its
own `lru_cache` that is *not* cleared. So the timed call reuses a put price computed
earlier and skips one full `integrate.quad`. Measured:

```
truth cache only cleared : 4.4 ms   <- what e2 prints
truth + put_price cleared: 8.1 ms   <- honest cold cost
```

That number appears in prose *and* in the E2 figure footer ("returns this same answer in
{ms:.0f} ms"). Add `M.put_price.cache_clear()` alongside it.

### 1.3 `e2`: `sds["crude MC"]` inside the loop hard-codes the `ESTIMATORS` ordering

```python
for name, fn, _ in M.ESTIMATORS:
    ...
    ratio = (sds["crude MC"] / sd) ** 2      # only defined if crude went first
```

Reordering `M.ESTIMATORS` raises `KeyError: 'crude MC'` halfway through printing the
table (verified). `_e2_figure` has the same dependency. The coupling is invisible from
either file. Either compute ratios after the loop, or name the baseline explicitly:

```python
BASELINE = "crude MC"
...
ratio = (sds[BASELINE] / sd) ** 2 if BASELINE in sds else float("nan")
```

### 1.4 CLI: `all` is only honored as the sole argument

```python
if names == ["all"]:
```

`python experiments.py e1 all` prints `unknown experiment(s): all` (verified). One line:

```python
if "all" in names:
    names = list(EXPERIMENTS)
```

---

## 2. Violations of the file's own stated conventions

The header commits to four rules. Two are not held everywhere, and given the project's
position that overclaiming is a substantive error, these are worth treating as findings
rather than nits.

### 2.1 "every figure carries the fair-pricing caveat in its footer"

E1's figure does not, and `e1()` never calls `_fair_pricing_note()` (verified: zero
occurrences of "caveat" in `e1 --no-plots` output). The standing caveat at the top of the
file says it applies to "every E1–E4 number".

Substantively I think E1 is *fine* without it: its estimand is `E[put payoff]`, which is
markup-free, so no "hedge wins" reading is available to caveat. But then the docstring is
what's wrong. Pick one — either add the footer, or narrow the convention to E2–E4 with
the one-line reason. Right now the code and its stated rule disagree, which is exactly
the failure mode the header is trying to prevent.

### 2.2 "every reported estimate carries its Monte Carlo standard error"

Several headline numbers don't:

| Number | Where | Why it matters |
|---|---|---|
| crash share of `Var(r)` / `Var(lg)` | `e2`, and 11.5pt bold in figure panel A | rests on ~1,259 crash draws out of 2e6 |
| variance-reduction ratios (`163x`) | `e2` table and figure panel C | ratio of two sample variances, 2,000 trials |
| `slope` and `n_star` | `e3` | `polyfit` output, no residual reported; `n_star` extrapolates |
| `true_se / naive_se` ratio | `e5` table and figure panel B | the headline of the experiment |
| `Var(mean) ~ T^slope` | `e5` | see 2.3 |

The crash share is the one I'd fix first. Across six independent seeds at `n_diag = 2e6`
the unhedged share ranges **61.5% – 65.1%** (sd ≈ 1.2pp) — so "64%" set in bold with no
interval is doing more work than the sample supports. The cheap fix is a bootstrap over
the crash indices; the cheaper one is raising `n_diag` (it's ~1s of runtime).

### 2.3 `e5`: the reported scaling exponent is fit from two points

```python
(t1, s1), (t2, s2) = rows[-2], rows[-1]
slope = 2 * np.log(s2 / s1) / np.log(t2 / t1)
```

A two-point slope has no residual, no standard error, and is maximally sensitive to noise
in either endpoint. It is then compared against a theoretical value (`2H-2`) in print,
which invites the reader to take the comparison seriously.

`curves[h]` already holds eight points of exactly this curve, computed a few lines below
for the figure and essentially free. Regress over the large-`T` tail of those instead and
report the fit's standard error. (The guard restricting the `2H-2` reference to `H > 1/2`
is correct and well-reasoned — keep it.)

---

## 3. Duplication and structure

### 3.1 `crash_share` exists twice

Defined as a closure inside `e2`, and again as `_crash_share` in `test_regressions.py`.
These are the same formula written out independently. `model.py`'s docstring says the
single-module refactor exists because "the exploratory phase duplicated the model across
five scripts with divergent parameterizations, which is exactly how the `exponnorm` bug
(B3) survived undetected." This is that pattern, at small scale: T3 and E2 can drift apart
and nothing would notice. Move it to `model.py`.

### 3.2 `_e2_figure` recomputes what `e2` already computed

`e2` passes the `crash_share` *closure* into the figure function, which calls it twice
more over 2M-element arrays whose shares were already printed. Pass the two floats.

### 3.3 `_save` calls `_mpl()` just to reach `plt.close`

Re-runs `plt.rcParams.update(_RC)` on every save. Harmless, but `import matplotlib.pyplot
as plt` locally (the import is already cached) or threading `plt` through is clearer about
intent.

---

## 4. Parameter fragility and drifting recorded numbers

### 4.1 `n_afford = 1000` is hard-coded against a comment that asserts a derived fact

```python
n_afford = 1000   # "N*p ~ 0.63 here"
```

Currently `n_afford * p = 0.628`, so the comment is true today. Move any model parameter
and it becomes silently false while E1's entire framing ("a budget where the run expects
fewer than one crash") continues to be printed. Derive it, or assert it:

```python
n_afford = int(round(0.63 / p))          # or: assert 0.4 < n_afford * p < 0.9
```

### 4.2 Recorded prose numbers have drifted and nothing checks them

`model.py` states "~190x" and IS weights "[0.0033, 1.4228]"; `_e2_figure`'s docstring
quotes "190x" again. This run printed **163.1x** and **[0.0033, 1.4227]**. The gap is
probably library-version noise rather than a regression, but the point stands: exact
figures are quoted in five places and pinned in none. Either add them to
`test_regressions.py` with an explicit tolerance, or stop quoting exact values in comments
and say "two orders of magnitude".

### 4.3 `M.truth.cache_clear()` is a global side effect inside a results function

Harmless because `truth()` is deterministic, but a timing measurement that mutates module
state is worth isolating in a `_time_cold_quadrature()` helper rather than sitting
mid-experiment.

---

## 5. Minor

- `from __future__ import annotations` with no annotations anywhere in the file.
- `zeros = lambda: np.zeros(paths)` and `se = lambda v: ...` in `e4` — PEP 8 E731; `def`
  also gives the frame a name in a traceback.
- `e3` recomputes the least-squares intercept by hand
  (`a = np.exp(np.log(v).mean() - slope*np.log(ns).mean())`) when `np.polyfit` already
  returns it: `slope, log_a = np.polyfit(np.log(ns), np.log(v), 1)`.
- `_fair_pricing_note()` ordering is inconsistent: `e2`/`e4` call it before plotting, `e3`
  after — so in `e3` the `saved …` line lands between the numbers and their caveat.
- `e4` uses `M.MU + edge` for the `g_h` reference line where `t["g_hedged"]` is already in
  scope and says what it means.
- `(ests == 0).mean()` is exact float equality. It is correct here (`np.maximum(..., 0.0)`
  returns exact zeros, and their mean is exactly 0.0), but it reads like a bug and would
  break silently if a nonzero floor were ever introduced. One line of comment.
- `fgn_cholesky` is uncached, so a second call to `e5` in one process repays O(t³). An
  `@lru_cache` in `model.py` would make the E5 figure/table split free.

---

## 6. Testing

`test_regressions.py` imports `model` only — `experiments.py` has no coverage at all. A
smoke test costing a few seconds would have caught 1.1 and 1.4:

```python
def test_e_smoke():
    X.PLOTS = False
    X.e1(trials=20); X.e2(n=200, trials=20, n_diag=50_000)
    X.e3(n_panel_a=128, trials_a=20, trials_b=10)
    X.e4(horizon=6, paths=2_000, n_track=100)
    X.e5(t_max=256, paths=40, ts=(64, 256))     # fails today — see 1.1
    assert X.main(["experiments.py", "e1", "all", "--no-plots"]) == 0  # fails — see 1.4
```

Note that both failing lines are cases where a *caller passes non-default arguments* —
which is the whole reason those arguments are exposed.

---

## What's right

Worth saying, because these are places the code is doing something subtle and getting it
right:

- **Common random numbers in `e4`** are implemented correctly — one `r` per month drives
  both bankrolls, so the difference isn't swamped by market noise.
- **The month-major loop in `e4`** with running state is the right call, and the `n_track`
  carve-out for the fan chart is justified rather than assumed.
- **`e5`'s autocorrelation check** — `np.corrcoef(r[:-lag].ravel(), r[lag:].ravel())` on a
  `(T, paths)` array is correct: C-order ravel preserves the `(t, p) → (t+lag, p)` pairing.
  This is easy to get backwards and it isn't.
- **The Cholesky prefix trick** (a prefix of an exact fGn path is a valid shorter path) is
  both correct and a real saving.
- **`e3`'s cross-validation** of the empirical `n_star` against E2's analytic `(2·sd/edge)²`
  is a genuine independent route to the same number, not a restatement.
- **Every log axis in the file is argued for** in the docstring rather than defaulted to,
  and in each case the argument is the right one (a linear axis would erase the finding).

---

## Suggested order

1. §1.1 (`e5` `ts` clipping) — silent wrong label, one line.
2. §1.2 (`put_price` cache) — a printed number is wrong by ~2x, one line.
3. §1.4 (CLI `all`) — one line.
4. §2.1 (E1 caveat: fix the code or the docstring) — decide which.
5. §2.2 crash-share SE — the number carries a bold callout it can't currently support.
6. §2.3 (`e5` two-point slope) — data for the better fit already exists.
7. §6 smoke test — locks 1–3 down.
8. §1.3, §3.1, §4.1, §4.2 — coupling and drift; no wrong numbers today.
