# Code review — `test_regressions.py`

Reviewed against `model.py` and `experiments.py`. Every finding below was verified by
running the code (numpy 2.2.6 / scipy 1.15.3). Baseline: **12/12 pass, ~2.3 s wall**,
both standalone and under pytest. Nothing here is broken today.

The file does something most test suites don't: it encodes *why* each number is the
number, and it treats retracted claims as first-class test subjects. That's the right
instinct and it should survive the edits below. The substantive criticism is not about
what's tested but about what isn't — the project's headline finding (the variance
ordering) has no test at all, and neither does any of section 6 of `model.py`.

---

## 1. Correctness and latent bugs

### 1.1 `main()` swallows only `AssertionError`, so any real error aborts the suite

```python
except AssertionError as exc:
```

A `ValueError`, `LinAlgError`, or an `ImportError` inside a test propagates out of
`main()`, kills the run, and skips every remaining test. No summary line, no list of
what passed. Verified by injecting a `ValueError` test: the traceback replaced the
`12/12 passed` line entirely.

This matters more here than in a normal suite, because several of these tests exercise
numerics that fail by raising rather than by asserting — `np.linalg.cholesky` raises
`LinAlgError`, `integrate.quad` raises on a non-finite integrand, and a shape mismatch
in `_crash_share` raises `IndexError`. Those are exactly the regressions you want
reported as failures, not as a crash.

```python
except AssertionError as exc:
    failures.append(name)
    print(f"  FAIL  {label:4s} {name[len(label) + 6:]:42s} {exc}")
except Exception as exc:                    # noqa: BLE001
    failures.append(name)
    print(f"  ERROR {label:4s} {name[len(label) + 6:]:42s} "
          f"{type(exc).__name__}: {exc}")
```

### 1.2 B4's docstring number contradicts B4's own output

Docstring:

> An earlier stage integrated over [-6, 6] and cut 0.47% of the value.

Actual output of the test:

```
a lo=-6 window would cut 0.09%
```

The test recomputes the truncation loss with `HI = 2.0`, not `6.0`, so it isn't measuring
the window the docstring describes. Either fix the docstring to say `[-6, 2]` and `0.09%`,
or make the demonstration match the retired window:

```python
tight, _ = integrate.quad(
    lambda r: M.log_growth(r) * M.pdf(r), -6.0, 6.0, limit=M.QUAD_LIMIT
)
```

A regression file whose stated number disagrees with its printed number undermines the
one thing it's for.

### 1.3 T6's docstring states a tolerance the test does not enforce

> T6: all three estimators are unbiased for g_hedged, within 1.4 MC-SE.

The gate is `assert abs(z) < 3.0`. The `1.4` is the *observed* z for `cv_is` at the fixed
seed, not the tolerance. A reader auditing coverage will believe the test is twice as
tight as it is. Say what's enforced:

> Unbiased within 3 MC-SE; observed at seed 20 is +1.0 / -0.7 / +1.4.

### 1.4 T5 pins an order statistic at half its tolerance budget

```python
assert abs(lo - 0.0033) < 5e-4 and abs(hi - 1.4228) < 5e-4, (lo, hi)
```

`hi` is the max of 200k draws, so it's a wobbly order statistic, not a converging
quantity. Measured:

| n       | `hi`      | distance from 1.4228 |
|---------|-----------|----------------------|
| 100 000 | 1.4227031 | 9.7e-5               |
| 200 000 | 1.4225494 | **2.5e-4**           |
| 400 000 | 1.4227850 | 1.5e-5               |

At the n the test actually uses, half the 5e-4 allowance is consumed by sampling noise
alone, and it isn't monotone in n (each n draws a different stream), so it can't be
tightened by sampling harder. The literal is also silently coupled to the hardcoded
`200_000` in the call.

Separately, `lo` is *exactly* `LAM / LAM_Q = 0.0033333...`, not approximately — because
`bulk_pdf` underflows to hard zero for a deep crash draw, leaving `f/q` as an exact ratio
of jump components. So `abs(lo - 0.0033) < 5e-4` is a float-underflow assertion wearing
statistical clothing, and the `lo >= bound_lo * (1 - 1e-9)` line above already covers it.

The analytic-bound asserts are the real content. Replace the literals with a statement of
what you actually want — that the sampler gets close enough to the bulk bound to show the
weights are bounded and tight:

```python
assert lo == bound_lo or lo >= bound_lo * (1 - 1e-9), (lo, bound_lo)
assert hi <= bound_hi * (1 + 1e-9), (hi, bound_hi)
assert hi > 0.99 * bound_hi, (hi, bound_hi)   # bulk is sampled densely enough
```

---

## 2. Coverage gaps

These are the biggest items in the review. Each corresponds to a claim `model.py` states
as verified and nothing checks.

### 2.1 The variance ordering — the headline finding — is untested

`model.py` section 5 records this in a comment and adds *"code downstream must not try to
'fix' it"*:

> `cv` alone INCREASES variance (~0.6x) … IS alone also FAILS (~0.8x) … `cv_is` WORKS (~190x)

T6 pins bias. T7 pins the convergence *rate*. Nothing pins the variance. This is the one
result the project is actually about, and it's the one an innocent-looking "optimization"
of `control_variate` or `LAM_Q` would silently destroy — with T6 and T7 still green,
because both bias and root-n rate survive a bad `LAM_Q`.

Measured now (seed 20, n=4096, 2000 trials), variance ratio vs crude:

```
+ control variate            0.52x
IS only (est_is)             0.74x
+ CV + importance sampling  167.12x
```

Note `167x`, against the recorded `190x`. Either the recorded figure was measured at
different settings or it has drifted; a test would have told you which. Suggested T8:

```python
def test_t8_variance_ordering(n=4096, trials=2000):
    """T8: cv alone and IS alone LOSE; only the composition wins. Verified
    0.52x / 0.74x / 167x vs crude at n=4096."""
    rng = np.random.default_rng(20)
    sd = {}
    for name, fn in [("crude", M.est_crude), ("cv", M.est_cv),
                     ("is", M.est_is), ("cv_is", M.est_cv_is)]:
        sd[name] = M.run_trials(fn, n, trials, rng).std(ddof=1)
    ratio = {k: (sd["crude"] / v) ** 2 for k, v in sd.items()}
    assert ratio["cv"] < 0.8, ratio       # CV alone is WORSE -- do not "fix"
    assert ratio["is"] < 0.9, ratio       # IS alone is WORSE
    assert ratio["cv_is"] > 100, ratio    # only the composition wins
    return "var ratio vs crude: " + ", ".join(
        f"{k} {v:.2f}x" for k, v in ratio.items() if k != "crude")
```

### 2.2 `est_is` is never called by any test

It exists solely as the exhibit for "IS alone fails" and it isn't in `ESTIMATORS`, so T6
and T7 skip it. It could break outright and the suite would stay green. T8 above fixes
this incidentally.

### 2.3 Section 6 of `model.py` (fGn / E5) has zero coverage

`fgn_cholesky`, `lrd_g_true`, and the `MU_A / SBAR / XI` block are entirely untested.
E5's whole argument is that the long-memory driver is *exact*, so an approximation
creeping in there would put the artefact under suspicion — which is precisely the failure
mode the rest of this file is built to prevent. Three cheap checks, all verified to pass:

```python
def test_t9_fgn_is_exact():
    """T9: the Cholesky factor reproduces the fGn autocovariance exactly, and
    H = 0.5 degenerates to white noise."""
    h, t_max = 0.75, 64
    L = M.fgn_cholesky(h, t_max)
    k = np.arange(t_max, dtype=float)
    gamma = 0.5 * ((k + 1) ** (2 * h) - 2 * k ** (2 * h) + np.abs(k - 1) ** (2 * h))
    idx = np.arange(t_max)
    err = np.abs(L @ L.T - gamma[np.abs(np.subtract.outer(idx, idx))]).max()
    assert err < 1e-9, err            # 1e-10 ridge is the only discrepancy
    assert abs(gamma[0] - 1.0) < 1e-12
    half = M.fgn_cholesky(0.5, 16)
    assert np.abs(half - np.eye(16) * half[0, 0]).max() < 1e-5   # H=.5 -> iid
    return f"fGn covariance exact to {err:.1e}; H=0.5 gives white noise"


def test_t10_lrd_growth_closed_form():
    """T10: lrd_g_true matches the simulated mean of r_t."""
    ...  # mu_a - 0.5*sbar^2*exp(2*xi^2) = -0.046847
```

Measured: `max |LLᵀ - Γ| = 1.0e-10` (the ridge), `lrd_g_true() = -0.046847`.

### 2.4 The load-bearing `E[r] = MU` pinning is asserted, not verified

T2 has:

```python
assert abs(t["g_unhedged"] - 0.005) < 1e-12       # exact by construction
```

`t["g_unhedged"]` is the literal `MU`, so this compares a constant to itself. It pins the
spec, which is fine, but it does *not* check the identity `M_BULK = MU + LAM*BETA ⟹
E[r] = MU` — the thing `model.py` calls "load-bearing" and on which the entire
control-variate estimator rests. Get `S`, `BETA`, or the sampler's sign wrong and this
test stays green while `est_cv` becomes biased. One line, verified:

```python
mean_r, _ = integrate.quad(lambda r: r * M.pdf(r), M.LO, M.HI, limit=M.QUAD_LIMIT)
assert abs(mean_r - M.MU) < 1e-9, mean_r        # measured residual 1.1e-13
```

### 2.5 T3 and T4 regenerate the same 2M-sample array

Same seed, same `n`, same `crash` mask, computed twice. They're two facts about one
draw. Merge them, or hoist to a small cached helper:

```python
@lru_cache(maxsize=1)
def _crash_sample(n=2_000_000, seed=11):
    x = M.sample_returns(np.random.default_rng(seed), n)
    return x, x < M.R_STAR
```

---

## 3. B1's source scan is both over- and under-inclusive

The guard is a substring sweep over `*.py` for `("standard_t", "stats.t.", "nu - 2",
"nu-2")`. Four problems, in increasing order of how much they'd cost you:

**Under-inclusive — the ban is trivial to reintroduce accidentally.** None of these are
caught:

```python
from scipy.stats import t
scale = sigma / np.sqrt(df / (df - 2))     # renamed variable
rv = stats.nct(...)                         # noncentral t
```

The markers encode one *spelling* of the bug, not the bug. If the scan is meant to be
load-bearing, match the concept — `\bt\.(pdf|rvs|cdf)\b`, `stats.t\b`, `standard_t`,
and `/\s*\(?\s*\w+\s*-\s*2` — or accept that it's a reminder and say so in the docstring.

**Not word-bounded.** Verified false positives: `"nu - 2"` matches `menu - 2`; `"nu-2"`
matches `manu-2`. Contrived, but a `for menu-2 in ...` style name in a future file fails
a test that then reads "Student-t model reappeared", which is a confusing thing to debug.

**Non-recursive glob.** `src_dir.glob("*.py")` doesn't descend. The day anything moves
into a subpackage, the guard silently stops guarding it and still reports PASS. Use
`rglob` with an explicit exclusion for `.venv` and `.old-files`, or —

**Better: allowlist the modules of record instead of globbing.** The scan currently reads
`Calculation.py` and `MemoryCalculator.py`, which aren't part of the model pipeline, and
any scratch file dropped in the folder becomes a test input. The set you actually care
about is small and known:

```python
GUARDED = ("model.py", "experiments.py")
for name in GUARDED:
    path = src_dir / name
    assert path.exists(), f"{name} is missing -- B1's guard is not running"
    ...
```

The existence assert matters: as written, if `model.py` were renamed the loop would
iterate over nothing and B1 would pass vacuously.

---

## 4. pytest compatibility

The module docstring promises `pytest test_regressions.py` works. It does — verified,
**12 passed, 12 warnings, 2.33 s** — but every test trips
`PytestReturnNotNoneWarning` because they all `return` a detail string:

```
PytestReturnNotNoneWarning: Test functions should return None, but
test_regressions.py::test_b2_premium_must_scale_with_p returned <class 'str'>.
```

Two consequences. First, under the very common CI setting `filterwarnings = error`, all
12 fail. Second, pytest discards the detail strings, so the informative output — which is
the best feature of this file — exists only in the standalone runner.

The return-value design is worth keeping; just declare it. Add to `pyproject.toml` or
`pytest.ini`:

```ini
[pytest]
filterwarnings = ignore::pytest.PytestReturnNotNoneWarning
```

and add a line to the module docstring saying the returns are deliberate and carry the
detail line for the standalone runner.

Relatedly, `requirements.txt` currently lists `__future__`, `sys`, `time`, `model`, and
`functools` as installable packages, so `pip install -r requirements.txt` fails outright.
pytest isn't listed either, and isn't in `.venv` — the pytest path advertised in the
docstring can't be exercised as the repo ships.

---

## 5. Minor

**5.1** `stats` is imported and never used (ruff `F401`). `from scipy import integrate`.

**5.2** `retired = lambda nu: ...` in B1 (`E731`) — make it a `def`; it's the retired
convention and deserves a name and a docstring more than most.

**5.3** The runner's label slicing, `name[len(label) + 6:]`, silently assumes a two-
character label. `test_t10_...` would print `0_...`. If T8–T10 above go in, this breaks
the moment you reach T10.

**5.4** `{:38s}` is too narrow: `jump_probability_is_not_crash_probability` is 41 chars,
so B5's detail column is visibly out of alignment in the current output. Use 42.

**5.5** Discovery is `sorted(globals().items())`, so B1–B5 run *before* T1–T7, inverting
the order the module docstring presents them in. Sort with a key that puts `t` first, or
reorder the docstring.

**5.6** `_crash_share` divides by `np.var(v)` and indexes `v[crash]` with no guard for an
empty mask. At n=2M there are 1261 crashes so it's fine today, but it's a general-looking
helper one `n=10_000` away from a silent `nan` (`np.var` of an empty slice warns and
returns `nan`, and `nan < 0.05` is `False`, so it'd fail with an uninformative message).
One line: `assert crash.sum() > 100, crash.sum()`.

**5.7** B4 calls `M.truth(1.0, -40.0)`, which is a different `lru_cache` key from
`M.truth()` despite being the identical computation — one redundant quadrature. Costs
milliseconds; mentioned only because the cache is load-bearing elsewhere
(`experiments.py` clears it deliberately).

---

## What's right

Worth stating explicitly, because these are the parts an edit should not casually undo:

- **T1 is correctly identified as the most important test.** Sampler and density are
  written independently, every estimator mixes them, and a mismatch corrupts IS weights
  silently. Cross-checking them is the right foundation and the 4-MC-SE framing is honest.
- **Tolerances are derived, not chosen.** B4 sets its tolerance from `quad`'s own reported
  error (1.3e-8) rather than from ambition; T1 uses the binomial MC-SE; T7 states why
  ±0.1 is the honest number at 400 trials. This is unusual and good.
- **T5 asserts the analytic bound, not just the empirical value.** The point is that
  finite IS variance holds *by construction* — asserting `f/q ∈ [LAM/LAM_Q,
  (1-LAM)/(1-LAM_Q)]` proves that; asserting the observed min/max wouldn't.
- **The B-series carries the retraction history in prose.** B1 and B2 both explain the
  false claim that was published off the bug. A bare assertion would not stop someone
  re-deriving the same error from the same reasoning, and the docstrings say so.
- **T3's assertion direction is the interesting one.** It pins a finding that *reversed*
  the project's founding premise, and the `share_u > 10 * share_h` line makes the
  reversal itself the thing under test rather than two loose numbers.

---

## Suggested order

1. §1.1 — `main()` catching only `AssertionError`. One-line fix, largest downside if left.
2. §2.1 / §2.2 — add T8, the variance ordering. This is the headline result and it is
   currently unprotected.
3. §1.2 / §1.3 — reconcile B4's `0.47%` and T6's `1.4 MC-SE` with what the code does.
4. §2.4 — one-line quadrature check on `E[r] = MU`.
5. §1.4 — de-brittle T5's hardcoded `hi`.
6. §3 — narrow B1's scan to an allowlist, with an existence assert.
7. §2.3 — fGn coverage (T9/T10), before E5's numbers get quoted anywhere further.
8. §4 — the pytest warning filter and `requirements.txt`.
9. §5 — minor items; §5.3 becomes mandatory if you add T10.
