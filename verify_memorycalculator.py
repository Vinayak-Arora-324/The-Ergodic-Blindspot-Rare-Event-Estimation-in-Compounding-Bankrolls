"""Verify the MemoryCalculator estimator fix against known-H ground truth.

Two independent fGn generators are used:

  1. model.fgn_cholesky  -- the project's own exact Cholesky simulator.
     Convenient, but circular: it shares a codebase (and could share a
     convention error) with the estimators under test.

  2. davies_harte        -- circulant-embedding simulator implemented here
     from the autocovariance definition, independent of model.py.  If both
     agree with the target H, a shared-convention artefact is ruled out.
"""

import sys
import warnings

import numpy as np

import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model
from MemoryCalculator import (
    INCREMENTS,
    PROFILE,
    higuchi_fd,
    hurst_dfa,
    hurst_variogram,
    mfdfa,
)


_CHOL_CACHE = {}


def fgn_cholesky_sample(H, n, rng):
    # O(n^3) factorization; identical for every seed at fixed (H, n).
    if (H, n) not in _CHOL_CACHE:
        _CHOL_CACHE[(H, n)] = model.fgn_cholesky(H, n)
    return _CHOL_CACHE[(H, n)] @ rng.standard_normal(n)


def davies_harte(H, n, rng):
    """Exact fGn via circulant embedding -- independent of model.py."""
    k = np.arange(n)
    g = 0.5 * ((k + 1.0) ** (2 * H) - 2.0 * k ** (2 * H) + np.abs(k - 1.0) ** (2 * H))
    # Build the circulant first row and take its (real, non-negative) spectrum.
    row = np.concatenate([g, [0.0], g[:0:-1]])
    lam = np.fft.fft(row).real
    lam = np.maximum(lam, 0.0)
    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.fft(np.sqrt(lam / (2 * m)) * z)
    return w.real[:n]


N = 2048
SEEDS = 12
H_GRID = [0.30, 0.50, 0.75, 0.90]

rows = []
for H in H_GRID:
    for gen_name, gen in (("cholesky", fgn_cholesky_sample), ("davies-harte", davies_harte)):
        vg, dfa, fd, h2 = [], [], [], []
        vg_bug, dfa_bug = [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 * s + int(H * 100))
            fgn = gen(H, N, rng)

            # --- fixed: increments in, estimator integrates internally -----
            vg.append(hurst_variogram(fgn, input_type=INCREMENTS))
            dfa.append(hurst_dfa(fgn, input_type=INCREMENTS))
            fd.append(higuchi_fd(fgn, input_type=INCREMENTS))
            h2.append(mfdfa(fgn, q_list=[2], input_type=INCREMENTS)[2])

            # --- reproduce the old behaviour: no integration at all --------
            vg_bug.append(hurst_variogram(fgn, input_type=PROFILE))
            dfa_bug.append(hurst_dfa(fgn, input_type=PROFILE))

        rows.append(
            dict(
                H=H,
                gen=gen_name,
                vg=np.mean(vg), vg_sd=np.std(vg),
                dfa=np.mean(dfa), dfa_sd=np.std(dfa),
                h2=np.mean(h2),
                fd=np.mean(fd),
                vg_bug=np.mean(vg_bug),
                dfa_bug=np.mean(dfa_bug),
            )
        )

hdr = (
    f"{'true H':>7} {'generator':>13} | {'variogram':>16} {'DFA':>16} "
    f"{'MFDFA h(2)':>11} {'Higuchi 2-FD':>13} | {'OLD vg':>7} {'OLD dfa':>8}"
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['H']:>7.2f} {r['gen']:>13} | "
        f"{r['vg']:>8.3f}+-{r['vg_sd']:<6.3f} "
        f"{r['dfa']:>8.3f}+-{r['dfa_sd']:<6.3f} "
        f"{r['h2']:>11.3f} {2 - r['fd']:>13.3f} | "
        f"{r['vg_bug']:>7.3f} {r['dfa_bug']:>8.3f}"
    )

print("\nMax |H_hat - H| over the grid (fixed estimators):")
for key, label in (("vg", "variogram"), ("dfa", "DFA"), ("h2", "MFDFA h(2)")):
    err = max(abs(r[key] - r["H"]) for r in rows)
    print(f"  {label:>12}: {err:.3f}")

# --- profile input path: passing prices directly must agree with increments
rng = np.random.default_rng(7)
fgn = davies_harte(0.75, N, rng)
prof = np.cumsum(fgn - fgn.mean())
print("\nContract check (H=0.75), same data both ways:")
print(f"  increments -> {hurst_dfa(fgn, input_type=INCREMENTS):.4f}")
print(f"  profile    -> {hurst_dfa(prof, input_type=PROFILE):.4f}")

# --- deprecated alias still works and warns
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from MemoryCalculator import hurst_rs

    val = hurst_rs(fgn)
    print(f"\nhurst_rs alias -> {val:.4f}, warned: {any(issubclass(x.category, DeprecationWarning) for x in w)}")

# --- predict_direction end to end on a persistent series
import pandas as pd

from MemoryCalculator import predict_direction

prices = pd.Series(np.exp(np.cumsum(davies_harte(0.75, 1000, np.random.default_rng(3)) * 0.01)) * 100)
H, direction = predict_direction(prices, window=400)
print(f"predict_direction on true-H=0.75 series -> H={H:.3f}, {direction}")
