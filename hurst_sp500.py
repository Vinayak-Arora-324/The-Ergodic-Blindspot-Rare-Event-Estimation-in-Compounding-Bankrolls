"""Estimate the Hurst exponent of the S&P 500 with MemoryCalculator's estimators.

    python hurst_sp500.py                      # ^GSPC, daily, 1990-present
    python hurst_sp500.py --ticker ^VIX
    python hurst_sp500.py --start 2000-01-01 --interval 1wk
    python hurst_sp500.py --no-download        # use the cached CSV only

Why this is not just "print H"
------------------------------
The project README already flags that "Hurst estimation is notorious for
spurious long memory."  A point estimate of H = 0.54 on real data means
nothing on its own, because these estimators have a sampling distribution
that is wide at finite n and not centred on 0.5 for every estimator.  On
synthetic fGn (see verify_memorycalculator.py) DFA sits ~0.02 high at
H = 0.3 and the variogram ~0.08 low at H = 0.9.

So this script reports three things for every estimator:

  1. the point estimate on the real series;
  2. a **shuffle null** -- the same returns randomly permuted, which destroys
     all temporal dependence while preserving the marginal distribution
     exactly (fat tails, skew, everything).  Any true H must be judged
     against where the estimator lands on that null, not against 0.5;
  3. a **moving-block bootstrap CI** on the estimate itself, with a block
     length long enough to preserve local dependence.

If the point estimate falls inside the shuffle null's spread, the series
gives no evidence of long memory that this estimator can detect at this n,
regardless of how far the number sits from 0.5.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from MemoryCalculator import (
    INCREMENTS,
    higuchi_fd,
    hurst_dfa,
    hurst_variogram,
    log_returns,
    mfdfa,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def cache_path(ticker, interval):
    safe = ticker.replace("^", "").replace("/", "-")
    return os.path.join(HERE, f"data_{safe}_{interval}.csv")


def download(ticker, start, end, interval):
    """Fetch a close-price series from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance is not installed. Install it into the project venv:\n"
            "    .venv/bin/pip install yfinance"
        )

    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        progress=False, auto_adjust=True,
    )
    if df is None or len(df) == 0:
        raise SystemExit(
            f"yfinance returned no rows for {ticker}. Check the ticker and "
            "your network connection."
        )
    # yfinance returns a MultiIndex column frame for a single ticker in
    # recent versions; flatten to the Close series either way.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def load_series(ticker, start, end, interval, allow_download=True):
    path = cache_path(ticker, interval)
    if allow_download:
        try:
            s = download(ticker, start, end, interval)
            s.to_csv(path, header=["Close"])
            print(f"Downloaded {len(s)} rows for {ticker} -> cached at "
                  f"{os.path.basename(path)}")
            return s
        except SystemExit:
            raise
        except Exception as exc:
            print(f"Download failed ({exc.__class__.__name__}: {exc}).")
            if not os.path.exists(path):
                raise SystemExit("No cached copy available either; aborting.")
            print("Falling back to the cached copy.")

    if not os.path.exists(path):
        raise SystemExit(
            f"No cached file at {path}. Run once without --no-download."
        )
    df = pd.read_csv(path, index_col=0)
    # Tolerate the multi-row header yfinance writes when to_csv is called on
    # a MultiIndex frame: a 'Ticker' row and a bare 'Date' row sit above the
    # data, so both the index and the values need coercing before use.
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce")
    try:
        s.index = pd.to_datetime(s.index, errors="coerce", format="ISO8601")
    except (ValueError, TypeError):
        s.index = pd.to_datetime(s.index, errors="coerce", format="mixed")
    s = s[s.index.notna()].dropna().sort_index()
    if len(s) == 0:
        raise SystemExit(f"Cached file {path} contained no usable rows.")
    print(f"Loaded {len(s)} cached rows for {ticker}")
    return s


# ---------------------------------------------------------------------------
# Estimation with an honest error bar
# ---------------------------------------------------------------------------
ESTIMATORS = {
    "variogram": lambda r: hurst_variogram(r, input_type=INCREMENTS),
    "DFA": lambda r: hurst_dfa(r, input_type=INCREMENTS),
    "MFDFA h(2)": lambda r: mfdfa(r, q_list=[2], input_type=INCREMENTS)[2],
    "Higuchi 2-FD": lambda r: 2.0 - higuchi_fd(r, input_type=INCREMENTS),
}


def shuffle_null(returns, fn, n_rep, rng):
    """H_hat on randomly permuted returns: true H = 0.5, same marginals."""
    out = np.empty(n_rep)
    for i in range(n_rep):
        out[i] = fn(rng.permutation(returns))
    return out


def block_bootstrap(returns, fn, n_rep, block, rng):
    """Moving-block bootstrap: resample blocks to preserve local dependence."""
    n = len(returns)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_rep)
    for i in range(n_rep):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([returns[s: s + block] for s in starts])[:n]
        out[i] = fn(sample)
    return out


def analyse(returns, n_rep, block, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn in ESTIMATORS.items():
        point = fn(returns)
        null = shuffle_null(returns, fn, n_rep, rng)
        boot = block_bootstrap(returns, fn, n_rep, block, rng)
        # Two-sided empirical p-value against the shuffled null.
        centre = null.mean()
        p = np.mean(np.abs(null - centre) >= abs(point - centre))
        rows.append(
            dict(
                name=name,
                point=point,
                null_mean=centre,
                null_sd=null.std(ddof=1),
                z=(point - centre) / null.std(ddof=1),
                p=p,
                lo=np.percentile(boot, 2.5),
                hi=np.percentile(boot, 97.5),
            )
        )
    return rows


def print_table(rows, label):
    print(f"\n{label}")
    hdr = (
        f"{'estimator':>13} {'H_hat':>7} {'95% block-boot':>18} "
        f"{'shuffled null':>17} {'z':>7} {'p':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        nul = f"{r['null_mean']:.3f}+-{r['null_sd']:.3f}"
        print(
            f"{r['name']:>13} {r['point']:>7.3f} {ci:>18} {nul:>17} "
            f"{r['z']:>7.2f} {r['p']:>7.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reps", type=int, default=200,
                    help="null / bootstrap replications per estimator")
    ap.add_argument("--block", type=int, default=250,
                    help="moving-block bootstrap block length")
    ap.add_argument("--abs-returns", action="store_true",
                    help="also analyse |returns| (the volatility proxy, where "
                         "long memory is actually a stylized fact)")
    args = ap.parse_args()

    prices = load_series(
        args.ticker, args.start, args.end, args.interval,
        allow_download=not args.no_download,
    )
    r = log_returns(prices).values
    print(f"\n{args.ticker}: {len(r)} log returns, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"mean {r.mean():.2e}  sd {r.std():.4f}  "
          f"skew {pd.Series(r).skew():.2f}  kurtosis {pd.Series(r).kurt():.1f}")

    print_table(analyse(r, args.reps, args.block), "LOG RETURNS")

    if args.abs_returns:
        print_table(
            analyse(np.abs(r) - np.abs(r).mean(), args.reps, args.block),
            "ABSOLUTE RETURNS (volatility proxy)",
        )

    print(
        "\nReading this table: compare H_hat against the shuffled-null column,\n"
        "not against 0.5. The null is what each estimator returns on data with\n"
        "identical marginals and no memory whatsoever, at this exact sample\n"
        "size. p is the two-sided empirical probability of seeing a deviation\n"
        "that large under no memory."
    )


if __name__ == "__main__":
    main()
