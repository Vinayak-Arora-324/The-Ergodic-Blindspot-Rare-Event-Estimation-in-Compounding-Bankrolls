"""Check whether S&P 500 returns or volatility show a lasting pattern.

    python -m blindspot.sp500_memory            # fetch, estimate, write figure
    python -m blindspot.sp500_memory --no-download

The model uses the Hurst exponent `H` to control volatility memory. On roughly
36 years of S&P 500 data, return direction is close to memoryless, while large
and small moves cluster over time. This supports putting memory in volatility
rather than using it to predict whether the market rises or falls next.

The comparison keeps each return's size in its original position and randomly
changes only its sign. This removes directional patterns without destroying
calm and turbulent periods. Simply shuffling all returns would destroy both and
could make ordinary volatility clustering look like directional memory.
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from .memory import (
    INCREMENTS,
    higuchi_fd,
    hurst_dfa,
    hurst_variogram,
    log_returns,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TICKER = "^GSPC"
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(PROJECT_ROOT, "data", "sp500_daily.csv")

# Scale range probed.  The 50-day default is too short to say anything about
# *long* memory, and the answer is scale-dependent for volatility: DFA on |r|
# gives H = 0.82 out to 50 days but 0.95 out to 250-1000, where it plateaus.
# Returns are flat at 0.45-0.49 across every choice.  250 days ~ one trading
# year, and spans 1.4 decades of scale, which is enough to see a straight line
# on log-log rather than assert one.
MAX_LAG = 250
K_MAX = 50          # Higuchi's scale parameter; probes lags 1..K_MAX

ESTIMATORS = {
    "variogram": lambda r: hurst_variogram(r, max_lag=MAX_LAG,
                                           input_type=INCREMENTS),
    "DFA": lambda r: hurst_dfa(r, max_lag=MAX_LAG, input_type=INCREMENTS),
    "Higuchi": lambda r: 2.0 - higuchi_fd(r, k_max=K_MAX,
                                          input_type=INCREMENTS),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _prepare_prices(prices, start, end):
    """Clean prices and select [start, end), matching the download API."""
    out = pd.to_numeric(prices, errors="coerce")
    out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out[out.index.notna()].dropna().sort_index()
    if not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index < end]
    if out.empty:
        raise ValueError("no valid prices in the requested date range")
    return out


def _save_prices(prices):
    """Replace the cache atomically after the prices have been validated."""
    parent = os.path.dirname(CACHE)
    os.makedirs(parent, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=parent,
                                         suffix=".csv", delete=False) as handle:
            temporary = handle.name
            prices.to_csv(handle, header=["Close"], index_label="Date")
        os.replace(temporary, CACHE)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_prices(start, end, allow_download=True):
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be before end (end is exclusive)")
    if allow_download:
        try:
            import yfinance as yf

            df = yf.download(TICKER, start=start, end=end, progress=False,
                             auto_adjust=True)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = _prepare_prices(close, start, end)
            _save_prices(close)
            print(f"Downloaded {len(close)} rows -> {os.path.basename(CACHE)}")
            return close
        except Exception as exc:
            print(f"Download failed ({exc.__class__.__name__}: {exc}); "
                  "using cache.")

    if not os.path.exists(CACHE):
        raise SystemExit(f"No cached data at {CACHE}; run without "
                         "--no-download once.")
    s = pd.read_csv(CACHE, index_col=0)
    col = "Close" if "Close" in s.columns else s.columns[0]
    try:
        out = _prepare_prices(s[col], start, end)
    except ValueError as exc:
        raise SystemExit(f"Cached data cannot satisfy the request: {exc}") from exc
    print(f"Loaded {len(out)} cached rows")
    return out


# ---------------------------------------------------------------------------
# Estimation against a null that keeps volatility clustering intact
# ---------------------------------------------------------------------------
def sign_null(x, fn, n_rep, rng):
    """Build a no-directional-memory baseline while preserving return sizes."""
    mag = np.abs(x)
    return np.array([fn(rng.choice([-1.0, 1.0], size=len(mag)) * mag)
                     for _ in range(n_rep)])


def block_bootstrap(x, fn, n_rep, block, rng):
    n, nb = len(x), int(np.ceil(len(x) / block))
    out = np.empty(n_rep)
    for i in range(n_rep):
        starts = rng.integers(0, n - block + 1, size=nb)
        out[i] = fn(np.concatenate([x[s: s + block] for s in starts])[:n])
    return out


def analyse(x, n_rep, block, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, fn in ESTIMATORS.items():
        point = fn(x)
        null = sign_null(x, fn, n_rep, rng)
        boot = block_bootstrap(x, fn, n_rep, block, rng)
        sd = null.std(ddof=1)
        rows.append(dict(
            name=name, point=point,
            null_mean=null.mean(), null_sd=sd,
            z=(point - null.mean()) / sd,
            p=np.mean(np.abs(null - null.mean()) >= abs(point - null.mean())),
            lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
        ))
    return rows


def print_table(rows, label):
    print(f"\n{label}")
    hdr = (f"{'method':>10} {'H':>7} {'95% range':>16} "
           f"{'no-memory range':>17} {'z':>7} {'p':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        null = f"{r['null_mean']:.3f}+-{r['null_sd']:.3f}"
        print(f"{r['name']:>10} {r['point']:>7.3f} {ci:>16} {null:>15} "
              f"{r['z']:>7.2f} {r['p']:>7.3f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def figure(returns, vol, rows_r, rows_v, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- left: the DFA scaling itself -------------------------------------
    for series, lab, c in ((returns, "log returns", "tab:blue"),
                           (vol, "|log returns|", "tab:red")):
        H, lags, F = hurst_dfa(series, max_lag=MAX_LAG, input_type=INCREMENTS,
                               return_curve=True)
        ax1.loglog(lags, F / F[0], "o", ms=4, color=c, alpha=0.75,
                   label=f"{lab}   H = {H:.3f}")
        fit = np.exp(np.polyval(np.polyfit(np.log(lags), np.log(F / F[0]), 1),
                                np.log(lags)))
        ax1.loglog(lags, fit, "-", color=c, lw=1.4)
    ref = np.array(lags, dtype=float)
    ax1.loglog(ref, (ref / ref[0]) ** 0.5, "--", color="grey", lw=1.2,
               label="H = 0.5 (no memory)")
    ax1.set_xlabel("window length (trading days)")
    ax1.set_ylabel("relative size of fluctuations")
    ax1.set_title("How fluctuations grow over longer windows")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3, which="both")

    # --- right: estimate vs its own null ----------------------------------
    names = [r["name"] for r in rows_r]
    y = np.arange(len(names))
    for rows, lab, c, off in ((rows_r, "log returns", "tab:blue", -0.15),
                              (rows_v, "|log returns|", "tab:red", 0.15)):
        pts = np.array([r["point"] for r in rows])
        lo = np.array([r["lo"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        nm = np.array([r["null_mean"] for r in rows])
        ns = np.array([r["null_sd"] for r in rows])
        # A percentile interval may exclude the original point estimate.
        # Draw its endpoints independently instead of making negative errors.
        ax2.hlines(y + off, lo, hi, color=c, zorder=3)
        ax2.vlines(lo, y + off - 0.035, y + off + 0.035, color=c, zorder=3)
        ax2.vlines(hi, y + off - 0.035, y + off + 0.035, color=c, zorder=3)
        ax2.plot(pts, y + off, "o", color=c, ms=6, label=lab, zorder=4)
        for yy, m, s in zip(y + off, nm, ns):
            ax2.add_patch(plt.Rectangle((m - 2 * s, yy - 0.09), 4 * s, 0.18,
                                        color=c, alpha=0.18, zorder=1))
            ax2.plot([m, m], [yy - 0.09, yy + 0.09], color=c, lw=1.2,
                     alpha=0.6, zorder=2)
    ax2.axvline(0.5, color="grey", ls="--", lw=1.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Hurst exponent (0.5 means no lasting pattern)")
    ax2.set_title("estimate and 95% range vs no-directional-memory baseline")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3, axis="x")
    ax2.set_ylim(-0.6, len(names) - 0.4)

    fig.suptitle("S&P 500: memory in returns and volatility", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--block", type=int, default=250)
    args = ap.parse_args()

    prices = load_prices(args.start, args.end, not args.no_download)
    r = log_returns(prices).values
    vol = np.abs(r) - np.abs(r).mean()

    print(f"\n{TICKER}: {len(r)} log returns, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"daily spread (sd) {r.std():.4f}  asymmetry {pd.Series(r).skew():.2f}  "
          f"tail weight (excess kurtosis) {pd.Series(r).kurt():.1f}")

    rows_r = analyse(r, args.reps, args.block)
    rows_v = analyse(vol, args.reps, args.block)
    print_table(rows_r, "LOG RETURNS (market direction)")
    print_table(rows_v, "ABSOLUTE RETURNS (size of market moves)")

    figure(r, vol, rows_r, rows_v,
           os.path.abspath(os.path.join(
               HERE, "..", "figures", "market_memory.png")))

    print("\nEach H estimate is compared with the same method applied to a focused")
    print("no-directional-memory baseline. That baseline preserves the original")
    print("return sizes and volatility clustering, and randomizes only their signs.")


if __name__ == "__main__":
    main()
