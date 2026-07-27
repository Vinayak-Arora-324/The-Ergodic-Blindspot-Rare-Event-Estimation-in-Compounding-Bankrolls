"""
Sample Changes
STAGE 2 - Does the tail hedge compound or bleed, and does the ANSWER depend on the estimator?

Target quantity (the ergodic object):
    g = E[ log( (1-c)*exp(r) + c*payoff(r)/price ) ]
Because per-period returns are iid, the time-average growth rate of a finite
bankroll IS this single-period expectation. No 120-period loop needed - the loop
would only add Monte Carlo noise to a quantity that was never path-dependent.
What makes it hard is the log: a crash contributes a rare, large term.

Model: light-tailed bulk + rare negative jump (the honest shape for equity crashes,
and the structure CGMY generalizes).
"""
import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(7)

# ---------------- Model ----------------
lam, beta, s = 1e-3, 1.5, 0.05     # jump prob, mean jump size, bulk vol
TARGET_MU = 0.005                  # E[r] -> unhedged growth g_u = 0.005 exactly
m = TARGET_MU + lam * beta

def f(r):                          # true density
    return ((1 - lam) * stats.norm.pdf(r, m, s)
            + lam * stats.exponnorm.pdf(-r, beta / s, loc=-m, scale=s))

def draw(n, rate):                 # sample with jump prob = rate
    jump = rng.random(n) < rate
    r = m + s * rng.standard_normal(n)
    r[jump] -= rng.exponential(beta, jump.sum())
    return r

# ---------------- Hedge ----------------
S0, K, c = 100.0, 50.0, 0.0005
payoff = lambda r: np.maximum(K - S0 * np.exp(r), 0.0)
LO, HI = -40.0, 2.0
price, _ = integrate.quad(lambda r: payoff(r) * f(r), LO, HI, limit=1500)
log_growth = lambda r: np.log((1 - c) * np.exp(r) + c * payoff(r) / price)

# ---------------- Ground truth (quadrature, no sampling) ----------------
g_h, _ = integrate.quad(lambda r: log_growth(r) * f(r), LO, HI, limit=1500)
g_u = TARGET_MU
r_star = np.log(K / S0)
p_crash, _ = integrate.quad(f, LO, r_star, limit=1500)

# ---------------- Importance sampling (defensive: oversample jumps) ----------------
LAM_Q = 0.30
def q(r):
    return ((1 - LAM_Q) * stats.norm.pdf(r, m, s)
            + LAM_Q * stats.exponnorm.pdf(-r, beta / s, loc=-m, scale=s))

# ---------------- Compare at affordable N ----------------
N, trials = 5_000, 2000
crude = np.array([log_growth(draw(N, lam)).mean() for _ in range(trials)])
imp   = np.array([np.mean(log_growth(x := draw(N, LAM_Q)) * (f(x) / q(x)))
                  for _ in range(trials)])

print(f"P(crash)  {p_crash:.2e}   (~{N*p_crash:.1f} crashes per run of N={N:,})")
print(f"TRUTH     g_unhedged {g_u:+.6f}")
print(f"TRUTH     g_hedged   {g_h:+.6f}   -> hedge {'COMPOUNDS' if g_h>g_u else 'BLEEDS'} "
      f"(edge {g_h-g_u:+.6f})")
print(f"\n{'estimator':22s} {'median':>10} {'sd':>10} {'RMSE':>10}   says COMPOUNDS")
for name, e in (("crude MC", crude), ("importance sampling", imp)):
    rmse = np.sqrt(((e - g_h) ** 2).mean())
    print(f"{name:22s} {np.median(e):+10.6f} {e.std():10.6f} {rmse:10.6f}   {(e > g_u).mean():6.1%}")

# bounded weights => finite variance, the reason IS is trustworthy here
x = draw(200_000, LAM_Q)
w = f(x) / q(x)
print(f"\nIS weights bounded in [{w.min():.4f}, {w.max():.4f}] -> finite variance by construction")
print(f"variance reduction factor: {(crude.std()/imp.std())**2:.1f}x")
