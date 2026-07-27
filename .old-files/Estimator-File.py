import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(0)

# ---- Model: one-period log return, fat-tailed (Student-t) ----
S0    = 100.0
nu    = 3.0            # tail heaviness; nu=3 => finite mean & var, heavy tail
sigma = 0.05          # target stdev of the log return
t_scale = sigma / np.sqrt(nu / (nu - 2.0))   # scale so sd(return) = sigma
mu    = 0.0

def sample_returns(n):
    return mu + t_scale * rng.standard_t(nu, size=n)

def ret_pdf(r):
    z = (r - mu) / t_scale
    return stats.t.pdf(z, nu) / t_scale

# ---- Hedge: deep OTM put, only pays in a crash ----
K = 55.0              # 45% OTM: crash is genuinely rare
def put_payoff(r):
    S = S0 * np.exp(r)
    return np.maximum(K - S, 0.0)

# ---- Ground truth via quadrature (no sampling) ----
true_val, _ = integrate.quad(lambda r: put_payoff(r) * ret_pdf(r), -6.0, 6.0, limit=400)
r_star  = np.log(K / S0)
p_crash = stats.t.cdf((r_star - mu) / t_scale, nu)   # P(put finishes ITM)

# ---- Crude MC at an affordable N, sampled many times ----
N_afford = 5_000
trials   = 3000
ests = np.array([put_payoff(sample_returns(N_afford)).mean() for _ in range(trials)])

print(f"crash prob (put ITM)      : {p_crash:.2e}  (~{N_afford*p_crash:.1f} crashes per run)")
print(f"TRUE E[payoff] (quad)     : {true_val:.5f}")
print(f"crude MC  mean of estimates: {ests.mean():.5f}   <- ~unbiased in expectation")
print(f"crude MC  MEDIAN estimate  : {np.median(ests):.5f}   <- the TYPICAL run")
print(f"          std of estimates : {ests.std():.5f}")
print(f"          coeff of variation: {ests.std()/ests.mean():.2f}")
print(f"frac of runs < 50% of true : {(ests < 0.5*true_val).mean():.2%}")
print(f"frac of runs that saw 0    : {(ests == 0).mean():.2%}")
print("five individual affordable runs:",
      [f"{put_payoff(sample_returns(N_afford)).mean():.4f}" for _ in range(5)])
