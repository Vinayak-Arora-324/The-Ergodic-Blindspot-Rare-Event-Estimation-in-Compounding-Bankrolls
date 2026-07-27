"""
Monte Carlo convergence figure for the estimators in stage2_diagnosis.py.

Panel A: sampling distribution of each estimator at one affordable N.
Panel B: sd(estimator) vs N, log-log, against the edge-resolution threshold.

Estimators (all unbiased, all targeting g_h = E[log_growth]):
  crude       : mean( lg(x) ),                  x ~ f
  +CV         : mean( D(x) ) + MU,              x ~ f,  D = lg - r,  E[r]=MU known
  +CV+IS      : mean( D(y) * f(y)/q(y) ) + MU,  y ~ q,  q = jump rate inflated to 0.3
"""
import numpy as np
from scipy import stats, integrate
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20)

# ---------------- model ----------------
lam, beta, s, MU = 1e-3, 1.5, 0.05, 0.005
m = MU + lam * beta
LAM_Q = 0.30
S0, K, c = 100.0, 50.0, 0.0005
LO, HI = -40.0, 2.0
Kp = beta / s

def bulk(r):
    return np.exp(-0.5 * ((r - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

def jumpc(r):
    """density of Normal(m,s) - Exp(beta), i.e. exponnorm(-r; K=beta/s, loc=-m, scale=s)"""
    y = (-r + m) / s
    return (1 / (2 * Kp)) * np.exp(1 / (2 * Kp**2) - y / Kp) * erfc((1 / Kp - y) / np.sqrt(2)) / s

f = lambda r: (1 - lam) * bulk(r) + lam * jumpc(r)
q = lambda r: (1 - LAM_Q) * bulk(r) + LAM_Q * jumpc(r)

# verify closed-form densities against scipy
chk = np.array([-8.0, -3.0, -0.5, 0.0, 0.05, 0.4])
ref = ((1 - lam) * stats.norm.pdf(chk, m, s)
       + lam * stats.exponnorm.pdf(-chk, Kp, loc=-m, scale=s))
assert np.allclose(f(chk), ref, rtol=1e-10), "density mismatch vs scipy"
print("density check vs scipy: OK")

def draw(n, rate):
    j = rng.random(n) < rate
    r = m + s * rng.standard_normal(n)
    r[j] -= rng.exponential(beta, j.sum())
    return r

# ---------------- estimand ----------------
payoff = lambda r: np.maximum(K - S0 * np.exp(r), 0.0)
price, _ = integrate.quad(lambda r: payoff(r) * f(r), LO, HI, limit=1500)
lg = lambda r: np.log((1 - c) * np.exp(r) + c * payoff(r) / price)
D  = lambda r: lg(r) - r

g_h, _ = integrate.quad(lambda r: lg(r) * f(r), LO, HI, limit=1500)
g_u  = MU
edge = g_h - g_u
thresh = edge / 2.0          # sd needed to separate g_h from g_u at 2 sigma
print(f"g_u {g_u:+.6f}   g_h {g_h:+.6f}   edge {edge:+.6f}   sd threshold {thresh:.2e}")

# ---------------- estimators ----------------
def est_crude(N):  return lg(draw(N, lam)).mean()
def est_cv(N):     return D(draw(N, lam)).mean() + MU
def est_cvis(N):
    y = draw(N, LAM_Q)
    return (D(y) * (f(y) / q(y))).mean() + MU

METHODS = [("crude MC", est_crude, "#c0392b"),
           ("+ control variate", est_cv, "#e67e22"),
           ("+ CV + importance sampling", est_cvis, "#1f6f4a")]

# ---------------- Panel A: sampling distributions at fixed N ----------------
N_A, T_A = 4096, 2000
samples = {name: np.array([fn(N_A) for _ in range(T_A)]) for name, fn, _ in METHODS}
print(f"\nunbiasedness check at N={N_A} (truth {g_h:+.6f}):")
for name, _, _ in METHODS:
    e = samples[name]
    mc_se = e.std() / np.sqrt(T_A)
    print(f"  {name:28s} mean {e.mean():+.6f}  bias {e.mean()-g_h:+.2e} "
          f"({abs(e.mean()-g_h)/mc_se:4.1f} MC-SE)  median {np.median(e):+.6f}")

# ---------------- Panel B: sd vs N ----------------
Ns = np.array([64, 256, 1024, 4096, 16384, 65536])
T_B = 300
sds = {name: [] for name, _, _ in METHODS}
for N in Ns:
    for name, fn, _ in METHODS:
        e = np.array([fn(int(N)) for _ in range(T_B)])
        sds[name].append(e.std())
sds = {k: np.array(v) for k, v in sds.items()}

print(f"\n{'estimator':28s} {'slope':>7} {'N to resolve sign':>18}")
Nstar = {}
for name, _, _ in METHODS:
    v = sds[name]
    slope = np.polyfit(np.log(Ns), np.log(v), 1)[0]
    # sd = A * N^slope  -> solve sd = thresh
    A = np.exp(np.log(v).mean() - slope * np.log(Ns).mean())
    Nstar[name] = (thresh / A) ** (1.0 / slope)
    print(f"{name:28s} {slope:7.3f} {Nstar[name]:18,.0f}")

# ---------------- figure ----------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

for name, _, col in METHODS:
    e = samples[name]
    axA.hist(e, bins=70, histtype="step", lw=1.9, color=col, density=True,
             label=f"{name}  (sd {e.std():.1e})")
axA.axvline(g_h, color="k", lw=1.6, label=f"truth $g_h$ = {g_h:.5f}")
axA.axvline(g_u, color="k", ls=":", lw=1.6, label=f"$g_u$ = {g_u:.5f} (no hedge)")
axA.axvspan(g_u - 4e-4, g_u + 4e-4, color="0.85", zorder=0)
axA.set_yscale("log")
axA.set_xlim(g_h - 0.0045, g_h + 0.0045)
axA.set_xlabel("estimate of geometric growth rate $g$")
axA.set_ylabel("density")
axA.set_title(f"A. Sampling distribution at affordable N = {N_A:,}\n"
              "the gap between the two black lines is the whole decision",
              fontsize=10.5)
axA.legend(fontsize=7.6, loc="upper left")

for name, _, col in METHODS:
    axB.plot(Ns, sds[name], "o-", color=col, lw=1.9, ms=5, label=name)
axB.axhline(thresh, color="k", ls="--", lw=1.5)
axB.text(70, thresh * 1.15, "sd needed to call the sign at $2\\sigma$",
         fontsize=8.5, va="bottom")
for name, _, col in METHODS:
    n = Nstar[name]
    if Ns[0] <= n <= Ns[-1] * 40:
        axB.plot([n], [thresh], "v", color=col, ms=9, zorder=5)
        axB.annotate(f"N≈{n:,.0f}", (n, thresh), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=8, color=col)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("samples per run, N")
axB.set_ylabel("sd of the estimator")
axB.set_title("B. Convergence: variance reduction moves the intercept,\n"
              "not the $N^{-1/2}$ slope", fontsize=10.5)
axB.legend(fontsize=8.4, loc="lower left")
axB.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("mc_estimator_convergence.png", dpi=170)
print("\nsaved figure")
