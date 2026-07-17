import numpy as np
from scipy import stats, integrate

rng = np.random.default_rng(11)
lam, beta, s = 1e-3, 1.5, 0.05
TARGET_MU = 0.005
m = TARGET_MU + lam * beta
f = lambda r: ((1-lam)*stats.norm.pdf(r, m, s)
               + lam*stats.exponnorm.pdf(-r, beta/s, loc=-m, scale=s))
def draw(n, rate):
    j = rng.random(n) < rate
    r = m + s*rng.standard_normal(n)
    r[j] -= rng.exponential(beta, j.sum())
    return r

S0, K, c = 100.0, 50.0, 0.0005
payoff = lambda r: np.maximum(K - S0*np.exp(r), 0.0)
LO, HI = -40.0, 2.0
price, _ = integrate.quad(lambda r: payoff(r)*f(r), LO, HI, limit=1500)
lg = lambda r: np.log((1-c)*np.exp(r) + c*payoff(r)/price)
D  = lambda r: lg(r) - r          # control variate: E[r]=TARGET_MU known exactly

g_h, _ = integrate.quad(lambda r: lg(r)*f(r), LO, HI, limit=1500)
edge = g_h - TARGET_MU

# where does the variance live?
x = draw(2_000_000, lam)
print(f"sd of log_growth (crude target) : {lg(x).std():.4f}   <- bulk noise, s={s}")
print(f"sd of D          (after CV)     : {D(x).std():.4f}   <- rare-event only")
print(f"edge to resolve                 : {edge:+.6f}")
print(f"\nN needed for crude MC to resolve sign at 2 sigma : {int((2*lg(x).std()/edge)**2):,}")
print(f"N needed after control variate                   : {int((2*D(x).std()/edge)**2):,}")

# CV + IS
LAM_Q = 0.30
q = lambda r: ((1-LAM_Q)*stats.norm.pdf(r, m, s)
               + LAM_Q*stats.exponnorm.pdf(-r, beta/s, loc=-m, scale=s))
N, trials = 5_000, 1500
cv    = np.array([D(draw(N, lam)).mean() + TARGET_MU for _ in range(trials)])
cv_is = np.array([np.mean(D(y := draw(N, LAM_Q)) * (f(y)/q(y))) + TARGET_MU
                  for _ in range(trials)])
crude = np.array([lg(draw(N, lam)).mean() for _ in range(trials)])
print(f"\n{'estimator':28s} {'sd':>10}  var-reduction vs crude")
for nm, e in (("crude MC", crude), ("+ control variate", cv), ("+ CV + importance sampling", cv_is)):
    print(f"{nm:28s} {e.std():10.6f}  {(crude.std()/e.std())**2:8.1f}x")

# the decisive point
import time
t0 = time.time(); integrate.quad(lambda r: lg(r)*f(r), LO, HI, limit=1500); t1 = time.time()
print(f"\nquadrature gives the EXACT answer in {1000*(t1-t0):.1f} ms. No Monte Carlo needed at all.")
