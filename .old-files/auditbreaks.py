import numpy as np
from scipy import stats, integrate

# --- CHECK 1: exponnorm density really matches the sampler (rule out silent mismatch) ---
rng = np.random.default_rng(3)
lam, beta, s, MU = 1e-3, 1.5, 0.05, 0.005
m = MU + lam * beta
def f(r):
    return ((1-lam)*stats.norm.pdf(r, m, s)
            + lam*stats.exponnorm.pdf(-r, beta/s, loc=-m, scale=s))
n = 4_000_000
j = rng.random(n) < lam
r = m + s*rng.standard_normal(n)
r[j] -= rng.exponential(beta, j.sum())
p_emp = (r < np.log(0.5)).mean()
p_quad, _ = integrate.quad(f, -40, np.log(0.5), limit=1500)
print(f"C1 density vs sampler: P(crash) empirical {p_emp:.2e} vs quad {p_quad:.2e}")

# --- CHECK 2: sweep2's scale convention jumps discontinuously at nu=2 ---
sigma = 0.06
for nu in (1.8, 2.0, 2.5, 3.0):
    scale = sigma/np.sqrt(nu/(nu-2.0)) if nu > 2 else sigma
    print(f"C2 nu={nu}: t-scale used = {scale:.4f}")

# --- CHECK 3: is 'hedge only wins when p~1e-3' a c-grid artifact? size c ~ p at tiny lam ---
S0 = 100.0
def edge(lam, beta, s, K, c):
    m = MU + lam*beta
    fl = lambda r: ((1-lam)*stats.norm.pdf(r, m, s)
                    + lam*stats.exponnorm.pdf(-r, beta/s, loc=-m, scale=s))
    pay = lambda r: max(K - S0*np.exp(r), 0.0)
    price, _ = integrate.quad(lambda r: pay(r)*fl(r), -60, 2, limit=2000)
    g, _ = integrate.quad(lambda r: np.log((1-c)*np.exp(r) + c*pay(r)/price)*fl(r),
                          -60, 2, limit=2000)
    p, _ = integrate.quad(fl, -60, np.log(K/S0), limit=2000)
    return g - MU, p

for lam_t in (1e-3, 1e-4, 1e-5):
    d_grid, p = edge(lam_t, 1.5, 0.05, 50.0, 2e-4)      # sweep3's smallest c
    d_sized, _ = edge(lam_t, 1.5, 0.05, 50.0, 0.8*p)    # c sized to p
    print(f"C3 lam={lam_t:.0e} p={p:.1e}: edge(c=2e-4)={d_grid:+.2e}  edge(c~p)={d_sized:+.2e}")

# --- CHECK 4: stage1 'TRUE' truncated at [-6,6]; how much mass was cut? ---
nu_, sig_ = 3.0, 0.05
sc = sig_/np.sqrt(nu_/(nu_-2))
f1 = lambda r: stats.t.pdf(r/sc, nu_)/sc
pay1 = lambda r: max(55.0 - 100.0*np.exp(r), 0.0)
t6, _  = integrate.quad(lambda r: pay1(r)*f1(r), -6, 6, limit=800)
t40, _ = integrate.quad(lambda r: pay1(r)*f1(r), -40, 6, limit=1200)
print(f"C4 stage1 truth: [-6,6] {t6:.6f} vs [-40,6] {t40:.6f}  ({100*(t40-t6)/t40:.2f}% cut)")

# --- CHECK 5: is the estimand rare-event dominated once hedged? variance decomposition ---
lg = None
K, c = 50.0, 0.0005
payv = lambda r: np.maximum(K - S0*np.exp(r), 0.0)
price, _ = integrate.quad(lambda r: payv(r)*f(r), -40, 2, limit=1500)
lgf = lambda r: np.log((1-c)*np.exp(r) + c*payv(r)/price)
x = r[:2_000_000]
crash = x < np.log(K/S0)
v = lgf(x)
print(f"C5 hedged lg in crash: mean {v[crash].mean():+.3f} (bounded!)  "
      f"crash share of Var(lg): {np.var(v[crash])*crash.mean()/np.var(v):.1%}")
print(f"C5 UNhedged r in crash: mean {x[crash].mean():+.3f}  "
      f"crash share of Var(r): {(np.var(x[crash])+ (x[crash].mean()-x.mean())**2)*crash.mean()/np.var(x):.1%}")
