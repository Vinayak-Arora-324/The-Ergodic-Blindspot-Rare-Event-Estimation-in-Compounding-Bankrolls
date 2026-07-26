"""
THE THIRD LIE - long memory makes the error bar dishonest.

Model: r_t = mu_a - sig_t^2/2 + sig_t * Z_t,   Z iid N(0,1)
       log sig_t = log(sbar) + xi * G_t,       G = fractional Gaussian noise, Hurst H
Returns are serially UNcorrelated (matches the stylized fact), but the growth
estimand inherits long memory through the variance drag sig^2/2.

Truth: g = mu_a - 0.5 * sbar^2 * exp(2 xi^2)   (closed form, since Var(G_t)=1)

Experiment: estimate g by the time average of log-growth over one path of
length T (i.e., one backtest). Compare
  - TRUE SE   : sd of that estimator across many independent paths
  - NAIVE SE  : the iid-formula sd/sqrt(T) a practitioner reports from one path
"""
import numpy as np

rng = np.random.default_rng(9)

def chol_fgn(H, T):
    """Exact fGn: Cholesky of the covariance. Prefix of a path is a valid shorter path."""
    k = np.arange(T, dtype=float)
    g = 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))
    C = g[np.abs(np.subtract.outer(np.arange(T), np.arange(T)))]
    return np.linalg.cholesky(C + 1e-10 * np.eye(T))

T_MAX, M = 4096, 1200
mu_a, sbar, xi = 0.01, 0.15, 0.9
g_true = mu_a - 0.5 * sbar**2 * np.exp(2 * xi**2)

for H in (0.8, 0.1):
    L = chol_fgn(H, T_MAX)
    G = L @ rng.standard_normal((T_MAX, M))          # M independent fGn paths
    sig = sbar * np.exp(xi * G)
    lg = mu_a - 0.5 * sig**2 + sig * rng.standard_normal((T_MAX, M))

    # sanity: returns uncorrelated, |returns| clustered?
    a = lg[:-100, 0] - lg[:, 0].mean(); b = lg[100:, 0] - lg[:, 0].mean()
    ac_r  = np.corrcoef(lg[:-100, :].ravel(), lg[100:, :].ravel())[0, 1]
    ac_ab = np.corrcoef(np.abs(lg[:-100, :]).ravel(), np.abs(lg[100:, :]).ravel())[0, 1]

    print(f"\nH = {H}   (true g = {g_true:+.5f})")
    print(f"corr(r_t, r_t+100) = {ac_r:+.3f}   corr(|r_t|,|r_t+100|) = {ac_ab:+.3f}")
    print(f"{'T':>6} {'mean est':>10} {'TRUE SE':>9} {'NAIVE SE':>9} {'ratio':>6} {'ESS':>7}")
    rows = []
    for T in (256, 1024, 4096):
        means    = lg[:T].mean(axis=0)               # one estimate per path
        true_se  = means.std()
        naive_se = (lg[:T].std(axis=0) / np.sqrt(T)).mean()
        ess      = lg[:T].var(axis=0).mean() / true_se**2
        rows.append((T, true_se))
        print(f"{T:6d} {means.mean():+10.5f} {true_se:9.5f} {naive_se:9.5f} "
              f"{true_se/naive_se:6.2f} {ess:7.0f}")
    (T1, s1), (T2, s2) = rows[-2], rows[-1]
    slope = 2 * np.log(s2 / s1) / np.log(T2 / T1)
    print(f"Var(mean) ~ T^{slope:+.2f}    (iid: -1.00, pure-LRD theory: {2*H-2:+.2f})")
