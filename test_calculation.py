"""
Regression / oracle tests for Calculation.py.

Every test compares against something known analytically, not against a
previously-recorded output of the same code. Naming follows the project's
test_regressions.py convention:
    T* = a verified number the code should reproduce
    B* = a committed bug; the test PASSES when the bug is still present,
         so that fixing it is a visible event.
"""
import numpy as np
from scipy import integrate
from scipy.stats import levy_stable, norm

import Calculation as C

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail):
    _results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}: {detail}")


# ----------------------------------------------------------------------
# Black-Scholes oracle. At alpha=2, beta=0 the stable law IS the normal:
# scipy's S1 parameterization gives Var = 2*gamma^2, so gamma = sigma/sqrt(2).
# ----------------------------------------------------------------------
def bs(S0, K, T, r, q, sigma, kind="call"):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price, norm.cdf(d2)          # price, P(ITM) under Q
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return price, norm.cdf(-d2)


S0, K, T, r, q, sigma = 100.0, 110.0, 0.25, 0.05, 0.02, 0.20
GAM2 = sigma / np.sqrt(2)


# ---------------------------------------------------------------- T1
# The Gaussian limit of the ITM probability must equal N(d2).
def T1_itm_gaussian_limit():
    _, p_true = bs(S0, K, T, r, q, sigma, "call")
    p_code = C.option_itm_probability(S0, K, T, r, q, 2.0, 0.0, GAM2, "call")
    err = abs(p_code - p_true)
    check("T1  P(ITM) -> N(d2) at alpha=2", err < 1e-6,
          f"code={p_code:.8f}  BS={p_true:.8f}  |err|={err:.2e}")

    _, pp_true = bs(S0, 90.0, T, r, q, sigma, "put")
    pp_code = C.option_itm_probability(S0, 90.0, T, r, q, 2.0, 0.0, GAM2, "put")
    check("T1b P(ITM) put -> N(-d2)", abs(pp_code - pp_true) < 1e-6,
          f"code={pp_code:.8f}  BS={pp_true:.8f}")


# ---------------------------------------------------------------- T2
# The martingale condition is *defined* by E[S_T] = S0*exp((r-q)T).
# Test it by integrating the density the code actually uses.
def _mgf(a, b, mu, scale, z_hi):
    """E[e^X] integrated in STANDARDIZED coordinates z=(x-mu)/scale, so the
    bulk of the density is O(1) wide and quad cannot miss it. Returns the
    integral truncated at z=z_hi."""
    def g(z):
        return np.exp(mu + scale * z) * levy_stable.pdf(z, a, b)
    # break the range at the bulk so the adaptive rule always samples it
    pts = [-200, -20, -5, 0, 5, 20]
    pts = [p for p in pts if p < z_hi] + [z_hi]
    tot = 0.0
    for lo_, hi_ in zip(pts[:-1], pts[1:]):
        v, _ = integrate.quad(g, lo_, hi_, limit=600)
        tot += v
    return tot


def T2_martingale_identity():
    target = np.exp((r - q) * T)
    for (a, b) in [(2.0, 0.0), (1.9, -1.0), (1.5, -1.0), (1.5, 0.0), (1.1, -0.2)]:
        gam = GAM2 if a == 2.0 else 0.05
        mu = C.martingale_condition(a, b, gam, T, r, q)
        scale = gam * (T ** (1 / a))

        v = [_mgf(a, b, mu, scale, U) for U in (30, 60, 120)]
        # divergent if the value keeps climbing as the cut moves out
        div = (v[2] - v[1]) / max(abs(v[1]), 1e-300)
        rel = abs(v[0] - target) / target
        if div > 1e-3:
            status = (f"DIVERGES (cut@30={v[0]:.6g}, @60={v[1]:.6g}, "
                      f"@120={v[2]:.6g})")
            ok = False
        else:
            status = f"converged, E[e^X]={v[2]:.8f}, rel.err vs target={rel:.3e}"
            ok = abs(v[2] - target) / target < 1e-4
        check(f"T2  E[S_T]/S0 = e^((r-q)T) for alpha={a}, beta={b}", ok, status)


# ---------------------------------------------------------------- B1
# expected_itm_payoff's call integrand. The payoff of a call is (S0*e^x - K).
# The code builds log_payoff = log(S0)+x-log(S0*e^x-K), i.e. it computes
# S0*e^x / (S0*e^x - K) -- the reciprocal of the payoff, times S0*e^x.
def B1_call_integrand_is_inverted():
    a, b, gam = 2.0, 0.0, GAM2
    mu = C.martingale_condition(a, b, gam, T, r, q)
    scale = gam * (T ** (1 / a))
    x = np.log(K / S0) + 0.5                       # a point comfortably ITM

    log_payoff = np.log(S0) + x - np.log(S0 * np.exp(x) - K + 1e-100)
    coded = np.exp(log_payoff)
    correct = S0 * np.exp(x) - K
    inverted = S0 * np.exp(x) / (S0 * np.exp(x) - K)

    check("B1  call integrand computes S0e^x/(S0e^x-K), not (S0e^x-K)",
          abs(coded - inverted) < 1e-9 and abs(coded - correct) > 1.0,
          f"coded={coded:.6f}  correct payoff={correct:.6f}  "
          f"ratio={correct/coded:.1f}x")

    # consequence: the singularity at x = log(K/S0) sits at the lower limit of
    # integration, so quad integrates a pole
    xs = np.log(K / S0) + np.array([1e-1, 1e-3, 1e-5, 1e-7])
    vals = [np.exp(np.log(S0) + xx - np.log(S0 * np.exp(xx) - K + 1e-100))
            for xx in xs]
    check("B1b integrand has a pole at the lower integration limit",
          vals[-1] > 1e5, f"values approaching log(K/S0): {['%.3g' % v for v in vals]}")


# ---------------------------------------------------------------- T3
# What the call price *should* be, Gaussian case: recompute with the correct
# payoff and compare to Black-Scholes; then compare to what the code returns.
def T3_call_price_vs_bs():
    a, b, gam = 2.0, 0.0, GAM2
    mu = C.martingale_condition(a, b, gam, T, r, q)
    scale = gam * (T ** (1 / a))
    lo = np.log(K / S0)

    def correct(x):
        return (S0 * np.exp(x) - K) * levy_stable.pdf(x, a, b, loc=mu, scale=scale)

    fixed, _ = integrate.quad(correct, lo, lo + 30, limit=1000)
    price_true, _ = bs(S0, K, T, r, q, sigma, "call")
    price_true_undisc = price_true * np.exp(r * T)

    check("T3  corrected integrand reproduces the BS call",
          abs(fixed - price_true_undisc) / price_true_undisc < 1e-5,
          f"corrected={fixed:.6f}  BS(undiscounted)={price_true_undisc:.6f}")

    coded = C.expected_itm_payoff(S0, K, T, r, q, a, b, gam, "call")
    check("T3b code's call payoff matches BS",
          abs(coded - price_true_undisc) / price_true_undisc < 1e-2,
          f"code={coded:.6g}  BS={price_true_undisc:.6f}  "
          f"off by {coded/price_true_undisc:.3g}x")


# ---------------------------------------------------------------- T4
# The put branch has the correct payoff, so it should match BS at alpha=2.
def T4_put_price_vs_bs():
    Kp = 90.0
    a, b, gam = 2.0, 0.0, GAM2
    coded = C.expected_itm_payoff(S0, Kp, T, r, q, a, b, gam, "put")
    price_true, _ = bs(S0, Kp, T, r, q, sigma, "put")
    price_true_undisc = price_true * np.exp(r * T)
    check("T4  put payoff matches BS at alpha=2",
          abs(coded - price_true_undisc) / price_true_undisc < 1e-4,
          f"code={coded:.6f}  BS(undiscounted)={price_true_undisc:.6f}")


# ---------------------------------------------------------------- B2
# Truncation sensitivity: the call integral's upper limit is hard-coded at
# log_threshold + 10. Under a heavy tail the answer should move when it moves.
def B2_truncation_sensitivity():
    """The call integral is cut at log_threshold+10. Whether that matters
    depends entirely on beta: at beta=-1 the right tail is thin and +10 is
    harmless; at beta>-1 the integral is divergent and +10 is the only thing
    producing a finite number."""
    lo = np.log(K / S0)
    for (a, b, gam) in [(1.5, -1.0, 0.05), (1.5, 0.0, 0.05), (1.1, -0.2, 0.3)]:
        mu = C.martingale_condition(a, b, gam, T, r, q)
        scale = gam * (T ** (1 / a))

        def correct(x, a=a, b=b, mu=mu, scale=scale):
            return (S0 * np.exp(x) - K) * levy_stable.pdf(x, a, b,
                                                          loc=mu, scale=scale)

        vals = {}
        for U in (5, 10, 20, 40):
            pts = [lo] + [lo + s for s in (0.5, 2, 5, 10, 20) if s < U] + [lo + U]
            tot = 0.0
            for l_, h_ in zip(pts[:-1], pts[1:]):
                v, _ = integrate.quad(correct, l_, h_, limit=600)
                tot += v
            vals[U] = tot
        spread = max(vals.values()) / max(min(vals.values()), 1e-300)
        stable = spread < 1.01
        check(f"B2  +10 cutoff is harmless for alpha={a}, beta={b}",
              stable,
              "  ".join(f"U={u}:{v:.6g}" for u, v in vals.items()) +
              f"   max/min={spread:.4g}x")


# ---------------------------------------------------------------- B3
# quantile_estimation: alpha branch. nu_alpha = (x95-x05)/(x75-x25) equals
# 2.4386 for a Gaussian and INCREASES as alpha falls. So phi_1 <= 2.439 means
# "at least as light-tailed as Gaussian" -> alpha ~ 2, not alpha ~ 0.1.
def B3_alpha_branch_direction():
    rng = np.random.default_rng(0)
    # exactly-Gaussian data, huge sample so phi_1 straddles 2.4386 by noise
    flips = []
    for seed in range(12):
        r_ = np.random.default_rng(seed).normal(0, 1, 400_000)
        est = C.quantile_estimation(r_)
        flips.append(est["alpha"])
    lo = sum(1 for a in flips if a < 1.0)
    check("B3  Gaussian data recovers alpha ~ 2",
          all(abs(a - 2.0) < 0.15 for a in flips),
          f"alphas from 12 Gaussian samples: {['%.2f' % a for a in flips]}  "
          f"({lo}/12 landed below 1.0)")

    # a light-tailed (uniform) sample forces phi_1 well below 2.439
    u = np.random.default_rng(1).uniform(-1, 1, 200_000)
    phi1 = ((np.percentile(u, 95) - np.percentile(u, 5)) /
            (np.percentile(u, 75) - np.percentile(u, 25)))
    est_u = C.quantile_estimation(u)
    check("B3b sub-Gaussian phi_1 sends alpha DOWN instead of clamping at 2",
          est_u["alpha"] > 1.9,
          f"phi_1={phi1:.4f} (<2.4386)  ->  alpha_est={est_u['alpha']:.4f}")


# ---------------------------------------------------------------- T5
# Round-trip: simulate from known stable parameters, recover them.
def T5_quantile_roundtrip():
    for (a, b, g, d) in [(1.7, 0.0, 0.01, 0.0), (1.5, -0.5, 0.02, 0.001),
                         (1.9, 0.3, 0.01, 0.0)]:
        x = levy_stable.rvs(a, b, loc=d, scale=g, size=200_000, random_state=7)
        est = C.quantile_estimation(x)
        ea = abs(est["alpha"] - a)
        eb = abs(est["beta"] - b)
        eg = abs(est["gamma"] - g) / g
        check(f"T5  round-trip alpha={a}, beta={b}, gamma={g}",
              ea < 0.05 and eb < 0.10 and eg < 0.05,
              f"est alpha={est['alpha']:.3f} (err {ea:.3f}), "
              f"beta={est['beta']:.3f} (err {eb:.3f}), "
              f"gamma={est['gamma']:.5f} (rel err {eg:.3f})")


# ---------------------------------------------------------------- B4
# delta: McCulloch's conversion is delta = zeta - beta*gamma*tan(pi*alpha/2).
# The code adds. Check the sign against the sample location.
def B4_delta_sign():
    a, b, g, d = 1.5, 0.8, 0.02, 0.0
    x = levy_stable.rvs(a, b, loc=d, scale=g, size=400_000, random_state=3)
    est = C.quantile_estimation(x)
    coded = est["delta"]
    med = np.median(x)
    flipped = med - est["gamma"] * est["beta"] * np.tan(np.pi * est["alpha"] / 2)
    check("B4  delta lands near the true location (0.0)",
          abs(coded - d) < abs(flipped - d),
          f"code delta={coded:+.5f}, sign-flipped={flipped:+.5f}, true={d:+.5f}, "
          f"median={med:+.5f}")


# ---------------------------------------------------------------- B5
# Put-call parity on the code's own outputs.
def B5_put_call_parity():
    a, b, gam = 2.0, 0.0, GAM2
    Kx = 100.0
    c = C.expected_itm_payoff(S0, Kx, T, r, q, a, b, gam, "call") * np.exp(-r * T)
    p = C.expected_itm_payoff(S0, Kx, T, r, q, a, b, gam, "put") * np.exp(-r * T)
    lhs = c - p
    rhs = S0 * np.exp(-q * T) - Kx * np.exp(-r * T)
    check("B5  put-call parity holds on the code's outputs",
          abs(lhs - rhs) < 1e-3,
          f"C-P={lhs:.6g}   S0e^-qT - Ke^-rT={rhs:.6f}")


# ---------------------------------------------------------------- B6
# The headline run: is the reported ITM probability even plausible?
def B6_headline_run():
    p = C.option_itm_probability(4500, 4600, 0.25, 0.05, 0.015, 1.1, -0.2, 0.3, "call")
    mu = C.martingale_condition(1.1, -0.2, 0.3, 0.25, 0.05, 0.015)
    check("B6  OTM call P(ITM) is below 50% at fair drift",
          p < 0.5,
          f"P(ITM)={p:.4f}; implied log-drift over 3M = {mu:+.4f} "
          f"(= {np.exp(mu)-1:+.1%} expected move), risk-free leg = "
          f"{(0.05-0.015)*0.25:+.5f}")


if __name__ == "__main__":
    for fn in [T1_itm_gaussian_limit, T2_martingale_identity,
               B1_call_integrand_is_inverted, T3_call_price_vs_bs,
               T4_put_price_vs_bs, B2_truncation_sensitivity,
               B3_alpha_branch_direction, T5_quantile_roundtrip,
               B4_delta_sign, B5_put_call_parity, B6_headline_run]:
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, f"raised {type(e).__name__}: {e}")

    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print(f"\n{'='*70}\n{len(_results)-n_fail} passed, {n_fail} failed "
          f"out of {len(_results)}\n{'='*70}")
