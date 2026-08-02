import numpy as np, time
from scipy.stats import levy_stable, norm
import calculation_fixed as F

S0,K,T,r,q,sigma = 100.,110.,0.25,0.05,0.02,0.20
G2 = sigma/np.sqrt(2)
def bs(S0,K,T,r,q,s,kind="call"):
    d1=(np.log(S0/K)+(r-q+0.5*s*s)*T)/(s*np.sqrt(T)); d2=d1-s*np.sqrt(T)
    if kind=="call": return S0*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S0*np.exp(-q*T)*norm.cdf(-d1)

print("T3b call vs BS:")
c=F.expected_itm_payoff(S0,K,T,r,q,2.,0.,G2,'call'); t=bs(S0,K,T,r,q,sigma)*np.exp(r*T)
print(f"   fixed={c:.8f}  BS={t:.8f}  relerr={abs(c-t)/t:.2e}")

print("B5 put-call parity (K=100):")
c=F.expected_itm_payoff(S0,100.,T,r,q,2.,0.,G2,'call')*np.exp(-r*T)
p=F.expected_itm_payoff(S0,100.,T,r,q,2.,0.,G2,'put')*np.exp(-r*T)
print(f"   C-P={c-p:.8f}   S0e^-qT-Ke^-rT={S0*np.exp(-q*T)-100*np.exp(-r*T):.8f}")

print("T2 martingale, beta=-1 (only well-posed case):")
for a in (1.9,1.7,1.5,1.3):
    mu=F.martingale_condition(a,-1.,0.05,T,r,q); s=0.05*T**(1/a)
    from scipy import integrate
    tot=0
    for l,h in zip([-300,-30,-5,0,5,30],[-30,-5,0,5,30,200]):
        tot+=integrate.quad(lambda z: np.exp(mu+s*z)*levy_stable.pdf(z,a,-1.),l,h,limit=800)[0]
    print(f"   alpha={a}: E[S_T]/S0={tot:.8f}  target={np.exp((r-q)*T):.8f}  relerr={abs(tot-np.exp((r-q)*T))/np.exp((r-q)*T):.2e}")

print("T2 beta>-1 now refuses instead of returning a truncation artifact:")
for a,b in [(1.5,0.0),(1.1,-0.2)]:
    try: F.martingale_condition(a,b,0.05,T,r,q); print(f"   alpha={a},beta={b}: RETURNED A NUMBER (bad)")
    except ValueError as e: print(f"   alpha={a},beta={b}: ValueError -> {str(e)[:70]}...")

print("B3 Gaussian data, 12 seeds (was flipping 1.10 <-> 2.00):")
al=[F.quantile_estimation(np.random.default_rng(s).normal(0,1,200_000))['alpha'] for s in range(12)]
print("   ", [f"{a:.2f}" for a in al])

print("T5 round-trip:")
for (a,b,g) in [(1.7,0.0,0.01),(1.5,-0.5,0.02),(1.9,0.3,0.01),(1.3,0.6,0.05)]:
    x=levy_stable.rvs(a,b,loc=0.,scale=g,size=300_000,random_state=7)
    e=F.quantile_estimation(x)
    print(f"   true(a={a},b={b:+.1f},g={g}) -> est(a={e['alpha']:.3f},b={e['beta']:+.3f},"
          f"g={e['gamma']:.5f},d={e['delta']:+.5f})  |da|={abs(e['alpha']-a):.3f} "
          f"|db|={abs(e['beta']-b):.3f} relg={abs(e['gamma']-g)/g:.3f}")
