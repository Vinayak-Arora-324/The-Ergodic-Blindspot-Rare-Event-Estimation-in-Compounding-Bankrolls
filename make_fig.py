import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import levy_stable
import Calculation as C, calculation_fixed as F

S0,K,T,r,q = 100.,110.,0.25,0.05,0.02
fig,ax=plt.subplots(1,2,figsize=(13,5))

# --- A: truncation sensitivity -------------------------------------------
lo=np.log(K/S0)
Us=np.array([2,4,6,8,10,14,20,28,40])
for (a,b,g,lab,col,ls) in [(1.5,-1.0,0.05,r"$\beta=-1$  (finite mean)","#1a7f5a","-"),
                           (1.5, 0.0,0.05,r"$\beta=0$   (infinite mean)","#c1440e","-"),
                           (1.1,-0.2,0.30,r"$\alpha{=}1.1,\beta{=}-0.2$  (script default)","#7b2d8e","--")]:
    mu=C.martingale_condition(a,b,g,T,r,q); sc=g*T**(1/a)
    f=lambda x: (S0*np.exp(x)-K)*levy_stable.pdf(x,a,b,loc=mu,scale=sc)
    ys=[]
    for U in Us:
        pts=[lo]+[lo+s for s in (0.25,1,3,8,20) if s<U]+[lo+U]
        ys.append(sum(integrate.quad(f,l,h,limit=600)[0] for l,h in zip(pts[:-1],pts[1:])))
    ax[0].plot(Us,np.maximum(ys,1e-8),ls,color=col,lw=2,marker='o',ms=4,label=lab)
ax[0].axvline(10,color='k',lw=1.2,ls=':',zorder=0)
ax[0].annotate("hard-coded cutoff\n(log_threshold + 10)",xy=(10,4e3),xytext=(15,1e1),
               fontsize=9,arrowprops=dict(arrowstyle='->',lw=1.2))
ax[0].set_yscale("log"); ax[0].set_xlabel("upper integration limit, in log-return units above the strike")
ax[0].set_ylabel("expected ITM call payoff")
ax[0].set_title("A. The reported price is a property of the cutoff",fontsize=11,loc='left')
ax[0].legend(fontsize=9,loc='upper left'); ax[0].grid(alpha=.25)

# --- B: alpha estimator transfer function --------------------------------
truth=np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.95,2.0])
old=[];new=[]
for a in truth:
    x=levy_stable.rvs(a,0.0,size=300_000,random_state=11)
    old.append(C.quantile_estimation(x)['alpha']); new.append(F.quantile_estimation(x)['alpha'])
ax[1].plot([1.05,2.02],[1.05,2.02],color='k',lw=1,ls='--',label="exact recovery")
ax[1].plot(truth,old,'o-',color="#c1440e",lw=2,ms=6,label="Calculation.py")
ax[1].plot(truth,new,'s-',color="#1a7f5a",lw=2,ms=5,label="grid-inverted McCulloch")
ax[1].axhline(1.1,color="#c1440e",lw=1,ls=':',alpha=.7)
ax[1].text(1.30,1.135,"np.clip floor at 1.1 — pinned here for every true $\\alpha\\leq1.6$",
           fontsize=8.5,color="#c1440e")
ax[1].annotate("true $\\alpha=2$ (Gaussian) collapses to 1.10:\nthe branch test flips sign at $\\nu_\\alpha=2.4386$",
           xy=(2.0,1.11),xytext=(1.55,1.62),fontsize=8.5,color="#c1440e",
           arrowprops=dict(arrowstyle='->',lw=1.2,color="#c1440e"))
ax[1].set_xlabel(r"true $\alpha$"); ax[1].set_ylabel(r"estimated $\alpha$")
ax[1].set_title(r"B. The $\alpha$ estimator is saturated, not noisy",fontsize=11,loc='left')
ax[1].legend(fontsize=9,loc='upper left'); ax[1].grid(alpha=.25)

fig.suptitle("Calculation.py — two failure modes that return a confident finite number",
             fontsize=12.5,y=0.99)
fig.text(0.5,0.005,"Panel A: for $\\beta>-1$ the exponential $\\alpha$-stable model has "
         "$E[S_T]=\\infty$; the integral does not converge, so the +10 cutoff is the only "
         "thing making the answer finite.",ha='center',fontsize=8.5,style='italic')
plt.tight_layout(rect=[0,0.03,1,0.97]); plt.savefig("calculation_audit.png",dpi=150)
print("wrote calculation_audit.png")
print("old:",[f"{v:.2f}" for v in old]); print("new:",[f"{v:.2f}" for v in new])
