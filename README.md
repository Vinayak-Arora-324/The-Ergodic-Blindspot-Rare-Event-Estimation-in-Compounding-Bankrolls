# The Main Question

## In human language

The document is careful to say the project is *not* "does tail hedging work?" The main question is one level up:

**"When the truth about a strategy lives in rare events, can the simulation I can actually afford to run tell me that truth — or does my answer secretly depend on which estimator I chose?"**

The concrete, motivating version: a tail-hedge (buying deep out-of-the-money Nifty puts) bleeds a small premium almost every day and pays off enormously in a rare crash. Whether the strategy *compounds* a real bankroll or *drains* it over years is decided almost entirely by those rare paths. So if your Monte Carlo simulation, at an affordable number of samples, systematically mis-measures rare events — which crude Monte Carlo does — then it can't honestly tell you whether the hedge is insurance or a money-pit.

The proposed finding is that the verdict *flips* depending on estimator choice: "does it compound or bleed?" gets a different answer from crude MC than from a proper rare-event estimator (importance sampling / SMC), at the same computational budget. The deliverable is honesty about the estimator, not a verdict on the strategy.

## In mathematical language

Fix a model: asset dynamics follow a tempered-stable (CGMY) process, and a fixed hedging rule generates a wealth path $W_0, W_1, \dots, W_T$ for a finite bankroll.

The quantity you actually care about is the **time-average (geometric) growth rate**:

$$g = \lim_{T\to\infty} \frac{1}{T}\,\log\frac{W_T}{W_0}$$

This is the ergodic object — path-dependent and, under fat tails, *not* equal to the ensemble average $\log \mathbb{E}[W_T/W_0]$; the two diverge, which is the ergodicity-economics point.

You can't compute $g$ analytically, so you estimate it by simulating $N$ paths: call the estimator $\hat{g}_N$. The main question is then:

$$\text{What are the bias, variance, and relative error of } \hat{g}_N \text{ at affordable } N,$$

$$\text{under crude Monte Carlo versus a rare-event (SMC / importance-sampling) estimator?}$$

And the sharpest form of it: **does $\operatorname{sign}(\hat{g}_N)$ agree with $\operatorname{sign}(g)$?** — because the sign is precisely "compounds vs. bleeds."

### Why this is nontrivial

$g$ is dominated by events of small probability $p$ (the crash payoffs). For crude Monte Carlo, the relative error of estimating a probability-$p$ event scales like

$$\sqrt{\frac{1-p}{Np}}$$

which explodes as $p \to 0$ at fixed budget $N$ — most runs see *zero* crashes and report pure premium-bleed. Rare-event estimators exist to control exactly this. So the claim under study is:

$$\hat{g}_N^{\text{crude}} < 0 < \hat{g}_N^{\text{rare-event}} \quad \text{(or vice versa) at the same } N$$

i.e., the strategy's apparent viability is an artifact of estimator choice — an estimator-honesty result, with the finance question as the test case rather than the object of study.

### The closing caveat

Even the best estimator only answers the question *inside the model*. The fully honest final quantity is the sensitivity of the answer — how much $\hat{g}$ moves as you vary estimator and model — a bracketing of your own ignorance rather than a point verdict.
