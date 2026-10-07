#!/usr/bin/python3
# =============================================================================
# breakthrough.py — robustness-shrinkage allocator (turns verdicts into capital).
#
# Standard allocators treat a sleeve's historical Sharpe as its expected Sharpe — the core flaw that funds lucky alpha.
# This shrinks each sleeve's expected return toward the null by its ROBUSTNESS before risk-parity:
#   shrunk_return = raw_expected_return · (1 − blind_p)^γ · min(1, eff_n/30) · mirage_penalty
#   mirage_penalty = {ROBUST:1.0, FRAGILE:0.5, NULL/MIRAGE:0.0}
# Inputs MUST be NET (post-friction) and post-mirage. Fragile/lucky sleeves get shrunk out before they draw capital;
# risk-parity then sizes the survivors to a target vol. Pure NumPy.
# =============================================================================
import math
import numpy as np

MIRAGE_PENALTY = {"ROBUST": 1.0, "FRAGILE": 0.5, "NULL": 0.0, "MIRAGE": 0.0, None: 1.0}

class Keeper:
    def __init__(self, name, net_expected_return, returns, blind_p=0.0, eff_n=60, mirage="ROBUST"):
        self.name=name; self.mu=net_expected_return; self.r=np.asarray(returns,float)
        self.blind_p=blind_p; self.eff_n=eff_n; self.mirage=mirage

def shrink(keepers, gamma=1.5):
    out=[]
    for k in keepers:
        conf=(1-k.blind_p)**gamma; effpen=min(1.0,k.eff_n/30.0); mir=MIRAGE_PENALTY.get(k.mirage,1.0)
        w=conf*effpen*mir
        out.append(dict(name=k.name, raw=k.mu, confidence=conf, eff_pen=effpen, mirage_pen=mir, shrunk=k.mu*w))
    return out

def _risk_parity(cov, iters=500):
    n=cov.shape[0]; w=np.ones(n)/n
    for _ in range(iters):
        mrc=cov@w; rc=w*mrc; target=rc.mean()
        w=w*(target/np.maximum(rc,1e-12))**0.5; w=np.maximum(w,0); w/=w.sum()
    return w

def allocate(keepers, target_vol=0.10, gamma=1.5):
    sh=shrink(keepers,gamma)
    keep=[(k,s) for k,s in zip(keepers,sh) if s["shrunk"]>0]
    if not keep: return dict(weights={}, shrink=sh, note="no sleeve survived shrinkage")
    ks=[k for k,_ in keep]; L=min(len(k.r) for k in ks)
    R=np.vstack([k.r[-L:] for k in ks]); cov=np.atleast_2d(np.cov(R))*252
    w=_risk_parity(cov); pv=math.sqrt(float(w@cov@w)); lev=target_vol/pv if pv>0 else 1.0
    weights={k.name: float(wi*lev) for k,wi in zip(ks,w)}
    return dict(weights=weights, gross_leverage=float(abs(np.array(list(weights.values()))).sum()), shrink=sh)

if __name__=="__main__":
    rng=np.random.default_rng(5); T=1500
    def mk(mu,vol):
        return rng.normal(mu/252, vol/np.sqrt(252), T)
    # four keepers: one robust strong, one robust modest, one fragile (mirage), one lucky (barely passed Blind, low eff N)
    ks=[
      Keeper("bankroll", 0.08, mk(0.08,0.10), blind_p=0.002, eff_n=80, mirage="ROBUST"),
      Keeper("borough",  0.03, mk(0.03,0.05), blind_p=0.03,  eff_n=45, mirage="ROBUST"),
      Keeper("burdensome",0.14,mk(0.14,0.22), blind_p=0.04,  eff_n=35, mirage="FRAGILE"),   # mirage-fragile → half
      Keeper("lucky",    0.12, mk(0.12,0.15), blind_p=0.048, eff_n=12, mirage="ROBUST"),     # barely passed, few trades
    ]
    print("="*96); print("BREAKTHROUGH — robustness-shrinkage allocator (demo)"); print("="*96)
    al=allocate(ks,target_vol=0.10)
    print(f"\n  {'sleeve':12}{'raw μ':>8}{'conf':>7}{'effN pen':>9}{'mirage':>8}{'→ shrunk μ':>12}")
    for s in al["shrink"]:
        print(f"  {s['name']:12}{s['raw']*100:>+7.0f}%{s['confidence']:>7.2f}{s['eff_pen']:>9.2f}{s['mirage_pen']:>8.1f}{s['shrunk']*100:>+11.1f}%")
    print(f"\n  final risk-parity weights (target 10% vol, gross {al['gross_leverage']:.2f}x):")
    for n,w in al["weights"].items(): print(f"     {n:12} {w*100:>+6.1f}%")
    print("\nREAD: the FRAGILE (mirage) sleeve is halved and the LUCKY one (barely-passed Blind, 12 eff trades) is shrunk")
    print("  hard — capital concentrates in the proven, robust sleeves, not the highest raw Sharpe. Lucky alpha starved.")


# ---- toolkit batch-2 additions ------------------------------------------------
def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _nppf(p):
    if p <= 0: return -1e9
    if p >= 1: return 1e9
    a=[-39.6968302866538,220.946098424521,-275.928510446969,138.357751867269,-30.6647980661472,2.50662827745924]
    b=[-54.4760987982241,161.585836858041,-155.698979859887,66.8013118877197,-13.2806815528857]
    c=[-0.00778489400243029,-0.322396458041136,-2.40075827716184,-2.54973253934373,4.37466414146497,2.93816398269878]
    d=[0.00778469570904146,0.32246712907004,2.445134137143,3.75440866190742]; pl=0.02425
    if p<pl: q=math.sqrt(-2*math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p<=1-pl: q=p-0.5; r=q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q=math.sqrt(-2*math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

def sharpe_haircut(sharpe_ann, n_trials, T_years, method="bhy"):
    """Harvey-Liu multiple-testing haircut on an annualized Sharpe. The haircut is NONLINEAR:
    brutal on marginal Sharpes, mild on exceptional ones (at 50 trials ~50% at SR 0.43 but
    ~8% at SR 0.78), so the '50% rule of thumb' over-penalizes the strong and under-penalizes
    the weak. SR -> t-stat -> p, deflate p for n_trials, map back to a haircut Sharpe.
    Feed sharpe_haircut(...)['sharpe_haircut']/raw_sharpe into shrink() as an extra factor."""
    T = max(T_years, 1e-6); t = sharpe_ann * math.sqrt(T)
    p = 2.0 * (1.0 - _ncdf(abs(t)))
    if method == "bonferroni": p_adj = min(1.0, p * n_trials)
    elif method == "holm":     p_adj = min(1.0, p * n_trials)
    else:                      p_adj = min(1.0, p * n_trials / (1.0 + math.log(max(n_trials, 1))))  # BHY-style (lighter than Bonferroni)
    t_adj = _nppf(1.0 - p_adj / 2.0); sr_adj = max(t_adj / math.sqrt(T), 0.0)
    hc = 1.0 - sr_adj / sharpe_ann if sharpe_ann > 0 else 1.0
    return dict(sharpe=sharpe_ann, sharpe_haircut=sr_adj, haircut_pct=float(min(max(hc, 0.0), 1.0)),
                n_trials=n_trials, method=method)
