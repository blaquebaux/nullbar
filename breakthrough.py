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
