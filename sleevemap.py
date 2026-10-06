#!/usr/bin/python3
# =============================================================================
# sleevemap.py — sleeve-correlation & factor-exposure map (is the book actually diversified?).
#
# ~55 sleeves, but many are momentum/carry/vol/credit variants. Before allocating, prove the keeper book isn't secretly
# ONE bet. Two views:
#   • correlation_map     — NxN correlation + an effective-number-of-bets (1/ΣΣ wρw at equal weight) + which sleeves cluster
#   • factor_exposure     — regress the combined book on {MKT, momentum, value, low-vol, credit, trend} → how much of the
#                           "alpha" is just factor beta, and what residual (true diversifying alpha) survives
# Pure NumPy. Demo pulls a handful of this session's sleeve PROXIES + factor ETFs.
# =============================================================================
import numpy as np

def correlation_map(returns_dict):
    names=list(returns_dict); L=min(len(returns_dict[n]) for n in names)
    R=np.vstack([np.asarray(returns_dict[n],float)[-L:] for n in names]); C=np.corrcoef(R)
    w=np.ones(len(names))/len(names)
    eff_bets=1.0/float(w@C@w)                                   # effective number of independent bets (equal weight)
    # crude cluster: greedily group names with pairwise corr > 0.5
    clusters=[]; used=set()
    for i,ni in enumerate(names):
        if ni in used: continue
        grp=[ni]; used.add(ni)
        for j,nj in enumerate(names):
            if nj not in used and C[i,j]>0.5: grp.append(nj); used.add(nj)
        clusters.append(grp)
    return dict(names=names, corr=C, eff_bets=eff_bets, avg_corr=float((C.sum()-len(names))/(len(names)**2-len(names))), clusters=clusters)

def factor_exposure(book_rets, factors_dict):
    """OLS of the combined book on factor returns. Returns betas, R² (how much is factor beta), and the annualized
    residual alpha (the diversifying part not explained by factors)."""
    y=np.asarray(book_rets,float); names=list(factors_dict); L=min([len(y)]+[len(factors_dict[f]) for f in names])
    y=y[-L:]; X=np.column_stack([np.ones(L)]+[np.asarray(factors_dict[f],float)[-L:] for f in names])
    beta,*_=np.linalg.lstsq(X,y,rcond=None); resid=y-X@beta
    r2=1-resid.var()/y.var() if y.var()>0 else float("nan")
    return dict(alpha_ann=float(beta[0]*252), betas={f:float(b) for f,b in zip(names,beta[1:])},
                factor_r2=float(r2), resid_sharpe=float(resid.mean()/resid.std()*np.sqrt(252)) if resid.std()>0 else float("nan"))

if __name__=="__main__":
    import os,json,urllib.request,math
    H={"APCA-API-KEY-ID":os.environ["ALPACA_KEY_ID"],"APCA-API-SECRET-KEY":os.environ["ALPACA_SECRET_KEY"]}
    def bars(s):
        u=(f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2018-01-01&end=2026-08-01&adjustment=all&feed=sip&limit=10000")
        d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=40)); return {b["t"][:10]:b["c"] for b in d.get("bars",{}).get(s,[])}
    def rets(s): b=bars(s); dts=sorted(b); px=np.array([b[d] for d in dts]); return dict(zip(dts[1:],px[1:]/px[:-1]-1))
    def al(syms):
        R={s:rets(s) for s in syms}; dts=sorted(set.intersection(*[set(R[s]) for s in syms])); return {s:np.array([R[s][d] for d in dts]) for s in syms}
    print("="*96); print("SLEEVEMAP — correlation & factor-exposure of the keeper book (demo)"); print("="*96)
    # sleeve PROXIES (long-only stand-ins for the directional tilt of several keepers) + factors
    syms=["MTUM","QUAL","USMV","PFF","GDX","EMB","MUB","SPY","VLUE","HYG","TLT","DBC"]
    A=al(syms)
    sleeves={"bulwark~MTUM":A["MTUM"],"burdensome~(credit+eq)":0.5*A["HYG"]+0.5*A["SPY"],"bowed~PFF":A["PFF"],
             "bullion~GDX":A["GDX"],"bazaar~EMB":A["EMB"],"borough~MUB":A["MUB"]}
    cm=correlation_map(sleeves)
    print(f"\n  effective number of independent bets (6 sleeves, equal wt): {cm['eff_bets']:.1f}   avg pairwise corr {cm['avg_corr']:+.2f}")
    print(f"  clusters (corr>0.5): {cm['clusters']}")
    book=np.mean(np.vstack(list(sleeves.values())),axis=0)
    fx=factor_exposure(book, {"MKT":A["SPY"],"MOM":A["MTUM"]-A["SPY"],"VAL":A["VLUE"]-A["SPY"],"LOWVOL":A["USMV"]-A["SPY"],"CREDIT":A["HYG"]-A["TLT"],"COMMOD":A["DBC"]})
    print(f"\n  combined book vs factors:  factor-R² {fx['factor_r2']*100:.0f}%   residual alpha {fx['alpha_ann']*100:+.1f}%/yr   residual Sharpe {fx['resid_sharpe']:+.2f}")
    print("   betas: "+", ".join(f"{f} {b:+.2f}" for f,b in fx["betas"].items()))
    print("\nREAD: if effective-bets ≪ sleeve-count and factor-R² is high, the 'diversified' book is mostly factor beta in")
    print("  disguise. Allocate on the RESIDUAL (diversifying) alpha, and collapse correlated clusters to one risk budget.")
