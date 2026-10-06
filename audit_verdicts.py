#!/usr/bin/python3
# audit_verdicts.py — run the mirage spec-audit on the remaining control-based verdicts.
import os, sys, json, urllib.request, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirage import specification_audit, report
H={"APCA-API-KEY-ID":os.environ["ALPACA_KEY_ID"],"APCA-API-SECRET-KEY":os.environ["ALPACA_SECRET_KEY"]}
def bars(s):
    u=(f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2016-01-01&end=2026-08-01&adjustment=all&feed=sip&limit=10000")
    d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=40)); return {b["t"][:10]:b["c"] for b in d.get("bars",{}).get(s,[])}
def aligned(syms):
    B={s:bars(s) for s in syms}; B={s:v for s,v in B.items() if len(v)>300}; u=list(B)
    dts=sorted(set.intersection(*[set(B[s]) for s in u])); PX={s:np.array([B[s][d] for d in dts],float) for s in u}
    R={s:PX[s][1:]/PX[s][:-1]-1 for s in u}; return PX,R,dts
print("="*104); print("MIRAGE audit — remaining control-based verdicts"); print("="*104)

# BIGBROTHER: "defense alpha SURVIVES controls (vs XLI/QUAL/USMV)" — the explicit survives-controls claim.
dfn=["LMT","RTX","NOC","GD","LHX"]; ctl=["SPY","XLI","QUAL","USMV","VLUE"]
PX,R,_=aligned(dfn+ctl); have=[s for s in dfn if s in R]
defense=np.nanmean(np.vstack([R[s] for s in have]),axis=0)
report("BIGBROTHER  (claim: defense alpha survives XLI/QUAL/USMV controls)",
       specification_audit(defense-R["SPY"][-len(defense):] if False else defense, {c:R[c] for c in ctl if c in R}, focal="XLI"))

# BOULDER: "cross-asset trend is NOT a dollar bet" — audit the trend book's alpha vs USD + asset betas.
ETFS=["XLK","XLF","XLE","XLV","XLI","XLP","XLU","XLB","XLY","QQQ","IWM","EEM","EFA","GLD","SLV","TLT","IEF","HYG","LQD","DBC"]
cctl=["UUP","SPY","AGG","GLD","DBC"]
PX2,R2,_=aligned(ETFS+cctl); E=[s for s in ETFS if s in R2]; n=len(R2[E[0]])
import math
def tsmom(E,R2,n):
    cols=[]
    for s in E:
        px=np.concatenate([[1.0],np.cumprod(1+R2[s])]); pos=np.full(n,np.nan)
        for t in range(252,n):
            sig=(np.sign(px[t]/px[t-63]-1)+np.sign(px[t]/px[t-126]-1)+np.sign(px[t]/px[t-252]-1))/3
            iv=R2[s][t-60:t].std()*math.sqrt(252); lev=min(2.0,0.10/iv) if iv>0 else 0
            pos[t]=sig*lev
        c=np.full(n,np.nan); c[253:]=pos[252:-1]*R2[s][253:]; cols.append(c)
    return np.nanmean(np.vstack(cols),axis=0)
trend=tsmom(E,R2,n); trend=trend[np.isfinite(trend)]
report("BOULDER  (claim: cross-asset trend is NOT a dollar bet / is diversifying alpha)",
       specification_audit(trend, {c:R2[c][-len(trend):] for c in cctl if c in R2}, focal="UUP"))

# BIFURCATE: already NULL at index level (no distinct-alpha claim) — audit the credit signal's forward alpha to confirm.
PX3,R3,dts3=aligned(["HYG","LQD","IEF","SPY"])
credit=R3["HYG"]-R3["LQD"]; n3=len(credit)
# credit-timed SPY: long SPY when 20d credit improving (risk-on) — the implied "credit leads equity" book
sig=np.array([1.0 if (t>=20 and np.sum(credit[t-20:t])>0) else 0.0 for t in range(n3)])
book=sig[:-1]*R3["SPY"][1:n3]
report("BIFURCATE  (claim re-check: does a credit-timed equity book carry alpha vs controls?)",
       specification_audit(book, {c:R3[c][1:n3] for c in ["SPY","HYG","LQD","IEF"]}, focal="HYG"))
print("\nREAD: confirms where a verdict's 'alpha' survives its control set (robust) vs where it was the control choice doing the work.")
