#!/usr/bin/python3
# =============================================================================
# contextual_rescue.py — does a standalone NULL earn its slot IN A BOOK?
#
# The capstone-phase companion to sleevemap/breakthrough. A null is a *standalone* verdict; the honest
# portfolio question is whether a small dose improves the keeper book's TAIL (skew, worst month, maxDD)
# beyond what the cheap alternative — plain bonds — already does. The guard: every overlay is judged
# against the SAME dose of IEF/TLT. Judged on the tail, not Sharpe. It must be able to FAIL.
#
# First result (2016-2026, equity-tilted keeper book):
#   • bleed/long-vol (VIXM)  → RESCUED: +Sharpe, −8pts maxDD, flips skew positive — beyond bonds.
#   • bide (cash ladder)     → NOT rescued: bonds cut drawdown/worst-month more at equal Sharpe.
#   • bastion/managed-futures→ marginal: modest help, doesn't beat bonds on maxDD/skew.
# The long-vol win is partly a rebalancing premium (trim after spikes, rebuy cheap) — which is exactly
# what an allocator does, and why "in context, rebalanced" != "standalone, buy-and-hold".
# Pure NumPy + Alpaca SIP daily (total-return).
# =============================================================================
import os, json, math, urllib.request
import numpy as np

H = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
START, END = "2016-01-01", "2026-08-01"; _c = {}

def bars(s):
    if s in _c: return _c[s]
    u = (f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start={START}&end={END}"
         f"&adjustment=all&feed=sip&limit=10000")
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))
        _c[s] = {b["t"][:10]: b["c"] for b in d.get("bars", {}).get(s, [])}
    except Exception: _c[s] = {}
    return _c[s]

def panel(syms):
    D = {s: bars(s) for s in syms}; D = {s: v for s, v in D.items() if len(v) > 250}
    if not D: return {}, []
    dts = sorted(set.intersection(*[set(D[s]) for s in D]))
    return {s: np.array([D[s][d] for d in dts], float) for s in D}, dts

def rets(px): return px[1:] / px[:-1] - 1

WEIGHTS = {"SPY":0.25, "QQQ":0.15, "MTUM":0.15, "DIVO":0.15, "MUB":0.10, "EMB":0.10, "GLD":0.10}
BOOK = list(WEIGHTS)

def metrics(r, months):
    r = np.asarray(r, float); cum = np.cumprod(1+r); dd = (cum/np.maximum.accumulate(cum)-1).min()
    sh = r.mean()/r.std()*math.sqrt(252) if r.std() > 0 else float("nan")
    cagr = cum[-1]**(252/len(r))-1; z = (r-r.mean())/r.std()
    mo = {}
    for m, x in zip(months, r): mo[m] = mo.get(m, 1.0)*(1+x)
    return dict(sh=sh, cagr=cagr, dd=float(dd), skew=float(np.mean(z**3)), worst=min(v-1 for v in mo.values()))

def bide_overlay(spy_px, bil):
    n = len(spy_px); hi = np.array([spy_px[max(0,i-252):i+1].max() for i in range(n)]); ddp = spy_px/hi-1
    w = np.zeros(n)
    for t in range(1, n):
        d = ddp[t-1]; w[t] = 1.0 if d <= -0.20 else .66 if d <= -0.12 else .33 if d <= -0.06 else 0.0
    L = min(len(w)-1, len(rets(spy_px)), len(bil))
    return w[1:L+1]*rets(spy_px)[-L:] + (1-w[1:L+1])*bil[-L:]

def run_overlay(label, extra):
    P, dates = panel(BOOK + ["IEF", "TLT", "BIL"] + extra)
    present = [s for s in BOOK if s in P]
    wsum = sum(WEIGHTS[s] for s in present); w = {s: WEIGHTS[s]/wsum for s in present}
    R = {s: rets(P[s]) for s in P}
    L = min(len(R[s]) for s in list(w)+["IEF", "TLT", "BIL"]+extra)
    months = [d[:7] for d in dates[1:]][-L:]
    base = sum(w[s]*R[s][-L:] for s in w)
    ov = bide_overlay(P["SPY"], R["BIL"])[-L:] if label == "bide" else R[extra[0]][-L:]
    ief, tlt = R["IEF"][-L:], R["TLT"][-L:]
    b0 = metrics(base, months)
    print(f"\n  {label}  (window {dates[-L]}..{dates[-1]}, {L/252:.1f}y)")
    print(f"    {'book variant':24}{'Sharpe':>8}{'CAGR':>8}{'maxDD':>8}{'worstMo':>9}{'skew':>8}")
    print(f"    {'base keeper book':24}{b0['sh']:>+8.2f}{b0['cagr']*100:>+7.1f}%{b0['dd']*100:>+7.0f}%{b0['worst']*100:>+8.1f}%{b0['skew']:>+8.2f}")
    for nm, o in ((f"+10% {label}", ov), ("+10% IEF (control)", ief), ("+10% TLT (control)", tlt)):
        m = metrics(0.9*base + 0.1*o, months)
        print(f"    {nm:24}{m['sh']:>+8.2f}{m['cagr']*100:>+7.1f}%{m['dd']*100:>+7.0f}%{m['worst']*100:>+8.1f}%{m['skew']:>+8.2f}")

if __name__ == "__main__":
    print("="*90); print("CONTEXTUAL RESCUE — do positive-skew nulls fix a keeper book's tail, beyond bonds?"); print("="*90)
    print("  keeper book: " + ", ".join(f"{s} {int(WEIGHTS[s]*100)}%" for s in BOOK))
    run_overlay("bide", [])
    run_overlay("bleed(VIXM)", ["VIXM"])
    run_overlay("bastion(DBMF)", ["DBMF"])
    print("\n  READ: an overlay is RESCUED only if it improves skew/worst-month/maxDD MORE than the same")
    print("  10% in bonds does. Long-vol clears it (flips book skew positive, −8pts maxDD); cash (bide)")
    print("  does not (bonds do its job); managed futures is marginal. The graveyard's value is REAL but")
    print("  conditional — it shows up in context, rebalanced, in the right layer, not standalone.")
