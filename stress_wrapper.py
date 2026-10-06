#!/usr/bin/python3
# =============================================================================
# stress_wrapper.py — regime-filter / inversion wrapper (the ACTING complement to stress.py's diagnostics).
#
# stress.py MEASURES regime fragility; this one ACTS on it — a functional wrapper that intercepts the raw signal before
# the optimizer, to implement the negative tests (filter/invert/asymmetric) WITHOUT rewriting a sleeve's alpha logic.
#
#   PassThrough  — no change
#   FilterOnly   — zero the signal unless the regime condition holds (trade only in-regime)
#   InvertOnly   — flip signal sign when the regime holds (e.g. invert bidirectional when VIX rising)
#   Asymmetric   — longs only out-of-regime, shorts only in-regime (e.g. only short in high-VIX)
#
# Pure NumPy. A RegimeFilter pairs a name, an indicator (bool per bar), and a mode.
# =============================================================================
import numpy as np

PASS_THROUGH, FILTER_ONLY, INVERT_ONLY, ASYMMETRIC = "PassThrough", "FilterOnly", "InvertOnly", "Asymmetric"

class RegimeFilter:
    def __init__(self, name, regime_mask, mode=FILTER_ONLY):
        self.name=name; self.regime=np.asarray(regime_mask,bool); self.mode=mode

def apply_stress(raw_signal, rf: RegimeFilter):
    """Transform a raw signal (>0 long, <0 short, 0 cash) by the regime filter. Returns the stressed signal."""
    s=np.array(raw_signal,float); g=rf.regime[-len(s):]; out=s.copy()
    if rf.mode==PASS_THROUGH: return out
    if rf.mode==FILTER_ONLY:
        out[~g]=0.0                                             # trade only when regime active
    elif rf.mode==INVERT_ONLY:
        out[g]=-out[g]                                         # flip sign in-regime
    elif rf.mode==ASYMMETRIC:
        out[(~g)&(s<0)]=0.0                                   # no shorts out-of-regime
        out[(g)&(s>0)]=0.0                                    # no longs in-regime  (e.g. only short in panic)
    return out

# pre-built regime indicators (return a bool mask from a market-data dict of arrays) ------------------
def rising_vol(vix_level, lookback=20):
    v=np.asarray(vix_level,float); m=np.zeros(len(v),bool)
    for t in range(lookback,len(v)): m[t]=v[t]>np.mean(v[t-lookback:t])
    return m
def high_correlation_panic(avg_pairwise_corr, pct=0.80):
    c=np.asarray(avg_pairwise_corr,float); thr=np.nanquantile(c[np.isfinite(c)],pct); return c>thr
def crowded_long(short_interest_pct, valuation_pct):
    return (np.asarray(short_interest_pct,float)>=0.90)&(np.asarray(valuation_pct,float)>=0.90)

if __name__=="__main__":
    rng=np.random.default_rng(7); n=1000
    sig=np.sign(rng.normal(size=n))                            # a raw long/short signal
    vix=50+np.cumsum(rng.normal(0,1,n))                        # a wandering vol level
    rf_filter=RegimeFilter("rising-vol FILTER", rising_vol(vix), FILTER_ONLY)
    rf_invert=RegimeFilter("rising-vol INVERT", rising_vol(vix), INVERT_ONLY)
    rf_asym =RegimeFilter("panic ASYMMETRIC (short-only in-regime)", rising_vol(vix), ASYMMETRIC)
    print("="*90); print("STRESS-WRAPPER — regime filter / inversion (demo)"); print("="*90)
    for rf in (rf_filter,rf_invert,rf_asym):
        out=apply_stress(sig,rf); g=rf.regime
        print(f"\n  {rf.name}:")
        print(f"    in-market fraction {np.mean(np.abs(out)>0)*100:4.0f}%   (regime active {np.mean(g)*100:.0f}% of bars)")
        print(f"    in-regime: longs {int(np.sum((g)&(out>0)))}, shorts {int(np.sum((g)&(out<0)))}  | out-regime: longs {int(np.sum((~g)&(out>0)))}, shorts {int(np.sum((~g)&(out<0)))}")
    print("\nREAD: implement the negative tests (bevy crowded-long inversion, bidirectional fade-only-in-tightening, bend")
    print("  trade-only-in-panic) by wrapping the raw signal — no change to alpha logic. stress.py measures; this acts.")
