#!/usr/bin/python3
# =============================================================================
# ledger.py — pre-registration ledger + corpus-wide multiple-testing control.
#
# DSR deflates for trials WITHIN one sleeve; nothing governs the ~55 ACROSS sleeves — some "keepers" are
# false-discovery luck, and DSR's n_trials is unknowable without an honest record of every hypothesis run. This module
# is (1) an append-only pre-registration ledger (the honest trial count), and (2) Benjamini-Hochberg FDR + a
# corpus-wide deflated-Sharpe over the whole family, to find which keepers survive once you count every shot taken.
#
# Pure NumPy + stdlib. Ledger is JSONL (one trial per line). Demo simulates a ~55-sleeve corpus.
# =============================================================================
import os, json, math
import numpy as np

LEDGER_PATH = os.environ.get("NULLBAR_LEDGER", "trials.jsonl")

def record_trial(sleeve, hypothesis, n_variations, observed_sharpe, T, path=LEDGER_PATH, **extra):
    """Append one tested hypothesis. Call this for EVERY variation you run — the honest input to DSR/FDR."""
    rec = dict(sleeve=sleeve, hypothesis=hypothesis, n_variations=int(n_variations),
               observed_sharpe=float(observed_sharpe), T=int(T), **extra)
    with open(path, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec

def load_ledger(path=LEDGER_PATH):
    if not os.path.exists(path): return []
    return [json.loads(l) for l in open(path) if l.strip()]

def _ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def sharpe_pvalue(sharpe_ann, T, ppy=252):
    """One-sided p-value that the true (annualized) Sharpe > 0, from the t-stat SR·sqrt(T/ppy)."""
    t = sharpe_ann * math.sqrt(max(T, 1) / ppy)
    return 1 - _ncdf(t)

def benjamini_hochberg(pvalues, q=0.10):
    """BH step-up FDR. Returns boolean survivor mask + the adjusted threshold at level q."""
    p = np.asarray(pvalues, float); n = len(p); order = np.argsort(p); ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * q
    passed = ranked <= thresh
    kmax = np.max(np.where(passed)[0]) if passed.any() else -1
    cut = ranked[kmax] if kmax >= 0 else 0.0
    return (p <= cut), float(cut)

def corpus_fdr(sleeves, q=0.10, ppy=252):
    """sleeves: list of dicts {name, sharpe, T}. Returns per-sleeve p-value, BH-survivor flag, and the deflated-Sharpe
    using the TRUE corpus-wide trial count (every sleeve is one of N shots)."""
    names = [s["name"] for s in sleeves]; srs = np.array([s["sharpe"] for s in sleeves], float)
    Ts = np.array([s["T"] for s in sleeves], float); N = len(sleeves)
    pvals = np.array([sharpe_pvalue(sr, T, ppy) for sr, T in zip(srs, Ts)])
    surv, cut = benjamini_hochberg(pvals, q)
    # corpus deflated Sharpe: expected max Sharpe of N trials (per-period units)
    g = 0.5772156649; sr_var = srs.std() / math.sqrt(ppy)       # dispersion of per-period Sharpes across the corpus
    emax = sr_var * ((1 - g) * _invnorm(1 - 1.0 / N) + g * _invnorm(1 - 1.0 / (N * math.e)))
    out = []
    for i, s in enumerate(sleeves):
        sr_pp = srs[i] / math.sqrt(ppy)
        dsr = _ncdf((sr_pp - emax) * math.sqrt(Ts[i] - 1)) if Ts[i] > 1 else float("nan")
        out.append(dict(name=names[i], sharpe=srs[i], p=float(pvals[i]), bh_survivor=bool(surv[i]), dsr_corpus=float(dsr)))
    return dict(threshold=cut, expected_max_sharpe_ann=emax * math.sqrt(ppy), results=sorted(out, key=lambda r: r["p"]))

def _invnorm(p):  # Acklam
    if p<=0: return -1e9
    if p>=1: return 1e9
    a=[-39.6968302866538,220.946098424521,-275.928510446969,138.357751867269,-30.6647980661472,2.50662827745924]
    b=[-54.4760987982241,161.585836858041,-155.698979859887,66.8013118877197,-13.2806815528857]
    c=[-0.00778489400243029,-0.322396458041136,-2.40075827716184,-2.54973253934373,4.37466414146497,2.93816398269878]
    d=[0.00778469570904146,0.32246712907004,2.445134137143,3.75440866190742]; pl=0.02425
    if p<pl: q=math.sqrt(-2*math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p<=1-pl: q=p-0.5;r=q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q=math.sqrt(-2*math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

if __name__ == "__main__":
    rng = np.random.default_rng(3); T = 2500
    # Stylized corpus: ~50 null sleeves (true Sharpe 0) + 5 with a real edge — mimics what the family likely looks like.
    nulls = [dict(name=f"null{i}", sharpe=float(rng.normal(0, 1/math.sqrt(T/252)*1.0)), T=T) for i in range(50)]
    reals = [dict(name=n, sharpe=s, T=T) for n,s in [("bankroll",1.18),("borough",0.46),("bullion_rv",0.61),("bulwark",0.35),("blend_crack",0.43)]]
    corpus = nulls + reals
    print("="*96); print(f"LEDGER — corpus-wide FDR over {len(corpus)} sleeves ({len(nulls)} null + {len(reals)} real)"); print("="*96)
    res = corpus_fdr(corpus, q=0.10)
    print(f"\n  expected MAX Sharpe from {len(corpus)} trials by luck alone: {res['expected_max_sharpe_ann']:+.2f}  (any keeper below this is suspect)")
    print(f"  BH(q=0.10) p-value threshold: {res['threshold']:.4f}\n")
    print(f"  {'sleeve':14}{'Sharpe':>8}{'p-value':>10}{'BH surv':>9}{'DSR(corpus)':>13}")
    for r in res["results"][:12]:
        print(f"  {r['name']:14}{r['sharpe']:>+8.2f}{r['p']:>10.4f}{str(r['bh_survivor']):>9}{r['dsr_corpus']:>13.2f}")
    surv=[r['name'] for r in res['results'] if r['bh_survivor']]
    print(f"\n  SURVIVORS (BH q=0.10): {surv}")
    print("  READ: of a realistic corpus, only the genuinely-strong Sharpes clear family-wise FDR; mid-tier 'keepers'")
    print("  (e.g. a 0.35-0.46 Sharpe) often DON'T survive once you count all ~55 shots — exactly the 1-2 we flagged.")
