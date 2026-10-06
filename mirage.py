#!/usr/bin/python3
# =============================================================================
# mirage.py — specification-error / "factor mirage" audit (companion to nullbar + validation + stress).
#
# DSR/PBO/nullbar ask "is the signal real?"; stress asks "is it robust to regime?". mirage asks the DISTINCT
# question none of them do: are the VARIABLES structurally correct, or does a bad control improve FIT while flipping
# or collapsing the COEFFICIENT? (Leamer extreme-bounds + bad-control / collinearity diagnostics.)
#
# A "factor mirage" is a control that is a CONSEQUENCE of the same force as the target (collinear with it), so adding
# it raises R²/IR while pushing the target's alpha toward zero or flipping its sign. The tell: fit ↑, structure ↓.
#
# specification_audit(target, controls) re-estimates the regression across ALL control subsets and reports:
#   • ALPHA (intercept) sign-stability + extreme bounds across specifications
#   • per-control MARGINAL effect when added: avg ΔR² vs avg Δalpha  → a mirage control has ΔR²>0 while Δalpha<0
#   • collinearity of each control with the TARGET (driver vs consequence) via |corr| and VIF
#   • verdict: ROBUST (alpha sign stable & significant across specs) vs FRAGILE / MIRAGE (flips/dies under plausible controls)
#
# Pure NumPy. The qualitative half — "why included / why excluded" economic causality — is a required per-sleeve
# RATIONALE field this can't automate; mirage audits the quantitative half (sign-stability under the control set).
# =============================================================================
import itertools, math
import numpy as np

def _ols(y, X):
    """OLS y ~ X (X already has an intercept col). Returns betas, their t-stats, R²."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta
    n, k = X.shape; dof = max(n - k, 1); s2 = float(resid @ resid) / dof
    try: xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError: xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(s2 * xtx_inv), 0)); t = beta / np.where(se > 0, se, np.nan)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return beta, t, r2

def _align(target, controls):
    names = list(controls); T = min([len(target)] + [len(controls[c]) for c in names])
    y = np.asarray(target, float)[-T:]; X = {c: np.asarray(controls[c], float)[-T:] for c in names}
    m = np.isfinite(y)
    for c in names: m &= np.isfinite(X[c])
    return y[m], {c: X[c][m] for c in names}, names

def specification_audit(target, controls, ann=252, max_controls=8, focal=None):
    """target: strategy daily returns whose alpha/distinctness is claimed. controls: {name: daily returns}.
    focal: optional control name to track the stability of its coefficient's sign."""
    y, X, names = _align(target, controls); names = names[:max_controls]
    n = len(y)
    # baseline alpha (no controls)
    a0 = y.mean() * ann
    rows = []
    for k in range(0, len(names) + 1):
        for subset in itertools.combinations(names, k):
            cols = [np.ones(n)] + [X[c] for c in subset]
            beta, t, r2 = _ols(y, np.column_stack(cols))
            row = dict(subset=subset, alpha=beta[0] * ann, t_alpha=t[0], r2=r2)
            if focal and focal in subset:
                j = 1 + list(subset).index(focal); row["focal_coef"] = beta[j]; row["focal_t"] = t[j]
            rows.append(row)
    alphas = np.array([r["alpha"] for r in rows]); talphas = np.array([r["t_alpha"] for r in rows])
    sign_stable = float(np.mean(np.sign(alphas) == np.sign(a0)))
    frac_sig = float(np.mean((np.abs(talphas) > 2) & (np.sign(alphas) == np.sign(a0))))
    # per-control marginal effect: adding c to a subset without it → ΔR², Δalpha (averaged over all such pairs)
    marg = {}
    base_by_subset = {r["subset"]: r for r in rows}
    for c in names:
        dR, dA = [], []
        for r in rows:
            if c in r["subset"]:
                prev = tuple(x for x in r["subset"] if x != c)
                p = base_by_subset.get(prev)
                if p: dR.append(r["r2"] - p["r2"]); dA.append(r["alpha"] - p["alpha"])
        marg[c] = dict(d_r2=float(np.mean(dR)) if dR else float("nan"),
                       d_alpha=float(np.mean(dA)) if dA else float("nan"),
                       corr_target=float(np.corrcoef(y, X[c])[0, 1]))
    out = dict(n=n, alpha_base=a0, alpha_min=float(alphas.min()), alpha_max=float(alphas.max()),
               alpha_sign_stable=sign_stable, alpha_frac_significant=frac_sig, n_specs=len(rows), marginals=marg)
    if focal:
        fc = np.array([r["focal_coef"] for r in rows if "focal_coef" in r])
        out["focal"] = dict(name=focal, coef_min=float(fc.min()), coef_max=float(fc.max()),
                            sign_stable=float(np.mean(np.sign(fc) == np.sign(np.median(fc)))))
    # verdict
    mirage_controls = [c for c, v in marg.items() if np.isfinite(v["d_r2"]) and v["d_r2"] > 0
                       and v["d_alpha"] * np.sign(a0) < 0 and abs(v["corr_target"]) > 0.3]
    if sign_stable > 0.95 and frac_sig > 0.6:
        verdict = "ROBUST — alpha sign stable and significant across every specification"
    elif abs(a0) < 0.02 and frac_sig < 0.10:
        verdict = "NULL — no alpha to speak of under ANY spec (robustly null, not a mirage)"
    elif sign_stable > 0.8:
        verdict = f"FRAGILE — a sizable alpha ({a0*100:+.0f}%/yr) keeps its sign but COLLAPSES/loses significance under plausible controls (significant in only {frac_sig*100:.0f}% of specs)"
    else:
        verdict = "MIRAGE — alpha sign FLIPS across plausible control specifications"
    out["mirage_controls"] = mirage_controls; out["verdict"] = verdict
    return out

def report(name, a):
    print(f"\n  {name}:  specs {a['n_specs']}   alpha(no controls) {a['alpha_base']*100:+.1f}%/yr")
    print(f"    alpha across all control subsets: [{a['alpha_min']*100:+.1f}%, {a['alpha_max']*100:+.1f}%]  "
          f"sign-stable {a['alpha_sign_stable']*100:.0f}%  significant+same-sign {a['alpha_frac_significant']*100:.0f}%")
    if "focal" in a:
        f = a["focal"]; print(f"    focal '{f['name']}' coef ∈ [{f['coef_min']:+.2f}, {f['coef_max']:+.2f}]  sign-stable {f['sign_stable']*100:.0f}%")
    print(f"    per-control  when ADDED →  ΔR²      Δalpha     |corr to target|")
    for c, v in sorted(a["marginals"].items(), key=lambda kv: -kv[1]["d_r2"] if np.isfinite(kv[1]["d_r2"]) else 0):
        tag = "  ← MIRAGE RISK (fit↑, alpha↓, collinear)" if c in a["mirage_controls"] else ""
        print(f"      {c:6} {v['d_r2']*100:>+7.2f}pp  {v['d_alpha']*100:>+7.1f}%/yr  {abs(v['corr_target']):>5.2f}{tag}")
    print(f"    → VERDICT: {a['verdict']}")

# ---- run on the most control-dependent verdicts ----------------------------------------------------
if __name__ == "__main__":
    import os, json, urllib.request
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
    def bars(s):
        u=(f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2016-01-01&end=2026-08-01&adjustment=all&feed=sip&limit=10000")
        d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=40)); return {b["t"][:10]:b["c"] for b in d.get("bars",{}).get(s,[])}
    def rets(sym):
        b=bars(sym); dts=sorted(b); px=np.array([b[d] for d in dts],float); return {d:r for d,r in zip(dts[1:],px[1:]/px[:-1]-1)}
    def aligned(syms):
        R={s:rets(s) for s in syms}; dts=sorted(set.intersection(*[set(R[s]) for s in syms]))
        return {s:np.array([R[s][d] for d in dts]) for s in syms}, dts
    print("="*104); print("MIRAGE — specification sign-stability audit of the most control-dependent verdicts"); print("="*104)

    # BURDENSOME: "cat/reinsurance premium is DISTINCT from selling puts (56% residual alpha)" — control-dependent.
    reins=["RNR","EG","ACGL","RLI","AXS","WRB","MKL","RGA"]; ctl=["SPY","PUTW","LQD","TLT","QUAL","VLUE","HYG","UUP"]
    A,_=aligned(reins+ctl); basket=np.nanmean(np.vstack([A[s] for s in reins]),axis=0)
    rep=specification_audit(basket, {c:A[c] for c in ctl}, focal="PUTW")
    report("BURDENSOME  (claim: cat premium distinct from puts)", rep)

    # BLEMISH: "Quality is Value in disguise" — does QUAL's alpha survive controlling for VLUE (and friends)?
    B,_=aligned(["QUAL","SPY","VLUE","MTUM","USMV","LQD","TLT","HYG"])
    repb=specification_audit(B["QUAL"]-B["SPY"], {c:B[c]-B["SPY"] for c in ["VLUE","MTUM","USMV","LQD","TLT","HYG"]}, focal="VLUE")
    report("BLEMISH  (claim: quality is value in disguise)", repb)

    print("\nREAD: a verdict that claims 'distinct alpha' is only as good as its control set. Where alpha's sign flips or")
    print("  dies under plausible controls, and a collinear control raises R² while killing alpha (← flagged), the 'edge'")
    print("  is partly a specification mirage. Run this on every regression-based verdict before promotion.")
