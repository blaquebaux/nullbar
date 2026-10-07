#!/usr/bin/python3
# =============================================================================
# attribution.py — decision-chain / alpha-attribution audit (the new layer).
#
# The eight gates test a strategy's INTERNAL validity, all sharing one premise: the
# trading logic is the alpha source. This layer asks the prior question — after stripping
# away DATA, ASSET-SELECTION, and EXECUTION choices, does the logic still add incremental
# return? The cheapest, most powerful cut is asset-selection: fix the logic, vary the
# universe. If a handful of names carry the edge, the "strategy alpha" is asset-selection
# alpha and the logic is merely a carrier.
#
#   asset_attribution    — how concentrated is the edge across names (a few names => carrier)
#   universe_attribution — does the edge survive removing the richest names / shrinking the pool
# Pure NumPy. (Data-vendor and execution-algo sublayers need multiple feeds — deferred.)
# =============================================================================
import numpy as np

def asset_attribution(asset_contribs):
    """asset_contribs: dict name -> total P&L contribution. Returns the edge's concentration:
    top-1 / top-3 share of positive contribution and the effective number of names (1/HHI).
    Few names dominating => the logic is a vessel for asset selection, not an edge."""
    names = list(asset_contribs)
    v = np.array([asset_contribs[n] for n in names], float)
    pos = np.clip(v, 0, None); tot = pos.sum()
    if tot <= 0: return dict(verdict="NO POSITIVE CONTRIBUTION", eff_names=float("nan"), n=len(names))
    w = pos / tot; order = np.argsort(w)[::-1]
    top1, top3, eff = float(w[order[0]]), float(w[order[:3]].sum()), float(1.0 / np.sum(w**2))
    verdict = "ASSET-SELECTION ALPHA (logic is a carrier)" if (top3 > 0.75 or eff < 3) else "DISTRIBUTED (logic plausibly adds)"
    return dict(top1_share=top1, top3_share=top3, eff_names=eff, n=len(names),
                top_names=[names[i] for i in order[:3]], verdict=verdict)

def universe_attribution(sharpe_by_universe):
    """sharpe_by_universe: dict label -> Sharpe of the SAME logic on different pools
    (e.g. 'full', 'large_cap', 'sp500', 'ex_top10'). If the edge collapses once the richest
    names are removed, it was asset selection, not logic."""
    v = dict(sharpe_by_universe)
    base = v.get("full", max(v.values()))
    ex = v.get("ex_top10", v.get("sp500"))
    drop = None if ex is None else base - ex
    collapse = drop is not None and base > 0 and drop > 0.5 * base
    return dict(sharpe_by_universe=v, base=base, ex_richest=ex, drop=drop,
                verdict="ASSET-SELECTION (edge collapses off the top names)" if collapse else "ROBUST ACROSS UNIVERSES")

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    print("=" * 80); print("ATTRIBUTION — decision-chain audit (asset-selection cut)"); print("=" * 80)
    # a 'strategy' whose entire edge is two names
    carrier = {f"stk{i}": float(rng.normal(0, 1)) for i in range(20)}; carrier["NVDA"] = 40.0; carrier["AAPL"] = 18.0
    distributed = {f"stk{i}": float(abs(rng.normal(2, 0.4))) for i in range(20)}
    for name, d in (("secretly two names", carrier), ("genuinely spread", distributed)):
        r = asset_attribution(d)
        print(f"\n  {name:20} top1 {r['top1_share']*100:4.0f}%  top3 {r['top3_share']*100:4.0f}%  effN {r['eff_names']:4.1f}  -> {r['verdict']}")
    u = universe_attribution({"full": 0.95, "ex_top10": 0.18})
    print(f"\n  universe test: full {u['base']:+.2f} -> ex-top10 {u['ex_richest']:+.2f}  => {u['verdict']}")
    print("\nREAD: fix the logic, vary the universe. An edge that lives in a few names is asset-")
    print("  selection alpha wearing a strategy's clothes — the gate the other eight assume away.")
