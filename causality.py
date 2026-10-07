#!/usr/bin/python3
# =============================================================================
# causality.py — causal prefix-invariance gate (the lookahead detector).
#
# A hard gate, not a statistic. A position taken at time t MUST NOT change when
# future bars are later revealed. DSR/PBO/mirage cannot catch look-ahead if the
# backtest itself peeks at the future (a leaky oracle posts Sharpe 35 and sails
# through every statistical screen). This runs the strategy on truncated prefixes
# of the price series and flags any past position that moves once more data appears
# — catching full-period normalization, centered windows, and subtle peeks.
#
# strategy_fn(prices[:k]) must return an array of positions, one per bar it saw.
# Pure NumPy.
# =============================================================================
import numpy as np

class LookaheadError(Exception):
    """Raised when a past position changes as future bars are revealed."""

def prefix_invariance(strategy_fn, prices, probes=24, tol=1e-9, raise_on_fail=False):
    """Reveal the series in growing prefixes; a causal strategy's position at bar i
    is identical whether it saw k bars or the whole series. Compares each prefix's
    positions against the full-series positions on their overlap. Returns a verdict
    dict; with raise_on_fail=True, raises LookaheadError on the first violation."""
    prices = np.asarray(prices, float); T = len(prices)
    if T < 10: return dict(is_causal=True, n_violations=0, violations=[], verdict="CAUSAL", note="series too short")
    full = np.asarray(strategy_fn(prices), float)
    cuts = sorted(set(np.linspace(max(5, T // 20), T, probes, dtype=int).tolist()))
    violations = []
    for k in cuts:
        pk = np.asarray(strategy_fn(prices[:k]), float)
        m = min(len(pk), len(full), k)
        d = np.abs(pk[:m] - full[:m])
        bad = np.where(d > tol)[0]
        if len(bad):
            v = dict(prefix=int(k), first_changed_bar=int(bad[0]), n_changed=int(len(bad)), max_delta=float(d[bad].max()))
            violations.append(v)
            if raise_on_fail:
                raise LookaheadError(f"position at bar {v['first_changed_bar']} changed when prefix grew to {k} "
                                     f"(Δ={v['max_delta']:.3g}) — the strategy peeks at the future")
    leak = len(violations) > 0
    return dict(is_causal=not leak, n_violations=len(violations), violations=violations[:5],
                verdict="LOOKAHEAD" if leak else "CAUSAL")

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, 500))
    # CAUSAL: 20-day momentum sign using only trailing data
    def causal(p):
        p = np.asarray(p, float); pos = np.zeros(len(p))
        for t in range(21, len(p)): pos[t] = np.sign(p[t-1] - p[t-21])
        return pos
    # LEAKY: z-score against the FULL-series mean/std (uses the future)
    def leaky(p):
        p = np.asarray(p, float); z = (p - p.mean()) / (p.std() + 1e-9); return np.sign(-z)
    print("=" * 78); print("CAUSALITY — prefix-invariance (lookahead) gate"); print("=" * 78)
    for name, fn in (("causal 20d-momentum", causal), ("leaky full-period z-score", leaky)):
        r = prefix_invariance(fn, px)
        print(f"\n  {name:26} -> {r['verdict']}  (violations: {r['n_violations']})")
        if r["violations"]: print(f"     first: bar {r['violations'][0]['first_changed_bar']}, Δ={r['violations'][0]['max_delta']:.3g}")
    print("\nREAD: the leaky rule returns different PAST positions once it sees more of the series —")
    print("  correctly-timestamped data is not enough; this gate proves position integrity.")
