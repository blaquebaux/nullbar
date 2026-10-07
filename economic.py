#!/usr/bin/python3
# =============================================================================
# economic.py — economic-significance gate (McCloskey-Ziliak: statistical != economic).
#
# Every other gate tests whether alpha is statistically real. This tests whether it is
# big enough to MATTER. A coefficient can be reliably non-zero and still too small to
# clear costs. A sleeve must pass a PRE-REGISTERED economic hurdle (minimum "oomph"),
# independent of its t-stat — a significant but trivial edge is not a keeper.
# Pure stdlib.
# =============================================================================
import math

def economic_significance(net_alpha_ann, hurdle=0.02, t_stat=None, min_t=2.0):
    """net_alpha_ann: annualized alpha ALREADY net of realistic costs (post-friction).
    hurdle: pre-registered minimum annual alpha to be economically meaningful (default 2%/yr).
    A keeper must clear BOTH the economic hurdle and (if a t-stat is given) significance."""
    econ = net_alpha_ann >= hurdle
    stat = True if t_stat is None else abs(t_stat) >= min_t
    if econ and stat:        v = "KEEPER"
    elif stat and not econ:  v = "TRIVIAL — significant but below the economic hurdle"
    elif econ and not stat:  v = "NOISY — large but not statistically distinguishable"
    else:                    v = "REJECT — neither economically nor statistically material"
    return dict(net_alpha_ann=net_alpha_ann, hurdle=hurdle, clears_hurdle=econ,
                t_stat=t_stat, significant=stat, is_keeper=(econ and stat), verdict=v)

def oomph_curve(net_alpha_ann, t_stat, hurdles=(0.0, 0.01, 0.02, 0.03, 0.05)):
    """An indifference map between effect size and significance: which hurdles the sleeve
    clears. Makes the economic-vs-statistical trade-off explicit rather than implicit."""
    return [dict(hurdle=h, keeper=economic_significance(net_alpha_ann, h, t_stat)["is_keeper"]) for h in hurdles]

if __name__ == "__main__":
    print("=" * 78); print("ECONOMIC SIGNIFICANCE — the minimum-oomph gate"); print("=" * 78)
    cases = [("bankroll-like", 0.08, 3.1), ("borough (tax carry)", 0.021, 2.3),
             ("a significant triviality", 0.004, 2.8), ("big but noisy", 0.09, 1.2)]
    print(f"\n  {'sleeve':24}{'net α/yr':>10}{'t':>7}  verdict")
    for n, a, t in cases:
        r = economic_significance(a, hurdle=0.02, t_stat=t)
        print(f"  {n:24}{a*100:>+9.1f}%{t:>7.1f}  {r['verdict']}")
    print("\nREAD: a 0.4%/yr edge with t=2.8 is STATISTICALLY real and ECONOMICALLY trivial — not a keeper.")
    print("  The hurdle is pre-registered so the bar can't be moved to fit the result.")
