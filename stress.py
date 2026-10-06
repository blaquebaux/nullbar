#!/usr/bin/python3
# =============================================================================
# stress.py — structural / failure-mode layer (companion to nullbar.gate + validation.py).
#
# Statistical significance isn't enough. Four checks that catch what Sharpe/PBO/DSR can't:
#   1. point_in_time         — evaluate on the date data ARRIVED, not the date it describes (kills backfill look-ahead)
#   2. failure_mode registry — different strategy SPECIES fail differently; classify first, stress the TYPE's killer
#   3. second_order_crowding — when SHORTING ITSELF is the crowded trade (the Oct-2025 quant-quake mechanism)
#   4. corroboration_score   — structural vs statistical alpha: many agreeing configs (structural) vs one razor-tuned fit
#
# Pure NumPy; the __main__ demo is synthetic (no data dependency) so it runs anywhere.
# =============================================================================
import numpy as np

# ---- 1. data-arrival / point-in-time ----------------------------------------------------------------
def point_in_time(values, arrived_idx, n=None):
    """Make each observation usable only from its ARRIVAL index, not the date it describes. Vendor 'historical'
    panels backfill new entities/corrected maps across all history, so a backtest 'knows' things the analyst that day
    couldn't. Pass each value's arrival index; returns a series that is NaN until arrival. Production latency is a
    DISTRIBUTION — test on the tail arrival, not the median."""
    arrived_idx = list(arrived_idx); n = n or (max(arrived_idx) + 1 if arrived_idx else 0)
    out = np.full(n, np.nan)
    for v, a in zip(values, arrived_idx):
        if 0 <= a < n and not np.isfinite(out[a]): out[a] = v
    return out

# ---- 2. strategy-type failure modes (classify first, then stress the species' specific killer) -------
FAILURE_MODES = {
 "equity_stat_arb": "Crowding + short squeeze — the SHORT side is the risk. → second_order_crowding() on the short leg.",
 "trend_following": "Chop + synchronized vol-target deleveraging (all CTAs sell liquid futures at once). → whipsaw-regime P&L + a vol-target-unwind scenario.",
 "market_neutral":  "Basis volatility + factor crowding together (2025). → factor-crowding + funding/basis shock.",
 "carry":           "Risk-off crash (funders rally). → conditional return in the worst market quintile; check negative skew.",
 "short_vol":       "Vol gap / tail. → worst single-day move + a Volmageddon-style scenario (see brace/bleacher/budget).",
 "mean_reversion":  "Trending breakouts / regime change. → P&L when trend strength is high (the inverse of its comfort zone).",
}
def failure_mode(strategy_type):
    """Return the species-specific failure mode to stress-test — because one Sharpe/DD/corr ruler across all types is
    false comfort, not analysis."""
    return FAILURE_MODES.get(strategy_type, "UNKNOWN type — classify before stress-testing; a generic ruler is false comfort.")

# ---- 3. second-order crowding (when shorting itself is the crowded trade) ----------------------------
def second_order_crowding(short_leg_rets, crowded_basket_rets, thresh=0.03):
    """On days the most-shorted / crowded basket RALLIES past `thresh`, does the short leg take synchronized,
    NON-LINEAR losses? (Oct-2025: long low-vol / short high-vol-low-quality unwound when the most-shorted basket
    ripped.) Returns the short leg's conditional loss on rally days vs normal, the worst such day, and a flag."""
    s = np.asarray(short_leg_rets, float); c = np.asarray(crowded_basket_rets, float)
    m = min(len(s), len(c)); s, c = s[-m:], c[-m:]
    rally = c > thresh
    cond = float(np.nanmean(s[rally])) if rally.sum() > 3 else float("nan")
    base = float(np.nanmean(s)); worst = float(np.nanmin(s[rally])) if rally.sum() > 3 else float("nan")
    # convexity: is the loss steeper than a linear response to the basket's rally size?
    nonlinear = False
    if rally.sum() > 5:
        big = c[rally] > np.nanquantile(c[rally], 0.5)
        nonlinear = bool(np.nanmean(s[rally][big]) < np.nanmean(s[rally][~big]) * 1.5)
    return dict(rally_days=int(rally.sum()), short_leg_on_rally=cond, short_leg_normal=base,
                worst_day=worst, nonlinear=nonlinear, flag=bool(np.isfinite(cond) and cond < base - 0.005))

# ---- 4. structural vs statistical alpha (corroboration, not razor-fit) -------------------------------
def corroboration_score(config_returns):
    """Structural-alpha proxy: of many INDEPENDENT configurations of the idea, what fraction agree on a positive
    edge? In prior work the COUNT of corroborating configs was +0.40 correlated with realized return while per-config
    fit quality (CPE) was -0.33 — i.e. agreement across configs is structural, a single razor-tuned config is
    statistical/overfit. High score = robust to specification; low = one lucky setting."""
    M = np.asarray(config_returns, float)
    if M.ndim != 2 or M.shape[1] < 2: return float("nan")
    srs = M.mean(0) / (M.std(0) + 1e-12)
    return float(np.mean(srs > 0))

# ---- demo (synthetic, no data) ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0); T = 1500
    print("="*96); print("STRESS — structural / failure-mode layer (synthetic demo)"); print("="*96)
    # 3. second-order crowding: a crowded 'most-shorted' basket, and a short leg that unwinds non-linearly when it rips
    crowded = rng.normal(0.0005, 0.03, T)
    short_leg = -0.3*crowded - 1.8*np.maximum(crowded-0.03, 0) + rng.normal(0, 0.006, T)   # convex loss on big rallies
    sc = second_order_crowding(short_leg, crowded, thresh=0.03)
    print(f"\n2nd-order crowding: on {sc['rally_days']} basket-rally days the short leg returns {sc['short_leg_on_rally']*100:+.2f}%/day vs {sc['short_leg_normal']*100:+.2f}% normal,")
    print(f"   worst {sc['worst_day']*100:+.1f}%, non-linear unwind={sc['nonlinear']}, FLAG={sc['flag']}  → when shorting itself is crowded, the short leg is the risk (Oct-2025).")
    # 4. corroboration: 20 configs that mostly agree (structural) vs 20 noisy configs
    struct = np.column_stack([0.0003 + rng.normal(0,0.01,T) for _ in range(20)])
    noisy  = np.column_stack([rng.normal(0,0.01,T) for _ in range(20)])
    print(f"\ncorroboration: structural idea (configs agree) → {corroboration_score(struct):.2f}   razor/noise → {corroboration_score(noisy):.2f}  (high = robust to spec; low = one lucky setting)")
    # 2. failure-mode registry
    print(f"\nfailure_mode('equity_stat_arb'): {failure_mode('equity_stat_arb')}")
    print(f"failure_mode('trend_following'): {failure_mode('trend_following')}")
    # 1. point-in-time
    pit = point_in_time([10,11,12], arrived_idx=[2,4,6], n=7)
    print(f"\npoint_in_time: values[10,11,12] arriving at idx[2,4,6] → {np.where(np.isfinite(pit),pit,-1).astype(int).tolist()} (−1 = not yet knowable; evaluate on ARRIVAL, not description date).")
    print("\nREAD: classify the species, test its specific killer, respect data-arrival, and demand corroboration across")
    print("  configs — significance (nullbar+validation) is necessary but not sufficient; these catch the structural traps.")
