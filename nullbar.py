#!/usr/bin/python3
# =============================================================================
# nullbar.py — Blaque Baux universal null-testing gate.
#
# Every sleeve's verdict should survive three questions this module answers, from the strategy's own daily positions
# and the asset returns — no new data required:
#   (1) RANDOM-ENTRY BASELINE — does the SIGNAL beat taking the SAME trade (same hold, same cost) from EVERY bar?
#       That every-bar baseline is the limit of infinitely many random entries; a signal that doesn't beat it is timing
#       nothing, just collecting the asset's drift.
#   (2) CLUSTER-CORRECTED "BAND OF LUCK" — circularly shift the whole position pattern through time many times, keeping
#       the signals' spacing/clustering intact, and rebuild the P&L each shift. The actual book's percentile in that
#       band is a p-value that, unlike a naive t-stat, is NOT inflated by overlapping/clustered trades.
#   (3) EFFECTIVE N — how many INDEPENDENT trades you really have after overlap. Overlapping holds make 1,000 "trades"
#       behave like 30; Effective N (Newey-West variance-inflation on the daily P&L) is the honest count, and the
#       NW-corrected t-stat is the honest significance.
#
# Usage (any sleeve):
#   from nullbar import gate
#   g = gate(pos, asset_rets, hold=21, cost=2e-4)   # pos: daily position series (signal pattern); asset_rets: daily
#   print(g["verdict"])                             # PASS / FLAG / FAIL, with the three diagnostics
#
# Pure NumPy, deterministic (seeded). The __main__ demo fetches SPY + a few names via Alpaca and shows the gate
# discriminating a real-ish signal from noise and exposing clustering — but the core functions take plain arrays.
# =============================================================================
import numpy as np

# ---- core diagnostics (data-agnostic: arrays in, verdict out) ---------------------------------------
def _ann_sharpe(r, ppy=252):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return r.mean() / r.std() * np.sqrt(ppy) if (len(r) > 5 and r.std() > 0) else float("nan")

def newey_west(r, lags):
    """Newey-West (Bartlett) long-run variance of the mean of daily returns r, and the variance-inflation factor
    (VIF) vs the iid assumption. `lags` ~ the holding period (overlap horizon)."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]; n = len(r)
    if n < 10 or r.std() == 0: return dict(vif=float("nan"), tstat=float("nan"), eff_n_days=float("nan"))
    x = r - r.mean(); gamma0 = np.mean(x * x)
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)                      # Bartlett kernel
        s += 2.0 * w * np.mean(x[k:] * x[:-k])
    vif = s / gamma0 if gamma0 > 0 else float("nan")    # variance inflation from autocorrelation/overlap
    se = np.sqrt(s / n)                                  # NW standard error of the mean
    tstat = r.mean() / se if se > 0 else float("nan")
    eff_n_days = n / vif if vif and vif > 0 else float("nan")
    return dict(vif=vif, tstat=tstat, eff_n_days=eff_n_days)

def effective_n(pos, asset_rets, hold):
    """Effective number of INDEPENDENT trades. Naive = every in-market bar starts a notional hold-day trade; those
    trades OVERLAP (consecutive ones share hold−1 days), so the TRADE-return series is autocorrelated. Effective N =
    naive / VIF, with VIF the Bartlett variance-inflation of the trade-return series — overlap → VIF≫1 → eff ≪ naive.
    Also returns the NW t-stat of the daily P&L (overlap-corrected significance of the mean)."""
    pos = np.asarray(pos, float); r = np.asarray(asset_rets, float)
    m = min(len(pos), len(r)); pos, r = pos[-m:], r[-m:]
    starts = [t for t in range(m - hold) if abs(pos[t]) > 1e-9 and np.all(np.isfinite(r[t:t + hold]))]
    tr = np.array([np.sign(pos[t]) * (np.prod(1 + r[t:t + hold]) - 1) for t in starts])   # overlapping trade returns
    naive = len(tr); nw = newey_west(pos * r, lags=hold)
    if naive < 5 or tr.std() == 0:
        return dict(naive_trades=naive, eff_trades=float(naive), vif=1.0, nw_tstat=nw["tstat"])
    x = tr - tr.mean(); g0 = np.mean(x * x); L = min(hold, naive - 2); s = g0
    for k in range(1, L + 1):
        s += 2.0 * (1.0 - k / (L + 1.0)) * np.mean(x[k:] * x[:-k])
    vif = max(s / g0, 1e-6)
    return dict(naive_trades=naive, eff_trades=naive / vif, vif=vif, nw_tstat=nw["tstat"])

def random_entry_baseline(asset_rets, hold, cost=0.0):
    """The every-bar baseline: the return of the SAME trade (hold `hold` days, pay `cost` round-trip) entered on
    EVERY possible bar — the limit of random-timed entries. Returns the full distribution of per-trade returns."""
    r = np.asarray(asset_rets, float); n = len(r)
    out = [np.prod(1 + r[t:t + hold]) - 1 - cost for t in range(0, n - hold) if np.all(np.isfinite(r[t:t + hold]))]
    return np.array(out)

def beats_random_entry(trade_rets, baseline_dist, n_boot=5000, seed=0):
    """Percentile of the strategy's mean trade return within the sampling distribution of the mean of the same number
    of RANDOM-entry trades. >0.95 = the signal's timing genuinely beats random entry; ~0.5 = it times nothing."""
    tr = np.asarray(trade_rets, float); tr = tr[np.isfinite(tr)]
    bl = np.asarray(baseline_dist, float); bl = bl[np.isfinite(bl)]
    if len(tr) < 3 or len(bl) < len(tr): return float("nan")
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(bl, size=len(tr), replace=True).mean() for _ in range(n_boot)])
    return float(np.mean(tr.mean() > boot))

def cluster_corrected_pvalue(pos, asset_rets, n_shifts=5000, seed=0):
    """Band of luck: circularly shift the position pattern (preserving its spacing/clustering) `n_shifts` times and
    recompute aggregate P&L; p = fraction of shifts whose P&L >= the actual. Not inflated by overlapping trades."""
    pos = np.asarray(pos, float); r = np.asarray(asset_rets, float)
    m = min(len(pos), len(r)); pos, r = pos[-m:], r[-m:]
    ok = np.isfinite(pos) & np.isfinite(r); pos, r = np.where(ok, pos, 0.0), np.where(ok, r, 0.0)
    actual = float(np.sum(pos * r))
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, m - 1, size=n_shifts)
    band = np.array([np.sum(np.roll(pos, int(s)) * r) for s in shifts])
    p = float(np.mean(band >= actual))                  # one-sided (for a long-biased/positive-edge book)
    return dict(actual=actual, p_value=p, band_mean=float(band.mean()), band_sd=float(band.std()))

# ---- the universal gate -----------------------------------------------------------------------------
def gate(pos, asset_rets, hold=21, cost=2e-4, n_shifts=5000, min_eff_trades=20, seed=0):
    """One call: random-entry baseline percentile, cluster-corrected p-value, Effective N + NW t-stat, and a PASS/
    FLAG/FAIL verdict. `pos` = daily position series (the signal pattern); `asset_rets` = daily returns it trades."""
    pos = np.asarray(pos, float); r = np.asarray(asset_rets, float)
    m = min(len(pos), len(r)); pos, r = pos[-m:], r[-m:]
    strat = pos * r
    en = effective_n(pos, r, hold)
    cc = cluster_corrected_pvalue(pos, r, n_shifts=n_shifts, seed=seed)
    # per-trade returns for the random-entry comparison (approx: in-market daily P&L compounded per hold-block)
    base = random_entry_baseline(r, hold, cost=cost)
    inm = strat[np.abs(pos) > 1e-9]
    trade_block = np.array([np.prod(1 + inm[i:i + hold]) - 1 for i in range(0, max(0, len(inm) - hold), hold)]) if len(inm) > hold else np.array([])
    be = beats_random_entry(trade_block, base, seed=seed) if len(trade_block) >= 3 else float("nan")
    sh = _ann_sharpe(strat)
    passes = (cc["p_value"] < 0.05) and (np.isfinite(en["eff_trades"]) and en["eff_trades"] >= min_eff_trades) and \
             (np.isfinite(en["nw_tstat"]) and abs(en["nw_tstat"]) > 2.0) and (not np.isfinite(be) or be > 0.90)
    flag = (cc["p_value"] < 0.10) and (np.isfinite(en["eff_trades"]) and en["eff_trades"] >= 10)
    verdict = "PASS" if passes else ("FLAG" if flag else "FAIL")
    return dict(verdict=verdict, sharpe=sh, naive_trades=en["naive_trades"], eff_trades=en["eff_trades"],
                vif=en["vif"], nw_tstat=en["nw_tstat"], cluster_p=cc["p_value"], beats_random_entry=be)

def report(name, g):
    print(f"  {name:28} {g['verdict']:5}  Sharpe {g['sharpe']:+.2f}  trades {g['naive_trades']:>4}→eff {g['eff_trades'] if not np.isfinite(g['eff_trades']) else round(g['eff_trades'],0)!s:>5}  "
          f"VIF {g['vif']:.1f}  NW-t {g['nw_tstat']:+.2f}  cluster-p {g['cluster_p']:.3f}  beats-random {g['beats_random_entry'] if not np.isfinite(g['beats_random_entry']) else round(g['beats_random_entry'],2)!s}")

# ---- demo: the gate discriminating real vs noise, and exposing overlap-inflated significance --------
if __name__ == "__main__":
    import os, json, urllib.request
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
    def bars(s):
        u = (f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2016-01-01&end=2026-08-01&adjustment=all&feed=sip&limit=10000")
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))
        return {b["t"][:10]: b["c"] for b in d.get("bars", {}).get(s, [])}
    spx = bars("SPY"); dates = sorted(spx); px = np.array([spx[d] for d in dates], float); r = px[1:]/px[:-1]-1; n = len(r)
    print("="*110); print(f"NULLBAR — universal null-testing gate (demo on SPY, {dates[0]}→{dates[-1]})"); print("="*110)
    rng = np.random.default_rng(1)
    # (1) PLANTED-EDGE oracle — DEMO ONLY, deliberately uses same-day return (look-ahead) to build a signal with a
    #     GENUINE edge, to prove the gate RECOGNIZES real timing (not to trade). ~70% accurate, non-overlapping daily.
    planted = np.array([1.0 if (0.6*np.sign(r[t]) + 0.4*(rng.random()-0.5) > 0) else -1.0 for t in range(n)])
    # (2) pure NOISE positions (no predictive content)
    noise = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    # (3) heavily OVERLAPPING always-long: naive trade count huge, independent N tiny — pure SPY beta, not timing
    overlap = np.ones(n)
    print("\n  signal                        verdict  Sharpe   naive→effective        significance")
    report("planted-edge oracle (demo)",  gate(planted, r, hold=1))
    report("pure-noise timing",           gate(noise, r, hold=1))
    report("always-long (overlap, h=63)", gate(overlap, r, hold=63))
    print("\nREAD: the planted oracle PASSES (beats-random ~1, cluster-p ~0, real Effective N) — the gate recognizes a genuine")
    print("  edge. Pure noise FAILS (cluster-p ~0.5, beats-random ~0.5 — times nothing). Always-long exposes the OVERLAP TRAP:")
    print("  a huge naive trade count collapses to a tiny Effective N (VIF ≫ 1) — its 'significance' is one long beta bet, not")
    print("  many independent trades. Gate every sleeve; a result that fails here is luck, clustering, or beta — not alpha.")
