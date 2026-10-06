# Blaque Baux Nullbar

**The family's universal null-testing gate. Every sleeve's verdict should survive it.**

`nullbar` answers three questions about any strategy — from its own daily positions and the asset returns, no new data —
and returns a **PASS / FLAG / FAIL** with the diagnostics. It's the formalization of the standing house rule: *an
extraordinary result is a red flag until it clears the null.*

> **Not investment advice.** Educational/research software. See [DISCLAIMER](DISCLAIMER.md) and [LICENSE](LICENSE).

```python
from nullbar import gate
g = gate(pos, asset_rets, hold=21, cost=2e-4)   # pos: daily position/signal series; asset_rets: daily returns traded
print(g["verdict"])                             # PASS / FLAG / FAIL + sharpe, eff trades, VIF, NW t-stat, cluster-p, beats-random
```
```bash
python3 nullbar.py   # demo (needs Alpaca data keys) — shows the gate on a planted edge, noise, and an overlap trap
```

## The three tests

1. **Random-entry baseline** — does the *signal* beat taking the **same trade** (same hold, same cost) from **every
   bar**? That every-bar baseline is the limit of infinitely many random entries; a signal that doesn't beat it is
   timing nothing, just collecting the asset's drift. (`beats_random_entry` → want ≳ 0.90.)
2. **Cluster-corrected "band of luck"** — circularly shift the whole position pattern through time thousands of times,
   keeping the signals' spacing/clustering intact, and rebuild the P&L each shift. The actual book's percentile in that
   band is a p-value that — unlike a naive t-stat — is **not inflated by overlapping/clustered trades**. (`cluster_p` →
   want < 0.05.)
3. **Effective N** — how many *independent* trades you really have. Every in-market bar starts a notional hold-day
   trade; those trades **overlap** (consecutive ones share hold−1 days), so the trade-return series is autocorrelated.
   Effective N = naive / Bartlett-VIF — overlap → VIF ≫ 1 → **eff ≪ naive**. Plus a Newey-West t-stat on the daily P&L.
   (`eff_trades` → want ≳ 20; `nw_tstat` → want |t| > 2.)

## Demo output (SPY, 2016–2026)

| signal | verdict | Sharpe | naive→effective | significance |
|---|---|---|---|---|
| planted-edge oracle *(look-ahead, demo only)* | **PASS** | +13.78 | 2657→**1976** | cluster-p 0.000, beats-random 1.0 |
| pure-noise timing | **FAIL** | −0.32 | 2657→2673 | cluster-p 0.84, beats-random 0.0 |
| always-long (overlap, h=63) | **FAIL** | +0.88 | 2595→**75** (VIF 34.8) | cluster-p 1.00, beats-random 0.48 |

The gate recognizes a genuine edge (planted), rejects noise, and exposes the **overlap trap**: an always-long book's
2,595 "trades" collapse to **75 effective** — its significance is *one long beta bet*, not many independent trades, and
it doesn't beat random entry (it's beta, not timing alpha).

## Use it as a universal gate

Run `gate(...)` on every sleeve's book. A result that **fails here is luck, clustering, or beta — not alpha.** It is the
counterpart to the family's recurring findings (e.g. `brunt`'s Sharpe-13 artifact, `beaming`'s fill artifact,
`bollinger`'s selection) — a single, reusable check that turns "looks good" into "survives the null." `nullbar_2` =
block-bootstrap confidence intervals + a multiple-testing (deflated-Sharpe) correction across the whole corpus.

## Status

**Toolkit.** Pure-NumPy, deterministic (seeded), importable; core functions take plain arrays (no data dependency), the
demo fetches SPY. Not a sleeve — a gate for all of them.

---

# Expanded suite (Oct 2026): selection-bias, execution & structural layers

`nullbar.gate` is the **entry/overlap** layer. Two companion modules complete the loop.

## `validation.py` — selection-bias & execution

| function | question | fail |
|---|---|---|
| `deflated_sharpe` | is the Sharpe just the MAX of many trials? (Bailey–López de Prado) | DSR < 0.5 |
| `pbo_cscv` | given everything tried, is the SELECTED strategy overfit? (CSCV) | PBO > 0.5 |
| `edge_decay` | how much edge survives IS→OOS? (consistency-weighted grade) | grade D/F |
| `basso_random_entry` | does trade MANAGEMENT alone profit (no entry rule)? | profit factor ≈ 1 |
| `monte_carlo_trades` | how wide is the luck-of-sequencing distribution? | CI overlaps zero |
| `falsify(...)` | stacks all of the above **+ nullbar.gate** into one PASS/FAIL | any layer fails |

Demo (SPY): PBO separates stable real-skill selection (~0.14) from noise (~0.37); DSR deflates a lucky best-of-60 Sharpe
as the honest trial count rises; Basso shows random entry + a trailing stop ≈ breakeven-to-positive in a trend — *the
exit is the edge, not the entry.*

## `stress.py` — structural / failure-mode layer

Significance is necessary, not sufficient. Four checks Sharpe/PBO/DSR can't make:

- **`point_in_time`** — evaluate on the date data *arrived*, not the date it *describes*; vendor panels backfill
  entities/corrected maps across history, so a backtest "knows" what the analyst that day couldn't. Latency is a
  distribution — test the tail.
- **`failure_mode`** (registry) — different strategy *species* fail differently (stat-arb → short squeeze; trend → chop +
  synchronized vol-target deleveraging; market-neutral → basis + crowding). Classify first, stress the type's killer; one
  ruler across all is false comfort.
- **`second_order_crowding`** — when *shorting itself* is the crowded trade (Oct-2025 quant quake): on days the
  most-shorted basket rallies, does the short leg take synchronized, non-linear losses? (The mechanism `bevy` couldn't see
  on large-cap post-2009 data — the short side as the squeeze source.)
- **`corroboration_score`** — structural vs statistical alpha: many *independent configs agreeing* (structural, robust to
  spec) beats one razor-tuned fit (statistical, overfit) — in prior work corroborating-config count was +0.40 with
  realized return while per-config fit was −0.33.

```bash
python3 validation.py   # selection-bias & execution demo (needs Alpaca keys)
python3 stress.py       # structural layer demo (synthetic, no data)
```

**The full loop:** signal (nullbar.gate) → selection (PBO, DSR) → decay (edge_decay) → execution (Basso, Monte Carlo) →
structure (stress.py). A sleeve that clears every layer has earned its verdict; one that fails any is luck, clustering,
beta, selection, or a backfill illusion — not alpha. `nullbar_2` = wire `falsify()` as a CI gate across the whole corpus.

## `mirage.py` — specification / factor-mirage audit

Asks the question significance tests can't: **are the variables structurally correct, or does a bad control improve fit
while flipping/collapsing the coefficient?** (Leamer extreme-bounds + bad-control diagnostics.) `specification_audit`
re-estimates the regression across **all control subsets** and reports alpha sign-stability + extreme bounds, each
control's marginal ΔR² vs Δalpha (a mirage control raises R² while killing alpha, collinear with the target), and a
ROBUST / FRAGILE / NULL / MIRAGE verdict. First audits (see `mirage.py`): **burdensome** "distinct from puts" came back
**FRAGILE** (raw +16%/yr alpha collapses to +2.6%, significant in only 2% of specs — largely equity/quality/credit beta);
**blemish** came back **robustly NULL** (confirmed). The qualitative half — *why* each variable is in/out — is a required
per-sleeve rationale field it can't automate.

See **[SPECS.md](SPECS.md)** for the full governed pipeline and the specs for the modules still to build (Friction,
corpus-wide FDR + pre-registration ledger, Breakthrough shrinkage, sleeve-correlation/factor map, Stress-wrapper).

## Complete suite (Oct 2026)

All layers are now built: `nullbar.py` (entry/overlap) · `validation.py` (PBO/DSR/edge-decay/Basso/Monte-Carlo) ·
`mirage.py` (specification) · `stress.py` (structural diagnostics) · `friction.py` (Almgren-Chriss impact + capacity) ·
`ledger.py` (pre-registration + corpus-wide FDR) · `breakthrough.py` (robustness-shrinkage allocator) ·
`sleevemap.py` (correlation + factor map) · `stress_wrapper.py` (regime filter/inversion). Run each file for its demo; see
[SPECS.md](SPECS.md) for the governed pipeline and two corpus-level findings (most mid-tier keepers don't survive
family-wise FDR; the keeper book is ~1.8 effective bets). Port to the Julia `base` engine for the real data feeds.
