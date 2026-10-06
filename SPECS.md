# Blaque Baux — governed research pipeline: module specs

The toolkit enforces one pipeline. A sleeve is promoted to the `breakthrough` allocator only if it survives **every**
layer. Built layers are in this repo (Python prototype); the rest are specs to port into the `base` Julia engine (the
system of record), where the data (ADV, borrow, PIT universe) actually lives.

```
generate_alpha → STRESS(regime filter) → FRICTION(honest cost) → BLIND(time-shift null) →
                 MIRAGE(spec sign-stability) → FDR/LEDGER(corpus-wide) → BREAKTHROUGH(robustness-shrunk allocation)
```

| layer | fragility it catches | status |
|---|---|---|
| `nullbar.gate` / Blind | statistical (luck, clustering, overlap) | ✅ built |
| `validation` (PBO, DSR, edge-decay, Basso, MC) | selection / execution illusion | ✅ built |
| `stress` (diagnostics) + **Stress-wrapper** | regime / structural | ✅ diag built; wrapper = spec |
| `mirage` | specification / factor-mirage | ✅ built |
| **Friction** | execution (liquidity evaporation) | ⬜ spec |
| **FDR + pre-registration ledger** | corpus-wide multiple testing | ⬜ spec |
| **Breakthrough shrinkage** | allocating to lucky vs proven alpha | ⬜ spec |
| **Sleeve-correlation / factor map** | portfolio secretly one bet | ⬜ spec |

---

## 1. Friction — honest, non-linear execution cost  *(top build priority — re-grades existing keepers)*

Square-root market impact (Almgren-Chriss) scaled by ADV participation + volatility, plus state-dependent short borrow.
Flat-bps is a lie; this is what kills thin keepers.

```julia
abstract type FrictionModel end
struct InstitutionalFriction <: FrictionModel
    impact_coeff::Float64    # √-impact coefficient (~0.1–0.5)
    borrow_base_bps::Float64 # base annual borrow
    htb_multiplier::Float64  # Hard-to-Borrow multiplier when utilization > ~80%
end
# slippage(trade) = half_spread + impact_coeff · vol · sqrt(trade_usd / ADV_usd)
# borrow_drag(short, day) = base_rate · (htb_multiplier if short_utilization > 0.80 else 1)
# apply_friction!(TradeLog, data, model)  → gross trade returns become NET, direction-aware
```
- **Data deps (not in Alpaca → Julia `base` feed):** ADV, bid-ask spread, realized vol, short utilization/borrow.
- **Extension — CAPACITY curve:** sweep trade_size/AUM → Sharpe-vs-AUM and the break-even AUM per sleeve. A 0.6-Sharpe
  microstructure/small-cap sleeve may hold $5M, not $500M. This is the deliverable, not just per-trade bps.
- **Feeds:** net TradeLog → Blind, mirage, and Breakthrough (which must shrink on NET, not gross, returns).

## 2. Stress-wrapper — regime filter / inversion (production transform)  *(complements stress.py diagnostics)*

A functional wrapper that intercepts the raw signal before the optimizer — to *implement* the negative tests, not just
measure them.

```julia
@enum InversionMode PassThrough FilterOnly InvertOnly Asymmetric
struct RegimeFilter; name::String; indicator_func::Function; mode::InversionMode; end
apply_stress(raw_signal, dates, data, filter) → stressed_signal
# FilterOnly: zero signal unless regime active; InvertOnly: flip when active; Asymmetric: longs-only in A / shorts-only in B
```
- Pre-built filters: crowded-long inversion (SI top-decile ∧ valuation top-decile), high-correlation-panic filter, rising-vol gate.
- **Fix from the draft:** re-check the `Asymmetric` branch (its long/short zeroing reads inverted vs the comment).
- `stress.py` *measures* the fragility (crowding, failure-mode, PIT, corroboration); this *acts* on it. Both ship.

## 3. FDR + pre-registration ledger — corpus-wide multiple testing  *(my addition; most urgent given ~55 sleeves)*

DSR deflates for trials *within* a sleeve; nothing governs the **~55 across** sleeves — some "keepers" are false-discovery
luck, and DSR's `n_trials` is unknowable without an honest record of every hypothesis run.

```julia
struct TrialRecord; sleeve::String; hypothesis::String; date::Date; n_variations::Int; observed_sharpe::Float64; end
# ledger.jsonl  — append-only; EVERY tested variation logged (the honest input to DSR/PBO)
corpus_fdr(records; method=:benjamini_hochberg) → per-sleeve q-values + family-wise survivor set
deflated_sharpe_corpus(keepers) → each keeper's DSR using the TRUE corpus-wide trial count, not a self-reported one
```
- Output: of the ~55 sleeves, which "keepers" survive Benjamini-Hochberg at q<0.1 / deflated-Sharpe>0.5 **after** counting
  all shots taken. Expect 1–2 current keepers to not survive.
- Pairs with `mirage`: FDR governs *across* tests, mirage governs *within* a test's specification.

## 4. Breakthrough shrinkage — robustness-weighted allocation  *(turns verdicts into capital)*

Standard allocators treat historical Sharpe as expected Sharpe — the core flaw. Shrink expected return toward the null by
each sleeve's *robustness* before risk-parity.

```julia
struct KeeperSleeve; name; raw_expected_return; raw_vol; blind_result; mirage_verdict; correlation_vector; end
# confidence_weight = (1 - blind_p)^γ                     # barely-passing Blind ⇒ heavy shrink
# eff_n_penalty     = min(1, effective_n / 30)            # too few independent trades ⇒ haircut
# mirage_penalty    = {ROBUST:1.0, FRAGILE:0.5, NULL/MIRAGE:0.0}   # spec-fragile ⇒ shrink hard   ← add this
# shrunk_return = raw_expected_return · confidence_weight · eff_n_penalty · mirage_penalty
# allocate: risk-parity on cov, drop shrunk_return ≤ 0, scale to target vol
```
- **Inputs must be NET (post-Friction) and post-mirage** — not gross. A FRAGILE mirage verdict (e.g. burdensome) gets its
  expected return halved; a NULL gets zero. This is the single switch that stops lucky/spec-fragile alpha drawing capital.
- A cleaner v2 shrinks the *posterior* Sharpe via the NW t-stat (proper Bayesian) rather than the (1−p)^γ heuristic.

## 5. Sleeve-correlation / factor-exposure map — is the book actually diversified?  *(my addition)*

~55 sleeves, but many are momentum/carry/vol/credit variants. Before allocating, prove the keeper book isn't *one bet*.

```julia
correlation_map(keeper_returns)  → NxN corr + hierarchical clusters (which sleeves are secretly the same trade)
factor_exposure(keeper_book, factors)  → regress the combined book on {MKT, momentum, value, credit, short-vol, trend/CTA}
                                        → how much of "alpha" is just factor beta; residual = true diversifying alpha
```
- Output: the breakthrough portfolio's factor decomposition + an effective-number-of-bets (1/ΣΣρ) metric.

---

## Mirage (built) — first results

Ran the sign-stability audit on the two most control-dependent verdicts (see `mirage.py`):
- **burdensome** — "cat premium distinct from puts": **FRAGILE.** The +15.7%/yr raw alpha collapses to **+2.6%** and is
  significant in **only 2%** of control specs; SPY/QUAL/PUTW/HYG each raise R² while killing alpha (collinear ~0.6). The
  "distinct specialty premium" was largely equity+quality+credit beta — the narrow SPY+PUTW control set flattered it.
- **blemish** — "quality is value in disguise": **NULL (robustly).** Quality-over-SPY alpha is ~0 under every spec; no
  edge to be a mirage, verdict confirmed.

**Action:** fold the burdensome finding back into its README (downgrade the "distinct" language), and run mirage on the
other control-based verdicts (`bifurcate`, `boulder`, `bigbrother`) before any capital logic.

---

# STATUS UPDATE (Oct 6 2026) — all modules BUILT

Every spec above is now a runnable module in this repo (Python prototype; port to Julia `base` where the data lives):
`friction.py` · `ledger.py` · `breakthrough.py` · `sleevemap.py` · `stress_wrapper.py` (+ earlier `nullbar/validation/stress/mirage`).

**Two demo findings worth acting on** (stylized inputs, but the lesson is real — re-run with the real sleeve returns):

- **Corpus FDR (`ledger.py`)**: with ~55 sleeves, the **expected max Sharpe by luck alone is ≈ +0.89**. Over a realistic
  corpus (50 null + 5 real), only a ~1.2-Sharpe sleeve (bankroll-like) clears Benjamini-Hochberg q<0.10 — the **mid-tier
  "keepers" (Sharpe 0.35–0.61: borough, bulwark, blend-crack, bullion-RV) do NOT survive family-wise FDR**, and a lucky
  null can sneak in. This confirms the standing skepticism: across 55 shots, only the strongest edge is real; the rest is
  selection luck until proven otherwise.
- **Sleeve map (`sleevemap.py`)**: a 6-keeper proxy book shows **~1.8 effective bets, 68% factor-R², ~0 residual alpha** —
  the "diversified" book is largely MKT/low-vol/momentum beta wearing six names. Allocate on the *residual*, collapse
  correlated clusters to one risk budget.

**Next real step (not a demo):** run `ledger.corpus_fdr` and `sleevemap` on the *actual* returns of all ~55 sleeves — it
will almost certainly cut the keeper list and show the book is more concentrated than it looks. That's the honest
corpus-level verdict the individual sleeve READMEs can't give.
