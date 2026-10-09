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

---

# REAL-RETURNS CORPUS AUDIT (Oct 6 2026) — ran it on the actual corpus

Replaces the stylized demo above with the real numbers. Also ported the whole toolkit into the canonical Julia engine
(`blaquebaux/base` → `src/module_14_nullbar`, `module Nullbar`, stdlib-only; verified standalone).

**1. Corpus FDR on the real ~44 independent sleeve tests** (infra/capstones/known artifacts excluded; top candidates'
headline Sharpes verified NET from each README's verdict):

- **Expected max Sharpe by luck alone ≈ +0.93.** Any single sleeve under ~0.9 Sharpe is statistically indistinguishable
  from the best-of-luck draw across 44 shots.
- On **raw strategy Sharpe**, ~12 sleeves "survive" BH q<0.10 — but that list is polluted: `bridle`/`blemish`/`baton`
  are *documented nulls* whose high Sharpe is the **raw asset** (high-β BAB, quality factor), not an edge; and
  `bull`/`broad`/`bitdollar` are **factor beta** (corr≈1), not alpha. The pollution is the lesson — raw Sharpe ranks beta.
- On **edge-over-benchmark (alpha) Sharpe**, with documented nulls set to 0 and beta sleeves reduced to residual, only
  **4 survive family-wise BH (q<0.10, p-threshold 0.0075): `bankroll` (+1.08), `blunt` (+1.00), `buzz` (+0.88),
  `bind` (+0.86)** — and `buzz`/`bind` are **not yet mirage-audited**. The confirmed, spec-robust, non-beta, FDR-surviving
  edges are therefore **just `bankroll` and `blunt`** pending a mirage pass on the other two.
- **`boom` (+0.55 beta-neutral momentum) is a genuine edge but does NOT survive** family-wise (p≈0.06 > 0.0075) — real,
  yet not distinguishable from luck once you count all 44 shots. The honest fate of a mid-tier edge in a large corpus.

**2. Sleeve map on the real keeper proxies (live Alpaca):** the 6-sleeve proxy book = **1.8 effective bets, +0.48 avg
pairwise corr, 68% factor-R², residual alpha +0.7%/yr, residual Sharpe ≈ 0.00.** Betas: MKT +0.63, LOWVOL +0.20,
MOM +0.19. Three sleeves collapse into one cluster. The "diversified" book is MKT+low-vol+momentum beta wearing six names.

**Bottom line:** across the real corpus, **2 confirmed diversifying edges** (bankroll, blunt), 2 FDR-survivors pending
mirage (buzz, bind), and a keeper book that is ~1.8 independent bets of mostly factor beta. This is the corpus-level
verdict the individual READMEs can't give — and it's exactly what the `breakthrough` shrinkage allocator is built to act
on: shrink the beta/lucky sleeves toward zero, size the few residual edges. Headline Sharpe drops; honest Sharpe rises.

---

# TOOLKIT BATCH-2 (Oct 7 2026) — eight new gates

Eight additions beyond the original pipeline, built from a triage of a large proposal list.
**Honesty note:** the proposal cited several 2026 "papers/packages" (LAZARUS, holdout-first,
monte-neo, DeePM, an MSCI note) that could not be verified; these are built on the *established
methods* behind them (CUSUM, Hansen SPA, Harvey-Leybourne-Newbold / Fair-Shiller encompassing,
Leamer EBA, the Harvey-Liu haircut, robustness/plateau scoring, McCloskey-Ziliak, lookahead
gates, trades-per-parameter), not on the unverifiable claims. Dupes were merged (search-gaming
appeared 3×, encompassing 2×, CUSUM 2×); two proposed "additions" were sleeves, not layers
(term-structure momentum, prediction-market OFI) and are parked for the `start <name>` track;
RF+SHAP regime decomposition and a critical-slowing EWS were deferred (break stdlib-only / too
speculative). All eight below are runnable in `nullbar` (Python) and ported to Julia `base`
(module_14, `module Nullbar`), stdlib-only, smoke-tested.

| # | gate | module | catches |
|---|---|---|---|
| 1 | `prefix_invariance` (lookahead gate) | `causality.py` | a position that changes when future bars appear — look-ahead DSR/PBO can't see |
| 2 | `parameter_budget` | `validation.py` | degrees-of-freedom overfit (trades-per-parameter < 50:1) |
| 3 | `plateau_score` | `validation.py` | peak-selection overfit (isolated spike vs broad ridge) |
| 4 | `economic_significance` | `economic.py` | statistically real but economically trivial alpha (pre-registered hurdle) |
| 5 | `sharpe_haircut` | `breakthrough.py` | uniform multiple-testing penalty — the haircut is nonlinear (brutal on 0.4s, mild on 1.0s) |
| 6 | `forecast_encompassing` | `sleevemap.py` | redundant keepers (does edge A encompass edge B at the forecast level) |
| 7 | `monotone_corpus_bar` | `ledger.py` | a search gaming its own FDR bar by diluting N — variance floor + monotone high-water-mark |
| 8 | `asset_attribution` / `universe_attribution` | `attribution.py` | **new layer** — alpha that is asset-selection, not logic (fix logic, vary universe) |

**Most consequential for the corpus:** #4 (economic gate) and #6 (encompassing). The economic
hurdle can cut a statistically-significant-but-trivial "survivor"; encompassing asks whether the
two confirmed edges (bankroll, blunt) are genuinely independent or one encompasses the other —
the next real-returns check to run once their forecast series are persisted.

**Still parked** (deliberately): CUSUM live-break monitor and regime-dependence diagnostic
(need a live book / overlap existing layers), Step-SPA (heavier bootstrap, small corpus),
data-correction provenance and causal-sieve (cheap, low urgency), the leakage-safe search
registry and monotone-bar enforcement inside an automated loop (belong to a future
"base automates discovery" project), and the two new sleeves above.

---

# CONTEXTUAL RESCUE (Oct 9 2026) — the first capstone-phase result

`contextual_rescue.py` answers the question the per-sleeve READMEs can't: a null is a *standalone*
verdict — does a small dose earn its slot **in a book**? Each overlay is judged against the same 10%
in bonds (IEF/TLT), on the **tail** (skew, worst month, maxDD), not Sharpe. It must be able to fail.

Run on an equity-tilted keeper book (SPY/QQQ/MTUM/DIVO/MUB/EMB/GLD), 2016–2026:

| overlay (+10%) | Sharpe | maxDD | worst mo | skew | verdict |
|---|---|---|---|---|---|
| base keeper book | +1.01 | −26% | −8.6% | −0.35 | — |
| **bleed / long-vol (VIXM)** | **+1.15** | **−18%** | **−6.0%** | **+0.13** | **RESCUED** |
| +10% IEF / TLT (control) | +1.02 / +1.01 | −23% / −22% | −7.5% / −8.0% | −0.33 / −0.31 | the bar |
| bide (cash ladder) | +0.99 | −25% | −8.4% | −0.31 | not rescued (bonds win) |
| bastion / managed futures (DBMF) | +1.00 | −24% | −7.4% | −0.36 | marginal |

**Finding:** the convex tail hedge is the null that earns its keep — a 10% long-vol overlay raises book
Sharpe, cuts maxDD by 8 points, and **flips book skew positive**, which bonds cannot do. The patient-cash
posture (`bide`) is *not* rescued — bonds cut drawdown/worst-month more at equal Sharpe, so it stays
shelved even in context (the narrative lost to the control). Managed futures is marginal. The long-vol
win is partly a **rebalancing premium** (trim after spikes, rebuy cheap) — which is exactly what an
allocator does, and why "in context, rebalanced" ≠ "standalone, buy-and-hold." **The graveyard's value
is real but conditional:** it shows up in the right layer, rebalanced, at a small dose — not standalone.
This is the empirical backbone for the preservation capstone.
