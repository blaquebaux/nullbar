#!/usr/bin/python3
# =============================================================================
# friction.py — honest, non-linear execution cost + capacity (execution-fragility layer).
#
# A flat "N bps round-trip" is a backtester's lie: real impact is non-linear (√ of participation), state-dependent
# (scales with volatility), and asymmetric (shorts pay borrow). This module implements the Almgren-Chriss square-root
# market-impact model, dynamic short borrow, and — the real deliverable — a CAPACITY curve (net Sharpe vs AUM) and the
# break-even AUM at which impact eats the edge. It re-grades thin "keepers" on what they'd actually net.
#
#   one-way slippage = half_spread + impact_coeff · daily_vol · sqrt(trade_usd / ADV_usd)
#   short borrow/day = base_rate · (htb_multiplier if utilization > ~0.80 else 1)
#
# ADV & daily-vol come from Alpaca (price×volume, returns). Spread/borrow/utilization are assumptions here (Alpaca
# lacks them) — wire real values in the Julia `base` feed. Pure NumPy.
# =============================================================================
import math
import numpy as np

class StaticFriction:
    def __init__(self, half_spread_bps=2.5, commission_bps=1.0):
        self.half_spread_bps, self.commission_bps = half_spread_bps, commission_bps
    def one_way(self, *a, **k): return (self.half_spread_bps + self.commission_bps) / 1e4

class InstitutionalFriction:
    def __init__(self, impact_coeff=0.3, half_spread_bps=2.5, borrow_base_bps=25.0, htb_multiplier=5.0):
        self.impact_coeff, self.half_spread_bps = impact_coeff, half_spread_bps
        self.borrow_base_bps, self.htb_multiplier = borrow_base_bps, htb_multiplier
    def one_way(self, trade_usd, adv_usd, daily_vol):
        part = trade_usd / max(adv_usd, 1.0)
        return self.half_spread_bps / 1e4 + self.impact_coeff * daily_vol * math.sqrt(part)
    def borrow_day(self, utilization=0.0):
        base = self.borrow_base_bps / 1e4 / 252.0
        return base * (self.htb_multiplier if utilization > 0.80 else 1.0)

def apply_friction(gross_trade_rets, trade_usd, adv_usd, daily_vol, directions, holds, model, utilization=None):
    """Convert gross per-trade returns to NET, direction-aware (slippage both legs; borrow on shorts over the hold)."""
    g = np.asarray(gross_trade_rets, float); net = np.empty_like(g)
    adv = np.broadcast_to(adv_usd, g.shape); dv = np.broadcast_to(daily_vol, g.shape)
    tu = np.broadcast_to(trade_usd, g.shape); dr = np.broadcast_to(directions, g.shape); hd = np.broadcast_to(holds, g.shape)
    util = np.zeros_like(g) if utilization is None else np.broadcast_to(utilization, g.shape)
    for i in range(len(g)):
        ow = model.one_way(tu[i], adv[i], dv[i]) if isinstance(model, InstitutionalFriction) else model.one_way()
        rt = 2 * ow
        borrow = (model.borrow_day(util[i]) * hd[i]) if (isinstance(model, InstitutionalFriction) and dr[i] < 0) else 0.0
        net[i] = (1 + g[i]) * (1 - rt - borrow) - 1
    return net

def capacity_curve(gross_ann_return, gross_ann_vol, annual_turnover, adv_usd, daily_vol, model,
                   aum_grid=None, sharpe_floor=0.3):
    """Net Sharpe vs AUM, and the break-even AUM where impact eats the edge. annual_turnover = $ traded per $ capital
    per year (e.g. 10 = rebalances its book ~5x round-trip). At each AUM the average trade participates
    (AUM·turnover)/ADV of the daily volume → √-impact drag scales up with size."""
    aum_grid = aum_grid if aum_grid is not None else np.array([1e6,5e6,1e7,5e7,1e8,5e8,1e9,5e9])
    rows = []
    for aum in aum_grid:
        trade_usd = aum * annual_turnover / 252.0                 # avg daily $ traded (turnover spread over the year)
        ow = model.one_way(trade_usd, adv_usd, daily_vol) if isinstance(model, InstitutionalFriction) else model.one_way()
        drag = annual_turnover * ow                               # annual cost = turnover × one-way (each $ traded pays ~one-way)
        net_ret = gross_ann_return - drag
        rows.append((aum, net_ret, net_ret / gross_ann_vol if gross_ann_vol > 0 else float("nan"), drag))
    arr = np.array(rows)
    below = arr[arr[:,2] < sharpe_floor]
    breakeven = float(below[0,0]) if len(below) else float("inf")
    return dict(grid=arr, breakeven_aum=breakeven)

# ---- demo ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os, json, urllib.request
    H={"APCA-API-KEY-ID":os.environ["ALPACA_KEY_ID"],"APCA-API-SECRET-KEY":os.environ["ALPACA_SECRET_KEY"]}
    def bars(s):
        u=(f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day&start=2024-01-01&end=2026-08-01&adjustment=all&feed=sip&limit=10000")
        d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=40)); return d.get("bars",{}).get(s,[])
    def adv_vol(s):
        b=bars(s); c=np.array([x["c"] for x in b]); v=np.array([x["v"] for x in b])
        advusd=float(np.median(c[-60:]*v[-60:])); dvol=float((c[1:]/c[:-1]-1)[-60:].std())
        return advusd, dvol
    print("="*100); print("FRICTION — Almgren-Chriss impact + capacity (demo)"); print("="*100)
    model=InstitutionalFriction(impact_coeff=0.3)
    for s in ["AAPL","IWM","CRAK"]:
        adv,dv=adv_vol(s)
        print(f"\n  {s}: dollar-ADV ${adv/1e6:.0f}M, daily vol {dv*100:.1f}%")
        for tusd in [1e6,1e7,1e8]:
            print(f"     trade ${tusd/1e6:>4.0f}M → one-way slippage {model.one_way(tusd,adv,dv)*1e4:6.1f} bps  (participation {tusd/adv*100:.0f}% of ADV)")
    # capacity curve for a 0.8-Sharpe, 15% ann-return, 12%-vol sleeve trading an IWM-liquidity book, turnover 12x/yr
    adv,dv=adv_vol("IWM")
    cap=capacity_curve(0.15, 0.12, annual_turnover=12.0, adv_usd=adv, daily_vol=dv, model=model)
    print(f"\n  CAPACITY (gross 15%/yr @12% vol, 12x turnover, IWM-liquidity):")
    print(f"     {'AUM':>8}{'net ret':>10}{'net Sharpe':>12}{'cost drag':>11}")
    for aum,nr,ns,dg in cap["grid"]:
        print(f"     ${aum/1e6:>6.0f}M{nr*100:>+9.1f}%{ns:>+12.2f}{dg*100:>+10.1f}%")
    print(f"     → break-even AUM (net Sharpe < 0.3): ${cap['breakeven_aum']/1e6:,.0f}M" if np.isfinite(cap['breakeven_aum']) else "     → scales past $5B at this liquidity")
    print("\nREAD: slippage is non-linear in trade size; the capacity curve says how much capital a sleeve holds before")
    print("  impact eats its Sharpe. A thin-but-real small-cap/microstructure sleeve may cap at tens of $M, not billions.")
