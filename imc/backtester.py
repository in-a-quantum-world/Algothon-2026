"""
IMC Prosperity Backtester
==========================
Reads historical_data.csv (or synthetic_data.csv), reconstructs TradingState
at each timestamp, runs the Trader, simulates fills, tracks PnL.

CSV format expected (IMC standard export format):
  timestamp, symbol,
  bid_price_1, bid_volume_1, bid_price_2, bid_volume_2, bid_price_3, bid_volume_3,
  ask_price_1, ask_volume_1, ask_price_2, ask_volume_2, ask_price_3, ask_volume_3,
  mid_price, profit_and_loss

Usage:
  python backtester.py                         # uses historical_data.csv
  python backtester.py my_data.csv             # custom file
  python backtester.py my_data.csv --plot      # show charts
"""

from __future__ import annotations
import sys
import os
import csv
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from trader_framework import (
    Order, OrderDepth, TradingState, Trade, Listing,
    POSITION_LIMITS, DEFAULT_POSITION_LIMIT,
)
from trader import Trader


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _float(val: str) -> Optional[float]:
    try:
        v = float(val)
        return v if v == v else None   # filter NaN
    except (ValueError, TypeError):
        return None


def _int(val: str) -> Optional[int]:
    f = _float(val)
    return int(f) if f is not None else None


def build_order_depth(row: dict) -> OrderDepth:
    """Parse up to 3 levels of bid/ask from a CSV row."""
    buy_orders: Dict[int, int] = {}
    sell_orders: Dict[int, int] = {}

    for i in range(1, 4):
        bp = _int(row.get(f"bid_price_{i}", ""))
        bv = _int(row.get(f"bid_volume_{i}", ""))
        ap = _int(row.get(f"ask_price_{i}", ""))
        av = _int(row.get(f"ask_volume_{i}", ""))

        if bp is not None and bv is not None and bv > 0:
            buy_orders[bp] = bv
        if ap is not None and av is not None:
            # IMC stores ask volumes as negative; accept either sign
            sell_orders[ap] = -abs(av)

    return OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)


def group_by_timestamp(rows: List[dict]) -> List[Tuple[int, Dict[str, dict]]]:
    """Group rows by timestamp → list of (ts, {symbol: row})."""
    groups: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for row in rows:
        ts = _int(row.get("timestamp", "0")) or 0
        sym = row.get("product", row.get("symbol", "UNKNOWN")).strip()
        groups[ts][sym] = row
    return sorted(groups.items())


# ─────────────────────────────────────────────────────────────────────────────
# Order matching simulation
# ─────────────────────────────────────────────────────────────────────────────

def match_orders(
    our_orders: List[Order],
    order_depth: OrderDepth,
    market_trades_this_tick: List[Trade],
    position: int,
    limit: int,
) -> Tuple[List[Trade], int, float]:
    """
    Returns (fills, new_position, cash_delta).

    Matching rules (mirrors IMC engine):
    1. Aggressive orders: our bid >= market ask → fill immediately.
    2. Passive market-making fills: if market bots trade at a price
       that crosses our resting quote, attribute the fill to us.
    """
    fills: List[Trade] = []
    cash_delta = 0.0
    pos = position

    for order in our_orders:
        remaining = order.quantity  # positive = buy, negative = sell

        if remaining > 0:
            # We are buying: fill against resting asks
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if ask_px > order.price or remaining <= 0:
                    break
                available = abs(order_depth.sell_orders[ask_px])
                fill_qty = min(remaining, available, limit - pos)
                if fill_qty <= 0:
                    break
                fills.append(Trade(order.symbol, ask_px, fill_qty))
                cash_delta -= ask_px * fill_qty
                pos += fill_qty
                remaining -= fill_qty
        else:
            # We are selling: fill against resting bids
            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_px < order.price or remaining >= 0:
                    break
                available = order_depth.buy_orders[bid_px]
                fill_qty = min(-remaining, available, limit + pos)
                if fill_qty <= 0:
                    break
                fills.append(Trade(order.symbol, bid_px, -fill_qty))
                cash_delta += bid_px * fill_qty
                pos -= fill_qty
                remaining += fill_qty

    # ── Passive MM fills ──────────────────────────────────────────────
    # If market bots traded at a price that crosses our resting quote,
    # attribute fill to us (IMC engine behaviour for resting orders).
    resting_bids = {o.price: o.quantity for o in our_orders if o.quantity > 0}
    resting_asks = {o.price: -o.quantity for o in our_orders if o.quantity < 0}

    for trade in market_trades_this_tick:
        # Market sold to us (at our bid or below)
        for bid_px, bid_qty in list(resting_bids.items()):
            if trade.price <= bid_px and bid_qty > 0:
                fill_qty = min(bid_qty, abs(trade.quantity), limit - pos)
                if fill_qty > 0:
                    fills.append(Trade(order.symbol if our_orders else trade.symbol,
                                       bid_px, fill_qty))
                    cash_delta -= bid_px * fill_qty
                    pos += fill_qty
                    resting_bids[bid_px] -= fill_qty

        # Market bought from us (at our ask or above)
        for ask_px, ask_qty in list(resting_asks.items()):
            if trade.price >= ask_px and ask_qty > 0:
                fill_qty = min(ask_qty, abs(trade.quantity), limit + pos)
                if fill_qty > 0:
                    fills.append(Trade(order.symbol if our_orders else trade.symbol,
                                       ask_px, -fill_qty))
                    cash_delta += ask_px * fill_qty
                    pos -= fill_qty
                    resting_asks[ask_px] -= fill_qty

    return fills, pos, cash_delta


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic market-trade simulator
# ─────────────────────────────────────────────────────────────────────────────

_SIM_RNG = random.Random(0)   # fixed seed → deterministic replays

def simulate_market_trades(
    sym: str,
    od: OrderDepth,
    n_trades: int = 3,
    max_vol: int = 8,
) -> List[Trade]:
    """
    Simulate `n_trades` random market trades per tick for `sym`.

    Each trade is a random buy or sell hitting the NBBO:
      - market buy  → trades at best_ask  (lifts our resting ask)
      - market sell → trades at best_bid  (hits our resting bid)

    This is how IMC's engine creates market_trades: other bots continuously
    send market/aggressive orders that cross the spread, and those crosses
    are attributed to any resting order at that price level.
    """
    bb = od.best_bid()
    ba = od.best_ask()
    if bb is None or ba is None:
        return []

    trades = []
    for _ in range(n_trades):
        vol = _SIM_RNG.randint(1, max_vol)
        if _SIM_RNG.random() < 0.5:
            trades.append(Trade(sym, ba, vol))   # market buy hits best ask
        else:
            trades.append(Trade(sym, bb, -vol))  # market sell hits best bid
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Backtester engine
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, csv_path: str, verbose: bool = False):
        self.csv_path = csv_path
        self.verbose = verbose
        self.trader = Trader()

        self.positions: Dict[str, int] = defaultdict(int)
        self.cash: Dict[str, float] = defaultdict(float)   # cash per product
        self.pnl_history: Dict[str, List[float]] = defaultdict(list)
        self.ts_history: List[int] = []
        self.trade_log: List[dict] = []

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        rows = load_csv(self.csv_path)
        timestamps = group_by_timestamp(rows)

        prev_own_trades: Dict[str, List[Trade]] = defaultdict(list)

        for ts, sym_rows in timestamps:
            # ── Build TradingState ────────────────────────────────────
            order_depths: Dict[str, OrderDepth] = {}
            listings: Dict[str, Listing] = {}
            market_trades: Dict[str, List[Trade]] = defaultdict(list)

            for sym, row in sym_rows.items():
                od = build_order_depth(row)
                order_depths[sym] = od
                listings[sym] = Listing(sym, sym, "SEASHELLS")
                market_trades[sym] = simulate_market_trades(sym, od)

            state = TradingState(
                timestamp=ts,
                listings=listings,
                order_depths=order_depths,
                own_trades=dict(prev_own_trades),
                market_trades=dict(market_trades),
                position=dict(self.positions),
                observations={},
            )

            # ── Run trader ────────────────────────────────────────────
            try:
                all_orders: Dict[str, List[Order]] = self.trader.run(state)
            except Exception as e:
                print(f"[Backtester] ERROR at ts={ts}: {e}")
                all_orders = {}

            # ── Simulate fills ────────────────────────────────────────
            new_own_trades: Dict[str, List[Trade]] = defaultdict(list)

            for sym, orders in all_orders.items():
                od = order_depths.get(sym, OrderDepth())
                limit = POSITION_LIMITS.get(sym, DEFAULT_POSITION_LIMIT)
                pos = self.positions[sym]

                fills, new_pos, cash_delta = match_orders(
                    orders, od, market_trades.get(sym, []), pos, limit
                )

                self.positions[sym] = new_pos
                self.cash[sym] += cash_delta
                new_own_trades[sym].extend(fills)

                for fill in fills:
                    self.trade_log.append({
                        "timestamp": ts,
                        "symbol": sym,
                        "price": fill.price,
                        "quantity": fill.quantity,
                        "side": "BUY" if fill.quantity > 0 else "SELL",
                    })

                if self.verbose and fills:
                    print(f"  [{sym}] fills={fills} pos={new_pos} cash_delta={cash_delta:.0f}")

            prev_own_trades = new_own_trades

            # ── Mark-to-market PnL ────────────────────────────────────
            self.ts_history.append(ts)
            for sym, row in sym_rows.items():
                mid = _float(row.get("mid_price", ""))
                if mid is None:
                    od = order_depths.get(sym)
                    mid = od.mid_price() if od else 0.0
                if mid is None:
                    mid = 0.0
                mtm = self.cash[sym] + self.positions[sym] * mid
                self.pnl_history[sym].append(mtm)

        return self._final_pnl()

    # ------------------------------------------------------------------
    def _final_pnl(self) -> Dict[str, float]:
        return {sym: hist[-1] if hist else 0.0 for sym, hist in self.pnl_history.items()}

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        pnl = self._final_pnl()
        total = sum(pnl.values())
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"{'Product':<20} {'Final PnL':>12}")
        print("-" * 35)
        for sym, p in sorted(pnl.items(), key=lambda x: -x[1]):
            print(f"{sym:<20} {p:>12.2f}")
        print("-" * 35)
        print(f"{'TOTAL':<20} {total:>12.2f}")
        print(f"Total trades executed: {len(self.trade_log)}")

    # ------------------------------------------------------------------
    def plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Backtester] matplotlib not installed. Skipping plot.")
            return

        syms = list(self.pnl_history.keys())
        fig, axes = plt.subplots(len(syms) + 1, 1,
                                 figsize=(12, 3 * (len(syms) + 1)),
                                 sharex=True)
        if len(syms) == 0:
            print("[Backtester] No data to plot.")
            return

        # Total PnL
        total_pnl = [
            sum(self.pnl_history[s][i] if i < len(self.pnl_history[s]) else 0
                for s in syms)
            for i in range(len(self.ts_history))
        ]
        axes[0].plot(self.ts_history, total_pnl, color="black", linewidth=2)
        axes[0].set_title("Total PnL")
        axes[0].set_ylabel("PnL (seashells)")
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].grid(True, alpha=0.3)

        for ax, sym in zip(axes[1:], syms):
            ax.plot(self.ts_history, self.pnl_history[sym], label=sym)
            ax.set_title(f"{sym} PnL")
            ax.set_ylabel("PnL")
            ax.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestamp")
        plt.tight_layout()
        plt.savefig("backtest_results.png", dpi=150)
        print("[Backtester] Chart saved: backtest_results.png")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMC Prosperity Backtester")
    parser.add_argument("csv", nargs="?", default="historical_data.csv",
                        help="Path to CSV data file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        # Try relative to imc/ dir, then project root
        here = os.path.dirname(__file__)
        candidates = [
            os.path.join(here, csv_path),
            os.path.join(here, "..", csv_path),
            csv_path,
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

    if not os.path.exists(csv_path):
        print(f"[Backtester] ERROR: file not found: {csv_path}")
        sys.exit(1)

    print(f"[Backtester] Loading {csv_path} ...")
    bt = Backtester(csv_path, verbose=args.verbose)
    bt.run()
    bt.print_summary()

    if args.plot:
        bt.plot()

    return bt


if __name__ == "__main__":
    main()
