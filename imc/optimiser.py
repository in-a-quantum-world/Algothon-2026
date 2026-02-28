"""
IMC Prosperity Parameter Optimiser
=====================================
Grid-searches strategy parameters by running the backtester multiple times.

Searches:
  FixedFairMM  : spread ∈ [1, 2, 3, 4, 5]
  RollingMidMM : window ∈ [5, 10, 20, 50, 100]
  SpreadArb    : z_threshold ∈ [0.5, 1.0, 1.5, 2.0, 2.5]
  AS-MM        : gamma ∈ [0.05, 0.1, 0.3, 0.7, 1.0]  ×  order_size ∈ [1, 2, 5]

Usage:
  python optimiser.py                          # uses historical_data.csv
  python optimiser.py synthetic_data.csv
  python optimiser.py --as-mm-only             # run only the AS-MM grid search
"""

from __future__ import annotations
import sys
import os
import copy
import argparse
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from backtester import Backtester


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _total_pnl(bt: Backtester) -> float:
    return sum(bt._final_pnl().values())


def _pnl_for_products(bt: Backtester, products: List[str]) -> float:
    pnl = bt._final_pnl()
    return sum(pnl.get(p, 0.0) for p in products)


def run_backtest(csv_path: str, patch_fn) -> Backtester:
    """
    Run one backtest after applying `patch_fn` which mutates the trader config.
    We import trader module fresh each run so config changes take effect.
    """
    import importlib
    import trader as trader_mod
    import trader_framework
    import strategy_modules

    # Force-reload so our config changes are picked up
    importlib.reload(strategy_modules)
    importlib.reload(trader_mod)

    patch_fn(trader_mod)

    importlib.reload(trader_mod)  # reload again after patching module-level dicts

    bt = Backtester(csv_path, verbose=False)
    # Swap trader instance so it uses the patched config
    import importlib
    bt.trader = trader_mod.Trader()
    bt.run()
    return bt


# ─────────────────────────────────────────────────────────────────────────────
# Grid searches
# ─────────────────────────────────────────────────────────────────────────────

def optimise_fixed_fair(csv_path: str, symbols: List[str]) -> Dict[str, Any]:
    """Grid-search spread for FixedFairMarketMaker products."""
    spreads = [1, 2, 3, 4, 5]
    results: List[Tuple[int, float]] = []

    print("\n[Optimiser] FixedFairMM spread search:", spreads)
    for spread in spreads:
        import trader as trader_mod

        def patch(mod, s=spread, syms=symbols):
            for sym in syms:
                if sym in mod.FIXED_FAIR_PRODUCTS:
                    mod.FIXED_FAIR_PRODUCTS[sym]["spread"] = s

        bt = Backtester(csv_path, verbose=False)
        bt.trader = _make_trader_with_patch(lambda mod, s=spread, syms=symbols: [
            mod.FIXED_FAIR_PRODUCTS.__setitem__(sym, {**mod.FIXED_FAIR_PRODUCTS[sym], "spread": s})
            for sym in syms if sym in mod.FIXED_FAIR_PRODUCTS
        ])
        bt.run()
        pnl = _pnl_for_products(bt, symbols)
        results.append((spread, pnl))
        print(f"  spread={spread:2d} → PnL={pnl:.2f}")

    best_spread, best_pnl = max(results, key=lambda x: x[1])
    print(f"  ✓ Best spread={best_spread} PnL={best_pnl:.2f}")
    return {"spread": best_spread, "pnl": best_pnl}


def optimise_rolling_mid(csv_path: str, symbols: List[str]) -> Dict[str, Any]:
    """Grid-search window size for RollingMidMarketMaker products."""
    windows = [5, 10, 20, 50, 100]
    results: List[Tuple[int, float]] = []

    print("\n[Optimiser] RollingMidMM window search:", windows)
    for window in windows:
        bt = Backtester(csv_path, verbose=False)
        bt.trader = _make_trader_with_patch(lambda mod, w=window, syms=symbols: [
            mod.ROLLING_MID_PRODUCTS.__setitem__(sym, {**mod.ROLLING_MID_PRODUCTS[sym], "window": w})
            for sym in syms if sym in mod.ROLLING_MID_PRODUCTS
        ])
        bt.run()
        pnl = _pnl_for_products(bt, symbols)
        results.append((window, pnl))
        print(f"  window={window:3d} → PnL={pnl:.2f}")

    best_window, best_pnl = max(results, key=lambda x: x[1])
    print(f"  ✓ Best window={best_window} PnL={best_pnl:.2f}")
    return {"window": best_window, "pnl": best_pnl}


def optimise_spread_arb(csv_path: str, basket_symbols: List[str]) -> Dict[str, Any]:
    """Grid-search z_threshold for SpreadArbitrage."""
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    results: List[Tuple[float, float]] = []

    # Collect all component symbols too
    import trader as trader_mod
    related_products: List[str] = list(basket_symbols)
    for sym in basket_symbols:
        cfg = trader_mod.BASKET_CONFIGS.get(sym, {})
        related_products.extend(cfg.get("components", {}).keys())

    print("\n[Optimiser] SpreadArb z_threshold search:", thresholds)
    for z in thresholds:
        bt = Backtester(csv_path, verbose=False)
        bt.trader = _make_trader_with_patch(lambda mod, zz=z, syms=basket_symbols: [
            mod.BASKET_CONFIGS.__setitem__(sym, {**mod.BASKET_CONFIGS[sym], "z_threshold": zz})
            for sym in syms if sym in mod.BASKET_CONFIGS
        ])
        bt.run()
        pnl = _pnl_for_products(bt, related_products)
        results.append((z, pnl))
        print(f"  z={z:.1f} → PnL={pnl:.2f}")

    best_z, best_pnl = max(results, key=lambda x: x[1])
    print(f"  ✓ Best z_threshold={best_z} PnL={best_pnl:.2f}")
    return {"z_threshold": best_z, "pnl": best_pnl}


# ─────────────────────────────────────────────────────────────────────────────
# AS-MM grid search
# ─────────────────────────────────────────────────────────────────────────────

def _passive_baseline_pnl(csv_path: str, symbols: List[str]) -> float:
    """
    Run a backtest where the listed symbols use passive 1-lot quotes
    (best bid / best ask) instead of the AS-MM.  Every other product is
    handled by its normal strategy so the comparison is apples-to-apples.
    """
    from trader import Trader
    from trader_framework import Order, POSITION_LIMITS, DEFAULT_POSITION_LIMIT

    class _PassiveTrader:
        def __init__(self, syms):
            self._inner = Trader()
            self._passive_syms = set(syms)

        def run(self, state):
            result = self._inner.run(state)
            for sym in self._passive_syms:
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                pos   = state.get_position(sym)
                limit = POSITION_LIMITS.get(sym, DEFAULT_POSITION_LIMIT)
                bb, ba = od.best_bid(), od.best_ask()
                orders = []
                if bb and min(1, limit - pos) > 0:
                    orders.append(Order(sym, bb,  min(1, limit - pos)))
                if ba and min(1, limit + pos) > 0:
                    orders.append(Order(sym, ba, -min(1, limit + pos)))
                result[sym] = orders
            return result

    bt = Backtester(csv_path, verbose=False)
    bt.trader = _PassiveTrader(symbols)
    bt.run()
    return _pnl_for_products(bt, symbols)


def optimise_as_mm(csv_path: str, symbols: List[str]) -> Dict[str, Any]:
    """
    2-D grid search over gamma × order_size for AvellanedaStoikovMarketMaker.

    Strategy: pre-populate trader.as_mm[sym] with a custom AS-MM instance
    before the backtest starts.  Trader.run() checks `if sym not in self.as_mm`
    before lazy-creating, so our pre-seeded instance is always used.
    """
    from trader import Trader
    from strategy_modules import AvellanedaStoikovMarketMaker

    gammas      = [0.05, 0.1, 0.3, 0.7, 1.0]
    order_sizes = [1, 2, 5]

    # ── Passive baseline ─────────────────────────────────────────────
    passive_pnl = _passive_baseline_pnl(csv_path, symbols)

    results: List[Tuple[float, int, float]] = []

    col_w = 10
    print(f"\n[Optimiser] AS-MM 2-D grid  symbols={symbols}")
    print(f"  gamma values   : {gammas}")
    print(f"  order_size values: {order_sizes}")
    print(f"  passive baseline : {passive_pnl:.2f}\n")
    print(f"  {'gamma':>6}  {'order_size':>10}  {'PnL':>10}  {'vs passive':>10}")
    print(f"  {'-'*42}")

    for gamma in gammas:
        for order_size in order_sizes:
            bt = Backtester(csv_path, verbose=False)
            t  = Trader()
            # Pre-seed the AS-MM instance with our trial params
            for sym in symbols:
                t.as_mm[sym] = AvellanedaStoikovMarketMaker(
                    gamma=gamma,
                    order_size=order_size,
                )
            bt.trader = t
            bt.run()
            pnl  = _pnl_for_products(bt, symbols)
            delta = pnl - passive_pnl
            sign  = "+" if delta >= 0 else ""
            results.append((gamma, order_size, pnl))
            print(
                f"  gamma={gamma:<5}  order_size={order_size:<2}  "
                f"PnL={pnl:>10.2f}  vs passive={sign}{delta:.2f}"
            )

    best_gamma, best_order_size, best_pnl = max(results, key=lambda x: x[2])
    delta_best = best_pnl - passive_pnl
    sign_best  = "+" if delta_best >= 0 else ""

    print(f"\n  {'─'*42}")
    print(f"  ✓ Best  gamma={best_gamma}  order_size={best_order_size}")
    print(f"    AS-MM PnL    : {best_pnl:.2f}")
    print(f"    Passive PnL  : {passive_pnl:.2f}")
    print(f"    Improvement  : {sign_best}{delta_best:.2f}")

    return {
        "gamma": best_gamma,
        "order_size": best_order_size,
        "pnl": best_pnl,
        "passive_pnl": passive_pnl,
        "improvement": delta_best,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a Trader with a patched config
# ─────────────────────────────────────────────────────────────────────────────

def _make_trader_with_patch(patch_fn):
    """
    Create a fresh Trader instance after applying patch_fn to trader module.
    patch_fn receives the trader module and mutates its config dicts.
    """
    import importlib
    import trader as trader_mod
    patch_fn(trader_mod)
    return trader_mod.Trader()


# ─────────────────────────────────────────────────────────────────────────────
# Full optimisation run
# ─────────────────────────────────────────────────────────────────────────────

def run_optimisation(csv_path: str) -> Dict[str, Dict[str, Any]]:
    import trader as trader_mod

    best_params: Dict[str, Dict[str, Any]] = {}

    # ── FixedFairMM ───────────────────────────────────────────────────
    ff_syms = list(trader_mod.FIXED_FAIR_PRODUCTS.keys())
    if ff_syms:
        best_params["FixedFairMM"] = optimise_fixed_fair(csv_path, ff_syms)

    # ── RollingMidMM ──────────────────────────────────────────────────
    rm_syms = list(trader_mod.ROLLING_MID_PRODUCTS.keys())
    if rm_syms:
        best_params["RollingMidMM"] = optimise_rolling_mid(csv_path, rm_syms)

    # ── SpreadArb ─────────────────────────────────────────────────────
    basket_syms = list(trader_mod.BASKET_CONFIGS.keys())
    if basket_syms:
        best_params["SpreadArb"] = optimise_spread_arb(csv_path, basket_syms)

    # Note: AS-MM symbols aren't enumerable at import time (lazy-created).
    # Run run_as_mm_optimisation(csv_path, ["PRODUCT_E", ...]) separately.

    # ── Print final summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTIMISATION RESULTS")
    print("=" * 60)
    for strategy, params in best_params.items():
        print(f"\n{strategy}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    return best_params


def run_as_mm_optimisation(csv_path: str, symbols: List[str]) -> Dict[str, Any]:
    """Entry point for optimising only the AS-MM parameters for given symbols."""
    print("\n" + "=" * 60)
    print("AS-MM PARAMETER OPTIMISATION")
    print("=" * 60)
    result = optimise_as_mm(csv_path, symbols)
    print("\n" + "=" * 60)
    print("AS-MM OPTIMISATION COMPLETE")
    print("=" * 60)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMC Prosperity Parameter Optimiser")
    parser.add_argument("csv", nargs="?", default="historical_data.csv")
    parser.add_argument(
        "--as-mm-only",
        action="store_true",
        help="Run only the AS-MM gamma/order_size grid search",
    )
    parser.add_argument(
        "--as-mm-symbols",
        nargs="+",
        default=["PRODUCT_E"],
        metavar="SYM",
        help="Symbols to optimise for AS-MM (default: PRODUCT_E)",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
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
        print(f"[Optimiser] ERROR: file not found: {csv_path}")
        sys.exit(1)

    if args.as_mm_only:
        run_as_mm_optimisation(csv_path, args.as_mm_symbols)
    else:
        run_optimisation(csv_path)
        run_as_mm_optimisation(csv_path, args.as_mm_symbols)


if __name__ == "__main__":
    main()
