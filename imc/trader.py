"""
IMC Prosperity – Main Trader (SUBMISSION FILE)
===============================================
Upload *this* file to the IMC platform.

The engine calls:  Trader().run(state: TradingState) -> Dict[str, List[Order]]

All persistent state lives in self.*  Trader.run() must be stateless across
calls except for self.* state.
"""

from __future__ import annotations
import sys
import os

# Allow running directly from imc/ directory or from project root
sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, List

from trader_framework import (
    Order, TradingState, POSITION_LIMITS, DEFAULT_POSITION_LIMIT
)
from strategy_modules import (
    FixedFairMarketMaker,
    RollingMidMarketMaker,
    SpreadArbitrage,
    CrossMarketArbitrage,
    OptionsStrategy,
    AvellanedaStoikovMarketMaker,
)


# ─────────────────────────────────────────────────────────────────────────────
# Product routing configuration
# Edit this to add / reconfigure products as the competition progresses.
# ─────────────────────────────────────────────────────────────────────────────

FIXED_FAIR_PRODUCTS: Dict[str, dict] = {
    # symbol -> FixedFairMarketMaker kwargs
    "AMETHYSTS":  {"fair_value": 10000, "spread": 2, "clear_spread": 1},
    "PRODUCT_A":  {"fair_value": 10000, "spread": 2, "clear_spread": 1},
}

ROLLING_MID_PRODUCTS: Dict[str, dict] = {
    # symbol -> RollingMidMarketMaker kwargs
    "STARFRUIT": {"window": 20, "spread": 2, "mm_min_size": 15},
    "PRODUCT_B": {"window": 20, "spread": 2, "mm_min_size": 10},
}

BASKET_CONFIGS: Dict[str, dict] = {
    # basket_symbol -> SpreadArbitrage kwargs
    "GIFT_BASKET": {
        "components": {"CHOCOLATE": 4, "STRAWBERRIES": 6, "ROSES": 1},
        "hardcoded_mean": 380.0,
        "std_window": 50,
        "z_threshold": 1.5,
    },
    "PRODUCT_C": {
        "components": {"X": 4, "Y": 6, "Z": 1},
        "hardcoded_mean": 380.0,
        "std_window": 50,
        "z_threshold": 1.5,
    },
}

CROSS_MARKET_PRODUCTS: Dict[str, dict] = {
    # symbol -> CrossMarketArbitrage kwargs
    "ORCHIDS": {
        "obs_bid_key": "SOUTH_ORCHID_BID",
        "obs_ask_key": "SOUTH_ORCHID_ASK",
        "transport_cost": 1.0,
        "import_tariff": 0.1,
        "export_tariff": 0.0,
        "min_edge": 2.0,
    },
}

OPTIONS_CONFIGS: Dict[str, dict] = {
    # option_symbol -> OptionsStrategy kwargs (must include underlying_symbol)
    "COCONUT_COUPON": {
        "underlying_symbol": "COCONUT",
        "option_symbol": "COCONUT_COUPON",
        "strike": 10000.0,
        "t_days": 250.0,
        "vol_window": 50,
        "vol_threshold": 1.5,
    },
    "PRODUCT_D_OPTION": {
        "underlying_symbol": "PRODUCT_D",
        "option_symbol": "PRODUCT_D_OPTION",
        "strike": 10000.0,
        "t_days": 250.0,
        "vol_window": 50,
        "vol_threshold": 1.5,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Trader class
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    def __init__(self):
        # ── Fixed-fair market makers ──────────────────────────────────
        self.fixed_fair: Dict[str, FixedFairMarketMaker] = {
            sym: FixedFairMarketMaker(**kwargs)
            for sym, kwargs in FIXED_FAIR_PRODUCTS.items()
        }

        # ── Rolling-mid market makers ─────────────────────────────────
        self.rolling_mid: Dict[str, RollingMidMarketMaker] = {
            sym: RollingMidMarketMaker(**kwargs)
            for sym, kwargs in ROLLING_MID_PRODUCTS.items()
        }

        # ── Basket / spread arbitrage ─────────────────────────────────
        self.baskets: Dict[str, SpreadArbitrage] = {
            sym: SpreadArbitrage(basket_symbol=sym, **{k: v for k, v in kwargs.items()})
            for sym, kwargs in BASKET_CONFIGS.items()
        }

        # ── Cross-market arbitrage ────────────────────────────────────
        self.cross_market: Dict[str, CrossMarketArbitrage] = {
            sym: CrossMarketArbitrage(**kwargs)
            for sym, kwargs in CROSS_MARKET_PRODUCTS.items()
        }

        # ── Options ───────────────────────────────────────────────────
        self.options: Dict[str, OptionsStrategy] = {
            sym: OptionsStrategy(**kwargs)
            for sym, kwargs in OPTIONS_CONFIGS.items()
        }

        # Track which products are handled by each strategy
        self._basket_component_products: set = set()
        for cfg in BASKET_CONFIGS.values():
            self._basket_component_products.update(cfg["components"].keys())

        self._options_underlying: Dict[str, str] = {
            cfg["option_symbol"]: cfg["underlying_symbol"]
            for cfg in OPTIONS_CONFIGS.values()
        }

        # ── Avellaneda-Stoikov fallback (lazy: created on first encounter) ──
        self.as_mm: Dict[str, AvellanedaStoikovMarketMaker] = {}

    # ------------------------------------------------------------------
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        print(f"\n{'='*60}")
        print(f"[Trader] Timestamp={state.timestamp}  Positions={dict(state.position)}")

        # ── 1. Update all rolling-state strategies ────────────────────
        for sym, strat in self.rolling_mid.items():
            strat.update_state(sym, state)

        for sym, strat in self.baskets.items():
            strat.update_state(state)

        for sym, strat in self.cross_market.items():
            strat.update_state(sym, state)

        for sym, strat in self.options.items():
            strat.update_state(state)

        # ── Update AS-MM state for every product not claimed by another strategy ─
        for sym in state.order_depths:
            if self._is_as_mm_product(sym):
                if sym not in self.as_mm:
                    self.as_mm[sym] = AvellanedaStoikovMarketMaker()
                self.as_mm[sym].update_state(sym, state)

        # ── 2. Generate orders per product ────────────────────────────
        for product in state.order_depths:
            orders = self.trade_product(product, state, result)
            if orders:
                # Merge — a product might already have orders from basket logic
                if product in result:
                    result[product].extend(orders)
                else:
                    result[product] = orders

        return result

    # ------------------------------------------------------------------
    def trade_product(
        self,
        symbol: str,
        state: TradingState,
        partial_result: Dict[str, List[Order]],
    ) -> List[Order]:
        """
        Route a single product to the correct strategy.
        Returns orders for `symbol` only (basket logic is handled separately).
        """
        position = state.get_position(symbol)
        limit = POSITION_LIMITS.get(symbol, DEFAULT_POSITION_LIMIT)

        # ── Fixed-fair market maker ───────────────────────────────────
        if symbol in self.fixed_fair:
            return self.fixed_fair[symbol].generate_orders(symbol, state, position, limit)

        # ── Rolling-mid market maker ──────────────────────────────────
        if symbol in self.rolling_mid:
            return self.rolling_mid[symbol].generate_orders(symbol, state, position, limit)

        # ── Basket arbitrage (one call generates orders for all legs) ─
        if symbol in self.baskets and symbol not in partial_result:
            positions = {s: state.get_position(s) for s in state.order_depths}
            limits = {s: POSITION_LIMITS.get(s, DEFAULT_POSITION_LIMIT)
                      for s in state.order_depths}
            basket_orders = self.baskets[symbol].generate_orders(state, positions, limits)
            # Push component orders into partial_result directly
            for sym, orders in basket_orders.items():
                if sym != symbol:
                    partial_result[sym] = orders
            return basket_orders.get(symbol, [])

        # ── Skip basket component products (handled above) ────────────
        if symbol in self._basket_component_products:
            return []

        # ── Cross-market arbitrage ────────────────────────────────────
        if symbol in self.cross_market:
            return self.cross_market[symbol].generate_orders(symbol, state, position, limit)

        # ── Options (one call generates orders for option + underlying) ─
        if symbol in self.options and symbol not in partial_result:
            positions = {s: state.get_position(s) for s in state.order_depths}
            limits = {s: POSITION_LIMITS.get(s, DEFAULT_POSITION_LIMIT)
                      for s in state.order_depths}
            opt_orders = self.options[symbol].generate_orders(state, positions, limits)
            for sym, orders in opt_orders.items():
                if sym != symbol:
                    partial_result[sym] = orders
            return opt_orders.get(symbol, [])

        # ── Skip underlying if already handled by options strategy ────
        if symbol in self._options_underlying.values():
            return []

        # ── Fallback: Avellaneda-Stoikov market maker ─────────────────
        # as_mm[symbol] is guaranteed to exist (created in run() above)
        return self.as_mm[symbol].generate_orders(symbol, state, position, limit)

    # ------------------------------------------------------------------
    def _is_as_mm_product(self, symbol: str) -> bool:
        """True for products not claimed by any specialised strategy."""
        return (
            symbol not in self.fixed_fair
            and symbol not in self.rolling_mid
            and symbol not in self.baskets
            and symbol not in self._basket_component_products
            and symbol not in self.cross_market
            and symbol not in self.options
            and symbol not in self._options_underlying.values()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from trader_framework import OrderDepth, Listing

    depth = OrderDepth(
        buy_orders={9998: 10, 9997: 5},
        sell_orders={10002: -10, 10003: -5},
    )
    state = TradingState(
        timestamp=0,
        listings={"PRODUCT_A": Listing("PRODUCT_A", "PRODUCT_A", "SEASHELLS")},
        order_depths={"PRODUCT_A": depth},
        own_trades={},
        market_trades={},
        position={"PRODUCT_A": 0},
        observations={},
    )
    trader = Trader()
    orders = trader.run(state)
    print("\nOrders:", orders)
