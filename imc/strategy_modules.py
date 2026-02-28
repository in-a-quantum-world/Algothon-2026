"""
IMC Prosperity Strategy Modules
================================
Five self-contained strategies. Each exposes a single entry-point:

    orders = StrategyClass(params).generate_orders(symbol, state, position, limit)

All strategies also expose `update_state(symbol, state)` so persistent state
(rolling windows, spread history, etc.) can be maintained in Trader.__init__.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Deque

from trader_framework import Order, OrderDepth, TradingState




# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clamp_qty(desired: int, position: int, limit: int) -> int:
    """Return the largest quantity we can actually trade without breaching limit."""
    if desired > 0:
        return min(desired, limit - position)
    elif desired < 0:
        return max(desired, -limit - position)
    return 0


def best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders.keys()) if od.buy_orders else None


def best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders.keys()) if od.sell_orders else None


def rolling_mean(dq: Deque[float]) -> float:
    return sum(dq) / len(dq) if dq else 0.0


def rolling_std(dq: Deque[float]) -> float:
    if len(dq) < 2:
        return 1.0
    m = rolling_mean(dq)
    return math.sqrt(sum((x - m) ** 2 for x in dq) / (len(dq) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – FixedFairMarketMaker
# ─────────────────────────────────────────────────────────────────────────────

class FixedFairMarketMaker:
    """
    For products with a stable, known fair value (e.g. Amethysts @ 10000).

    Parameters
    ----------
    fair_value : int
        The known fair price of the product.
    spread : int
        Half-spread around fair value for our passive quotes.
    clear_spread : int
        Extra discount/premium applied when position-clearing.
    """

    def __init__(self, fair_value: int, spread: int = 2, clear_spread: int = 1):
        self.fair_value = fair_value
        self.spread = spread
        self.clear_spread = clear_spread

    # ------------------------------------------------------------------
    def generate_orders(
        self,
        symbol: str,
        state: TradingState,
        position: int,
        limit: int,
    ) -> List[Order]:
        od = state.order_depths.get(symbol)
        if od is None:
            return []

        orders: List[Order] = []
        fv = self.fair_value

        # ── 1. Take any mispriced resting orders ──────────────────────
        # Buy everything offered below fair value
        for ask_px, ask_vol in sorted(od.sell_orders.items()):
            if ask_px < fv:
                qty = clamp_qty(-ask_vol, position, limit)  # ask_vol is negative
                if qty > 0:
                    print(f"[FixedFairMM] {symbol}: TAKE ask {ask_px} qty={qty}")
                    orders.append(Order(symbol, ask_px, qty))
                    position += qty

        # Sell everything bid above fair value
        for bid_px, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if bid_px > fv:
                qty = clamp_qty(-bid_vol, position, limit)  # sell = negative
                if qty < 0:
                    print(f"[FixedFairMM] {symbol}: TAKE bid {bid_px} qty={qty}")
                    orders.append(Order(symbol, bid_px, qty))
                    position += qty

        # ── 2. Position clearing (zero-EV trades to reduce risk) ──────
        if position > 0:
            # We're long: place a sell at fair value (zero EV but reduces risk)
            clear_qty = clamp_qty(-position, position, limit)
            if clear_qty < 0:
                clear_px = fv - self.clear_spread
                print(f"[FixedFairMM] {symbol}: CLEAR long, sell {abs(clear_qty)}@{clear_px}")
                orders.append(Order(symbol, clear_px, clear_qty))
                position += clear_qty
        elif position < 0:
            clear_qty = clamp_qty(-position, position, limit)
            if clear_qty > 0:
                clear_px = fv + self.clear_spread
                print(f"[FixedFairMM] {symbol}: CLEAR short, buy {clear_qty}@{clear_px}")
                orders.append(Order(symbol, clear_px, clear_qty))
                position += clear_qty

        # ── 3. Passive market-making quotes ───────────────────────────
        buy_qty = clamp_qty(limit, position, limit)
        sell_qty = clamp_qty(-limit, position, limit)

        if buy_qty > 0:
            bid_px = fv - self.spread
            print(f"[FixedFairMM] {symbol}: QUOTE bid {bid_px} qty={buy_qty}")
            orders.append(Order(symbol, bid_px, buy_qty))

        if sell_qty < 0:
            ask_px = fv + self.spread
            print(f"[FixedFairMM] {symbol}: QUOTE ask {ask_px} qty={sell_qty}")
            orders.append(Order(symbol, ask_px, sell_qty))

        return orders


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 – RollingMidMarketMaker
# ─────────────────────────────────────────────────────────────────────────────

class RollingMidMarketMaker:
    """
    For slowly-trending products (e.g. Starfruit / Bananas).

    Key insight from top IMC teams: filter the order book for the
    market-maker bot (consistently large size ~20-30 lots on both sides)
    and use *its* mid price rather than the noisy NBBO mid.

    Parameters
    ----------
    window : int
        Number of ticks in the rolling fair-value window.
    spread : int
        Half-spread for passive quotes.
    mm_min_size : int
        Minimum volume that qualifies a level as "market-maker size".
    """

    def __init__(self, window: int = 20, spread: int = 2, mm_min_size: int = 15):
        self.window = window
        self.spread = spread
        self.mm_min_size = mm_min_size
        self.mid_history: Deque[float] = deque(maxlen=window)
        self.fair_value: Optional[float] = None

    # ------------------------------------------------------------------
    def _extract_mm_mid(self, od: OrderDepth) -> Optional[float]:
        """
        Find the 'market-maker bot' mid by looking for thick levels.
        Falls back to NBBO mid if nothing qualifies.
        """
        mm_bid, mm_ask = None, None

        for px, vol in od.buy_orders.items():
            if vol >= self.mm_min_size:
                if mm_bid is None or px > mm_bid:
                    mm_bid = px

        for px, vol in od.sell_orders.items():
            if abs(vol) >= self.mm_min_size:
                if mm_ask is None or px < mm_ask:
                    mm_ask = px

        if mm_bid is not None and mm_ask is not None:
            return (mm_bid + mm_ask) / 2.0

        # fallback
        bb = best_bid(od)
        ba = best_ask(od)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    # ------------------------------------------------------------------
    def update_state(self, symbol: str, state: TradingState) -> None:
        od = state.order_depths.get(symbol)
        if od is None:
            return
        mid = self._extract_mm_mid(od)
        if mid is not None:
            self.mid_history.append(mid)
            self.fair_value = rolling_mean(self.mid_history)

    # ------------------------------------------------------------------
    def generate_orders(
        self,
        symbol: str,
        state: TradingState,
        position: int,
        limit: int,
    ) -> List[Order]:
        if self.fair_value is None or len(self.mid_history) < 3:
            return []

        od = state.order_depths.get(symbol)
        if od is None:
            return []

        fv = round(self.fair_value)
        orders: List[Order] = []

        # ── 1. Take mispriced orders ──────────────────────────────────
        for ask_px, ask_vol in sorted(od.sell_orders.items()):
            if ask_px < fv:
                qty = clamp_qty(-ask_vol, position, limit)
                if qty > 0:
                    print(f"[RollingMM] {symbol}: TAKE ask {ask_px} qty={qty} fv={fv:.1f}")
                    orders.append(Order(symbol, ask_px, qty))
                    position += qty

        for bid_px, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if bid_px > fv:
                qty = clamp_qty(-bid_vol, position, limit)
                if qty < 0:
                    print(f"[RollingMM] {symbol}: TAKE bid {bid_px} qty={qty} fv={fv:.1f}")
                    orders.append(Order(symbol, bid_px, qty))
                    position += qty

        # ── 2. Position clearing ──────────────────────────────────────
        if position > limit * 0.5:
            qty = clamp_qty(-int(position * 0.5), position, limit)
            if qty < 0:
                print(f"[RollingMM] {symbol}: CLEAR long {qty}@{fv}")
                orders.append(Order(symbol, fv, qty))
                position += qty
        elif position < -limit * 0.5:
            qty = clamp_qty(-int(position * 0.5), position, limit)
            if qty > 0:
                print(f"[RollingMM] {symbol}: CLEAR short {qty}@{fv}")
                orders.append(Order(symbol, fv, qty))
                position += qty

        # ── 3. Passive quotes ─────────────────────────────────────────
        buy_qty = clamp_qty(limit, position, limit)
        sell_qty = clamp_qty(-limit, position, limit)

        if buy_qty > 0:
            print(f"[RollingMM] {symbol}: QUOTE bid {fv - self.spread} qty={buy_qty}")
            orders.append(Order(symbol, fv - self.spread, buy_qty))
        if sell_qty < 0:
            print(f"[RollingMM] {symbol}: QUOTE ask {fv + self.spread} qty={sell_qty}")
            orders.append(Order(symbol, fv + self.spread, sell_qty))

        return orders


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 – SpreadArbitrage  (ETF basket)
# ─────────────────────────────────────────────────────────────────────────────

class SpreadArbitrage:
    """
    ETF / basket arbitrage.

    basket_price  =  sum(weight_i * component_i_price)  +  premium
    spread        =  basket_price - synthetic_price
    Trade when z-score exceeds threshold.

    Parameters
    ----------
    basket_symbol : str
    components : Dict[str, int]  symbol -> weight in basket
    hardcoded_mean : float       stable long-run spread mean
    std_window : int             rolling window for spread std
    z_threshold : float          entry z-score
    """

    def __init__(
        self,
        basket_symbol: str,
        components: Dict[str, int],
        hardcoded_mean: float = 380.0,
        std_window: int = 50,
        z_threshold: float = 1.5,
    ):
        self.basket_symbol = basket_symbol
        self.components = components
        self.hardcoded_mean = hardcoded_mean
        self.std_window = std_window
        self.z_threshold = z_threshold
        self.spread_history: Deque[float] = deque(maxlen=std_window)

    # ------------------------------------------------------------------
    def _synthetic_price(self, state: TradingState) -> Optional[float]:
        total = 0.0
        for sym, weight in self.components.items():
            od = state.order_depths.get(sym)
            if od is None:
                return None
            mid = od.mid_price()
            if mid is None:
                return None
            total += weight * mid
        return total

    # ------------------------------------------------------------------
    def update_state(self, state: TradingState) -> None:
        basket_od = state.order_depths.get(self.basket_symbol)
        if basket_od is None:
            return
        basket_mid = basket_od.mid_price()
        synthetic = self._synthetic_price(state)
        if basket_mid is not None and synthetic is not None:
            self.spread_history.append(basket_mid - synthetic)

    # ------------------------------------------------------------------
    def generate_orders(
        self,
        state: TradingState,
        positions: Dict[str, int],
        limits: Dict[str, int],
    ) -> Dict[str, List[Order]]:
        """Returns orders keyed by symbol (basket + all components)."""
        result: Dict[str, List[Order]] = {}

        if len(self.spread_history) < 10:
            return result

        basket_od = state.order_depths.get(self.basket_symbol)
        synthetic = self._synthetic_price(state)
        if basket_od is None or synthetic is None:
            return result

        current_spread = (basket_od.mid_price() or 0) - synthetic
        std = rolling_std(self.spread_history)
        if std < 1e-6:
            return result

        z = (current_spread - self.hardcoded_mean) / std
        print(f"[SpreadArb] spread={current_spread:.1f} z={z:.2f} std={std:.2f}")

        basket_limit = limits.get(self.basket_symbol, 60)
        basket_pos = positions.get(self.basket_symbol, 0)

        if z > self.z_threshold:
            # Basket expensive vs synthetic → sell basket, buy components
            sell_qty = clamp_qty(-1, basket_pos, basket_limit)
            if sell_qty < 0:
                px = best_bid(basket_od)
                if px:
                    print(f"[SpreadArb] SELL basket {sell_qty}@{px}")
                    result[self.basket_symbol] = [Order(self.basket_symbol, px, sell_qty)]

            for sym, weight in self.components.items():
                comp_od = state.order_depths.get(sym)
                if not comp_od:
                    continue
                comp_limit = limits.get(sym, 20)
                comp_pos = positions.get(sym, 0)
                buy_qty = clamp_qty(weight, comp_pos, comp_limit)
                if buy_qty > 0:
                    px = best_ask(comp_od)
                    if px:
                        print(f"[SpreadArb] BUY {sym} {buy_qty}@{px}")
                        result[sym] = [Order(sym, px, buy_qty)]

        elif z < -self.z_threshold:
            # Basket cheap vs synthetic → buy basket, sell components
            buy_qty = clamp_qty(1, basket_pos, basket_limit)
            if buy_qty > 0:
                px = best_ask(basket_od)
                if px:
                    print(f"[SpreadArb] BUY basket {buy_qty}@{px}")
                    result[self.basket_symbol] = [Order(self.basket_symbol, px, buy_qty)]

            for sym, weight in self.components.items():
                comp_od = state.order_depths.get(sym)
                if not comp_od:
                    continue
                comp_limit = limits.get(sym, 20)
                comp_pos = positions.get(sym, 0)
                sell_qty = clamp_qty(-weight, comp_pos, comp_limit)
                if sell_qty < 0:
                    px = best_bid(comp_od)
                    if px:
                        print(f"[SpreadArb] SELL {sym} {sell_qty}@{px}")
                        result[sym] = [Order(sym, px, sell_qty)]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4 – CrossMarketArbitrage  (e.g. Orchids)
# ─────────────────────────────────────────────────────────────────────────────

class CrossMarketArbitrage:
    """
    Exploits price differences between the local exchange and an
    external market whose prices appear in state.observations.

    Parameters
    ----------
    obs_bid_key : str    key in observations for external best bid
    obs_ask_key : str    key in observations for external best ask
    transport_cost : float   one-way shipping cost per unit
    import_tariff : float    tariff on imports
    export_tariff : float    tariff on exports
    min_edge : float         minimum profit per lot to trade
    fill_window : int        window to track fill rate (adaptive edge)
    """

    def __init__(
        self,
        obs_bid_key: str = "SOUTH_ORCHID_BID",
        obs_ask_key: str = "SOUTH_ORCHID_ASK",
        transport_cost: float = 1.0,
        import_tariff: float = 0.1,
        export_tariff: float = 0.0,
        min_edge: float = 2.0,
        fill_window: int = 20,
    ):
        self.obs_bid_key = obs_bid_key
        self.obs_ask_key = obs_ask_key
        self.transport_cost = transport_cost
        self.import_tariff = import_tariff
        self.export_tariff = export_tariff
        self.min_edge = min_edge
        self.fill_window = fill_window

        self._sent: int = 0
        self._filled: int = 0
        self._fill_history: Deque[float] = deque(maxlen=fill_window)
        self._edge_adjust: float = 0.0  # adaptive component

    # ------------------------------------------------------------------
    def _fill_rate(self) -> float:
        if not self._fill_history:
            return 1.0
        return rolling_mean(self._fill_history)

    def update_state(self, symbol: str, state: TradingState) -> None:
        """Call each tick to update fill history."""
        own = state.own_trades.get(symbol, [])
        if self._sent > 0:
            filled_this_tick = len(own)
            self._fill_history.append(min(1.0, filled_this_tick / max(self._sent, 1)))
        self._sent = 0

        # Adaptive: if fill rate drops, widen required edge
        fr = self._fill_rate()
        if fr < 0.3:
            self._edge_adjust = 1.0
        elif fr < 0.6:
            self._edge_adjust = 0.5
        else:
            self._edge_adjust = 0.0

    # ------------------------------------------------------------------
    def generate_orders(
        self,
        symbol: str,
        state: TradingState,
        position: int,
        limit: int,
    ) -> List[Order]:
        obs = state.observations
        ext_bid = obs.get(self.obs_bid_key)
        ext_ask = obs.get(self.obs_ask_key)
        if ext_bid is None or ext_ask is None:
            return []

        od = state.order_depths.get(symbol)
        if od is None:
            return []

        required_edge = self.min_edge + self._edge_adjust
        orders: List[Order] = []

        # Import: buy from external market, sell locally
        # Cost to import = ext_ask + transport + tariff
        import_cost = ext_ask + self.transport_cost + self.import_tariff * ext_ask
        local_bid = best_bid(od)
        if local_bid is not None:
            edge = local_bid - import_cost
            if edge >= required_edge:
                qty = clamp_qty(-limit, position, limit)  # sell locally
                if qty < 0:
                    print(f"[CrossArb] {symbol}: IMPORT arb edge={edge:.2f}, SELL local {qty}@{local_bid}")
                    orders.append(Order(symbol, local_bid, qty))
                    self._sent += abs(qty)

        # Export: buy locally, sell to external market
        # Revenue from export = ext_bid - transport - tariff
        export_revenue = ext_bid - self.transport_cost - self.export_tariff * ext_bid
        local_ask = best_ask(od)
        if local_ask is not None:
            edge = export_revenue - local_ask
            if edge >= required_edge:
                qty = clamp_qty(limit, position, limit)
                if qty > 0:
                    print(f"[CrossArb] {symbol}: EXPORT arb edge={edge:.2f}, BUY local {qty}@{local_ask}")
                    orders.append(Order(symbol, local_ask, qty))
                    self._sent += qty

        return orders


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5 – OptionsStrategy  (vanilla calls, e.g. Coconut Coupons)
# ─────────────────────────────────────────────────────────────────────────────

class OptionsStrategy:
    """
    Vanilla European call option trading via implied-vol mean reversion.

    Assumes:
      - Underlying: `underlying_symbol`
      - Option:     `option_symbol`  (coupon / call)
      - Strike K, time-to-expiry T (in days), r=0

    Trades when implied vol deviates from historical mean.
    Uses Black-Scholes delta for hedging the underlying.

    Parameters
    ----------
    underlying_symbol : str
    option_symbol : str
    strike : float
    t_days : float          time to expiry in days
    vol_window : int        rolling window for historical iv
    vol_threshold : float   z-score threshold to enter
    """

    def __init__(
        self,
        underlying_symbol: str,
        option_symbol: str,
        strike: float = 10000.0,
        t_days: float = 250.0,
        vol_window: int = 50,
        vol_threshold: float = 1.5,
    ):
        self.underlying_symbol = underlying_symbol
        self.option_symbol = option_symbol
        self.K = strike
        self.T_days = t_days
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.iv_history: Deque[float] = deque(maxlen=vol_window)
        self.current_delta: float = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        """Black-Scholes call price. T in years."""
        if T <= 0 or S <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N = OptionsStrategy._norm_cdf
        return S * N(d1) - K * math.exp(-r * T) * N(d2)

    @staticmethod
    def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        if T <= 0 or S <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return OptionsStrategy._norm_cdf(d1)

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.0,
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> Optional[float]:
        """Bisection to invert Black-Scholes for implied vol."""
        lo, hi = 1e-4, 5.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            price = OptionsStrategy.bs_call_price(S, K, T, mid, r)
            if abs(price - market_price) < tol:
                return mid
            if price < market_price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    def update_state(self, state: TradingState) -> None:
        und_od = state.order_depths.get(self.underlying_symbol)
        opt_od = state.order_depths.get(self.option_symbol)
        if und_od is None or opt_od is None:
            return

        S = und_od.mid_price()
        opt_mid = opt_od.mid_price()
        if S is None or opt_mid is None:
            return

        T_years = self.T_days / 365.0
        iv = self.implied_vol(opt_mid, S, self.K, T_years)
        if iv is not None and 0.01 <= iv <= 3.0:
            self.iv_history.append(iv)

        # Update delta for hedging
        if self.iv_history:
            self.current_delta = self.bs_delta(S, self.K, T_years, rolling_mean(self.iv_history))

    # ------------------------------------------------------------------
    def generate_orders(
        self,
        state: TradingState,
        positions: Dict[str, int],
        limits: Dict[str, int],
    ) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        if len(self.iv_history) < 10:
            return result

        und_od = state.order_depths.get(self.underlying_symbol)
        opt_od = state.order_depths.get(self.option_symbol)
        if und_od is None or opt_od is None:
            return result

        S = und_od.mid_price()
        if S is None:
            return result

        T_years = self.T_days / 365.0
        mean_iv = rolling_mean(self.iv_history)
        std_iv = rolling_std(self.iv_history)
        if std_iv < 1e-6:
            return result

        opt_mid = opt_od.mid_price()
        if opt_mid is None:
            return result

        current_iv = self.implied_vol(opt_mid, S, self.K, T_years)
        if current_iv is None:
            return result

        z = (current_iv - mean_iv) / std_iv
        print(f"[Options] IV={current_iv:.4f} mean={mean_iv:.4f} z={z:.2f} delta={self.current_delta:.3f}")

        opt_limit = limits.get(self.option_symbol, 600)
        opt_pos = positions.get(self.option_symbol, 0)

        if z > self.vol_threshold:
            # IV too high → sell options (collect premium, expect vol to fall)
            qty = clamp_qty(-5, opt_pos, opt_limit)
            if qty < 0:
                px = best_bid(opt_od)
                if px:
                    print(f"[Options] SELL option {qty}@{px} (high IV)")
                    result[self.option_symbol] = [Order(self.option_symbol, px, qty)]

        elif z < -self.vol_threshold:
            # IV too low → buy options (buy cheap gamma)
            qty = clamp_qty(5, opt_pos, opt_limit)
            if qty > 0:
                px = best_ask(opt_od)
                if px:
                    print(f"[Options] BUY option {qty}@{px} (low IV)")
                    result[self.option_symbol] = [Order(self.option_symbol, px, qty)]

        # Delta hedge: target zero net delta
        und_limit = limits.get(self.underlying_symbol, 300)
        und_pos = positions.get(self.underlying_symbol, 0)
        target_hedge = -round(opt_pos * self.current_delta)
        hedge_trade = target_hedge - und_pos
        hedge_qty = clamp_qty(hedge_trade, und_pos, und_limit)

        if abs(hedge_qty) >= 1:
            px = best_ask(und_od) if hedge_qty > 0 else best_bid(und_od)
            if px:
                print(f"[Options] DELTA HEDGE {self.underlying_symbol} {hedge_qty}@{px}")
                result[self.underlying_symbol] = [Order(self.underlying_symbol, px, hedge_qty)]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 6 – AvellanedaStoikovMarketMaker
# ─────────────────────────────────────────────────────────────────────────────

class AvellanedaStoikovMarketMaker:
    """
    Optimal market-making based on Avellaneda & Stoikov (2008).

    Core idea: the market maker has a risk-aversion parameter gamma that
    causes her to shade her reservation price away from her inventory, and
    an order-arrival intensity k that controls how wide the spread needs to
    be to attract counterparties.

    Formulas
    --------
    sigma        = rolling std of mid-price differences  (floored at 0.01)
    t            = min(tick_count / 500, 0.999)          (normalised time in [0,1))
    time_left    = T - t

    reservation_price = mid - q * gamma * sigma^2 * time_left
    optimal_spread    = gamma * sigma^2 * time_left  +  (2/gamma) * ln(1 + gamma/k)
    bid = reservation_price - spread/2     (rounded to int)
    ask = reservation_price + spread/2     (rounded to int)

    Parameters
    ----------
    gamma        : float  risk-aversion / inventory penalty  (default 0.1)
    k            : float  order-arrival rate per unit spread (default 1.5)
    T            : float  total time horizon, normalised     (default 1.0)
    sigma_window : int    rolling window for vol estimation  (default 20)
    order_size   : int    lots per quote                     (default 5)
    max_spread   : float  hard cap on spread to avoid bad
                          startup-period quotes              (default 10.0)
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        T: float = 1.0,
        sigma_window: int = 20,
        order_size: int = 5,
        max_spread: float = 10.0,
    ):
        self.gamma = gamma
        self.k = k
        self.T = T
        self.sigma_window = sigma_window
        self.order_size = order_size
        self.max_spread = max_spread

        # sigma_window + 1 so we always have enough diffs
        self.mid_history: Deque[float] = deque(maxlen=sigma_window + 1)
        self.tick_count: int = 0

    # ------------------------------------------------------------------
    def _sigma(self) -> float:
        """Rolling std of mid-price first-differences, floored at 0.01."""
        if len(self.mid_history) < 2:
            return 0.01
        diffs = [
            self.mid_history[i] - self.mid_history[i - 1]
            for i in range(1, len(self.mid_history))
        ]
        if len(diffs) < 2:
            return 0.01
        mean_d = sum(diffs) / len(diffs)
        var = sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1)
        return max(math.sqrt(var), 0.01)

    # ------------------------------------------------------------------
    def update_state(self, symbol: str, state: TradingState) -> None:
        """Record current mid price and advance tick counter."""
        od = state.order_depths.get(symbol)
        if od is None:
            return
        mid = od.mid_price()
        if mid is not None:
            self.mid_history.append(mid)
        self.tick_count += 1

    # ------------------------------------------------------------------
    def generate_orders(
        self,
        symbol: str,
        state: TradingState,
        position: int,
        limit: int,
    ) -> List[Order]:
        if len(self.mid_history) < 2:
            return []

        od = state.order_depths.get(symbol)
        if od is None:
            return []
        mid = od.mid_price()
        if mid is None:
            return []

        sigma = self._sigma()
        t = min(self.tick_count / 500.0, 0.999)
        time_left = self.T - t

        # ── A-S reservation price ─────────────────────────────────────
        # Positive inventory (long) → shade bid downward to attract sellers
        reservation_price = mid - position * self.gamma * sigma ** 2 * time_left

        # ── A-S optimal spread ────────────────────────────────────────
        spread = (
            self.gamma * sigma ** 2 * time_left
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )
        spread = min(spread, self.max_spread)

        bid_px = int(reservation_price - spread / 2.0)
        ask_px = int(reservation_price + spread / 2.0)

        # Guarantee at least 1 tick between bid and ask
        if bid_px >= ask_px:
            ask_px = bid_px + 1

        orders: List[Order] = []
        buy_qty  = clamp_qty( self.order_size, position, limit)
        sell_qty = clamp_qty(-self.order_size, position, limit)

        if buy_qty > 0:
            print(
                f"[AS-MM] {symbol}: BID {bid_px} qty={buy_qty:+d} | "
                f"rp={reservation_price:.1f}  spread={spread:.2f}  "
                f"σ={sigma:.3f}  t={t:.3f}  q={position}"
            )
            orders.append(Order(symbol, bid_px, buy_qty))

        if sell_qty < 0:
            print(
                f"[AS-MM] {symbol}: ASK {ask_px} qty={sell_qty:+d} | "
                f"rp={reservation_price:.1f}  spread={spread:.2f}  "
                f"σ={sigma:.3f}  t={t:.3f}  q={position}"
            )
            orders.append(Order(symbol, ask_px, sell_qty))

        return orders
