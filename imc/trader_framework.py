"""
IMC Prosperity Trading Framework - Core Data Types
===================================================
Matches IMC's exact API. Do NOT modify these classes —
the submission engine expects this exact interface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class Order:
    """A single order placed by our trader."""
    symbol: str
    price: int
    quantity: int  # positive = buy, negative = sell

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({self.symbol} {side} {abs(self.quantity)}@{self.price})"


@dataclass
class Trade:
    """A completed trade (ours or market participants')."""
    symbol: str
    price: int
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class OrderDepth:
    """
    Live order book for a single product.
    buy_orders:  price -> volume (volume is positive)
    sell_orders: price -> volume (volume is negative in IMC convention)
    """
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)

    def best_bid(self) -> Optional[int]:
        return max(self.buy_orders.keys()) if self.buy_orders else None

    def best_ask(self) -> Optional[int]:
        return min(self.sell_orders.keys()) if self.sell_orders else None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    def spread(self) -> Optional[int]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None:
            return ba - bb
        return None


@dataclass
class Listing:
    """Product listing metadata."""
    symbol: str
    product: str
    denomination: str


@dataclass
class TradingState:
    """
    The full state snapshot delivered to Trader.run() each timestamp.
    This exactly mirrors IMC's TradingState object.
    """
    timestamp: int
    listings: Dict[str, Listing]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]       # trades we executed last tick
    market_trades: Dict[str, List[Trade]]    # trades between other bots last tick
    position: Dict[str, int]                 # our current net position per product
    observations: Dict[str, Any]             # extra market data (e.g. conversion rates)

    def get_position(self, symbol: str) -> int:
        return self.position.get(symbol, 0)

    def get_order_depth(self, symbol: str) -> Optional[OrderDepth]:
        return self.order_depths.get(symbol)


# ── Position limits by product (update when competition reveals products) ──────
POSITION_LIMITS: Dict[str, int] = {
    "PRODUCT_A": 20,
    "PRODUCT_B": 20,
    "PRODUCT_C": 60,
    "PRODUCT_D": 600,
    # Basket legs
    "X": 250,
    "Y": 350,
    "Z": 60,
    # Add real IMC products here as they are revealed
    "AMETHYSTS":  20,
    "STARFRUIT":  20,
    "ORCHIDS":    100,
    "CHOCOLATE":  250,
    "STRAWBERRIES": 350,
    "ROSES":      60,
    "GIFT_BASKET": 60,
    "COCONUT":    300,
    "COCONUT_COUPON": 600,
}

DEFAULT_POSITION_LIMIT = 20
