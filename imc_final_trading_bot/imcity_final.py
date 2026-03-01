

import json
import os
import time
import threading
import math
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import cached_property
from threading import Thread, Lock
from traceback import format_exc
from typing import Any, Callable, Literal

import requests
import sseclient

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these if not using env vars
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "cmi_url":   os.getenv("CMI_URL",    "http://localhost:8080"),
    "username":  os.getenv("CMI_USER",   "your_username"),
    "password":  os.getenv("CMI_PASS",   "your_password"),
    "gemini_key":os.getenv("GEMINI_KEY", "your_gemini_key"),
    # Strategy toggles
    "use_gemini":        True,   # set False to run pure rule-based
    "use_market_making": True,
    "use_arbitrage":     True,
    "use_alpha_signals": True,
    # Risk params
    "mm_spread_ticks":   2,      # half-spread in ticks for market making
    "mm_size":           3,      # lots per MM quote
    "max_position":      90,     # stay well inside ±100 limit
    "arb_threshold":     5,      # min misprice to fire arb order (in price units)
    "gemini_interval":  30,      # seconds between Gemini agent calls
    "data_fetch_interval": 10,   # seconds between external data fetch
}

STANDARD_HEADERS = {"Content-Type": "application/json; charset=utf-8"}

# ─────────────────────────────────────────────────────────────────────────────
# BASE BOT FRAMEWORK  (from IMC template – unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class DictLikeFrozenDataclassMapping(Mapping):
    def __getitem__(self, key):       return getattr(self, key)
    def __iter__(self):               return iter(self.__annotations__)
    def __len__(self):                return len(self.__annotations__)
    def to_dict(self):                return asdict(self)
    def keys(self):                   return self.__annotations__.keys()
    def values(self):                 return [getattr(self, k) for k in self.keys()]
    def items(self):                  return [(k, getattr(self, k)) for k in self.keys()]


@dataclass(frozen=True)
class Product(DictLikeFrozenDataclassMapping):
    symbol: str
    tickSize: float
    startingPrice: int
    contractSize: int


@dataclass(frozen=True)
class Trade(DictLikeFrozenDataclassMapping):
    timestamp: str
    product: str
    buyer: str
    seller: str
    volume: int
    price: float


@dataclass(frozen=True)
class Order(DictLikeFrozenDataclassMapping):
    price: float
    volume: int
    own_volume: int


@dataclass(frozen=True)
class OrderBook(DictLikeFrozenDataclassMapping):
    product: str
    tick_size: float
    buy_orders: list[Order]
    sell_orders: list[Order]


class Side(StrEnum):
    BUY  = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderRequest:
    product: str
    price: float
    side: Side
    volume: int


@dataclass(frozen=True)
class OrderResponse:
    id: str
    status: Literal["ACTIVE", "PART_FILLED"]
    product: str
    side: Side
    price: float
    volume: int
    filled: int
    user: str
    timestamp: str
    targetUser: str | None = None
    message:    str | None = None


class _SSEThread(Thread):
    def __init__(self, bearer, url, handle_orderbook, handle_trade_event):
        super().__init__(daemon=True)
        self._bearer             = bearer
        self._url                = url
        self._handle_orderbook   = handle_orderbook
        self._handle_trade_event = handle_trade_event
        self._http_stream        = None
        self._client             = None
        self._closed             = False

    def run(self):
        while not self._closed:
            try:
                self._consume()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                pass
            except Exception:
                if not self._closed:
                    print("SSE error, reconnecting…")
                    print(format_exc())

    def close(self):
        self._closed = True
        if self._http_stream: self._http_stream.close()
        if self._client:      self._client.close()

    def _consume(self):
        headers = {"Authorization": self._bearer, "Accept": "text/event-stream; charset=utf-8"}
        self._http_stream = requests.get(self._url, stream=True, headers=headers, timeout=30)
        self._client = sseclient.SSEClient(self._http_stream)
        for event in self._client.events():
            if event.event == "order":
                self._on_order_event(json.loads(event.data))
            elif event.event == "trade":
                data   = json.loads(event.data)
                trades = data if isinstance(data, list) else [data]
                fields = {f.name for f in Trade.__dataclass_fields__.values()}
                for t in trades:
                    self._handle_trade_event(Trade(**{k: v for k, v in t.items() if k in fields}))

    def _on_order_event(self, data):
        buy_orders = sorted(
            [Order(price=float(p), volume=v["marketVolume"], own_volume=v["userVolume"])
             for p, v in data["buyOrders"].items()],
            key=lambda o: -o.price)
        sell_orders = sorted(
            [Order(price=float(p), volume=v["marketVolume"], own_volume=v["userVolume"])
             for p, v in data["sellOrders"].items()],
            key=lambda o: o.price)
        self._handle_orderbook(OrderBook(data["productsymbol"], data["tickSize"], buy_orders, sell_orders))


class BaseBot(ABC):
    def __init__(self, cmi_url, username, password):
        self._cmi_url        = cmi_url.rstrip("/")
        self.username        = username
        self._password       = password
        self._sse_thread     = None
        self.trades: list[Trade] = []
        self._trade_watermark    = None
        self._last_trade_fetch   = None

    @cached_property
    def auth_token(self):
        r = requests.post(f"{self._cmi_url}/api/user/authenticate",
                          headers=STANDARD_HEADERS,
                          json={"username": self.username, "password": self._password})
        r.raise_for_status()
        return r.headers["Authorization"]

    def start(self):
        if self._sse_thread: raise RuntimeError("Already running.")
        self._sse_thread = _SSEThread(
            bearer=self.auth_token,
            url=f"{self._cmi_url}/api/market/stream",
            handle_orderbook=self.on_orderbook,
            handle_trade_event=self.on_trades)
        self._sse_thread.start()

    def stop(self):
        if self._sse_thread:
            self._sse_thread.close()
            self._sse_thread.join(timeout=5)
            self._sse_thread = None

    @abstractmethod
    def on_orderbook(self, orderbook: OrderBook): ...
    @abstractmethod
    def on_trades(self, trade: Trade): ...

    def get_market_trades(self):
        params = {}
        if self._trade_watermark: params["from"] = self._trade_watermark
        r = requests.get(f"{self._cmi_url}/api/trade", params=params, headers=self._auth_headers())
        self._last_trade_fetch = time.monotonic()
        if not r.ok:
            print(f"Failed to fetch trades: {r.status_code}")
            return self.trades
        for raw in r.json():
            t = Trade(**raw)
            if self._trade_watermark is None or t.timestamp > self._trade_watermark:
                self.trades.append(t)
                self._trade_watermark = t.timestamp
        return self.trades

    def send_order(self, order: OrderRequest) -> OrderResponse | None:
        r = requests.post(f"{self._cmi_url}/api/order", json=asdict(order), headers=self._auth_headers())
        if r.ok: return OrderResponse(**r.json())
        print(f"Order failed: {r.text}")
        return None

    def send_orders(self, orders: list[OrderRequest]) -> list[OrderResponse]:
        results = []
        def _send(o):
            res = self.send_order(o)
            if res: results.append(res)
        threads = [Thread(target=_send, args=(o,)) for o in orders]
        for t in threads: t.start()
        for t in threads: t.join()
        return results

    def cancel_order(self, order_id):
        requests.delete(f"{self._cmi_url}/api/order/{order_id}", headers=self._auth_headers())

    def cancel_all_orders(self):
        orders  = self.get_orders()
        threads = [Thread(target=self.cancel_order, args=(o["id"],)) for o in orders]
        for t in threads: t.start()
        for t in threads: t.join()

    def get_orders(self, product=None):
        params = {"productsymbol": product} if product else {}
        r = requests.get(f"{self._cmi_url}/api/order/current-user", params=params, headers=self._auth_headers())
        return r.json() if r.ok else []

    def get_products(self):
        r = requests.get(f"{self._cmi_url}/api/product", headers=self._auth_headers())
        r.raise_for_status()
        return [Product(**p) for p in r.json()]

    def get_positions(self):
        r = requests.get(f"{self._cmi_url}/api/position/current-user", headers=self._auth_headers())
        return {p["product"]: p["netPosition"] for p in r.json()} if r.ok else {}

    def get_orderbook(self, product):
        r = requests.get(f"{self._cmi_url}/api/product/{product}/order-book/current-user",
                         headers=self._auth_headers())
        r.raise_for_status()
        data = r.json()
        buy_orders  = sorted([Order(price=e["price"], volume=e["volume"], own_volume=e["userOrderVolume"])
                               for e in data.get("buy", [])],  key=lambda o: -o.price)
        sell_orders = sorted([Order(price=e["price"], volume=e["volume"], own_volume=e["userOrderVolume"])
                               for e in data.get("sell", [])], key=lambda o:  o.price)
        return OrderBook(data["product"], data["tickSize"], buy_orders, sell_orders)

    def get_pnl(self):
        r = requests.get(f"{self._cmi_url}/api/profit/current-user", headers=self._auth_headers())
        return r.json() if r.ok else {}

    def _auth_headers(self):
        return {**STANDARD_HEADERS, "Authorization": self.auth_token}


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_thames_level() -> float | None:
    """Fetch latest Thames tidal level from Environment Agency API."""
    try:
        url = ("https://environment.data.gov.uk/flood-monitoring/id/measures/"
               "0006-level-tidal_level-i-15_min-mAOD/readings?_sorted&_limit=2")
        r = requests.get(url, timeout=8)
        items = r.json().get("items", [])
        if items:
            val = items[0].get("value")
            return float(val) if val is not None else None
    except Exception as e:
        print(f"[Thames] fetch error: {e}")
    return None


def fetch_weather() -> dict | None:
    """Fetch current temperature (F) and humidity for London from open-meteo."""
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               "?latitude=51.5074&longitude=-0.1278"
               "&current=temperature_2m,relative_humidity_2m"
               "&temperature_unit=fahrenheit&forecast_days=1")
        r = requests.get(url, timeout=8)
        cur = r.json().get("current", {})
        temp = cur.get("temperature_2m")
        hum  = cur.get("relative_humidity_2m")
        if temp is not None and hum is not None:
            return {"temp_f": round(float(temp)), "humidity": float(hum)}
    except Exception as e:
        print(f"[Weather] fetch error: {e}")
    return None


def fetch_lhr_flights() -> dict | None:
    """
    Fetch LHR flight counts.  Uses public AviationStack free tier (no key needed for basic).
    Falls back to a simple estimate if unavailable.
    """
    try:
        # Public endpoint – no key required for a sample
        url = "https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR"
        # NOTE: RapidAPI requires a key.  If you have one set RAPIDAPI_KEY env var.
        key = os.getenv("RAPIDAPI_KEY", "")
        if not key:
            raise ValueError("No RapidAPI key")
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"}
        now   = time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
        later = time.strftime("%Y-%m-%dT%H:%M", time.gmtime(time.time() + 3600))
        r = requests.get(url, headers=headers, params={"withLeg": "true", "direction": "Both",
                                                        "withCancelled": "false",
                                                        "from": now, "to": later}, timeout=8)
        data = r.json()
        arr  = len(data.get("arrivals",   {}).get("items", []))
        dep  = len(data.get("departures", {}).get("items", []))
        return {"arrivals": arr, "departures": dep}
    except Exception as e:
        print(f"[LHR] fetch error (using estimate): {e}")
        # Heathrow averages ~1300 movements/day → ~27/30min
        return {"arrivals": 27, "departures": 27}


# ─────────────────────────────────────────────────────────────────────────────
# FAIR VALUE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class FairValueEstimator:
    """Estimates settlement prices from real-world signals."""

    def __init__(self):
        self.lock          = Lock()
        self.thames_level  = None   # mAOD
        self.weather       = None   # {"temp_f": X, "humidity": Y}
        self.lhr           = None   # {"arrivals": A, "departures": D}
        # Accumulators for cumulative products
        self.wx_sum_acc    = 0.0    # cumulative (T*H)/100
        self.tide_strangle = 0.0    # cumulative strangle value *100
        self.lhr_index_acc = 0.0    # cumulative airport metric
        self._prev_tide    = None
        self._update_count = 0

    def update(self, thames, weather, lhr):
        with self.lock:
            self._update_count += 1
            if thames is not None:
                if self._prev_tide is not None:
                    diff = abs(thames - self._prev_tide)
                    # Strangle with strikes 0.2 and 0.25
                    strangle = max(0, diff - 0.25) + max(0, 0.2 - diff)
                    self.tide_strangle += strangle * 100
                self._prev_tide   = thames
                self.thames_level = thames

            if weather is not None:
                self.weather = weather
                th = weather["temp_f"] * weather["humidity"] / 100.0
                self.wx_sum_acc += th

            if lhr is not None:
                self.lhr = lhr
                total = lhr["arrivals"] + lhr["departures"]
                if total > 0:
                    metric = 100 * (lhr["arrivals"] - lhr["departures"]) / total
                    self.lhr_index_acc += metric

    def fair_values(self) -> dict[str, float | None]:
        with self.lock:
            fv = {}

            # Market 1 – TIDE_SPOT: ABS(water_level) * 1000
            if self.thames_level is not None:
                fv["TIDE_SPOT"] = abs(self.thames_level) * 1000
            else:
                fv["TIDE_SPOT"] = None

            # Market 2 – TIDE_SWING: cumulative strangle
            fv["TIDE_SWING"] = self.tide_strangle if self.tide_strangle else None

            # Market 3 – WX_SPOT: T(F) * humidity at settlement
            if self.weather:
                fv["WX_SPOT"] = self.weather["temp_f"] * self.weather["humidity"]
            else:
                fv["WX_SPOT"] = None

            # Market 4 – WX_SUM: cumulative (T*H)/100
            fv["WX_SUM"] = self.wx_sum_acc if self.wx_sum_acc else None

            # Market 5 – LHR_COUNT: total arrivals + departures (24h)
            if self.lhr:
                fv["LHR_COUNT"] = self.lhr["arrivals"] + self.lhr["departures"]
            else:
                fv["LHR_COUNT"] = None

            # Market 6 – LHR_INDEX: cumulative airport metric (absolute)
            fv["LHR_INDEX"] = abs(self.lhr_index_acc) if self.lhr_index_acc else None

            # Market 7 – LON_ETF: TIDE_SPOT + WX_SPOT + LHR_COUNT
            if all(fv.get(k) is not None for k in ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]):
                fv["LON_ETF"] = fv["TIDE_SPOT"] + fv["WX_SPOT"] + fv["LHR_COUNT"]
            else:
                fv["LON_ETF"] = None

            # Market 8 – LON_FLY: option package on LON_ETF
            if fv.get("LON_ETF") is not None:
                S = fv["LON_ETF"]
                # +2 P6200 +1 C6200 -2 C6600 +3 C7000
                val  = 2  * max(0, 6200 - S)
                val += 1  * max(0, S - 6200)
                val += -2 * max(0, S - 6600)
                val += 3  * max(0, S - 7000)
                fv["LON_FLY"] = val
            else:
                fv["LON_FLY"] = None

            return fv


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI AI AGENT
# ─────────────────────────────────────────────────────────────────────────────

class GeminiAgent:
    """Calls Gemini to suggest trade actions given market state."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._available = False
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            self._available = True
            print("[Gemini] Agent initialised ✓")
        except Exception as e:
            print(f"[Gemini] Not available: {e}")

    def suggest_trades(self, market_state: dict) -> list[dict]:
        """Returns list of {product, side, price, volume, reason}."""
        if not self._available:
            return []
        try:
            import google.generativeai as genai
            prompt = f"""
You are an expert algorithmic trader competing in a simulated London data trading competition.

MARKETS:
- TIDE_SPOT:  settles to ABS(Thames tidal height) * 1000
- TIDE_SWING: settles to cumulative strangle value on 15-min tide differences
- WX_SPOT:    settles to Temperature(F) * Humidity at Sunday 12:00
- WX_SUM:     settles to sum of (T*H)/100 over all 15-min intervals
- LHR_COUNT:  settles to total arrivals + departures at Heathrow (24h)
- LHR_INDEX:  settles to ABS(sum of 100*(arr-dep)/(arr+dep) over 30-min slots)
- LON_ETF:    settles to TIDE_SPOT + WX_SPOT + LHR_COUNT
- LON_FLY:    option package on LON_ETF: +2 P6200 +1 C6200 -2 C6600 +3 C7000

POSITION LIMITS: ±100 per product (stay within ±90 to be safe)
SCORING: normalised PnL per product group — so accuracy of fair value prediction matters most.

CURRENT MARKET STATE:
{json.dumps(market_state, indent=2)}

Based on the fair value estimates vs current market prices, identify the best trades.
For each trade, output a JSON array of objects with keys:
  product (string), side ("BUY" or "SELL"), price (number), volume (integer 1-10), reason (string)

Respond ONLY with a valid JSON array, no other text. If no good trades, return [].
Be aggressive but stay within position limits. Focus on products where fair value differs most from mid-price.
"""
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            trades = json.loads(text)
            print(f"[Gemini] Suggested {len(trades)} trades")
            return trades if isinstance(trades, list) else []
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STRATEGY BOT
# ─────────────────────────────────────────────────────────────────────────────

class IMCStrategyBot(BaseBot):
    """
    Multi-strategy bot:
      1. Fair-value alpha signals from real London data
      2. Market making around fair value
      3. ETF arbitrage (LON_ETF vs constituents)
      4. Gemini AI agent for strategic overrides
    """

    PRODUCTS = ["TIDE_SPOT", "TIDE_SWING", "WX_SPOT", "WX_SUM",
                "LHR_COUNT", "LHR_INDEX", "LON_ETF", "LON_FLY"]

    # LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT  (arbitrage legs)
    ETF_LEGS = {"LON_ETF": 1, "TIDE_SPOT": -1, "WX_SPOT": -1, "LHR_COUNT": -1}

    def __init__(self, cfg: dict):
        super().__init__(cfg["cmi_url"], cfg["username"], cfg["password"])
        self.cfg        = cfg
        self.fv         = FairValueEstimator()
        self.gemini     = GeminiAgent(cfg["gemini_key"]) if cfg["use_gemini"] else None
        self._lock      = Lock()

        # Live market state
        self.orderbooks: dict[str, OrderBook] = {}
        self.mid_prices: dict[str, float]     = {}
        self.positions:  dict[str, int]       = {}
        self.recent_trades: dict[str, deque]  = defaultdict(lambda: deque(maxlen=20))

        # Track our pending quote order IDs per product to cancel/replace
        self.quote_ids: dict[str, list[str]]  = defaultdict(list)

        self._running = False

    # ── callbacks ──────────────────────────────────────────────────────────

    def on_orderbook(self, ob: OrderBook):
        with self._lock:
            self.orderbooks[ob.product] = ob
            if ob.buy_orders and ob.sell_orders:
                mid = (ob.buy_orders[0].price + ob.sell_orders[0].price) / 2
                self.mid_prices[ob.product] = mid
            elif ob.buy_orders:
                self.mid_prices[ob.product] = ob.buy_orders[0].price
            elif ob.sell_orders:
                self.mid_prices[ob.product] = ob.sell_orders[0].price

        # Trigger strategies on each book update
        self._run_strategies(ob.product)

    def on_trades(self, trade: Trade):
        self.recent_trades[trade.product].append(trade)

    # ── strategy dispatcher ────────────────────────────────────────────────

    def _run_strategies(self, product: str):
        try:
            fair_values = self.fv.fair_values()
            positions   = self.get_positions()
            with self._lock:
                self.positions = positions

            orders_to_send = []

            if self.cfg["use_alpha_signals"]:
                orders_to_send += self._alpha_signal_orders(product, fair_values, positions)

            if self.cfg["use_market_making"]:
                orders_to_send += self._market_make(product, fair_values, positions)

            if self.cfg["use_arbitrage"] and product in ("LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"):
                orders_to_send += self._etf_arbitrage(fair_values, positions)

            if orders_to_send:
                self.send_orders(orders_to_send)

        except Exception:
            print(format_exc())

    # ── alpha signal: trade towards fair value ─────────────────────────────

    def _alpha_signal_orders(self, product: str, fvs: dict, positions: dict) -> list[OrderRequest]:
        fv = fvs.get(product)
        if fv is None:
            return []
        with self._lock:
            ob = self.orderbooks.get(product)
        if ob is None:
            return []
        if not ob.buy_orders or not ob.sell_orders:
            return []

        best_bid = ob.buy_orders[0].price
        best_ask = ob.sell_orders[0].price
        mid      = (best_bid + best_ask) / 2
        pos      = positions.get(product, 0)
        tick     = ob.tick_size or 1.0
        orders   = []

        edge = fv - mid  # positive → we think it's underpriced → BUY

        if edge > tick * 2:
            # Aggressively cross spread if edge is large
            price = best_ask if edge > tick * 5 else best_bid + tick
            size  = min(5, self.cfg["max_position"] - pos)
            if size > 0:
                orders.append(OrderRequest(product=product, price=price,
                                           side=Side.BUY, volume=size))
                print(f"[Alpha] BUY {product} @ {price} | edge={edge:.1f}")

        elif edge < -tick * 2:
            price = best_bid if edge < -tick * 5 else best_ask - tick
            size  = min(5, self.cfg["max_position"] + pos)
            if size > 0:
                orders.append(OrderRequest(product=product, price=price,
                                           side=Side.SELL, volume=size))
                print(f"[Alpha] SELL {product} @ {price} | edge={edge:.1f}")

        return orders

    # ── market making ──────────────────────────────────────────────────────

    def _market_make(self, product: str, fvs: dict, positions: dict) -> list[OrderRequest]:
        fv = fvs.get(product)
        with self._lock:
            ob = self.orderbooks.get(product)
        if ob is None or not ob.buy_orders or not ob.sell_orders:
            return []

        tick  = ob.tick_size or 1.0
        pos   = positions.get(product, 0)
        mid   = (ob.buy_orders[0].price + ob.sell_orders[0].price) / 2
        ref   = fv if fv else mid
        half  = tick * self.cfg["mm_spread_ticks"]
        size  = self.cfg["mm_size"]
        orders = []

        # Cancel old quotes first (fire and forget)
        old_ids = self.quote_ids.pop(product, [])
        for oid in old_ids:
            Thread(target=self.cancel_order, args=(oid,), daemon=True).start()

        # Skew quotes based on position (lean against inventory)
        skew = -pos * tick * 0.5

        bid_price = round((ref - half + skew) / tick) * tick
        ask_price = round((ref + half + skew) / tick) * tick

        max_p = self.cfg["max_position"]
        if pos < max_p:
            orders.append(OrderRequest(product=product, price=bid_price,
                                       side=Side.BUY,  volume=size))
        if pos > -max_p:
            orders.append(OrderRequest(product=product, price=ask_price,
                                       side=Side.SELL, volume=size))

        return orders

    # ── ETF arbitrage ──────────────────────────────────────────────────────

    def _etf_arbitrage(self, fvs: dict, positions: dict) -> list[OrderRequest]:
        """
        LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
        If ETF trades rich vs sum of legs → sell ETF, buy legs.
        If ETF trades cheap                → buy ETF, sell legs.
        """
        orders = []
        with self._lock:
            etf_ob   = self.orderbooks.get("LON_ETF")
            tide_ob  = self.orderbooks.get("TIDE_SPOT")
            wx_ob    = self.orderbooks.get("WX_SPOT")
            lhr_ob   = self.orderbooks.get("LHR_COUNT")

        for ob in [etf_ob, tide_ob, wx_ob, lhr_ob]:
            if ob is None or not ob.buy_orders or not ob.sell_orders:
                return []

        etf_bid  = etf_ob.buy_orders[0].price
        etf_ask  = etf_ob.sell_orders[0].price
        leg_sum_ask = (tide_ob.sell_orders[0].price +
                       wx_ob.sell_orders[0].price   +
                       lhr_ob.sell_orders[0].price)
        leg_sum_bid = (tide_ob.buy_orders[0].price +
                       wx_ob.buy_orders[0].price   +
                       lhr_ob.buy_orders[0].price)

        threshold = self.cfg["arb_threshold"]

        # ETF too expensive relative to legs → sell ETF, buy legs
        if etf_bid - leg_sum_ask > threshold:
            pos_etf  = positions.get("LON_ETF",   0)
            pos_tide = positions.get("TIDE_SPOT",  0)
            pos_wx   = positions.get("WX_SPOT",    0)
            pos_lhr  = positions.get("LHR_COUNT",  0)
            size = 2
            print(f"[ARB] ETF rich: etf_bid={etf_bid} vs legs_ask={leg_sum_ask} | diff={etf_bid-leg_sum_ask:.1f}")
            if pos_etf > -self.cfg["max_position"]:
                orders.append(OrderRequest("LON_ETF",   etf_bid,                      Side.SELL, size))
            if pos_tide < self.cfg["max_position"]:
                orders.append(OrderRequest("TIDE_SPOT", tide_ob.sell_orders[0].price, Side.BUY,  size))
            if pos_wx < self.cfg["max_position"]:
                orders.append(OrderRequest("WX_SPOT",   wx_ob.sell_orders[0].price,   Side.BUY,  size))
            if pos_lhr < self.cfg["max_position"]:
                orders.append(OrderRequest("LHR_COUNT", lhr_ob.sell_orders[0].price,  Side.BUY,  size))

        # ETF too cheap → buy ETF, sell legs
        elif leg_sum_bid - etf_ask > threshold:
            pos_etf  = positions.get("LON_ETF",   0)
            pos_tide = positions.get("TIDE_SPOT",  0)
            pos_wx   = positions.get("WX_SPOT",    0)
            pos_lhr  = positions.get("LHR_COUNT",  0)
            size = 2
            print(f"[ARB] ETF cheap: legs_bid={leg_sum_bid} vs etf_ask={etf_ask} | diff={leg_sum_bid-etf_ask:.1f}")
            if pos_etf < self.cfg["max_position"]:
                orders.append(OrderRequest("LON_ETF",   etf_ask,                     Side.BUY,  size))
            if pos_tide > -self.cfg["max_position"]:
                orders.append(OrderRequest("TIDE_SPOT", tide_ob.buy_orders[0].price, Side.SELL, size))
            if pos_wx > -self.cfg["max_position"]:
                orders.append(OrderRequest("WX_SPOT",   wx_ob.buy_orders[0].price,   Side.SELL, size))
            if pos_lhr > -self.cfg["max_position"]:
                orders.append(OrderRequest("LHR_COUNT", lhr_ob.buy_orders[0].price,  Side.SELL, size))

        return orders

    # ── Gemini agent loop ──────────────────────────────────────────────────

    def _gemini_loop(self):
        while self._running:
            time.sleep(self.cfg["gemini_interval"])
            if not self._running:
                break
            try:
                fvs       = self.fv.fair_values()
                positions = self.get_positions()
                with self._lock:
                    mids = dict(self.mid_prices)

                state = {
                    "fair_values":  {k: round(v, 2) if v else None for k, v in fvs.items()},
                    "mid_prices":   {k: round(v, 2) for k, v in mids.items()},
                    "positions":    positions,
                    "pnl":          self.get_pnl(),
                    "thames_level": self.fv.thames_level,
                    "weather":      self.fv.weather,
                    "lhr":          self.fv.lhr,
                }

                suggestions = self.gemini.suggest_trades(state)
                orders = []
                for s in suggestions:
                    try:
                        prod = s["product"]
                        side = Side.BUY if s["side"].upper() == "BUY" else Side.SELL
                        price = float(s["price"])
                        vol   = max(1, min(10, int(s.get("volume", 3))))
                        pos   = positions.get(prod, 0)
                        # Respect position limits
                        if side == Side.BUY  and pos >= self.cfg["max_position"]: continue
                        if side == Side.SELL and pos <= -self.cfg["max_position"]: continue
                        orders.append(OrderRequest(prod, price, side, vol))
                        print(f"[Gemini] {side} {vol}x {prod} @ {price} — {s.get('reason','')}")
                    except Exception:
                        pass
                if orders:
                    self.send_orders(orders)
            except Exception:
                print(format_exc())

    # ── external data fetch loop ───────────────────────────────────────────

    def _data_loop(self):
        while self._running:
            thames  = fetch_thames_level()
            weather = fetch_weather()
            lhr     = fetch_lhr_flights()
            self.fv.update(thames, weather, lhr)
            fvs = self.fv.fair_values()
            print(f"\n[Data] Thames={thames} | Wx={weather} | LHR={lhr}")
            print(f"[FairValues] {json.dumps({k: round(v,1) if v else None for k,v in fvs.items()})}\n")
            time.sleep(self.cfg["data_fetch_interval"])

    # ── start / stop ───────────────────────────────────────────────────────

    def run(self):
        self._running = True
        print("=" * 60)
        print("  IMC Algothon Bot — starting")
        print("=" * 60)

        # Kick off background threads
        Thread(target=self._data_loop,   daemon=True, name="DataLoop").start()

        if self.cfg["use_gemini"] and self.gemini and self.gemini._available:
            Thread(target=self._gemini_loop, daemon=True, name="GeminiLoop").start()

        # Connect SSE stream
        self.start()
        print("[Bot] SSE stream connected.  Trading live.")
        print("      Ctrl-C to stop.\n")

        try:
            while True:
                time.sleep(5)
                pnl = self.get_pnl()
                pos = self.get_positions()
                print(f"[Status] PnL={pnl} | Positions={pos}")
        except KeyboardInterrupt:
            print("\n[Bot] Stopping…")
        finally:
            self._running = False
            self.cancel_all_orders()
            self.stop()
            print("[Bot] Stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── override credentials here or use env vars ──
    # CONFIG["cmi_url"]    = "http://YOUR_EXCHANGE_URL"
    # CONFIG["username"]   = "your_username"
    # CONFIG["password"]   = "your_password"
    # CONFIG["gemini_key"] = "your_gemini_api_key"

    bot = IMCStrategyBot(CONFIG)
    bot.run()
