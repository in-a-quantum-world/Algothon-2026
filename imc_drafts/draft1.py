

import json
import math
import time
import threading
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import StrEnum
from functools import cached_property
from threading import Thread
from traceback import format_exc
from typing import Any, Callable, Literal

import requests
import sseclient

STANDARD_HEADERS = {"Content-Type": "application/json; charset=utf-8"}


class DictLikeFrozenDataclassMapping(Mapping):
    """Mixin class to allow frozen dataclasses behave like a dict."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__annotations__)

    def __len__(self) -> int:
        return len(self.__annotations__)

    def to_dict(self) -> dict:
        return asdict(self)

    def keys(self):
        return self.__annotations__.keys()

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]


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
    BUY = "BUY"
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
    message: str | None = None


class _SSEThread(Thread):
    """Background thread that consumes the CMI SSE stream and dispatches events."""

    def __init__(
        self,
        bearer: str,
        url: str,
        handle_orderbook: Callable[[OrderBook], Any],
        handle_trade_event: Callable[[Trade], Any],
    ):
        super().__init__(daemon=True)
        self._bearer = bearer
        self._url = url
        self._handle_orderbook = handle_orderbook
        self._handle_trade_event = handle_trade_event
        self._http_stream: requests.Response | None = None
        self._client: sseclient.SSEClient | None = None
        self._closed = False

    def run(self):
        while not self._closed:
            try:
                self._consume()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                pass
            except Exception:
                if not self._closed:
                    print("SSE error, reconnecting...")
                    print(format_exc())

    def close(self):
        self._closed = True
        if self._http_stream:
            self._http_stream.close()
        if self._client:
            self._client.close()

    def _consume(self):
        headers = {
            "Authorization": self._bearer,
            "Accept": "text/event-stream; charset=utf-8",
        }
        self._http_stream = requests.get(self._url, stream=True, headers=headers, timeout=30)
        self._client = sseclient.SSEClient(self._http_stream)

        for event in self._client.events():
            if event.event == "order":
                self._on_order_event(json.loads(event.data))
            elif event.event == "trade":
                data = json.loads(event.data)
                trades = data if isinstance(data, list) else [data]
                trade_fields = {f.name for f in Trade.__dataclass_fields__.values()}
                for t in trades:
                    self._handle_trade_event(Trade(**{k: v for k, v in t.items() if k in trade_fields}))

    def _on_order_event(self, data: dict[str, Any]):
        buy_orders = sorted(
            [
                Order(price=float(price), volume=v["marketVolume"], own_volume=v["userVolume"])
                for price, v in data["buyOrders"].items()
            ],
            key=lambda o: -o.price,
        )
        sell_orders = sorted(
            [
                Order(price=float(price), volume=v["marketVolume"], own_volume=v["userVolume"])
                for price, v in data["sellOrders"].items()
            ],
            key=lambda o: o.price,
        )
        self._handle_orderbook(OrderBook(data["productsymbol"], data["tickSize"], buy_orders, sell_orders))


class BaseBot(ABC):
    """Base bot for CMI Exchange.
    """

    def __init__(self, cmi_url: str, username: str, password: str):
        self._cmi_url = cmi_url.rstrip("/")
        self.username = username
        self._password = password
        self._sse_thread: _SSEThread | None = None

        # Incremental trade state
        self.trades: list[Trade] = []
        self._trade_watermark: str | None = None
        self._last_trade_fetch: float | None = None

    @cached_property
    def auth_token(self) -> str:
        response = requests.post(
            f"{self._cmi_url}/api/user/authenticate",
            headers=STANDARD_HEADERS,
            json={"username": self.username, "password": self._password},
        )
        response.raise_for_status()
        return response.headers["Authorization"]

    # -- lifecycle --

    def start(self) -> None:
        if self._sse_thread:
            raise RuntimeError("Bot already running. Call stop() first.")
        self._sse_thread = _SSEThread(
            bearer=self.auth_token,
            url=f"{self._cmi_url}/api/market/stream",
            handle_orderbook=self.on_orderbook,
            handle_trade_event=self.on_trades,
        )
        self._sse_thread.start()

    def stop(self) -> None:
        if self._sse_thread:
            self._sse_thread.close()
            self._sse_thread.join(timeout=5)
            self._sse_thread = None

    # -- callbacks --

    @abstractmethod
    def on_orderbook(self, orderbook: OrderBook) -> None: ...

    @abstractmethod
    def on_trades(self, trade: Trade) -> None: ...

    # -- market trades (incremental) --

    def get_market_trades(self) -> list[Trade]:
        """Fetch new market trades from the exchange and append to self.trades.

        Uses incremental loading: only requests trades newer than the last
        seen timestamp. Returns the full accumulated list.
        """
        params: dict[str, str] = {}
        if self._trade_watermark:
            params["from"] = self._trade_watermark
        response = requests.get(
            f"{self._cmi_url}/api/trade",
            params=params,
            headers=self._auth_headers(),
        )
        self._last_trade_fetch = time.monotonic()
        if not response.ok:
            print(f"Failed to fetch trades: {response.status_code}")
            return self.trades

        new_trades = []
        for raw in response.json():
            trade = Trade(**raw)
            if self._trade_watermark is None or trade.timestamp > self._trade_watermark:
                new_trades.append(trade)

        if new_trades:
            self.trades.extend(new_trades)
            self._trade_watermark = new_trades[-1].timestamp

        return self.trades

    @property
    def last_trade_fetch_age(self) -> float | None:
        """Seconds since last get_market_trades() call, or None if never called."""
        if self._last_trade_fetch is None:
            return None
        return time.monotonic() - self._last_trade_fetch

    # -- trading helpers --

    def send_order(self, order: OrderRequest) -> OrderResponse | None:
        response = requests.post(
            f"{self._cmi_url}/api/order",
            json=asdict(order),
            headers=self._auth_headers(),
        )
        if response.ok:
            return OrderResponse(**response.json())
        print(f"Order failed: {response.text}")
        return None

    def send_orders(self, orders: list[OrderRequest]) -> list[OrderResponse]:
        results: list[OrderResponse] = []

        def _send(o: OrderRequest):
            r = self.send_order(o)
            if r:
                results.append(r)

        threads = [Thread(target=_send, args=(o,)) for o in orders]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def cancel_order(self, order_id: str) -> None:
        requests.delete(f"{self._cmi_url}/api/order/{order_id}", headers=self._auth_headers())

    def cancel_all_orders(self) -> None:
        orders = self.get_orders()
        threads = [Thread(target=self.cancel_order, args=(o["id"],)) for o in orders]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def get_orders(self, product: str | None = None) -> list[dict]:
        params = {"productsymbol": product} if product else {}
        response = requests.get(
            f"{self._cmi_url}/api/order/current-user",
            params=params,
            headers=self._auth_headers(),
        )
        return response.json() if response.ok else []

    def get_products(self) -> list[Product]:
        response = requests.get(f"{self._cmi_url}/api/product", headers=self._auth_headers())
        response.raise_for_status()
        return [Product(**p) for p in response.json()]

    def get_positions(self) -> dict[str, int]:
        response = requests.get(
            f"{self._cmi_url}/api/position/current-user",
            headers=self._auth_headers(),
        )
        if response.ok:
            return {p["product"]: p["netPosition"] for p in response.json()}
        return {}

    def get_orderbook(self, product: str) -> OrderBook:
        response = requests.get(
            f"{self._cmi_url}/api/product/{product}/order-book/current-user",
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        data = response.json()
        buy_orders = sorted(
            [Order(price=e["price"], volume=e["volume"], own_volume=e["userOrderVolume"]) for e in data.get("buy", [])],
            key=lambda o: -o.price,
        )
        sell_orders = sorted(
            [Order(price=e["price"], volume=e["volume"], own_volume=e["userOrderVolume"]) for e in data.get("sell", [])],
            key=lambda o: o.price,
        )
        return OrderBook(data["product"], data["tickSize"], buy_orders, sell_orders)

    def get_pnl(self) -> dict:
        response = requests.get(
            f"{self._cmi_url}/api/profit/current-user",
            headers=self._auth_headers(),
        )
        return response.json() if response.ok else {}

    # -- internals --

    def _auth_headers(self) -> dict[str, str]:
        return {**STANDARD_HEADERS, "Authorization": self.auth_token}


# ============================================================================
# DATA FETCHING
# ============================================================================

class DataFetcher:
    """Fetches real-time London data for fair value computation."""

    TIDE_URL = (
        "http://environment.data.gov.uk/flood-monitoring/id/measures/"
        "0006-level-tidal_level-i-15_min-mAOD/readings?_sorted&_limit=200"
    )
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    WEATHER_PARAMS = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/London",
        "forecast_days": 2,
        "past_days": 1,
        "temperature_unit": "fahrenheit",
    }

    def __init__(self):
        self.tide_readings = []
        self.weather_data = []
        self.last_tide_fetch = 0
        self.last_weather_fetch = 0

    def fetch_tide_data(self):
        try:
            if time.time() - self.last_tide_fetch < 60:
                return self.tide_readings
            resp = requests.get(self.TIDE_URL, timeout=10)
            if resp.ok:
                data = resp.json()
                readings = []
                for item in data.get("items", []):
                    dt = datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00"))
                    readings.append((dt, item["value"]))
                readings.sort(key=lambda x: x[0])
                self.tide_readings = readings
                self.last_tide_fetch = time.time()
                print(f"[DATA] Fetched {len(readings)} tide readings")
        except Exception as e:
            print(f"[DATA] Tide fetch error: {e}")
        return self.tide_readings

    def fetch_weather_data(self):
        try:
            if time.time() - self.last_weather_fetch < 120:
                return self.weather_data
            resp = requests.get(self.WEATHER_URL, params=self.WEATHER_PARAMS, timeout=10)
            if resp.ok:
                data = resp.json()
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                humids = hourly.get("relative_humidity_2m", [])
                self.weather_data = [(datetime.fromisoformat(t), temp, hum)
                                     for t, temp, hum in zip(times, temps, humids)]
                self.last_weather_fetch = time.time()
                print(f"[DATA] Fetched {len(self.weather_data)} weather points")
        except Exception as e:
            print(f"[DATA] Weather fetch error: {e}")
        return self.weather_data


# ============================================================================
# FAIR VALUE CALCULATOR
# ============================================================================

class FairValueCalculator:
    """Computes fair values for all 8 markets."""

    def __init__(self, data_fetcher: DataFetcher):
        self.data = data_fetcher
        self.fair_values = {}
        self.confidence = {}

    def update_all(self):
        self.data.fetch_tide_data()
        self.data.fetch_weather_data()
        self._compute_tide_spot()
        self._compute_tide_swing()
        self._compute_wx_spot()
        self._compute_wx_sum()
        self._compute_lhr_count()
        self._compute_lhr_index()
        self._compute_lon_etf()
        self._compute_lon_fly()
        return self.fair_values

    def _compute_tide_spot(self):
        """Market 1: ABS(tidal height MAOD) * 1000 at Sunday 12:00 PM."""
        readings = self.data.tide_readings
        if not readings:
            return
        latest_val = readings[-1][1]
        now = datetime.now(timezone.utc)
        target = now.replace(hour=12, minute=0, second=0, microsecond=0)
        closest = min(readings, key=lambda r: abs((r[0] - target).total_seconds()))
        if abs((closest[0] - target).total_seconds()) < 1800:
            settlement_est = abs(closest[1]) * 1000
            self.confidence["TIDE_SPOT"] = 0.9
        else:
            settlement_est = abs(latest_val) * 1000
            self.confidence["TIDE_SPOT"] = 0.5
        self.fair_values["TIDE_SPOT"] = round(settlement_est)
        print(f"[FV] TIDE_SPOT = {self.fair_values['TIDE_SPOT']} (latest: {latest_val:.3f}m)")

    def _compute_tide_swing(self):
        """Market 2: Sum of strangle(0.2-0.25) on 15m diffs * 100."""
        readings = self.data.tide_readings
        if len(readings) < 10:
            return
        total_strangle = 0.0
        for i in range(1, len(readings)):
            dt_diff = (readings[i][0] - readings[i-1][0]).total_seconds()
            if abs(dt_diff - 900) < 120:
                diff = abs(readings[i][1] - readings[i-1][1])
                strangle_val = max(0, 0.2 - diff) + max(0, diff - 0.25)
                total_strangle += strangle_val
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=24)
        actual = len([r for r in readings if r[0] >= window_start])
        scale = 96 / max(actual, 1)
        settlement_est = total_strangle * 100 * min(scale, 2.0)
        self.fair_values["TIDE_SWING"] = round(settlement_est)
        self.confidence["TIDE_SWING"] = min(actual / 96, 1.0)
        print(f"[FV] TIDE_SWING = {self.fair_values['TIDE_SWING']} ({actual} readings)")

    def _compute_wx_spot(self):
        """Market 3: Temperature(F) * Humidity at Sunday 12:00 PM."""
        weather = self.data.weather_data
        if not weather:
            return
        now = datetime.now()
        target = now.replace(hour=12, minute=0, second=0, microsecond=0)
        closest = min(weather, key=lambda w: abs((w[0] - target).total_seconds()))
        temp_f = round(closest[1])
        humidity = round(closest[2])
        self.fair_values["WX_SPOT"] = temp_f * humidity
        self.confidence["WX_SPOT"] = 0.7
        print(f"[FV] WX_SPOT = {self.fair_values['WX_SPOT']} (T={temp_f}F, H={humidity}%)")

    def _compute_wx_sum(self):
        """Market 4: Sum of (Temp*Humidity)/100 for all 15-min intervals."""
        weather = self.data.weather_data
        if not weather:
            return
        now = datetime.now()
        window_start = now - timedelta(hours=24)
        total = 0.0
        count = 0
        for dt, temp, hum in weather:
            if dt >= window_start:
                total += (round(temp) * round(hum)) / 100 * 4
                count += 1
        self.fair_values["WX_SUM"] = round(total)
        self.confidence["WX_SUM"] = min(count / 24, 1.0)
        print(f"[FV] WX_SUM = {self.fair_values['WX_SUM']} ({count} hours)")

    def _compute_lhr_count(self):
        """Market 5: Total flights at LHR in 24h. Estimated without API key."""
        self.fair_values["LHR_COUNT"] = 1000
        self.confidence["LHR_COUNT"] = 0.3
        print(f"[FV] LHR_COUNT = 1000 (estimated)")

    def _compute_lhr_index(self):
        """Market 6: Airport metric. Small value as arrivals ~ departures."""
        self.fair_values["LHR_INDEX"] = 15
        self.confidence["LHR_INDEX"] = 0.2
        print(f"[FV] LHR_INDEX = 15 (estimated)")

    def _compute_lon_etf(self):
        """Market 7: ETF = Market1 + Market3 + Market5."""
        m1 = self.fair_values.get("TIDE_SPOT", 0)
        m3 = self.fair_values.get("WX_SPOT", 0)
        m5 = self.fair_values.get("LHR_COUNT", 0)
        if m1 and m3 and m5:
            self.fair_values["LON_ETF"] = m1 + m3 + m5
            self.confidence["LON_ETF"] = 0.5
            print(f"[FV] LON_ETF = {self.fair_values['LON_ETF']} ({m1}+{m3}+{m5})")

    def _compute_lon_fly(self):
        """Market 8: +2 P6200 +C6200 -2 C6600 +3 C7000."""
        S = self.fair_values.get("LON_ETF", 0)
        if not S:
            return
        fly = (2 * max(0, 6200 - S) + max(0, S - 6200)
               - 2 * max(0, S - 6600) + 3 * max(0, S - 7000))
        self.fair_values["LON_FLY"] = round(fly)
        self.confidence["LON_FLY"] = 0.4
        print(f"[FV] LON_FLY = {self.fair_values['LON_FLY']} (ETF={S})")


# ============================================================================
# TRADING BOT
# ============================================================================

class IMCityBot(BaseBot):
    """Multi-strategy trading bot for IMCity Challenge."""

    MAX_POSITION = 80
    MM_SPREAD_TICKS = 3
    MM_VOLUME = 2
    ALPHA_THRESHOLD = 0.03
    ALPHA_VOLUME = 5

    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        self.data_fetcher = DataFetcher()
        self.fv_calculator = FairValueCalculator(self.data_fetcher)
        self.orderbooks = {}
        self.positions = {}
        self.tick_sizes = {}
        self.fair_values = {}
        self.mid_prices = {}
        self.last_fv_update = 0
        self.last_mm_cycle = 0
        self.last_position_check = 0
        self.last_alpha_trade = {}
        self.lock = threading.Lock()

    def run(self):
        print(f"\n{'='*60}")
        print(f"  IMCity Trading Bot")
        print(f"  User: {self.username}")
        print(f"{'='*60}\n")

        self._init_products()
        self._update_positions()
        self.start()

        try:
            while True:
                try:
                    self._trading_loop()
                except Exception as e:
                    print(f"[ERROR] {e}")
                    traceback.print_exc()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[BOT] Shutting down...")
            self.cancel_all_orders()
            self.stop()

    def _init_products(self):
        products = self.get_products()
        for p in products:
            self.tick_sizes[p.symbol] = p.tickSize
            print(f"[INIT] {p.symbol}: tick={p.tickSize}, start={p.startingPrice}")

    def _update_positions(self):
        try:
            self.positions = self.get_positions()
            pos_str = ", ".join(f"{k}:{v}" for k, v in self.positions.items() if v != 0)
            if pos_str:
                print(f"[POS] {pos_str}")
        except Exception as e:
            print(f"[ERROR] Position update: {e}")

    # ── Callbacks ──

    def on_orderbook(self, orderbook: OrderBook):
        with self.lock:
            self.orderbooks[orderbook.product] = orderbook
            self.tick_sizes[orderbook.product] = orderbook.tick_size
            if orderbook.buy_orders and orderbook.sell_orders:
                self.mid_prices[orderbook.product] = (
                    orderbook.buy_orders[0].price + orderbook.sell_orders[0].price) / 2

    def on_trades(self, trade: Trade):
        if trade.buyer == self.username or trade.seller == self.username:
            side = "BOUGHT" if trade.buyer == self.username else "SOLD"
            print(f"[TRADE] {side} {trade.volume}x {trade.product} @ {trade.price}")

    # ── Trading Loop ──

    def _trading_loop(self):
        now = time.time()

        if now - self.last_fv_update > 30:
            try:
                self.fair_values = self.fv_calculator.update_all()
                self.last_fv_update = now
            except Exception as e:
                print(f"[ERROR] FV: {e}")

        if now - self.last_position_check > 10:
            self._update_positions()
            self.last_position_check = now

        if now - self.last_mm_cycle > 3:
            self._market_making_cycle()
            self.last_mm_cycle = now

        self._check_etf_arbitrage()
        self._check_alpha_signals()

    # ── Market Making ──

    def _market_making_cycle(self):
        for product in list(self.orderbooks.keys()):
            try:
                self._market_make_product(product)
            except Exception as e:
                print(f"[ERROR] MM {product}: {e}")

    def _market_make_product(self, product: str):
        ob = self.orderbooks.get(product)
        if not ob:
            return

        tick = self.tick_sizes.get(product, 1)
        pos = self.positions.get(product, 0)
        fv = self.fair_values.get(product)
        ref_price = fv if fv else self.mid_prices.get(product)
        if not ref_price:
            return

        try:
            for o in self.get_orders(product):
                self.cancel_order(o["id"])
        except:
            pass

        confidence = self.fv_calculator.confidence.get(product, 0.3)
        spread = max(self.MM_SPREAD_TICKS * tick, tick) / max(confidence, 0.2)
        skew = -pos * tick * 0.5

        bid_price = math.floor((ref_price - spread + skew) / tick) * tick
        ask_price = math.ceil((ref_price + spread + skew) / tick) * tick
        if bid_price >= ask_price:
            ask_price = bid_price + tick

        orders = []
        buy_vol = self.MM_VOLUME if pos < self.MAX_POSITION - self.MM_VOLUME else 1
        sell_vol = self.MM_VOLUME if pos > -self.MAX_POSITION + self.MM_VOLUME else 1

        if pos < self.MAX_POSITION:
            orders.append(OrderRequest(product, round(bid_price, 4), Side.BUY, buy_vol))
        if pos > -self.MAX_POSITION:
            orders.append(OrderRequest(product, round(ask_price, 4), Side.SELL, sell_vol))

        if orders:
            self.send_orders(orders)

    # ── ETF Arbitrage ──

    def _check_etf_arbitrage(self):
        etf_mid = self.mid_prices.get("LON_ETF")
        tide_mid = self.mid_prices.get("TIDE_SPOT")
        wx_mid = self.mid_prices.get("WX_SPOT")
        lhr_mid = self.mid_prices.get("LHR_COUNT")

        if not all([etf_mid, tide_mid, wx_mid, lhr_mid]):
            return

        constituent_sum = tide_mid + wx_mid + lhr_mid
        spread = etf_mid - constituent_sum
        threshold = 20
        etf_pos = self.positions.get("LON_ETF", 0)

        if abs(spread) > threshold:
            direction = "SELL_ETF" if spread > 0 else "BUY_ETF"
            if (direction == "SELL_ETF" and etf_pos > -self.MAX_POSITION + 3) or \
               (direction == "BUY_ETF" and etf_pos < self.MAX_POSITION - 3):
                print(f"[ARB] ETF spread={spread:.0f}, {direction}")
                self._execute_arb(direction)

    def _execute_arb(self, direction: str):
        orders = []
        vol = 1
        if direction == "SELL_ETF":
            legs = [("LON_ETF", Side.SELL), ("TIDE_SPOT", Side.BUY),
                    ("WX_SPOT", Side.BUY), ("LHR_COUNT", Side.BUY)]
        else:
            legs = [("LON_ETF", Side.BUY), ("TIDE_SPOT", Side.SELL),
                    ("WX_SPOT", Side.SELL), ("LHR_COUNT", Side.SELL)]

        for prod, side in legs:
            ob = self.orderbooks.get(prod)
            if not ob:
                return
            price_list = ob.buy_orders if side == Side.SELL else ob.sell_orders
            if price_list:
                orders.append(OrderRequest(prod, price_list[0].price, side, vol))

        if len(orders) == 4:
            self.send_orders(orders)

    # ── Alpha Signals ──

    def _check_alpha_signals(self):
        now = time.time()
        for product, fv in self.fair_values.items():
            if not fv or fv == 0:
                continue
            mid = self.mid_prices.get(product)
            if not mid or mid == 0:
                continue
            if now - self.last_alpha_trade.get(product, 0) < 10:
                continue

            deviation = (mid - fv) / fv
            pos = self.positions.get(product, 0)
            confidence = self.fv_calculator.confidence.get(product, 0.3)
            if confidence < 0.4:
                continue

            ob = self.orderbooks.get(product)
            if not ob:
                continue

            if deviation > self.ALPHA_THRESHOLD and pos > -self.MAX_POSITION + self.ALPHA_VOLUME:
                if ob.buy_orders:
                    vol = min(self.ALPHA_VOLUME, self.MAX_POSITION + pos)
                    if vol > 0:
                        result = self.send_order(
                            OrderRequest(product, ob.buy_orders[0].price, Side.SELL, vol))
                        if result:
                            print(f"[ALPHA] SELL {vol}x {product} "
                                  f"(mid={mid:.0f} fv={fv:.0f} dev={deviation:+.1%})")
                            self.last_alpha_trade[product] = now

            elif deviation < -self.ALPHA_THRESHOLD and pos < self.MAX_POSITION - self.ALPHA_VOLUME:
                if ob.sell_orders:
                    vol = min(self.ALPHA_VOLUME, self.MAX_POSITION - pos)
                    if vol > 0:
                        result = self.send_order(
                            OrderRequest(product, ob.sell_orders[0].price, Side.BUY, vol))
                        if result:
                            print(f"[ALPHA] BUY {vol}x {product} "
                                  f"(mid={mid:.0f} fv={fv:.0f} dev={deviation:+.1%})")
                            self.last_alpha_trade[product] = now


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import sys

    EXCHANGE_URL = "https://cmi.example.com"   # ← Replace with exchange URL
    USERNAME = "your_username"                  # ← Replace with team username
    PASSWORD = "your_password"                  # ← Replace with team password

    # Command line override: python bot_template.py URL USER PASS
    if len(sys.argv) >= 4:
        EXCHANGE_URL = sys.argv[1]
        USERNAME = sys.argv[2]
        PASSWORD = sys.argv[3]

    bot = IMCityBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run()
