"""
IMC Prosperity Synthetic Data Generator
=========================================
Generates historical_data.csv in IMC format for testing the full pipeline.

Products
--------
PRODUCT_A  stable price 10000, N(0, σ=3), limit=20      → FixedFairMM
PRODUCT_B  slow random walk from 5000, σ=2/step, lim=20  → RollingMidMM
X, Y, Z    basket components for PRODUCT_C
PRODUCT_C  PRODUCT_C = 4*X + 6*Y + 1*Z + premium        → SpreadArb
PRODUCT_D  slow RW ~10000, + call option PRODUCT_D_OPTION → OptionsStrategy
PRODUCT_E  slow random walk from 2000, σ=3/step, lim=20  → AS-MM fallback

Usage:
  python generate_data.py              # writes historical_data.csv (1000 steps)
  python generate_data.py --steps 500 --out my_data.csv
"""

from __future__ import annotations
import csv
import math
import os
import sys
import random
import argparse
from typing import List, Dict, Any

RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes call price (mirrors strategy_modules.py)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or S <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


# ─────────────────────────────────────────────────────────────────────────────
# Order-book generator helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_order_book(mid: float, spread: int = 2, depth_levels: int = 3, lot_size: int = 20):
    """
    Returns up to `depth_levels` bids and asks around `mid`.
    Volumes are drawn uniformly from [5, lot_size].
    """
    half = spread / 2
    bids, asks = [], []
    for i in range(depth_levels):
        bid_px = int(mid - half - i)
        ask_px = int(mid + half + i)
        vol = random.randint(5, lot_size)
        bids.append((bid_px, vol))
        asks.append((ask_px, vol))
    return bids, asks


def row_for(ts: int, sym: str, mid: float, bids, asks) -> dict:
    r: Dict[str, Any] = {
        "timestamp": ts,
        "product": sym,
        "mid_price": round(mid, 2),
        "profit_and_loss": 0,
    }
    for i, (px, vol) in enumerate(bids[:3], 1):
        r[f"bid_price_{i}"] = px
        r[f"bid_volume_{i}"] = vol
    for i in range(len(bids) + 1, 4):
        r[f"bid_price_{i}"] = ""
        r[f"bid_volume_{i}"] = ""
    for i, (px, vol) in enumerate(asks[:3], 1):
        r[f"ask_price_{i}"] = px
        r[f"ask_volume_{i}"] = vol
    for i in range(len(asks) + 1, 4):
        r[f"ask_price_{i}"] = ""
        r[f"ask_volume_{i}"] = ""
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Price process generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_stable(steps: int, fair: float, sigma: float) -> List[float]:
    """Stable product: mean-reverting around `fair`."""
    prices = []
    p = fair
    for _ in range(steps):
        p = fair + random.gauss(0, sigma)
        prices.append(p)
    return prices


def gen_random_walk(steps: int, start: float, sigma: float) -> List[float]:
    """Slow random walk with no drift."""
    prices = [start]
    for _ in range(steps - 1):
        prices.append(prices[-1] + random.gauss(0, sigma))
    return prices


def gen_basket_components(
    steps: int,
    x_start: float = 1500.0,
    y_start: float = 800.0,
    z_start: float = 3000.0,
    sigma: float = 1.5,
    basket_premium: float = 380.0,
    spread_sigma: float = 40.0,
) -> tuple:
    """
    X, Y, Z components + PRODUCT_C basket.
    PRODUCT_C ≈ 4*X + 6*Y + 1*Z + premium + spread_noise
    spread mean-reverts around basket_premium.
    """
    xs = gen_random_walk(steps, x_start, sigma)
    ys = gen_random_walk(steps, y_start, sigma)
    zs = gen_random_walk(steps, z_start, sigma)

    spread_noise = gen_mean_reverting(steps, 0, 0.9, spread_sigma)
    cs = [4 * xs[i] + 6 * ys[i] + 1 * zs[i] + basket_premium + spread_noise[i]
          for i in range(steps)]
    return xs, ys, zs, cs


def gen_mean_reverting(steps: int, mean: float, phi: float, sigma: float) -> List[float]:
    """AR(1) process with mean reversion: x_t = phi*x_{t-1} + (1-phi)*mean + e_t."""
    xs = [mean]
    for _ in range(steps - 1):
        xs.append(phi * xs[-1] + (1 - phi) * mean + random.gauss(0, sigma))
    return xs


def gen_option_series(
    steps: int,
    und_start: float = 10000.0,
    strike: float = 10000.0,
    total_t_days: float = 250.0,
    vol: float = 0.16,
    und_sigma: float = 2.0,
) -> tuple:
    """
    Underlying slow random walk + call option priced by BS.
    Vol has small AR(1) noise around historical mean.
    """
    und_prices = gen_random_walk(steps, und_start, und_sigma)
    vol_process = gen_mean_reverting(steps, vol, 0.95, 0.01)

    opt_prices = []
    for i, (S, iv) in enumerate(zip(und_prices, vol_process)):
        T = max(0.001, (total_t_days - i) / 365.0)
        iv_clamped = max(0.01, min(3.0, iv))
        opt_prices.append(bs_call_price(S, strike, T, iv_clamped))

    return und_prices, opt_prices


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = [
    "timestamp", "product",
    "bid_price_1", "bid_volume_1",
    "bid_price_2", "bid_volume_2",
    "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1",
    "ask_price_2", "ask_volume_2",
    "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]


def generate(steps: int = 1000, out_path: str = "historical_data.csv") -> None:
    random.seed(RANDOM_SEED)

    print(f"[DataGen] Generating {steps} timesteps → {out_path}")

    # Generate all price series
    a_prices = gen_stable(steps, 10000.0, 3.0)
    b_prices = gen_random_walk(steps, 5000.0, 2.0)
    xs, ys, zs, cs = gen_basket_components(steps)
    d_prices, d_opt_prices = gen_option_series(steps)
    e_prices = gen_random_walk(steps, 2000.0, 3.0)  # unknown product → AS-MM fallback

    # Timestamp step = 100 (IMC convention)
    timestamps = [100 * i for i in range(steps)]

    rows: List[dict] = []

    for i, ts in enumerate(timestamps):
        products = {
            "PRODUCT_A": (a_prices[i], 2, 20),
            "PRODUCT_B": (b_prices[i], 2, 20),
            "X":         (xs[i],       1, 30),
            "Y":         (ys[i],       1, 30),
            "Z":         (zs[i],       2, 25),
            "PRODUCT_C": (cs[i],       3, 30),
            "PRODUCT_D": (d_prices[i], 3, 25),
            "PRODUCT_D_OPTION": (d_opt_prices[i], 2, 20),
            "PRODUCT_E": (e_prices[i], 2, 20),
        }

        for sym, (mid, spread, lot) in products.items():
            bids, asks = make_order_book(mid, spread=spread, lot_size=lot)
            rows.append(row_for(ts, sym, mid, bids, asks))

    # Write CSV
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DataGen] Wrote {len(rows)} rows across {len(products)} products.")
    print(f"[DataGen] Products: {list(products.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMC Synthetic Data Generator")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", default="historical_data.csv")
    args = parser.parse_args()

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(__file__), out)

    generate(args.steps, out)


if __name__ == "__main__":
    main()
