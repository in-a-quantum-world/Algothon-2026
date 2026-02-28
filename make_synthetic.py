"""
make_synthetic.py
-----------------
Generate synthetic cross-sectional return-prediction data.

Mimics a realistic factor-model setup:
  - N stocks × T dates panel
  - 5 alpha factors (momentum, value, quality, size, low_vol)
  - 4 sectors
  - Target: 1-period forward return (linear combination of factors + noise)

Column names deliberately chosen to test the auto-inference logic.
Run standalone: python make_synthetic.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate(save_path: str = "synthetic_data.csv",
             n_stocks: int = 80,
             n_dates: int = 120,
             seed: int = 42) -> pd.DataFrame:
    """
    Generate and save synthetic panel data.
    Returns the DataFrame.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Universe ────────────────────────────
    dates   = pd.bdate_range("2022-01-01", periods=n_dates, freq="W-FRI")
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]
    sectors = rng.choice(["Tech", "Finance", "Healthcare", "Energy"],
                          size=n_stocks)
    sector_map = dict(zip(tickers, sectors))

    # ── 2. Latent factor returns (daily, per stock) ─
    # Each factor has persistence (AR-1) so it's temporally correlated
    def ar1_panel(n, T, rho=0.85, sigma=1.0):
        """AR(1) panel: shape (n, T)"""
        X = np.zeros((n, T))
        X[:, 0] = rng.normal(0, sigma, n)
        for t in range(1, T):
            X[:, t] = rho * X[:, t-1] + rng.normal(0, sigma * np.sqrt(1-rho**2), n)
        return X

    momentum = ar1_panel(n_stocks, n_dates, rho=0.92)   # slow-moving
    value    = ar1_panel(n_stocks, n_dates, rho=0.80)
    quality  = ar1_panel(n_stocks, n_dates, rho=0.75)
    size     = rng.normal(0, 1, (n_stocks, 1)) * np.ones((1, n_dates))  # static
    low_vol  = ar1_panel(n_stocks, n_dates, rho=0.70)

    # ── 3. Forward return = factor loadings + noise ─
    # True "alpha" signal
    true_signal = (
        0.20 * momentum +
        0.15 * value    +
        0.10 * quality  -
        0.05 * size     +
        0.10 * low_vol
    )
    noise = rng.normal(0, 1.0, (n_stocks, n_dates))
    # Shift by 1 to get FORWARD return (target at time t = return from t to t+1)
    fwd_return = np.roll(true_signal + noise, -1, axis=1)
    fwd_return[:, -1] = np.nan   # last date has no forward return

    # ── 4. Build long-format DataFrame ──────────
    rows = []
    for ti, ticker in enumerate(tickers):
        for di, dt in enumerate(dates):
            rows.append({
                "date":        dt.strftime("%Y-%m-%d"),
                "stock_id":    ticker,
                "sector":      sector_map[ticker],
                "momentum":    round(momentum[ti, di], 6),
                "value":       round(value[ti, di], 6),
                "quality":     round(quality[ti, di], 6),
                "size":        round(size[ti, di], 6),
                "low_vol":     round(low_vol[ti, di], 6),
                "fwd_return":  round(fwd_return[ti, di], 6),
            })

    df = pd.DataFrame(rows)

    # Add a bit of noise to make it realistic
    for col in ["momentum", "value", "quality", "size", "low_vol"]:
        df[col] += rng.normal(0, 0.05, len(df))

    # Inject ~1% missing values per feature
    for col in ["momentum", "value", "quality"]:
        mask = rng.random(len(df)) < 0.01
        df.loc[mask, col] = np.nan

    df.to_csv(save_path, index=False)
    print(f"[make_synthetic] Generated {len(df):,} rows "
          f"({n_stocks} stocks × {n_dates} dates) → {save_path}")
    print(f"  Columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data.csv"
    generate(path)
