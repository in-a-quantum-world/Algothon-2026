"""
features.py
-----------
Modular feature engineering toolkit for financial time series.

All functions:
  - Accept explicit column names → no hardcoded assumptions
  - Return a DataFrame (or modified copy) so they can be chained
  - Are safe to call independently — pick what fits your data

Typical usage in main.py:
    from features import (
        cross_sectional_zscore, rolling_stats, lag_features, group_demean,
        add_all_features
    )
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


# ──────────────────────────────────────────────
# Cross-sectional normalisation
# ──────────────────────────────────────────────

def cross_sectional_zscore(df: pd.DataFrame,
                            date_col: str,
                            feature_cols: list,
                            suffix: str = "_csz") -> pd.DataFrame:
    """
    For each date, z-score normalise each feature_col across all stocks.
    Adds columns named  <col><suffix>.
    Works best for panel/cross-sectional data.
    """
    df = df.copy()

    def _zscore_row(x):
        mu, sigma = x.mean(), x.std()
        return (x - mu) / sigma if sigma > 1e-8 else x * 0.0

    grp = df.groupby(date_col)[feature_cols]
    normed = grp.transform(_zscore_row)
    normed.columns = [f"{c}{suffix}" for c in normed.columns]
    df = pd.concat([df, normed], axis=1)
    print(f"[cross_sectional_zscore] Added {len(normed.columns)} columns")
    return df


def cross_sectional_rank(df: pd.DataFrame,
                          date_col: str,
                          feature_cols: list,
                          suffix: str = "_csr") -> pd.DataFrame:
    """
    Convert features to cross-sectional percentile rank within each date.
    Robust to outliers. Adds columns named <col><suffix>.
    """
    df = df.copy()
    grp = df.groupby(date_col)[feature_cols]
    ranked = grp.transform(lambda x: x.rank(pct=True))
    ranked.columns = [f"{c}{suffix}" for c in ranked.columns]
    df = pd.concat([df, ranked], axis=1)
    print(f"[cross_sectional_rank] Added {len(ranked.columns)} columns")
    return df


# ──────────────────────────────────────────────
# Time-series / rolling features (per stock)
# ──────────────────────────────────────────────

def rolling_stats(df: pd.DataFrame,
                  feature_cols: list,
                  windows: list = [5, 10, 20],
                  id_col: str = None,
                  date_col: str = None,
                  include: list = ("mean", "std", "zscore")) -> pd.DataFrame:
    """
    Add rolling mean, std, and/or z-score for each feature_col × window.

    If id_col is given, rolling is computed within each stock separately.
    If date_col is given, df is sorted by it before rolling.

    include: subset of ("mean", "std", "zscore", "min", "max")
    """
    df = df.copy()

    # Sort so rolling windows are temporally correct
    sort_cols = [c for c in [id_col, date_col] if c is not None]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    new_cols = {}

    def _compute(group, col, w):
        rm  = group[col].rolling(w, min_periods=1)
        mn  = rm.mean()
        sd  = rm.std()
        results = {}
        if "mean" in include:
            results[f"{col}_roll{w}_mean"] = mn
        if "std" in include:
            results[f"{col}_roll{w}_std"] = sd
        if "zscore" in include:
            results[f"{col}_roll{w}_z"] = (group[col] - mn) / sd.replace(0, np.nan)
        if "min" in include:
            results[f"{col}_roll{w}_min"] = rm.min()
        if "max" in include:
            results[f"{col}_roll{w}_max"] = rm.max()
        return results

    if id_col and id_col in df.columns:
        for col in feature_cols:
            for w in windows:
                parts = df.groupby(id_col, group_keys=False).apply(
                    lambda g: pd.DataFrame(_compute(g, col, w), index=g.index)
                )
                new_cols.update({k: parts[k] for k in parts.columns})
    else:
        for col in feature_cols:
            for w in windows:
                new_cols.update(_compute(df, col, w))

    added = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, added], axis=1)
    print(f"[rolling_stats] Added {len(added.columns)} columns "
          f"(windows={windows})")
    return df


# ──────────────────────────────────────────────
# Lag features
# ──────────────────────────────────────────────

def lag_features(df: pd.DataFrame,
                 feature_cols: list,
                 lags: list = [1, 2, 3, 5],
                 id_col: str = None,
                 date_col: str = None) -> pd.DataFrame:
    """
    Add lagged versions of feature_cols.
    If id_col given, lags are per-stock (no bleed across stocks).
    """
    df = df.copy()
    sort_cols = [c for c in [id_col, date_col] if c is not None]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    new_cols = {}

    def _lag_one(group, col, lag):
        return group[col].shift(lag)

    if id_col and id_col in df.columns:
        for col in feature_cols:
            for lag in lags:
                key = f"{col}_lag{lag}"
                new_cols[key] = df.groupby(id_col, group_keys=False)[col].shift(lag)
    else:
        for col in feature_cols:
            for lag in lags:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)

    added = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, added], axis=1)
    print(f"[lag_features] Added {len(added.columns)} columns (lags={lags})")
    return df


# ──────────────────────────────────────────────
# Group demeaning (sector-neutral features)
# ──────────────────────────────────────────────

def group_demean(df: pd.DataFrame,
                 feature_cols: list,
                 group_col: str,
                 date_col: str = None,
                 suffix: str = "_gd") -> pd.DataFrame:
    """
    Subtract group (e.g. sector) mean from each feature, optionally within
    each date. This makes features sector-neutral.
    Adds columns named <col><suffix>.
    """
    df = df.copy()
    group_keys = [c for c in [date_col, group_col] if c is not None]

    def _demean(x):
        return x - x.mean()

    normed = df.groupby(group_keys)[feature_cols].transform(_demean)
    normed.columns = [f"{c}{suffix}" for c in normed.columns]
    df = pd.concat([df, normed], axis=1)
    print(f"[group_demean] Added {len(normed.columns)} columns "
          f"(group={group_col})")
    return df


# ──────────────────────────────────────────────
# Returns / momentum
# ──────────────────────────────────────────────

def price_to_returns(df: pd.DataFrame,
                     price_col: str,
                     id_col: str = None,
                     periods: list = [1, 5, 20],
                     log_returns: bool = True) -> pd.DataFrame:
    """
    Convert a price column into forward/backward returns over multiple periods.
    Adds <price_col>_ret<n> columns.
    """
    df = df.copy()
    new_cols = {}

    def _rets(series):
        result = {}
        for p in periods:
            if log_returns:
                result[f"{price_col}_ret{p}"] = np.log(series / series.shift(p))
            else:
                result[f"{price_col}_ret{p}"] = series.pct_change(p)
        return pd.DataFrame(result, index=series.index)

    if id_col and id_col in df.columns:
        parts = df.groupby(id_col, group_keys=False).apply(
            lambda g: _rets(g[price_col])
        )
        for col in parts.columns:
            new_cols[col] = parts[col]
    else:
        parts = _rets(df[price_col])
        for col in parts.columns:
            new_cols[col] = parts[col]

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(f"[price_to_returns] Added {len(new_cols)} return columns")
    return df


# ──────────────────────────────────────────────
# Volatility / risk features
# ──────────────────────────────────────────────

def realized_vol(df: pd.DataFrame,
                 return_col: str,
                 windows: list = [10, 20],
                 id_col: str = None) -> pd.DataFrame:
    """Rolling realised volatility (std of returns)."""
    df = df.copy()
    new_cols = {}
    for w in windows:
        key = f"{return_col}_rvol{w}"
        if id_col and id_col in df.columns:
            new_cols[key] = df.groupby(id_col, group_keys=False)[return_col] \
                              .transform(lambda x: x.rolling(w, min_periods=2).std())
        else:
            new_cols[key] = df[return_col].rolling(w, min_periods=2).std()
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(f"[realized_vol] Added {len(new_cols)} vol columns")
    return df


# ──────────────────────────────────────────────
# Interaction features
# ──────────────────────────────────────────────

def pairwise_ratios(df: pd.DataFrame,
                    col_a: str, col_b: str,
                    clip: float = 10.0) -> pd.DataFrame:
    """Add col_a / col_b ratio (clipped for stability)."""
    df = df.copy()
    name = f"{col_a}_over_{col_b}"
    df[name] = (df[col_a] / df[col_b].replace(0, np.nan)).clip(-clip, clip)
    return df


# ──────────────────────────────────────────────
# Winsorisation / outlier handling
# ──────────────────────────────────────────────

def winsorise(df: pd.DataFrame,
              feature_cols: list,
              lower: float = 0.01,
              upper: float = 0.99) -> pd.DataFrame:
    """Clip each feature to its [lower, upper] quantile. Modifies in-place copy."""
    df = df.copy()
    for col in feature_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    print(f"[winsorise] Winsorised {len(feature_cols)} columns "
          f"at [{lower:.0%}, {upper:.0%}]")
    return df


# ──────────────────────────────────────────────
# One-stop convenience wrapper
# ──────────────────────────────────────────────

def add_all_features(df: pd.DataFrame,
                     col_map: dict,
                     raw_feature_cols: list,
                     rolling_windows: list = [5, 10, 20],
                     lags: list = [1, 2, 3],
                     do_cs_zscore: bool = True,
                     do_rolling: bool = True,
                     do_lags: bool = True,
                     do_group_demean: bool = True,
                     do_winsorise: bool = True) -> pd.DataFrame:
    """
    Convenience function: run all applicable feature engineering steps
    based on which columns exist in col_map.

    Designed to be called once from main.py — but each sub-function can
    also be called independently for finer control.
    """
    date_col   = col_map.get("date")
    id_col     = col_map.get("id")
    sector_col = col_map.get("sector")

    if do_winsorise:
        df = winsorise(df, raw_feature_cols)

    if do_cs_zscore and date_col and id_col:
        df = cross_sectional_zscore(df, date_col, raw_feature_cols)
        df = cross_sectional_rank(df, date_col, raw_feature_cols)

    if do_rolling and raw_feature_cols:
        df = rolling_stats(df, raw_feature_cols, windows=rolling_windows,
                           id_col=id_col, date_col=date_col)

    if do_lags and raw_feature_cols:
        df = lag_features(df, raw_feature_cols, lags=lags,
                          id_col=id_col, date_col=date_col)

    if do_group_demean and sector_col and sector_col in df.columns:
        df = group_demean(df, raw_feature_cols, sector_col,
                          date_col=date_col)

    return df


# ──────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from data_loader import load_csv, infer_columns, get_feature_cols

    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data.csv"
    df = load_csv(path)
    col_map = infer_columns(df, verbose=False)
    feat_cols = get_feature_cols(df, col_map)

    print(f"Raw feature cols: {feat_cols}")
    df_feat = add_all_features(df, col_map, feat_cols)
    print(f"\nShape after feature engineering: {df_feat.shape}")
    print(df_feat.iloc[:3, -8:])
