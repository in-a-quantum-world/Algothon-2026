"""
data_loader.py
--------------
Flexible CSV loader for financial ML datasets.
Handles any column naming — nothing is hardcoded.
Adapt DATE_COL, ID_COL, TARGET_COL at the top of main.py once you see the data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# Column-name heuristics (override in main.py)
# ──────────────────────────────────────────────

DATE_KEYWORDS   = ["date", "time", "datetime", "period", "month", "year", "day"]
ID_KEYWORDS     = ["stock", "ticker", "id", "symbol", "asset", "code", "name", "company"]
TARGET_KEYWORDS = ["fwd_return", "forward_return", "fwd_ret", "return", "ret",
                   "target", "label", "forward", "fwd", "excess", "y_true"]
SECTOR_KEYWORDS = ["sector", "industry", "group", "category", "gics"]


def load_csv(path: str, **read_kwargs) -> pd.DataFrame:
    """Load any CSV into a DataFrame. Passes kwargs to pd.read_csv."""
    df = pd.read_csv(path, **read_kwargs)
    print(f"[load_csv] Loaded {path}  →  shape {df.shape}")
    return df


def infer_columns(df: pd.DataFrame,
                  date_col=None, id_col=None,
                  target_col=None, sector_col=None,
                  verbose=True) -> dict:
    """
    Best-effort inference of special columns.
    Returns dict with keys: date, id, target, sector (any can be None).
    Pass explicit names to skip inference for that column.
    """
    cols_lower = {c: c.lower() for c in df.columns}

    def _find(keywords, override):
        if override and override in df.columns:
            return override
        # Iterate keywords in priority order (longer/more specific first)
        # so "fwd_return" beats "fwd" and "return"
        for kw in sorted(keywords, key=len, reverse=True):
            for col, col_l in cols_lower.items():
                # Whole-token match: kw must be surrounded by start/end or _/-
                import re
                if re.search(r'(^|[_\-])' + re.escape(kw) + r'([_\-]|$)', col_l):
                    return col
                # Fallback: substring match only for longer keywords (≥4 chars)
                if len(kw) >= 4 and kw in col_l:
                    return col
        return None

    result = {
        "date":   _find(DATE_KEYWORDS,   date_col),
        "id":     _find(ID_KEYWORDS,     id_col),
        "target": _find(TARGET_KEYWORDS, target_col),
        "sector": _find(SECTOR_KEYWORDS, sector_col),
    }

    if verbose:
        print("[infer_columns]")
        for k, v in result.items():
            print(f"  {k:8s} → {v}")

    return result


def summarise(df: pd.DataFrame, col_map: dict, save_path="data_summary.txt") -> str:
    """
    Print and save a human-readable data summary.
    Returns the summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DATA SUMMARY")
    lines.append("=" * 60)

    lines.append(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    lines.append("\n--- Column map ---")
    for k, v in col_map.items():
        lines.append(f"  {k:8s} → {v}")

    lines.append("\n--- dtypes ---")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col:<35s} {str(dtype)}")

    lines.append("\n--- Missing values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("  (none)")
    else:
        for col, n in missing.items():
            lines.append(f"  {col:<35s} {n:,}  ({100*n/len(df):.1f}%)")

    lines.append("\n--- Numeric describe ---")
    lines.append(df.describe().to_string())

    # Date range if we have a date column
    dc = col_map.get("date")
    if dc and dc in df.columns:
        lines.append(f"\n--- Date range ---")
        try:
            dates = pd.to_datetime(df[dc])
            lines.append(f"  {dates.min()}  →  {dates.max()}")
            lines.append(f"  Unique dates: {dates.nunique():,}")
        except Exception:
            lines.append("  (could not parse as dates)")

    # Target distribution
    tc = col_map.get("target")
    if tc and tc in df.columns:
        lines.append(f"\n--- Target ({tc}) ---")
        t = df[tc].dropna()
        lines.append(f"  unique values : {t.nunique()}")
        lines.append(f"  mean / std    : {t.mean():.4f} / {t.std():.4f}")
        lines.append(f"  min / max     : {t.min():.4f} / {t.max():.4f}")
        # Are we doing classification or regression?
        task = "classification" if t.nunique() <= 20 else "regression"
        lines.append(f"  inferred task : {task}")

    summary = "\n".join(lines)
    print(summary)

    if save_path:
        Path(save_path).write_text(summary)
        print(f"\n[summarise] Saved → {save_path}")

    return summary


def train_test_split(df: pd.DataFrame,
                     date_col=None,
                     test_frac: float = 0.2,
                     test_start_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / test.

    Strategy:
      1. If test_start_date given  → everything before is train, after is test.
      2. If date_col exists        → split by unique dates (last test_frac of dates).
      3. Otherwise                 → row-index split.

    Returns (train_df, test_df).
    """
    if test_start_date is not None and date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        cutoff = pd.to_datetime(test_start_date)
        train = df[dates < cutoff].copy()
        test  = df[dates >= cutoff].copy()
        print(f"[split] Date cutoff {cutoff.date()}: "
              f"train {len(train):,} / test {len(test):,}")
        return train, test

    if date_col and date_col in df.columns:
        unique_dates = sorted(df[date_col].unique())
        n_test = max(1, int(len(unique_dates) * test_frac))
        test_dates = set(unique_dates[-n_test:])
        train = df[~df[date_col].isin(test_dates)].copy()
        test  = df[ df[date_col].isin(test_dates)].copy()
        print(f"[split] By date: train {len(train):,} rows "
              f"({len(unique_dates)-n_test} dates) / "
              f"test {len(test):,} rows ({n_test} dates)")
        return train, test

    # Fallback: row-index split
    n = len(df)
    cut = int(n * (1 - test_frac))
    train = df.iloc[:cut].copy()
    test  = df.iloc[cut:].copy()
    print(f"[split] By row index: train {len(train):,} / test {len(test):,}")
    return train, test


def get_feature_cols(df: pd.DataFrame, col_map: dict,
                     exclude_extra: list = None) -> list:
    """
    Return all numeric columns that are NOT the special columns.
    Useful for quickly getting X columns.
    """
    special = set(v for v in col_map.values() if v is not None)
    if exclude_extra:
        special.update(exclude_extra)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in special]
    return feature_cols


# ──────────────────────────────────────────────
# Quick sanity-check when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data.csv"
    df = load_csv(path)
    col_map = infer_columns(df)
    summarise(df, col_map)
    train, test = train_test_split(df, date_col=col_map["date"])
    feat_cols = get_feature_cols(df, col_map)
    print(f"\nFeature columns ({len(feat_cols)}): {feat_cols}")
