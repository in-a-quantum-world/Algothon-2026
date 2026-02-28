"""
main.py
-------
Full pipeline runner. Entry point for the hackathon.

Usage:
    python main.py data.csv
    python main.py data.csv --target fwd_ret --date date --id ticker
    python main.py data.csv --task regression --test-frac 0.25

────────────────────────────────────────────────
QUICK ADAPTATION GUIDE (read this before the challenge starts)
────────────────────────────────────────────────
1. If column names are obvious → just run:  python main.py data.csv
   The pipeline auto-detects date / id / target / sector columns.

2. If auto-detection is wrong → override via CLI:
   python main.py data.csv --target ret_1d --date trading_date --id stock_id

3. If you want to skip features or certain models → edit the PIPELINE CONFIG
   section below (marked with 🔧).

4. Task is auto-detected (classification vs regression) from target cardinality.
   Force it with: --task classification

5. Tune models by editing their kwargs in models.py get_all_models().
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── local modules ──
from data_loader import (
    load_csv, infer_columns, summarise,
    train_test_split, get_feature_cols
)
from features import add_all_features
from models import get_all_models, detect_task
from evaluate import (
    full_evaluate, print_summary_table,
    plot_cumulative_pnl, plot_feature_importance,
    plot_ic_over_time, plot_confusion_matrix_chart,
    plot_pred_vs_actual, ic_by_period
)
from submit import make_submission


# ══════════════════════════════════════════════
# 🔧 PIPELINE CONFIG — edit these freely
# ══════════════════════════════════════════════

ROLLING_WINDOWS   = [5, 10, 20]          # rolling stat windows
LAG_PERIODS       = [1, 2, 3]            # lag feature periods
TEST_FRACTION     = 0.20                 # last 20% of dates for test
DO_CS_ZSCORE      = True                 # cross-sectional z-score
DO_ROLLING        = True                 # rolling mean/std/z
DO_LAGS           = True                 # lag features
DO_GROUP_DEMEAN   = True                 # sector demeaning
DO_WINSORISE      = True                 # winsorise outliers
DROP_NA_ROWS      = True                 # drop rows with NaN features
ANNUALISATION     = 252                  # trading days per year

# ══════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Imperial Algothon ML Pipeline")
    p.add_argument("csv_path", nargs="?", default="synthetic_data.csv",
                   help="Path to input CSV")
    p.add_argument("--target",   default=None, help="Target column name")
    p.add_argument("--date",     default=None, help="Date column name")
    p.add_argument("--id",       default=None, help="ID/stock column name")
    p.add_argument("--sector",   default=None, help="Sector/group column name")
    p.add_argument("--task",     default="auto",
                   choices=["auto", "classification", "regression"])
    p.add_argument("--test-frac", type=float, default=TEST_FRACTION)
    p.add_argument("--test-date", default=None,
                   help="Hard cutoff date for test split (YYYY-MM-DD)")
    p.add_argument("--no-features", action="store_true",
                   help="Skip feature engineering (use raw columns only)")
    p.add_argument("--submission", default="submission.csv")
    return p.parse_args()


def run_pipeline(csv_path, target_col=None, date_col=None, id_col=None,
                 sector_col=None, task="auto", test_frac=TEST_FRACTION,
                 test_date=None, no_features=False,
                 submission_path="submission.csv"):
    """
    Core pipeline. Can also be imported and called from a notebook.
    Returns (best_model, best_preds, test_df, col_map, feature_cols).
    """

    # ── 1. Load data ──────────────────────────
    print("\n" + "═"*60)
    print("STEP 1 · Load & Summarise")
    print("═"*60)
    df = load_csv(csv_path)
    col_map = infer_columns(df,
                            date_col=date_col,
                            id_col=id_col,
                            target_col=target_col,
                            sector_col=sector_col)

    summarise(df, col_map, save_path="data_summary.txt")

    _date_col   = col_map["date"]
    _id_col     = col_map["id"]
    _target_col = col_map["target"]
    _sector_col = col_map["sector"]

    if _target_col is None:
        print("\n[ERROR] Could not identify target column.")
        print("  Pass --target <col_name>  to specify it manually.")
        sys.exit(1)

    # ── 2. Feature engineering ─────────────────
    print("\n" + "═"*60)
    print("STEP 2 · Feature Engineering")
    print("═"*60)
    raw_feat_cols = get_feature_cols(df, col_map)
    print(f"Raw feature columns ({len(raw_feat_cols)}): {raw_feat_cols}")

    if no_features:
        df_feat = df.copy()
        print("[features] Skipped (--no-features flag)")
    else:
        df_feat = add_all_features(
            df, col_map, raw_feat_cols,
            rolling_windows=ROLLING_WINDOWS,
            lags=LAG_PERIODS,
            do_cs_zscore=DO_CS_ZSCORE,
            do_rolling=DO_ROLLING,
            do_lags=DO_LAGS,
            do_group_demean=DO_GROUP_DEMEAN,
            do_winsorise=DO_WINSORISE,
        )

    # ── 3. Train / test split ──────────────────
    print("\n" + "═"*60)
    print("STEP 3 · Train / Test Split")
    print("═"*60)
    train_df, test_df = train_test_split(
        df_feat,
        date_col=_date_col,
        test_frac=test_frac,
        test_start_date=test_date,
    )

    # ── 4. Prepare X, y ───────────────────────
    all_feat_cols = get_feature_cols(df_feat, col_map)
    print(f"\nTotal features after engineering: {len(all_feat_cols)}")

    # Drop rows where target or features are NaN
    train_df = train_df.dropna(subset=[_target_col])
    test_df  = test_df.dropna(subset=[_target_col])
    if DROP_NA_ROWS:
        train_df = train_df.dropna(subset=all_feat_cols)
        test_df  = test_df.dropna(subset=all_feat_cols)

    print(f"Train: {len(train_df):,} rows   Test: {len(test_df):,} rows")

    X_train = train_df[all_feat_cols].values
    y_train = train_df[_target_col].values
    X_test  = test_df[all_feat_cols].values
    y_test  = test_df[_target_col].values

    # ── 5. Fit & evaluate models ──────────────
    print("\n" + "═"*60)
    print("STEP 4 · Model Training & Evaluation")
    print("═"*60)

    # Resolve task once (all models share the same task)
    resolved_task = task
    if task == "auto":
        resolved_task = detect_task(pd.Series(y_train))

    models     = get_all_models(task=task)
    all_metrics   = []
    all_preds     = {}   # model_name → predictions on test set
    trained_models = {}

    for model in models:
        print(f"\n  ── {model.name} ──")
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            date_vals = (test_df[_date_col].values
                         if _date_col and _date_col in test_df.columns
                         else None)

            m = full_evaluate(model.name, preds, y_test,
                              task=resolved_task,
                              date_col_values=date_vals,
                              ann_factor=ANNUALISATION)
            all_metrics.append(m)
            all_preds[model.name]      = preds
            trained_models[model.name] = model
            print(f"    ✓ Sharpe={m['sharpe']:.3f}  IC={m['ic']:.3f}")
        except Exception as e:
            print(f"    ✗ {model.name} failed: {e}")

    # ── 6. Summary table ──────────────────────
    print("\n" + "═"*60)
    print("STEP 5 · Summary")
    print("═"*60)
    summary_df = print_summary_table(all_metrics)
    summary_df.to_csv("model_summary.csv")
    print("\n[main] Saved model_summary.csv")

    # ── 7. Charts ─────────────────────────────
    print("\n" + "═"*60)
    print("STEP 6 · Charts")
    print("═"*60)
    plot_cumulative_pnl(all_preds, y_test)

    for name, model in trained_models.items():
        imp = model.feature_importance(all_feat_cols)
        if imp is not None:
            plot_feature_importance(imp, name)

        if resolved_task == "classification":
            plot_confusion_matrix_chart(all_preds[name], y_test, name)
        else:
            plot_pred_vs_actual(np.array(all_preds[name]), y_test, name)

        if _date_col and _date_col in test_df.columns:
            ic_df = ic_by_period(all_preds[name], y_test,
                                  test_df[_date_col].values)
            plot_ic_over_time(ic_df, name)

    # ── 8. Pick best model & save submission ──
    print("\n" + "═"*60)
    print("STEP 7 · Submission")
    print("═"*60)

    # Rank by Sharpe (highest = best)
    best_name = max(all_metrics, key=lambda m: m.get("sharpe", -999))["model"]
    best_preds = all_preds[best_name]
    best_model = trained_models[best_name]
    print(f"Best model: {best_name}  (Sharpe={max(m['sharpe'] for m in all_metrics):.4f})")

    # Build submission DataFrame
    id_vals = (test_df[_id_col].values
               if _id_col and _id_col in test_df.columns
               else np.arange(len(test_df)))
    date_vals_sub = (test_df[_date_col].values
                     if _date_col and _date_col in test_df.columns
                     else None)

    make_submission(
        predictions=best_preds,
        id_values=id_vals,
        date_values=date_vals_sub,
        save_path=submission_path,
        model_name=best_name,
    )

    return best_model, best_preds, test_df, col_map, all_feat_cols


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Check the CSV exists; if not and it's the default, generate synthetic data
    if not Path(args.csv_path).exists():
        if args.csv_path == "synthetic_data.csv":
            print("[main] synthetic_data.csv not found — generating it now...")
            from make_synthetic import generate
            generate("synthetic_data.csv")
        else:
            print(f"[ERROR] File not found: {args.csv_path}")
            sys.exit(1)

    run_pipeline(
        csv_path=args.csv_path,
        target_col=args.target,
        date_col=args.date,
        id_col=args.id,
        sector_col=args.sector,
        task=args.task,
        test_frac=args.test_frac,
        test_date=args.test_date,
        no_features=args.no_features,
        submission_path=args.submission,
    )
