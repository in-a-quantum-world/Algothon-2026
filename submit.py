"""
submit.py
---------
Format predictions into a clean submission CSV.

Configurable column names — adapt to whatever format the challenge requires.

Usage from command line:
    python submit.py --preds submission.csv --id-col stock --pred-col prediction

Usage as a function:
    from submit import make_submission
    make_submission(predictions, id_values, date_values, save_path="final_sub.csv")
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def make_submission(predictions: np.ndarray,
                    id_values: np.ndarray,
                    date_values: np.ndarray = None,
                    save_path: str = "submission.csv",
                    model_name: str = "",
                    id_col: str = "id",
                    date_col: str = "date",
                    pred_col: str = "prediction",
                    rank_transform: bool = False,
                    clip_preds: bool = False,
                    clip_range: tuple = (-3.0, 3.0)) -> pd.DataFrame:
    """
    Build and save a submission DataFrame.

    Parameters
    ----------
    predictions   : model output (probabilities or regression values)
    id_values     : row identifiers (stock ID, etc.)
    date_values   : date for each row (optional, included if not None)
    save_path     : where to save the CSV
    model_name    : printed in console output for clarity
    id_col        : column name for IDs in the output CSV
    date_col      : column name for dates in the output CSV
    pred_col      : column name for predictions in the output CSV
    rank_transform: convert predictions to cross-sectional percentile ranks
                    (useful for submission formats that require ranks)
    clip_preds    : whether to clip predictions to clip_range
    clip_range    : (min, max) for clipping

    Returns
    -------
    submission DataFrame
    """
    preds = np.array(predictions, dtype=float)

    # Optional transforms before saving
    if clip_preds:
        preds = np.clip(preds, clip_range[0], clip_range[1])
        print(f"[submit] Clipped predictions to {clip_range}")

    if rank_transform:
        preds = _rank_transform(preds)
        print("[submit] Applied rank transform → [0, 1]")

    # Build DataFrame
    sub = pd.DataFrame({id_col: id_values})
    if date_values is not None:
        sub[date_col] = date_values
    sub[pred_col] = preds

    # Save
    sub.to_csv(save_path, index=False)

    print(f"\n[submit] {'Model: '+model_name+' | ' if model_name else ''}"
          f"Saved {len(sub):,} rows → {save_path}")
    print(f"  Columns  : {list(sub.columns)}")
    print(f"  Pred stats: "
          f"mean={preds.mean():.4f}  std={preds.std():.4f}  "
          f"min={preds.min():.4f}  max={preds.max():.4f}")
    print(sub.head(5).to_string(index=False))

    return sub


def _rank_transform(arr: np.ndarray) -> np.ndarray:
    """Convert array to percentile ranks in [0, 1]."""
    from scipy.stats import rankdata
    return rankdata(arr) / len(arr)


def reformat_submission(input_csv: str,
                         output_csv: str,
                         rename_map: dict = None,
                         drop_cols: list = None,
                         keep_cols: list = None) -> pd.DataFrame:
    """
    Load an existing submission CSV and reformat it.
    rename_map  : {old_name: new_name}
    drop_cols   : columns to remove
    keep_cols   : if provided, keep only these columns (after rename)
    """
    sub = pd.read_csv(input_csv)

    if rename_map:
        sub = sub.rename(columns=rename_map)

    if drop_cols:
        sub = sub.drop(columns=[c for c in drop_cols if c in sub.columns])

    if keep_cols:
        sub = sub[[c for c in keep_cols if c in sub.columns]]

    sub.to_csv(output_csv, index=False)
    print(f"[reformat] {input_csv} → {output_csv}  shape={sub.shape}")
    return sub


def validate_submission(sub_path: str,
                         expected_ids: np.ndarray = None,
                         pred_col: str = "prediction",
                         id_col: str = "id") -> bool:
    """
    Basic validation checks on a submission file.
    Returns True if all checks pass.
    """
    sub = pd.read_csv(sub_path)
    ok = True
    print(f"\n[validate] Checking {sub_path} ...")

    if pred_col not in sub.columns:
        print(f"  ✗ Missing prediction column: '{pred_col}'")
        ok = False
    else:
        n_nan = sub[pred_col].isna().sum()
        if n_nan > 0:
            print(f"  ✗ {n_nan} NaN predictions")
            ok = False
        else:
            print(f"  ✓ No NaN predictions")

    if expected_ids is not None and id_col in sub.columns:
        sub_ids = set(sub[id_col].tolist())
        exp_ids = set(expected_ids.tolist())
        missing = exp_ids - sub_ids
        extra   = sub_ids - exp_ids
        if missing:
            print(f"  ✗ Missing IDs: {len(missing)}")
            ok = False
        if extra:
            print(f"  ✗ Extra IDs: {len(extra)}")
            ok = False
        if not missing and not extra:
            print(f"  ✓ All {len(exp_ids)} IDs present")

    if ok:
        print("  ✓ Submission looks valid")
    return ok


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Submission formatter")
    p.add_argument("--preds",    default="submission.csv",
                   help="Input CSV with predictions")
    p.add_argument("--output",   default="final_submission.csv")
    p.add_argument("--id-col",   default="id")
    p.add_argument("--pred-col", default="prediction")
    p.add_argument("--date-col", default="date")
    p.add_argument("--rank",     action="store_true",
                   help="Convert predictions to percentile ranks")
    args = p.parse_args()

    sub = pd.read_csv(args.preds)
    preds = sub[args.pred_col].values

    id_col  = args.id_col  if args.id_col  in sub.columns else sub.columns[0]
    ids     = sub[id_col].values
    dates   = sub[args.date_col].values if args.date_col in sub.columns else None

    make_submission(
        predictions=preds,
        id_values=ids,
        date_values=dates,
        save_path=args.output,
        id_col=id_col,
        pred_col=args.pred_col,
        rank_transform=args.rank,
    )

    validate_submission(args.output, pred_col=args.pred_col, id_col=id_col)
