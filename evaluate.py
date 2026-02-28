"""
evaluate.py
-----------
Comprehensive evaluation for both classification and regression tasks.

Financial-specific metrics:
  - Sharpe ratio (treating predictions as position sizes)
  - Information Coefficient (IC) — Spearman rank correlation
  - IC by time period (ICIR)
  - Cumulative PnL chart

Standard ML metrics:
  - Classification: accuracy, F1, ROC-AUC, confusion matrix
  - Regression: RMSE, MAE, R²

All charts saved as PNGs in ./charts/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, mean_absolute_error, r2_score
)

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Core financial metrics
# ──────────────────────────────────────────────

def sharpe_ratio(predictions: np.ndarray,
                 actuals: np.ndarray,
                 ann_factor: float = 252) -> float:
    """
    Treat predictions as position sizes, actuals as next-period returns.
    PnL = prediction × actual return.
    Returns annualised Sharpe ratio.
    """
    pnl = np.array(predictions) * np.array(actuals)
    std = pnl.std()
    if std < 1e-10:
        return 0.0
    return float((pnl.mean() / std) * np.sqrt(ann_factor))


def information_coefficient(predictions: np.ndarray,
                             actuals: np.ndarray) -> float:
    """Spearman rank correlation between predictions and actuals."""
    p = np.array(predictions)
    a = np.array(actuals)
    mask = ~(np.isnan(p) | np.isnan(a))
    if mask.sum() < 3:
        return np.nan
    corr, _ = sp_stats.spearmanr(p[mask], a[mask])
    return float(corr)


def ic_by_period(predictions: np.ndarray,
                 actuals: np.ndarray,
                 period_labels: np.ndarray) -> pd.DataFrame:
    """
    Compute IC for each unique period (e.g. date or month).
    Returns a DataFrame with columns: [period, ic].
    """
    df = pd.DataFrame({
        "pred":   predictions,
        "actual": actuals,
        "period": period_labels,
    })
    results = []
    for period, grp in df.groupby("period"):
        if len(grp) < 3:
            continue
        ic = information_coefficient(grp["pred"].values, grp["actual"].values)
        results.append({"period": period, "ic": ic})
    return pd.DataFrame(results).set_index("period")


def icir(ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC). Measures IC stability."""
    ic = ic_series.dropna()
    if ic.std() < 1e-10:
        return 0.0
    return float(ic.mean() / ic.std())


# ──────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────

def classification_metrics(predictions: np.ndarray,
                            actuals: np.ndarray,
                            threshold: float = 0.5) -> dict:
    """
    predictions: raw probabilities (or scores)
    actuals:     true binary labels
    """
    y_true = np.array(actuals)
    y_prob = np.array(predictions)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan

    return metrics


# ──────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────

def regression_metrics(predictions: np.ndarray,
                        actuals: np.ndarray) -> dict:
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ──────────────────────────────────────────────
# Full evaluation (auto-detects task)
# ──────────────────────────────────────────────

def full_evaluate(model_name: str,
                  predictions: np.ndarray,
                  actuals: np.ndarray,
                  task: str,
                  date_col_values: np.ndarray = None,
                  ann_factor: float = 252) -> dict:
    """
    Run all metrics appropriate for the task.
    Returns a flat dict suitable for printing / DataFrame assembly.
    """
    metrics = {"model": model_name, "task": task}

    if task == "classification":
        metrics.update(classification_metrics(predictions, actuals))
    else:
        metrics.update(regression_metrics(predictions, actuals))

    # Financial metrics (always)
    metrics["sharpe"] = sharpe_ratio(predictions, actuals, ann_factor)
    metrics["ic"]     = information_coefficient(predictions, actuals)

    if date_col_values is not None:
        ic_df = ic_by_period(predictions, actuals, date_col_values)
        metrics["icir"]   = icir(ic_df["ic"])
        metrics["ic_mean"] = float(ic_df["ic"].mean())
        metrics["ic_std"]  = float(ic_df["ic"].std())

    return metrics


# ──────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────

def plot_cumulative_pnl(results_dict: dict,
                        actuals: np.ndarray,
                        save_name: str = "cumulative_pnl.png") -> str:
    """
    Plot cumulative PnL for each model.
    results_dict: {model_name: predictions_array}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    actuals = np.array(actuals)

    for name, preds in results_dict.items():
        pnl = np.array(preds) * actuals
        cum_pnl = np.cumsum(pnl)
        ax.plot(cum_pnl, label=name, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative PnL (predictions as position sizes)")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = str(CHART_DIR / save_name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
    return path


def plot_feature_importance(importance_series: pd.Series,
                             model_name: str,
                             top_n: int = 20,
                             save_name: str = None) -> str:
    """Bar chart of top-N feature importances."""
    if importance_series is None or importance_series.empty:
        return None

    top = importance_series.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    top[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")

    fname = save_name or f"feat_importance_{model_name.lower().replace(' ', '_')}.png"
    path = str(CHART_DIR / fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
    return path


def plot_ic_over_time(ic_df: pd.DataFrame,
                      model_name: str,
                      save_name: str = None) -> str:
    """Bar chart of IC by period."""
    if ic_df is None or ic_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["green" if v > 0 else "red" for v in ic_df["ic"]]
    ax.bar(range(len(ic_df)), ic_df["ic"].values, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(ic_df["ic"].mean(), color="navy", linewidth=1.2,
               linestyle="--", label=f"Mean IC = {ic_df['ic'].mean():.3f}")
    ax.set_xticks(range(0, len(ic_df), max(1, len(ic_df)//10)))
    ax.set_xticklabels(
        [str(ic_df.index[i]) for i in range(0, len(ic_df), max(1, len(ic_df)//10))],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_title(f"{model_name} — IC by Period")
    ax.set_ylabel("IC (Spearman)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fname = save_name or f"ic_over_time_{model_name.lower().replace(' ', '_')}.png"
    path = str(CHART_DIR / fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
    return path


def plot_confusion_matrix_chart(predictions: np.ndarray,
                                 actuals: np.ndarray,
                                 model_name: str,
                                 threshold: float = 0.5,
                                 save_name: str = None) -> str:
    """Confusion matrix heatmap for classification."""
    y_pred = (np.array(predictions) >= threshold).astype(int)
    cm = confusion_matrix(actuals, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} — Confusion Matrix")

    fname = save_name or f"confusion_{model_name.lower().replace(' ', '_')}.png"
    path = str(CHART_DIR / fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
    return path


def plot_pred_vs_actual(predictions: np.ndarray,
                         actuals: np.ndarray,
                         model_name: str,
                         save_name: str = None) -> str:
    """Scatter plot of predictions vs actuals for regression."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actuals, predictions, alpha=0.3, s=10, color="steelblue")
    lims = [min(actuals.min(), predictions.min()),
            max(actuals.max(), predictions.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} — Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = save_name or f"pred_vs_actual_{model_name.lower().replace(' ', '_')}.png"
    path = str(CHART_DIR / fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
    return path


# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────

def print_summary_table(all_metrics: list) -> pd.DataFrame:
    """
    Pretty-print a comparison table of all models.
    all_metrics: list of dicts from full_evaluate().
    Returns a DataFrame for saving.
    """
    df = pd.DataFrame(all_metrics)
    df = df.set_index("model")

    # Round numbers for display
    numeric = df.select_dtypes(include=[np.number])
    df[numeric.columns] = numeric.round(4)

    sep = "=" * 80
    print(f"\n{sep}")
    print("MODEL COMPARISON")
    print(sep)
    print(df.to_string())
    print(sep)
    return df


# ──────────────────────────────────────────────
# Quick test when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    actuals = np.random.randn(n)
    predictions = actuals * 0.4 + np.random.randn(n) * 0.9

    m = full_evaluate("TestModel", predictions, actuals, "regression")
    for k, v in m.items():
        print(f"  {k}: {v}")

    plot_pred_vs_actual(predictions, actuals, "TestModel")
    plot_cumulative_pnl({"TestModel": predictions}, actuals)
