"""
models.py
---------
Consistent model interface for fast experimentation.

Every model exposes:
    .fit(X_train, y_train)
    .predict(X_test)           → raw predictions (probabilities for classif.)
    .predict_label(X_test)     → hard class labels (classif. only)
    .evaluate(X_test, y_test)  → dict of metrics
    .feature_importance(feat_names) → pd.Series or None

Task detection:
    Pass task="auto" (default) to detect classification vs regression from y.
    Or force task="classification" / task="regression".

Add models here as you discover what the challenge needs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Task detection
# ──────────────────────────────────────────────

CLASSIFICATION_THRESHOLD = 20   # If target has ≤ this many unique values → classif.

def detect_task(y: pd.Series, threshold: int = CLASSIFICATION_THRESHOLD) -> str:
    """Return 'classification' or 'regression'."""
    n_unique = y.nunique()
    task = "classification" if n_unique <= threshold else "regression"
    print(f"[detect_task] {n_unique} unique target values → {task}")
    return task


# ──────────────────────────────────────────────
# Base wrapper
# ──────────────────────────────────────────────

class BaseModel:
    """Common interface all models inherit."""

    name: str = "base"
    task: str = "auto"      # "auto", "classification", "regression"

    def __init__(self, task="auto", **kwargs):
        self.task = task
        self._fitted_task = None   # resolved after fit()
        self._model = None

    def _resolve_task(self, y):
        if self.task == "auto":
            return detect_task(pd.Series(y))
        return self.task

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        """
        Returns:
          - classification: probability of positive class (1D array)
          - regression:     continuous predictions
        """
        raise NotImplementedError

    def predict_label(self, X_test, threshold=0.5):
        """Hard labels for classification. Passthrough for regression."""
        preds = self.predict(X_test)
        if self._fitted_task == "classification":
            return (preds >= threshold).astype(int)
        return preds

    def feature_importance(self, feat_names=None) -> pd.Series:
        return None

    def evaluate(self, X_test, y_test) -> dict:
        """Quick scalar metrics. Full eval is in evaluate.py."""
        preds = self.predict(X_test)
        y = np.array(y_test)
        metrics = {}
        if self._fitted_task == "classification":
            labels = (preds >= 0.5).astype(int)
            metrics["accuracy"] = accuracy_score(y, labels)
            metrics["f1"]       = f1_score(y, labels, average="weighted", zero_division=0)
        else:
            metrics["rmse"] = np.sqrt(mean_squared_error(y, preds))
            metrics["mae"]  = mean_absolute_error(y, preds)
            metrics["r2"]   = r2_score(y, preds)
        # Sharpe of raw predictions as signal
        pnl = preds * y
        metrics["sharpe"] = _signal_sharpe(pnl)
        metrics["ic"]     = _information_coefficient(preds, y)
        return metrics


def _signal_sharpe(pnl: np.ndarray, ann_factor: float = 252) -> float:
    """Annualised Sharpe of a daily PnL series."""
    pnl = np.array(pnl)
    if pnl.std() < 1e-8:
        return 0.0
    return float((pnl.mean() / pnl.std()) * np.sqrt(ann_factor))


def _information_coefficient(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Spearman rank correlation between predictions and actuals."""
    from scipy.stats import spearmanr
    preds   = np.array(preds)
    actuals = np.array(actuals)
    mask    = ~(np.isnan(preds) | np.isnan(actuals))
    if mask.sum() < 3:
        return np.nan
    corr, _ = spearmanr(preds[mask], actuals[mask])
    return float(corr)


# ──────────────────────────────────────────────
# Baseline: mean / majority-class predictor
# ──────────────────────────────────────────────

class BaselineModel(BaseModel):
    name = "Baseline"

    def __init__(self, task="auto"):
        super().__init__(task=task)
        self._constant = None

    def fit(self, X_train, y_train):
        self._fitted_task = self._resolve_task(pd.Series(y_train))
        y = np.array(y_train)
        if self._fitted_task == "classification":
            # majority class
            vals, counts = np.unique(y, return_counts=True)
            self._constant = vals[np.argmax(counts)]
        else:
            self._constant = y.mean()
        print(f"[{self.name}] Constant prediction: {self._constant:.4f}")
        return self

    def predict(self, X_test):
        n = len(X_test)
        if self._fitted_task == "classification":
            # return probability = 1 if majority class is 1, else 0
            return np.full(n, float(self._constant == 1))
        return np.full(n, self._constant)


# ──────────────────────────────────────────────
# Random Forest
# ──────────────────────────────────────────────

class RandomForestModel(BaseModel):
    name = "RandomForest"

    def __init__(self, task="auto", n_estimators=200, max_depth=6,
                 n_jobs=-1, random_state=42, **kwargs):
        super().__init__(task=task)
        self._params = dict(n_estimators=n_estimators, max_depth=max_depth,
                            n_jobs=n_jobs, random_state=random_state, **kwargs)

    def fit(self, X_train, y_train):
        self._fitted_task = self._resolve_task(pd.Series(y_train))
        X = np.array(X_train)
        y = np.array(y_train)
        if self._fitted_task == "classification":
            self._model = RandomForestClassifier(**self._params)
        else:
            self._model = RandomForestRegressor(**self._params)
        self._model.fit(X, y)
        return self

    def predict(self, X_test):
        X = np.array(X_test)
        if self._fitted_task == "classification":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def feature_importance(self, feat_names=None):
        if self._model is None:
            return None
        imp = self._model.feature_importances_
        idx = feat_names if feat_names is not None else range(len(imp))
        return pd.Series(imp, index=idx).sort_values(ascending=False)


# ──────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────

class XGBoostModel(BaseModel):
    name = "XGBoost"

    def __init__(self, task="auto", n_estimators=300, max_depth=5,
                 learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, random_state=42, **kwargs):
        super().__init__(task=task)
        self._params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state, n_jobs=-1,
            eval_metric="logloss" if task == "classification" else "rmse",
            **kwargs
        )

    def fit(self, X_train, y_train):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("pip install xgboost")
        self._fitted_task = self._resolve_task(pd.Series(y_train))
        X = np.array(X_train)
        y = np.array(y_train)
        if self._fitted_task == "classification":
            self._params["objective"] = "binary:logistic"
            self._model = xgb.XGBClassifier(**self._params)
        else:
            self._params["objective"] = "reg:squarederror"
            self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(X, y, verbose=False)
        return self

    def predict(self, X_test):
        X = np.array(X_test)
        if self._fitted_task == "classification":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def feature_importance(self, feat_names=None):
        if self._model is None:
            return None
        imp = self._model.feature_importances_
        idx = feat_names if feat_names is not None else range(len(imp))
        return pd.Series(imp, index=idx).sort_values(ascending=False)


# ──────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────

class LightGBMModel(BaseModel):
    name = "LightGBM"

    def __init__(self, task="auto", n_estimators=300, max_depth=6,
                 learning_rate=0.05, num_leaves=31,
                 subsample=0.8, colsample_bytree=0.8,
                 random_state=42, **kwargs):
        super().__init__(task=task)
        self._params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, num_leaves=num_leaves,
            subsample=subsample, colsample_bytree=colsample_bytree,
            random_state=random_state, n_jobs=-1,
            verbose=-1,
            **kwargs
        )

    def fit(self, X_train, y_train):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("pip install lightgbm")
        self._fitted_task = self._resolve_task(pd.Series(y_train))
        X = np.array(X_train)
        y = np.array(y_train)
        if self._fitted_task == "classification":
            self._params["objective"] = "binary"
            self._model = lgb.LGBMClassifier(**self._params)
        else:
            self._params["objective"] = "regression"
            self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(X, y)
        return self

    def predict(self, X_test):
        X = np.array(X_test)
        if self._fitted_task == "classification":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def feature_importance(self, feat_names=None):
        if self._model is None:
            return None
        imp = self._model.feature_importances_
        idx = feat_names if feat_names is not None else range(len(imp))
        return pd.Series(imp, index=idx).sort_values(ascending=False)


# ──────────────────────────────────────────────
# Ridge / Lasso (handy for regression, fast)
# ──────────────────────────────────────────────

class RidgeModel(BaseModel):
    name = "Ridge"

    def __init__(self, task="regression", alpha=1.0):
        super().__init__(task=task)
        self._alpha = alpha

    def fit(self, X_train, y_train):
        from sklearn.linear_model import Ridge, LogisticRegression
        self._fitted_task = self._resolve_task(pd.Series(y_train))
        X = np.array(X_train)
        y = np.array(y_train)
        if self._fitted_task == "classification":
            self._model = LogisticRegression(C=1/self._alpha, max_iter=500)
        else:
            self._model = Ridge(alpha=self._alpha)
        self._model.fit(X, y)
        return self

    def predict(self, X_test):
        X = np.array(X_test)
        if self._fitted_task == "classification":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def feature_importance(self, feat_names=None):
        if self._model is None or not hasattr(self._model, "coef_"):
            return None
        coef = self._model.coef_.ravel()
        idx  = feat_names if feat_names is not None else range(len(coef))
        return pd.Series(np.abs(coef), index=idx).sort_values(ascending=False)


# ──────────────────────────────────────────────
# Model registry — easy to add more
# ──────────────────────────────────────────────

def get_all_models(task="auto") -> list:
    """Return one instance of every model, sharing the same task setting."""
    return [
        BaselineModel(task=task),
        RidgeModel(task=task),
        RandomForestModel(task=task),
        XGBoostModel(task=task),
        LightGBMModel(task=task),
    ]


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=20, noise=0.1)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)

    for m in get_all_models(task="regression"):
        m.fit(X[:400], y[:400])
        metrics = m.evaluate(X[400:], y[400:])
        print(f"{m.name:15s}  RMSE={metrics.get('rmse', '—'):.3f}  "
              f"Sharpe={metrics.get('sharpe', '—'):.3f}  "
              f"IC={metrics.get('ic', '—'):.3f}")
