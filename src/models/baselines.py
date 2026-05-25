"""Baseline ML models: Logistic Regression, LightGBM, XGBoost on multiple feature sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.utils import setup_logger

log = setup_logger("models.baselines")


@dataclass
class BaselineResult:
    name: str
    feature_set: str
    model: Any
    y_pred_proba_val: np.ndarray
    y_pred_proba_test: np.ndarray


def _to_dense_float(X) -> np.ndarray:
    """Convert pandas nullable dtypes (Float64/Int64) to plain numpy float64."""
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X.astype("float64", errors="ignore").fillna(0.0).to_numpy(dtype=np.float64)
    arr = np.asarray(X)
    if arr.dtype == object:
        arr = np.asarray(arr, dtype=np.float64)
    if arr.dtype.kind in {"i", "u"}:
        return arr.astype(np.float64)
    return np.nan_to_num(arr.astype(np.float64), nan=0.0)


def _build_feature_matrix(
    feature_set: str,
    X_struct: np.ndarray | None,
    X_tfidf: csr_matrix | None,
    X_sent: np.ndarray | None,
) -> Any:
    parts = []
    if feature_set in ("structured", "structured_plus_tfidf", "structured_plus_sentiment", "all"):
        assert X_struct is not None, "X_struct required"
        parts.append(csr_matrix(_to_dense_float(X_struct)))
    if feature_set in ("tfidf", "structured_plus_tfidf", "all"):
        assert X_tfidf is not None, "X_tfidf required"
        parts.append(X_tfidf)
    if feature_set in ("structured_plus_sentiment", "all"):
        assert X_sent is not None, "X_sent required"
        parts.append(csr_matrix(_to_dense_float(X_sent)))
    if not parts:
        raise ValueError(f"unknown feature_set: {feature_set}")
    if len(parts) == 1:
        return parts[0]
    return hstack(parts).tocsr()


def train_logistic_regression(X_train, y_train, params: dict):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(**params)
    clf.fit(X_train, y_train)
    return clf


def train_lightgbm(X_train, y_train, params: dict, X_val=None, y_val=None):
    import lightgbm as lgb

    # LightGBM doesn't accept class_weight kw in the sklearn constructor consistently;
    # convert to sample weights for safety.
    sample_weight = None
    cw = params.pop("class_weight", None)
    if cw == "balanced":
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        w_pos = (n_pos + n_neg) / (2 * n_pos)
        w_neg = (n_pos + n_neg) / (2 * n_neg)
        sample_weight = np.where(y_train == 1, w_pos, w_neg)

    clf = lgb.LGBMClassifier(**params)
    if X_val is not None and y_val is not None:
        clf.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def train_xgboost(X_train, y_train, params: dict, X_val=None, y_val=None):
    import xgboost as xgb

    if params.get("scale_pos_weight") == "auto":
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        params = {**params, "scale_pos_weight": n_neg / n_pos}

    clf = xgb.XGBClassifier(eval_metric="auc", **params)
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    clf.fit(X_train, y_train, **fit_kwargs)
    return clf


def predict_proba_binary(model, X) -> np.ndarray:
    """Return P(y=1) as a 1-D array, robust to sklearn / lightgbm / xgboost APIs."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        return expit(model.decision_function(X))
    raise AttributeError(f"{type(model)} has neither predict_proba nor decision_function")


def save_model(model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)


__all__ = [
    "BaselineResult",
    "_build_feature_matrix",
    "train_logistic_regression",
    "train_lightgbm",
    "train_xgboost",
    "predict_proba_binary",
    "save_model",
    "load_model",
]
