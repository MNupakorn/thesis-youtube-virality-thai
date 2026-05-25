"""Multi-modal hybrid model (proposed extension).

Fuses:
1. Title text embedding from WangchanBERTa (CLS, frozen by default).
2. Thai sentiment probability vector (4-dim) + arousal/valence/polar.
3. Engineered title features (length, emoji, caps, lexicon scores).
4. Channel + temporal + video numerical features.

Trains two heads on the fused vector:
- MLP (PyTorch) trained with focal loss.
- LightGBM (often competitive + gives SHAP for free).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.sentiment import SENTIMENT_FEATURE_COLUMNS
from src.features.structural import STRUCTURED_FEATURE_COLUMNS
from src.utils import setup_logger

log = setup_logger("models.hybrid")


def _train_mode(module) -> None:
    getattr(module, "train")()


def _inference_mode(module) -> None:
    getattr(module, "eval")()


@dataclass
class HybridFeatureMatrix:
    X: np.ndarray
    feature_names: list[str]


def assemble_hybrid_features(
    df: pd.DataFrame,
    title_embeddings: np.ndarray,
    use_sentiment: bool = True,
    use_structured: bool = True,
    use_embedding: bool = True,
) -> HybridFeatureMatrix:
    parts: list[np.ndarray] = []
    names: list[str] = []

    if use_embedding:
        parts.append(title_embeddings.astype(np.float32))
        names.extend([f"emb_{i}" for i in range(title_embeddings.shape[1])])

    if use_sentiment:
        sent_cols = [c for c in SENTIMENT_FEATURE_COLUMNS if c in df.columns]
        if sent_cols:
            block = df[sent_cols].fillna(0.0).astype(np.float32).to_numpy()
            parts.append(block)
            names.extend(sent_cols)

    if use_structured:
        struct_cols = [c for c in STRUCTURED_FEATURE_COLUMNS if c in df.columns]
        block = df[struct_cols].fillna(0.0).astype(np.float32).to_numpy()
        parts.append(block)
        names.extend(struct_cols)

    X = np.concatenate(parts, axis=1)
    log.info(f"hybrid feature matrix: {X.shape},  blocks={len(parts)}")
    return HybridFeatureMatrix(X=X, feature_names=names)


class _MLP:
    """Tiny MLP classifier (PyTorch)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = "gelu",
    ):
        import torch.nn as nn

        layers: list = []
        last = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = hidden_dim
        layers.append(nn.Linear(last, 2))
        self.net = nn.Sequential(*layers)

    def parameters(self):
        return self.net.parameters()

    def to(self, device):
        self.net.to(device)
        return self

    def __call__(self, x):
        return self.net(x)


def train_mlp_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    class_weight_pos: float | None = None,
    device: str = "auto",
) -> dict:
    """Train the MLP fusion head with focal loss + early stopping on val ROC-AUC."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.transformer_finetune import FocalLoss
    from src.utils import get_device

    dev = get_device(device)
    log.info(f"hybrid MLP training on {dev}, in_dim={X_train.shape[1]}")

    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xva_t = torch.tensor(X_val, dtype=torch.float32, device=dev)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=cfg.get("batch_size", 128),
        shuffle=True,
        drop_last=False,
    )

    model = _MLP(
        in_dim=X_train.shape[1],
        hidden_dim=cfg["fusion"]["hidden_dim"],
        num_layers=cfg["fusion"]["num_layers"],
        dropout=cfg["fusion"]["dropout"],
        activation=cfg["fusion"]["activation"],
    ).to(dev)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    loss_fn = (
        FocalLoss(gamma=cfg.get("focal_gamma", 2.0), alpha=class_weight_pos)
        if cfg.get("loss") == "focal"
        else torch.nn.CrossEntropyLoss()
    )

    best_auc = -1.0
    best_state = None
    patience = cfg.get("early_stopping_patience", 5)
    waited = 0

    for epoch in range(cfg["epochs"]):
        _train_mode(model.net)
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

        _inference_mode(model.net)
        with torch.no_grad():
            logits_val = model(Xva_t)
            probs_val = torch.softmax(logits_val, dim=-1)[:, 1].cpu().numpy()
        auc = float(roc_auc_score(y_val, probs_val))
        log.info(f"epoch {epoch + 1}/{cfg['epochs']}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.net.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                log.info(f"early stop at epoch {epoch + 1} (best val_auc={best_auc:.4f})")
                break

    if best_state is not None:
        model.net.load_state_dict(best_state)
    _inference_mode(model.net)

    return {"model": model, "best_val_auc": best_auc}


def predict_mlp(model, X: np.ndarray, device: str = "auto") -> np.ndarray:
    import torch

    from src.utils import get_device

    dev = get_device(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    _inference_mode(model.net)
    with torch.no_grad():
        logits = model(Xt)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def train_gbm_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
):
    """Train a LightGBM head on the fused features (great for SHAP)."""
    import lightgbm as lgb

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": 42,
        "is_unbalance": True,
        **(params or {}),
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return clf


def save_hybrid_artifacts(out_dir: str | Path, **artifacts) -> None:
    import joblib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        joblib.dump(obj, out_dir / f"{name}.joblib")
    log.info(f"saved {list(artifacts)} -> {out_dir}")
