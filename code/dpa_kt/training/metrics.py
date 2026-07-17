"""Evaluation metrics computed on flattened, selectmasked predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """AUC / ACC / RMSE from probabilities and binary labels."""
    out: dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:  # only one class present
        out["auc"] = float("nan")
    out["acc"] = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_prob)))
    return out
