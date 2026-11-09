"""Métricas de evaluación para clasificación multietiqueta."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Calcula métricas macro a partir de probabilidades.

    Args:
        y_true: array (N, C) binario.
        y_prob: array (N, C) de probabilidades en [0, 1].
        threshold: umbral para convertir probabilidades en etiquetas.

    Returns:
        dict con auroc, auprc, precision, recall, f1.
    """
    y_pred = (y_prob >= threshold).astype(int)
    results = {}
    for i in range(y_true.shape[1]):
        # Algunas clases pueden no tener positivos en un batch pequeño
        if len(np.unique(y_true[:, i])) < 2:
            continue
        results.setdefault("auroc", []).append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        results.setdefault("auprc", []).append(average_precision_score(y_true[:, i], y_prob[:, i]))
    return {
        "auroc": float(np.mean(results.get("auroc", [np.nan]))),
        "auprc": float(np.mean(results.get("auprc", [np.nan]))),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
