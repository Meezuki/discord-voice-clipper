"""
Shared binary classification and KWS evaluation metrics:
Accuracy, Precision, Recall, F1, FRR, FAR, ROC-AUC, Confusion Matrix, and threshold calibration.
"""

import numpy as np


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes exact ROC-AUC via trapezoidal rule without external dependencies."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Computes confusion matrix and standard KWS metrics at a given threshold."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Rejection Rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate
    auc = compute_roc_auc(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "samples": total,
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "frr": round(frr, 4),
        "far": round(far, 4),
        "roc_auc": round(auc, 4),
    }


def calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: np.ndarray | None = None,
    target_thr: float = 0.75,
) -> float:
    """
    Selects the decision threshold that maximizes F1 score on validation data,
    breaking ties in favor of lower False Acceptance Rate (FAR) and conservative target_thr.
    """
    if candidates is None:
        candidates = np.linspace(0.30, 0.95, 66)

    best_thr = 0.5
    best_score = (-1.0, -1.0, -1.0)
    for thr in candidates:
        m = compute_binary_metrics(y_true, y_prob, float(thr))
        # Prefer high F1, then low FAR, then proximity to conservative target_thr
        score = (m["f1"], -m["far"], -abs(float(thr) - target_thr))
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return round(best_thr, 3)
