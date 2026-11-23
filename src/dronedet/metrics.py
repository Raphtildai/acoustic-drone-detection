"""Threshold-dependent and threshold-free metrics, false-alarm budgets and causal smoothing."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def best_f1_threshold(y, p) -> float:
    """F1-optimal threshold on a 0.005 grid in [0.05, 0.95] (the paper's validation rule)."""
    y, p = np.asarray(y), np.asarray(p)
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t


def binary_metrics(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    two = len(np.unique(y)) > 1
    return {"threshold": float(threshold), "accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
            "auroc": float(roc_auc_score(y, p)) if two else float("nan"),
            "auprc": float(average_precision_score(y, p)) if two else float("nan"),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def smooth(t, p, k: int, step: float):
    """Causal moving average over the last k windows, restarted wherever the window start
    times `t` are not consecutive (gap != step). Usable in real time."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    if k <= 1:
        return p.copy()
    out, start = np.empty_like(p), 0
    for i in range(len(p)):
        if i and abs(t[i] - t[i - 1] - step) > 1e-3:
            start = i
        out[i] = p[max(start, i - k + 1): i + 1].mean()
    return out


def budget_threshold(y, p, budget_per_h: float, window_s: float) -> float:
    """Lowest threshold whose false-alarm rate on (y, p) is <= budget per hour of drone-free audio."""
    y, p = np.asarray(y), np.asarray(p)
    neg_h = np.sum(y == 0) * window_s / 3600
    for thr in np.unique(p):
        if np.sum((p >= thr) & (y == 0)) / neg_h <= budget_per_h:
            return float(thr)
    return float("inf")


def operating_point(y, p, thr: float, window_s: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    return {"threshold": float(thr), "recall": float(np.mean(pred[y == 1])) if np.any(y == 1) else float("nan"),
            "false_alarms_per_hour": float(np.sum(pred & (y == 0)) / max(np.sum(y == 0) * window_s / 3600, 1e-12)),
            "precision": float(np.sum(pred & (y == 1)) / max(np.sum(pred), 1))}
