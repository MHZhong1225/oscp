"""Conformal prediction utilities for multiclass experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile.

    Returns the ceil((n + 1) * (1 - alpha)) / n empirical quantile with the
    usual conservative infinity fallback when the rank exceeds n.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.inf
    rank = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        return np.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least ambiguous class score: s(x, y) = 1 - p_y(x)."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    return 1.0 - probs[np.arange(y.size), y]


def deterministic_aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic APS score: sum of probabilities at least as large as p_y."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    true_probs = probs[np.arange(y.size), y]
    return np.sum(np.where(probs >= true_probs[:, None], probs, 0.0), axis=1)


def all_label_scores(probs: np.ndarray, score: str = "lac") -> np.ndarray:
    """Score for every candidate label, shape [n, K]."""
    probs = np.asarray(probs, dtype=float)
    if score == "lac":
        return 1.0 - probs
    if score == "aps":
        out = np.empty_like(probs)
        for y in range(probs.shape[1]):
            py = probs[:, y]
            out[:, y] = np.sum(np.where(probs >= py[:, None], probs, 0.0), axis=1)
        return out
    raise ValueError(f"unknown score: {score}")


def label_scores(probs: np.ndarray, y: np.ndarray, score: str = "lac") -> np.ndarray:
    """Calibration scores for observed labels."""
    if score == "lac":
        return lac_scores(probs, y)
    if score == "aps":
        return deterministic_aps_scores(probs, y)
    raise ValueError(f"unknown score: {score}")


def prediction_sets_from_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray | float,
    score: str = "lac",
) -> np.ndarray:
    """Return boolean conformal sets for the requested multiclass score."""
    probs = np.asarray(probs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(probs.shape[0], float(thresholds))
    return all_label_scores(probs, score=score) <= thresholds[:, None]


@dataclass(frozen=True)
class SetResult:
    """Prediction sets and the per-unit thresholds that generated them."""

    sets: np.ndarray
    thresholds: np.ndarray
    reference_sizes: np.ndarray


def sets_from_reference_masks(
    cal_scores: np.ndarray,
    test_probs: np.ndarray,
    reference_masks: np.ndarray,
    alpha: float,
    score: str = "lac",
) -> SetResult:
    """Build one conformal set per test unit from calibration reference masks."""
    n_test = test_probs.shape[0]
    thresholds = np.empty(n_test, dtype=float)
    reference_sizes = reference_masks.sum(axis=1).astype(int)
    for j in range(n_test):
        thresholds[j] = conformal_quantile(cal_scores[reference_masks[j]], alpha)
    sets = prediction_sets_from_thresholds(test_probs, thresholds, score=score)
    return SetResult(sets=sets, thresholds=thresholds, reference_sizes=reference_sizes)
