"""Baseline methods for relational operation-selected conformal prediction."""

from __future__ import annotations

import numpy as np

from methods.conformal import all_label_scores, conformal_quantile, label_scores
from methods.relational_core import (
    EdgeSelection,
    EdgeSetResult,
    RelationalSelectionConfig,
    complete_batches,
    compute_support_scores,
    reference_mask_top_capacity,
)


def action_wise_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    relation: np.ndarray,
    alpha: float,
    method: str = "Action-wise CP",
) -> EdgeSetResult:
    """Conformalize downstream action support, then expand actions to labels."""
    relation = np.asarray(relation, dtype=int)
    cal_labels = np.asarray(cal_labels, dtype=int)
    cal_action_support = compute_support_scores(cal_probs, relation)
    true_action_mask = relation[cal_labels].astype(bool)
    supported_scores = np.where(true_action_mask, cal_action_support, -np.inf)
    cal_scores = 1.0 - np.max(supported_scores, axis=1)
    q = conformal_quantile(cal_scores, alpha)

    edge_probs = test_probs[selection.unit_indices]
    edge_action_support = compute_support_scores(edge_probs, relation)
    action_sets = (1.0 - edge_action_support) <= q
    label_sets = (action_sets.astype(int) @ relation.T.astype(int)) > 0
    return EdgeSetResult(
        selection=selection,
        sets=label_sets.astype(bool),
        thresholds=np.full(selection.n_edges, q),
        reference_sizes=np.full(selection.n_edges, cal_scores.size, dtype=int),
        method=method,
    )


def _weighted_quantile(values: np.ndarray, q: float) -> float:
    values = np.sort(np.asarray(values, dtype=float).ravel())
    if values.size == 0:
        raise ValueError("Cannot compute a quantile for an empty block.")
    index = int(np.ceil(q * values.size) - 1)
    index = max(0, min(values.size - 1, index))
    return float(values[index])


def _make_quantile_grid(values: np.ndarray, num_bin: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Cannot build a grid from empty values.")
    qs = np.linspace(0.0, 1.0, max(2, int(num_bin)))
    return np.unique(np.quantile(values, qs, method="inverted_cdf"))


def _make_dense_grid(values: np.ndarray, num_bin: int, padding: float = 0.05) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1e-8)
    return np.linspace(max(0.0, lo - padding * span), hi + padding * span, max(2, int(num_bin)))


def _isotonic_quantile_predict(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    quantile_level: float,
) -> np.ndarray:
    """PAVA quantile isotonic calibrator used by SC-CP."""
    order = np.argsort(x, kind="mergesort")
    x_sorted = np.asarray(x, dtype=float).ravel()[order]
    y_sorted = np.asarray(y, dtype=float).ravel()[order]
    blocks: list[dict[str, np.ndarray | float]] = []
    for x_value, y_value in zip(x_sorted, y_sorted):
        blocks.append(
            {
                "x_left": float(x_value),
                "x_right": float(x_value),
                "values": np.array([float(y_value)]),
                "prediction": float(_weighted_quantile(np.array([y_value]), quantile_level)),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["prediction"] > blocks[-1]["prediction"]:
            right = blocks.pop()
            left = blocks.pop()
            merged_values = np.concatenate([left["values"], right["values"]])
            blocks.append(
                {
                    "x_left": left["x_left"],
                    "x_right": right["x_right"],
                    "values": merged_values,
                    "prediction": float(_weighted_quantile(merged_values, quantile_level)),
                }
            )

    out = np.empty(np.asarray(x_new).size, dtype=float)
    for i, value in enumerate(np.asarray(x_new, dtype=float).ravel()):
        for block in blocks:
            if float(block["x_left"]) <= value <= float(block["x_right"]):
                out[i] = float(block["prediction"])
                break
        else:
            out[i] = float(blocks[0]["prediction"] if value < blocks[0]["x_left"] else blocks[-1]["prediction"])
    return out


def _histogram_quantile_predict(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    quantile_level: float,
    num_bin: int,
) -> np.ndarray:
    grid = _make_quantile_grid(x, num_bin)
    ids = np.searchsorted(grid, np.asarray(x, dtype=float), side="left")
    ids = np.clip(ids, 0, grid.size - 1)
    values = np.empty(grid.size, dtype=float)
    global_q = _weighted_quantile(y, quantile_level)
    for idx in range(grid.size):
        mask = ids == idx
        values[idx] = _weighted_quantile(np.asarray(y)[mask], quantile_level) if np.any(mask) else global_q
    new_ids = np.searchsorted(grid, np.asarray(x_new, dtype=float), side="left")
    new_ids = np.clip(new_ids, 0, grid.size - 1)
    return values[new_ids]


def _linear_extrapolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x.size == 1:
        return np.full(x_new.size, float(y[0]))
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    return np.interp(x_new, x, y, left=y[0], right=y[-1])


def self_calibrating_score_thresholds(
    cal_scores: np.ndarray,
    cal_predictor: np.ndarray,
    test_predictor: np.ndarray,
    alpha: float,
    num_bin_predictor: int = 15,
    num_bin_score: int = 60,
) -> np.ndarray:
    """SC-CP Venn-Abers quantile calibration for conformity-score thresholds."""
    cal_scores = np.asarray(cal_scores, dtype=float).ravel()
    cal_predictor = np.asarray(cal_predictor, dtype=float).ravel()
    test_predictor = np.asarray(test_predictor, dtype=float).ravel()
    quantile_level = 1.0 - alpha

    predictor_grid = _make_quantile_grid(cal_predictor, num_bin_predictor)
    score_grid = _make_dense_grid(cal_scores, num_bin_score)
    threshold_paths = np.zeros((predictor_grid.size, score_grid.size), dtype=float)

    for pred_index, pred in enumerate(predictor_grid):
        pred_augmented = np.concatenate([cal_predictor, np.array([pred])])
        for score_index, score_candidate in enumerate(score_grid):
            score_augmented = np.concatenate([cal_scores, np.array([score_candidate])])
            calibrated = _isotonic_quantile_predict(
                pred_augmented,
                score_augmented,
                np.array([pred]),
                quantile_level,
            )
            threshold_paths[pred_index, score_index] = calibrated[0]

    histogram_bins = min(10, max(4, num_bin_predictor // 2))
    baseline = _histogram_quantile_predict(
        cal_predictor,
        cal_scores,
        predictor_grid,
        quantile_level,
        histogram_bins,
    )
    venn_bounds = np.column_stack(
        [np.min(threshold_paths, axis=1), np.max(threshold_paths, axis=1)]
    )
    midpoints = np.mean(venn_bounds, axis=1)
    widths = venn_bounds[:, 1] - venn_bounds[:, 0]
    score_span = max(float(np.max(score_grid) - np.min(score_grid)), 1e-8)
    point_threshold = midpoints + widths / score_span * (baseline - midpoints)
    point_threshold = np.maximum(point_threshold, 0.0)
    return _linear_extrapolate(predictor_grid, point_threshold, test_predictor)


def relational_marginal_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    score: str = "lac",
    method: str = "Marginal CP",
) -> EdgeSetResult:
    """Apply one marginal conformal threshold to every selected edge."""
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    q = conformal_quantile(cal_scores, alpha)
    edge_scores = all_label_scores(test_probs[selection.unit_indices], score=score)
    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= q,
        thresholds=np.full(selection.n_edges, q),
        reference_sizes=np.full(selection.n_edges, cal_scores.size, dtype=int),
        method=method,
    )


def relational_self_calibrating_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    score: str = "lac",
    method: str = "SC-CP",
    num_bin_predictor: int = 15,
    num_bin_score: int = 60,
) -> EdgeSetResult:
    """Self-Calibrating CP baseline with prediction-dependent thresholds."""
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    cal_predictor = 1.0 - np.max(cal_probs, axis=1)
    edge_probs = test_probs[selection.unit_indices]
    edge_predictor = 1.0 - np.max(edge_probs, axis=1)
    thresholds = self_calibrating_score_thresholds(
        cal_scores,
        cal_predictor,
        edge_predictor,
        alpha,
        num_bin_predictor=num_bin_predictor,
        num_bin_score=num_bin_score,
    )
    edge_scores = all_label_scores(edge_probs, score=score)
    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=np.full(selection.n_edges, cal_scores.size, dtype=int),
        method=method,
    )


def relational_bonferroni_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    divisor: int,
    score: str = "lac",
) -> EdgeSetResult:
    return relational_marginal_cp(
        cal_probs,
        cal_labels,
        test_probs,
        selection,
        alpha / divisor,
        score=score,
        method=f"Bonferroni CP{divisor}",
    )


def relational_jomi_unit_top(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_support: np.ndarray,
    test_probs: np.ndarray,
    test_support: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    score: str = "lac",
    method: str = "JOMI",
) -> EdgeSetResult:
    """JOMI-style focal-unit baseline for relational selection."""
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    edge_scores = all_label_scores(test_probs[selection.unit_indices], score=score)
    thresholds = np.empty(selection.n_edges, dtype=float)
    ref_sizes = np.empty(selection.n_edges, dtype=int)

    batches = complete_batches(test_support.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)

    mask_cache: dict[tuple[int, int], np.ndarray] = {}
    for e, (j, batch_id) in enumerate(zip(selection.unit_indices, selection.batch_indices)):
        local_j = int(j - batch_start[batch_id])
        key = (int(batch_id), local_j)
        mask = mask_cache.get(key)
        if mask is None:
            batch = batches[batch_id]
            mask = np.zeros(cal_scores.size, dtype=bool)
            for a, cap in enumerate(config.capacities):
                if cap == 0:
                    continue
                mask |= reference_mask_top_capacity(
                    cal_support[:, a],
                    test_support[batch, a],
                    local_j,
                    int(cap),
                )
            mask_cache[key] = mask
        ref_sizes[e] = int(np.sum(mask))
        thresholds[e] = conformal_quantile(cal_scores[mask], alpha)

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=ref_sizes,
        method=method,
    )


__all__ = [
    "action_wise_cp",
    "relational_bonferroni_cp",
    "relational_jomi_unit_top",
    "relational_marginal_cp",
    "relational_self_calibrating_cp",
    "self_calibrating_score_thresholds",
]
