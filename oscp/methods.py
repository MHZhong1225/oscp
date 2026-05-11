"""Conformal methods and baselines for operation-selected evaluation."""

from __future__ import annotations

import numpy as np

from oscp.conformal import (
    SetResult,
    conformal_quantile,
    prediction_sets_from_thresholds,
    sets_from_reference_masks,
)
from oscp.selection import (
    OPERATIONS,
    SelectionConfig,
    apply_selection_to_single_batch,
    bottom_reference_mask_for_selected,
    split_batches,
    top_reference_mask_for_selected,
)


def marginal_cp(cal_scores: np.ndarray, test_probs: np.ndarray, alpha: float) -> SetResult:
    q = conformal_quantile(cal_scores, alpha)
    thresholds = np.full(test_probs.shape[0], q)
    sets = prediction_sets_from_thresholds(test_probs, thresholds)
    refs = np.full(test_probs.shape[0], cal_scores.size, dtype=int)
    return SetResult(sets=sets, thresholds=thresholds, reference_sizes=refs)


def bonferroni_cp(
    cal_scores: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
    divisor: int,
) -> SetResult:
    return marginal_cp(cal_scores, test_probs, alpha / divisor)


def mondrian_by_predicted_class(
    cal_scores: np.ndarray,
    cal_probs: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
) -> SetResult:
    cal_group = np.argmax(cal_probs, axis=1)
    test_group = np.argmax(test_probs, axis=1)
    masks = cal_group[None, :] == test_group[:, None]
    return sets_from_reference_masks(cal_scores, test_probs, masks, alpha)


def risk_bin_cp(
    cal_scores: np.ndarray,
    cal_risk: np.ndarray,
    test_probs: np.ndarray,
    test_risk: np.ndarray,
    alpha: float,
    n_bins: int = 10,
) -> SetResult:
    edges = np.quantile(cal_risk, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    cal_bin = np.searchsorted(edges[1:-1], cal_risk, side="right")
    test_bin = np.searchsorted(edges[1:-1], test_risk, side="right")
    masks = cal_bin[None, :] == test_bin[:, None]
    return sets_from_reference_masks(cal_scores, test_probs, masks, alpha)


def naive_action_wise_cp(
    cal_scores: np.ndarray,
    cal_risk: np.ndarray,
    test_probs: np.ndarray,
    test_ops: np.ndarray,
    alpha: float,
    config: SelectionConfig,
) -> SetResult:
    """Calibrate by operations selected inside calibration batches."""
    cal_ops = np.full(cal_risk.size, "unused", dtype=object)
    for batch in split_batches(cal_risk.size, config.batch_size):
        batch_ops = apply_selection_to_single_batch(cal_risk[batch], config)
        cal_ops[batch] = batch_ops

    masks = np.zeros((test_probs.shape[0], cal_scores.size), dtype=bool)
    for op in OPERATIONS:
        masks[test_ops == op] = cal_ops[None, :] == op
    unused = test_ops == "unused"
    if np.any(unused):
        masks[unused] = True
    return sets_from_reference_masks(cal_scores, test_probs, masks, alpha)


def oscp_top_bottom(
    cal_scores: np.ndarray,
    cal_risk: np.ndarray,
    test_probs: np.ndarray,
    test_risk: np.ndarray,
    test_ops: np.ndarray,
    alpha: float,
    config: SelectionConfig,
) -> SetResult:
    """Efficient OSCP for top-B urgent and bottom-B routine operations."""
    masks = np.zeros((test_probs.shape[0], cal_scores.size), dtype=bool)

    for batch in split_batches(test_risk.size, config.batch_size):
        batch_risk = test_risk[batch]
        for local_idx, global_idx in enumerate(batch):
            op = test_ops[global_idx]
            if op == "urgent":
                masks[global_idx] = top_reference_mask_for_selected(
                    cal_risk, batch_risk, local_idx, config.urgent_b
                )
            elif op == "routine":
                masks[global_idx] = bottom_reference_mask_for_selected(
                    cal_risk, batch_risk, local_idx, config.routine_b
                )
            elif op == "review":
                masks[global_idx] = _top_bottom_review_mask_for_unit(
                    cal_risk, batch_risk, local_idx, op, config
                )
    unused = test_ops == "unused"
    if np.any(unused):
        masks[unused] = True
    return sets_from_reference_masks(cal_scores, test_probs, masks, alpha)


def generic_selection_conditional_cp(
    cal_scores: np.ndarray,
    cal_risk: np.ndarray,
    test_probs: np.ndarray,
    test_risk: np.ndarray,
    test_ops: np.ndarray,
    alpha: float,
    config: SelectionConfig,
) -> SetResult:
    """Brute-force reference sets by replacing each selected test unit."""
    masks = np.zeros((test_probs.shape[0], cal_scores.size), dtype=bool)
    for batch in split_batches(test_risk.size, config.batch_size):
        batch_risk = test_risk[batch]
        for local_idx, global_idx in enumerate(batch):
            masks[global_idx] = _top_bottom_review_mask_for_unit(
                cal_risk, batch_risk, local_idx, test_ops[global_idx], config
            )
    unused = test_ops == "unused"
    if np.any(unused):
        masks[unused] = True
    return sets_from_reference_masks(cal_scores, test_probs, masks, alpha)


def jomi_selection_conditional_cp(
    cal_scores: np.ndarray,
    cal_risk: np.ndarray,
    test_probs: np.ndarray,
    test_risk: np.ndarray,
    test_ops: np.ndarray,
    alpha: float,
    config: SelectionConfig,
) -> SetResult:
    """JOMI baseline via swap-based selection-conditional reference sets."""
    return generic_selection_conditional_cp(
        cal_scores,
        cal_risk,
        test_probs,
        test_risk,
        test_ops,
        alpha,
        config,
    )


def _top_bottom_review_mask_for_unit(
    cal_risk: np.ndarray,
    batch_risk: np.ndarray,
    selected_local_index: int,
    operation: str,
    config: SelectionConfig,
) -> np.ndarray:
    """Vectorized replacement mask for the top/bottom/review selector."""
    if operation == "urgent":
        return top_reference_mask_for_selected(
            cal_risk, batch_risk, selected_local_index, config.urgent_b
        )
    if operation == "routine":
        return bottom_reference_mask_for_selected(
            cal_risk, batch_risk, selected_local_index, config.routine_b
        )
    if operation == "review":
        others = np.delete(batch_risk, selected_local_index)
        bottom_threshold = np.partition(others, config.routine_b - 1)[config.routine_b - 1]
        top_threshold = np.partition(others, others.size - config.urgent_b)[
            others.size - config.urgent_b
        ]
        return (cal_risk > bottom_threshold) & (cal_risk < top_threshold)
    return np.ones(cal_risk.size, dtype=bool)
