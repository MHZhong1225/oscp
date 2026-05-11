"""Our relational OSCP methods."""

from __future__ import annotations

from typing import Callable

import numpy as np

from methods.conformal import all_label_scores, conformal_quantile, label_scores
from methods.relational_core import (
    EdgeSelection,
    EdgeSetResult,
    RelationalSelectionConfig,
    complete_batches,
    reference_mask_top_capacity,
    top_edge_set_for_batch,
)


def relational_oscp_top(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_support: np.ndarray,
    test_probs: np.ndarray,
    test_support: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    score: str = "lac",
    method: str = "OSCP",
) -> EdgeSetResult:
    """Closed-form OSCP for per-action top-capacity selection."""
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    edge_scores = all_label_scores(test_probs[selection.unit_indices], score=score)
    thresholds = np.empty(selection.n_edges, dtype=float)
    ref_sizes = np.empty(selection.n_edges, dtype=int)

    batches = complete_batches(test_support.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)

    for e, (j, a, batch_id) in enumerate(
        zip(selection.unit_indices, selection.action_indices, selection.batch_indices)
    ):
        local_j = int(j - batch_start[batch_id])
        batch = batches[batch_id]
        mask = reference_mask_top_capacity(
            cal_support[:, a],
            test_support[batch, a],
            local_j,
            int(config.capacities[a]),
        )
        ref_sizes[e] = int(np.sum(mask))
        thresholds[e] = conformal_quantile(cal_scores[mask], alpha)

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=ref_sizes,
        method=method,
    )


def relational_swap_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_support: np.ndarray,
    test_probs: np.ndarray,
    test_support: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    selection_rule: Callable[[np.ndarray], set[tuple[int, int]]] | None = None,
    score: str = "lac",
    method: str = "Generic swap sc CP",
) -> EdgeSetResult:
    """Generic replacement OSCP implementation for arbitrary edge selection rules."""
    if selection_rule is None:
        selection_rule = lambda batch_support: top_edge_set_for_batch(
            batch_support, config.capacities
        )

    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    edge_scores = all_label_scores(test_probs[selection.unit_indices], score=score)
    thresholds = np.empty(selection.n_edges, dtype=float)
    ref_sizes = np.empty(selection.n_edges, dtype=int)

    batches = complete_batches(test_support.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)

    for e, (j, a, batch_id) in enumerate(
        zip(selection.unit_indices, selection.action_indices, selection.batch_indices)
    ):
        local_j = int(j - batch_start[batch_id])
        batch = batches[batch_id]
        base_batch_support = test_support[batch].copy()
        mask = np.zeros(cal_scores.size, dtype=bool)
        for i in range(cal_scores.size):
            swapped = base_batch_support.copy()
            swapped[local_j] = cal_support[i]
            mask[i] = (local_j, int(a)) in selection_rule(swapped)
        ref_sizes[e] = int(np.sum(mask))
        thresholds[e] = conformal_quantile(cal_scores[mask], alpha)

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=ref_sizes,
        method=method,
    )


__all__ = ["relational_oscp_top", "relational_swap_cp"]
