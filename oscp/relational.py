"""Relational operation-selection conditional conformal prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from oscp.conformal import (
    all_label_scores,
    conformal_quantile,
    label_scores,
)


@dataclass(frozen=True)
class RelationSpec:
    """Many-to-many relation between labels and downstream actions."""

    label_names: tuple[str, ...]
    action_names: tuple[str, ...]
    relation: np.ndarray

    def __post_init__(self) -> None:
        relation = np.asarray(self.relation, dtype=float)
        if relation.shape != (len(self.label_names), len(self.action_names)):
            raise ValueError("relation must have shape [n_labels, n_actions]")
        if np.any(relation < 0):
            raise ValueError("relation weights must be nonnegative")
        object.__setattr__(self, "relation", relation)


@dataclass(frozen=True)
class RelationalSelectionConfig:
    """Top-capacity edge selection for each action in complete batches."""

    batch_size: int
    capacities: np.ndarray

    def __post_init__(self) -> None:
        capacities = np.asarray(self.capacities, dtype=int)
        if capacities.ndim != 1:
            raise ValueError("capacities must be a 1D array")
        if np.any(capacities < 0):
            raise ValueError("capacities must be nonnegative")
        if np.any(capacities > self.batch_size):
            raise ValueError("capacities cannot exceed batch_size")
        object.__setattr__(self, "capacities", capacities)


@dataclass(frozen=True)
class EdgeSelection:
    """Selected unit-action edges from batch-level downstream selection."""

    unit_indices: np.ndarray
    action_indices: np.ndarray
    batch_indices: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(self.unit_indices.size)


@dataclass(frozen=True)
class EdgeSetResult:
    """Edge-specific conformal outputs."""

    selection: EdgeSelection
    sets: np.ndarray
    thresholds: np.ndarray
    reference_sizes: np.ndarray
    method: str


def compute_support_scores(probs: np.ndarray, relation: np.ndarray) -> np.ndarray:
    """Map label probabilities to action support scores, shape [n, M]. g(x)=p^​(x)R. """
    return np.asarray(probs, dtype=float) @ np.asarray(relation, dtype=float)


def action_wise_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    relation: np.ndarray,
    alpha: float,
    method: str = "Action-wise CP",
) -> EdgeSetResult:
    """Action-wise conformal baseline.

    This ablation conformalizes downstream action support scores rather than
    complete label sets or operation-selected calibration strata. For a
    calibration label y, the score is 1 minus the strongest supported action.
    Test action sets are expanded back to label sets by action overlap.
    """
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
    """SC-CP Venn-Abers quantile calibration for conformity-score thresholds.

    This mirrors the public SelfCalibratingConformal quantile workflow: a
    score-quantile predictor is calibrated with isotonic Venn-Abers quantile
    loss, then interpolated to new predictor values.
    """
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


def complete_batches(n: int, batch_size: int) -> list[np.ndarray]:
    usable = (n // batch_size) * batch_size
    return [
        np.arange(start, start + batch_size)
        for start in range(0, usable, batch_size)
    ]


def select_top_edges(
    support: np.ndarray,
    config: RelationalSelectionConfig,
) -> EdgeSelection:
    """Select top-capacity unit-action edges in every complete batch."""
    support = np.asarray(support, dtype=float)
    if support.shape[1] != config.capacities.size:
        raise ValueError("support width must match number of capacities")

    units: list[int] = []
    actions: list[int] = []
    batches: list[int] = []
    for batch_id, batch in enumerate(complete_batches(support.shape[0], config.batch_size)):
        batch_support = support[batch]
        for a, cap in enumerate(config.capacities):
            if cap == 0:
                continue
            order = np.argsort(batch_support[:, a], kind="mergesort")
            selected_local = order[-cap:]
            selected_global = batch[selected_local]
            units.extend(selected_global.tolist())
            actions.extend([a] * cap)
            batches.extend([batch_id] * cap)
    return EdgeSelection(
        unit_indices=np.asarray(units, dtype=int),
        action_indices=np.asarray(actions, dtype=int),
        batch_indices=np.asarray(batches, dtype=int),
    )


def top_edge_set_for_batch(
    batch_support: np.ndarray,
    capacities: np.ndarray,
) -> set[tuple[int, int]]:
    """Local unit-action edge set for a single batch."""
    edges: set[tuple[int, int]] = set()
    for a, cap in enumerate(capacities):
        if cap == 0:
            continue
        order = np.argsort(batch_support[:, a], kind="mergesort")
        for local_j in order[-cap:]:
            edges.add((int(local_j), int(a)))
    return edges


def reference_mask_top_capacity(
    cal_support_a: np.ndarray,
    batch_support_a: np.ndarray,
    local_j: int,
    capacity: int,
) -> np.ndarray:
    """Calibration points that would be selected after replacing unit j."""
    if capacity <= 0:
        return np.zeros(cal_support_a.size, dtype=bool)
    others = np.delete(batch_support_a, local_j)
    if others.size < capacity:
        threshold = -np.inf
    else:
        threshold = np.partition(others, others.size - capacity)[others.size - capacity]
    return cal_support_a >= threshold


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
    sets = edge_scores <= q
    return EdgeSetResult(
        selection=selection,
        sets=sets,
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
    """Self-Calibrating CP baseline with prediction-dependent thresholds.

    The score-quantile predictor is the model uncertainty ``1 - max_k p_k(x)``.
    SC-CP then calibrates the observed conformity scores against this predictor
    with Venn-Abers isotonic quantile calibration.
    """
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
        method=f"Bonferroni CP (alpha/{divisor})",
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
    method: str = "JOMI (unit-selected)",
) -> EdgeSetResult:
    """Original JOMI-style focal-unit baseline for relational selection.

    JOMI conditions on a focal test unit being selected. In the relational
    experiments, a patient can be selected through one or more action edges, so
    this baseline calibrates on the event that the unit would be selected by at
    least one action after replacement. The resulting unit-level set is reported
    on every selected edge of that unit for direct comparison with edge-level
    OSCP.
    """
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    edge_scores = all_label_scores(test_probs[selection.unit_indices], score=score)
    thresholds = np.empty(selection.n_edges, dtype=float)
    ref_sizes = np.empty(selection.n_edges, dtype=int)

    batches = complete_batches(test_support.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)

    mask_cache: dict[tuple[int, int], np.ndarray] = {}
    for e, (j, batch_id) in enumerate(
        zip(selection.unit_indices, selection.batch_indices)
    ):
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
    method: str = "Generic swap selection-conditional CP",
) -> EdgeSetResult:
    """Generic replacement implementation for arbitrary edge selection rules.

    The selection rule receives one batch support matrix, shape [m, M], and
    returns local selected edges as ``(local_unit_index, action_index)`` pairs.
    This is slower than closed-form top-capacity OSCP, but it is the right
    extension point for knapsack, fairness-constrained, or preliminary-set
    triggered operation selection.
    """
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


def relational_patient_union(result: EdgeSetResult) -> dict[int, np.ndarray]:
    """Union edge-specific sets for patients selected by one or more actions."""
    out: dict[int, np.ndarray] = {}
    for j, pred_set in zip(result.selection.unit_indices, result.sets):
        if int(j) not in out:
            out[int(j)] = pred_set.copy()
        else:
            out[int(j)] |= pred_set
    return out


def _decision_ambiguity(label_sets: np.ndarray, relation: np.ndarray) -> float:
    """Average number of distinct action signatures represented by label sets."""
    relation = np.asarray(relation, dtype=int)
    signatures, inverse = np.unique(relation, axis=0, return_inverse=True)
    del signatures
    ambiguities = []
    for pred_set in np.asarray(label_sets, dtype=bool):
        labels = np.where(pred_set)[0]
        ambiguities.append(len(set(inverse[labels].tolist())))
    return float(np.mean(ambiguities))


def _action_union_size(label_sets: np.ndarray, relation: np.ndarray) -> float:
    """Average number of downstream actions represented by label sets."""
    relation = np.asarray(relation, dtype=int)
    sizes = []
    for pred_set in np.asarray(label_sets, dtype=bool):
        labels = np.where(pred_set)[0]
        if labels.size == 0:
            sizes.append(0)
        else:
            sizes.append(int(relation[labels].max(axis=0).sum()))
    return float(np.mean(sizes))


def evaluate_edge_sets(
    result: EdgeSetResult,
    labels: np.ndarray,
    relation: np.ndarray,
    action_names: tuple[str, ...],
    nominal: float = 0.90,
    critical_label: int | None = None,
    critical_action: str | None = None,
) -> dict[str, float | str]:
    """Summarize patient-level CP metrics plus edge-level relational metrics."""
    y_edge = labels[result.selection.unit_indices]
    covered = result.sets[np.arange(result.selection.n_edges), y_edge]
    sizes = result.sets.sum(axis=1)
    patient_sets = relational_patient_union(result)
    patient_indices = np.array(sorted(patient_sets), dtype=int)
    patient_labels = labels[patient_indices]
    patient_matrix = np.stack([patient_sets[j] for j in patient_indices], axis=0)
    patient_covered = patient_matrix[np.arange(patient_indices.size), patient_labels]
    patient_sizes = patient_matrix.sum(axis=1)
    explains = np.array(
        [
            bool(np.any(pred_set & (relation[:, a] > 0)))
            for pred_set, a in zip(result.sets, result.selection.action_indices)
        ]
    )
    row: dict[str, float | str] = {
        "method": result.method,
        "coverage": float(np.mean(patient_covered)),
        "size": float(np.mean(patient_sizes)),
        "decision_ambiguity": _decision_ambiguity(patient_matrix, relation),
        "action_union_size": _action_union_size(patient_matrix, relation),
        "edge_cov": float(np.mean(covered)),
        "avg_size": float(np.mean(sizes)),
        "avg_ref_size": float(np.mean(result.reference_sizes)),
        "explain_rate": float(np.mean(explains)),
        "n_edges": int(result.selection.n_edges),
        "n_patients": int(patient_indices.size),
    }

    action_covs = []
    under_gaps = []
    for a, name in enumerate(action_names):
        mask = result.selection.action_indices == a
        if not np.any(mask):
            row[f"{name}_cov"] = np.nan
            row[f"{name}_size"] = np.nan
            row[f"{name}_ref"] = np.nan
            row[f"{name}_explain"] = np.nan
            continue
        cov = float(np.mean(covered[mask]))
        row[f"{name}_cov"] = cov
        row[f"{name}_size"] = float(np.mean(sizes[mask]))
        row[f"{name}_ref"] = float(np.mean(result.reference_sizes[mask]))
        row[f"{name}_explain"] = float(np.mean(explains[mask]))
        action_covs.append(cov)
        under_gaps.append(max(nominal - cov, 0.0))

    if action_covs:
        row["action_cov_gap"] = float(np.mean(np.abs(np.asarray(action_covs) - nominal)))
        row["worst_under_gap"] = float(np.max(under_gaps))

    if critical_label is not None and critical_action is not None:
        action_idx = action_names.index(critical_action)
        mask = (result.selection.action_indices == action_idx) & (y_edge == critical_label)
        row[f"{critical_action}_critical_n"] = int(np.sum(mask))
        row[f"{critical_action}_critical_miss_rate"] = (
            float(np.mean(~covered[mask])) if np.any(mask) else np.nan
        )
    return row


def aggregate_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat(
        [df.assign(seed=i) for i, df in enumerate(runs)],
        ignore_index=True,
    )
    numeric = all_rows.select_dtypes(include=[np.number]).columns.drop("seed")
    means = all_rows.groupby("method", sort=False)[numeric].mean()
    stds = all_rows.groupby("method", sort=False)[numeric].std(ddof=1).add_suffix("_std")
    return means.join(stds).reset_index()
