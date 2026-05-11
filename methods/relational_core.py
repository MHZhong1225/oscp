"""Shared relational data structures, selection rules, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    """Map label probabilities to action support scores, shape [n, M]."""
    return np.asarray(probs, dtype=float) @ np.asarray(relation, dtype=float)


def complete_batches(n: int, batch_size: int) -> list[np.ndarray]:
    usable = (n // batch_size) * batch_size
    return [np.arange(start, start + batch_size) for start in range(0, usable, batch_size)]


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
    relation = np.asarray(relation, dtype=int)
    _, inverse = np.unique(relation, axis=0, return_inverse=True)
    ambiguities = []
    for pred_set in np.asarray(label_sets, dtype=bool):
        labels = np.where(pred_set)[0]
        ambiguities.append(len(set(inverse[labels].tolist())))
    return float(np.mean(ambiguities))


def _action_union_size(label_sets: np.ndarray, relation: np.ndarray) -> float:
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
        "avg_size": float(np.mean(patient_sizes)),
        "decision_ambiguity": _decision_ambiguity(patient_matrix, relation),
        "action_union_size": _action_union_size(patient_matrix, relation),
        "edge_cov": float(np.mean(covered)),
        "edge_size": float(np.mean(sizes)),
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
