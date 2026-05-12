from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.conformal import all_label_scores, conformal_quantile, label_scores
from methods.relational_core import (
    EdgeSelection,
    EdgeSetResult,
    RelationalSelectionConfig,
    complete_batches,
    select_top_edges,
)
from methods.oscp.relational_oscp import relational_oscp_top


DEFAULT_LABELS = (
    "benign",
    "mild",
    "moderate",
    "severe",
    "critical",
)

DEFAULT_ACTIONS = (
    "cbc",
    "bmp",
    "lactate",
    "troponin",
)

SupportMode = Literal["oracle", "learned_hidden", "entropy_gain"]
BatchContextMode = Literal["none", "clustered"]


@dataclass(frozen=True)
class AcquisitionData:
    x_base: np.ndarray
    x_groups: np.ndarray
    y: np.ndarray
    best_action: np.ndarray
    hidden_action: np.ndarray
    route_hardness: np.ndarray


@dataclass(frozen=True)
class AcquisitionSplit:
    x_base_train: np.ndarray
    x_groups_train: np.ndarray
    y_train: np.ndarray
    best_action_train: np.ndarray
    hidden_action_train: np.ndarray
    route_hardness_train: np.ndarray

    x_base_val: np.ndarray
    x_groups_val: np.ndarray
    y_val: np.ndarray
    best_action_val: np.ndarray
    hidden_action_val: np.ndarray
    route_hardness_val: np.ndarray

    x_base_cal: np.ndarray
    x_groups_cal: np.ndarray
    y_cal: np.ndarray
    best_action_cal: np.ndarray
    hidden_action_cal: np.ndarray
    route_hardness_cal: np.ndarray

    x_base_test: np.ndarray
    x_groups_test: np.ndarray
    y_test: np.ndarray
    best_action_test: np.ndarray
    hidden_action_test: np.ndarray
    route_hardness_test: np.ndarray


@dataclass(frozen=True)
class AcquisitionModels:
    base_model: object
    action_models: tuple[object, ...]
    utility_models: tuple[object, ...]


@dataclass
class FeatureSubsetClassifier:
    model: object
    feature_indices: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x[:, self.feature_indices])


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def generate_synthetic_diagnostic_acquisition(
    n: int = 30000,
    seed: int = 0,
    d_base: int = 12,
    n_actions: int = 4,
    d_group: int = 4,
    n_classes: int = 5,
    noise_scale: float = 0.8,
    hidden_rate: float = 0.10,
    hidden_signal: float = 6.0,
    hidden_critical_prob: float = 0.85,
    base_feature_dim_for_generation: int = 4,
) -> AcquisitionData:
    """Generate a routed hidden-diagnostic synthetic dataset.

    The current label predictor should only see x_base[:, :base_feature_dim].
    Hidden high-severity mechanisms are determined from later routing features,
    so acquisition can find them but the current predictor cannot.
    """
    if n_actions != 4:
        raise ValueError("This synthetic generator currently expects exactly 4 actions.")
    if d_base < 12:
        raise ValueError("d_base must be at least 12.")
    if n_classes < 3:
        raise ValueError("n_classes must be at least 3.")

    rng = np.random.default_rng(seed)
    x_base = rng.normal(size=(n, d_base))
    x_groups = rng.normal(size=(n, n_actions, d_group))

    # Current-visible features generate ordinary label risk.
    visible = x_base[:, :base_feature_dim_for_generation]
    w_visible = rng.normal(scale=0.8, size=(visible.shape[1], n_classes))
    logits = visible @ w_visible

    # Routing features are deliberately outside the default current predictor.
    route_scores = np.column_stack(
        [
            x_base[:, 4] - 0.50 * x_base[:, 5],
            x_base[:, 6] + 0.50 * x_base[:, 7],
            -x_base[:, 8] + x_base[:, 9],
            x_base[:, 10] - x_base[:, 11],
        ]
    )
    winner = np.argmax(route_scores, axis=1)
    route_hardness = np.max(route_scores, axis=1)

    hidden_action = np.full(n, -1, dtype=int)
    threshold = np.quantile(route_hardness, 1.0 - hidden_rate)
    hidden = route_hardness >= threshold
    hidden_action[hidden] = winner[hidden]

    # The correct diagnostic action reveals a strong marker.
    for a in range(n_actions):
        mask = hidden_action == a
        if np.any(mask):
            x_groups[mask, a, 0] += hidden_signal
            if d_group > 1:
                x_groups[mask, a, 1] += 0.50 * hidden_signal

    # Non-hidden diagnostic features still contain ordinary information.
    for a in range(n_actions):
        w_group = rng.normal(scale=0.45, size=(d_group, n_classes))
        logits += 0.25 * (x_groups[:, a, :] @ w_group)

    logits += rng.normal(scale=noise_scale, size=logits.shape)
    probs = _softmax(logits)
    y = np.array([rng.choice(n_classes, p=p) for p in probs], dtype=int)

    severe_class = n_classes - 2
    critical_class = n_classes - 1
    draw = rng.random(n)
    y[hidden & (draw < hidden_critical_prob)] = critical_class
    y[hidden & (draw >= hidden_critical_prob)] = severe_class

    best_action = np.argmax(route_scores, axis=1).astype(int)
    best_action[hidden] = hidden_action[hidden]

    return AcquisitionData(
        x_base=x_base,
        x_groups=x_groups,
        y=y,
        best_action=best_action,
        hidden_action=hidden_action,
        route_hardness=route_hardness,
    )


def make_acquisition_splits(data: AcquisitionData, seed: int = 0) -> AcquisitionSplit:
    """Create 50/15/15/20 train/val/cal/test splits."""
    idx = np.arange(data.y.size)

    train_idx, rest_idx = train_test_split(
        idx,
        train_size=0.50,
        random_state=seed,
        stratify=data.y,
    )
    val_idx, rest_idx = train_test_split(
        rest_idx,
        train_size=0.30,
        random_state=seed + 1,
        stratify=data.y[rest_idx],
    )
    cal_idx, test_idx = train_test_split(
        rest_idx,
        train_size=3.0 / 7.0,
        random_state=seed + 2,
        stratify=data.y[rest_idx],
    )

    return AcquisitionSplit(
        x_base_train=data.x_base[train_idx],
        x_groups_train=data.x_groups[train_idx],
        y_train=data.y[train_idx],
        best_action_train=data.best_action[train_idx],
        hidden_action_train=data.hidden_action[train_idx],
        route_hardness_train=data.route_hardness[train_idx],
        x_base_val=data.x_base[val_idx],
        x_groups_val=data.x_groups[val_idx],
        y_val=data.y[val_idx],
        best_action_val=data.best_action[val_idx],
        hidden_action_val=data.hidden_action[val_idx],
        route_hardness_val=data.route_hardness[val_idx],
        x_base_cal=data.x_base[cal_idx],
        x_groups_cal=data.x_groups[cal_idx],
        y_cal=data.y[cal_idx],
        best_action_cal=data.best_action[cal_idx],
        hidden_action_cal=data.hidden_action[cal_idx],
        route_hardness_cal=data.route_hardness[cal_idx],
        x_base_test=data.x_base[test_idx],
        x_groups_test=data.x_groups[test_idx],
        y_test=data.y[test_idx],
        best_action_test=data.best_action[test_idx],
        hidden_action_test=data.hidden_action[test_idx],
        route_hardness_test=data.route_hardness[test_idx],
    )


def _fit_multiclass_logistic(x: np.ndarray, y: np.ndarray, seed: int = 0) -> object:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    model.fit(x, y)
    return model


def _concat_action_features(
    x_base: np.ndarray,
    x_groups: np.ndarray,
    action_index: int,
) -> np.ndarray:
    return np.concatenate([x_base, x_groups[:, action_index, :]], axis=1)


def fit_acquisition_models(
    split: AcquisitionSplit,
    seed: int = 0,
    entropy_weight: float = 0.2,
    critical_gain_weight: float = 4.0,
    base_feature_dim: int | None = 4,
    support_mode: SupportMode = "entropy_gain",
) -> AcquisitionModels:
    """Fit current predictor, post-action predictors, and support scorers."""
    n_actions = split.x_groups_train.shape[1]

    if base_feature_dim is None or base_feature_dim <= 0:
        feature_indices = np.arange(split.x_base_train.shape[1])
    else:
        feature_indices = np.arange(min(base_feature_dim, split.x_base_train.shape[1]))

    base_raw = _fit_multiclass_logistic(
        split.x_base_train[:, feature_indices],
        split.y_train,
        seed=seed,
    )
    base_model = FeatureSubsetClassifier(base_raw, feature_indices)

    base_train_probs = base_model.predict_proba(split.x_base_train)
    base_train_entropy = entropy(base_train_probs)
    critical_class = int(np.max(split.y_train))
    base_train_critical = base_train_probs[:, critical_class]

    action_models: list[object] = []
    realized_utilities = np.zeros((split.y_train.size, n_actions), dtype=float)

    for a in range(n_actions):
        x_train_a = _concat_action_features(split.x_base_train, split.x_groups_train, a)
        action_model = _fit_multiclass_logistic(
            x_train_a,
            split.y_train,
            seed=seed + 100 + a,
        )
        action_models.append(action_model)

        action_train_probs = action_model.predict_proba(x_train_a)
        action_train_critical = action_train_probs[:, critical_class]

        realized_utilities[:, a] = (
            entropy_weight * (base_train_entropy - entropy(action_train_probs))
            + critical_gain_weight
            * np.maximum(action_train_critical - base_train_critical, 0.0)
        )

    utility_models: list[object] = []
    for a in range(n_actions):
        reg = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=seed + 200 + a,
            n_jobs=-1,
        )

        if support_mode == "learned_hidden":
            target = (split.hidden_action_train == a).astype(float)
        elif support_mode in {"oracle", "entropy_gain"}:
            target = realized_utilities[:, a]
        else:
            raise ValueError(f"unknown support_mode: {support_mode}")

        reg.fit(split.x_base_train, target)
        utility_models.append(reg)

    return AcquisitionModels(
        base_model=base_model,
        action_models=tuple(action_models),
        utility_models=tuple(utility_models),
    )


def acquisition_support_scores(
    x_base: np.ndarray,
    utility_models: Sequence[object],
    mode: SupportMode = "entropy_gain",
    hidden_action: np.ndarray | None = None,
    n_actions: int | None = None,
    seed: int = 0,
    oracle_boost: float = 1.0,
    oracle_noise: float = 0.02,
    clip_negative: bool = True,
) -> np.ndarray:
    """Compute acquisition support scores g_a(x_base)."""
    if mode == "oracle":
        if hidden_action is None or n_actions is None:
            raise ValueError("oracle mode requires hidden_action and n_actions.")

        rng = np.random.default_rng(seed)
        scores = oracle_noise * rng.normal(size=(x_base.shape[0], n_actions))
        for a in range(n_actions):
            scores[hidden_action == a, a] += oracle_boost
        return scores

    if mode not in {"learned_hidden", "entropy_gain"}:
        raise ValueError(f"unknown support mode: {mode}")

    scores = np.column_stack([m.predict(x_base) for m in utility_models])
    if clip_negative:
        scores = np.maximum(scores, 0.0)
    return scores


def reorder_for_batch_context(
    arrays: dict[str, np.ndarray],
    key: np.ndarray,
    mode: BatchContextMode = "none",
) -> dict[str, np.ndarray]:
    """Reorder cal/test arrays before batching.

    clustered mode groups examples with similar routing hardness together. This
    makes realized top-B thresholds vary across batches, which is useful for
    evaluating conditional coverage within hard selected-edge strata.
    """
    if mode == "none":
        return arrays
    if mode != "clustered":
        raise ValueError(f"unknown batch_context mode: {mode}")

    order = np.argsort(key, kind="mergesort")
    return {name: value[order] for name, value in arrays.items()}


def edge_selection_thresholds(
    support: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
) -> np.ndarray:
    """Threshold each selected edge had to beat in its realized batch.

    This mirrors the closed-form OSCP reference-set threshold, but returns the
    numeric realized threshold for diagnostics / stratified evaluation.
    """
    support = np.asarray(support, dtype=float)
    thresholds = np.empty(selection.n_edges, dtype=float)
    batches = complete_batches(support.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)

    for e, (j, a, batch_id) in enumerate(
        zip(selection.unit_indices, selection.action_indices, selection.batch_indices)
    ):
        local_j = int(j - batch_start[batch_id])
        batch = batches[int(batch_id)]
        values = np.delete(support[batch, int(a)], local_j)
        capacity = int(config.capacities[int(a)])
        if values.size < capacity:
            thresholds[e] = -np.inf
        else:
            thresholds[e] = np.partition(values, values.size - capacity)[
                values.size - capacity
            ]
    return thresholds


def marginal_edge_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    score: str = "lac",
    method: str = "Marginal CP",
) -> EdgeSetResult:
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    q = conformal_quantile(cal_scores, alpha)

    edge_probs = test_probs[selection.unit_indices]
    edge_scores = all_label_scores(edge_probs, score=score)

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= q,
        thresholds=np.full(selection.n_edges, q),
        reference_sizes=np.full(selection.n_edges, cal_scores.size, dtype=int),
        method=method,
    )


def actionwise_selected_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_support: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    score: str = "lac",
    method: str = "Action-wise CP",
) -> EdgeSetResult:
    """One threshold per diagnostic action selected from calibration batches."""
    cal_scores = label_scores(cal_probs, cal_labels, score=score)
    cal_selection = select_top_edges(cal_support, config)

    global_q = conformal_quantile(cal_scores, alpha)
    n_actions = config.capacities.size
    action_thresholds = np.full(n_actions, global_q, dtype=float)
    action_ref_sizes = np.zeros(n_actions, dtype=int)

    for a in range(n_actions):
        mask_edges = cal_selection.action_indices == a
        ref_units = cal_selection.unit_indices[mask_edges]
        if ref_units.size > 0:
            action_thresholds[a] = conformal_quantile(cal_scores[ref_units], alpha)
            action_ref_sizes[a] = int(ref_units.size)

    edge_probs = test_probs[selection.unit_indices]
    edge_scores = all_label_scores(edge_probs, score=score)
    thresholds = action_thresholds[selection.action_indices]
    ref_sizes = action_ref_sizes[selection.action_indices]

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=ref_sizes,
        method=method,
    )


def _patient_union_sets(result: EdgeSetResult) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for unit, pred_set in zip(result.selection.unit_indices, result.sets):
        unit = int(unit)
        if unit not in out:
            out[unit] = pred_set.copy()
        else:
            out[unit] |= pred_set
    return out


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def evaluate_acquisition_edge_sets(
    result: EdgeSetResult,
    labels: np.ndarray,
    action_names: Sequence[str],
    nominal: float = 0.90,
    critical_label: int | None = None,
    oracle_best_action: np.ndarray | None = None,
    hidden_action: np.ndarray | None = None,
    edge_difficulty: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Evaluate selected diagnostic-action coverage.

    edge_difficulty is typically the realized top-B threshold. High-difficulty
    edges correspond to batches / actions where selection was more stringent.
    """
    y_edge = labels[result.selection.unit_indices]
    covered = result.sets[np.arange(result.selection.n_edges), y_edge]
    edge_sizes = result.sets.sum(axis=1)

    patient_sets = _patient_union_sets(result)
    patient_indices = np.array(sorted(patient_sets), dtype=int)
    patient_matrix = np.stack([patient_sets[int(j)] for j in patient_indices], axis=0)
    patient_labels = labels[patient_indices]
    patient_covered = patient_matrix[np.arange(patient_indices.size), patient_labels]

    row: dict[str, float | str] = {
        "method": result.method,
        "coverage": float(np.mean(patient_covered)),
        "edge_cov": float(np.mean(covered)),
        "edge_size": float(np.mean(edge_sizes)),
        "avg_ref_size": float(np.mean(result.reference_sizes)),
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
            row[f"{name}_selected_n"] = 0
            continue

        cov_a = float(np.mean(covered[mask]))
        row[f"{name}_cov"] = cov_a
        row[f"{name}_size"] = float(np.mean(edge_sizes[mask]))
        row[f"{name}_ref"] = float(np.mean(result.reference_sizes[mask]))
        row[f"{name}_selected_n"] = int(np.sum(mask))
        action_covs.append(cov_a)
        under_gaps.append(max(nominal - cov_a, 0.0))

    if action_covs:
        row["action_cov_gap"] = float(np.mean(np.abs(np.asarray(action_covs) - nominal)))
        row["worst_under_gap"] = float(np.max(under_gaps))

    if critical_label is not None:
        critical_mask = y_edge == critical_label
        row["critical_selected_n"] = int(np.sum(critical_mask))
        row["critical_miss_rate"] = (
            float(np.mean(~covered[critical_mask])) if np.any(critical_mask) else np.nan
        )

    if oracle_best_action is not None:
        selected_best = (
            result.selection.action_indices
            == oracle_best_action[result.selection.unit_indices]
        )
        row["oracle_action_hit_rate"] = float(np.mean(selected_best))

    if hidden_action is not None:
        row["hidden_test_rate"] = float(np.mean(hidden_action >= 0))
        hidden_edge = hidden_action[result.selection.unit_indices] >= 0
        row["hidden_selected_n"] = int(np.sum(hidden_edge))
        row["hidden_selected_rate"] = float(np.mean(hidden_edge))
        hidden_action_match = (
            result.selection.action_indices
            == hidden_action[result.selection.unit_indices]
        )
        row["hidden_action_hit_rate"] = (
            float(np.mean(hidden_action_match[hidden_edge]))
            if np.any(hidden_edge)
            else np.nan
        )

    if edge_difficulty is not None:
        finite = np.isfinite(edge_difficulty)
        if np.any(finite):
            median = np.median(edge_difficulty[finite])
            q75 = np.quantile(edge_difficulty[finite], 0.75)
            hard = finite & (edge_difficulty >= median)
            very_hard = finite & (edge_difficulty >= q75)
            easy = finite & (edge_difficulty < median)

            row["easy_edge_cov"] = _safe_mean(covered[easy])
            row["hard_edge_cov"] = _safe_mean(covered[hard])
            row["very_hard_edge_cov"] = _safe_mean(covered[very_hard])
            row["hard_edge_under_gap"] = (
                max(nominal - float(row["hard_edge_cov"]), 0.0)
                if not np.isnan(row["hard_edge_cov"])
                else np.nan
            )
            row["very_hard_under_gap"] = (
                max(nominal - float(row["very_hard_edge_cov"]), 0.0)
                if not np.isnan(row["very_hard_edge_cov"])
                else np.nan
            )
            row["threshold_cov_gap"] = (
                abs(float(row["hard_edge_cov"]) - float(row["easy_edge_cov"]))
                if not np.isnan(row["hard_edge_cov"]) and not np.isnan(row["easy_edge_cov"])
                else np.nan
            )

    return row


def run_one_acquisition_oscp(
    seed: int,
    n: int,
    alpha: float,
    batch_size: int,
    capacities: Sequence[int],
    score: str = "lac",
    label_names: Sequence[str] = DEFAULT_LABELS,
    action_names: Sequence[str] = DEFAULT_ACTIONS,
    hidden_rate: float = 0.10,
    hidden_signal: float = 6.0,
    hidden_critical_prob: float = 0.85,
    entropy_weight: float = 0.2,
    critical_gain_weight: float = 4.0,
    base_feature_dim: int | None = 4,
    support_mode: SupportMode = "entropy_gain",
    oracle_boost: float = 1.0,
    oracle_noise: float = 0.02,
    batch_context: BatchContextMode = "clustered",
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run one synthetic diagnostic-acquisition OSCP experiment."""
    data = generate_synthetic_diagnostic_acquisition(
        n=n,
        seed=seed,
        n_actions=len(action_names),
        n_classes=len(label_names),
        hidden_rate=hidden_rate,
        hidden_signal=hidden_signal,
        hidden_critical_prob=hidden_critical_prob,
        base_feature_dim_for_generation=base_feature_dim or 4,
    )
    split = make_acquisition_splits(data, seed=seed)

    models = fit_acquisition_models(
        split,
        seed=seed,
        entropy_weight=entropy_weight,
        critical_gain_weight=critical_gain_weight,
        base_feature_dim=base_feature_dim,
        support_mode=support_mode,
    )

    config = RelationalSelectionConfig(
        batch_size=batch_size,
        capacities=np.asarray(capacities, dtype=int),
    )

    cal_probs = models.base_model.predict_proba(split.x_base_cal)
    test_probs = models.base_model.predict_proba(split.x_base_test)

    cal_support = acquisition_support_scores(
        split.x_base_cal,
        models.utility_models,
        mode=support_mode,
        hidden_action=split.hidden_action_cal,
        n_actions=len(action_names),
        seed=seed + 300,
        oracle_boost=oracle_boost,
        oracle_noise=oracle_noise,
    )
    test_support = acquisition_support_scores(
        split.x_base_test,
        models.utility_models,
        mode=support_mode,
        hidden_action=split.hidden_action_test,
        n_actions=len(action_names),
        seed=seed + 400,
        oracle_boost=oracle_boost,
        oracle_noise=oracle_noise,
    )

    # Reorder calibration and test pools before batching to create heterogeneous
    # screening rounds. This does not alter labels/features, only batch grouping.
    cal_arrays = reorder_for_batch_context(
        {
            "probs": cal_probs,
            "support": cal_support,
            "labels": split.y_cal,
            "hidden": split.hidden_action_cal,
            "best": split.best_action_cal,
            "route": split.route_hardness_cal,
        },
        key=split.route_hardness_cal,
        mode=batch_context,
    )
    test_arrays = reorder_for_batch_context(
        {
            "probs": test_probs,
            "support": test_support,
            "labels": split.y_test,
            "hidden": split.hidden_action_test,
            "best": split.best_action_test,
            "route": split.route_hardness_test,
        },
        key=split.route_hardness_test,
        mode=batch_context,
    )

    cal_probs = cal_arrays["probs"]
    cal_support = cal_arrays["support"]
    cal_labels = cal_arrays["labels"]

    test_probs = test_arrays["probs"]
    test_support = test_arrays["support"]
    test_labels = test_arrays["labels"]
    test_hidden = test_arrays["hidden"]
    test_best = test_arrays["best"]
    test_route = test_arrays["route"]

    selection = select_top_edges(test_support, config)
    edge_difficulty = edge_selection_thresholds(test_support, selection, config)

    methods = [
        marginal_edge_cp(
            cal_probs=cal_probs,
            cal_labels=cal_labels,
            test_probs=test_probs,
            selection=selection,
            alpha=alpha,
            score=score,
        ),
        actionwise_selected_cp(
            cal_probs=cal_probs,
            cal_labels=cal_labels,
            cal_support=cal_support,
            test_probs=test_probs,
            selection=selection,
            config=config,
            alpha=alpha,
            score=score,
        ),
        relational_oscp_top(
            cal_probs=cal_probs,
            cal_labels=cal_labels,
            cal_support=cal_support,
            test_probs=test_probs,
            test_support=test_support,
            selection=selection,
            config=config,
            alpha=alpha,
            score=score,
            method="OSCP",
        ),
    ]

    rows = [
        evaluate_acquisition_edge_sets(
            result=r,
            labels=test_labels,
            action_names=action_names,
            nominal=1.0 - alpha,
            critical_label=len(label_names) - 1,
            oracle_best_action=test_best,
            hidden_action=test_hidden,
            edge_difficulty=edge_difficulty,
        )
        for r in methods
    ]

    hidden_test = test_hidden >= 0
    selected_hidden = hidden_test[selection.unit_indices]
    selected_hidden_action_match = (
        selection.action_indices == test_hidden[selection.unit_indices]
    )

    diagnostics: dict[str, float | int | str] = {
        "seed": seed,
        "support_mode": support_mode,
        "batch_context": batch_context,
        "n_cal": int(cal_labels.size),
        "n_test": int(test_labels.size),
        "n_edges": int(selection.n_edges),
        "n_selected_patients": int(np.unique(selection.unit_indices).size),
        "edge_patient_ratio": float(
            selection.n_edges / np.unique(selection.unit_indices).size
        ),
        "mean_support": float(np.mean(test_support)),
        "max_support": float(np.max(test_support)),
        "hidden_test_rate": float(np.mean(hidden_test)),
        "hidden_selected_rate": float(np.mean(selected_hidden)),
        "hidden_action_hit_rate": (
            float(np.mean(selected_hidden_action_match[selected_hidden]))
            if np.any(selected_hidden)
            else np.nan
        ),
        "mean_edge_threshold": float(np.mean(edge_difficulty[np.isfinite(edge_difficulty)])),
        "mean_route_hardness": float(np.mean(test_route)),
    }

    return pd.DataFrame(rows), diagnostics


def aggregate_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat(
        [df.assign(seed=i) for i, df in enumerate(runs)],
        ignore_index=True,
    )
    numeric = all_rows.select_dtypes(include=[np.number]).columns.drop("seed")
    means = all_rows.groupby("method", sort=False)[numeric].mean()
    stds = all_rows.groupby("method", sort=False)[numeric].std(ddof=1).add_suffix("_std")
    return means.join(stds).reset_index()
