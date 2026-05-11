"""Batch-dependent downstream operation selection rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OPERATIONS = ("routine", "review", "urgent")


@dataclass(frozen=True)
class SelectionConfig:
    """Top/bottom capacity selection within each test batch."""

    batch_size: int = 100
    urgent_b: int = 10
    routine_b: int = 30


def split_batches(n: int, batch_size: int) -> list[np.ndarray]:
    """Split the first n units into complete batches."""
    usable = (n // batch_size) * batch_size
    return [np.arange(start, start + batch_size) for start in range(0, usable, batch_size)]


def select_operations(risk: np.ndarray, config: SelectionConfig) -> np.ndarray:
    """Assign each unit in complete batches to routine/review/urgent."""
    risk = np.asarray(risk, dtype=float)
    op = np.full(risk.size, "unused", dtype=object)
    for batch in split_batches(risk.size, config.batch_size):
        order = batch[np.argsort(risk[batch], kind="mergesort")]
        routine = order[: config.routine_b]
        urgent = order[-config.urgent_b :]
        review = np.setdiff1d(batch, np.concatenate([routine, urgent]), assume_unique=False)
        op[routine] = "routine"
        op[review] = "review"
        op[urgent] = "urgent"
    return op


def selected_mask(op: np.ndarray, operation: str) -> np.ndarray:
    return np.asarray(op, dtype=object) == operation


def top_reference_mask_for_selected(
    cal_risk: np.ndarray,
    batch_risk: np.ndarray,
    selected_local_index: int,
    b: int,
) -> np.ndarray:
    """Calibration points that would enter top-B after replacing selected unit."""
    others = np.delete(batch_risk, selected_local_index)
    threshold = np.partition(others, others.size - b)[others.size - b]
    return cal_risk >= threshold


def bottom_reference_mask_for_selected(
    cal_risk: np.ndarray,
    batch_risk: np.ndarray,
    selected_local_index: int,
    b: int,
) -> np.ndarray:
    """Calibration points that would enter bottom-B after replacing selected unit."""
    others = np.delete(batch_risk, selected_local_index)
    threshold = np.partition(others, b - 1)[b - 1]
    return cal_risk <= threshold


def apply_selection_to_single_batch(
    risk: np.ndarray,
    config: SelectionConfig,
) -> np.ndarray:
    """Operation labels for one batch-sized risk vector."""
    if risk.size != config.batch_size:
        raise ValueError("risk vector must have exactly config.batch_size entries")
    order = np.argsort(risk, kind="mergesort")
    op = np.full(risk.size, "review", dtype=object)
    op[order[: config.routine_b]] = "routine"
    op[order[-config.urgent_b :]] = "urgent"
    return op
