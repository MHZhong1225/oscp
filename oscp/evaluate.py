"""Evaluation metrics for operation-selected conformal experiments."""

from __future__ import annotations

import pandas as pd


def to_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percent_cols = [
        c
        for c in out.columns
        if c.endswith("_cov")
        or c.endswith("_gap")
        or c.endswith("_miss_rate")
        or c == "marginal_cov"
    ]
    for c in percent_cols:
        out[c] = 100.0 * out[c]
    return out
