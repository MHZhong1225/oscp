#!/usr/bin/env python
"""Run ChEMBL36 target-screening edge-label OSCP experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.chembl_oscp import (  # noqa: E402
    aggregate_runs,
    edge_label_actionwise_cp,
    edge_label_bonferroni_cp,
    edge_label_jomi_unit_top,
    edge_label_marginal_cp,
    edge_label_oscp_top,
    edge_label_selection_thresholds,
    edge_label_self_calibrating_cp,
    evaluate_edge_label_sets,
    fit_chembl_multitask_model,
    load_chembl36_data,
    make_chembl_splits,
    select_top_edges_masked,
)
from methods.relational_core import RelationalSelectionConfig  # noqa: E402


def to_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percent_cols = [
        c
        for c in out.columns
        if c.endswith("_cov")
        or c.endswith("_rate")
        or c.endswith("_under_gap")
        or c.endswith("_cov_gap")
        or c.endswith("_threshold_cov_gap")
        or c in {"coverage", "edge_cov", "active_cov", "inactive_cov"}
    ]
    for col in percent_cols:
        if col in out.columns:
            out[col] = 100.0 * out[col]
    return out


def existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def round_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        decimals = 6 if col == "time_sec" or col == "time_sec_std" else 4
        out[col] = out[col].round(decimals)
    return out


def run_one(
    seed: int,
    data_root: Path,
    edges_path: Path | None,
    sqlite_path: Path | None,
    n_targets: int,
    use_existing_split: bool,
    force_rebuild: bool,
    fingerprint: str,
    n_bits: int,
    radius: int,
    alpha: float,
    batch_size: int,
    capacity: int,
    max_iter: int,
    C: float,
    cuda: int | None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    data = load_chembl36_data(
        data_root=data_root,
        edges_path=edges_path,
        n_targets=n_targets,
        use_existing_split=use_existing_split,
        seed=seed,
        force_rebuild=force_rebuild,
        sqlite_path=sqlite_path,
    )
    split, resolved_fingerprint = make_chembl_splits(
        data,
        fingerprint=fingerprint,
        n_bits=n_bits,
        radius=radius,
    )
    models, model_info = fit_chembl_multitask_model(
        split,
        seed=seed,
        max_iter=max_iter,
        C=C,
        cuda=cuda,
    )

    cal_probs = models.predict_active_proba(split.x_cal)
    test_probs = models.predict_active_proba(split.x_test)

    config = RelationalSelectionConfig(
        batch_size=batch_size,
        capacities=np.full(len(data.action_names), capacity, dtype=int),
    )
    selection = select_top_edges_masked(test_probs, split.observed_test, config)
    if selection.n_edges == 0:
        raise ValueError(
            f"No selected edges. Reduce --batch-size below n_test={split.y_test.shape[0]} "
            "or lower target filtering."
        )
    edge_difficulty = edge_label_selection_thresholds(
        test_probs,
        split.observed_test,
        selection,
        config,
    )

    timed_results = []

    def add_timed_result(make_result) -> None:
        start = time.perf_counter()
        timed_results.append((make_result(), time.perf_counter() - start))

    for make_result in (
        lambda: edge_label_marginal_cp(
            cal_probs=cal_probs,
            cal_labels=split.y_cal,
            cal_observed=split.observed_cal,
            test_probs=test_probs,
            selection=selection,
            alpha=alpha,
        ),
        # lambda: edge_label_bonferroni_cp(
        #     cal_probs=cal_probs,
        #     cal_labels=split.y_cal,
        #     cal_observed=split.observed_cal,
        #     test_probs=test_probs,
        #     selection=selection,
        #     alpha=alpha,
        #     divisor=len(data.action_names),
        # ),
        lambda: edge_label_actionwise_cp(
            cal_probs=cal_probs,
            cal_labels=split.y_cal,
            cal_observed=split.observed_cal,
            test_probs=test_probs,
            selection=selection,
            config=config,
            alpha=alpha,
        ),
        lambda: edge_label_self_calibrating_cp(
            cal_probs=cal_probs,
            cal_labels=split.y_cal,
            cal_observed=split.observed_cal,
            test_probs=test_probs,
            selection=selection,
            alpha=alpha,
        ),
        lambda: edge_label_jomi_unit_top(
            cal_probs=cal_probs,
            cal_labels=split.y_cal,
            cal_observed=split.observed_cal,
            test_probs=test_probs,
            selection=selection,
            config=config,
            alpha=alpha,
        ),
        lambda: edge_label_oscp_top(
            cal_probs=cal_probs,
            cal_labels=split.y_cal,
            cal_observed=split.observed_cal,
            test_probs=test_probs,
            test_observed=split.observed_test,
            selection=selection,
            config=config,
            alpha=alpha,
        ),
    ):
        add_timed_result(make_result)

    rows = []
    for result, time_sec in timed_results:
        row = evaluate_edge_label_sets(
            result=result,
            labels=split.y_test,
            action_names=data.action_names,
            nominal=1.0 - alpha,
            edge_difficulty=edge_difficulty,
        )
        row["time_sec"] = time_sec
        rows.append(row)

    selected_labels = split.y_test[selection.unit_indices, selection.action_indices]
    target_summary = (
        data.edges.groupby(["target_id", "target_chembl_id"], as_index=False)
        .agg(
            target_name=("target_name", "first"),
            n_edges=("y", "size"),
            n_active=("y", "sum"),
        )
        .sort_values("target_id")
    )
    target_summary["n_inactive"] = target_summary["n_edges"] - target_summary["n_active"]
    target_summary["selected_edges"] = [
        int(np.sum(selection.action_indices == a))
        for a in range(len(data.action_names))
    ]
    target_summary["selected_active_rate"] = [
        float(np.mean(selected_labels[selection.action_indices == a]))
        if np.any(selection.action_indices == a)
        else np.nan
        for a in range(len(data.action_names))
    ]

    diagnostics = {
        "seed": seed,
        "fingerprint": resolved_fingerprint,
        "n_bits": n_bits,
        "radius": radius,
        "n_targets": len(data.action_names),
        "batch_size": batch_size,
        "capacity": capacity,
        "model_type": model_info["model_type"],
        "device": model_info["device"],
        "mean_val_log_loss": model_info["mean_val_log_loss"],
        "n_train_compounds": int(split.x_train.shape[0]),
        "n_val_compounds": int(split.x_val.shape[0]),
        "n_cal_compounds": int(split.x_cal.shape[0]),
        "n_test_compounds": int(split.x_test.shape[0]),
        "n_cal_edges": int(split.observed_cal.sum()),
        "n_test_edges": int(split.observed_test.sum()),
        "n_selected_edges": int(selection.n_edges),
        "n_selected_compounds": int(np.unique(selection.unit_indices).size),
        "selected_active_rate": float(np.mean(selected_labels)),
        "mean_support": float(np.mean(test_probs)),
        "max_support": float(np.max(test_probs)),
        "mean_edge_threshold": float(np.mean(edge_difficulty[np.isfinite(edge_difficulty)])),
    }
    return pd.DataFrame(rows), diagnostics, target_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/chembl36"))
    parser.add_argument("--edges-path", type=Path, default=None)
    parser.add_argument("--sqlite-path", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--use-existing-split",
        action="store_true",
        help="Reuse split column from chembl_oscp_edges_split.parquet instead of creating seed-specific compound splits.",
    )
    parser.add_argument("--n-targets", type=int, default=30)
    parser.add_argument(
        "--fingerprint",
        choices=["auto", "morgan", "hashed_smiles"],
        default="auto",
        help="auto uses Morgan when RDKit is installed, otherwise hashed SMILES n-grams.",
    )
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--capacity", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index, defaults to 0.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/chembl36_oscp"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_cov_cols = [f"target_{a}_cov" for a in range(args.n_targets)]
    display_cols = [
        "method",
        "coverage",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        "selected_active_rate",
        "active_cov",
        "inactive_cov",
        "action_cov_gap",
        "worst_under_gap",
        "easy_edge_cov",
        "hard_edge_cov",
        "very_hard_edge_cov",
        "threshold_cov_gap",
        "time_sec",
        "n_edges",
        "n_compounds",
        *target_cov_cols,
    ]
    core_cols = [
        "method",
        "coverage",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        "selected_active_rate",
        "active_cov",
        "inactive_cov",
        "action_cov_gap",
        "worst_under_gap",
        "time_sec",
    ]

    runs: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    target_summaries: list[pd.DataFrame] = []

    for seed in range(args.seeds):
        df, diag, target_summary = run_one(
            seed=seed,
            data_root=args.data_root,
            edges_path=args.edges_path,
            sqlite_path=args.sqlite_path,
            n_targets=args.n_targets,
            use_existing_split=args.use_existing_split,
            force_rebuild=args.force_rebuild,
            fingerprint=args.fingerprint,
            n_bits=args.n_bits,
            radius=args.radius,
            alpha=args.alpha,
            batch_size=args.batch_size,
            capacity=args.capacity,
            max_iter=args.max_iter,
            C=args.C,
            cuda=args.cuda,
        )
        runs.append(df)
        diagnostics.append(diag)
        target_summaries.append(target_summary.assign(seed=seed))

        percent = round_for_output(to_percent_table(df))
        print(f"\nSeed {seed} diagnostics: {diag}")
        print(
            percent[existing_cols(percent, display_cols)].to_string(
                index=False,
                float_format=lambda x: f"{x:0.4f}",
            )
        )

    raw = pd.concat(
        [df.assign(seed=seed) for seed, df in enumerate(runs)],
        ignore_index=True,
    )
    summary = aggregate_runs(runs)
    display_summary = round_for_output(to_percent_table(summary))

    raw.to_csv(args.out_dir / "raw.csv", index=False)
    summary.to_csv(args.out_dir / "summary_full.csv", index=False)
    display_summary.to_csv(args.out_dir / "summary.csv", index=False)
    summary[existing_cols(summary, core_cols)].to_csv(args.out_dir / "core_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)
    pd.concat(target_summaries, ignore_index=True).to_csv(
        args.out_dir / "target_summary.csv",
        index=False,
    )

    print("\nAggregated means:")
    print(
        display_summary[existing_cols(display_summary, display_cols)].to_string(
            index=False,
            float_format=lambda x: f"{x:0.4f}",
        )
    )
    print(f"\nSaved CSV files under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
