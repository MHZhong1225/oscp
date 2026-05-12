#!/usr/bin/env python
"""Run relational OSCP on Nursery or BACH datasets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.evaluate import to_percent_table
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



from methods.trainer.datasets import (
    BACH_CRITICAL_CLASS,
    NURSERY_CRITICAL_CLASS,
    load_bach_splits,
    load_nursery_data,
    make_dataset_splits,
    fit_dataset_classifier,
)
from methods.baselines import (
    action_wise_cp,
    relational_bonferroni_cp,
    relational_jomi_unit_top,
    relational_marginal_cp,
    relational_self_calibrating_cp,
)
from methods.oscp import relational_oscp_top, relational_swap_cp
from methods.relational_core import (
    RelationSpec,
    RelationalSelectionConfig,
    aggregate_runs,
    compute_support_scores,
    evaluate_edge_sets,
    select_top_edges,
)


def dataset_relation(dataset: str) -> tuple[RelationSpec, np.ndarray, int, str]:
    if dataset == "nursery":
        label_names = ("not_recom", "very_recom", "priority", "spec_prior")
        action_names = ("reject", "routine_review", "priority_review", "special_priority")
        relation = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        capacities = np.array([8, 8, 6, 4], dtype=int)
        return (
            RelationSpec(label_names, action_names, relation),
            capacities,
            NURSERY_CRITICAL_CLASS,
            "reject",
        )

    if dataset == "bach":
        label_names = ("Benign", "InSitu", "Invasive", "Normal")
        action_names = ("routine_followup", "biopsy_review", "oncology_escalation")
        relation = np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=float,
        )
        capacities = np.array([8, 8, 6], dtype=int)
        return (
            RelationSpec(label_names, action_names, relation),
            capacities,
            BACH_CRITICAL_CLASS,
            "routine_followup",
        )

    raise ValueError(f"unknown dataset: {dataset}")



def round_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        decimals = 6 if col == "time_sec" or col == "time_sec_std" else 4
        out[col] = out[col].round(decimals)
    return out


def table_string(df: pd.DataFrame) -> str:
    formatters = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "time_sec" or col == "time_sec_std":
            formatters[col] = lambda x: f"{x:0.6f}"
        else:
            formatters[col] = lambda x: f"{x:0.4f}"
    return df.to_string(index=False, formatters=formatters)


def format_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col == "time_sec" or col == "time_sec_std":
            out[col] = out[col].map(lambda x: f"{x:0.6f}" if pd.notna(x) else "")
        else:
            out[col] = out[col].map(lambda x: f"{x:0.4f}" if pd.notna(x) else "")
    return out


def load_split(args: argparse.Namespace, seed: int):
    if args.dataset == "nursery":
        data = load_nursery_data(path=args.nursery_csv, n=args.n, seed=seed)
        return make_dataset_splits(data, seed=seed)
    if args.dataset == "bach":
        feature_device = "cpu" if args.cuda is None else f"cuda:{args.cuda}"
        return load_bach_splits(
            root=args.bach_root,
            seed=seed,
            image_size=args.image_size,
            backbone_name=args.backbone,
            pretrained=not args.no_pretrained,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            device=feature_device,
        )
    raise ValueError(f"unknown dataset: {args.dataset}")


def run_one(args: argparse.Namespace, seed: int) -> tuple[pd.DataFrame, dict]:
    split = load_split(args, seed)
    model, model_info = fit_dataset_classifier(split, seed=seed, cuda=args.cuda)

    relation_spec, capacities, critical_label, critical_action = dataset_relation(args.dataset)
    config = RelationalSelectionConfig(batch_size=args.batch_size, capacities=capacities)
    cal_probs = model.predict_proba(split.x_cal)
    test_probs = model.predict_proba(split.x_test)
    cal_support = compute_support_scores(cal_probs, relation_spec.relation)
    test_support = compute_support_scores(test_probs, relation_spec.relation)
    selection = select_top_edges(test_support, config)
    if selection.n_edges == 0:
        raise RuntimeError(
            f"No selected edges. Reduce --batch-size below n_test={split.y_test.size}."
        )

    timed_results = []

    def add_timed_result(make_result):
        start = time.perf_counter()
        result = make_result()
        timed_results.append((result, time.perf_counter() - start))

    add_timed_result(lambda: relational_marginal_cp(
        cal_probs,
        split.y_cal,
        test_probs,
        selection,
        args.alpha,
        score=args.score,
        method="Marginal CP",
    ))
    add_timed_result(lambda: relational_bonferroni_cp(
        cal_probs,
        split.y_cal,
        test_probs,
        selection,
        args.alpha,
        divisor=len(relation_spec.action_names),
        score=args.score,
    ))
    add_timed_result(lambda: action_wise_cp(
        cal_probs,
        split.y_cal,
        test_probs,
        selection,
        relation_spec.relation,
        args.alpha,
    ))
    if args.include_swap:
        add_timed_result(lambda: relational_swap_cp(
            cal_probs,
            split.y_cal,
            cal_support,
            test_probs,
            test_support,
            selection,
            config,
            args.alpha,
            score=args.score,
        ))
    add_timed_result(lambda: relational_self_calibrating_cp(
        cal_probs,
        split.y_cal,
        test_probs,
        selection,
        args.alpha,
        score=args.score,
    ))
    add_timed_result(lambda: relational_jomi_unit_top(
        cal_probs,
        split.y_cal,
        cal_support,
        test_probs,
        test_support,
        selection,
        config,
        args.alpha,
        score=args.score,
    ))
    add_timed_result(lambda: relational_oscp_top(
        cal_probs,
        split.y_cal,
        cal_support,
        test_probs,
        test_support,
        selection,
        config,
        args.alpha,
        score=args.score,
    ))

    rows = []
    for result, time_sec in timed_results:
        row = evaluate_edge_sets(
            result,
            split.y_test,
            relation_spec.relation,
            relation_spec.action_names,
            nominal=1.0 - args.alpha,
            critical_label=critical_label,
            critical_action=critical_action,
        )
        row["time_sec"] = time_sec
        rows.append(row)

    critical_action_idx = relation_spec.action_names.index(critical_action)
    critical_edges = selection.action_indices == critical_action_idx
    critical_units = selection.unit_indices[critical_edges]
    diagnostics = {
        "seed": seed,
        "dataset": args.dataset,
        "score": args.score,
        "device": model_info["device"],
        "val_log_loss": model_info["val_log_loss"],
        "n_cal": int(split.y_cal.size),
        "n_test": int(split.y_test.size),
        "n_edges": int(selection.n_edges),
        "n_selected_units": int(np.unique(selection.unit_indices).size),
        "edge_patient_ratio": float(selection.n_edges / np.unique(selection.unit_indices).size),
        f"{critical_action}_edges": int(np.sum(critical_edges)),
        f"{critical_action}_critical_rate": float(np.mean(split.hidden_test[critical_units])),
    }
    return pd.DataFrame(rows), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nursery", "bach"], required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n", type=int, default=None, help="Optional nursery subsample size.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--score", choices=["lac", "aps"], default="lac")
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device index. Omit for CPU.")
    parser.add_argument("--nursery-csv", type=Path, default=Path("dataset/nursery/nursery.csv"))
    parser.add_argument("--bach-root", type=Path, default=Path("/home/ubuntu/zmh/BrCPT/datasets/bach"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"],
        default="resnet18",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-swap", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed diagnostics and tables.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("results") / f"relational_{args.dataset}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    relation_spec, _, _, _ = dataset_relation(args.dataset)
    action_cov_cols = [f"{name}_cov" for name in relation_spec.action_names]
    display_cols = [
        "method",
        "coverage",
        "avg_size",
        "time_sec",
        "decision_ambiguity",
        "action_union_size",
        "edge_cov",
        *action_cov_cols,
        "edge_size",
        "action_cov_gap",
        "worst_under_gap",
        "explain_rate",
        "avg_ref_size",
    ]
    core_cols = [
        "method",
        "coverage",
        "avg_size",
        "time_sec",
        "decision_ambiguity",
        "action_union_size",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        *action_cov_cols,
    ]

    runs = []
    diagnostics = []
    seed_core_cols = [
        "method",
        "coverage",
        "avg_size",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        "time_sec",
    ]
    for seed in tqdm(range(args.seeds), desc="Seeds", unit="seed"):
        df, diag = run_one(args, seed)
        runs.append(df)
        diagnostics.append(diag)
        if args.verbose:
            print(f"\nSeed {seed} diagnostics: {diag}")
            print(table_string(round_for_output(to_percent_table(df)[display_cols])))
        else:
            print(f"\nSeed {seed} core metrics:")
            print(table_string(round_for_output(to_percent_table(df)[seed_core_cols])))

    raw = pd.concat([df.assign(seed=seed) for seed, df in enumerate(runs)], ignore_index=True)
    summary = aggregate_runs(runs)
    format_for_csv(raw).to_csv(args.out_dir / "raw.csv", index=False)
    format_for_csv(summary).to_csv(args.out_dir / "summary.csv", index=False)
    format_for_csv(summary[core_cols]).to_csv(args.out_dir / "core_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)

    print("\nAggregated means:")
    print(table_string(round_for_output(to_percent_table(summary)[display_cols])))
    print("\nCore metrics (raw scale):")
    print(table_string(round_for_output(summary[core_cols])))
    print(f"\nSaved CSV files under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
