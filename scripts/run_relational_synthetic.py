#!/usr/bin/env python
"""Run the relational operation-selection synthetic triage experiment."""

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



from methods.relational_core import (
    RelationSpec,
    RelationalSelectionConfig,
    aggregate_runs,
    compute_support_scores,
    evaluate_edge_sets,
    select_top_edges,
)
from methods.baselines import (
    action_wise_cp,
    relational_bonferroni_cp,
    relational_jomi_unit_top,
    relational_marginal_cp,
    relational_self_calibrating_cp,
)
from methods.oscp import relational_oscp_top, relational_swap_cp
from methods.trainer.synthetic import (
    CRITICAL_CLASS,
    LABELS,
    fit_base_classifier,
    generate_synthetic_triage,
    make_splits,
)


def triage_relation() -> tuple[RelationSpec, np.ndarray]:
    action_names = ("routine", "labs", "imaging", "urgent", "expert_review")
    relation = np.array(
        [
            # routine, labs, imaging, urgent, expert_review
            [1, 0, 0, 0, 0],  # benign
            [1, 1, 0, 0, 0],  # mild
            [0, 1, 1, 0, 1],  # moderate
            [0, 0, 1, 1, 1],  # severe
            [0, 0, 0, 1, 1],  # critical
        ],
        dtype=float,
    )
    capacities = np.array([30, 30, 20, 10, 15], dtype=int)
    spec = RelationSpec(tuple(LABELS.tolist()), action_names, relation)
    return spec, capacities


def run_one(
    seed: int,
    n: int,
    alpha: float,
    batch_size: int,
    score: str,
    cuda: int | None,
) -> tuple[pd.DataFrame, dict]:
    data = generate_synthetic_triage(n=n, seed=seed)
    split = make_splits(data, seed=seed)
    model, model_info = fit_base_classifier(split, seed=seed, cuda=cuda)

    relation_spec, capacities = triage_relation()
    config = RelationalSelectionConfig(batch_size=batch_size, capacities=capacities)

    cal_probs = model.predict_proba(split.x_cal)
    test_probs = model.predict_proba(split.x_test)
    cal_support = compute_support_scores(cal_probs, relation_spec.relation)
    test_support = compute_support_scores(test_probs, relation_spec.relation)
    selection = select_top_edges(test_support, config)

    timed_results = []
    timed_calls = [
        lambda: relational_marginal_cp(
            cal_probs,
            split.y_cal,
            test_probs,
            selection,
            alpha,
            score=score,
            method="Marginal CP",
        ),
        lambda: relational_bonferroni_cp(
            cal_probs,
            split.y_cal,
            test_probs,
            selection,
            alpha,
            divisor=len(relation_spec.action_names),
            score=score,
        ),
        lambda: action_wise_cp(
            cal_probs,
            split.y_cal,
            test_probs,
            selection,
            relation_spec.relation,
            alpha,
        ),
    ]
    if n <= 10000:
        timed_calls.append(
            lambda: relational_swap_cp(
                cal_probs,
                split.y_cal,
                cal_support,
                test_probs,
                test_support,
                selection,
                config,
                alpha,
                score=score,
            )
        )
    timed_calls.extend(
        [
            lambda: relational_self_calibrating_cp(
                cal_probs,
                split.y_cal,
                test_probs,
                selection,
                alpha,
                score=score,
            ),
            lambda: relational_jomi_unit_top(
                cal_probs,
                split.y_cal,
                cal_support,
                test_probs,
                test_support,
                selection,
                config,
                alpha,
                score=score,
            ),
            lambda: relational_oscp_top(
                cal_probs,
                split.y_cal,
                cal_support,
                test_probs,
                test_support,
                selection,
                config,
                alpha,
                score=score,
            ),
        ]
    )

    for make_result in timed_calls:
        start = time.perf_counter()
        timed_results.append((make_result(), time.perf_counter() - start))

    rows = []
    for result, time_sec in timed_results:
        row = evaluate_edge_sets(
            result,
            split.y_test,
            relation_spec.relation,
            relation_spec.action_names,
            nominal=1.0 - alpha,
            critical_label=CRITICAL_CLASS,
            critical_action="routine",
        )
        row["time_sec"] = time_sec
        rows.append(row)

    routine_edges = selection.action_indices == relation_spec.action_names.index("routine")
    routine_units = selection.unit_indices[routine_edges]
    diagnostics = {
        "seed": seed,
        "score": score,
        "device": model_info["device"],
        "val_log_loss": model_info["val_log_loss"],
        "n_cal": len(split.y_cal),
        "n_test": len(split.y_test),
        "n_edges": selection.n_edges,
        "n_selected_patients": int(np.unique(selection.unit_indices).size),
        "edge_patient_ratio": float(
            selection.n_edges / np.unique(selection.unit_indices).size
        ),
        "routine_edges": int(np.sum(routine_edges)),
        "routine_hidden_rate": float(np.mean(split.hidden_test[routine_units])),
    }
    return pd.DataFrame(rows), diagnostics




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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--score", choices=["lac", "aps"], default="lac")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index, defaults to 0.")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed diagnostics and tables.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/relational_synthetic"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
        df, diag = run_one(
            seed,
            args.n,
            args.alpha,
            args.batch_size,
            args.score,
            args.cuda,
        )
        runs.append(df)
        diagnostics.append(diag)
        if args.verbose:
            print(f"\nSeed {seed} diagnostics: {diag}")
            print(table_string(
                round_for_output(to_percent_table(df)[
                    [
                        "method",
                        "coverage",
                        "avg_size",
                        "time_sec",
                        "decision_ambiguity",
                        "action_union_size",
                        "edge_cov",
                        "routine_cov",
                        "labs_cov",
                        "imaging_cov",
                        "urgent_cov",
                        "expert_review_cov",
                        "edge_size",
                        "action_cov_gap",
                        "worst_under_gap",
                        "explain_rate",
                        "routine_critical_miss_rate",
                        "avg_ref_size",
                    ]
                ])
            ))
        else:
            print(f"\nSeed {seed} core metrics:")
            print(table_string(round_for_output(to_percent_table(df)[seed_core_cols])))

    raw = pd.concat(
        [df.assign(seed=seed) for seed, df in enumerate(runs)],
        ignore_index=True,
    )
    summary = aggregate_runs(runs)
    format_for_csv(raw).to_csv(args.out_dir / "raw.csv", index=False)
    format_for_csv(summary).to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)

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
        "routine_cov",
        "labs_cov",
        "imaging_cov",
        "urgent_cov",
        "expert_review_cov",
    ]
    format_for_csv(summary[core_cols]).to_csv(args.out_dir / "core_metrics.csv", index=False)

    display_cols = [
        "method",
        "coverage",
        "avg_size",
        "time_sec",
        "decision_ambiguity",
        "action_union_size",
        "edge_cov",
        "routine_cov",
        "labs_cov",
        "imaging_cov",
        "urgent_cov",
        "expert_review_cov",
        "edge_size",
        "action_cov_gap",
        "worst_under_gap",
        "explain_rate",
        "routine_critical_miss_rate",
        "avg_ref_size",
        "n_edges",
        "n_patients",
    ]
    print("\nAggregated means:")
    print(table_string(round_for_output(to_percent_table(summary)[display_cols])))
    print("\nCore metrics (raw scale):")
    print(table_string(round_for_output(summary[core_cols])))
    print(f"\nSaved CSV files under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
