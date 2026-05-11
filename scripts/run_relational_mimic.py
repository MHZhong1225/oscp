#!/usr/bin/env python
"""Run the relational operation-selection MIMIC-IV ICU triage experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oscp.mimic import (
    MIMIC_CRITICAL_CLASS,
    MIMIC_LABELS,
    fit_mimic_classifier,
    load_mimic_triage,
    make_mimic_splits,
)
from oscp.relational import (
    RelationSpec,
    RelationalSelectionConfig,
    action_wise_cp,
    aggregate_runs,
    compute_support_scores,
    evaluate_edge_sets,
    relational_bonferroni_cp,
    relational_jomi_unit_top,
    relational_marginal_cp,
    relational_oscp_top,
    relational_self_calibrating_cp,
    relational_swap_cp,
    select_top_edges,
)


def mimic_relation() -> tuple[RelationSpec, np.ndarray]:
    action_names = (
        "discharge_planning",
        "ward_monitoring",
        "stepdown_review",
        "icu_escalation",
        "mortality_review",
    )
    relation = np.array(
        [
            # discharge_planning, ward_monitoring, stepdown_review, icu_escalation, mortality_review
            [1, 1, 0, 0, 0],  # brief_stay
            [1, 1, 1, 0, 0],  # moderate_stay
            [0, 1, 1, 1, 0],  # extended_stay
            [0, 0, 1, 1, 1],  # prolonged_stay
            [0, 0, 0, 1, 1],  # hospital_death
        ],
        dtype=float,
    )
    capacities = np.array([25, 30, 25, 15, 10], dtype=int)
    spec = RelationSpec(tuple(MIMIC_LABELS.tolist()), action_names, relation)
    return spec, capacities


def run_one(
    seed: int,
    n: int | None,
    frac: float | None,
    alpha: float,
    batch_size: int,
    score: str,
    data_root: Path,
    cuda: int | None,
) -> tuple[pd.DataFrame, dict]:
    data = load_mimic_triage(root=data_root, n=n, frac=frac, seed=seed)
    split = make_mimic_splits(data, seed=seed)
    model, model_info = fit_mimic_classifier(split, seed=seed, cuda=cuda)

    relation_spec, capacities = mimic_relation()
    config = RelationalSelectionConfig(batch_size=batch_size, capacities=capacities)

    cal_probs = model.predict_proba(split.x_cal)
    test_probs = model.predict_proba(split.x_test)
    cal_support = compute_support_scores(cal_probs, relation_spec.relation)
    test_support = compute_support_scores(test_probs, relation_spec.relation)
    selection = select_top_edges(test_support, config)

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
            alpha,
            score=score,
            method="Marginal CP",
        ))
    add_timed_result(lambda: relational_jomi_unit_top(
            cal_probs,
            split.y_cal,
            cal_support,
            test_probs,
            test_support,
            selection,
            config,
            alpha,
            score=score,
        ))
    add_timed_result(lambda: relational_self_calibrating_cp(
            cal_probs,
            split.y_cal,
            test_probs,
            selection,
            alpha,
            score=score,
        ))
    add_timed_result(lambda: action_wise_cp(
            cal_probs,
            split.y_cal,
            test_probs,
            selection,
            relation_spec.relation,
            alpha,
        ))
    add_timed_result(lambda: relational_oscp_top(
            cal_probs,
            split.y_cal,
            cal_support,
            test_probs,
            test_support,
            selection,
            config,
            alpha,
            score=score,
        ))
    add_timed_result(lambda: relational_bonferroni_cp(
            cal_probs,
            split.y_cal,
            test_probs,
            selection,
            alpha,
            divisor=len(relation_spec.action_names),
            score=score,
        ))
    if split.y_test.size <= 5000:
        add_timed_result(lambda: relational_swap_cp(
                cal_probs,
                split.y_cal,
                cal_support,
                test_probs,
                test_support,
                selection,
                config,
                alpha,
                score=score,
            ))

    rows = []
    for result, time_sec in timed_results:
        row = evaluate_edge_sets(
            result,
            split.y_test,
            relation_spec.relation,
            relation_spec.action_names,
            nominal=1.0 - alpha,
            critical_label=MIMIC_CRITICAL_CLASS,
            critical_action="discharge_planning",
        )
        row["time_sec"] = time_sec
        rows.append(row)

    discharge_idx = relation_spec.action_names.index("discharge_planning")
    discharge_edges = selection.action_indices == discharge_idx
    discharge_units = selection.unit_indices[discharge_edges]
    diagnostics = {
        "seed": seed,
        "score": score,
        "device": model_info["device"],
        "sample_frac": frac if frac is not None else np.nan,
        "val_log_loss": model_info["val_log_loss"],
        "n_total": int(data.y.size),
        "n_cal": len(split.y_cal),
        "n_test": len(split.y_test),
        "n_edges": selection.n_edges,
        "n_selected_patients": int(np.unique(selection.unit_indices).size),
        "edge_patient_ratio": float(
            selection.n_edges / np.unique(selection.unit_indices).size
        ),
        "discharge_planning_edges": int(np.sum(discharge_edges)),
        "discharge_planning_death_rate": float(
            np.mean(split.hidden_test[discharge_units])
        ),
    }
    return pd.DataFrame(rows), diagnostics


def to_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percent_cols = [
        c
        for c in out.columns
        if c.endswith("_cov")
        or c.endswith("_explain")
        or c.endswith("_miss_rate")
        or c in {"coverage", "edge_cov", "explain_rate"}
    ]
    for c in percent_cols:
        out[c] = 100.0 * out[c]
    return out


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
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--score", choices=["lac", "aps"], default="lac")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index, defaults to 0.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/mimic-iv-3.1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/relational_mimic"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    relation_spec, _ = mimic_relation()
    action_cov_cols = [f"{name}_cov" for name in relation_spec.action_names]
    display_cols = [
        "method",
        "coverage",
        "size",
        "time_sec",
        "decision_ambiguity",
        "action_union_size",
        "edge_cov",
        *action_cov_cols,
        "avg_size",
        "action_cov_gap",
        "worst_under_gap",
        "explain_rate",
        "discharge_planning_critical_miss_rate",
        "avg_ref_size",
    ]
    summary_cols = [*display_cols, "n_edges", "n_patients"]
    core_cols = [
        "method",
        "coverage",
        "size",
        "time_sec",
        "decision_ambiguity",
        "action_union_size",
        "edge_cov",
        "avg_size",
        "avg_ref_size",
        *action_cov_cols,
    ]

    runs = []
    diagnostics = []
    for seed in range(args.seeds):
        df, diag = run_one(
            seed,
            args.n,
            args.frac,
            args.alpha,
            args.batch_size,
            args.score,
            args.data_root,
            args.cuda,
        )
        runs.append(df)
        diagnostics.append(diag)
        print(f"\nSeed {seed} diagnostics: {diag}")
        print(table_string(round_for_output(to_percent_table(df)[display_cols])))

    raw = pd.concat(
        [df.assign(seed=seed) for seed, df in enumerate(runs)],
        ignore_index=True,
    )
    summary = aggregate_runs(runs)
    display_summary = round_for_output(to_percent_table(summary)[summary_cols])
    format_for_csv(raw).to_csv(args.out_dir / "raw.csv", index=False)
    display_summary.to_csv(args.out_dir / "summary.csv", index=False)
    format_for_csv(summary).to_csv(args.out_dir / "summary_full.csv", index=False)
    format_for_csv(summary[core_cols]).to_csv(args.out_dir / "core_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)

    print("\nAggregated means:")
    print(table_string(display_summary))
    print("\nCore metrics (raw scale):")
    print(table_string(round_for_output(summary[core_cols])))
    print(f"\nSaved CSV files under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
