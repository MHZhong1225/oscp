#!/usr/bin/env python
"""Run MIMIC-IV-ED diagnostic-acquisition OSCP experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.acquisition_oscp import (  # noqa: E402
    actionwise_selected_cp,
    aggregate_runs,
    edge_selection_thresholds,
    evaluate_acquisition_edge_sets,
    marginal_edge_cp,
)
from methods.oscp import relational_oscp_top  # noqa: E402
from methods.relational_core import RelationalSelectionConfig, select_top_edges  # noqa: E402
from methods.trainer.mimic_ed import (  # noqa: E402
    MIMIC_ED_ACTIONS,
    MIMIC_ED_CRITICAL_CLASS,
    MIMIC_ED_LABELS,
    fit_mimic_ed_models,
    load_mimic_ed_acquisition,
    make_mimic_ed_splits,
)


def to_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percent_cols = [
        c
        for c in out.columns
        if c.endswith("_cov")
        or c.endswith("_miss_rate")
        or c.endswith("_hit_rate")
        or c.endswith("_selected_rate")
        or c.endswith("_test_rate")
        or c.endswith("_under_gap")
        or c.endswith("_cov_gap")
        or c.endswith("_threshold_cov_gap")
        or c
        in {
            "coverage",
            "edge_cov",
            "action_cov_gap",
            "worst_under_gap",
            "easy_edge_cov",
            "hard_edge_cov",
            "very_hard_edge_cov",
            "threshold_cov_gap",
        }
    ]
    for col in percent_cols:
        if col in out.columns:
            out[col] = 100.0 * out[col]
    return out


def existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _ordered_arrays(
    batch_mode: str,
    seed: int,
    probs: np.ndarray,
    support: np.ndarray,
    labels: np.ndarray,
    actions: np.ndarray,
    intime: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if batch_mode == "arrival":
        order = np.argsort(intime, kind="mergesort")
    elif batch_mode == "random":
        order = np.random.default_rng(seed).permutation(labels.size)
    else:
        raise ValueError(f"unknown batch_mode: {batch_mode}")
    return probs[order], support[order], labels[order], actions[order]


def run_one(
    seed: int,
    n: int | None,
    frac: float | None,
    alpha: float,
    batch_size: int,
    capacities: Sequence[int],
    score: str,
    ed_root: Path,
    hosp_root: Path,
    action_window_hours: float,
    batch_mode: str,
    chunksize: int,
    max_text_features: int,
    min_frequency: int,
    cuda: int | None,
    cache_dir: Path | None,
) -> tuple[pd.DataFrame, dict]:
    data = load_mimic_ed_acquisition(
        ed_root=ed_root,
        hosp_root=hosp_root,
        n=n,
        frac=frac,
        seed=seed,
        action_window_hours=action_window_hours,
        chunksize=chunksize,
        cache_dir=cache_dir,
    )
    split = make_mimic_ed_splits(data, seed=seed)
    outcome_model, support_model, model_info = fit_mimic_ed_models(
        split,
        seed=seed,
        max_text_features=max_text_features,
        min_frequency=min_frequency,
        cuda=cuda,
    )

    config = RelationalSelectionConfig(
        batch_size=batch_size,
        capacities=np.asarray(capacities, dtype=int),
    )

    cal_probs = outcome_model.predict_proba(split.x_cal)
    test_probs = outcome_model.predict_proba(split.x_test)
    cal_support = support_model.predict_support(split.x_cal)
    test_support = support_model.predict_support(split.x_test)

    cal_probs, cal_support, cal_labels, _ = _ordered_arrays(
        batch_mode=batch_mode,
        seed=seed + 1000,
        probs=cal_probs,
        support=cal_support,
        labels=split.y_cal,
        actions=split.actions_cal,
        intime=split.intime_cal,
    )
    test_probs, test_support, test_labels, test_actions = _ordered_arrays(
        batch_mode=batch_mode,
        seed=seed + 2000,
        probs=test_probs,
        support=test_support,
        labels=split.y_test,
        actions=split.actions_test,
        intime=split.intime_test,
    )

    selection = select_top_edges(test_support, config)
    if selection.n_edges == 0:
        raise ValueError(
            f"No selected edges. Reduce --batch-size below n_test={test_labels.size}."
        )
    edge_difficulty = edge_selection_thresholds(test_support, selection, config)

    timed_results = []

    def add_timed_result(make_result):
        start = time.perf_counter()
        result = make_result()
        timed_results.append((result, time.perf_counter() - start))

    add_timed_result(
        lambda: marginal_edge_cp(
            cal_probs=cal_probs,
            cal_labels=cal_labels,
            test_probs=test_probs,
            selection=selection,
            alpha=alpha,
            score=score,
        )
    )
    add_timed_result(
        lambda: actionwise_selected_cp(
            cal_probs=cal_probs,
            cal_labels=cal_labels,
            cal_support=cal_support,
            test_probs=test_probs,
            selection=selection,
            config=config,
            alpha=alpha,
            score=score,
        )
    )
    add_timed_result(
        lambda: relational_oscp_top(
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
        )
    )

    rows = []
    for result, time_sec in timed_results:
        row = evaluate_acquisition_edge_sets(
            result=result,
            labels=test_labels,
            action_names=MIMIC_ED_ACTIONS.tolist(),
            nominal=1.0 - alpha,
            critical_label=MIMIC_ED_CRITICAL_CLASS,
            edge_difficulty=edge_difficulty,
        )
        row["time_sec"] = time_sec
        rows.append(row)

    observed_selected = test_actions[
        selection.unit_indices,
        selection.action_indices,
    ]
    label_counts = np.bincount(data.y, minlength=MIMIC_ED_LABELS.size)
    diagnostics = {
        "seed": seed,
        "score": score,
        "batch_mode": batch_mode,
        "sample_frac": frac if frac is not None else np.nan,
        "device": model_info["device"],
        "action_window_hours": action_window_hours,
        "val_log_loss": model_info["val_log_loss"],
        "encoded_dim": model_info["encoded_dim"],
        "n_total": int(data.y.size),
        "n_cal": int(cal_labels.size),
        "n_test": int(test_labels.size),
        "n_edges": int(selection.n_edges),
        "n_selected_patients": int(np.unique(selection.unit_indices).size),
        "edge_patient_ratio": float(
            selection.n_edges / np.unique(selection.unit_indices).size
        ),
        "selected_observed_action_rate": float(np.mean(observed_selected)),
        "mean_support": float(np.mean(test_support)),
        "max_support": float(np.max(test_support)),
        "mean_edge_threshold": float(
            np.mean(edge_difficulty[np.isfinite(edge_difficulty)])
        ),
        **{
            f"label_{name}_n": int(label_counts[i])
            for i, name in enumerate(MIMIC_ED_LABELS)
        },
        **{
            f"{name}_prevalence": float(data.actions[:, i].mean())
            for i, name in enumerate(MIMIC_ED_ACTIONS)
        },
        **{
            f"{name}_selected_observed_rate": float(
                np.mean(
                    test_actions[
                        selection.unit_indices[selection.action_indices == i],
                        i,
                    ]
                )
            )
            for i, name in enumerate(MIMIC_ED_ACTIONS)
            if np.any(selection.action_indices == i)
        },
    }
    for key, value in model_info.items():
        if key.endswith("_val_log_loss"):
            diagnostics[key] = value

    return pd.DataFrame(rows), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[20, 10, 10, 15, 5],
        help="Per-action top-B capacities: lab cardiac infection medication escalation.",
    )
    parser.add_argument("--score", choices=["lac", "aps"], default="lac")
    parser.add_argument(
        "--batch-mode",
        choices=["arrival", "random"],
        default="arrival",
        help="arrival sorts cal/test ED stays by ED intime before fixed-size batching.",
    )
    parser.add_argument("--action-window-hours", type=float, default=6.0)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--max-text-features", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=20)
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index, defaults to 0.")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed diagnostics and tables.")
    parser.add_argument("--cache-dir", type=Path, default=Path("results/cache/mimic_ed"))
    parser.add_argument("--ed-root", type=Path, default=Path("dataset/mimic-iv-ed-2.2"))
    parser.add_argument("--hosp-root", type=Path, default=Path("dataset/mimic-iv-3.1"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/mimic_iv_ed_acquisition"),
    )
    args = parser.parse_args()

    if len(args.capacities) != MIMIC_ED_ACTIONS.size:
        raise ValueError(
            f"--capacities length must be {MIMIC_ED_ACTIONS.size}, got {len(args.capacities)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    display_cols = [
        "method",
        "coverage",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        "action_cov_gap",
        "worst_under_gap",
        "critical_miss_rate",
        "easy_edge_cov",
        "hard_edge_cov",
        "very_hard_edge_cov",
        "threshold_cov_gap",
        "hard_edge_under_gap",
        "very_hard_under_gap",
        "n_edges",
        "n_patients",
        *[f"{a}_cov" for a in MIMIC_ED_ACTIONS],
    ]
    core_display_cols = [
        "method",
        "coverage",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        "critical_miss_rate",
        "time_sec",
    ]

    runs: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for seed in tqdm(range(args.seeds), desc="Seeds", unit="seed"):
        df, diag = run_one(
            seed=seed,
            n=args.n,
            frac=args.frac,
            alpha=args.alpha,
            batch_size=args.batch_size,
            capacities=args.capacities,
            score=args.score,
            ed_root=args.ed_root,
            hosp_root=args.hosp_root,
            action_window_hours=args.action_window_hours,
            batch_mode=args.batch_mode,
            chunksize=args.chunksize,
            max_text_features=args.max_text_features,
            min_frequency=args.min_frequency,
            cuda=args.cuda,
            cache_dir=args.cache_dir,
        )
        runs.append(df)
        diagnostics.append(diag)
        if args.verbose:
            print(f"\nSeed {seed} diagnostics: {diag}")
            percent = to_percent_table(df)
            print(
                percent[existing_cols(percent, display_cols)].to_string(
                    index=False,
                    float_format=lambda x: f"{x:0.2f}",
                )
            )
        else:
            percent = to_percent_table(df)
            print(f"\nSeed {seed} core metrics:")
            print(
                percent[existing_cols(percent, core_display_cols)].to_string(
                    index=False,
                    float_format=lambda x: f"{x:0.2f}",
                )
            )

    raw = pd.concat(
        [df.assign(seed=seed) for seed, df in enumerate(runs)],
        ignore_index=True,
    )
    summary = aggregate_runs(runs)
    display_summary = to_percent_table(summary)

    raw.to_csv(args.out_dir / "raw.csv", index=False)
    summary.to_csv(args.out_dir / "summary_full.csv", index=False)
    display_summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)
    summary[existing_cols(summary, display_cols)].to_csv(
        args.out_dir / "core_metrics.csv",
        index=False,
    )

    print("\nAggregated means:")
    print(
        display_summary[existing_cols(display_summary, display_cols)].to_string(
            index=False,
            float_format=lambda x: f"{x:0.2f}",
        )
    )
    print(f"\nSaved CSV files under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
