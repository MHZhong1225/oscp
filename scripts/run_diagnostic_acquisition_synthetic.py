#!/usr/bin/env python
"""Run synthetic diagnostic-acquisition OSCP experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.acquisition_oscp import (
    DEFAULT_ACTIONS,
    aggregate_runs,
    run_one_acquisition_oscp,
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
            "oracle_action_hit_rate",
            "hidden_test_rate",
            "hidden_selected_rate",
            "hidden_action_hit_rate",
            "easy_edge_cov",
            "hard_edge_cov",
            "very_hard_edge_cov",
            "threshold_cov_gap",
        }
    ]
    for c in percent_cols:
        if c in out.columns:
            out[c] = 100.0 * out[c]
    return out


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        default=[5, 5, 5, 5],
        help="Per-diagnostic top-B capacities. Must match number of actions.",
    )
    parser.add_argument("--score", choices=["lac", "aps"], default="lac")

    parser.add_argument("--hidden-rate", type=float, default=0.10)
    parser.add_argument("--hidden-signal", type=float, default=6.0)
    parser.add_argument("--hidden-critical-prob", type=float, default=0.85)

    parser.add_argument("--entropy-weight", type=float, default=0.2)
    parser.add_argument("--critical-gain-weight", type=float, default=4.0)
    parser.add_argument(
        "--base-feature-dim",
        type=int,
        default=4,
        help="Number of base features visible to the current predictor. Use <=0 for all.",
    )

    parser.add_argument(
        "--support-mode",
        choices=["oracle", "learned_hidden", "entropy_gain"],
        default="entropy_gain",
        help="How to construct diagnostic action support scores.",
    )
    parser.add_argument("--oracle-boost", type=float, default=1.0)
    parser.add_argument("--oracle-noise", type=float, default=0.02)

    parser.add_argument(
        "--batch-context",
        choices=["none", "clustered"],
        default="clustered",
        help=(
            "Whether to reorder cal/test examples by route hardness before batching. "
            "clustered creates heterogeneous screening rounds and enables hard-edge diagnostics."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/diagnostic_acquisition_synthetic"),
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-seed diagnostics and tables.")
    args = parser.parse_args()

    if len(args.capacities) != len(DEFAULT_ACTIONS):
        raise ValueError(
            f"--capacities length must be {len(DEFAULT_ACTIONS)}, got {len(args.capacities)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[pd.DataFrame] = []
    diagnostics: list[dict] = []

    display_cols = [
        "method",
        "coverage",
        "edge_cov",
        "edge_size",
        "avg_ref_size",
        "action_cov_gap",
        "worst_under_gap",
        "critical_miss_rate",
        "oracle_action_hit_rate",
        "hidden_test_rate",
        "hidden_selected_rate",
        "hidden_action_hit_rate",
        "easy_edge_cov",
        "hard_edge_cov",
        "very_hard_edge_cov",
        "threshold_cov_gap",
        "hard_edge_under_gap",
        "very_hard_under_gap",
        "n_edges",
        "n_patients",
        *[f"{a}_cov" for a in DEFAULT_ACTIONS],
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

    for seed in tqdm(range(args.seeds), desc="Seeds", unit="seed"):
        df, diag = run_one_acquisition_oscp(
            seed=seed,
            n=args.n,
            alpha=args.alpha,
            batch_size=args.batch_size,
            capacities=args.capacities,
            score=args.score,
            hidden_rate=args.hidden_rate,
            hidden_signal=args.hidden_signal,
            hidden_critical_prob=args.hidden_critical_prob,
            entropy_weight=args.entropy_weight,
            critical_gain_weight=args.critical_gain_weight,
            base_feature_dim=None if args.base_feature_dim <= 0 else args.base_feature_dim,
            support_mode=args.support_mode,
            oracle_boost=args.oracle_boost,
            oracle_noise=args.oracle_noise,
            batch_context=args.batch_context,
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
