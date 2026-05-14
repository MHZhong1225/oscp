"""Edge-label OSCP utilities for ChEMBL target-screening experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from scipy import sparse
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from methods.baselines.relational import self_calibrating_score_thresholds
from methods.conformal import conformal_quantile
from methods.relational_core import (
    EdgeSelection,
    EdgeSetResult,
    RelationalSelectionConfig,
    reference_mask_top_capacity,
)


CHEMBL_FEATURE_CACHE_DIR = Path("results/cache/chembl36_features")


@dataclass(frozen=True)
class ChemblData:
    compounds: pd.DataFrame
    edges: pd.DataFrame
    action_names: tuple[str, ...]
    target_chembl_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChemblSplitData:
    x_train: sparse.csr_matrix
    x_val: sparse.csr_matrix
    x_cal: sparse.csr_matrix
    x_test: sparse.csr_matrix

    y_train: np.ndarray
    y_val: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray

    observed_train: np.ndarray
    observed_val: np.ndarray
    observed_cal: np.ndarray
    observed_test: np.ndarray

    train_compounds: pd.DataFrame
    val_compounds: pd.DataFrame
    cal_compounds: pd.DataFrame
    test_compounds: pd.DataFrame


class MultiTaskMLP(torch.nn.Module):
    """Shared compound encoder with one binary activity logit per target."""

    def __init__(self, d_in: int, n_targets: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ChemblMultiTaskActivityPredictor:
    model: MultiTaskMLP
    device: str
    batch_size: int = 4096

    def predict_active_proba(self, x: sparse.csr_matrix) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for ChEMBL multi-task training.")
        self.model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                stop = min(start + self.batch_size, x.shape[0])
                x_batch = _as_dense_float32(x[start:stop])
                x_tensor = torch.as_tensor(x_batch, dtype=torch.float32, device=self.device)
                batch_probs = torch.sigmoid(self.model(x_tensor)).cpu().numpy()
                probs.append(batch_probs)
        if probs:
            return np.vstack(probs)
        return np.empty((0, self.model.net[-1].out_features))


def extract_raw_chembl_edges(
    sqlite_path: Path,
    out_path: Path,
    include_censored_inactive: bool = False,
) -> pd.DataFrame:
    """
    Extract strict ChEMBL36 activity records for human direct single-protein binding tasks.

    Default behavior is the strict exact-only baseline:
        - Homo sapiens
        - SINGLE PROTEIN
        - assay confidence_score = 9
        - binding assay only: assay_type = 'B'
        - exact standardized nM measurements only: standard_relation = '='
        - pchembl_value is not null
        - standard_type in IC50/Ki/Kd
        - potential duplicates removed
        - molecule is mapped to its ChEMBL parent compound before aggregation

    If include_censored_inactive=True, the extractor also keeps censored inactive
    observations such as IC50 > 10000 nM. These rows are not treated as exact
    pChEMBL values later; they are used only as binary inactive evidence.
    """
    import sqlite3

    activity_condition = """
        (
            -- strict exact pChEMBL records
            act.pchembl_value IS NOT NULL
            AND act.standard_relation = '='
        )
    """
    if include_censored_inactive:
        activity_condition = """
        (
            (
                -- strict exact pChEMBL records
                act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='
            )
            OR
            (
                -- censored inactive records, e.g. IC50 > 10000 nM
                act.standard_relation IN ('>', '>=')
                AND act.standard_value >= 10000
            )
        )
        """

    query = f"""
    SELECT
        COALESCE(pmd.chembl_id, md.chembl_id) AS compound_chembl_id,
        COALESCE(pcs.canonical_smiles, cs.canonical_smiles) AS canonical_smiles,
        md.chembl_id AS original_compound_chembl_id,
        cs.canonical_smiles AS original_canonical_smiles,

        td.chembl_id AS target_chembl_id,
        td.pref_name AS target_name,
        td.target_type AS target_type,
        td.organism AS organism,

        a.chembl_id AS assay_chembl_id,
        a.assay_type AS assay_type,
        a.confidence_score AS confidence_score,

        act.standard_type AS standard_type,
        act.standard_relation AS standard_relation,
        act.standard_value AS standard_value,
        act.standard_units AS standard_units,
        act.pchembl_value AS pchembl_value,
        act.data_validity_comment AS data_validity_comment,
        act.potential_duplicate AS potential_duplicate

    FROM activities act
    JOIN assays a
        ON act.assay_id = a.assay_id
    JOIN target_dictionary td
        ON a.tid = td.tid
    JOIN molecule_dictionary md
        ON act.molregno = md.molregno
    JOIN compound_structures cs
        ON md.molregno = cs.molregno

    -- Map salts / alternative forms to the ChEMBL parent compound.
    LEFT JOIN molecule_hierarchy mh
        ON md.molregno = mh.molregno
    LEFT JOIN molecule_dictionary pmd
        ON COALESCE(mh.parent_molregno, md.molregno) = pmd.molregno
    LEFT JOIN compound_structures pcs
        ON COALESCE(mh.parent_molregno, md.molregno) = pcs.molregno

    WHERE
        COALESCE(pcs.canonical_smiles, cs.canonical_smiles) IS NOT NULL

        -- Human direct single-protein targets only.
        AND td.organism = 'Homo sapiens'
        AND td.target_type = 'SINGLE PROTEIN'
        AND a.confidence_score = 9

        -- Binding assays only. This avoids mixing binding and functional endpoints.
        AND a.assay_type = 'B'
        -- Standardized nM activity records.
        AND act.standard_units = 'nM'
        AND act.standard_value > 0
        AND act.standard_type IN ('IC50', 'Ki', 'Kd')
        -- Remove suspicious or duplicated records.
        AND (
            act.data_validity_comment IS NULL
            OR act.data_validity_comment = 'Manually validated'
        )
        AND (
            act.potential_duplicate IS NULL
            OR act.potential_duplicate = 0
        )

        AND {activity_condition}
    """

    with sqlite3.connect(sqlite_path) as conn:
        raw = pd.read_sql_query(query, conn)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out_path, index=False)
    return raw


def aggregate_compound_target_edges(
    raw: pd.DataFrame,
    out_path: Path | None = None,
    active_threshold: float = 6.0,
    inactive_threshold: float = 5.0,
    max_pchembl_range: float = 1.5,
    drop_gray_exact_pairs: bool = True,
) -> pd.DataFrame:
    """
    Aggregate repeated ChEMBL records into one binary parent-compound/target edge.

    Exact pChEMBL records are aggregated by median pChEMBL:
        - median pChEMBL >= active_threshold   -> active, y = 1
        - median pChEMBL <= inactive_threshold -> inactive, y = 0
        - inactive_threshold < median < active_threshold -> gray zone, dropped by default

    Optional censored inactive rows, if present in raw, are handled only as binary
    inactive evidence:
        - standard_relation in {'>', '>='} and standard_value >= 10000 nM -> y = 0 evidence

    Censored rows are never included in pChEMBL median/range calculations.
    Compound-target pairs with conflicting active and inactive evidence are dropped.
    """
    raw = raw.copy()
    raw["pchembl_value"] = pd.to_numeric(raw.get("pchembl_value"), errors="coerce")
    raw["standard_value"] = pd.to_numeric(raw.get("standard_value"), errors="coerce")

    required = [
        "compound_chembl_id",
        "canonical_smiles",
        "target_chembl_id",
        "target_name",
        "standard_relation",
        "standard_value",
        "standard_units",
    ]
    raw = raw.dropna(subset=required).copy()

    group_cols = [
        "compound_chembl_id",
        "canonical_smiles",
        "target_chembl_id",
        "target_name",
    ]

    exact = raw[
        raw["standard_relation"].eq("=")
        & raw["pchembl_value"].notna()
    ].copy()

    censored_inactive = raw[
        raw["standard_relation"].isin([">", ">="])
        & raw["standard_units"].eq("nM")
        & raw["standard_value"].ge(10000.0)
    ].copy()

    # Aggregate exact pChEMBL records.
    if exact.empty:
        exact_agg = pd.DataFrame(columns=group_cols)
    else:
        exact_agg = (
            exact.groupby(group_cols, as_index=False)
            .agg(
                pchembl_median=("pchembl_value", "median"),
                pchembl_min=("pchembl_value", "min"),
                pchembl_max=("pchembl_value", "max"),
                n_exact_measurements=("pchembl_value", "size"),
                assay_types=("assay_type", lambda x: ",".join(sorted(set(map(str, x))))),
                standard_types=("standard_type", lambda x: ",".join(sorted(set(map(str, x))))),
            )
        )
        exact_agg["pchembl_range"] = exact_agg["pchembl_max"] - exact_agg["pchembl_min"]
        exact_agg["exact_y"] = np.nan
        exact_agg.loc[exact_agg["pchembl_median"].ge(active_threshold), "exact_y"] = 1
        exact_agg.loc[exact_agg["pchembl_median"].le(inactive_threshold), "exact_y"] = 0
        exact_agg["exact_is_gray"] = exact_agg["exact_y"].isna()

        # Remove exact pairs with large experimental disagreement.
        exact_agg = exact_agg[
            (exact_agg["n_exact_measurements"] <= 1)
            | exact_agg["pchembl_range"].le(max_pchembl_range)
        ].copy()

    # Aggregate censored inactive evidence separately.
    if censored_inactive.empty:
        censored_agg = pd.DataFrame(columns=group_cols + ["n_censored_inactive"])
    else:
        censored_agg = (
            censored_inactive.groupby(group_cols, as_index=False)
            .agg(n_censored_inactive=("standard_value", "size"))
        )

    # Outer-merge exact and censored evidence so censored-only inactive pairs can be kept.
    agg = exact_agg.merge(censored_agg, on=group_cols, how="outer")
    if agg.empty:
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            agg.to_parquet(out_path, index=False)
        return agg

    if "n_exact_measurements" not in agg.columns:
        agg["n_exact_measurements"] = 0
    else:
        agg["n_exact_measurements"] = pd.to_numeric(
            agg["n_exact_measurements"],
            errors="coerce",
        ).fillna(0).astype(int)

    if "n_censored_inactive" not in agg.columns:
        agg["n_censored_inactive"] = 0
    else:
        agg["n_censored_inactive"] = pd.to_numeric(
            agg["n_censored_inactive"],
            errors="coerce",
        ).fillna(0).astype(int)

    if "exact_is_gray" not in agg.columns:
        agg["exact_is_gray"] = False
    else:
        agg["exact_is_gray"] = agg["exact_is_gray"].fillna(False).astype(bool)

    if "exact_y" not in agg.columns:
        agg["exact_y"] = np.nan

    agg["has_censored_inactive"] = agg["n_censored_inactive"].gt(0)

    # Start with exact labels, then add censored-only inactive labels.
    agg["y"] = agg["exact_y"]
    censored_only = agg["y"].isna() & agg["has_censored_inactive"] & ~agg["exact_is_gray"]
    agg.loc[censored_only, "y"] = 0

    # Drop gray exact pairs by default, even if they also have censored inactive evidence.
    if drop_gray_exact_pairs:
        agg = agg[~agg["exact_is_gray"]].copy()

    # Drop conflicts: exact active together with censored inactive evidence.
    conflict = agg["exact_y"].eq(1) & agg["has_censored_inactive"]
    agg = agg[~conflict].copy()

    # Drop unlabeled rows.
    agg = agg.dropna(subset=["y"]).copy()
    agg["y"] = agg["y"].astype(int)

    agg["evidence_source"] = "exact"
    agg.loc[
        agg["n_exact_measurements"].eq(0) & agg["has_censored_inactive"],
        "evidence_source",
    ] = "censored_inactive_only"
    agg.loc[
        agg["n_exact_measurements"].gt(0) & agg["has_censored_inactive"],
        "evidence_source",
    ] = "exact_plus_censored_inactive"

    # Keep stable column order for downstream code and analysis.
    preferred_cols = [
        "compound_chembl_id",
        "canonical_smiles",
        "target_chembl_id",
        "target_name",
        "y",
        "pchembl_median",
        "pchembl_min",
        "pchembl_max",
        "pchembl_range",
        "n_exact_measurements",
        "n_censored_inactive",
        "evidence_source",
        "assay_types",
        "standard_types",
    ]
    remaining_cols = [c for c in agg.columns if c not in preferred_cols]
    agg = agg[[c for c in preferred_cols if c in agg.columns] + remaining_cols]

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
    return agg


def select_chembl_targets(
    edges: pd.DataFrame,
    out_path: Path | None = None,
    n_targets: int = 30,
    min_edges: int = 1000,
    min_active: int = 100,
    min_inactive: int = 100,
) -> pd.DataFrame:
    """Select high-support targets and recode compound/target ids."""
    stats = (
        edges.groupby("target_chembl_id")
        .agg(
            n_edges=("y", "size"),
            n_active=("y", "sum"),
            target_name=("target_name", "first"),
        )
        .reset_index()
    )
    stats["n_inactive"] = stats["n_edges"] - stats["n_active"]
    eligible = stats[
        (stats["n_edges"] >= min_edges)
        & (stats["n_active"] >= min_active)
        & (stats["n_inactive"] >= min_inactive)
    ].copy()
    selected = (
        eligible.sort_values("n_edges", ascending=False)
        .head(n_targets)["target_chembl_id"]
        .tolist()
    )
    out = edges[edges["target_chembl_id"].isin(selected)].copy()
    target_order = (
        out.groupby("target_chembl_id")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    target_map = {target: i for i, target in enumerate(target_order)}
    out["target_id"] = out["target_chembl_id"].map(target_map).astype(int)
    out["compound_id"] = pd.Categorical(out["compound_chembl_id"]).codes.astype(int)
    out = out.sort_values(["compound_id", "target_id"]).reset_index(drop=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out


def prepare_chembl36_processed_files(
    data_root: Path,
    sqlite_path: Path | None = None,
    n_targets: int = 30,
    force: bool = False,
    include_censored_inactive: bool = False,
) -> Path:
    """Create raw, aggregated, and selected-target parquet files if needed."""
    data_root.mkdir(parents=True, exist_ok=True)

    suffix = "exact_plus_censored_inactive" if include_censored_inactive else "exact"
    raw_path = data_root / f"raw_human_direct_single_protein_binding_{suffix}.parquet"
    compound_target_path = data_root / f"compound_target_edges_{suffix}.parquet"
    selected_path = data_root / f"chembl_oscp_edges_{suffix}_top{n_targets}.parquet"

    if force or not raw_path.exists():
        if sqlite_path is None:
            sqlite_path = data_root / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
        extract_raw_chembl_edges(
            sqlite_path=sqlite_path,
            out_path=raw_path,
            include_censored_inactive=include_censored_inactive,
        )

    if force or not compound_target_path.exists():
        raw = pd.read_parquet(raw_path)
        aggregate_compound_target_edges(raw, compound_target_path)

    if force or not selected_path.exists():
        edges = pd.read_parquet(compound_target_path)
        select_chembl_targets(edges, selected_path, n_targets=n_targets)

    return selected_path


def _split_compounds(
    compounds: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    compound_ids = compounds["compound_chembl_id"].to_numpy()
    train_ids, rest_ids = train_test_split(
        compound_ids,
        train_size=0.50,
        random_state=seed,
    )
    val_ids, rest_ids = train_test_split(
        rest_ids,
        train_size=0.30,
        random_state=seed + 1,
    )
    cal_ids, test_ids = train_test_split(
        rest_ids,
        train_size=3.0 / 7.0,
        random_state=seed + 2,
    )
    split = pd.Series(index=compound_ids, dtype=object)
    split.loc[train_ids] = "train"
    split.loc[val_ids] = "val"
    split.loc[cal_ids] = "cal"
    split.loc[test_ids] = "test"
    out = compounds.copy()
    out["split"] = out["compound_chembl_id"].map(split)
    return out


def load_chembl36_data(
    data_root: Path = Path("dataset/chembl36"),
    edges_path: Path | None = None,
    n_targets: int = 30,
    use_existing_split: bool = False,
    seed: int = 0,
    force_rebuild: bool = False,
    sqlite_path: Path | None = None,
    include_censored_inactive: bool = False,
) -> ChemblData:
    """Load selected ChEMBL target-screening edges and compound split metadata."""
    if edges_path is None:
        suffix = "exact_plus_censored_inactive" if include_censored_inactive else "exact"
        selected_path = prepare_chembl36_processed_files(
            data_root=data_root,
            sqlite_path=sqlite_path,
            n_targets=n_targets,
            force=force_rebuild,
            include_censored_inactive=include_censored_inactive,
        )
        split_path = data_root / f"chembl_oscp_edges_{suffix}_top{n_targets}_split.parquet"
        if use_existing_split and split_path.exists() and not force_rebuild:
            edges_path = split_path
        else:
            edges_path = selected_path

    edges = pd.read_parquet(edges_path).copy()
    if n_targets is not None and edges["target_chembl_id"].nunique() > n_targets:
        edges = select_chembl_targets(edges, n_targets=n_targets)

    target_meta = (
        edges.groupby(["target_id", "target_chembl_id"], as_index=False)
        .agg(target_name=("target_name", "first"), n_edges=("y", "size"))
        .sort_values("target_id")
    )
    target_id_map = {
        old: new for new, old in enumerate(target_meta["target_id"].astype(int).tolist())
    }
    if any(old != new for old, new in target_id_map.items()):
        edges["target_id"] = edges["target_id"].map(target_id_map).astype(int)
        target_meta["target_id"] = target_meta["target_id"].map(target_id_map).astype(int)
        target_meta = target_meta.sort_values("target_id")

    compounds = (
        edges[["compound_chembl_id", "canonical_smiles"] + (["split"] if "split" in edges.columns else [])]
        .drop_duplicates("compound_chembl_id")
        .sort_values("compound_chembl_id")
        .reset_index(drop=True)
    )
    if "split" not in compounds.columns:
        compounds = _split_compounds(compounds, seed=seed)

    split_map = compounds.set_index("compound_chembl_id")["split"]
    edges["split"] = edges["compound_chembl_id"].map(split_map)
    edges = edges.dropna(subset=["split"]).copy()
    action_names = tuple(
        f"{row.target_name} ({row.target_chembl_id})"
        for row in target_meta.itertuples(index=False)
    )
    target_chembl_ids = tuple(target_meta["target_chembl_id"].tolist())
    return ChemblData(
        compounds=compounds,
        edges=edges,
        action_names=action_names,
        target_chembl_ids=target_chembl_ids,
    )


def _morgan_fingerprint_matrix(
    smiles: Sequence[str],
    n_bits: int,
    radius: int,
) -> sparse.csr_matrix:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows: list[int] = []
    cols: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        on_bits = list(fp.GetOnBits())
        rows.extend([i] * len(on_bits))
        cols.extend(on_bits)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(smiles), n_bits), dtype=np.float32)


def _compound_feature_cache_digest(compounds: pd.DataFrame) -> str:
    """Stable short digest so feature caches cannot collide across different datasets."""
    key_cols = ["compound_chembl_id", "canonical_smiles"]
    payload = (
        compounds[key_cols]
        .fillna("")
        .astype(str)
        .sort_values(key_cols)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha1(payload).hexdigest()[:12]


def _compound_feature_cache_path(
    cache_digest: str,
    n_rows: int,
    n_bits: int,
    radius: int,
) -> Path:
    return CHEMBL_FEATURE_CACHE_DIR / (
        f"morgan_{cache_digest}_n{n_rows}_bits{n_bits}_r{radius}.npz"
    )


def build_compound_features(
    compounds: pd.DataFrame,
    fingerprint: str = "auto",
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[sparse.csr_matrix, str]:
    """Build compound features. auto uses Morgan when RDKit is installed."""
    smiles = compounds["canonical_smiles"].fillna("").astype(str).tolist()
    cache_digest = _compound_feature_cache_digest(compounds)
    if fingerprint not in {"auto", "morgan"}:
        raise ValueError(f"unknown fingerprint: {fingerprint}")

    cache_path = _compound_feature_cache_path(
        cache_digest,
        len(smiles),
        n_bits,
        radius,
    )
    if cache_path.exists():
        return sparse.load_npz(cache_path).tocsr(), "morgan"
    features = _morgan_fingerprint_matrix(smiles, n_bits=n_bits, radius=radius)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(cache_path, features)
    return features, "morgan"


def _label_matrix(
    compounds: pd.DataFrame,
    edges: pd.DataFrame,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    local = {
        chembl_id: i
        for i, chembl_id in enumerate(compounds["compound_chembl_id"].tolist())
    }
    y = np.full((compounds.shape[0], n_actions), -1, dtype=np.int8)
    sub = edges[edges["compound_chembl_id"].isin(local)].copy()
    rows = sub["compound_chembl_id"].map(local).to_numpy(dtype=int)
    cols = sub["target_id"].to_numpy(dtype=int)
    y[rows, cols] = sub["y"].to_numpy(dtype=np.int8)
    return y, y >= 0


def make_chembl_splits(
    data: ChemblData,
    fingerprint: str = "auto",
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[ChemblSplitData, str]:
    """Create compound-level train/val/cal/test matrices and sparse label masks."""
    compounds = data.compounds.sort_values("compound_chembl_id").reset_index(drop=True)
    features, resolved_fingerprint = build_compound_features(
        compounds,
        fingerprint=fingerprint,
        n_bits=n_bits,
        radius=radius,
    )
    n_actions = len(data.action_names)

    split_frames = {}
    split_indices = {}
    for split_name in ["train", "val", "cal", "test"]:
        mask = compounds["split"].eq(split_name).to_numpy()
        split_indices[split_name] = np.where(mask)[0]
        split_frames[split_name] = compounds.loc[mask].reset_index(drop=True)

    matrices = {}
    observed = {}
    for split_name, frame in split_frames.items():
        split_edges = data.edges[data.edges["split"].eq(split_name)]
        matrices[split_name], observed[split_name] = _label_matrix(
            frame,
            split_edges,
            n_actions=n_actions,
        )

    return (
        ChemblSplitData(
            x_train=features[split_indices["train"]],
            x_val=features[split_indices["val"]],
            x_cal=features[split_indices["cal"]],
            x_test=features[split_indices["test"]],
            y_train=matrices["train"],
            y_val=matrices["val"],
            y_cal=matrices["cal"],
            y_test=matrices["test"],
            observed_train=observed["train"],
            observed_val=observed["val"],
            observed_cal=observed["cal"],
            observed_test=observed["test"],
            train_compounds=split_frames["train"],
            val_compounds=split_frames["val"],
            cal_compounds=split_frames["cal"],
            test_compounds=split_frames["test"],
        ),
        resolved_fingerprint,
    )


def _as_dense_float32(x: sparse.csr_matrix | np.ndarray) -> np.ndarray:
    x_np = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return x_np.astype(np.float32, copy=False)


def _masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    loss_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    observed = observed.to(dtype=loss_raw.dtype)
    return (loss_raw * observed).sum() / observed.sum().clamp_min(1.0)


def _choose_torch_device(cuda: int | None) -> str:
    if cuda is not None and cuda >= 0 and torch.cuda.is_available():
        return f"cuda:{cuda}"
    return "cpu"


def _evaluate_masked_val_loss(
    model: MultiTaskMLP,
    x: sparse.csr_matrix,
    y: np.ndarray,
    observed: np.ndarray,
    device: str,
    batch_size: int,
) -> float:
    rows = np.where(observed.any(axis=1))[0]
    if rows.size == 0:
        return np.nan
    model.eval()
    total_loss = 0.0
    total_weight = 0
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            batch = rows[start : start + batch_size]
            x_batch = _as_dense_float32(x[batch])
            mask_batch = observed[batch]
            y_batch = np.where(mask_batch, y[batch], 0).astype(np.float32, copy=False)
            x_tensor = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            y_tensor = torch.as_tensor(y_batch, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)
            n_obs = int(mask_batch.sum())
            batch_loss = float(
                _masked_bce_with_logits(model(x_tensor), y_tensor, mask_tensor).item()
            )
            total_loss += batch_loss * n_obs
            total_weight += n_obs
    return float(total_loss / max(total_weight, 1))


def fit_chembl_multitask_model(
    split: ChemblSplitData,
    seed: int = 0,
    max_iter: int = 200,
    C: float = 1.0,
    cuda: int | None = None,
) -> tuple[ChemblMultiTaskActivityPredictor, dict[str, Any]]:
    """Fit a multi-task active/inactive predictor with one output head per target."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for ChEMBL multi-task training.")

    _ = C
    n_actions = split.y_train.shape[1]
    device = _choose_torch_device(cuda)
    torch_device = torch.device(device)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    train_edges = split.observed_train.sum(axis=0).astype(int).tolist()
    val_edges = split.observed_val.sum(axis=0).astype(int).tolist()
    train_rows = np.where(split.observed_train.any(axis=1))[0]
    if train_rows.size == 0:
        raise ValueError("No observed ChEMBL training labels.")

    model = MultiTaskMLP(split.x_train.shape[1], n_actions).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    batch_size = 2048
    patience = 30
    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0

    for _ in range(max_iter):
        model.train()
        rng.shuffle(train_rows)
        for start in range(0, train_rows.size, batch_size):
            batch = train_rows[start : start + batch_size]
            x_batch = _as_dense_float32(split.x_train[batch])
            mask_batch = split.observed_train[batch]
            y_batch = np.where(mask_batch, split.y_train[batch], 0).astype(
                np.float32,
                copy=False,
            )
            x_tensor = torch.as_tensor(x_batch, dtype=torch.float32, device=torch_device)
            y_tensor = torch.as_tensor(y_batch, dtype=torch.float32, device=torch_device)
            mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=torch_device)

            optimizer.zero_grad()
            loss = _masked_bce_with_logits(model(x_tensor), y_tensor, mask_tensor)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate_masked_val_loss(
            model,
            split.x_val,
            split.y_val,
            split.observed_val,
            device,
            batch_size=4096,
        )
        early_stop_loss = val_loss
        if np.isnan(early_stop_loss):
            early_stop_loss = _evaluate_masked_val_loss(
                model,
                split.x_train,
                split.y_train,
                split.observed_train,
                device,
                batch_size=4096,
            )
        if early_stop_loss + 1e-6 < best_val_loss:
            best_val_loss = early_stop_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch_device)

    predictor = ChemblMultiTaskActivityPredictor(model=model, device=device)
    val_probs = predictor.predict_active_proba(split.x_val)
    val_losses: list[float] = []
    for a in range(n_actions):
        val_mask = split.observed_val[:, a]
        y_val = split.y_val[val_mask, a].astype(int)
        if y_val.size > 0:
            p_val = np.clip(val_probs[val_mask, a], 1e-6, 1.0 - 1e-6)
            val_losses.append(
                float(
                    log_loss(
                        y_val,
                        np.column_stack([1.0 - p_val, p_val]),
                        labels=[0, 1],
                    )
                )
            )
        else:
            val_losses.append(np.nan)

    return (
        predictor,
        {
            "model_type": "multi_task_mlp",
            "mean_val_log_loss": float(np.nanmean(val_losses)),
            "target_val_log_loss": val_losses,
            "target_train_edges": train_edges,
            "target_val_edges": val_edges,
            "device": device,
        },
    )


def complete_batches(n: int, batch_size: int) -> list[np.ndarray]:
    usable = (n // batch_size) * batch_size
    return [np.arange(start, start + batch_size) for start in range(0, usable, batch_size)]


def select_top_edges_masked(
    support: np.ndarray,
    observed: np.ndarray,
    config: RelationalSelectionConfig,
) -> EdgeSelection:
    """Select per-action top-B edges among observed retrospective labels."""
    support = np.asarray(support, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if support.shape != observed.shape:
        raise ValueError("support and observed must have the same shape")
    if support.shape[1] != config.capacities.size:
        raise ValueError("support width must match number of capacities")

    units: list[int] = []
    actions: list[int] = []
    batches: list[int] = []
    for batch_id, batch in enumerate(complete_batches(support.shape[0], config.batch_size)):
        for a, cap in enumerate(config.capacities):
            valid_local = np.where(observed[batch, a])[0]
            if cap <= 0 or valid_local.size == 0:
                continue
            take = min(int(cap), valid_local.size)
            values = support[batch[valid_local], a]
            order = np.argsort(values, kind="mergesort")[-take:]
            selected_global = batch[valid_local[order]]
            units.extend(selected_global.tolist())
            actions.extend([a] * selected_global.size)
            batches.extend([batch_id] * selected_global.size)
    return EdgeSelection(
        unit_indices=np.asarray(units, dtype=int),
        action_indices=np.asarray(actions, dtype=int),
        batch_indices=np.asarray(batches, dtype=int),
    )


def edge_label_selection_thresholds(
    support: np.ndarray,
    observed: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    thresholds = np.empty(selection.n_edges, dtype=float)
    batches = complete_batches(support.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)

    for e, (j, a, batch_id) in enumerate(
        zip(selection.unit_indices, selection.action_indices, selection.batch_indices)
    ):
        batch = batches[int(batch_id)]
        local_j = int(j - batch_start[int(batch_id)])
        valid_local = np.where(observed[batch, int(a)])[0]
        valid_local = valid_local[valid_local != local_j]
        values = support[batch[valid_local], int(a)]
        capacity = int(config.capacities[int(a)])
        if values.size < capacity:
            thresholds[e] = -np.inf
        else:
            thresholds[e] = np.partition(values, values.size - capacity)[values.size - capacity]
    return thresholds


def _binary_label_scores(p_active: np.ndarray, y: np.ndarray) -> np.ndarray:
    p_active = np.asarray(p_active, dtype=float)
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, 1.0 - p_active, p_active)


def _binary_all_scores(p_active: np.ndarray) -> np.ndarray:
    p_active = np.asarray(p_active, dtype=float)
    return np.column_stack([p_active, 1.0 - p_active])


def _selected_edge_probs(
    probs: np.ndarray,
    selection: EdgeSelection,
) -> np.ndarray:
    return probs[selection.unit_indices, selection.action_indices]


def _observed_binary_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    return _binary_label_scores(probs[observed], labels[observed])


def edge_label_marginal_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_observed: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    method: str = "Marginal CP",
) -> EdgeSetResult:
    scores = _observed_binary_scores(cal_probs, cal_labels, cal_observed)
    q = conformal_quantile(scores, alpha)
    p_edge = _selected_edge_probs(test_probs, selection)
    edge_scores = _binary_all_scores(p_edge)
    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= q,
        thresholds=np.full(selection.n_edges, q),
        reference_sizes=np.full(selection.n_edges, scores.size, dtype=int),
        method=method,
    )


def edge_label_actionwise_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_observed: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    method: str = "Action-wise CP",
) -> EdgeSetResult:
    cal_selection = select_top_edges_masked(cal_probs, cal_observed, config)
    n_actions = cal_probs.shape[1]
    thresholds = np.empty(n_actions, dtype=float)
    ref_sizes = np.empty(n_actions, dtype=int)

    for a in range(n_actions):
        selected_mask = cal_selection.action_indices == a
        selected_units = cal_selection.unit_indices[selected_mask]
        if selected_units.size:
            scores = _binary_label_scores(
                cal_probs[selected_units, a],
                cal_labels[selected_units, a],
            )
        else:
            obs = cal_observed[:, a]
            scores = _binary_label_scores(cal_probs[obs, a], cal_labels[obs, a])
        thresholds[a] = conformal_quantile(scores, alpha)
        ref_sizes[a] = int(scores.size)

    p_edge = _selected_edge_probs(test_probs, selection)
    edge_scores = _binary_all_scores(p_edge)
    edge_thresholds = thresholds[selection.action_indices]
    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= edge_thresholds[:, None],
        thresholds=edge_thresholds,
        reference_sizes=ref_sizes[selection.action_indices],
        method=method,
    )


def edge_label_bonferroni_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_observed: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    divisor: int,
    method: str | None = None,
) -> EdgeSetResult:
    adjusted_method = method or f"Bonferroni CP (alpha/{divisor})"
    return edge_label_marginal_cp(
        cal_probs=cal_probs,
        cal_labels=cal_labels,
        cal_observed=cal_observed,
        test_probs=test_probs,
        selection=selection,
        alpha=alpha / divisor,
        method=adjusted_method,
    )


def edge_label_self_calibrating_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_observed: np.ndarray,
    test_probs: np.ndarray,
    selection: EdgeSelection,
    alpha: float,
    method: str = "SC-CP",
    num_bin_predictor: int = 15,
    num_bin_score: int = 60,
) -> EdgeSetResult:
    cal_scores = _observed_binary_scores(cal_probs, cal_labels, cal_observed)
    cal_predictor = np.minimum(cal_probs[cal_observed], 1.0 - cal_probs[cal_observed])
    p_edge = _selected_edge_probs(test_probs, selection)
    edge_predictor = np.minimum(p_edge, 1.0 - p_edge)
    thresholds = self_calibrating_score_thresholds(
        cal_scores,
        cal_predictor,
        edge_predictor,
        alpha,
        num_bin_predictor=num_bin_predictor,
        num_bin_score=num_bin_score,
    )
    edge_scores = _binary_all_scores(p_edge)
    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=np.full(selection.n_edges, cal_scores.size, dtype=int),
        method=method,
    )


def edge_label_jomi_unit_top(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_observed: np.ndarray,
    test_probs: np.ndarray,
    test_observed: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    method: str = "JOMI-unit",
) -> EdgeSetResult:
    cal_scores_matrix = np.full(cal_probs.shape, np.nan, dtype=float)
    cal_scores_matrix[cal_observed] = _observed_binary_scores(
        cal_probs,
        cal_labels,
        cal_observed,
    )
    p_edge = _selected_edge_probs(test_probs, selection)
    edge_scores = _binary_all_scores(p_edge)
    thresholds = np.empty(selection.n_edges, dtype=float)
    ref_sizes = np.empty(selection.n_edges, dtype=int)

    batches = complete_batches(test_probs.shape[0], config.batch_size)
    batch_start = np.array([batch[0] for batch in batches], dtype=int)
    mask_cache: dict[tuple[int, int], np.ndarray] = {}

    for edge_index, (unit_index, batch_index) in enumerate(
        zip(selection.unit_indices, selection.batch_indices)
    ):
        local_index = int(unit_index - batch_start[int(batch_index)])
        cache_key = (int(batch_index), local_index)
        reference_mask = mask_cache.get(cache_key)
        if reference_mask is None:
            batch = batches[int(batch_index)]
            reference_mask = np.zeros(cal_probs.shape, dtype=bool)
            for action_index, capacity in enumerate(config.capacities):
                if capacity <= 0:
                    continue
                valid_local = np.where(test_observed[batch, action_index])[0]
                valid_local = valid_local[valid_local != local_index]
                other_scores = test_probs[batch[valid_local], action_index]
                if other_scores.size < int(capacity):
                    selection_threshold = -np.inf
                else:
                    selection_threshold = np.partition(
                        other_scores,
                        other_scores.size - int(capacity),
                    )[other_scores.size - int(capacity)]
                candidate_units = cal_probs[:, action_index] >= selection_threshold
                reference_mask[:, action_index] = (
                    cal_observed[:, action_index] & candidate_units
                )
            mask_cache[cache_key] = reference_mask
        reference_scores = cal_scores_matrix[reference_mask]
        ref_sizes[edge_index] = int(reference_scores.size)
        thresholds[edge_index] = conformal_quantile(reference_scores, alpha)

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=ref_sizes,
        method=method,
    )


def edge_label_oscp_top(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_observed: np.ndarray,
    test_probs: np.ndarray,
    test_observed: np.ndarray,
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    method: str = "OSCP",
) -> EdgeSetResult:
    p_edge = _selected_edge_probs(test_probs, selection)
    edge_scores = _binary_all_scores(p_edge)
    edge_thresholds = edge_label_selection_thresholds(
        test_probs,
        test_observed,
        selection,
        config,
    )
    thresholds = np.empty(selection.n_edges, dtype=float)
    ref_sizes = np.empty(selection.n_edges, dtype=int)
    threshold_cache: dict[tuple[int, float], tuple[float, int]] = {}

    for e, a in enumerate(selection.action_indices):
        key = (int(a), float(edge_thresholds[e]))
        cached = threshold_cache.get(key)
        if cached is None:
            ref = cal_observed[:, a] & (cal_probs[:, a] >= edge_thresholds[e])
            scores = _binary_label_scores(cal_probs[ref, a], cal_labels[ref, a])
            cached = (conformal_quantile(scores, alpha), int(scores.size))
            threshold_cache[key] = cached
        thresholds[e], ref_sizes[e] = cached

    return EdgeSetResult(
        selection=selection,
        sets=edge_scores <= thresholds[:, None],
        thresholds=thresholds,
        reference_sizes=ref_sizes,
        method=method,
    )


def evaluate_edge_label_sets(
    result: EdgeSetResult,
    labels: np.ndarray,
    action_names: Sequence[str],
    nominal: float = 0.90,
    edge_difficulty: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    y_edge = labels[result.selection.unit_indices, result.selection.action_indices].astype(int)
    covered = result.sets[np.arange(result.selection.n_edges), y_edge]
    sizes = result.sets.sum(axis=1)
    active = y_edge == 1

    row: dict[str, float | int | str] = {
        "method": result.method,
        "coverage": float(np.mean(covered)),
        "edge_cov": float(np.mean(covered)),
        "edge_size": float(np.mean(sizes)),
        "avg_ref_size": float(np.mean(result.reference_sizes)),
        "n_edges": int(result.selection.n_edges),
        "n_compounds": int(np.unique(result.selection.unit_indices).size),
        "selected_active_rate": float(np.mean(active)),
        "active_selected_n": int(np.sum(active)),
        "active_cov": float(np.mean(covered[active])) if np.any(active) else np.nan,
        "inactive_cov": float(np.mean(covered[~active])) if np.any(~active) else np.nan,
    }

    action_covs = []
    under_gaps = []
    for a, _ in enumerate(action_names):
        short = f"target_{a}"
        mask = result.selection.action_indices == a
        if not np.any(mask):
            row[f"{short}_cov"] = np.nan
            row[f"{short}_size"] = np.nan
            row[f"{short}_ref"] = np.nan
            row[f"{short}_selected_n"] = 0
            row[f"{short}_active_rate"] = np.nan
            continue
        cov = float(np.mean(covered[mask]))
        row[f"{short}_cov"] = cov
        row[f"{short}_size"] = float(np.mean(sizes[mask]))
        row[f"{short}_ref"] = float(np.mean(result.reference_sizes[mask]))
        row[f"{short}_selected_n"] = int(np.sum(mask))
        row[f"{short}_active_rate"] = float(np.mean(active[mask]))
        action_covs.append(cov)
        under_gaps.append(max(nominal - cov, 0.0))

    if action_covs:
        row["action_cov_gap"] = float(np.mean(np.abs(np.asarray(action_covs) - nominal)))
        row["worst_under_gap"] = float(np.max(under_gaps))

    if edge_difficulty is not None:
        finite = np.isfinite(edge_difficulty)
        if np.any(finite):
            median = np.median(edge_difficulty[finite])
            q75 = np.quantile(edge_difficulty[finite], 0.75)
            hard = finite & (edge_difficulty >= median)
            very_hard = finite & (edge_difficulty >= q75)
            easy = finite & (edge_difficulty < median)
            row["easy_edge_cov"] = float(np.mean(covered[easy])) if np.any(easy) else np.nan
            row["hard_edge_cov"] = float(np.mean(covered[hard])) if np.any(hard) else np.nan
            row["very_hard_edge_cov"] = (
                float(np.mean(covered[very_hard])) if np.any(very_hard) else np.nan
            )
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


def aggregate_runs(runs: list[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat(
        [df.assign(seed=i) for i, df in enumerate(runs)],
        ignore_index=True,
    )
    numeric = all_rows.select_dtypes(include=[np.number]).columns.drop("seed")
    means = all_rows.groupby("method", sort=False)[numeric].mean()
    stds = all_rows.groupby("method", sort=False)[numeric].std(ddof=1).add_suffix("_std")
    summary = pd.concat([means, stds], axis=1)
    return summary.reset_index()
