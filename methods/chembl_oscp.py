"""Edge-label OSCP utilities for ChEMBL target-screening experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
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
) -> pd.DataFrame:
    """Extract human single-protein pChEMBL activity edges from ChEMBL SQLite."""
    import sqlite3

    query = """
    SELECT
        md.chembl_id AS compound_chembl_id,
        cs.canonical_smiles AS canonical_smiles,
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
        act.data_validity_comment AS data_validity_comment
    FROM activities act
    JOIN assays a
        ON act.assay_id = a.assay_id
    JOIN target_dictionary td
        ON a.tid = td.tid
    JOIN molecule_dictionary md
        ON act.molregno = md.molregno
    JOIN compound_structures cs
        ON md.molregno = cs.molregno
    WHERE
        act.pchembl_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
        AND td.organism = 'Homo sapiens'
        AND td.target_type = 'SINGLE PROTEIN'
        AND a.confidence_score >= 8
        AND act.standard_relation = '='
        AND (
            act.data_validity_comment IS NULL
            OR act.data_validity_comment = 'Manually validated'
        )
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
    max_iqr: float = 1.5,
) -> pd.DataFrame:
    """Aggregate repeated ChEMBL measurements into one binary compound-target edge."""
    raw = raw.copy()
    raw["pchembl_value"] = pd.to_numeric(raw["pchembl_value"], errors="coerce")
    raw = raw.dropna(subset=["compound_chembl_id", "canonical_smiles", "target_chembl_id", "pchembl_value"])
    agg = (
        raw.groupby(
            ["compound_chembl_id", "canonical_smiles", "target_chembl_id", "target_name"],
            as_index=False,
        )
        .agg(
            pchembl_median=("pchembl_value", "median"),
            pchembl_iqr=("pchembl_value", lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25)),
            n_measurements=("pchembl_value", "size"),
        )
    )
    agg = agg[agg["pchembl_iqr"] <= max_iqr].copy()
    agg["y"] = np.nan
    agg.loc[agg["pchembl_median"] >= active_threshold, "y"] = 1
    agg.loc[agg["pchembl_median"] <= inactive_threshold, "y"] = 0
    agg = agg.dropna(subset=["y"]).copy()
    agg["y"] = agg["y"].astype(int)
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
) -> Path:
    """Create raw, aggregated, and selected-target parquet files if needed."""
    data_root.mkdir(parents=True, exist_ok=True)
    raw_path = data_root / "raw_human_single_protein.parquet"
    compound_target_path = data_root / "compound_target_edges.parquet"
    selected_path = data_root / "chembl_oscp_edges.parquet"

    if force or not raw_path.exists():
        if sqlite_path is None:
            sqlite_path = data_root / "chembl_36" / "chembl_36_sqlite" / "chembl_36.db"
        extract_raw_chembl_edges(sqlite_path, raw_path)

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
) -> ChemblData:
    """Load selected ChEMBL target-screening edges and compound split metadata."""
    if edges_path is None:
        split_path = data_root / "chembl_oscp_edges_split.parquet"
        selected_path = prepare_chembl36_processed_files(
            data_root=data_root,
            sqlite_path=sqlite_path,
            n_targets=n_targets,
            force=force_rebuild,
        )
        edges_path = split_path if use_existing_split and split_path.exists() and not force_rebuild else selected_path

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
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ModuleNotFoundError as exc:
        raise RuntimeError("RDKit is required for Morgan fingerprints.") from exc

    rows: list[int] = []
    cols: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        on_bits = list(fp.GetOnBits())
        rows.extend([i] * len(on_bits))
        cols.extend(on_bits)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(smiles), n_bits), dtype=np.float32)


def _hashed_smiles_matrix(
    smiles: Sequence[str],
    n_bits: int,
    ngram_min: int = 2,
    ngram_max: int = 5,
) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        n_features=n_bits,
        alternate_sign=False,
        norm=None,
        binary=True,
        dtype=np.float32,
    )
    return vectorizer.transform([str(s) for s in smiles]).tocsr()


def build_compound_features(
    compounds: pd.DataFrame,
    fingerprint: str = "auto",
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[sparse.csr_matrix, str]:
    """Build compound features. auto uses Morgan when RDKit is installed."""
    smiles = compounds["canonical_smiles"].fillna("").astype(str).tolist()
    if fingerprint not in {"auto", "morgan", "hashed_smiles"}:
        raise ValueError(f"unknown fingerprint: {fingerprint}")

    if fingerprint in {"auto", "morgan"}:
        try:
            cache_path = CHEMBL_FEATURE_CACHE_DIR / (
                f"morgan_n{len(smiles)}_bits{n_bits}_r{radius}.npz"
            )
            if cache_path.exists():
                return sparse.load_npz(cache_path).tocsr(), "morgan"
            features = _morgan_fingerprint_matrix(smiles, n_bits=n_bits, radius=radius)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(cache_path, features)
            return features, "morgan"
        except RuntimeError:
            if fingerprint == "morgan":
                raise
    cache_path = CHEMBL_FEATURE_CACHE_DIR / f"hashed_smiles_n{len(smiles)}_bits{n_bits}.npz"
    if cache_path.exists():
        return sparse.load_npz(cache_path).tocsr(), "hashed_smiles"
    features = _hashed_smiles_matrix(smiles, n_bits=n_bits)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(cache_path, features)
    return features, "hashed_smiles"


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
    losses = []
    weights = []
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
            losses.append(
                float(_masked_bce_with_logits(model(x_tensor), y_tensor, mask_tensor).item())
            )
            weights.append(n_obs)
    return float(np.average(losses, weights=weights))


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
    selection: EdgeSelection,
    config: RelationalSelectionConfig,
    alpha: float,
    method: str = "JOMI",
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
                candidate_units = reference_mask_top_capacity(
                    cal_probs[:, action_index],
                    test_probs[batch, action_index],
                    local_index,
                    int(capacity),
                )
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
    return means.join(stds).reset_index()
