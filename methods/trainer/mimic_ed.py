"""MIMIC-IV-ED diagnostic-acquisition helpers for OSCP experiments."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sklearn.frozen import FrozenEstimator
except ModuleNotFoundError:  # scikit-learn < 1.6
    FrozenEstimator = None


MIMIC_ED_LABELS = np.array(
    [
        "ed_discharge",
        "hospital_admission",
        "icu_24h",
        "death",
    ]
)
MIMIC_ED_CRITICAL_CLASS = 3

MIMIC_ED_ACTIONS = np.array(
    [
        "any_lab_6h",
        "cardiac_lab_6h",
        "infection_lab_6h",
        "ed_medication_6h",
        "escalation_decision",
    ]
)


MIMIC_ED_CACHE_VERSION = "v1"


@dataclass(frozen=True)
class MimicEDData:
    x: pd.DataFrame
    y: np.ndarray
    actions: np.ndarray
    subject_id: np.ndarray
    stay_id: np.ndarray
    intime: np.ndarray


@dataclass(frozen=True)
class MimicEDSplitData:
    x_train: pd.DataFrame
    y_train: np.ndarray
    actions_train: np.ndarray

    x_val: pd.DataFrame
    y_val: np.ndarray
    actions_val: np.ndarray

    x_cal: pd.DataFrame
    y_cal: np.ndarray
    actions_cal: np.ndarray
    intime_cal: np.ndarray

    x_test: pd.DataFrame
    y_test: np.ndarray
    actions_test: np.ndarray
    intime_test: np.ndarray


@dataclass
class MimicEDOutcomeClassifier:
    preprocess: Any
    classifier: Any

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.preprocess.transform(x)
        return self.classifier.predict_proba(encoded)


@dataclass
class TorchEncodedClassifier:
    linear: Any
    temperature: float

    def predict_proba(self, x: Any) -> np.ndarray:
        x_np = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        self.linear.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(
                x_np, dtype=torch.float32, device=self.linear.weight.device
            )
            logits = self.linear(x_tensor) / self.temperature
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


@dataclass
class ConstantBinaryClassifier:
    probability: float

    def predict_proba(self, x: Any) -> np.ndarray:
        n = x.shape[0]
        p = np.clip(float(self.probability), 0.0, 1.0)
        return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])


@dataclass
class MimicEDActionSupportModel:
    preprocess: Any
    classifiers: tuple[Any, ...]

    def predict_support(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.preprocess.transform(x)
        cols = []
        for clf in self.classifiers:
            probs = clf.predict_proba(encoded)
            if probs.shape[1] == 1:
                cols.append(np.zeros(encoded.shape[0], dtype=float))
            else:
                cols.append(probs[:, -1])
        return np.column_stack(cols)


def _one_hot_encoder(min_frequency: int = 20) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_frequency,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_frequency,
            sparse=True,
        )


def _build_preprocess(
    x_train: pd.DataFrame,
    max_text_features: int = 1000,
    min_frequency: int = 20,
) -> ColumnTransformer:
    """Build preprocessing for ED tabular and text features."""
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = ["chiefcomplaint"]
    categorical_cols = [
        c
        for c in x_train.columns
        if c not in numeric_cols and c not in text_cols
    ]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_cols,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    _one_hot_encoder(min_frequency=min_frequency),
                ),
                categorical_cols,
            ),
            (
                "chiefcomplaint",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=max_text_features,
                    min_df=2,
                ),
                "chiefcomplaint",
            ),
        ],
        sparse_threshold=0.3,
    )


def _calibrate_prefit(estimator: Any, x_val: Any, y_val: np.ndarray) -> Any:
    if FrozenEstimator is None:
        model = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
    else:
        model = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    model.fit(x_val, y_val)
    return model


def _to_dense(x: Any) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _fit_torch_linear_classifier(
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    device: str,
    lr: float = 0.03,
) -> tuple[TorchEncodedClassifier, float]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for CUDA training.")

    x_train_np = _to_dense(x_train)
    x_val_np = _to_dense(x_val)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    x_train_tensor = torch.as_tensor(x_train_np, dtype=torch.float32, device=torch_device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long, device=torch_device)
    x_val_tensor = torch.as_tensor(x_val_np, dtype=torch.float32, device=torch_device)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.long, device=torch_device)

    linear = torch.nn.Linear(x_train_np.shape[1], n_classes, device=torch_device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val_loss = float("inf")
    patience = 25
    stale_epochs = 0
    for _ in range(250):
        linear.train()
        optimizer.zero_grad()
        train_loss = F.cross_entropy(linear(x_train_tensor), y_train_tensor)
        train_loss.backward()
        optimizer.step()

        linear.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(linear(x_val_tensor), y_val_tensor).item()
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(linear.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        linear.load_state_dict(best_state)

    temperature = torch.nn.Parameter(torch.ones(1, device=torch_device))
    temp_optimizer = torch.optim.LBFGS(
        [temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        temp_optimizer.zero_grad()
        temp = temperature.clamp_min(1e-3)
        loss = F.cross_entropy(linear(x_val_tensor) / temp, y_val_tensor)
        loss.backward()
        return loss

    temp_optimizer.step(closure)
    model = TorchEncodedClassifier(
        linear=linear,
        temperature=float(temperature.detach().clamp_min(1e-3).item()),
    )
    val_loss = log_loss(
        y_val,
        model.predict_proba(x_val_np),
        labels=np.arange(n_classes),
    )
    return model, float(val_loss)


def _lab_item_sets(hosp_root: Path) -> tuple[set[int], set[int]]:
    labitems = pd.read_csv(
        hosp_root / "hosp" / "d_labitems.csv.gz",
        usecols=["itemid", "label", "category"],
    )
    text = (
        labitems["label"].fillna("").astype(str)
        + " "
        + labitems["category"].fillna("").astype(str)
    ).str.lower()

    cardiac = text.str.contains(
        r"troponin|ck-mb|creatine kinase.*mb|bnp|nt-probnp",
        regex=True,
    )
    infection = text.str.contains(
        r"lactate|procalcitonin|c-reactive protein|\bcrp\b",
        regex=True,
    )
    return (
        set(labitems.loc[cardiac, "itemid"].astype(int).tolist()),
        set(labitems.loc[infection, "itemid"].astype(int).tolist()),
    )


def _merge_timed_events(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    time_col: str,
    on: list[str],
) -> pd.DataFrame:
    if events.empty:
        return events
    events = events.copy()
    events[time_col] = pd.to_datetime(
        events[time_col],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )
    events = events.dropna(subset=[time_col])
    merged = events.merge(windows, on=on, how="inner")
    return merged[
        (merged[time_col] >= merged["intime"])
        & (merged[time_col] < merged["action_end"])
    ]


def _add_lab_actions(
    actions: np.ndarray,
    stays: pd.DataFrame,
    hosp_root: Path,
    chunksize: int,
) -> None:
    path = hosp_root / "hosp" / "labevents.csv.gz"
    if not path.exists():
        return

    cardiac_items, infection_items = _lab_item_sets(hosp_root)
    windows = stays[
        ["stay_id", "subject_id", "hadm_id", "intime", "action_end"]
    ].dropna(subset=["hadm_id"])
    if windows.empty:
        return
    windows = windows.copy()
    windows["hadm_id"] = windows["hadm_id"].astype("int64")
    hadm_ids = set(windows["hadm_id"].tolist())
    stay_to_row = {int(stay_id): i for i, stay_id in enumerate(stays["stay_id"])}

    for chunk in pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "itemid", "charttime"],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)].copy()
        if chunk.empty:
            continue

        hits = _merge_timed_events(
            chunk,
            windows,
            time_col="charttime",
            on=["subject_id", "hadm_id"],
        )
        if hits.empty:
            continue

        row_idx = hits["stay_id"].map(stay_to_row).to_numpy(dtype=int)
        actions[row_idx, 0] = 1

        itemids = hits["itemid"].astype(int)
        cardiac_idx = hits.loc[itemids.isin(cardiac_items), "stay_id"].map(stay_to_row)
        infection_idx = hits.loc[itemids.isin(infection_items), "stay_id"].map(stay_to_row)
        if cardiac_idx.size:
            actions[cardiac_idx.to_numpy(dtype=int), 1] = 1
        if infection_idx.size:
            actions[infection_idx.to_numpy(dtype=int), 2] = 1


def _add_microbiology_actions(
    actions: np.ndarray,
    stays: pd.DataFrame,
    hosp_root: Path,
    chunksize: int,
) -> None:
    path = hosp_root / "hosp" / "microbiologyevents.csv.gz"
    if not path.exists():
        return

    windows = stays[
        ["stay_id", "subject_id", "hadm_id", "intime", "action_end"]
    ].dropna(subset=["hadm_id"])
    if windows.empty:
        return
    windows = windows.copy()
    windows["hadm_id"] = windows["hadm_id"].astype("int64")
    hadm_ids = set(windows["hadm_id"].tolist())
    stay_to_row = {int(stay_id): i for i, stay_id in enumerate(stays["stay_id"])}

    for chunk in pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "charttime"],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id", "charttime"]).copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)].copy()
        if chunk.empty:
            continue
        hits = _merge_timed_events(
            chunk,
            windows,
            time_col="charttime",
            on=["subject_id", "hadm_id"],
        )
        if hits.empty:
            continue
        row_idx = hits["stay_id"].map(stay_to_row).to_numpy(dtype=int)
        actions[row_idx, 2] = 1


def _add_pyxis_actions(
    actions: np.ndarray,
    stays: pd.DataFrame,
    ed_root: Path,
    chunksize: int,
) -> None:
    path = ed_root / "ed" / "pyxis.csv.gz"
    if not path.exists():
        return

    windows = stays[["stay_id", "intime", "action_end"]].copy()
    stay_ids = set(windows["stay_id"].astype(int).tolist())
    stay_to_row = {int(stay_id): i for i, stay_id in enumerate(stays["stay_id"])}

    for chunk in pd.read_csv(
        path,
        usecols=["stay_id", "charttime"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if chunk.empty:
            continue
        hits = _merge_timed_events(
            chunk,
            windows,
            time_col="charttime",
            on=["stay_id"],
        )
        if hits.empty:
            continue
        row_idx = hits["stay_id"].map(stay_to_row).to_numpy(dtype=int)
        actions[row_idx, 3] = 1


def _make_action_matrix(
    df: pd.DataFrame,
    ed_root: Path,
    hosp_root: Path,
    action_window_hours: float,
    chunksize: int,
) -> np.ndarray:
    actions = np.zeros((df.shape[0], MIMIC_ED_ACTIONS.size), dtype=int)
    stays = df[
        ["stay_id", "subject_id", "hadm_id", "intime", "escalation_decision"]
    ].copy()
    stays["action_end"] = stays["intime"] + pd.to_timedelta(
        action_window_hours, unit="h"
    )

    _add_lab_actions(actions, stays, hosp_root=hosp_root, chunksize=chunksize)
    _add_microbiology_actions(actions, stays, hosp_root=hosp_root, chunksize=chunksize)
    _add_pyxis_actions(actions, stays, ed_root=ed_root, chunksize=chunksize)
    actions[:, 4] = stays["escalation_decision"].to_numpy(dtype=int)
    return actions


def _build_ed_table(
    ed_root: Path,
    hosp_root: Path,
) -> pd.DataFrame:
    ed = pd.read_csv(
        ed_root / "ed" / "edstays.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "gender",
            "race",
            "arrival_transport",
            "disposition",
        ],
    )
    triage = pd.read_csv(ed_root / "ed" / "triage.csv.gz")
    patients = pd.read_csv(
        hosp_root / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "anchor_age"],
    )
    admissions = pd.read_csv(
        hosp_root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "hospital_expire_flag", "deathtime"],
    )
    icu = pd.read_csv(
        hosp_root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "intime"],
    )

    ed["intime"] = pd.to_datetime(ed["intime"], errors="coerce")
    ed["outtime"] = pd.to_datetime(ed["outtime"], errors="coerce")
    ed["hadm_id"] = pd.to_numeric(ed["hadm_id"], errors="coerce")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
    icu["hadm_id"] = pd.to_numeric(icu["hadm_id"], errors="coerce")
    icu["icu_intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu_first = (
        icu.dropna(subset=["hadm_id", "icu_intime"])
        .sort_values("icu_intime")
        .groupby(["subject_id", "hadm_id"], as_index=False)["icu_intime"]
        .first()
    )

    df = (
        ed.merge(triage, on=["subject_id", "stay_id"], how="inner")
        .merge(patients, on="subject_id", how="inner")
        .merge(admissions, on=["subject_id", "hadm_id"], how="left")
        .merge(icu_first, on=["subject_id", "hadm_id"], how="left")
    )
    df = df.dropna(subset=["intime", "anchor_age"])
    df = df[df["anchor_age"] >= 18].copy()

    icu_hours = (
        df["icu_intime"] - df["intime"]
    ).dt.total_seconds() / 3600.0
    df["icu_24h"] = (icu_hours >= 0.0) & (icu_hours <= 24.0)
    df["hospital_expire_flag"] = df["hospital_expire_flag"].fillna(0).astype(int)
    df["death"] = (
        df["disposition"].fillna("").eq("EXPIRED")
        | df["hospital_expire_flag"].eq(1)
        | df["deathtime"].notna()
    )
    admitted_disposition = df["disposition"].fillna("").isin(["ADMITTED", "TRANSFER"])
    df["hospital_admission"] = df["hadm_id"].notna() | admitted_disposition
    df["escalation_decision"] = (
        df["hospital_admission"] | df["icu_24h"] | df["death"]
    ).astype(int)

    y = np.zeros(df.shape[0], dtype=int)
    y[df["hospital_admission"].to_numpy(dtype=bool)] = 1
    y[df["icu_24h"].to_numpy(dtype=bool)] = 2
    y[df["death"].to_numpy(dtype=bool)] = 3
    df["severity_label"] = y

    df = df.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    df["prior_ed_visits"] = df.groupby("subject_id").cumcount()
    admitted_so_far = df["hadm_id"].notna().astype(int)
    df["prior_admissions"] = (
        admitted_so_far.groupby(df["subject_id"]).cumsum() - admitted_so_far
    )
    df["arrival_hour"] = df["intime"].dt.hour.astype(float)
    df["arrival_dayofweek"] = df["intime"].dt.dayofweek.astype(float)

    numeric_cols = [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
        "acuity",
        "anchor_age",
        "prior_ed_visits",
        "prior_admissions",
        "arrival_hour",
        "arrival_dayofweek",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["chiefcomplaint"] = df["chiefcomplaint"].fillna("").astype(str)

    return df.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)


def _sample_ed_table(
    df: pd.DataFrame,
    n: int | None,
    frac: float | None,
    seed: int,
) -> pd.DataFrame:
    if frac is not None:
        sample_n = max(1, int(np.ceil(frac * df.shape[0])))
        return (
            df.sample(n=sample_n, random_state=seed)
            .sort_values(["subject_id", "intime", "stay_id"])
            .reset_index(drop=True)
        )
    if n is not None and 0 < n < df.shape[0]:
        return (
            df.sample(n=n, random_state=seed)
            .sort_values(["subject_id", "intime", "stay_id"])
            .reset_index(drop=True)
        )
    return df.copy()


def _cache_file_path(
    cache_dir: Path,
    ed_root: Path,
    hosp_root: Path,
    action_window_hours: float,
) -> Path:
    key = "|".join(
        [
            MIMIC_ED_CACHE_VERSION,
            str(ed_root.resolve()),
            str(hosp_root.resolve()),
            f"{action_window_hours:.6f}",
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"mimic_ed_acquisition_{digest}.pkl"


def _load_or_build_cached_ed_frame(
    ed_root: Path,
    hosp_root: Path,
    action_window_hours: float,
    chunksize: int,
    cache_dir: Path | None,
) -> pd.DataFrame:
    if cache_dir is None:
        base = _build_ed_table(ed_root=ed_root, hosp_root=hosp_root)
        actions = _make_action_matrix(
            base,
            ed_root=ed_root,
            hosp_root=hosp_root,
            action_window_hours=action_window_hours,
            chunksize=chunksize,
        )
        out = base.copy()
        for i, action_name in enumerate(MIMIC_ED_ACTIONS):
            out[action_name] = actions[:, i]
        return out

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_path(
        cache_dir=cache_dir,
        ed_root=ed_root,
        hosp_root=hosp_root,
        action_window_hours=action_window_hours,
    )
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    base = _build_ed_table(ed_root=ed_root, hosp_root=hosp_root)
    actions = _make_action_matrix(
        base,
        ed_root=ed_root,
        hosp_root=hosp_root,
        action_window_hours=action_window_hours,
        chunksize=chunksize,
    )
    out = base.copy()
    for i, action_name in enumerate(MIMIC_ED_ACTIONS):
        out[action_name] = actions[:, i]
    out.to_pickle(cache_path)
    return out


def load_mimic_ed_acquisition(
    ed_root: Path = Path("dataset/mimic-iv-ed-2.2"),
    hosp_root: Path = Path("dataset/mimic-iv-3.1"),
    n: int | None = None,
    frac: float | None = None,
    seed: int = 0,
    action_window_hours: float = 6.0,
    chunksize: int = 2_000_000,
    cache_dir: Path | None = None,
) -> MimicEDData:
    """Load ED-stay diagnostic-acquisition data.

    The feature set is limited to demographics, arrival metadata, triage
    vitals, chief complaint, and prior-utilization proxies. Action labels use
    events in [ED intime, ED intime + action_window_hours).
    """
    full_df = _load_or_build_cached_ed_frame(
        ed_root=ed_root,
        hosp_root=hosp_root,
        action_window_hours=action_window_hours,
        chunksize=chunksize,
        cache_dir=cache_dir,
    )
    df = _sample_ed_table(full_df, n=n, frac=frac, seed=seed)

    feature_cols = [
        "anchor_age",
        "gender",
        "race",
        "arrival_transport",
        "acuity",
        "chiefcomplaint",
        "pain",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "prior_ed_visits",
        "prior_admissions",
        "arrival_hour",
        "arrival_dayofweek",
    ]
    x = df[feature_cols].copy()
    actions = df[MIMIC_ED_ACTIONS.tolist()].to_numpy(dtype=int)
    return MimicEDData(
        x=x,
        y=df["severity_label"].to_numpy(dtype=int),
        actions=actions,
        subject_id=df["subject_id"].to_numpy(dtype=int),
        stay_id=df["stay_id"].to_numpy(dtype=int),
        intime=df["intime"].to_numpy(),
    )


def _stratify_labels_or_none(labels: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(labels, return_counts=True)
    if counts.size <= 1 or np.min(counts) < 2:
        return None
    return labels


def make_mimic_ed_splits(data: MimicEDData, seed: int = 0) -> MimicEDSplitData:
    """Create patient-level 50/15/15/20 train/val/cal/test splits."""
    subjects = np.unique(data.subject_id)
    subject_labels = (
        pd.Series(data.y)
        .groupby(data.subject_id)
        .max()
        .loc[subjects]
        .to_numpy()
    )

    train_subjects, rest_subjects, _, rest_labels = train_test_split(
        subjects,
        subject_labels,
        train_size=0.50,
        random_state=seed,
        stratify=_stratify_labels_or_none(subject_labels),
    )
    val_subjects, rest_subjects, _, rest_labels = train_test_split(
        rest_subjects,
        rest_labels,
        train_size=0.30,
        random_state=seed + 1,
        stratify=_stratify_labels_or_none(rest_labels),
    )
    cal_subjects, test_subjects = train_test_split(
        rest_subjects,
        train_size=3.0 / 7.0,
        random_state=seed + 2,
        stratify=_stratify_labels_or_none(rest_labels),
    )

    def mask(subject_subset: np.ndarray) -> np.ndarray:
        return np.isin(data.subject_id, subject_subset)

    train_mask = mask(train_subjects)
    val_mask = mask(val_subjects)
    cal_mask = mask(cal_subjects)
    test_mask = mask(test_subjects)

    return MimicEDSplitData(
        x_train=data.x.loc[train_mask],
        y_train=data.y[train_mask],
        actions_train=data.actions[train_mask],
        x_val=data.x.loc[val_mask],
        y_val=data.y[val_mask],
        actions_val=data.actions[val_mask],
        x_cal=data.x.loc[cal_mask],
        y_cal=data.y[cal_mask],
        actions_cal=data.actions[cal_mask],
        intime_cal=data.intime[cal_mask],
        x_test=data.x.loc[test_mask],
        y_test=data.y[test_mask],
        actions_test=data.actions[test_mask],
        intime_test=data.intime[test_mask],
    )


def _fit_sklearn_mimic_ed_models(
    split: MimicEDSplitData,
    seed: int = 0,
    max_text_features: int = 500,
    min_frequency: int = 20,
) -> tuple[MimicEDOutcomeClassifier, MimicEDActionSupportModel, dict[str, Any]]:
    """Fit p(Y | x_obs) and one calibrated g_a(x_obs) model per action."""
    preprocess = _build_preprocess(
        split.x_train,
        max_text_features=max_text_features,
        min_frequency=min_frequency,
    )
    x_train = preprocess.fit_transform(split.x_train)
    x_val = preprocess.transform(split.x_val)

    outcome_base = LogisticRegression(
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    )
    outcome_base.fit(x_train, split.y_train)
    if np.unique(split.y_val).size == MIMIC_ED_LABELS.size:
        outcome_classifier = _calibrate_prefit(outcome_base, x_val, split.y_val)
    else:
        outcome_classifier = outcome_base

    action_classifiers: list[Any] = []
    action_val_losses: dict[str, float] = {}
    for a, action_name in enumerate(MIMIC_ED_ACTIONS):
        y_action = split.actions_train[:, a]
        y_action_val = split.actions_val[:, a]
        is_constant = np.unique(y_action).size < 2
        if is_constant:
            clf: Any = ConstantBinaryClassifier(float(np.mean(y_action)))
            probs = np.clip(clf.predict_proba(x_val)[:, -1], 1e-6, 1.0 - 1e-6)
            action_val_loss = float(
                log_loss(y_action_val, np.column_stack([1.0 - probs, probs]), labels=[0, 1])
            )
        else:
            base = LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",
                random_state=seed + 100 + a,
            )
            base.fit(x_train, y_action)
            if np.unique(y_action_val).size == 2:
                clf = _calibrate_prefit(base, x_val, y_action_val)
            else:
                clf = base
            probs = np.clip(clf.predict_proba(x_val)[:, -1], 1e-6, 1.0 - 1e-6)
            action_val_loss = float(
                log_loss(
                    y_action_val,
                    np.column_stack([1.0 - probs, probs]),
                    labels=[0, 1],
                )
            )
        action_classifiers.append(clf)
        action_val_losses[f"{action_name}_val_log_loss"] = action_val_loss

    outcome_model = MimicEDOutcomeClassifier(
        preprocess=preprocess,
        classifier=outcome_classifier,
    )
    support_model = MimicEDActionSupportModel(
        preprocess=preprocess,
        classifiers=tuple(action_classifiers),
    )

    val_probs = outcome_model.predict_proba(split.x_val)
    info: dict[str, Any] = {
        "val_log_loss": float(
            log_loss(split.y_val, val_probs, labels=np.arange(MIMIC_ED_LABELS.size))
        ),
        "encoded_dim": int(x_train.shape[1]),
        "device": "cpu",
        "action_prevalence_train": dict(
            zip(MIMIC_ED_ACTIONS.tolist(), split.actions_train.mean(axis=0).tolist())
        ),
        **action_val_losses,
    }
    return outcome_model, support_model, info


def _fit_torch_mimic_ed_models(
    split: MimicEDSplitData,
    seed: int = 0,
    max_text_features: int = 500,
    min_frequency: int = 20,
    device: str = "cuda:0",
) -> tuple[MimicEDOutcomeClassifier, MimicEDActionSupportModel, dict[str, Any]]:
    preprocess = _build_preprocess(
        split.x_train,
        max_text_features=max_text_features,
        min_frequency=min_frequency,
    )
    x_train = preprocess.fit_transform(split.x_train)
    x_val = preprocess.transform(split.x_val)

    outcome_classifier, outcome_val_loss = _fit_torch_linear_classifier(
        x_train=x_train,
        y_train=split.y_train,
        x_val=x_val,
        y_val=split.y_val,
        n_classes=MIMIC_ED_LABELS.size,
        seed=seed,
        device=device,
    )

    action_classifiers: list[Any] = []
    action_val_losses: dict[str, float] = {}
    for a, action_name in enumerate(MIMIC_ED_ACTIONS):
        y_action = split.actions_train[:, a]
        y_action_val = split.actions_val[:, a]
        is_constant = np.unique(y_action).size < 2
        if is_constant:
            clf: Any = ConstantBinaryClassifier(float(np.mean(y_action)))
            probs = np.clip(clf.predict_proba(x_val)[:, -1], 1e-6, 1.0 - 1e-6)
            action_val_loss = float(
                log_loss(y_action_val, np.column_stack([1.0 - probs, probs]), labels=[0, 1])
            )
        else:
            clf, action_val_loss = _fit_torch_linear_classifier(
                x_train=x_train,
                y_train=y_action,
                x_val=x_val,
                y_val=y_action_val,
                n_classes=2,
                seed=seed + 100 + a,
                device=device,
            )
        action_classifiers.append(clf)
        action_val_losses[f"{action_name}_val_log_loss"] = action_val_loss

    outcome_model = MimicEDOutcomeClassifier(
        preprocess=preprocess,
        classifier=outcome_classifier,
    )
    support_model = MimicEDActionSupportModel(
        preprocess=preprocess,
        classifiers=tuple(action_classifiers),
    )

    info: dict[str, Any] = {
        "val_log_loss": outcome_val_loss,
        "encoded_dim": int(x_train.shape[1]),
        "device": device,
        "action_prevalence_train": dict(
            zip(MIMIC_ED_ACTIONS.tolist(), split.actions_train.mean(axis=0).tolist())
        ),
        **action_val_losses,
    }
    return outcome_model, support_model, info


def fit_mimic_ed_models(
    split: MimicEDSplitData,
    seed: int = 0,
    max_text_features: int = 500,
    min_frequency: int = 20,
    cuda: int | None = None,
) -> tuple[MimicEDOutcomeClassifier, MimicEDActionSupportModel, dict[str, Any]]:
    use_cuda = (
        cuda is not None and cuda >= 0 and torch is not None and torch.cuda.is_available()
    )
    device = f"cuda:{cuda}" if use_cuda else "cpu"
    if device == "cpu":
        return _fit_sklearn_mimic_ed_models(
            split,
            seed=seed,
            max_text_features=max_text_features,
            min_frequency=min_frequency,
        )
    return _fit_torch_mimic_ed_models(
        split,
        seed=seed,
        max_text_features=max_text_features,
        min_frequency=min_frequency,
        device=device,
    )
