"""MIMIC-IV ICU triage dataset helpers for OSCP experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MIMIC_LABELS = np.array(
    [
        "brief_stay",
        "moderate_stay",
        "extended_stay",
        "prolonged_stay",
        "hospital_death",
    ]
)
MIMIC_CRITICAL_CLASS = 4


@dataclass(frozen=True)
class MimicData:
    x: pd.DataFrame
    y: np.ndarray
    hidden_critical: np.ndarray


@dataclass(frozen=True)
class MimicSplitData:
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_val: pd.DataFrame
    y_val: np.ndarray
    x_cal: pd.DataFrame
    y_cal: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    hidden_test: np.ndarray


@dataclass
class TorchMimicClassifier:
    preprocess: Any
    linear: Any
    temperature: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for CUDA training.")
        encoded = self.preprocess.transform(x)
        x_np = encoded.toarray() if hasattr(encoded, "toarray") else np.asarray(encoded)
        self.linear.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=self.linear.weight.device)
            logits = self.linear(x_tensor) / self.temperature
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def _diagnosis_chapter(icd_code: object, icd_version: object) -> str:
    code = str(icd_code).strip().upper()
    version = str(icd_version).strip()
    if not code or code == "NAN":
        return "dx_unknown"
    if version == "10":
        return f"dx10_{code[0]}"
    digits = "".join(ch for ch in code[:3] if ch.isdigit())
    if not digits:
        return "dx9_other"
    value = int(digits)
    bins = [
        (139, "infectious"),
        (239, "neoplasm"),
        (279, "endocrine"),
        (289, "blood"),
        (319, "mental"),
        (389, "nervous"),
        (459, "circulatory"),
        (519, "respiratory"),
        (579, "digestive"),
        (629, "genitourinary"),
        (679, "pregnancy"),
        (709, "skin"),
        (739, "musculoskeletal"),
        (759, "congenital"),
        (779, "perinatal"),
        (799, "symptoms"),
        (999, "injury"),
    ]
    for upper, name in bins:
        if value <= upper:
            return f"dx9_{name}"
    return "dx9_other"


def _load_diagnosis_features(root: Path) -> pd.DataFrame:
    diagnoses = pd.read_csv(
        root / "hosp" / "diagnoses_icd.csv.gz",
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": "string", "icd_version": "string"},
    )
    diagnoses["dx_chapter"] = [
        _diagnosis_chapter(code, version)
        for code, version in zip(diagnoses["icd_code"], diagnoses["icd_version"])
    ]
    chapter_counts = pd.crosstab(diagnoses["hadm_id"], diagnoses["dx_chapter"])
    chapter_counts.columns = [f"n_{col}" for col in chapter_counts.columns]
    chapter_counts["diagnosis_count"] = diagnoses.groupby("hadm_id").size()
    return chapter_counts.reset_index()


def _severity_labels(df: pd.DataFrame) -> np.ndarray:
    y = np.full(df.shape[0], 3, dtype=int)
    alive = df["hospital_expire_flag"].to_numpy(dtype=int) == 0
    y[~alive] = MIMIC_CRITICAL_CLASS
    los = df["los"].to_numpy(dtype=float)
    finite_alive = alive & np.isfinite(los)
    y[finite_alive & (los < 1.1)] = 0
    y[finite_alive & (los >= 1.1) & (los < 1.9)] = 1
    y[finite_alive & (los >= 1.9) & (los < 3.6)] = 2
    y[finite_alive & (los >= 3.6)] = 3
    return y


def load_mimic_triage(
    root: Path = Path("dataset/mimic-iv-3.1"),
    n: int | None = None,
    frac: float | None = None,
    seed: int = 0,
) -> MimicData:
    """Load an ICU-stay-level MIMIC-IV severity classification table.

    The task uses admission and demographic fields plus diagnosis-chapter
    summaries to predict five severity classes derived from ICU LOS and
    hospital mortality. It intentionally avoids the very large chart/lab event
    tables so relational experiments stay lightweight and reproducible.
    """
    patients = pd.read_csv(
        root / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = pd.read_csv(
        root / "hosp" / "admissions.csv.gz",
        usecols=[
            "subject_id",
            "hadm_id",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ],
    )
    icu = pd.read_csv(
        root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "los"],
    )
    diagnosis_features = _load_diagnosis_features(root)

    df = (
        icu.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
        .merge(patients, on="subject_id", how="inner")
        .merge(diagnosis_features, on="hadm_id", how="left")
    )
    count_cols = [c for c in df.columns if c.startswith("n_dx") or c == "diagnosis_count"]
    df[count_cols] = df[count_cols].fillna(0.0)
    df = df.sort_values(["subject_id", "hadm_id", "stay_id"]).reset_index(drop=True)

    if frac is not None:
        sample_n = max(1, int(np.ceil(frac * df.shape[0])))
        df = (
            df.sample(n=sample_n, random_state=seed)
            .sort_values(["subject_id", "hadm_id", "stay_id"])
            .reset_index(drop=True)
        )
    elif n is not None and n > 0 and n < df.shape[0]:
        df = (
            df.sample(n=n, random_state=seed)
            .sort_values(["subject_id", "hadm_id", "stay_id"])
            .reset_index(drop=True)
        )

    y = _severity_labels(df)
    feature_cols = [
        "anchor_age",
        "gender",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "first_careunit",
        "diagnosis_count",
        *[c for c in df.columns if c.startswith("n_dx")],
    ]
    x = df[feature_cols].copy()
    hidden_critical = y == MIMIC_CRITICAL_CLASS
    return MimicData(x=x, y=y, hidden_critical=hidden_critical)


def make_mimic_splits(data: MimicData, seed: int = 0) -> MimicSplitData:
    """Create 50/15/15/20 train/val/cal/test splits."""
    idx = np.arange(data.y.size)
    train_idx, rest_idx = train_test_split(
        idx, train_size=0.50, random_state=seed, stratify=data.y
    )
    val_idx, rest_idx = train_test_split(
        rest_idx,
        train_size=0.30,
        random_state=seed + 1,
        stratify=data.y[rest_idx],
    )
    cal_idx, test_idx = train_test_split(
        rest_idx,
        train_size=3.0 / 7.0,
        random_state=seed + 2,
        stratify=data.y[rest_idx],
    )
    return MimicSplitData(
        x_train=data.x.iloc[train_idx],
        y_train=data.y[train_idx],
        x_val=data.x.iloc[val_idx],
        y_val=data.y[val_idx],
        x_cal=data.x.iloc[cal_idx],
        y_cal=data.y[cal_idx],
        x_test=data.x.iloc[test_idx],
        y_test=data.y[test_idx],
        hidden_test=data.hidden_critical[test_idx],
    )


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)



def _build_mimic_preprocess(split: MimicSplitData) -> ColumnTransformer:
    """Fit a calibrated multinomial logistic model for MIMIC severity labels."""
    numeric_cols = split.x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in split.x_train.columns if c not in numeric_cols]
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
                    _one_hot_encoder(),
                ),
                categorical_cols,
            ),
        ]
    )


def _fit_sklearn_mimic_classifier(split: MimicSplitData, seed: int = 0):
    preprocess = _build_mimic_preprocess(split)
    base = make_pipeline(
        preprocess,
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
        ),
    )
    base.fit(split.x_train, split.y_train)
    model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    model.fit(split.x_val, split.y_val)
    val_loss = log_loss(
        split.y_val,
        model.predict_proba(split.x_val),
        labels=np.arange(MIMIC_LABELS.size),
    )
    return model, {"val_log_loss": float(val_loss), "device": "cpu"}


def _fit_torch_mimic_classifier(split: MimicSplitData, seed: int, device: str):

    preprocess = _build_mimic_preprocess(split)
    x_train_enc = preprocess.fit_transform(split.x_train)
    x_val_enc = preprocess.transform(split.x_val)
    x_train = x_train_enc.toarray() if hasattr(x_train_enc, "toarray") else np.asarray(x_train_enc)
    x_val = x_val_enc.toarray() if hasattr(x_val_enc, "toarray") else np.asarray(x_val_enc)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
    y_train_tensor = torch.as_tensor(split.y_train, dtype=torch.long, device=torch_device)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=torch_device)
    y_val_tensor = torch.as_tensor(split.y_val, dtype=torch.long, device=torch_device)

    linear = torch.nn.Linear(x_train.shape[1], MIMIC_LABELS.size, device=torch_device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.03, weight_decay=1e-4)

    best_state = None
    best_val_loss = float("inf")
    patience = 40
    stale_epochs = 0
    for _ in range(400):
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
    model = TorchMimicClassifier(
        preprocess=preprocess,
        linear=linear,
        temperature=float(temperature.detach().clamp_min(1e-3).item()),
    )
    val_loss = log_loss(
        split.y_val,
        model.predict_proba(split.x_val),
        labels=np.arange(MIMIC_LABELS.size),
    )
    return model, {"val_log_loss": float(val_loss), "device": device}


def fit_mimic_classifier(split: MimicSplitData, seed: int = 0, cuda: int | None = None):
    device = "cpu" if cuda is None else f"cuda:{cuda}"
    if device == "cpu":
        return _fit_sklearn_mimic_classifier(split, seed=seed)
    return _fit_torch_mimic_classifier(split, seed=seed, device=device)
