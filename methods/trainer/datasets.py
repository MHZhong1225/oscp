"""Nursery and BACH dataset helpers for relational OSCP experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - optional CUDA dependency
    torch = None
    F = None
    DataLoader = None

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
except ImportError:  # pragma: no cover - optional image dependency
    transforms = None
    ImageFolder = None

from methods.configs import create_backbone


NURSERY_LABELS = np.array(["not_recom", "very_recom", "priority", "spec_prior"])
BACH_LABELS = np.array(["Benign", "InSitu", "Invasive", "Normal"])
NURSERY_CRITICAL_CLASS = 3
BACH_CRITICAL_CLASS = 2


@dataclass(frozen=True)
class DatasetData:
    x: np.ndarray
    y: np.ndarray
    hidden_critical: np.ndarray
    label_names: np.ndarray


@dataclass(frozen=True)
class DatasetSplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    hidden_test: np.ndarray
    label_names: np.ndarray


@dataclass
class TorchTabularClassifier:
    linear: Any
    mean: Any
    scale: Any
    temperature: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for CUDA training.")
        self.linear.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
            x_scaled = (x_tensor - self.mean) / self.scale
            logits = self.linear(x_scaled) / self.temperature
            return torch.softmax(logits, dim=1).cpu().numpy()


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_nursery_data(
    path: Path = Path("dataset/nursery/nursery.csv"),
    n: int | None = None,
    seed: int = 0,
) -> DatasetData:
    """Load the nursery tabular dataset with a fixed severity label order."""
    df = pd.read_csv(path)
    df = df[df["final evaluation"] != "recommend"].copy()
    label_to_id = {label: i for i, label in enumerate(NURSERY_LABELS)}
    df = df[df["final evaluation"].isin(label_to_id)].copy()
    y = df["final evaluation"].map(label_to_id).to_numpy(dtype=int)

    feature_cols = [
        "parents",
        "has_nurs",
        "form",
        "children",
        "housing",
        "finance",
        "social",
        "health",
    ]
    encoder = _one_hot_encoder()
    x = encoder.fit_transform(df[feature_cols].astype(str)).astype(np.float32)

    if n is not None and n > 0 and n < x.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(x.shape[0]), size=n, replace=False)
        x = x[idx]
        y = y[idx]

    hidden_critical = y == NURSERY_CRITICAL_CLASS
    return DatasetData(x=x, y=y, hidden_critical=hidden_critical, label_names=NURSERY_LABELS)


def _bach_transform(image_size: int):
    if transforms is None:
        raise RuntimeError("torchvision is required for BACH transforms.")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _extract_backbone_features(
    root: Path,
    split: str,
    backbone: Any,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if ImageFolder is None or DataLoader is None or torch is None:
        raise RuntimeError("torch and torchvision are required for BACH feature extraction.")
    dataset = ImageFolder(str(root / split), transform=_bach_transform(image_size))
    class_to_global = {label: i for i, label in enumerate(BACH_LABELS)}
    local_to_global = {
        local_idx: class_to_global[class_name]
        for class_name, local_idx in dataset.class_to_idx.items()
    }
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    features = []
    labels = []
    backbone.eval()
    with torch.no_grad():
        for x, local_y in loader:
            x = x.to(device)
            feat = backbone(x).flatten(1).cpu().numpy()
            y = np.asarray([local_to_global[int(v)] for v in local_y], dtype=int)
            features.append(feat)
            labels.append(y)
    return np.vstack(features).astype(np.float32), np.concatenate(labels).astype(int)


def load_bach_splits(
    root: Path = Path("/home/ubuntu/zmh/BrCPT/datasets/bach"),
    seed: int = 0,
    image_size: int = 224,
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    feature_batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cpu",
) -> DatasetSplitData:
    """Load BACH ImageFolder splits and extract frozen backbone features."""
    if torch is None:
        raise RuntimeError("PyTorch is required for BACH backbone features.")
    backbone, _ = create_backbone(backbone_name, pretrained=pretrained)
    backbone = backbone.to(device)
    x_train, y_train = _extract_backbone_features(
        root, "train", backbone, image_size, feature_batch_size, num_workers, device
    )
    x_val, y_val = _extract_backbone_features(
        root, "val", backbone, image_size, feature_batch_size, num_workers, device
    )
    x_holdout, y_holdout = _extract_backbone_features(
        root, "test", backbone, image_size, feature_batch_size, num_workers, device
    )
    idx = np.arange(y_holdout.size)
    cal_idx, test_idx = train_test_split(
        idx,
        train_size=0.5,
        random_state=seed,
        stratify=y_holdout if len(np.unique(y_holdout)) > 1 else None,
    )
    return DatasetSplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_cal=x_holdout[cal_idx],
        y_cal=y_holdout[cal_idx],
        x_test=x_holdout[test_idx],
        y_test=y_holdout[test_idx],
        hidden_test=y_holdout[test_idx] == BACH_CRITICAL_CLASS,
        label_names=BACH_LABELS,
    )


def make_dataset_splits(data: DatasetData, seed: int = 0) -> DatasetSplitData:
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
    return DatasetSplitData(
        x_train=data.x[train_idx],
        y_train=data.y[train_idx],
        x_val=data.x[val_idx],
        y_val=data.y[val_idx],
        x_cal=data.x[cal_idx],
        y_cal=data.y[cal_idx],
        x_test=data.x[test_idx],
        y_test=data.y[test_idx],
        hidden_test=data.hidden_critical[test_idx],
        label_names=data.label_names,
    )


def _fit_sklearn_classifier(split: DatasetSplitData, seed: int = 0):
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    base.fit(split.x_train, split.y_train)
    try:
        from sklearn.frozen import FrozenEstimator

        model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except ModuleNotFoundError:
        model = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    model.fit(split.x_val, split.y_val)
    val_loss = log_loss(
        split.y_val,
        model.predict_proba(split.x_val),
        labels=np.arange(split.label_names.size),
    )
    return model, {"val_log_loss": float(val_loss), "device": "cpu"}


def _fit_torch_classifier(split: DatasetSplitData, seed: int, device: str):
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for CUDA training.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    x_train = torch.as_tensor(split.x_train, dtype=torch.float32, device=torch_device)
    y_train = torch.as_tensor(split.y_train, dtype=torch.long, device=torch_device)
    x_val = torch.as_tensor(split.x_val, dtype=torch.float32, device=torch_device)
    y_val = torch.as_tensor(split.y_val, dtype=torch.long, device=torch_device)

    mean = x_train.mean(dim=0, keepdim=True)
    scale = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / scale
    x_val = (x_val - mean) / scale

    linear = torch.nn.Linear(split.x_train.shape[1], split.label_names.size, device=torch_device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.03, weight_decay=1e-4)

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    for _ in range(400):
        linear.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(linear(x_train), y_train)
        loss.backward()
        optimizer.step()

        linear.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(linear(x_val), y_val).item()
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(linear.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 40:
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
        temp_loss = F.cross_entropy(linear(x_val) / temp, y_val)
        temp_loss.backward()
        return temp_loss

    temp_optimizer.step(closure)
    model = TorchTabularClassifier(
        linear=linear,
        mean=mean,
        scale=scale,
        temperature=float(temperature.detach().clamp_min(1e-3).item()),
    )
    val_loss = log_loss(
        split.y_val,
        model.predict_proba(split.x_val),
        labels=np.arange(split.label_names.size),
    )
    return model, {"val_log_loss": float(val_loss), "device": device}


def fit_dataset_classifier(split: DatasetSplitData, seed: int = 0, cuda: int | None = None):
    device = "cpu" if cuda is None else f"cuda:{cuda}"
    if device == "cpu":
        return _fit_sklearn_classifier(split, seed=seed)
    return _fit_torch_classifier(split, seed=seed, device=device)
