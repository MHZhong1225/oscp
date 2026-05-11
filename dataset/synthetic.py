"""Synthetic triage data with low-risk hidden critical cases."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    F = None
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


LABELS = np.array(["benign", "mild", "moderate", "severe", "critical"])
SEVERE_CLASS = 3
CRITICAL_CLASS = 4


@dataclass(frozen=True)
class SyntheticData:
    x: np.ndarray
    y: np.ndarray
    latent_z: np.ndarray
    hidden_critical: np.ndarray


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    hidden_test: np.ndarray


@dataclass
class TorchCalibratedClassifier:
    linear: Any
    mean: Any
    scale: Any
    temperature: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for non-CPU device training.")
        self.linear.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
            x_scaled = (x_tensor - self.mean) / self.scale
            logits = self.linear(x_scaled) / self.temperature
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def generate_synthetic_triage(
    n: int = 30_000,
    d: int = 20,
    seed: int = 0,
    hidden_rate: float = 0.24,
) -> SyntheticData:
    """Generate a five-class diagnosis task.

    A small fraction of low-severity-looking units are relabeled as critical.
    Their observed features remain benign-like, so a standard classifier tends
    to assign low severe/critical risk while the true LAC score is large.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    beta = np.zeros(d)
    beta[:6] = np.array([1.2, -0.8, 0.7, 0.45, -0.55, 0.35])
    z = x @ beta + rng.normal(scale=0.9, size=n)

    y = np.digitize(z, bins=np.array([-1.25, -0.25, 0.75, 1.75])).astype(int)

    low_region = z < -1.05
    hidden = low_region & (rng.random(n) < hidden_rate)
    y[hidden] = CRITICAL_CLASS

    # Add mild nonlinear nuisance features to help tree-free models remain
    # imperfect in the hidden region without making the full task unrealistic.
    x[:, 6] = 0.6 * np.sin(x[:, 0]) + 0.4 * rng.normal(size=n)
    x[:, 7] = 0.5 * (x[:, 1] ** 2 - 1.0) + 0.5 * rng.normal(size=n)
    return SyntheticData(x=x, y=y, latent_z=z, hidden_critical=hidden)


def make_splits(data: SyntheticData, seed: int = 0) -> SplitData:
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
    return SplitData(
        x_train=data.x[train_idx],
        y_train=data.y[train_idx],
        x_val=data.x[val_idx],
        y_val=data.y[val_idx],
        x_cal=data.x[cal_idx],
        y_cal=data.y[cal_idx],
        x_test=data.x[test_idx],
        y_test=data.y[test_idx],
        hidden_test=data.hidden_critical[test_idx],
    )



def _fit_sklearn_classifier(split: SplitData, seed: int = 0):
    """Fit a deliberately simple classifier and calibrate probabilities on val."""
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=2.0,
            max_iter=2000,
            random_state=seed,
        ),
    )
    base.fit(split.x_train, split.y_train)
    model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    model.fit(split.x_val, split.y_val)
    val_loss = log_loss(split.y_val, model.predict_proba(split.x_val), labels=np.arange(5))
    return model, {"val_log_loss": float(val_loss), "device": "cpu"}


def _fit_torch_classifier(split: SplitData, seed: int, device: str):
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for non-CPU device training.")

    torch.manual_seed(seed)
    if device.startswith("cuda"):
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

    linear = torch.nn.Linear(split.x_train.shape[1], len(LABELS), device=torch_device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.05, weight_decay=1e-4)

    best_state = None
    best_val_loss = float("inf")
    patience = 40
    stale_epochs = 0
    for _ in range(400):
        linear.train()
        optimizer.zero_grad()
        train_loss = F.cross_entropy(linear(x_train), y_train)
        train_loss.backward()
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
        loss = F.cross_entropy(linear(x_val) / temp, y_val)
        loss.backward()
        return loss

    temp_optimizer.step(closure)
    model = TorchCalibratedClassifier(
        linear=linear,
        mean=mean,
        scale=scale,
        temperature=float(temperature.detach().clamp_min(1e-3).item()),
    )
    val_loss = log_loss(split.y_val, model.predict_proba(split.x_val), labels=np.arange(5))
    return model, {"val_log_loss": float(val_loss), "device": device}


def fit_base_classifier(split: SplitData, seed: int = 0, cuda: int | None = None):
    device = "cpu" if cuda is None else f"cuda:{cuda}"
    if device == "cpu":
        return _fit_sklearn_classifier(split, seed=seed)
    return _fit_torch_classifier(split, seed=seed, device=device)


def severe_critical_risk(probs: np.ndarray) -> np.ndarray:
    return probs[:, SEVERE_CLASS] + probs[:, CRITICAL_CLASS]
