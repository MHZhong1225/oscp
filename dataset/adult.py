import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dataloaders_adult(cfg):
    path = _cfg_get(cfg, "csv_path", _cfg_get(cfg, "adult_csv_path", "dataset/adult/adult.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    df = pd.read_csv(path).dropna().copy()
    target_col = _cfg_get(cfg, "target_col", "income")
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y = LabelEncoder().fit_transform(df[target_col].astype(str))
    features = df.drop(columns=[target_col]).copy()
    for col in features.columns:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))
    x = features.astype(np.float32).to_numpy()

    seed = int(_cfg_get(cfg, "seed", 42))
    n_use = int(_cfg_get(cfg, "n_use", 0) or x.shape[0])
    n_use = min(n_use, x.shape[0])
    x = x[:n_use]
    y = y[:n_use]

    idx = np.arange(n_use)
    train_cal_idx, test_idx = train_test_split(
        idx,
        test_size=float(_cfg_get(cfg, "test_frac", 0.2)),
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    train_idx, cal_idx = train_test_split(
        train_cal_idx,
        test_size=float(_cfg_get(cfg, "cal_frac_of_train_cal", 0.5)),
        random_state=seed,
        stratify=y[train_cal_idx] if len(np.unique(y[train_cal_idx])) > 1 else None,
    )

    batch_size = int(_cfg_get(cfg, "batch_size", 128))

    def make_loader(idxs, shuffle):
        ds = TensorDataset(
            torch.tensor(x[idxs], dtype=torch.float32),
            torch.tensor(y[idxs], dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    meta = SimpleNamespace(
        dataset_mode="adult",
        path=path,
        n_total=n_use,
        n_train=len(train_idx),
        n_cal=len(cal_idx),
        n_test=len(test_idx),
        feature_dim=int(x.shape[1]),
        num_classes=int(len(np.unique(y))),
    )
    return make_loader(train_idx, True), make_loader(cal_idx, False), make_loader(test_idx, False), meta
