import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dataloaders_nursery(cfg):
    path = _cfg_get(cfg, "csv_path", _cfg_get(cfg, "nursery_csv_path", "dataset/nursery/nursery.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    df = pd.read_csv(path)
    df = df[df["final evaluation"] != "recommend"].copy()

    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["final evaluation"])
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
    for col in feature_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    target_occ = int(_cfg_get(cfg, "bias_group_code", 1))
    mask_target = df["parents"] == target_occ
    seed = int(_cfg_get(cfg, "seed", 42))
    rng = np.random.default_rng(seed)

    df_bias = df[mask_target].sample(frac=0.1, random_state=seed).copy()
    noise = rng.uniform(-4, 4, size=len(df_bias))
    df_bias["label"] = np.clip(np.round(df_bias["label"].values + noise), 0, 3).astype(np.int64)
    df_final = (
        pd.concat([df[~mask_target], df_bias])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    x = df_final[feature_cols].astype(np.float32).to_numpy()
    y = df_final["label"].to_numpy()
    attrs = [df_final[col].to_numpy() for col in ["parents", "children", "finance", "social", "health"]]

    n_total = len(df_final)
    n_use = int(_cfg_get(cfg, "n_use", 0) or n_total)
    n_total = min(n_use, n_total)
    test_samples = int(_cfg_get(cfg, "test_samples", 500))
    test_samples = min(test_samples, max(1, n_total - 2))
    n_train_cal = n_total - test_samples
    train_end = n_train_cal // 2

    def make_loader(sl: slice, shuffle: bool, b_size: int):
        ds = TensorDataset(
            torch.tensor(x[sl], dtype=torch.float32),
            torch.tensor(y[sl], dtype=torch.long),
            *[torch.tensor(a[sl], dtype=torch.long) for a in attrs],
        )
        return DataLoader(ds, batch_size=b_size, shuffle=shuffle, drop_last=False)

    batch_size = int(_cfg_get(cfg, "batch_size", 25))
    train_loader = make_loader(slice(0, train_end), True, batch_size)
    cal_loader = make_loader(slice(train_end, n_train_cal), False, batch_size)
    test_loader = make_loader(slice(n_train_cal, n_total), False, test_samples)

    meta = SimpleNamespace(
        dataset_mode="nursery_afcp",
        path=path,
        n_total=n_total,
        n_train=train_end,
        n_cal=n_train_cal - train_end,
        n_test=test_samples,
        feature_dim=8,
        num_classes=4,
    )
    return train_loader, cal_loader, test_loader, meta
