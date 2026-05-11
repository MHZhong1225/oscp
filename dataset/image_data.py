import os
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class ImageDatasetWithAttrs(Dataset):
    """Wrapper to make ImageFolder return dummy attributes to match tabular loaders."""

    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.num_classes = len(self.dataset.classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, y, 0, 0


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _try_imagefolder_samples(root: str):
    try:
        ds = ImageFolder(root)
    except Exception:
        return None
    if len(ds) <= 0:
        return None
    return ds


def _collect_bach_samples(data_dir: str):
    all_dir = os.path.join(data_dir, "all")
    ds_all = _try_imagefolder_samples(all_dir)
    if ds_all is not None:
        return ds_all.samples, ds_all.class_to_idx

    ds_root = _try_imagefolder_samples(data_dir)
    if ds_root is not None:
        root_classes = set(ds_root.class_to_idx.keys())
        if not ({"train", "val", "test"} <= root_classes or root_classes <= {"train", "val", "test"}):
            return ds_root.samples, ds_root.class_to_idx

    parts = []
    for sub in ["train", "val", "test"]:
        p = os.path.join(data_dir, sub)
        ds = _try_imagefolder_samples(p)
        if ds is not None:
            parts.append((ds.samples, ds.class_to_idx))

    if not parts:
        raise FileNotFoundError(f"No valid ImageFolder root found under {data_dir}")

    base_class_to_idx = parts[0][1]
    all_samples = []
    for samples, class_to_idx in parts:
        if class_to_idx != base_class_to_idx:
            inv = {v: k for k, v in class_to_idx.items()}
            for path, y in samples:
                all_samples.append((path, int(base_class_to_idx[inv[int(y)]])))
        else:
            all_samples.extend([(p, int(y)) for p, y in samples])
    return all_samples, base_class_to_idx


def build_dataloaders_bach(cfg: SimpleNamespace) -> Tuple[Any, Any, Any, Any]:
    data_dir = _cfg_get(cfg, "image_data_dir", "/home/ubuntu/zmh/BrCPT/datasets/bach")
    seed = int(_cfg_get(cfg, "seed", 42))
    n_tra_cal = int(_cfg_get(cfg, "n_tra_cal", 0) or 0)
    test_samples = int(_cfg_get(cfg, "test_samples", 0) or 0)
    image_size = int(_cfg_get(cfg, "image_size", 224))

    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    samples, class_to_idx = _collect_bach_samples(data_dir)
    paths = [p for p, _y in samples]
    labels = np.asarray([int(y) for _p, y in samples], dtype=int)
    total_len = len(paths)
    if total_len < 3:
        raise RuntimeError(f"Insufficient images for BACH split: total={total_len} (need >= 3)")

    if test_samples <= 0:
        test_samples = min(100, max(1, int(round(0.2 * total_len))))
    test_samples = min(test_samples, total_len - 2)
    idx_all = np.arange(total_len)
    try:
        pool_idx, test_idx = train_test_split(
            idx_all,
            test_size=int(test_samples),
            random_state=seed,
            stratify=labels,
        )
    except Exception:
        pool_idx, test_idx = train_test_split(idx_all, test_size=int(test_samples), random_state=seed)

    rng = np.random.default_rng(seed)
    pool_idx = rng.permutation(np.asarray(pool_idx, dtype=int))
    if n_tra_cal > 0:
        pool_idx = pool_idx[: min(n_tra_cal, len(pool_idx))]
    if len(pool_idx) < 2:
        raise RuntimeError("Insufficient train+cal samples after split.")

    train_len = int(len(pool_idx) // 2)
    train_idx = pool_idx[:train_len]
    cal_idx = pool_idx[train_len:]

    class _PathWithAttrs(Dataset):
        def __init__(self, idxs, transform):
            self.idxs = np.asarray(idxs, dtype=int)
            self.transform = transform

        def __len__(self):
            return int(self.idxs.shape[0])

        def __getitem__(self, idx):
            i = int(self.idxs[idx])
            x = default_loader(paths[i])
            if self.transform is not None:
                x = self.transform(x)
            y = int(labels[i])
            return x, y, y, 0, 0

    batch_size = int(_cfg_get(cfg, "batch_size", 32))
    num_workers = int(_cfg_get(cfg, "num_workers", 6))
    train_dataset = _PathWithAttrs(train_idx, transform_train)
    cal_dataset = _PathWithAttrs(cal_idx, transform_test)
    test_dataset = _PathWithAttrs(test_idx, transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    meta = SimpleNamespace(
        dataset_mode="bach",
        path=data_dir,
        n_train=len(train_dataset),
        n_cal=len(cal_dataset),
        n_test=len(test_dataset),
        num_classes=len(class_to_idx) or 4,
        feature_dim=None,
        is_image=True,
    )
    return train_loader, cal_loader, test_loader, meta
