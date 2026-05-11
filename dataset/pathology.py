"""Pathology image dataset registry for supplemental medical experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
except ImportError:  # pragma: no cover - optional image training dependency
    transforms = None
    ImageFolder = None


DATASET_ROOT = Path("/home/ubuntu/zmh/BrCPT/datasets")


@dataclass(frozen=True)
class PathologySpec:
    name: str
    root: Path
    class_names: tuple[str, ...]
    action_names: tuple[str, ...]
    relation: np.ndarray


class ImageFolderWithAttrs(Dataset):
    """ImageFolder wrapper returning the tuple shape used by existing loaders."""

    def __init__(self, root: Path, class_to_idx: dict[str, int], transform=None):
        if ImageFolder is None:
            raise RuntimeError("torchvision is required to build pathology dataloaders.")
        self.dataset = ImageFolder(str(root), transform=transform)
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx)
        self.num_classes = len(class_to_idx)
        self._local_to_global = {
            local_idx: class_to_idx[class_name]
            for class_name, local_idx in self.dataset.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, local_y = self.dataset[idx]
        y = int(self._local_to_global[int(local_y)])
        return x, y, y, 0, 0


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_pathology_spec(name: str, magnification: str = "40X") -> PathologySpec:
    """Return dataset path, classes, and a decision relation matrix."""
    name = name.lower()
    if name == "bach":
        class_names = ("Benign", "InSitu", "Invasive", "Normal")
        action_names = ("routine_followup", "biopsy_review", "oncology_escalation")
        relation = np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=int,
        )
        return PathologySpec(name, DATASET_ROOT / "bach", class_names, action_names, relation)

    if name == "bracs":
        class_names = ("0_N", "1_PB", "2_UDH", "3_FEA", "4_ADH", "5_DCIS", "6_IC")
        action_names = (
            "routine_screening",
            "benign_review",
            "atypia_workup",
            "carcinoma_escalation",
        )
        relation = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            dtype=int,
        )
        return PathologySpec(name, DATASET_ROOT / "bracs", class_names, action_names, relation)

    if name == "breakhis":
        class_names = ("A", "DC", "F", "LC", "MC", "PC", "PT", "TA")
        action_names = ("benign_followup", "malignant_workup", "specialist_review")
        relation = np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 0, 0],
            ],
            dtype=int,
        )
        root = DATASET_ROOT / "breakhis" / str(magnification)
        return PathologySpec(f"breakhis_{magnification}", root, class_names, action_names, relation)

    raise ValueError(f"unknown pathology dataset: {name}")


def pathology_manifest(dataset: str = "all", magnification: str = "40X") -> pd.DataFrame:
    """Count images by dataset, split, and class without loading image bytes."""
    if dataset == "all":
        specs = [
            get_pathology_spec("bach"),
            get_pathology_spec("bracs"),
            *[get_pathology_spec("breakhis", mag) for mag in ("40X", "100X", "200X", "400X")],
        ]
    else:
        specs = [get_pathology_spec(dataset, magnification)]

    rows = []
    for spec in specs:
        for split in ("train", "val", "test"):
            split_root = spec.root / split
            for class_name in spec.class_names:
                class_root = split_root / class_name
                count = len(list(class_root.glob("*"))) if class_root.exists() else 0
                rows.append(
                    {
                        "dataset": spec.name,
                        "split": split,
                        "class": class_name,
                        "count": count,
                        "root": str(spec.root),
                    }
                )
    return pd.DataFrame(rows)


def build_pathology_dataloaders(cfg: SimpleNamespace):
    """Build train/val/test image loaders for BACH, BRACS, or BreakHis."""
    if transforms is None:
        raise RuntimeError("torchvision is required to build pathology dataloaders.")
    dataset = _cfg_get(cfg, "dataset", _cfg_get(cfg, "dataset_mode", "bach"))
    magnification = _cfg_get(cfg, "magnification", "40X")
    data_dir = _cfg_get(cfg, "image_data_dir", None)
    batch_size = int(_cfg_get(cfg, "batch_size", 32))
    num_workers = int(_cfg_get(cfg, "num_workers", 6))
    image_size = int(_cfg_get(cfg, "image_size", 224))

    spec = get_pathology_spec(dataset, magnification)
    root = Path(data_dir).expanduser() if data_dir is not None else spec.root
    class_to_idx = {class_name: i for i, class_name in enumerate(spec.class_names)}

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

    train_dataset = ImageFolderWithAttrs(root / "train", class_to_idx, transform_train)
    cal_dataset = ImageFolderWithAttrs(root / "val", class_to_idx, transform_test)
    test_dataset = ImageFolderWithAttrs(root / "test", class_to_idx, transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    meta = SimpleNamespace(
        dataset_mode=spec.name,
        path=str(root),
        n_train=len(train_dataset),
        n_cal=len(cal_dataset),
        n_test=len(test_dataset),
        num_classes=len(spec.class_names),
        class_names=spec.class_names,
        action_names=spec.action_names,
        relation=spec.relation,
        feature_dim=None,
        is_image=True,
    )
    return train_loader, cal_loader, test_loader, meta


def build_dataloaders_bach(cfg: SimpleNamespace):
    return build_pathology_dataloaders(SimpleNamespace(**{**vars(cfg), "dataset": "bach"}))


def build_dataloaders_bracs(cfg: SimpleNamespace):
    return build_pathology_dataloaders(SimpleNamespace(**{**vars(cfg), "dataset": "bracs"}))


def build_dataloaders_breakhis(cfg: SimpleNamespace):
    return build_pathology_dataloaders(SimpleNamespace(**{**vars(cfg), "dataset": "breakhis"}))
