import argparse
import glob
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _class_name(path: str) -> str:
    return Path(path).parent.name


def collect_images(src: Path) -> list[str]:
    patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tif", "**/*.tiff", "**/*.bmp"]
    images: list[str] = []
    for pattern in patterns:
        images.extend(glob.glob(str(src / pattern), recursive=True))
    return sorted(set(images))


def split_bach(src: Path, dst: Path, seed: int, train_frac: float, val_frac: float) -> None:
    images = collect_images(src)
    if not images:
        raise FileNotFoundError(f"No images found under {src}")

    labels = [_class_name(p) for p in images]
    train_val, test = train_test_split(
        images,
        test_size=max(0.0, 1.0 - train_frac - val_frac),
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    train_val_labels = [_class_name(p) for p in train_val]
    val_ratio = val_frac / max(train_frac + val_frac, 1e-12)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val_labels if len(set(train_val_labels)) > 1 else None,
    )

    splits = {"train": train, "val": val, "test": test}
    for split, files in splits.items():
        for src_file in tqdm(files, desc=f"copy {split}"):
            class_dir = _class_name(src_file)
            out_dir = dst / split / class_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, out_dir / Path(src_file).name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a BACH-style ImageFolder split.")
    parser.add_argument("--src", required=True, help="Input directory containing class subdirectories.")
    parser.add_argument("--dst", required=True, help="Output directory for train/val/test splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dst = Path(args.dst)
    if dst.exists() and args.overwrite:
        shutil.rmtree(dst)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} already exists and is not empty; pass --overwrite to replace it.")
    split_bach(Path(args.src), dst, args.seed, args.train_frac, args.val_frac)


if __name__ == "__main__":
    main()
