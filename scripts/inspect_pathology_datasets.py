from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.pathology import pathology_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "bach", "bracs", "breakhis"])
    parser.add_argument("--magnification", default="40X")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    df = pathology_manifest(args.dataset, args.magnification)
    print(df.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
