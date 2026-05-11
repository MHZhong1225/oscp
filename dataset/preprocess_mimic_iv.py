import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(root: Path, relative: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(root / relative, compression="infer", **kwargs)


def build_icu_stay_table(root: Path) -> pd.DataFrame:
    stays = read_csv(root, "icu/icustays.csv.gz")
    admissions = read_csv(root, "hosp/admissions.csv.gz")
    patients = read_csv(root, "hosp/patients.csv.gz")

    df = stays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(patients, on="subject_id", how="left")

    intime = pd.to_datetime(df["intime"])
    outtime = pd.to_datetime(df["outtime"])
    df["icu_los_hours"] = (outtime - intime).dt.total_seconds() / 3600.0
    df["hospital_expire_flag"] = df["hospital_expire_flag"].fillna(0).astype(int)
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")

    bins = [-np.inf, 24.0, 72.0, 168.0, np.inf]
    labels = ["short_stay", "medium_stay", "long_stay", "very_long_stay"]
    df["los_group"] = pd.cut(df["icu_los_hours"], bins=bins, labels=labels).astype(str)
    df.loc[df["hospital_expire_flag"] == 1, "los_group"] = "mortality"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact ICU-stay table from MIMIC-IV.")
    parser.add_argument("--root", default="dataset/mimic-iv-3.1")
    parser.add_argument("--out", default="dataset/mimic_iv_icu_stays.csv")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    df = build_icu_stay_table(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
